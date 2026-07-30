"""CORAL-style formal oracle interventions for Cluster-Normalized Poisoning.

**Step 1 only** (see `docs/CORAL_MMD_ORACLE_INTERVENTION_PLAN.md` and the plan
file `coral_mmd_oracle_intervention_plan_31a5d179.plan.md`): implements the
**PCA/subspace CORAL-style covariance-alignment transform** only.

Explicitly **out of scope for this module** (deferred to later steps of that
plan, not implemented here):

- Full-dimensional ridge-regularized CORAL (`Cov + lambda*I` inverted over
  all 384 dimensions).
- The MMD-minimizing gradient-based oracle optimizer.

Like `defense/cluster_normalized_poisoning.py`, this module is purely
additive: it is never imported by `defense/defense_runner.py`,
`defense/dispatch.py`, `defense/filterrag.py`, or `main.py`, and it never
calls an LLM/API. It only transforms already-encoded poison-passage
embeddings for a single retrieved-context query and hands the result to the
*unmodified* `defense/ragdefender_internals.py::concentration_stage1` /
`stage2_pair_frequency` functions, exactly like E0/E1.

**Why PCA/subspace, not naive full-dimensional CORAL (see the plan doc for
the full derivation):** with `n_poison = n_clean = 5` points in `d = 384`
dimensions, the mean-centered poison/clean covariance matrices have rank
`<= n - 1 = 4`. A naive `Cov^{-1/2}` over all 384 dimensions is undefined in
the 380-dimensional null space (division by zero), and even a ridge-
regularized inverse (`Cov + lambda*I`) is dominated by the arbitrary
`lambda` term in those null directions -- an artifact of the regularizer,
not of the data. This module instead:

1. Computes the poison/clean covariance eigenbasis via the **economy SVD**
   of the mean-centered data matrix directly (never forms the `d x d`
   covariance matrix explicitly), which is exact and numerically stable for
   `n << d`.
2. Restricts CORAL's whiten/recolor operation to the top
   `rank <= min(n_poison - 1, n_clean - 1)` singular directions of each
   group's own centered data -- the *only* directions with genuine
   signal -- and treats the null space as carrying zero information (no
   inversion, no regularizer) rather than approximating it.
3. At the default rank (`min(n_poison - 1, n_clean - 1)`, e.g. 4 when
   `n_poison = n_clean = 5`), this is an **exact** decomposition: the
   centered poison data matrix has true rank `<= n_poison - 1`, so no
   signal is discarded. Only if a caller explicitly requests a *smaller*
   rank does this become a lossy (variance-discarding) approximation --
   documented explicitly on `coral_pca_transform`, not silently assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from defense.cluster_normalized_poisoning import l2_normalize_rows

__all__ = [
    "CoralPcaTransformResult",
    "resolve_subspace_rank",
    "coral_pca_transform",
]

DEFAULT_EIGENVALUE_FLOOR = 1e-8


def resolve_subspace_rank(n_poison: int, n_clean: int, requested_rank: Optional[int] = None) -> int:
    """`rank <= min(n_poison - 1, n_clean - 1)` -- the maximum rank at which
    *both* groups' mean-centered covariance can have genuine (non-null)
    signal. `requested_rank=None` (the default used throughout this
    module's sweep) resolves to this maximum, i.e. no additional
    dimensionality reduction beyond what rank-deficiency already forces.
    A `requested_rank` above this ceiling is silently clamped down to it
    (never raised above the data's own maximum possible rank); a negative
    `requested_rank` raises `ValueError`."""
    max_rank = max(min(n_poison - 1, n_clean - 1), 0)
    if requested_rank is None:
        return max_rank
    if requested_rank < 0:
        raise ValueError(f"requested_rank must be >= 0, got {requested_rank}")
    return min(requested_rank, max_rank)


def _truncated_symmetric_sqrt_or_pinv(z_centered: np.ndarray, rank: int, invert: bool,
                                       eps: float = DEFAULT_EIGENVALUE_FLOOR) -> np.ndarray:
    """Truncated rank-`rank` (pseudo-)square-root of the mean-centered data
    `z_centered`'s (`n x d`) implicit covariance `Cov = z_centered^T @
    z_centered / (n - 1)`, computed via `z_centered`'s own **economy SVD**
    (`z_centered = U @ diag(s) @ Vt`, `s` descending) rather than ever
    forming the `d x d` covariance matrix explicitly -- `Cov`'s
    eigenvectors are exactly `Vt`'s rows, with eigenvalues `s**2 / (n-1)`.

    Returns the `d x d` symmetric matrix `sum_{i=1}^{rank} f(lambda_i) *
    v_i @ v_i^T`, where `f(lambda) = 1/sqrt(lambda)` if `invert=True`
    (whitening; a direction with `lambda_i <= eps` is treated as null and
    contributes nothing -- **never divided by**, unlike a naive/ridge
    inverse) else `f(lambda) = sqrt(max(lambda, 0))` (recoloring; guards
    only against floating-point noise). Directions beyond `rank` (or
    beyond how many singular vectors `z_centered` actually has, `<= n-1`
    for mean-centered data) also contribute nothing.
    """
    z_centered = np.asarray(z_centered, dtype=np.float64)
    n, d = z_centered.shape
    if n < 2:
        raise ValueError(f"_truncated_symmetric_sqrt_or_pinv requires n >= 2 rows, got n={n}")
    if rank < 0:
        raise ValueError(f"rank must be >= 0, got {rank}")

    _, s, vt = np.linalg.svd(z_centered, full_matrices=False)
    eigvals = (s ** 2) / (n - 1)

    out = np.zeros((d, d), dtype=np.float64)
    n_available = min(rank, eigvals.shape[0])
    for i in range(n_available):
        lam = float(eigvals[i])
        if invert:
            if lam <= eps:
                continue
            coeff = 1.0 / np.sqrt(lam)
        else:
            coeff = float(np.sqrt(max(lam, 0.0)))
        v = vt[i]
        out += coeff * np.outer(v, v)
    return out


@dataclass(frozen=True)
class CoralPcaTransformResult:
    """Every intermediate of one `coral_pca_transform` call, for logging
    and for the `beta=0` identity check (§ tests)."""

    z_poison_coral: np.ndarray   # pure CORAL-recolored poison (beta=1.0 pre-normalize state), shape (n_poison, d)
    z_poison_final: np.ndarray   # (1-beta)*z_poison_original + beta*z_poison_coral, L2-normalized -- what callers feed to cos_sim
    rank: int                    # resolved subspace rank actually used
    target_mean: np.ndarray      # mean added back after recoloring (== clean_mean by default)
    beta: float


def coral_pca_transform(z_poison: np.ndarray, z_clean: np.ndarray, beta: float,
                         rank: Optional[int] = None, eps: float = DEFAULT_EIGENVALUE_FLOOR,
                         target_mean: Optional[np.ndarray] = None) -> CoralPcaTransformResult:
    """PCA/subspace CORAL-style covariance-alignment transform (Step 1 of
    the CORAL/MMD oracle intervention plan). `z_clean` is **read-only** --
    this function never modifies or returns a transformed `z_clean`;
    callers must still pass the original, untransformed `z_clean` through
    unchanged when recombining, per the execution plan's fixed data-flow
    contract (same convention as `centroid_interpolate`/`anchor_interpolate`
    in `defense/cluster_normalized_poisoning.py`).

    Steps (see module docstring for the rank-deficiency rationale):

    1. Center `z_poison`/`z_clean` by their own means.
    2. Resolve the shared subspace `rank` (default
       `min(n_poison - 1, n_clean - 1)`, via `resolve_subspace_rank`).
    3. Whiten the centered poison data within its own top-`rank` singular
       directions (`Cp^{-1/2}` restricted to that subspace; a truncated
       pseudo-inverse, not a naive/ridge full-rank inverse).
    4. Recolor within the clean group's own top-`rank` singular directions
       (`Cc^{1/2}` restricted to that subspace).
    5. Add back `target_mean` (default: `mean(z_clean)` -- the documented
       default target-mean choice for this transform; a caller-supplied
       `target_mean`, e.g. an interpolated mean, is accepted but not used
       by default).
    6. Mix with the original, untransformed poison embeddings:
       `Zp_final = normalize((1 - beta) * Zp_original + beta * Zp_coral)`.
       At `beta=0.0` this reduces to `normalize(Zp_original)` exactly,
       regardless of `z_poison_coral` -- the identity/sanity control (same
       role as `alpha=1.0` for E0/E1).

    Raises `FloatingPointError` if any output is non-finite (should be
    unreachable given the eigenvalue floor in step 3, but is asserted
    explicitly rather than silently returned).
    """
    z_poison = np.asarray(z_poison, dtype=np.float64)
    z_clean = np.asarray(z_clean, dtype=np.float64)
    n_poison, n_clean = z_poison.shape[0], z_clean.shape[0]
    if n_poison < 2 or n_clean < 2:
        raise ValueError(
            f"coral_pca_transform requires >= 2 rows per group (mean-centered covariance is "
            f"undefined for n<2); got n_poison={n_poison}, n_clean={n_clean}."
        )
    if z_poison.shape[1] != z_clean.shape[1]:
        raise ValueError(
            f"z_poison and z_clean must share embedding dim; got {z_poison.shape[1]} vs {z_clean.shape[1]}"
        )

    resolved_rank = resolve_subspace_rank(n_poison, n_clean, rank)

    poison_mean = z_poison.mean(axis=0)
    clean_mean = z_clean.mean(axis=0)
    resolved_target_mean = clean_mean if target_mean is None else np.asarray(target_mean, dtype=np.float64)

    zp_centered = z_poison - poison_mean[None, :]
    zc_centered = z_clean - clean_mean[None, :]

    if resolved_rank == 0:
        # No shared signal subspace (only reachable if both groups are
        # exactly size 2, since rank = min(n_poison-1, n_clean-1) and the
        # >=2-row guard above already excludes n<2): the CORAL component
        # degenerates to a pure translation onto the target mean, since
        # there is no covariance direction left to whiten/recolor.
        z_poison_coral = np.tile(resolved_target_mean, (n_poison, 1))
    else:
        cp_inv_half = _truncated_symmetric_sqrt_or_pinv(zp_centered, resolved_rank, invert=True, eps=eps)
        cc_half = _truncated_symmetric_sqrt_or_pinv(zc_centered, resolved_rank, invert=False, eps=eps)
        # Both factors are symmetric d x d matrices restricted to the
        # shared rank-`resolved_rank` subspace; for row-vector data,
        # whiten-then-recolor is `z_row @ Cp_inv_half @ Cc_half` (see
        # module docstring derivation). `cp_inv_half`/`cc_half` are each
        # individually verified finite above (by construction: only
        # eigenvalues > eps are ever inverted), but their outer-product
        # construction can leave extremely small (denormal-range) entries
        # that some BLAS backends (observed with Apple's Accelerate
        # framework) flag with spurious divide-by-zero/overflow/invalid
        # RuntimeWarnings during the matmul below even though the actual
        # numeric result is finite and correct -- suppressed locally; the
        # explicit `isfinite` assertion after this block is the real
        # correctness guard, not this suppression.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            transform = cp_inv_half @ cc_half
            zp_coral_centered = zp_centered @ transform
        z_poison_coral = zp_coral_centered + resolved_target_mean[None, :]

    mixed = (1.0 - beta) * z_poison + beta * z_poison_coral
    z_poison_final = l2_normalize_rows(mixed)

    if not np.all(np.isfinite(z_poison_coral)):
        raise FloatingPointError("coral_pca_transform: z_poison_coral contains non-finite values")
    if not np.all(np.isfinite(z_poison_final)):
        raise FloatingPointError("coral_pca_transform: z_poison_final contains non-finite values")

    return CoralPcaTransformResult(
        z_poison_coral=z_poison_coral,
        z_poison_final=z_poison_final,
        rank=resolved_rank,
        target_mean=resolved_target_mean,
        beta=float(beta),
    )
