"""CORAL-style and MMD-minimizing formal oracle interventions for
Cluster-Normalized Poisoning.

**Steps 1, 2, and 3** of the CORAL/MMD oracle intervention plan (see
`docs/CORAL_MMD_ORACLE_INTERVENTION_PLAN.md` and the plan file
`coral_mmd_oracle_intervention_plan_31a5d179.plan.md`):

- **Step 1**: `coral_pca_transform` -- the **PCA/subspace CORAL-style
  covariance-alignment transform**, which restricts whiten/recolor to each
  group's own top-`rank` singular directions and treats the null space as
  carrying zero signal (no inversion at all in that space).
- **Step 2**: `coral_ridge_transform` -- the **full-dimensional
  ridge-regularized CORAL transform**, which instead inverts the *entire*
  `384 x 384` covariance (regularized with `+ lambda*I`) so that every
  dimension -- including the 380 directions with no genuine poison/clean
  signal at `n=5` -- participates in the whiten/recolor operation. This is
  a materially different, and deliberately harsher/less-conservative,
  oracle transform than Step 1: see `coral_ridge_transform`'s docstring for
  why its null-space behavior is fundamentally different from Step 1's.
- **Step 3**: `mmd_minimize_transform` -- a **direct gradient-based oracle
  optimizer** that moves the poison embeddings to directly minimize
  biased-squared-RBF MMD against the clean embeddings (via PyTorch
  autograd), rather than aligning second-order covariance statistics like
  Steps 1/2. This tests whether a *distribution-alignment objective
  applied directly*, rather than through the CORAL covariance-matching
  proxy, is a stronger oracle stress test for RAGDefender's Stage 2. This
  is **not** DAN (no discriminator network is trained; MMD here is used
  exactly as in the original CORAL/MMD literature background, as a direct,
  closed-form two-sample statistic optimized over frozen embeddings, never
  as a learned model).

Like `defense/cluster_normalized_poisoning.py`, this module is purely
additive: it is never imported by `defense/defense_runner.py`,
`defense/dispatch.py`, `defense/filterrag.py`, or `main.py`, and it never
calls an LLM/API. It only transforms already-encoded poison-passage
embeddings for a single retrieved-context query and hands the result to the
*unmodified* `defense/ragdefender_internals.py::concentration_stage1` /
`stage2_pair_frequency` functions, exactly like E0/E1.

**Why PCA/subspace (Step 1) is not naive full-dimensional CORAL:** with
`n_poison = n_clean = 5` points in `d = 384` dimensions, the mean-centered
poison/clean covariance matrices have rank `<= n - 1 = 4`. A naive
`Cov^{-1/2}` over all 384 dimensions is undefined in the 380-dimensional
null space (division by zero), and even a ridge-regularized inverse
(`Cov + lambda*I`) is dominated by the arbitrary `lambda` term in those null
directions -- an artifact of the regularizer, not of the data. Step 1
instead:

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

**Why Step 2 (full-ridge) is included anyway, despite Step 1's own
docstring calling the ridge variant's null-space behavior an "artifact of
the regularizer, not of the data":** Step 1's PCA/subspace transform (run
in `scripts/run_coral_pca_oracle_intervention.py`) monotonically reduced
CORAL/MMD distance but caused **0/6** residual-poison RAGDefender failures
across the tested queries -- a genuine negative result, but one that
leaves open whether that failure to disrupt Stage 2 is inherent to
*distribution alignment* as an attack family, or specific to *restricting*
the alignment to the (exact, 4-dimensional) signal subspace. Step 2 is the
oracle stress test that isolates that question: it applies the harsher,
less-conservative full-384-dimensional ridge transform (which *does*
perturb the null-space directions, unlike Step 1) and reports, per
`coral_ridge_transform`'s own numerical-limitations note, whether that
additional (regularizer-dominated, not data-dominated) perturbation changes
the Stage-2 outcome. See `CORAL_RIDGE_REPORT.md`'s Limitations section for
why any resulting failure is not more "real" than Step 1's, even if it
occurs.

**Why Step 3 (direct MMD minimization) is a distinct oracle from Steps
1/2:** CORAL (both variants) aligns the poison group's *second-order*
statistics (covariance) to the clean group's; two distributions can have
matched covariance while still being geometrically far apart or
differently shaped in ways CORAL cannot see -- and, per Steps 1/2's own
0/6 and 0/18 residual-poison-failure results, evidently did not disrupt
RAGDefender's Stage-2 poison-pair structure. MMD is a full two-sample
kernel statistic sensitive to more than just the covariance -- directly
minimizing it is a strictly more direct (if less structured) oracle
objective. It is implemented as a **direct gradient-based optimizer over
the frozen poison embeddings** (`mmd_minimize_transform`, this module),
never as a trained auxiliary model (e.g. a domain-adversarial network /
DAN -- DAN is literature background motivating why MMD is a meaningful
alignment objective, not something implemented or trained here), keeping
it in the same "oracle embedding-space intervention" family as Steps 1/2
and E0/E1: it still only ever produces a transformed `Zp_prime` for a
fixed, already-retrieved set of passages, never a learned, reusable
attack model or a natural-language rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch

from defense.cluster_normalized_poisoning import l2_normalize_rows
from defense.distribution_metrics import DEFAULT_MMD_GAMMA

__all__ = [
    "CoralPcaTransformResult",
    "resolve_subspace_rank",
    "coral_pca_transform",
    "CoralRidgeTransformResult",
    "coral_ridge_transform",
    "PreservationMetrics",
    "compute_preservation_metrics",
    "MmdTransformResult",
    "mmd_minimize_transform",
    "mmd_rbf_squared_raw",
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


def _ridge_covariance(z_centered: np.ndarray, lam: float) -> np.ndarray:
    """Full `d x d` ridge-regularized covariance `Cov + lambda*I` of
    mean-centered data `z_centered` (`n x d`), where `Cov = z_centered^T @
    z_centered / (n-1)`. Unlike `_truncated_symmetric_sqrt_or_pinv`, this
    **explicitly forms the full `d x d` matrix** -- the entire point of
    the ridge variant is that every dimension (including the `d - (n-1)`
    directions with zero true poison/clean signal at this sample size)
    participates in the subsequent inversion, not just the top-`rank`
    signal directions. `lam` must be strictly positive (asserted by the
    caller): `Cov` alone is only PSD (rank `<= n-1 < d`), so without `lam`
    the matrix is singular and `Cov^{-1/2}` would be undefined in the null
    space -- this is exactly the naive full-dimensional CORAL failure mode
    documented in the module docstring, which ridge regularization exists
    to avoid.
    """
    z_centered = np.asarray(z_centered, dtype=np.float64)
    n, d = z_centered.shape
    cov = (z_centered.T @ z_centered) / (n - 1)
    cov = (cov + cov.T) / 2.0  # symmetrize away floating-point rounding asymmetry before eigh
    return cov + lam * np.eye(d, dtype=np.float64)


def _full_symmetric_sqrt_or_pinv(cov_matrix: np.ndarray, invert: bool,
                                  eps: float = DEFAULT_EIGENVALUE_FLOOR) -> np.ndarray:
    """Full-rank (pseudo-)square-root of a `d x d` symmetric matrix (a
    ridge-regularized covariance, `Cov + lambda*I`), via `numpy.linalg.eigh`
    (which assumes/enforces symmetry) over **all** `d` eigenpairs -- the
    key numerical difference from `_truncated_symmetric_sqrt_or_pinv`,
    which only ever uses the top `rank <= n-1` eigenpairs. Because the
    input is already ridge-regularized (every true covariance eigenvalue,
    including the `d-(n-1)` structurally-zero ones, has `lam` added to it
    before this function ever sees it), no eigenvalue here should be
    `<= 0` in exact arithmetic; `eps` (default `1e-8`, independent of and
    much smaller than any swept `lambda in {1e-1, 1e-2, 1e-3}`) is applied
    only as a **numerical safety clip** against floating-point rounding in
    the eigendecomposition, never as a substitute for ridge regularization
    itself (which is `_ridge_covariance`'s `+ lam*I`, applied *before* this
    function runs) -- i.e. this function never performs an
    **unregularized** inverse: an un-ridge-regularized covariance is never
    passed to it by `coral_ridge_transform`.
    """
    cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)  # ascending; cov_matrix assumed symmetric
    eigvals_clipped = np.clip(eigvals, eps, None)
    if invert:
        coeffs = 1.0 / np.sqrt(eigvals_clipped)
    else:
        coeffs = np.sqrt(eigvals_clipped)
    return (eigvecs * coeffs[None, :]) @ eigvecs.T


@dataclass(frozen=True)
class CoralRidgeTransformResult:
    """Every intermediate of one `coral_ridge_transform` call, for logging
    and for the `beta=0` identity check (§ tests)."""

    z_poison_coral: np.ndarray   # pure CORAL-recolored poison (beta=1.0 pre-normalize state), shape (n_poison, d)
    z_poison_final: np.ndarray   # (1-beta)*z_poison_original + beta*z_poison_coral, L2-normalized -- what callers feed to cos_sim
    lam: float                   # ridge regularization strength actually used
    target_mean: np.ndarray      # mean added back after recoloring (== clean_mean by default)
    beta: float


def coral_ridge_transform(z_poison: np.ndarray, z_clean: np.ndarray, beta: float, lam: float,
                           eps: float = DEFAULT_EIGENVALUE_FLOOR,
                           target_mean: Optional[np.ndarray] = None) -> CoralRidgeTransformResult:
    """Full-dimensional ridge-regularized CORAL-style covariance-alignment
    transform (Step 2 of the CORAL/MMD oracle intervention plan). `z_clean`
    is **read-only**, same data-flow contract as `coral_pca_transform`.

    Steps (see module docstring for why this is a materially different,
    harsher oracle transform than Step 1's PCA/subspace variant):

    1. Center `z_poison`/`z_clean` by their own means.
    2. Form the **full** `d x d` ridge-regularized covariances
       `Cp = cov(Zp_centered) + lambda*I`, `Cc = cov(Zc_centered) +
       lambda*I` (`_ridge_covariance`) -- `lambda` must be strictly
       positive; see the `ValueError` below.
    3. `Cp^{-1/2}` (whiten) and `Cc^{1/2}` (recolor) via **full** symmetric
       eigendecomposition over all `d` eigenpairs (`_full_symmetric_sqrt_or_pinv`),
       with only an `eps`-eigenvalue floor for float-rounding safety --
       never an unregularized inverse (regularization is already baked
       into `Cp`/`Cc` themselves via `+ lambda*I`).
    4. `Zp_coral = (Zp - mean_p) @ Cp^{-1/2} @ Cc^{1/2} + target_mean`
       (`target_mean` defaults to `mean(Zc)`, same documented default as
       Step 1).
    5. Mix with the original, untransformed poison embeddings:
       `Zp_final = normalize((1 - beta) * Zp_original + beta * Zp_coral)`.
       At `beta=0.0` this reduces to `normalize(Zp_original)` exactly,
       regardless of `lambda` or `z_poison_coral` -- the identity/sanity
       control.

    Raises `ValueError` if `lam <= 0` (ridge regularization must be
    strictly positive for this full-dimensional variant -- an
    unregularized full-dimensional CORAL is mathematically undefined at
    this sample size, per the module docstring) or if the two groups have
    fewer than 2 rows / mismatched embedding dimensions. Raises
    `FloatingPointError` if any output is non-finite (should be
    unreachable given `lam > 0` and the eigenvalue floor, but is asserted
    explicitly rather than silently returned).
    """
    z_poison = np.asarray(z_poison, dtype=np.float64)
    z_clean = np.asarray(z_clean, dtype=np.float64)
    n_poison, n_clean = z_poison.shape[0], z_clean.shape[0]
    if n_poison < 2 or n_clean < 2:
        raise ValueError(
            f"coral_ridge_transform requires >= 2 rows per group (mean-centered covariance is "
            f"undefined for n<2); got n_poison={n_poison}, n_clean={n_clean}."
        )
    if z_poison.shape[1] != z_clean.shape[1]:
        raise ValueError(
            f"z_poison and z_clean must share embedding dim; got {z_poison.shape[1]} vs {z_clean.shape[1]}"
        )
    if lam <= 0:
        raise ValueError(
            f"coral_ridge_transform requires lam > 0 (an unregularized full-dimensional inverse is "
            f"undefined when n <= d, see module docstring); got lam={lam}"
        )

    poison_mean = z_poison.mean(axis=0)
    clean_mean = z_clean.mean(axis=0)
    resolved_target_mean = clean_mean if target_mean is None else np.asarray(target_mean, dtype=np.float64)

    zp_centered = z_poison - poison_mean[None, :]
    zc_centered = z_clean - clean_mean[None, :]

    # Same benign Apple Accelerate BLAS quirk documented in
    # coral_pca_transform: matmuls involving these tall-skinny (n=5, d=384)
    # data/eigenvector matrices can trip spurious divide-by-zero / overflow
    # / invalid RuntimeWarnings even though the numeric result is correct
    # and finite -- suppressed for this whole numerical core; the explicit
    # `isfinite` assertions below (on the actual returned arrays) are the
    # real correctness guard, not this suppression.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cp_ridge = _ridge_covariance(zp_centered, lam)
        cc_ridge = _ridge_covariance(zc_centered, lam)
        cp_inv_half = _full_symmetric_sqrt_or_pinv(cp_ridge, invert=True, eps=eps)
        cc_half = _full_symmetric_sqrt_or_pinv(cc_ridge, invert=False, eps=eps)
        # Both factors are symmetric d x d matrices (full rank, unlike Step
        # 1's rank-truncated factors); for row-vector data, whiten-then-recolor
        # is `z_row @ Cp_inv_half @ Cc_half` (see module docstring derivation).
        transform = cp_inv_half @ cc_half
        zp_coral_centered = zp_centered @ transform
    z_poison_coral = zp_coral_centered + resolved_target_mean[None, :]

    mixed = (1.0 - beta) * z_poison + beta * z_poison_coral
    z_poison_final = l2_normalize_rows(mixed)

    if not np.all(np.isfinite(z_poison_coral)):
        raise FloatingPointError("coral_ridge_transform: z_poison_coral contains non-finite values")
    if not np.all(np.isfinite(z_poison_final)):
        raise FloatingPointError("coral_ridge_transform: z_poison_final contains non-finite values")

    return CoralRidgeTransformResult(
        z_poison_coral=z_poison_coral,
        z_poison_final=z_poison_final,
        lam=float(lam),
        target_mean=resolved_target_mean,
        beta=float(beta),
    )


@dataclass(frozen=True)
class PreservationMetrics:
    """Per-row and aggregate perturbation-size ("how far did the poison
    embeddings move, and how well is their original direction preserved")
    metrics between the original (untransformed) and transformed poison
    embeddings for one query at one `beta`. See `compute_preservation_metrics`.
    """

    l2_displacements: np.ndarray      # shape (n_poison,): per-row ||z_transformed_i - z_original_i||_2
    original_cosines: np.ndarray      # shape (n_poison,): per-row cos(z_transformed_i, z_original_i)
    mean_l2_displacement: float
    max_l2_displacement: float
    mean_original_cosine: float
    min_original_cosine: float


def compute_preservation_metrics(z_poison_original: np.ndarray, z_poison_transformed: np.ndarray) -> PreservationMetrics:
    """Preservation/displacement metrics between a query's original
    (untransformed) poison embeddings and any transformed version of them
    (e.g. `CoralPcaTransformResult.z_poison_final` at a given `beta`),
    for fairly comparing how much perturbation different oracle
    interventions (E1 anchor-interpolation, CORAL-PCA, later ridge-CORAL /
    MMD) apply to reach a given RAGDefender outcome.

    Both inputs are **L2-normalized before comparison** (per-row), so that:

    - displacement/cosine are computed on the same unit-sphere
      representation RAGDefender itself operates on (cosine similarity),
      not on the embedder's raw, arbitrary-norm output;
    - `beta=0.0` (`z_poison_transformed == coral_pca_transform(..., beta=0.0).z_poison_final
      == normalize(z_poison_original)` exactly) gives **exact** identity:
      `mean/max_l2_displacement == 0.0` and `mean/min_original_cosine == 1.0`,
      not merely approximate.

    Returns a `PreservationMetrics` with both the per-row arrays (`shape
    (n_poison,)`, for tests/fine-grained inspection) and the four
    aggregate scalars used in `CORAL_PCA_SWEEP.csv`:
    `mean_poison_l2_displacement`, `max_poison_l2_displacement`,
    `mean_poison_original_cosine`, `min_poison_original_cosine`.
    """
    z_original = np.asarray(z_poison_original, dtype=np.float64)
    z_transformed = np.asarray(z_poison_transformed, dtype=np.float64)
    if z_original.shape != z_transformed.shape:
        raise ValueError(
            f"z_poison_original and z_poison_transformed must have the same shape; "
            f"got {z_original.shape} vs {z_transformed.shape}"
        )

    z_original_norm = l2_normalize_rows(z_original)
    z_transformed_norm = l2_normalize_rows(z_transformed)

    diffs = z_transformed_norm - z_original_norm
    l2_displacements = np.linalg.norm(diffs, axis=1)
    # Both rows are unit-norm, so the dot product is exactly the cosine
    # similarity; clip only to absorb floating-point noise just outside
    # [-1, 1] (e.g. 1.0000000000000002), never a substantive correction.
    original_cosines = np.clip(np.sum(z_transformed_norm * z_original_norm, axis=1), -1.0, 1.0)

    metrics = PreservationMetrics(
        l2_displacements=l2_displacements,
        original_cosines=original_cosines,
        mean_l2_displacement=float(np.mean(l2_displacements)),
        max_l2_displacement=float(np.max(l2_displacements)),
        mean_original_cosine=float(np.mean(original_cosines)),
        min_original_cosine=float(np.min(original_cosines)),
    )

    if not np.all(np.isfinite(metrics.l2_displacements)) or not np.all(np.isfinite(metrics.original_cosines)):
        raise FloatingPointError("compute_preservation_metrics: non-finite values in displacement/cosine arrays")

    return metrics


# --------------------------------------------------------------------------
# Step 3: direct MMD-minimizing oracle optimizer
# --------------------------------------------------------------------------

DEFAULT_MMD_LR = 0.05


def _pairwise_sq_euclidean_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """`(n_x, n_y)` squared Euclidean distances between rows of `x`/`y`,
    computed **without** an intermediate `sqrt` (unlike `torch.cdist`).
    This matters for autograd correctness here: `torch.cdist(a, a)`'s
    diagonal (`i == j`, distance exactly `0`) has a `sqrt` singularity in
    its backward pass (`d/dx sqrt(x) = 1/(2*sqrt(x))`, undefined at
    `x=0`), which would poison the gradient with `NaN`/`Inf` even though
    the *forward* squared-distance value is a perfectly well-behaved `0`
    -- the biased MMD estimator's `k_pp`/`k_cc` terms always include this
    diagonal (self-distance) by construction. Computing the squared
    distance directly via `((x_i - y_j)**2).sum()` has a well-defined
    (zero) gradient at `x_i == y_j`, avoiding this pitfall entirely."""
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n_x, n_y, d)
    return (diff ** 2).sum(dim=-1)


def _mmd_rbf_squared_torch(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Differentiable **biased** squared RBF-MMD between rows of `x`
    (`n_x, d`) and `y` (`n_y, d`):

        MMD^2(x, y) = mean(k_xx) + mean(k_yy) - 2 * mean(k_xy)
        k(a, b) = exp(-gamma * ||a - b||^2)

    Mathematically the same estimator as
    `defense.distribution_metrics.mmd_rbf_distance_from_gram` (verified
    numerically equal in `tests/test_mmd_intervention.py` when `x`/`y` are
    both unit-norm, since `mmd_rbf_distance_from_gram` also uses
    `||a-b||^2 = 2 - 2*cos(a,b)` for unit vectors, the same identity this
    function implements directly on raw row vectors) -- kept as a
    separate, PyTorch-native, differentiable implementation here purely
    because `mmd_rbf_distance_from_gram` operates on precomputed
    numpy Gram-matrix blocks, not on autograd-tracked tensors."""
    def kernel_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d2 = _pairwise_sq_euclidean_torch(a, b)
        return torch.exp(-gamma * d2).mean()

    return kernel_mean(x, x) + kernel_mean(y, y) - 2.0 * kernel_mean(x, y)


def mmd_rbf_squared_raw(z_x: np.ndarray, z_y: np.ndarray, gamma: float = DEFAULT_MMD_GAMMA) -> float:
    """Non-differentiable convenience wrapper around
    `_mmd_rbf_squared_torch` for plain numpy arrays (no gradient tracking)
    -- used by tests and by callers that want the biased RBF-MMD directly
    from raw (not necessarily unit-norm) embedding rows, without going
    through a cosine/Gram matrix first."""
    x_t = torch.tensor(np.asarray(z_x, dtype=np.float64))
    y_t = torch.tensor(np.asarray(z_y, dtype=np.float64))
    with torch.no_grad():
        return float(_mmd_rbf_squared_torch(x_t, y_t, gamma).item())


@dataclass(frozen=True)
class MmdTraceStep:
    """One row of an MMD optimization trace (the numerical-only fields
    `mmd_minimize_transform` itself can compute; RAGDefender-specific
    fields like `top_pair_pp`/`decision_label` are added by the caller
    via the `on_step` callback -- see `mmd_minimize_transform`)."""

    step: int
    z_poison_step: np.ndarray  # (n_poison, d), unit-norm, the optimizer's state at this step
    mmd_loss: float
    preservation_loss: float
    total_loss: float


@dataclass(frozen=True)
class MmdTransformResult:
    """Every intermediate of one `mmd_minimize_transform` call, for
    logging and for the `steps=0` identity check (§ tests)."""

    z_poison_final: np.ndarray  # final (unit-norm) transformed poison embeddings
    trace: List[MmdTraceStep]   # one entry per step, 0..steps inclusive (steps+1 entries)
    lambda_preserve: float
    gamma: float
    steps: int
    lr: float


def mmd_minimize_transform(z_poison: np.ndarray, z_clean: np.ndarray, lambda_preserve: float,
                            gamma: float = DEFAULT_MMD_GAMMA, steps: int = 0, lr: float = DEFAULT_MMD_LR,
                            seed: Optional[int] = None,
                            on_step: Optional[Callable[[int, np.ndarray, float, float, float], None]] = None
                            ) -> MmdTransformResult:
    """Direct gradient-based MMD-minimizing oracle transform (Step 3 of
    the CORAL/MMD oracle intervention plan). `z_clean` is **read-only**,
    same data-flow contract as `coral_pca_transform`/`coral_ridge_transform`.

    Unlike Steps 1/2 (which align covariance statistics in closed form),
    this directly optimizes the transformed poison embeddings `Zp_prime`
    with PyTorch autograd to minimize

        L = MMD_RBF(Zp_prime, Zc) + lambda_preserve * mean_i(||Zp_prime_i - Zp_original_i||^2)

    where `MMD_RBF` is the biased squared RBF-MMD (`_mmd_rbf_squared_torch`)
    and the second (preservation) term is the **mean over poison rows** of
    each row's squared L2 displacement from its original position (an
    average, not a raw Frobenius-norm sum, so `lambda_preserve` stays in a
    comparable numeric range to the `[0, ~2]`-bounded MMD term regardless
    of `n_poison`/`d`).

    **Both `Zp_original` and `Zc` are L2-normalized once up front, and
    `Zp_prime` is re-projected onto the unit sphere after every optimizer
    step** (`Zp_prime /= ||Zp_prime||` per row) -- this is the "project
    back to the unit sphere" option from the plan (simpler than an added
    penalty term), and it keeps the entire optimization in the same
    unit-sphere representation `compute_preservation_metrics` and
    RAGDefender's own cosine-similarity Stage 1/2 logic already operate
    on (rather than the arbitrary-norm raw embedding space CORAL's
    covariance alignment uses) -- a deliberate, documented design choice,
    not an oversight relative to Steps 1/2.

    At `steps=0`, the optimization loop never runs and `z_poison_final ==
    normalize(z_poison)` **exactly** -- the identity/sanity control (same
    role as `beta=0`/`alpha=1.0` for CORAL/E1). The returned `trace`
    always has `steps + 1` entries (`step=0` through `step=steps`
    inclusive): `step=0` is always the pre-optimization identity state
    (`preservation_loss == 0.0` there, exactly, since `Zp_prime ==
    Zp_original` at that point).

    If `on_step` is given, it is called once per trace entry as
    `on_step(step, z_poison_step, mmd_loss, preservation_loss,
    total_loss)` with the optimizer's **current** (already unit-normalized)
    poison embeddings at that step -- this lets a caller (e.g.
    `scripts/run_mmd_oracle_intervention.py`) recombine `z_poison_step`
    with the query's fixed clean embeddings and recompute
    RAGDefender's Stage 1/2 decision at every step for a full per-step
    trace CSV, without this module needing to know anything about
    `k`/`doc_ids`/`is_poison`/Stage 1/2 itself (same separation-of-concerns
    convention as the rest of this module).

    Raises `ValueError` for `<1` row in either group, mismatched embedding
    dims, `gamma <= 0`, `steps < 0`, or `lr <= 0`. Raises
    `FloatingPointError` if the final transformed embeddings are
    non-finite (should be unreachable for finite inputs and a
    finite-magnitude `lr`, but asserted explicitly).
    """
    z_poison = np.asarray(z_poison, dtype=np.float64)
    z_clean = np.asarray(z_clean, dtype=np.float64)
    n_poison, n_clean = z_poison.shape[0], z_clean.shape[0]
    if n_poison < 1 or n_clean < 1:
        raise ValueError(
            f"mmd_minimize_transform requires >= 1 row per group; got "
            f"n_poison={n_poison}, n_clean={n_clean}."
        )
    if z_poison.shape[1] != z_clean.shape[1]:
        raise ValueError(
            f"z_poison and z_clean must share embedding dim; got {z_poison.shape[1]} vs {z_clean.shape[1]}"
        )
    if gamma <= 0:
        raise ValueError(f"mmd_minimize_transform requires gamma > 0; got {gamma}")
    if steps < 0:
        raise ValueError(f"mmd_minimize_transform requires steps >= 0; got {steps}")
    if lr <= 0:
        raise ValueError(f"mmd_minimize_transform requires lr > 0; got {lr}")

    zp_unit = l2_normalize_rows(z_poison)

    if seed is not None:
        torch.manual_seed(seed)

    zp_t = torch.tensor(zp_unit, dtype=torch.float64, requires_grad=True)
    zp_orig_t = torch.tensor(zp_unit, dtype=torch.float64)  # fixed reference for the preservation term
    zc_t = torch.tensor(l2_normalize_rows(z_clean), dtype=torch.float64)  # fixed target

    trace: List[MmdTraceStep] = []

    def _record(step: int) -> None:
        with torch.no_grad():
            mmd_val = _mmd_rbf_squared_torch(zp_t, zc_t, gamma)
            preserve_val = torch.mean(torch.sum((zp_t - zp_orig_t) ** 2, dim=1))
            total_val = mmd_val + lambda_preserve * preserve_val
        z_step = zp_t.detach().numpy().astype(np.float64).copy()
        mmd_f, preserve_f, total_f = float(mmd_val.item()), float(preserve_val.item()), float(total_val.item())
        trace.append(MmdTraceStep(step=step, z_poison_step=z_step, mmd_loss=mmd_f,
                                   preservation_loss=preserve_f, total_loss=total_f))
        if on_step is not None:
            on_step(step, z_step, mmd_f, preserve_f, total_f)

    _record(0)  # pre-optimization identity state; preservation_loss == 0.0 exactly here

    if steps > 0:
        optimizer = torch.optim.SGD([zp_t], lr=lr)
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            mmd_loss_t = _mmd_rbf_squared_torch(zp_t, zc_t, gamma)
            preserve_loss_t = torch.mean(torch.sum((zp_t - zp_orig_t) ** 2, dim=1))
            total_loss_t = mmd_loss_t + lambda_preserve * preserve_loss_t
            total_loss_t.backward()
            optimizer.step()
            with torch.no_grad():
                zp_t /= zp_t.norm(dim=1, keepdim=True).clamp_min(1e-12)
            _record(step)

    z_poison_final = zp_t.detach().numpy().astype(np.float64)
    if not np.all(np.isfinite(z_poison_final)):
        raise FloatingPointError("mmd_minimize_transform: z_poison_final contains non-finite values")

    return MmdTransformResult(
        z_poison_final=z_poison_final,
        trace=trace,
        lambda_preserve=float(lambda_preserve),
        gamma=float(gamma),
        steps=int(steps),
        lr=float(lr),
    )
