"""CORAL-style formal oracle interventions for Cluster-Normalized Poisoning.

**Steps 1 and 2** of the CORAL/MMD oracle intervention plan (see
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

Explicitly **out of scope for this module** (deferred to a later step of
that plan, not implemented here): the MMD-minimizing gradient-based oracle
optimizer.

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
    "CoralRidgeTransformResult",
    "coral_ridge_transform",
    "PreservationMetrics",
    "compute_preservation_metrics",
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
