"""Distribution-matching diagnostic metrics (CORAL distance, RBF-kernel MMD)
for the Cluster-Normalized Poisoning oracle batch analysis (see
`docs/CLUSTER_NORMALIZED_POISONING_EXECUTION_PLAN.md`).

**Diagnostic metrics only.** This module computes CORAL/MMD *distances*
between the poison and clean embedding groups; it does **not** implement a
CORAL feature-alignment transform or an MMD-minimizing optimizer (candidates
B/C in the execution plan remain deferred, per that plan and the current
batch-analysis task). It never calls an LLM/API and never loads an embedder
-- it operates purely on already-saved cosine-similarity ("Gram") matrices.

**Why a cosine-similarity matrix is enough (no raw embeddings needed):**
`sentence_transformers.util.cos_sim(a, b)` (and this repo's own numpy
equivalents) always L2-normalize their inputs before taking the dot
product, so any `k x k` cosine similarity matrix `M` is *exactly* the Gram
matrix of the L2-normalized embedding *directions* -- `M[i, j] = z_hat_i .
z_hat_j` with `||z_hat_i|| = 1` -- regardless of the raw embeddings' scale.
Two standard identities then let us recover distribution-level statistics
(covariances, RBF kernels) of those unit-norm directions purely from `M`,
without ever needing the underlying `d`-dimensional embedding vectors:

1. For unit-norm vectors, `||z_hat_i - z_hat_j||^2 = 2 - 2 * M[i, j]`
   exactly (`squared_euclidean_from_cosine`) -- this feeds the RBF kernel
   used by `mmd_rbf_distance_from_gram`.
2. A group's mean-centered covariance matrix `C = Z_c^T Z_c / (n - 1)`
   (`Z_c` = mean-centered rows) has the same nonzero eigenvalues as its
   mean-centered Gram matrix `Z_c Z_c^T = H @ G @ H` (`H` = the standard
   kernel-PCA centering matrix, `G` = the un-centered Gram block, `@` =
   matrix product) -- so `trace(C^2) = ||H @ G @ H||_F^2 / (n - 1)^2`, and
   the cross term `trace(C_p @ C_c)` reduces analogously using the
   *doubly*-centered poison-clean cross Gram block. `coral_distance_from_gram`
   assembles the full CORAL distance `||Cov(P) - Cov(C)||_F^2` from exactly
   these three traces. This identity was verified numerically against a
   direct from-embeddings CORAL computation before being adopted here (see
   `tests/test_distribution_metrics.py`).

Because these are computed on the L2-normalized embedding *directions*
(exactly the representation RAGDefender's Stage 1/2 cosine-similarity logic
operates on), not on raw pre-normalization embedding vectors, they are
arguably a *more* faithful diagnostic of "what RAGDefender sees" than a
raw-embedding CORAL/MMD would be -- not merely a workaround for missing
data.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "squared_euclidean_from_cosine",
    "slice_gram_blocks",
    "coral_distance_from_gram",
    "mmd_rbf_distance_from_gram",
    "DEFAULT_MMD_GAMMA",
]

DEFAULT_MMD_GAMMA = 1.0


def squared_euclidean_from_cosine(cos: np.ndarray) -> np.ndarray:
    """`||x - y||^2 = 2 - 2*cos(x, y)`, exact for unit-norm `x`, `y` (see
    module docstring). Clips tiny negative floating-point noise (cosine
    values fractionally above 1.0 due to rounding) to zero."""
    cos = np.asarray(cos, dtype=np.float64)
    d2 = 2.0 - 2.0 * cos
    return np.clip(d2, 0.0, None)


def slice_gram_blocks(m: np.ndarray, poison_indices, clean_indices) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a `k x k` cosine/Gram matrix `m` into its poison-poison
    (`g_pp`), poison-clean (`g_pc`), and clean-clean (`g_cc`) blocks, given
    the original-order index lists from
    `defense.cluster_normalized_poisoning.split_poison_clean`."""
    m = np.asarray(m, dtype=np.float64)
    poison_indices = list(poison_indices)
    clean_indices = list(clean_indices)
    g_pp = m[np.ix_(poison_indices, poison_indices)]
    g_pc = m[np.ix_(poison_indices, clean_indices)]
    g_cc = m[np.ix_(clean_indices, clean_indices)]
    return g_pp, g_pc, g_cc


def _centering_matrix(n: int) -> np.ndarray:
    return np.eye(n) - np.ones((n, n)) / n


def coral_distance_from_gram(g_pp: np.ndarray, g_pc: np.ndarray, g_cc: np.ndarray) -> float:
    """Squared-Frobenius CORAL distance `||Cov(P) - Cov(C)||_F^2` between
    the poison group `P` and clean group `C`, computed only from the
    poison-poison (`g_pp`, `n_p x n_p`), poison-clean (`g_pc`, `n_p x n_c`),
    and clean-clean (`g_cc`, `n_c x n_c`) blocks of a cosine/Gram matrix --
    see the module docstring for the derivation. Both groups' covariances
    are mean-centered (standard CORAL convention).

    Requires `n_p >= 2` and `n_c >= 2` (a mean-centered covariance is
    undefined for a single point); raises `ValueError` otherwise. The
    result is always `>= 0` (up to floating-point noise, which is clipped),
    since it is a sum-of-squares Frobenius norm by construction; exactly
    `0.0` iff the two groups have identical mean-centered covariance
    structure (e.g. `P` and `C` are the same points)."""
    g_pp = np.asarray(g_pp, dtype=np.float64)
    g_pc = np.asarray(g_pc, dtype=np.float64)
    g_cc = np.asarray(g_cc, dtype=np.float64)
    n_p, n_c = g_pp.shape[0], g_cc.shape[0]
    if n_p < 2 or n_c < 2:
        raise ValueError(
            f"coral_distance_from_gram requires at least 2 points per group "
            f"(mean-centered covariance is undefined for n<2); got n_p={n_p}, n_c={n_c}."
        )
    h_p, h_c = _centering_matrix(n_p), _centering_matrix(n_c)
    g_pp_c = h_p @ g_pp @ h_p
    g_cc_c = h_c @ g_cc @ h_c
    g_pc_c = h_p @ g_pc @ h_c

    trace_pp = float(np.sum(g_pp_c ** 2)) / (n_p - 1) ** 2
    trace_cc = float(np.sum(g_cc_c ** 2)) / (n_c - 1) ** 2
    trace_pc = float(np.sum(g_pc_c ** 2)) / ((n_p - 1) * (n_c - 1))

    distance = trace_pp - 2.0 * trace_pc + trace_cc
    return max(distance, 0.0)  # guard tiny floating-point negatives


def mmd_rbf_distance_from_gram(g_pp: np.ndarray, g_pc: np.ndarray, g_cc: np.ndarray,
                                gamma: float = DEFAULT_MMD_GAMMA) -> float:
    """Biased RBF-kernel MMD^2 estimate between the poison group `P` and
    clean group `C`:

        MMD^2(P, C) = mean(k_pp) + mean(k_cc) - 2 * mean(k_pc)
        k(x, y) = exp(-gamma * ||x - y||^2)

    computed directly from the poison-poison/poison-clean/clean-clean
    cosine/Gram blocks via `squared_euclidean_from_cosine` (exact for the
    L2-normalized embedding directions a cosine matrix represents -- see
    module docstring). This is the standard *biased* estimator (its P-P and
    C-C sums include the `i == j` diagonal terms, where `k(x, x) = 1`); for
    the small, fixed group sizes here (typically `N=5` per group) the
    biased/unbiased distinction is numerically minor and the biased form is
    simpler to reason about.

    `gamma` is a **fixed, lightweight default** (not a per-query
    median-heuristic bandwidth) -- a deliberate simplicity choice for a
    diagnostic-only metric; see the execution plan's limitations. Always
    `>= 0` for any valid kernel bandwidth `gamma > 0`, since it is
    (empirically) exactly `||mean_embedding(P) - mean_embedding(C)||^2` in
    the RBF-induced reproducing kernel Hilbert space; `0.0` iff `P` and `C`
    are the same points (identical empirical kernel-mean embedding)."""
    def kernel_mean(cos_block: np.ndarray) -> float:
        d2 = squared_euclidean_from_cosine(cos_block)
        return float(np.mean(np.exp(-gamma * d2)))

    mmd2 = kernel_mean(g_pp) + kernel_mean(g_cc) - 2.0 * kernel_mean(g_pc)
    return max(mmd2, 0.0)  # guard tiny floating-point negatives
