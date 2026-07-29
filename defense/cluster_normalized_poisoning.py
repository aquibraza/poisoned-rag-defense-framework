"""Cluster-Normalized Poisoning: oracle embedding-space interventions.

Pure-math, side-by-side reimplementation companion to
`defense/ragdefender_internals.py`, for the oracle stress test described in
`docs/CLUSTER_NORMALIZED_POISONING_EXECUTION_PLAN.md`. This module is purely
additive: it is never imported by `defense/defense_runner.py`,
`defense/dispatch.py`, `defense/filterrag.py`, or `main.py`, and it never
calls an LLM/API. It only transforms already-encoded passage embeddings for
a single retrieved-context query and hands the result to the *unmodified*
`defense/ragdefender_internals.py::concentration_stage1` /
`stage2_pair_frequency` functions.

Interventions implemented (see the execution plan, section 4):

- **E0** (`centroid_interpolate`): clean-centroid interpolation baseline.
  Every poisoned embedding is pulled toward the *same* shared clean
  centroid. Kept only as a sanity baseline -- it can leave poison-poison
  similarity unchanged or even increase it (see module docstring caveat in
  the plan), so it is not treated as evidence of RAGDefender's fragility on
  its own.
- **E1** (`anchor_interpolate`, with `resolve_anchor_permutation`): clean-anchor
  interpolation. Each poisoned embedding is pulled toward a *different*
  clean anchor, chosen by one of four assignment strategies
  (`rank_aligned`, `nearest_bijection`, `farthest_bijection`, `random`).
  `nearest_bijection`/`farthest_bijection` solve the assignment problem by
  brute-force enumeration over `itertools.permutations` (tractable only
  because `N_poison` is small, e.g. 5, in the anchor queries this plan
  targets) -- **no SciPy dependency is introduced**.

Both interventions transform poisoned embeddings only; clean embeddings are
always passed through unmodified by the caller (this module never receives
or needs the full retrieved-context embedding matrix, only the poison/clean
sub-blocks already split by the caller).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ANCHOR_STRATEGIES",
    "AnchorAssignment",
    "l2_normalize_rows",
    "split_poison_clean",
    "recombine_poison_clean",
    "centroid_interpolate",
    "resolve_anchor_permutation",
    "anchor_interpolate",
    "is_bijection",
]

ANCHOR_STRATEGIES = ("rank_aligned", "nearest_bijection", "farthest_bijection", "random")
AnchorStrategy = Literal["rank_aligned", "nearest_bijection", "farthest_bijection", "random"]


def l2_normalize_rows(z: np.ndarray) -> np.ndarray:
    """L2-normalize each row of `z`. Purely cosmetic for this module's own
    callers -- `sentence_transformers.util.cos_sim` (used downstream by
    `scripts/run_cluster_normalized_poisoning.py`) already L2-normalizes
    its inputs internally, so this does not change any `cos_sim` value; it
    exists only for representational clarity, matching the plan's stated
    formulas (`z' = z' / ||z'||_2`)."""
    z = np.asarray(z, dtype=np.float64)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return z / norms


def split_poison_clean(z: np.ndarray, is_poison: Sequence[bool]) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """Split a `k x dim` embedding matrix into poison/clean sub-blocks,
    preserving original row order within each block. Returns
    `(z_poison, z_clean, poison_indices, clean_indices)` where the index
    lists are positions in the original `k`-length axis, so the split is
    always invertible via `recombine_poison_clean`."""
    z = np.asarray(z, dtype=np.float64)
    poison_indices = [i for i, p in enumerate(is_poison) if bool(p)]
    clean_indices = [i for i, p in enumerate(is_poison) if not bool(p)]
    return z[poison_indices], z[clean_indices], poison_indices, clean_indices


def recombine_poison_clean(
    z_poison: np.ndarray,
    z_clean: np.ndarray,
    poison_indices: Sequence[int],
    clean_indices: Sequence[int],
    k: int,
) -> np.ndarray:
    """Inverse of `split_poison_clean`: place `z_poison`/`z_clean` rows back
    at their original `k`-length positions."""
    if z_poison.shape[0] != len(poison_indices):
        raise ValueError("z_poison row count must match len(poison_indices)")
    if z_clean.shape[0] != len(clean_indices):
        raise ValueError("z_clean row count must match len(clean_indices)")
    dim = z_poison.shape[1] if z_poison.shape[0] else z_clean.shape[1]
    out = np.zeros((k, dim), dtype=np.float64)
    for row, idx in zip(z_poison, poison_indices):
        out[idx] = row
    for row, idx in zip(z_clean, clean_indices):
        out[idx] = row
    return out


# --------------------------------------------------------------------------
# E0: clean-centroid interpolation baseline (sanity baseline only)
# --------------------------------------------------------------------------

def centroid_interpolate(z_poison: np.ndarray, z_clean: np.ndarray, alpha: float) -> np.ndarray:
    """E0: `z'_poison = alpha * z_poison + (1 - alpha) * clean_centroid`,
    followed by L2-renormalization. `clean_centroid = mean(z_clean)`.

    **Known failure mode (see execution plan section 4/10):** pulling every
    poisoned row toward the same shared point can leave poison-poison
    pairwise similarity unchanged or even increase it. This is expected and
    is why E0 is only a sanity baseline, not the primary intervention.
    """
    z_poison = np.asarray(z_poison, dtype=np.float64)
    z_clean = np.asarray(z_clean, dtype=np.float64)
    if z_clean.shape[0] == 0:
        raise ValueError("centroid_interpolate: z_clean must be non-empty")
    clean_centroid = z_clean.mean(axis=0)
    transformed = alpha * z_poison + (1.0 - alpha) * clean_centroid[None, :]
    return l2_normalize_rows(transformed)


# --------------------------------------------------------------------------
# E1: clean-anchor interpolation (primary intervention)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorAssignment:
    """Resolved poison-to-clean anchor assignment for E1, plus a
    machine-checkable bijectivity record (this is the direct regression
    guard for the bug the execution plan's Rev 2 fixed: an earlier
    independent-argmax/argmin `nearest`/`farthest` definition could map
    multiple poison points to the same clean anchor)."""

    strategy: str
    pi: List[int]  # length N_poison; pi[i] = local clean index z_poison[i] is anchored to
    random_seed: Optional[int]
    is_bijection: bool
    objective_value: Optional[float]  # sum_i cos(z_p_i, z_c_pi(i)) at the chosen pi, for nearest/farthest_bijection only


def is_bijection(pi: Sequence[int], n: int) -> bool:
    """True iff `pi` is a permutation of `range(n)` (a bijection
    `{0..n-1} -> {0..n-1}`): same length as `n`, every value in range, no
    repeats."""
    pi = list(pi)
    return len(pi) == n and sorted(pi) == list(range(n))


def _pairwise_cosine(z_poison: np.ndarray, z_clean: np.ndarray) -> np.ndarray:
    """`N_poison x N_clean` cosine similarity matrix, computed directly
    with numpy (not `sentence_transformers.util.cos_sim`) since this
    function only needs to *choose* an assignment -- it never feeds a
    result into `defense/ragdefender_internals.py`. Mathematically
    identical to `util.cos_sim` (both L2-normalize then dot)."""
    zp = l2_normalize_rows(z_poison)
    zc = l2_normalize_rows(z_clean)
    return zp @ zc.T


def _brute_force_best_permutation(cos_matrix: np.ndarray, maximize: bool) -> Tuple[List[int], float]:
    """Exhaustively search all `N!` permutations of clean-local indices
    (`N = cos_matrix.shape[1]`) for the one that maximizes (or minimizes)
    `sum_i cos_matrix[i, pi(i)]`. Only tractable for small `N` (e.g. 5,
    N! = 120) -- see the execution plan's explicit `O(N!)` limitation.
    Uses `itertools.permutations` (stdlib only); no SciPy dependency.
    """
    n_poison, n_clean = cos_matrix.shape
    if n_poison != n_clean:
        raise ValueError(
            f"_brute_force_best_permutation requires N_poison == N_clean for a bijection "
            f"(got {n_poison} poison, {n_clean} clean); the execution plan's E1 "
            f"assignment strategies are only defined for equal-size groups."
        )
    best_pi: Optional[List[int]] = None
    best_score = -np.inf if maximize else np.inf
    for perm in itertools.permutations(range(n_clean)):
        score = float(sum(cos_matrix[i, perm[i]] for i in range(n_poison)))
        better = (score > best_score) if maximize else (score < best_score)
        if better:
            best_score = score
            best_pi = list(perm)
    assert best_pi is not None  # n_clean >= 1 is guaranteed by caller
    return best_pi, best_score


def resolve_anchor_permutation(
    z_poison: np.ndarray,
    z_clean: np.ndarray,
    strategy: AnchorStrategy,
    random_seed: Optional[int] = None,
) -> AnchorAssignment:
    """Resolve the poison-to-clean anchor assignment `pi` for E1, for one of
    the four strategies defined in the execution plan (section 4):

    - `rank_aligned`: `pi(i) = i` (requires `N_poison == N_clean`).
    - `nearest_bijection`: the permutation maximizing
      `sum_i cos(z_p_i, z_c_pi(i))`, found by brute force.
    - `farthest_bijection`: the permutation minimizing the same sum.
    - `random`: a uniformly random permutation from `random_seed`.

    Every branch returns an `AnchorAssignment` whose `pi` is verified to be
    a bijection (`is_bijection=True`) before returning -- if it is not,
    this function raises rather than silently returning an invalid
    assignment.
    """
    z_poison = np.asarray(z_poison, dtype=np.float64)
    z_clean = np.asarray(z_clean, dtype=np.float64)
    n_poison, n_clean = z_poison.shape[0], z_clean.shape[0]

    if strategy not in ANCHOR_STRATEGIES:
        raise ValueError(f"Unknown anchor_strategy {strategy!r}; must be one of {ANCHOR_STRATEGIES}")
    if n_poison != n_clean:
        raise ValueError(
            f"resolve_anchor_permutation requires N_poison == N_clean for a bijective "
            f"assignment (got {n_poison} poison, {n_clean} clean)."
        )

    objective_value: Optional[float] = None
    if strategy == "rank_aligned":
        pi = list(range(n_poison))
    elif strategy == "random":
        rng = np.random.default_rng(random_seed)
        pi = list(rng.permutation(n_clean))
    else:
        cos_matrix = _pairwise_cosine(z_poison, z_clean)
        maximize = strategy == "nearest_bijection"
        pi, objective_value = _brute_force_best_permutation(cos_matrix, maximize=maximize)

    bijection_ok = is_bijection(pi, n_clean)
    if not bijection_ok:
        raise RuntimeError(
            f"resolve_anchor_permutation produced a non-bijective pi for strategy={strategy!r}: {pi!r}. "
            "This should be unreachable -- every strategy is constructed to be a permutation by "
            "definition; this indicates a bug in this function, not in the caller."
        )

    return AnchorAssignment(
        strategy=strategy,
        pi=[int(x) for x in pi],
        random_seed=random_seed if strategy == "random" else None,
        is_bijection=bijection_ok,
        objective_value=objective_value,
    )


def anchor_interpolate(z_poison: np.ndarray, z_clean: np.ndarray, pi: Sequence[int], alpha: float) -> np.ndarray:
    """E1: `z'_p_i = alpha * z_p_i + (1 - alpha) * z_c_{pi(i)}`, followed by
    L2-renormalization, for each poisoned row `i` and its assigned clean
    anchor `pi[i]` (from `resolve_anchor_permutation`)."""
    z_poison = np.asarray(z_poison, dtype=np.float64)
    z_clean = np.asarray(z_clean, dtype=np.float64)
    pi = list(pi)
    if len(pi) != z_poison.shape[0]:
        raise ValueError(f"len(pi)={len(pi)} must equal z_poison.shape[0]={z_poison.shape[0]}")
    anchors = z_clean[pi]  # N_poison x dim, row i = z_clean[pi[i]]
    transformed = alpha * z_poison + (1.0 - alpha) * anchors
    return l2_normalize_rows(transformed)
