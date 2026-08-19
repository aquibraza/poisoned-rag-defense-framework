"""Regime-B Stage-1 boundary-sensitivity oracle -- pure, offline library.

Implements PHASES 1-5 of the "REGIME-B STAGE-1 BOUNDARY-SENSITIVITY ORACLE"
task as a set of testable pure functions operating on already-computed
Stage-1 statistics (`defense.ragdefender_internals.concentration_stage1_paper`
output) and raw similarity matrices. No file I/O, no Stella, no retrieval,
no network -- this module never touches disk.

==========================================================================
CONCEPTUAL RULE (see module callers / report for full statement)
==========================================================================
Stage-1 `adv_flag` indices are NOT predicted-poisoned passages -- they only
contribute to the INTEGER count `N_adv`. Every function in this module that
selects a PRIMARY candidate does so using ONLY Stage-1 margins/flags, never
`is_poison`. Functions that accept an `is_poison` array (for diagnostic
labeling AFTER selection) document this explicitly in their docstring.

==========================================================================
PHASES
==========================================================================
1.  `passage_margins` -- per-passage mean/median margins + flags (Phase 1).
1A. `classify_binding_condition` -- A/B/C/D boundary classification.
1B. `median_rank_gap_analysis` -- rank5/rank6/gap + near-tie counts.
1C. `mean_gate_candidates` -- above_median & !above_mean passage indices.
2.  `success_vs_failure_row` -- per-query descriptive-statistics row.
3.  `statistic_space_oracle_for_query` -- idealized vector-space minimal
    perturbation (mean-only / median-only / combined arms).
4.  `matrix_oracle_for_candidate` -- realizable *symmetric-matrix*-space
    minimal alpha via monotonic bisection or deterministic grid fallback.
5.  `stage2_causal_check` -- rerun the UNCHANGED Stage 2 on a perturbed
    matrix at the fixed n_adv=5 and classify A (count-fix-only) vs.
    B (count-fix-but-Stage2-changes).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from defense import ragdefender_internals as ri

EPS = 1e-9  # strict-inequality safety margin added on top of exact crossing points


# ---------------------------------------------------------------------------
# PHASE 1 -- exact Stage-1 boundary decomposition
# ---------------------------------------------------------------------------

def passage_margins(stage1: "ri.ConcentrationResultPaper") -> List[dict]:
    """Per-passage margins/flags (Phase 1). Never touches `is_poison`."""
    k = len(stage1.s_mean)
    rows = []
    for i in range(k):
        rows.append(
            {
                "index": i,
                "s_mean_i": float(stage1.s_mean[i]),
                "s_median_i": float(stage1.s_median[i]),
                "mean_margin_i": float(stage1.s_mean[i] - stage1.s_bar),
                "median_margin_i": float(stage1.s_median[i] - stage1.s_tilde),
                "above_mean_i": bool(stage1.above_mean[i]),
                "above_median_i": bool(stage1.above_median[i]),
                "and_flag_i": bool(stage1.adv_flag[i]),
            }
        )
    return rows


def query_level_counts(stage1: "ri.ConcentrationResultPaper") -> dict:
    return {
        "n_above_mean": int(stage1.above_mean.sum()),
        "n_above_median": int(stage1.above_median.sum()),
        "n_and": int(stage1.n_adv_estimated),
    }


# ---------------------------------------------------------------------------
# PHASE 1A -- classify the binding Stage-1 count-boundary condition
# ---------------------------------------------------------------------------

BINDING_CEILING_REACHED = "D. CEILING-REACHED"
BINDING_MEDIAN_LIMITED = "A. MEDIAN-LIMITED"
BINDING_MEAN_GATED = "B. MEAN-GATED"
BINDING_BOTH_LIMITED = "C. BOTH-LIMITED"

VALID_BINDING_LABELS = (
    BINDING_MEDIAN_LIMITED,
    BINDING_MEAN_GATED,
    BINDING_BOTH_LIMITED,
    BINDING_CEILING_REACHED,
)


def classify_binding_condition(n_above_median: int, n_and: int, ceiling: int = 5) -> str:
    """Priority-ordered classification (Phase 1A). `ceiling` is `floor(k/2)`
    (5 for k=10, the Regime-B structural ceiling)."""
    if n_and >= ceiling:
        return BINDING_CEILING_REACHED
    if n_above_median < ceiling:
        return BINDING_MEDIAN_LIMITED
    if n_above_median == ceiling and n_and < ceiling:
        return BINDING_MEAN_GATED
    # n_above_median > ceiling is structurally impossible (at most `ceiling`
    # values can be strictly greater than the `ceiling`-th order statistic
    # of a `2*ceiling`-length vector) -- if we ever reach here, the observed
    # configuration does not fit A or B cleanly.
    return BINDING_BOTH_LIMITED


# ---------------------------------------------------------------------------
# PHASE 1B -- median tie / separation analysis
# ---------------------------------------------------------------------------

def median_rank_gap_analysis(s_median: np.ndarray, tolerances=(1e-8, 1e-6, 1e-4)) -> dict:
    """Rank5/rank6/gap under the torch-style lower-middle convention, plus
    near-tie counts around the threshold at several diagnostic tolerances.
    Purely descriptive -- does not change production comparison semantics.
    """
    s_median = np.asarray(s_median, dtype=np.float64)
    k = s_median.shape[0]
    sorted_vals = np.sort(s_median)
    idx_rank5 = (k - 1) // 2  # torch-style median index (0-indexed), e.g. 4 for k=10
    idx_rank6 = idx_rank5 + 1
    rank5 = float(sorted_vals[idx_rank5])
    rank6 = float(sorted_vals[idx_rank6]) if idx_rank6 < k else None
    median_gap = (rank6 - rank5) if rank6 is not None else None
    s_tilde = ri._torch_style_median_1d(s_median)  # noqa: SLF001
    assert math.isclose(s_tilde, rank5, rel_tol=0, abs_tol=1e-12), "s_tilde must equal rank5 by construction"

    tie_counts = {}
    for tol in tolerances:
        tie_counts[tol] = int(np.sum(np.abs(s_median - s_tilde) <= tol))

    return {
        "median_rank5": rank5,
        "median_rank6": rank6,
        "median_gap": median_gap,
        "rank5_equals_rank6": (rank6 is not None and rank5 == rank6),
        "tie_counts_by_tolerance": tie_counts,
    }


# ---------------------------------------------------------------------------
# PHASE 1C -- mean-gate analysis
# ---------------------------------------------------------------------------

def mean_gate_candidates(stage1: "ri.ConcentrationResultPaper") -> List[int]:
    """Indices where `above_median=True` and `above_mean=False` -- i.e. the
    passages that clear the median condition but are the reason the AND
    count falls short of the median-side ceiling. Selection uses ONLY
    Stage-1 flags, never `is_poison`."""
    return [
        i
        for i in range(len(stage1.s_mean))
        if bool(stage1.above_median[i]) and not bool(stage1.above_mean[i])
    ]


def mean_rank_descending(s_mean: np.ndarray, i: int) -> int:
    """1-indexed rank of `s_mean[i]` among all `s_mean` values, descending
    (rank 1 = largest)."""
    s_mean = np.asarray(s_mean, dtype=np.float64)
    return int(np.sum(s_mean > s_mean[i]) + 1)


# ---------------------------------------------------------------------------
# PHASE 2 -- success vs. failure descriptive comparison
# ---------------------------------------------------------------------------

def passage_shortfall(mean_margin_i: float, median_margin_i: float) -> float:
    """Combined 'distance to satisfying BOTH Stage-1 tests' for a passage
    that does not currently satisfy both -- sum of the (non-negative)
    shortfalls on each test independently. Zero if the passage already
    satisfies both (mean_margin_i>0 and median_margin_i>0)."""
    return max(0.0, -mean_margin_i) + max(0.0, -median_margin_i)


def success_vs_failure_row(stage1: "ri.ConcentrationResultPaper", ceiling: int = 5) -> dict:
    """One query's worth of Phase-2 comparison statistics."""
    margins = passage_margins(stage1)
    counts = query_level_counts(stage1)
    mgap = median_rank_gap_analysis(stage1.s_median)

    mean_margins = np.array([m["mean_margin_i"] for m in margins])
    median_margins = np.array([m["median_margin_i"] for m in margins])

    positive_mean = mean_margins[mean_margins > 0]
    negative_mean = mean_margins[mean_margins <= 0]
    positive_median = median_margins[median_margins > 0]
    negative_median = median_margins[median_margins <= 0]

    shortfalls = [
        passage_shortfall(m["mean_margin_i"], m["median_margin_i"])
        for m in margins
        if not m["and_flag_i"]
    ]

    return {
        "n_above_mean": counts["n_above_mean"],
        "n_above_median": counts["n_above_median"],
        "n_and": counts["n_and"],
        "median_gap": mgap["median_gap"],
        "min_positive_mean_margin": float(positive_mean.min()) if positive_mean.size else None,
        "closest_negative_mean_margin": float(negative_mean.max()) if negative_mean.size else None,
        "min_positive_median_margin": float(positive_median.min()) if positive_median.size else None,
        "closest_negative_median_margin": float(negative_median.max()) if negative_median.size else None,
        "n_near_ties_1e-6": mgap["tie_counts_by_tolerance"].get(1e-6, 0),
        "smallest_shortfall_to_and": min(shortfalls) if shortfalls else 0.0,
    }


def descriptive_summary(values: List[float]) -> dict:
    """median/IQR/range only -- no inferential statistics (Phase 2 rule)."""
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"median": None, "iqr": None, "range": None, "n": 0}
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "median": float(np.median(arr)),
        "iqr": float(q3 - q1),
        "range": float(arr.max() - arr.min()),
        "n": int(arr.size),
    }


# ---------------------------------------------------------------------------
# PHASE 3 -- statistic-space minimal oracle (idealized, not matrix-realizable)
# ---------------------------------------------------------------------------

def minimal_mean_only_delta(s_mean: np.ndarray, i: int) -> Optional[float]:
    """Minimal delta added to `s_mean[i]` ALONE such that, after recomputing
    the global mean `s_bar' = mean(s_mean with index i replaced)`, the
    strict condition `s_mean[i]+delta > s_bar'` holds. Exact closed form
    (s_bar is an exact linear function of delta). Returns 0.0-ish (EPS) if
    already satisfied; the caller is responsible for checking whether this
    arm is even meaningful (see `statistic_space_oracle_for_query`)."""
    s_mean = np.asarray(s_mean, dtype=np.float64)
    k = s_mean.shape[0]
    s_bar = float(s_mean.mean())
    margin = s_bar - float(s_mean[i])  # > 0 means currently below/at threshold
    delta_exact = margin * k / (k - 1)
    delta = max(0.0, delta_exact) + EPS
    # Verify by direct recomputation (defensive; closed form should be exact).
    new_mean = s_mean.copy()
    new_mean[i] += delta
    new_bar = float(new_mean.mean())
    assert new_mean[i] > new_bar, "minimal_mean_only_delta: closed-form crossing point failed verification"
    return float(delta)


def _monotonic_or_grid_search(
    predicate: Callable[[float], bool],
    lo: float = 0.0,
    hi: float = 4.0,
    coarse_steps: int = 400,
    bisection_iters: int = 50,
) -> Tuple[Optional[float], bool, List[Tuple[float, bool]]]:
    """Generic minimal-delta search for a predicate assumed non-decreasing
    (False, ..., False, True, ..., True) as delta increases from `lo`.

    Returns (minimal_delta_or_None, is_monotonic, coarse_path). Verifies
    monotonicity on the coarse grid; if violated, falls back to the FIRST
    grid point where the predicate is True (deterministic, no bisection
    assumption), per the task's "deterministic bounded grid + local
    refinement" fallback instruction.
    """
    grid = np.linspace(lo, hi, coarse_steps + 1)
    path = [(float(d), bool(predicate(d))) for d in grid]

    if not path[-1][1]:
        return None, True, path  # not achievable within [lo, hi]

    # Monotonicity check: no True followed later by a False.
    seen_true = False
    is_monotonic = True
    first_true_idx = None
    for idx, (_d, val) in enumerate(path):
        if val:
            if first_true_idx is None:
                first_true_idx = idx
            seen_true = True
        elif seen_true and not val:
            is_monotonic = False

    if is_monotonic:
        lo_b, hi_b = path[first_true_idx - 1][0] if first_true_idx > 0 else lo, path[first_true_idx][0]
        for _ in range(bisection_iters):
            mid = (lo_b + hi_b) / 2.0
            if predicate(mid):
                hi_b = mid
            else:
                lo_b = mid
        return float(hi_b), True, path

    # Non-monotonic: deterministic grid fallback -- first True on the coarse
    # grid, then a local finer refinement immediately below it (never
    # assuming monotonicity holds in that refinement window either; we just
    # take the smallest delta found in the finer window that is True and
    # immediately preceded by a False in that same finer scan, else keep
    # the coarse value).
    coarse_first_true = path[first_true_idx][0]
    coarse_prev = path[first_true_idx - 1][0] if first_true_idx > 0 else lo
    fine_grid = np.linspace(coarse_prev, coarse_first_true, 200)
    best = coarse_first_true
    for d in fine_grid:
        if predicate(float(d)) and float(d) < best:
            best = float(d)
    return float(best), False, path


def minimal_median_only_delta(
    s_median: np.ndarray, i: int, hi: float = 4.0
) -> Tuple[Optional[float], bool]:
    """Minimal delta added to `s_median[i]` ALONE such that, after
    recomputing the GLOBAL median `s_tilde'` (torch-style lower-middle) over
    the perturbed vector, `s_median[i]+delta > s_tilde'` holds. Order-
    statistic-based (non-smooth step function of delta) -- uses monotonic
    bisection with an empirical monotonicity check, falling back to a
    deterministic grid otherwise. Returns (delta_or_None, is_monotonic)."""
    s_median = np.asarray(s_median, dtype=np.float64)

    def predicate(delta: float) -> bool:
        perturbed = s_median.copy()
        perturbed[i] += delta
        new_tilde = ri._torch_style_median_1d(perturbed)  # noqa: SLF001
        return perturbed[i] > new_tilde

    delta, is_monotonic, _path = _monotonic_or_grid_search(predicate, lo=0.0, hi=hi)
    if delta is None:
        return None, is_monotonic
    return delta + EPS, is_monotonic


@dataclass
class StatisticOracleResult:
    candidate_index: int
    mean_only_delta: Optional[float]
    median_only_delta: Optional[float]
    median_only_monotonic: Optional[bool]
    combined_mean_delta: Optional[float]
    combined_median_delta: Optional[float]
    combined_magnitude: Optional[float]  # Euclidean norm of the two components
    resulting_n_adv_mean_only: Optional[int]
    resulting_n_adv_median_only: Optional[int]
    resulting_n_adv_combined: Optional[int]
    sensitivity_class: str  # mean-sensitive / median-sensitive / both-sensitive / not-repairable


def _recompute_n_adv_after_vector_perturbation(
    s_mean: np.ndarray, s_median: np.ndarray, i: int, mean_delta: float, median_delta: float
) -> int:
    new_mean = s_mean.copy()
    new_mean[i] += mean_delta
    new_median = s_median.copy()
    new_median[i] += median_delta
    new_bar = float(new_mean.mean())
    new_tilde = ri._torch_style_median_1d(new_median)  # noqa: SLF001
    above_mean = new_mean > new_bar
    above_median = new_median > new_tilde
    return int((above_mean & above_median).sum())


def statistic_space_oracle_for_candidate(
    s_mean: np.ndarray, s_median: np.ndarray, i: int
) -> StatisticOracleResult:
    """Phase 3 for ONE candidate passage `i` (assumed currently non-AND).
    Since `s_bar` depends only on the `s_mean` vector and `s_tilde` depends
    only on the `s_median` vector, the mean-only and median-only arms are
    mathematically independent of each other -- this is exploited directly
    rather than approximated."""
    s_mean = np.asarray(s_mean, dtype=np.float64)
    s_median = np.asarray(s_median, dtype=np.float64)
    s_bar = float(s_mean.mean())
    s_tilde = ri._torch_style_median_1d(s_median)  # noqa: SLF001
    currently_above_mean = bool(s_mean[i] > s_bar)
    currently_above_median = bool(s_median[i] > s_tilde)

    mean_delta_needed = minimal_mean_only_delta(s_mean, i) if not currently_above_mean else 0.0
    median_delta_needed, median_monotonic = (
        minimal_median_only_delta(s_median, i) if not currently_above_median else (0.0, True)
    )

    # Mean-only arm is only a valid ROUTE to AND=True if median already holds.
    mean_only_delta = mean_delta_needed if currently_above_median else None
    resulting_n_adv_mean_only = None
    if mean_only_delta is not None:
        resulting_n_adv_mean_only = _recompute_n_adv_after_vector_perturbation(
            s_mean, s_median, i, mean_only_delta, 0.0
        )

    # Median-only arm is only a valid ROUTE to AND=True if mean already holds.
    median_only_delta = median_delta_needed if (currently_above_mean and median_delta_needed is not None) else None
    resulting_n_adv_median_only = None
    if median_only_delta is not None:
        resulting_n_adv_median_only = _recompute_n_adv_after_vector_perturbation(
            s_mean, s_median, i, 0.0, median_only_delta
        )

    # Combined arm: apply BOTH minimal deltas simultaneously (independent by
    # construction) -- always achievable unless the median-only crossing
    # search itself failed to find a delta within its bound.
    combined_mean_delta = mean_delta_needed
    combined_median_delta = median_delta_needed
    combined_magnitude = None
    resulting_n_adv_combined = None
    if combined_median_delta is not None:
        combined_magnitude = float(np.hypot(combined_mean_delta, combined_median_delta))
        resulting_n_adv_combined = _recompute_n_adv_after_vector_perturbation(
            s_mean, s_median, i, combined_mean_delta, combined_median_delta
        )

    if mean_only_delta is not None and resulting_n_adv_mean_only and resulting_n_adv_mean_only >= 5:
        sensitivity_class = "mean-sensitive"
    elif median_only_delta is not None and resulting_n_adv_median_only and resulting_n_adv_median_only >= 5:
        sensitivity_class = "median-sensitive"
    elif resulting_n_adv_combined is not None and resulting_n_adv_combined >= 5:
        sensitivity_class = "both-sensitive"
    else:
        sensitivity_class = "not-repairable"

    return StatisticOracleResult(
        candidate_index=i,
        mean_only_delta=mean_only_delta,
        median_only_delta=median_only_delta,
        median_only_monotonic=median_monotonic if median_delta_needed is not None else None,
        combined_mean_delta=combined_mean_delta,
        combined_median_delta=combined_median_delta,
        combined_magnitude=combined_magnitude,
        resulting_n_adv_mean_only=resulting_n_adv_mean_only,
        resulting_n_adv_median_only=resulting_n_adv_median_only,
        resulting_n_adv_combined=resulting_n_adv_combined,
        sensitivity_class=sensitivity_class,
    )


def statistic_space_oracle_for_query(
    stage1: "ri.ConcentrationResultPaper",
) -> List[StatisticOracleResult]:
    """Phase 3 over every currently-non-AND candidate in one query. Selection
    of candidates is purely defense-native (all non-AND indices), never
    `is_poison`-based."""
    non_and = [i for i in range(len(stage1.s_mean)) if not bool(stage1.adv_flag[i])]
    return [
        statistic_space_oracle_for_candidate(stage1.s_mean, stage1.s_median, i) for i in non_and
    ]


def best_statistic_oracle_result(results: List[StatisticOracleResult]) -> Optional[StatisticOracleResult]:
    """Defense-native selection of the single best (smallest achievable
    single-statistic delta, preferring mean-only/median-only over combined)
    candidate -- never uses `is_poison`."""
    single_arm = [
        r for r in results if r.sensitivity_class in ("mean-sensitive", "median-sensitive")
    ]

    def _cost(r: StatisticOracleResult) -> float:
        if r.sensitivity_class == "mean-sensitive":
            return r.mean_only_delta
        return r.median_only_delta

    if single_arm:
        return min(single_arm, key=_cost)
    combined = [r for r in results if r.sensitivity_class == "both-sensitive"]
    if combined:
        return min(combined, key=lambda r: r.combined_magnitude)
    return None


# ---------------------------------------------------------------------------
# PHASE 4 -- symmetric similarity-matrix oracle
# ---------------------------------------------------------------------------

def perturb_boost(matrix: np.ndarray, i: int, alpha: float) -> np.ndarray:
    """S'_ij = clip(S_ij + alpha, -1, 1) for all j != i, symmetric,
    diagonal unchanged."""
    matrix = np.asarray(matrix, dtype=np.float64)
    perturbed = matrix.copy()
    k = matrix.shape[0]
    for j in range(k):
        if j == i:
            continue
        new_val = np.clip(matrix[i, j] + alpha, -1.0, 1.0)
        perturbed[i, j] = new_val
        perturbed[j, i] = new_val
    return perturbed


def perturb_decrease(matrix: np.ndarray, i: int, alpha: float) -> np.ndarray:
    """S'_ij = clip(S_ij - alpha, -1, 1) for all j != i, symmetric,
    diagonal unchanged."""
    return perturb_boost(matrix, i, -alpha)


def n_adv_after_matrix_perturbation(matrix: np.ndarray) -> int:
    return ri.concentration_stage1_paper(matrix).n_adv_estimated


@dataclass
class MatrixOracleResult:
    candidate_index: int
    mode: str  # "boost" or "decrease"
    alpha: Optional[float]
    is_monotonic: bool
    n_adv_path: List[Tuple[float, int]]
    achieved_n_adv: Optional[int]
    perturbed_matrix: Optional[np.ndarray] = field(default=None, repr=False)


def matrix_oracle_for_candidate(
    matrix: np.ndarray,
    i: int,
    target_n_adv: int = 5,
    mode: str = "boost",
    alpha_max: float = 2.0,
    coarse_steps: int = 200,
) -> MatrixOracleResult:
    """Phase 4A/4B for ONE candidate passage `i`. `mode='boost'` raises
    `S_ij` for all `j != i`; `mode='decrease'` lowers it. Does NOT assume
    monotonicity -- verifies it empirically via a coarse grid over the full
    `N_adv(alpha)` path, falling back to a deterministic grid+local-refine
    if non-monotonic."""
    perturb_fn = perturb_boost if mode == "boost" else perturb_decrease

    def predicate(alpha: float) -> bool:
        return n_adv_after_matrix_perturbation(perturb_fn(matrix, i, alpha)) >= target_n_adv

    delta, is_monotonic, path_bool = _monotonic_or_grid_search(
        predicate, lo=0.0, hi=alpha_max, coarse_steps=coarse_steps
    )
    # Re-derive the full N_adv(alpha) integer path (not just the boolean
    # predicate) at the same grid resolution for non-monotonicity reporting.
    n_adv_path = [
        (d, n_adv_after_matrix_perturbation(perturb_fn(matrix, i, d))) for d, _ in path_bool
    ]

    achieved = None
    perturbed_matrix = None
    if delta is not None:
        perturbed_matrix = perturb_fn(matrix, i, delta)
        achieved = n_adv_after_matrix_perturbation(perturbed_matrix)

    return MatrixOracleResult(
        candidate_index=i,
        mode=mode,
        alpha=delta,
        is_monotonic=is_monotonic,
        n_adv_path=n_adv_path,
        achieved_n_adv=achieved,
        perturbed_matrix=perturbed_matrix,
    )


def matrix_oracle_for_query(
    matrix: np.ndarray,
    non_and_indices: List[int],
    target_n_adv: int = 5,
    alpha_max: float = 2.0,
) -> List[MatrixOracleResult]:
    """Phase 4A over every currently-non-AND candidate (defense-native
    selection -- `non_and_indices` must be derived from Stage-1 flags only,
    never from `is_poison`). Tries `boost` first; if boost cannot reach the
    target for a candidate, also tries `decrease` (Phase 4B)."""
    results = []
    for i in non_and_indices:
        boost = matrix_oracle_for_candidate(matrix, i, target_n_adv, mode="boost", alpha_max=alpha_max)
        results.append(boost)
        if boost.alpha is None:
            decrease = matrix_oracle_for_candidate(
                matrix, i, target_n_adv, mode="decrease", alpha_max=alpha_max
            )
            results.append(decrease)
    return results


def best_matrix_oracle_result(results: List[MatrixOracleResult]) -> Optional[MatrixOracleResult]:
    """Defense-native selection: smallest alpha among all candidates/modes
    that reached the target -- never uses `is_poison`."""
    achieving = [r for r in results if r.alpha is not None]
    if not achieving:
        return None
    return min(achieving, key=lambda r: r.alpha)


# ---------------------------------------------------------------------------
# PHASE 5 -- causal check through the UNCHANGED Stage 2
# ---------------------------------------------------------------------------

STAGE2_LABEL_COUNT_FIX_SUCCESSFUL = "A. COUNT FIX + STAGE2 STILL SUCCESSFUL"
STAGE2_LABEL_COUNT_FIX_DEGRADED = "B. COUNT FIX BUT STAGE2 CHANGES"


def stage2_causal_check(perturbed_matrix: np.ndarray, is_poison: np.ndarray, n_adv: int = 5) -> dict:
    """Reruns the UNCHANGED `stage2_pair_frequency` on the PERTURBED matrix
    at the fixed `n_adv`. `is_poison` is used ONLY to score the resulting
    (already-selected) removed-index set, never to influence Stage 2's
    selection itself."""
    stage2 = ri.stage2_pair_frequency(perturbed_matrix, n_adv=n_adv, p=2.0)
    removed = set(stage2.selected_indices)
    removed_poison = sum(1 for idx in removed if is_poison[idx])
    removed_clean = sum(1 for idx in removed if not is_poison[idx])
    n_poison = int(np.asarray(is_poison).sum())
    residual_poison = n_poison - removed_poison
    label = (
        STAGE2_LABEL_COUNT_FIX_SUCCESSFUL
        if (residual_poison == 0 and removed_clean == 0)
        else STAGE2_LABEL_COUNT_FIX_DEGRADED
    )
    return {
        "removed_indices": sorted(removed),
        "removed_poison": removed_poison,
        "removed_clean": removed_clean,
        "residual_poison": residual_poison,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Perturbation effect-size metrics (shared by Phase 4/5 reporting)
# ---------------------------------------------------------------------------

def perturbation_effect_size(original: np.ndarray, perturbed: np.ndarray) -> dict:
    original = np.asarray(original, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    k = original.shape[0]
    off_diag_mask = ~np.eye(k, dtype=bool)
    diff = perturbed - original
    off_diag_diff = diff[off_diag_mask]
    n_off_diag = off_diag_mask.sum()
    n_changed = int(np.sum(np.abs(off_diag_diff) > 1e-12))
    return {
        "max_abs_off_diag_change": float(np.abs(off_diag_diff).max()) if off_diag_diff.size else 0.0,
        "mean_abs_off_diag_change": float(np.abs(off_diag_diff).mean()) if off_diag_diff.size else 0.0,
        "frobenius_norm_diff": float(np.linalg.norm(diff)),
        "fraction_off_diag_changed": float(n_changed / n_off_diag) if n_off_diag else 0.0,
    }
