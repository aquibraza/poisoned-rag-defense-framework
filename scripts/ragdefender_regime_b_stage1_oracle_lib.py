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


ALPHA_TOL = 1e-6  # local-refinement stopping resolution for the grid search below


@dataclass
class GridSearchResult:
    """Full result of scanning a Boolean predicate over `[lo, hi]` -- see
    `_monotonic_or_grid_search` docstring. Fixes the V1 bug where only the
    grid ENDPOINT was checked for reachability (a false negative whenever
    the predicate is transiently True somewhere in the interior but False
    at `hi`)."""

    reachable: bool
    earliest_success_alpha: Optional[float]  # refined estimate, or None if unreachable
    is_monotonic: bool  # True iff no True->False transition occurs anywhere on the coarse grid
    n_false_to_true_transitions: int
    n_true_to_false_transitions: int
    n_success_windows: int
    endpoint_successful: bool
    success_windows: List[Tuple[int, int]] = field(default_factory=list)  # coarse-grid index pairs, inclusive
    coarse_path: List[Tuple[float, bool]] = field(default_factory=list, repr=False)
    verified: bool = False  # predicate(earliest_success_alpha) re-checked True post-refinement
    just_below_alpha: Optional[float] = None
    just_below_successful: Optional[bool] = None


def _refine_earliest_success(
    predicate: Callable[[float], bool],
    bracket_lo: float,
    bracket_hi: float,
    refine_levels: int = 40,
    alpha_tol: float = ALPHA_TOL,
    dense_points: int = 25,
) -> float:
    """Deterministic local refinement of the EARLIEST false->true crossing
    inside `[bracket_lo, bracket_hi]` (where `predicate(bracket_hi)` is
    already known True). Does NOT assume monotonicity even locally --
    each refinement level re-scans a dense grid and only narrows to the
    immediately preceding false->true subinterval of THAT scan, exactly as
    specified (Phase 4, STEP 1A)."""
    lo, hi = float(bracket_lo), float(bracket_hi)
    best = hi
    for _level in range(refine_levels):
        if (hi - lo) <= alpha_tol:
            break
        dense_grid = np.linspace(lo, hi, dense_points)
        dense_bools = [bool(predicate(float(d))) for d in dense_grid]
        true_idx = next((i for i, v in enumerate(dense_bools) if v), None)
        if true_idx is None:
            # Should not happen since `hi` (the last dense point) is True by
            # construction of the bracket -- defensive fallback only.
            break
        best = float(dense_grid[true_idx])
        if true_idx == 0:
            hi = best
            break
        lo, hi = float(dense_grid[true_idx - 1]), best
    return best


def _monotonic_or_grid_search(
    predicate: Callable[[float], bool],
    lo: float = 0.0,
    hi: float = 4.0,
    coarse_steps: int = 400,
    refine_levels: int = 40,
    alpha_tol: float = ALPHA_TOL,
) -> GridSearchResult:
    """Full-path Boolean grid search over `[lo, hi]` -- CORRECTED version.

    V1 BUG (fixed here): the previous implementation checked ONLY
    `path[-1][1]` (the grid endpoint) to decide reachability, incorrectly
    returning "unreachable" whenever the predicate was transiently True in
    the interior of `[lo, hi]` but False again at `hi`. This function
    instead scans the ENTIRE coarse grid for ANY True value.

    Returns a `GridSearchResult`:
    - `reachable` is True iff the predicate is True at ANY sampled point.
    - `is_monotonic` (non-decreasing) iff there is zero True->False
      transition anywhere on the coarse grid -- i.e. once True, the
      predicate never reverts to False for the remainder of `[lo, hi]`.
    - `earliest_success_alpha` is the refined (not exactly-minimal, see
      `_refine_earliest_success`) alpha at the START of the FIRST
      contiguous successful window, regardless of what happens at later
      alpha (including if the predicate later reverts to False).
    """
    grid = np.linspace(lo, hi, coarse_steps + 1)
    path = [(float(d), bool(predicate(d))) for d in grid]
    bools = [v for _, v in path]

    success_indices = [i for i, v in enumerate(bools) if v]
    if not success_indices:
        return GridSearchResult(
            reachable=False,
            earliest_success_alpha=None,
            is_monotonic=True,  # vacuously monotonic: no True ever appears
            n_false_to_true_transitions=0,
            n_true_to_false_transitions=0,
            n_success_windows=0,
            endpoint_successful=False,
            success_windows=[],
            coarse_path=path,
            verified=False,
        )

    n_f2t = 0
    n_t2f = 0
    windows: List[List[int]] = []
    for i in range(len(bools)):
        prev = bools[i - 1] if i > 0 else False
        cur = bools[i]
        if (not prev) and cur:
            n_f2t += 1
            windows.append([i, i])
        elif prev and (not cur):
            n_t2f += 1
            windows[-1][1] = i - 1
        elif cur:
            # Still inside the currently-open window -- extend its end index
            # as we go, so a window that runs all the way to the end of the
            # grid is correctly closed at the LAST True index, not left at
            # its start index.
            windows[-1][1] = i

    is_monotonic = n_t2f == 0
    endpoint_successful = bools[-1]
    first_window = windows[0]

    bracket_lo = path[first_window[0] - 1][0] if first_window[0] > 0 else lo
    bracket_hi = path[first_window[0]][0]
    earliest_alpha = _refine_earliest_success(predicate, bracket_lo, bracket_hi, refine_levels, alpha_tol)

    verified = bool(predicate(earliest_alpha))
    just_below = None
    just_below_successful = None
    below_candidate = np.nextafter(earliest_alpha, -np.inf)
    if below_candidate >= lo:
        just_below = float(below_candidate)
        just_below_successful = bool(predicate(just_below))

    return GridSearchResult(
        reachable=True,
        earliest_success_alpha=earliest_alpha,
        is_monotonic=is_monotonic,
        n_false_to_true_transitions=n_f2t,
        n_true_to_false_transitions=n_t2f,
        n_success_windows=len(windows),
        endpoint_successful=endpoint_successful,
        success_windows=[(w[0], w[1]) for w in windows],
        coarse_path=path,
        verified=verified,
        just_below_alpha=just_below,
        just_below_successful=just_below_successful,
    )


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

    result = _monotonic_or_grid_search(predicate, lo=0.0, hi=hi)
    if not result.reachable:
        return None, result.is_monotonic
    return result.earliest_success_alpha + EPS, result.is_monotonic


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
# PHASE 4 -- symmetric bounded matrix-space oracle
#
# TERMINOLOGY (V2 correction): this phase explores perturbations that are
# symmetric, diagonal-preserving, and clipped to [-1,1] -- properties that
# are NECESSARY but NOT SUFFICIENT for a matrix to be a valid cosine Gram
# matrix of any embedding (it must also be positive semi-definite -- see
# `gram_matrix_validity` below). V1 of this report used the word
# "realizable" for this perturbation family; that wording is WITHDRAWN.
# Use "symmetric bounded matrix-space oracle" (LEVEL 1) or, for the PSD-
# valid subset, "abstract unit-vector-compatible matrix perturbation"
# (LEVEL 2). NEVER "embedding-realizable", "Stella-realizable", or
# "text-realizable" (LEVELS 3/4) -- those require a separate experiment
# this module does not perform.
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


def gram_matrix_validity(matrix: np.ndarray, sym_atol: float = 1e-8, diag_atol: float = 1e-6) -> dict:
    """PSD / Gram-matrix validity diagnostics (Phase 4, STEP 4).

    Symmetric + diagonal-preserving + [-1,1]-bounded (guaranteed by
    `perturb_boost`/`perturb_decrease`) does NOT by itself imply `matrix`
    is a valid cosine Gram matrix of any set of unit vectors -- it must
    ALSO be positive semi-definite. This function checks that directly via
    `eigvalsh`, and separately confirms the symmetry/diagonal properties
    (belt-and-suspenders, since the perturbation functions already
    guarantee them by construction).
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    is_symmetric = bool(np.allclose(matrix, matrix.T, atol=sym_atol))
    diag_near_one = bool(np.allclose(np.diag(matrix), 1.0, atol=diag_atol))
    # Symmetrize defensively before eigvalsh (which assumes symmetry and
    # only reads the lower triangle by default) -- does not change the
    # result for an already-symmetric input, but protects against any
    # sub-tolerance asymmetry from floating-point roundoff.
    symmetrized = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetrized)
    min_eigenvalue = float(eigenvalues.min())
    n_negative_eigenvalues = int(np.sum(eigenvalues < -1e-12))
    return {
        "is_symmetric": is_symmetric,
        "diag_near_one": diag_near_one,
        "min_eigenvalue": min_eigenvalue,
        "n_negative_eigenvalues": n_negative_eigenvalues,
        "psd_valid_tol_1e8": min_eigenvalue >= -1e-8,
        "psd_valid_tol_1e6": min_eigenvalue >= -1e-6,
    }


@dataclass
class MatrixOracleResult:
    candidate_index: int
    mode: str  # "boost" or "decrease"
    alpha: Optional[float]  # earliest-detected successful alpha (refined), or None if unreachable
    is_monotonic: bool
    reachable: bool
    endpoint_successful: bool
    n_false_to_true_transitions: int
    n_true_to_false_transitions: int
    n_success_windows: int
    n_adv_path: List[Tuple[float, int]]
    achieved_n_adv: Optional[int]
    verified: bool
    gram: Optional[dict] = None  # PSD/Gram validity of the winning perturbed matrix, if alpha is not None
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
    `S_ij` for all `j != i`; `mode='decrease'` lowers it. Uses the
    CORRECTED `_monotonic_or_grid_search` (full-path scan -- see its
    docstring for the V1 bug this fixes), so a transient success whose
    grid endpoint is False is still detected and returned, not missed."""
    perturb_fn = perturb_boost if mode == "boost" else perturb_decrease

    def predicate(alpha: float) -> bool:
        return n_adv_after_matrix_perturbation(perturb_fn(matrix, i, alpha)) >= target_n_adv

    search = _monotonic_or_grid_search(predicate, lo=0.0, hi=alpha_max, coarse_steps=coarse_steps)
    # Re-derive the full N_adv(alpha) integer path (not just the boolean
    # predicate) at the same grid resolution for non-monotonicity reporting.
    n_adv_path = [
        (d, n_adv_after_matrix_perturbation(perturb_fn(matrix, i, d))) for d, _ in search.coarse_path
    ]

    achieved = None
    perturbed_matrix = None
    gram = None
    if search.reachable:
        perturbed_matrix = perturb_fn(matrix, i, search.earliest_success_alpha)
        achieved = n_adv_after_matrix_perturbation(perturbed_matrix)
        gram = gram_matrix_validity(perturbed_matrix)

    return MatrixOracleResult(
        candidate_index=i,
        mode=mode,
        alpha=search.earliest_success_alpha,
        is_monotonic=search.is_monotonic,
        reachable=search.reachable,
        endpoint_successful=search.endpoint_successful,
        n_false_to_true_transitions=search.n_false_to_true_transitions,
        n_true_to_false_transitions=search.n_true_to_false_transitions,
        n_success_windows=search.n_success_windows,
        n_adv_path=n_adv_path,
        achieved_n_adv=achieved,
        verified=search.verified,
        gram=gram,
        perturbed_matrix=perturbed_matrix,
    )


def matrix_oracle_for_query(
    matrix: np.ndarray,
    non_and_indices: List[int],
    target_n_adv: int = 5,
    alpha_max: float = 2.0,
) -> List[MatrixOracleResult]:
    """Phase 4A+4B over every currently-non-AND candidate (defense-native
    selection -- `non_and_indices` must be derived from Stage-1 flags only,
    never from `is_poison`). V2 CORRECTION: always runs BOTH `boost` AND
    `decrease` for every candidate (V1's "only try decrease if boost fails"
    shortcut is removed for this correction pass, so a second successful
    mode can never be silently hidden by control flow)."""
    results = []
    for i in non_and_indices:
        results.append(matrix_oracle_for_candidate(matrix, i, target_n_adv, mode="boost", alpha_max=alpha_max))
        results.append(matrix_oracle_for_candidate(matrix, i, target_n_adv, mode="decrease", alpha_max=alpha_max))
    return results


def select_matrix_winner(
    results: List[MatrixOracleResult],
    original_matrix: np.ndarray,
    require_psd: bool = False,
    psd_tol: str = "1e8",
) -> Optional[MatrixOracleResult]:
    """Phase-4 STEP 7 winner selection with the documented tie-break:
    (1) smallest reachable/refined alpha; (2) smaller Frobenius norm of
    `S'-S`; (3) lower candidate index. If `require_psd`, restricts to the
    PSD-valid subset FIRST (`psd_tol` in {"1e8", "1e6"}), then applies the
    identical tie-break -- this is how the separate `best_psd_valid_winner`
    is derived (STEP 4A/7). Never uses `is_poison`."""
    achieving = [r for r in results if r.alpha is not None]
    if require_psd:
        key = "psd_valid_tol_1e8" if psd_tol == "1e8" else "psd_valid_tol_1e6"
        achieving = [r for r in achieving if r.gram is not None and r.gram[key]]
    if not achieving:
        return None

    def _frobenius(r: MatrixOracleResult) -> float:
        return float(np.linalg.norm(r.perturbed_matrix - original_matrix))

    min_alpha = min(r.alpha for r in achieving)
    tol = 1e-12
    tied_alpha = [r for r in achieving if abs(r.alpha - min_alpha) <= tol]
    if len(tied_alpha) == 1:
        return tied_alpha[0]
    min_frob = min(_frobenius(r) for r in tied_alpha)
    tied_frob = [r for r in tied_alpha if abs(_frobenius(r) - min_frob) <= 1e-12]
    if len(tied_frob) == 1:
        return tied_frob[0]
    return min(tied_frob, key=lambda r: r.candidate_index)


# ---------------------------------------------------------------------------
# PHASE 5 -- causal check through the UNCHANGED Stage 2
# ---------------------------------------------------------------------------

STAGE2_LABEL_COUNT_FIX_SUCCESSFUL = "A. COUNT FIX + STAGE2 STILL SUCCESSFUL"
STAGE2_LABEL_COUNT_FIX_DEGRADED = "B. COUNT FIX BUT STAGE2 CHANGES"


def stage2_causal_check(perturbed_matrix: np.ndarray, is_poison: np.ndarray, n_adv: int = 5) -> dict:
    """Reruns the UNCHANGED `stage2_pair_frequency` on the PERTURBED matrix
    at the fixed `n_adv`. `is_poison` is used ONLY to score the resulting
    (already-selected) removed-index set, never to influence Stage 2's
    selection itself. Also reports the PP/PC/CC pair composition of the
    top-pair set and the full frequency-score vector (Phase 5, STEP 8)."""
    is_poison = np.asarray(is_poison, dtype=bool)
    k = len(is_poison)
    stage2 = ri.stage2_pair_frequency(perturbed_matrix, n_adv=n_adv, p=2.0)
    removed = set(stage2.selected_indices)
    removed_poison = sum(1 for idx in removed if is_poison[idx])
    removed_clean = sum(1 for idx in removed if not is_poison[idx])
    n_poison = int(is_poison.sum())
    n_clean = k - n_poison
    residual_poison = n_poison - removed_poison
    residual_clean = n_clean - removed_clean
    label = (
        STAGE2_LABEL_COUNT_FIX_SUCCESSFUL
        if (residual_poison == 0 and removed_clean == 0)
        else STAGE2_LABEL_COUNT_FIX_DEGRADED
    )
    pp = pc = cc = 0
    for x, y, _sim in stage2.top_pairs:
        if is_poison[x] and is_poison[y]:
            pp += 1
        elif (not is_poison[x]) and (not is_poison[y]):
            cc += 1
        else:
            pc += 1
    return {
        "removed_indices": sorted(removed),
        "removed_poison": removed_poison,
        "removed_clean": removed_clean,
        "residual_poison": residual_poison,
        "residual_clean": residual_clean,
        "n_pairs": stage2.n_pairs,
        "pp_count": pp,
        "pc_count": pc,
        "cc_count": cc,
        "frequency_scores": stage2.frequency_scores.tolist(),
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


# ---------------------------------------------------------------------------
# MUTUAL-MEDIAN MECHANISM VALIDATION (V2 correction, Phase 5/STEP 5)
#
# The V1 report's headline mechanistic claim -- "the exact rank5/rank6
# s_median tie in 11/14 MEDIAN-LIMITED failures is caused by a
# 'mutual-median-match' passage pair" -- was demonstrated on ONE worked
# example, not verified query-by-query. This section makes that check
# explicit and exhaustive, using PROVIDER SETS (not single indices) to
# correctly handle rows with more than one neighbor tied at the median
# value.
# ---------------------------------------------------------------------------

def median_provider_indices(matrix: np.ndarray, i: int) -> set:
    """Every neighbor `j != i` whose similarity `S_ij` equals passage `i`'s
    own (self-excluded, torch-style) per-passage median `s_median_i` --
    i.e. the SET of off-diagonal neighbors that supply the median-ranked
    value in row `i`. A set, not a single index, because more than one
    neighbor can carry exactly the median value when there are ties within
    a single row's 9 off-diagonal similarities."""
    matrix = np.asarray(matrix, dtype=np.float64)
    k = matrix.shape[0]
    row = matrix[i, :]
    others = [j for j in range(k) if j != i]
    off_diag_vals = np.array([row[j] for j in others], dtype=np.float64)
    median_val = ri._torch_style_median_1d(off_diag_vals)  # noqa: SLF001
    return {j for j in others if row[j] == median_val}


def identify_tied_boundary_passages(s_median: np.ndarray) -> dict:
    """Identifies the passage index/indices occupying the tied rank-5/
    rank-6 boundary of `s_median` (torch-style lower-middle convention,
    `k=10` -> 0-indexed positions 4 and 5). Handles ties of width >2 by
    returning every index whose `s_median` value equals the tied boundary
    value, not just the two canonical rank-5/rank-6 positions."""
    s_median = np.asarray(s_median, dtype=np.float64)
    k = s_median.shape[0]
    sorted_order = np.argsort(s_median, kind="stable")
    rank5_pos = (k - 1) // 2
    rank6_pos = rank5_pos + 1
    rank5_idx = int(sorted_order[rank5_pos])
    rank6_idx = int(sorted_order[rank6_pos]) if rank6_pos < k else None
    is_tied = rank6_idx is not None and s_median[rank5_idx] == s_median[rank6_idx]
    tied_value = float(s_median[rank5_idx])
    tied_indices = sorted(i for i in range(k) if s_median[i] == tied_value) if is_tied else []
    return {
        "rank5_idx": rank5_idx,
        "rank6_idx": rank6_idx,
        "is_tied": is_tied,
        "tied_value": tied_value if is_tied else None,
        "tied_indices": tied_indices,
    }


def mutual_median_validation_for_query(matrix: np.ndarray, s_median: np.ndarray) -> dict:
    """Full STEP 5 validation for one query: identifies the tied
    rank5/rank6 boundary passages, computes each tied passage's median-
    provider SET, and determines whether at least one MUTUAL match exists
    (`j in providers(i)` AND `i in providers(j)`) among the tied passages.

    `mutual_median_match` is True iff such a pair exists. This is checked
    directly from the matrix/statistics -- no assumption, no conditional
    no-op assertion (see STEP 5A)."""
    boundary = identify_tied_boundary_passages(s_median)
    if not boundary["is_tied"]:
        return {
            **boundary,
            "providers": {},
            "mutual_median_match": False,
            "mutual_pairs": [],
            "shared_matrix_entry": None,
        }

    tied_indices = boundary["tied_indices"]
    providers = {i: median_provider_indices(matrix, i) for i in tied_indices}

    mutual_pairs = set()
    for i in tied_indices:
        for j in providers[i]:
            if j in tied_indices and i in providers.get(j, set()):
                mutual_pairs.add(tuple(sorted((i, j))))

    shared_matrix_entry = None
    if mutual_pairs:
        i0, j0 = sorted(mutual_pairs)[0]
        shared_matrix_entry = float(matrix[i0, j0])

    return {
        **boundary,
        "providers": {i: sorted(v) for i, v in providers.items()},
        "mutual_median_match": len(mutual_pairs) > 0,
        "mutual_pairs": sorted(mutual_pairs),
        "shared_matrix_entry": shared_matrix_entry,
    }
