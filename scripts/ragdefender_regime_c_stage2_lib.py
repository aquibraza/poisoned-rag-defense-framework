"""Pure-function library for the Regime-C Stage-2 Identification-Capacity
study.

Scope: OFFLINE Stage-2 diagnostic only. Holds the supplied Stage-2 count
FIXED at the true poison count `M` throughout -- Stage 1 is never
consulted here (no `adv_flag`, no `concentration_stage1_paper` call
anywhere in this module). Every function is a pure, deterministic
transformation of a similarity matrix (+ a `p=2.0` exponent) and, for
diagnostics ONLY, a ground-truth `is_poison` label array. `is_poison` is
NEVER used to influence the PRODUCTION Stage-2 recomputation
(`stage2_original_top_pairs` + `compute_frequency_and_selection` with the
real matrix) -- only to CLASSIFY/SCORE pairs and passages after the fact,
and, explicitly and only in Phases 5-7, to construct GROUND-TRUTH UPPER-
BOUND ORACLE pair sets (documented as such, never claimed as a deployable
mechanism).

All frequency-score math mirrors `defense/ragdefender_internals.py
::stage2_pair_frequency` EXACTLY (same sort key, same `math.copysign(abs
(sim)**p, sim)` contribution rule, same plain-dict insertion-order tie-
break under a stable `sorted()`, verified byte-for-byte against it in
`tests/test_ragdefender_regime_c_stage2.py`).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-9
Pair = Tuple[int, int, float]  # (i, j, sim), i < j


# ---------------------------------------------------------------------------
# Basic combinatorics / classification
# ---------------------------------------------------------------------------

def n_choose_2(n: int) -> int:
    if n < 2:
        return 0
    return n * (n - 1) // 2


def classify_pair(i: int, j: int, is_poison: np.ndarray) -> str:
    pi, pj = bool(is_poison[i]), bool(is_poison[j])
    if pi and pj:
        return "PP"
    if (not pi) and (not pj):
        return "CC"
    return "PC"


def pair_signed_score(sim: float, p: float = 2.0) -> float:
    return math.copysign(abs(sim) ** p, sim)


# ---------------------------------------------------------------------------
# Faithful mirror of ri.stage2_pair_frequency's two internal stages,
# exposed separately so oracle phases (5/6/7) can substitute a custom
# pair list into the SECOND stage while phases 1/2 use the real FIRST
# stage on the frozen similarity matrix.
# ---------------------------------------------------------------------------

def all_pairs_sorted(matrix: np.ndarray) -> List[Pair]:
    """Every unordered pair (i<j), sorted desc by RAW (signed) similarity
    -- identical sort key to `ri.stage2_pair_frequency`, but NOT truncated
    to `n_pairs` (needed for Phase 1B's beyond-the-boundary analysis)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    k = matrix.shape[0]
    pairs = [(i, j, float(matrix[i, j])) for i in range(k) for j in range(i + 1, k)]
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs


def stage2_original_top_pairs(matrix: np.ndarray, n_adv: int) -> Tuple[List[Pair], List[Pair], int]:
    """Stage 2's FIRST stage on the real, frozen similarity matrix: sort
    ALL pairs by raw similarity descending, keep the top `n_pairs =
    max(1, C(n_adv,2))`. Returns (top_pairs, all_pairs_sorted, n_pairs)."""
    if n_adv <= 0:
        return [], all_pairs_sorted(matrix), 0
    all_pairs = all_pairs_sorted(matrix)
    n_pairs = max(1, n_choose_2(n_adv))
    return all_pairs[:n_pairs], all_pairs, n_pairs


@dataclass
class SelectionResult:
    """Output of Stage 2's SECOND stage (frequency scoring + top-`n_adv`
    selection) applied to an ARBITRARY pair list -- identical algorithm to
    `ri.stage2_pair_frequency`'s tail half, byte-for-byte, so it can be
    reused for the real P_top (Phase 1/2) as well as every oracle pair-set
    variant (Phase 5/6/7)."""

    freq: Dict[int, float]  # only indices with >=1 incident pair in the given list
    ranked: List[Tuple[int, float]]  # sorted(freq.items(), key=score, reverse=True) -- FULL, not truncated
    selected_indices: List[int]  # ranked[:n_adv] indices -- may be SHORTER than n_adv if len(ranked)<n_adv
    frequency_scores: np.ndarray  # length k, 0.0 for every index absent from freq


def compute_frequency_and_selection(pair_list: Sequence[Pair], k: int, n_adv: int, p: float = 2.0) -> SelectionResult:
    """Byte-for-byte mirror of `ri.stage2_pair_frequency`'s scoring tail:
    given an explicit pair list (already the intended P_top -- no further
    sorting/truncation is performed here), accumulate
    `math.copysign(abs(sim)**p, sim)` per incident index in a plain dict
    (first-insertion order preserved, exactly like `collections.Counter`),
    then a stable `sorted(..., reverse=True)` for the final ranking. This
    is the ENTIRE tie-break contract production relies on; replicated
    verbatim rather than approximated."""
    freq: Dict[int, float] = {}
    for x, y, sim in pair_list:
        contribution = pair_signed_score(sim, p)
        for idx in (x, y):
            freq[idx] = freq.get(idx, 0.0) + contribution
    ranked = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    selected_indices = [idx for idx, _ in ranked[:n_adv]] if n_adv > 0 else []
    frequency_scores = np.zeros(k, dtype=np.float64)
    for idx, score in freq.items():
        frequency_scores[idx] = score
    return SelectionResult(
        freq=freq, ranked=ranked, selected_indices=selected_indices, frequency_scores=frequency_scores
    )


def extended_ranking_with_zero_degree(selection: SelectionResult, k: int) -> List[Tuple[int, float]]:
    """Extends `selection.ranked` (which only contains indices with >=1
    incident pair) to a FULL length-`k` ranking by appending every
    zero-degree index (score exactly 0.0, ascending-index tie-break) after
    all real entries. Production Stage 2 never defines a "rank" for a
    zero-degree passage (it is simply absent from `freq`); this extension
    exists ONLY for Phase 2A's M/M+1 boundary diagnostic and is documented
    as a diagnostic convention, not a production behavior."""
    present = {idx for idx, _ in selection.ranked}
    zero_degree = [idx for idx in range(k) if idx not in present]
    return list(selection.ranked) + [(idx, 0.0) for idx in zero_degree]


def removal_outcome(selection: SelectionResult, is_poison: np.ndarray, m_poison: int) -> dict:
    removed = set(selection.selected_indices)
    removed_poison = sum(1 for idx in removed if is_poison[idx])
    removed_clean = sum(1 for idx in removed if not is_poison[idx])
    residual_poison = m_poison - removed_poison
    return {
        "removed_indices": sorted(removed),
        "removed_poison": removed_poison,
        "removed_clean": removed_clean,
        "residual_poison": residual_poison,
        "success": bool(removed_poison == m_poison and removed_clean == 0),
    }


# ---------------------------------------------------------------------------
# Phase 1A/1B -- P_top composition and boundary margins
# ---------------------------------------------------------------------------

def pair_class_counts(pairs: Sequence[Pair], is_poison: np.ndarray) -> dict:
    n_pp = n_pc = n_cc = 0
    for i, j, _sim in pairs:
        cls = classify_pair(i, j, is_poison)
        if cls == "PP":
            n_pp += 1
        elif cls == "PC":
            n_pc += 1
        else:
            n_cc += 1
    return {"n_PP": n_pp, "n_PC": n_pc, "n_CC": n_cc}


def ptop_composition(
    top_pairs: Sequence[Pair], all_pairs: Sequence[Pair], n_pairs: int, is_poison: np.ndarray, m_poison: int, c_clean: int
) -> dict:
    counts = pair_class_counts(top_pairs, is_poison)
    total_pp = n_choose_2(m_poison)
    total_pc = m_poison * c_clean
    total_cc = n_choose_2(c_clean)

    pp_sims = [sim for i, j, sim in top_pairs if classify_pair(i, j, is_poison) == "PP"]
    pc_sims = [sim for i, j, sim in top_pairs if classify_pair(i, j, is_poison) == "PC"]
    cc_sims = [sim for i, j, sim in top_pairs if classify_pair(i, j, is_poison) == "CC"]

    first_pc_rank = None
    first_cc_rank = None
    for rank, (i, j, _sim) in enumerate(all_pairs, start=1):
        cls = classify_pair(i, j, is_poison)
        if cls == "PC" and first_pc_rank is None:
            first_pc_rank = rank
        if cls == "CC" and first_cc_rank is None:
            first_cc_rank = rank
        if first_pc_rank is not None and first_cc_rank is not None:
            break

    selected_boundary_sim = all_pairs[n_pairs - 1][2] if 0 < n_pairs <= len(all_pairs) else None
    next_pair_sim = all_pairs[n_pairs][2] if n_pairs < len(all_pairs) else None
    pair_cutoff_margin = (
        selected_boundary_sim - next_pair_sim if (selected_boundary_sim is not None and next_pair_sim is not None) else None
    )

    return {
        "n_pairs": n_pairs,
        "total_possible_PP": total_pp,
        "total_possible_PC": total_pc,
        "total_possible_CC": total_cc,
        "n_PP_selected": counts["n_PP"],
        "n_PC_selected": counts["n_PC"],
        "n_CC_selected": counts["n_CC"],
        "PP_share": counts["n_PP"] / n_pairs if n_pairs else None,
        "PC_share": counts["n_PC"] / n_pairs if n_pairs else None,
        "CC_share": counts["n_CC"] / n_pairs if n_pairs else None,
        "n_PP_missing": total_pp - counts["n_PP"],
        "pair_set_pure_pp": bool(counts["n_PP"] == total_pp),
        "highest_PC_sim": max(pc_sims) if pc_sims else None,
        "highest_CC_sim": max(cc_sims) if cc_sims else None,
        "lowest_PP_sim": min(pp_sims) if pp_sims else None,
        "first_PC_rank": first_pc_rank,
        "first_CC_rank": first_cc_rank,
        "selected_boundary_sim": selected_boundary_sim,
        "next_pair_sim": next_pair_sim,
        "pair_cutoff_margin": pair_cutoff_margin,
        "max_PC_minus_min_PP": (max(pc_sims) - min(pp_sims)) if (pc_sims and pp_sims) else None,
        "max_CC_minus_min_PP": (max(cc_sims) - min(pp_sims)) if (cc_sims and pp_sims) else None,
    }


# ---------------------------------------------------------------------------
# Phase 2 -- passage frequency-score / incident detail
# ---------------------------------------------------------------------------

def passage_incident_detail(pairs: Sequence[Pair], is_poison: np.ndarray, k: int, p: float = 2.0) -> Dict[int, dict]:
    detail = {
        i: {
            "n_PP_incident": 0,
            "n_PC_incident": 0,
            "n_CC_incident": 0,
            "signed_PP_contribution": 0.0,
            "signed_PC_contribution": 0.0,
            "signed_CC_contribution": 0.0,
            "degree": 0,
        }
        for i in range(k)
    }
    for x, y, sim in pairs:
        cls = classify_pair(x, y, is_poison)
        contribution = pair_signed_score(sim, p)
        for idx in (x, y):
            detail[idx][f"n_{cls}_incident"] += 1
            detail[idx][f"signed_{cls}_contribution"] += contribution
            detail[idx]["degree"] += 1
    return detail


def boundary_classification(rank_m_label: str, rank_m1_label: str) -> str:
    if rank_m_label == "poison" and rank_m1_label == "clean":
        return "A. POISON/CLEAN"
    if rank_m_label == "clean" and rank_m1_label == "poison":
        return "B. CLEAN/POISON"
    return "C. SAME-LABEL BOUNDARY"


# ---------------------------------------------------------------------------
# Phase 5 -- pure-PP pair-set oracle
# ---------------------------------------------------------------------------

def pure_pp_pair_set(matrix: np.ndarray, is_poison: np.ndarray) -> List[Pair]:
    """ALL true PP pairs (i<j, both poison) with their REAL similarity
    values from the frozen matrix -- exactly `C(M,2)` pairs, which equals
    the production Stage-2 pair budget for `n_adv=M`. GROUND-TRUTH UPPER-
    BOUND DIAGNOSTIC ORACLE: uses `is_poison` to construct the pair set,
    never to alter production selection."""
    matrix = np.asarray(matrix, dtype=np.float64)
    k = matrix.shape[0]
    return [(i, j, float(matrix[i, j])) for i in range(k) for j in range(i + 1, k) if is_poison[i] and is_poison[j]]


# ---------------------------------------------------------------------------
# Phase 6 -- pair-class ablation
# ---------------------------------------------------------------------------

def ablation_variants(top_pairs: Sequence[Pair], is_poison: np.ndarray) -> Dict[str, List[Pair]]:
    def cls(pair):
        return classify_pair(pair[0], pair[1], is_poison)

    return {
        "A_original": list(top_pairs),
        "B_remove_CC": [pr for pr in top_pairs if cls(pr) != "CC"],
        "C_remove_PC": [pr for pr in top_pairs if cls(pr) != "PC"],
        "D_pp_only": [pr for pr in top_pairs if cls(pr) == "PP"],
    }


def classify_ablation_driver(success_b: bool, success_c: bool, success_d: bool) -> str:
    """Deterministic classification rule (documented, STEP per Phase 6):

    - CC-driven: removing CC alone (variant B) succeeds, removing PC alone
      (variant C) does not.
    - PC-driven: removing PC alone (variant C) succeeds, removing CC alone
      (variant B) does not.
    - mixed PC+CC: EITHER (both B and C independently succeed -- each
      alone is sufficient, contributions overlap/are redundant) OR (only
      D -- removing BOTH -- succeeds, i.e. neither alone suffices but
      together they do).
    - PP-weighting/other: even D (the PP-only SUBSET of the original
      P_top, NOT the full true-PP set of Phase 5) still fails -- the
      failure is not explained by non-PP intrusion alone; the specific PP
      edges that survived in P_top are themselves insufficient (e.g. some
      poison passage had zero PP-only degree in the original P_top).
    """
    if success_b and success_c:
        return "mixed PC+CC (either alone sufficient)"
    if success_b and not success_c:
        return "CC-driven"
    if success_c and not success_b:
        return "PC-driven"
    if success_d:
        return "mixed PC+CC (only both together sufficient)"
    return "PP-weighting/other"


# ---------------------------------------------------------------------------
# Phase 7 -- minimal pair-swap oracle
# ---------------------------------------------------------------------------

MAX_EXACT_SWAP_COMBINATIONS = 200_000


@dataclass
class SwapSearchResult:
    swap_count: Optional[int]
    is_exact: bool
    removed_pairs: List[Pair] = field(default_factory=list)
    added_pairs: List[Pair] = field(default_factory=list)
    resulting_pair_list: List[Pair] = field(default_factory=list)
    outcome: Optional[dict] = None
    searched_up_to: int = 0


def minimal_pair_swap_search(
    top_pairs: Sequence[Pair],
    matrix: np.ndarray,
    is_poison: np.ndarray,
    k: int,
    m_poison: int,
    max_combinations: int = MAX_EXACT_SWAP_COMBINATIONS,
) -> SwapSearchResult:
    """SUPERSEDED BY V2 -- see `certified_minimum_pair_swap_search` below.

    V1 CERTIFICATION BUG (fixed in V2, kept here only for historical
    comparison/tests): when `C(q,r)**2` exceeds `max_combinations` for
    some swap count `r`, this function evaluates only ONE deterministic
    GREEDY candidate at that level instead of exhausting it. If that
    greedy candidate fails but a LARGER `r'` later falls back under the
    threshold and is found exhaustively successful, this function still
    returns `is_exact=True` for `r'` -- even though smaller levels were
    never exhaustively ruled out, so `is_exact=True` does NOT certify
    `swap_count` as the true global minimum. Confirmed to have actually
    understated correctness on one real query in this population (this
    function reported r=10 "exact"; the true certified minimum, found by
    `certified_minimum_pair_swap_search`, is r=8). Do not use this
    function's output as a minimality claim.

    Minimum number of (remove one non-PP selected pair, add one
    currently-unselected true-PP pair) swaps needed so that recomputing
    the SAME Eq.(6)/(7) frequency-score selection over the swapped pair
    list yields `removed_poison == M, removed_clean == 0`.

    Deterministic search order: candidate removals are the currently-
    selected non-PP pairs sorted ASCENDING by similarity (weakest/most
    suspicious first); candidate additions are the currently-unselected
    true PP pairs sorted DESCENDING by similarity (strongest available
    true edge first). `itertools.combinations` over these pre-sorted lists
    enumerates in a fixed, reproducible order; the FIRST successful
    combination found (removal combo outer loop, addition combo inner
    loop, both in that fixed order) is returned -- documented, not
    arbitrary. Guaranteed feasible at `swap_count == len(non_pp_pairs)`
    (full swap to the Phase-5 pure-PP set), so the search never needs to
    exceed that bound.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    non_pp_pairs = sorted(
        [pr for pr in top_pairs if classify_pair(pr[0], pr[1], is_poison) != "PP"], key=lambda t: t[2]
    )
    selected_set = {(pr[0], pr[1]) for pr in top_pairs}
    unselected_pp_pairs = sorted(
        [
            (i, j, float(matrix[i, j]))
            for i in range(k)
            for j in range(i + 1, k)
            if is_poison[i] and is_poison[j] and (i, j) not in selected_set
        ],
        key=lambda t: t[2],
        reverse=True,
    )

    max_swap = len(non_pp_pairs)  # full swap to pure-PP is always feasible (proof: see report)
    base_pairs = list(top_pairs)

    for swap_count in range(1, max_swap + 1):
        removal_combos = list(itertools.combinations(non_pp_pairs, swap_count))
        addition_combos_len = len(list(itertools.combinations(unselected_pp_pairs, swap_count))) if unselected_pp_pairs else 0
        total_combos = len(removal_combos) * max(addition_combos_len, 1)
        is_exact = total_combos <= max_combinations

        if is_exact:
            for removal_combo in removal_combos:
                remove_set = {(pr[0], pr[1]) for pr in removal_combo}
                remaining = [pr for pr in base_pairs if (pr[0], pr[1]) not in remove_set]
                for addition_combo in itertools.combinations(unselected_pp_pairs, swap_count):
                    candidate = remaining + list(addition_combo)
                    selection = compute_frequency_and_selection(candidate, k, m_poison)
                    outcome = removal_outcome(selection, is_poison, m_poison)
                    if outcome["success"]:
                        return SwapSearchResult(
                            swap_count=swap_count,
                            is_exact=True,
                            removed_pairs=list(removal_combo),
                            added_pairs=list(addition_combo),
                            resulting_pair_list=candidate,
                            outcome=outcome,
                            searched_up_to=swap_count,
                        )
        else:
            # Deterministic greedy fallback for this swap_count: remove the
            # `swap_count` weakest non-PP pairs, add the `swap_count`
            # strongest unselected PP pairs (both lists are already sorted
            # that way) -- a single candidate, not exhaustive.
            removal_combo = tuple(non_pp_pairs[:swap_count])
            addition_combo = tuple(unselected_pp_pairs[:swap_count])
            remove_set = {(pr[0], pr[1]) for pr in removal_combo}
            remaining = [pr for pr in base_pairs if (pr[0], pr[1]) not in remove_set]
            candidate = remaining + list(addition_combo)
            selection = compute_frequency_and_selection(candidate, k, m_poison)
            outcome = removal_outcome(selection, is_poison, m_poison)
            if outcome["success"]:
                return SwapSearchResult(
                    swap_count=swap_count,
                    is_exact=False,
                    removed_pairs=list(removal_combo),
                    added_pairs=list(addition_combo),
                    resulting_pair_list=candidate,
                    outcome=outcome,
                    searched_up_to=swap_count,
                )

    # Should not happen given the full-swap feasibility proof, but return a
    # well-formed "not found" result rather than raising.
    return SwapSearchResult(swap_count=None, is_exact=True, searched_up_to=max_swap)


# ---------------------------------------------------------------------------
# Phase 8 -- optional score-space epsilon oracle
# ---------------------------------------------------------------------------

def minimal_score_epsilon(frequency_scores: np.ndarray, is_poison: np.ndarray, m_poison: int, eps: float = EPS) -> Optional[float]:
    """Minimum additive correction needed so every poison score would
    exceed every clean score, evaluated directly in frequency-score space
    (NOT matrix-realizable; diagnostic only). Returns the smallest
    `epsilon_needed = f_clean - f_poison + eps` over every clean/poison
    inversion at the current ranking, or `None` if no inversion exists."""
    poison_scores = [frequency_scores[i] for i in range(len(frequency_scores)) if is_poison[i]]
    clean_scores = [frequency_scores[i] for i in range(len(frequency_scores)) if not is_poison[i]]
    if not poison_scores or not clean_scores:
        return None
    max_clean = max(clean_scores)
    min_poison = min(poison_scores)
    if max_clean < min_poison:
        return None
    return float(max_clean - min_poison + eps)


# ===========================================================================
# REGIME-C STAGE-2 V2 VALIDATION PASS
#
# Corrects three V1 issues:
#   (1) the pair-swap "exact minimum" certification bug (a greedy-only
#       smaller level could be silently skipped while a larger,
#       exhaustively-searched level was mislabeled `is_exact=True`);
#   (2) the vague "PP-weighting/other" ablation category, replaced by the
#       exact graph-theoretic PP-COVERAGE-LIMITED mechanism (Steps 5/5A/5B);
#   (3) overclaiming pure-PP-oracle causality and `score_overlap` causality
#       (corrected in the V2 report text, not in this library -- the
#       library only computes numbers, never causal prose).
# ===========================================================================

# ---------------------------------------------------------------------------
# Step 2 -- pair-count identity (q_selected_non_pp == q_missing_pp)
# ---------------------------------------------------------------------------

def pair_count_identity(top_pairs: Sequence[Pair], matrix: np.ndarray, is_poison: np.ndarray, k: int) -> dict:
    """`q_selected_non_pp` (selected PC+CC pairs currently in P_top) must
    equal `q_missing_pp` (true PP pairs NOT currently in P_top), because
    `N_pairs = C(M,2)` exactly equals the total number of true PP pairs:
    every selected non-PP pair displaces exactly one true PP pair out of
    the budget. Verified explicitly here rather than assumed."""
    matrix = np.asarray(matrix, dtype=np.float64)
    non_pp_pairs = [pr for pr in top_pairs if classify_pair(pr[0], pr[1], is_poison) != "PP"]
    selected_set = {(pr[0], pr[1]) for pr in top_pairs}
    missing_pp_pairs = [
        (i, j, float(matrix[i, j]))
        for i in range(k)
        for j in range(i + 1, k)
        if is_poison[i] and is_poison[j] and (i, j) not in selected_set
    ]
    q_selected_non_pp = len(non_pp_pairs)
    q_missing_pp = len(missing_pp_pairs)
    return {
        "q_selected_non_pp": q_selected_non_pp,
        "q_missing_pp": q_missing_pp,
        "identity_holds": bool(q_selected_non_pp == q_missing_pp),
        "non_pp_pairs": non_pp_pairs,
        "missing_pp_pairs": missing_pp_pairs,
    }


# ---------------------------------------------------------------------------
# Step 5/5A -- PP vertex-coverage analysis (replaces "PP-weighting/other")
# ---------------------------------------------------------------------------

def pp_coverage_analysis(top_pairs: Sequence[Pair], is_poison: np.ndarray, m_poison: int) -> dict:
    """Graph-theoretic diagnostic (OUR interpretation of the pair-frequency
    procedure, not a claim about how RAGDefender "reasons"): build
    `G_PP(P_top)` -- vertices = the M poison passages, edges = the
    currently-SELECTED PP pairs inside `P_top` (i.e. `top_pairs` restricted
    to the PP class; a strict subgraph of the complete `K_M`, generally
    NOT all `C(M,2)` true PP pairs). Computes each poison vertex's degree
    in this subgraph and whether every poison vertex has degree >= 1
    ("complete coverage").

    THEOREM (proved and verified computationally, see
    `tests/test_ragdefender_regime_c_stage2_v2.py`): because this pair set
    contains ONLY PP edges, no clean vertex can ever appear in `freq`
    (degree 0, absent). Therefore:
      - if coverage is complete (every poison vertex has PP degree>=1),
        `len(freq) == M` exactly and `ranked[:M]` MUST be exactly the M
        poison vertices, regardless of edge weights/signs (identical
        argument to the Phase-5 complete-K_M proof, just restricted to a
        possibly-incomplete subgraph) -- the PP-only selection SUCCEEDS.
      - if coverage is incomplete (>=1 poison vertex has PP degree 0),
        `len(freq) < M` (that vertex is absent) -- the PP-only selection
        necessarily UNDER-selects (`removed_poison < M`), and because no
        clean vertex is ever present, `removed_clean` is always exactly 0
        for this pair set. The PP-only selection's ONLY possible failure
        mode is under-coverage, never clean-passage displacement.
    """
    poison_indices = [i for i in range(len(is_poison)) if is_poison[i]]
    assert len(poison_indices) == m_poison
    pp_only = [pr for pr in top_pairs if classify_pair(pr[0], pr[1], is_poison) == "PP"]
    degree = {i: 0 for i in poison_indices}
    for i, j, _sim in pp_only:
        degree[i] += 1
        degree[j] += 1
    degrees = [degree[i] for i in poison_indices]
    uncovered = [i for i in poison_indices if degree[i] == 0]
    covered_count = m_poison - len(uncovered)
    return {
        "n_poison_vertices": m_poison,
        "n_poison_covered_by_PP": covered_count,
        "n_poison_uncovered_by_PP": len(uncovered),
        "uncovered_poison_indices": uncovered,
        "min_poison_pp_degree": min(degrees) if degrees else None,
        "median_poison_pp_degree": float(np.median(degrees)) if degrees else None,
        "max_poison_pp_degree": max(degrees) if degrees else None,
        "pp_vertex_coverage_complete": bool(len(uncovered) == 0),
        "pp_only_pairs": pp_only,
    }


def classify_mechanism_v2(pp_vertex_coverage_complete: bool, success_variant_c_remove_pc: bool) -> str:
    """Step 5B/5C primary mechanism classification for a failed query:

    - B. PP-COVERAGE-LIMITED: `pp_vertex_coverage_complete` is False --
      the PP-only subgraph of the ORIGINAL P_top structurally cannot cover
      every poison vertex, so no non-PP ablation can repair the query
      (theorem above); this is checked and reported FIRST, independent of
      any PC/CC ablation outcome.
    - A. PC-CONTRIBUTION-DRIVEN: coverage IS complete (so the PP-only
      subgraph alone would already succeed), AND removing the selected PC
      pairs alone (keeping PP+CC, no refill) already restores complete
      identification -- i.e. PC intrusion is what displaced poison in
      this case, not a coverage gap.
    - C. OTHER/UNEXPLAINED: coverage is complete but removing PC alone
      does NOT repair the query (would indicate CC pairs, or a PP+CC
      interaction, drive the residual failure) -- recomputed, not forced.
    """
    if not pp_vertex_coverage_complete:
        return "B. PP-COVERAGE-LIMITED"
    if success_variant_c_remove_pc:
        return "A. PC-CONTRIBUTION-DRIVEN"
    return "C. OTHER/UNEXPLAINED"


# ---------------------------------------------------------------------------
# Step 1/1A/2/2A/2B/2C -- corrected, CERTIFIED minimum pair-swap search
# ---------------------------------------------------------------------------

DEFAULT_HARD_COMBINATION_CAP = 20_000_000
TIE_EPSILON = 1e-9


@dataclass
class SwapAuditRow:
    r: int
    n_removal_combos: int
    n_addition_combos: int
    total_candidates: int
    exhaustive: bool
    combinations_examined: int
    success_found: bool
    first_success_removal: Optional[Pair] = None  # only meaningful for r==1 display; full combo in result


@dataclass
class CertifiedSwapResult:
    identity: dict
    certified_min_swap_count: Optional[int]
    minimum_certified: bool
    largest_fully_exhausted_r: int
    n_combinations_examined: int
    removed_pairs: List[Pair] = field(default_factory=list)
    added_pairs: List[Pair] = field(default_factory=list)
    outcome: Optional[dict] = None
    certified_lower_bound: Optional[int] = None
    successful_upper_bound: Optional[int] = None
    audit_rows: List[SwapAuditRow] = field(default_factory=list)


def _pair_vectors(pairs: Sequence[Pair], k: int, p: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """(contribution, degree) matrices, one row per pair, shape (len(pairs), k).
    Both are exactly additive across pairs (no interaction term), so a
    candidate pair set's total contribution/degree vector is EXACTLY the
    base vector plus/minus the individual per-pair rows for
    added/removed pairs -- no approximation."""
    n = len(pairs)
    contrib = np.zeros((n, k), dtype=np.float64)
    degree = np.zeros((n, k), dtype=np.int64)
    for idx, (i, j, sim) in enumerate(pairs):
        c = pair_signed_score(sim, p)
        contrib[idx, i] += c
        contrib[idx, j] += c
        degree[idx, i] += 1
        degree[idx, j] += 1
    return contrib, degree


def certified_minimum_pair_swap_search(
    top_pairs: Sequence[Pair],
    matrix: np.ndarray,
    is_poison: np.ndarray,
    k: int,
    m_poison: int,
    p: float = 2.0,
    hard_combination_cap: int = DEFAULT_HARD_COMBINATION_CAP,
    tie_epsilon: float = TIE_EPSILON,
) -> CertifiedSwapResult:
    """V2, CORRECTED replacement for `minimal_pair_swap_search`.

    Certifies `minimum_certified=True` ONLY if EVERY swap count
    `1, ..., r-1` was EXHAUSTIVELY searched with no success AND swap count
    `r` was exhaustively searched until a success was found (Step 1). No
    greedy shortcut is ever used to produce a "True" certification: if a
    swap-count level cannot be exhausted within `hard_combination_cap`
    candidates, the search stops immediately and returns
    `minimum_certified=False` with an explicit `[certified_lower_bound,
    successful_upper_bound]` interval (Step 2C) -- it NEVER continues past
    an un-exhausted level to look for a later "exact" success, which is
    precisely the V1 bug.

    Speed (Step 2B): per-pair contribution/degree vectors are precomputed
    once; a candidate pair set's frequency/degree vectors are obtained by
    exact vector addition/subtraction from the ORIGINAL P_top's vectors
    (no pair-list reconstruction or dict rebuild per candidate). A vertex
    is "present" (eligible for ranking, matching production's `freq` dict
    semantics) iff its degree in the candidate set is > 0 -- tracked
    explicitly via the degree vector, never inferred from a merely-zero
    score. Success (`removed_poison==M, removed_clean==0`) is determined
    by strict comparison of the minimum present-poison score against the
    maximum present-clean score; ties within `tie_epsilon` (essentially
    impossible with real float64 cosine similarities, but included for
    correctness) trigger an exact fallback recomputation via
    `compute_frequency_and_selection` on the fully-reconstructed candidate
    pair list, so no candidate is ever misclassified by the fast path.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    identity = pair_count_identity(top_pairs, matrix, is_poison, k)
    non_pp_pairs = sorted(identity["non_pp_pairs"], key=lambda t: t[2])  # weakest removal candidate first
    unselected_pp_pairs = sorted(identity["missing_pp_pairs"], key=lambda t: t[2], reverse=True)  # strongest addition first
    q_removal = len(non_pp_pairs)
    q_addition = len(unselected_pp_pairs)

    is_poison_arr = np.asarray(is_poison, dtype=bool)
    base_pairs = list(top_pairs)
    base_selection = compute_frequency_and_selection(base_pairs, k, m_poison, p)
    base_freq = base_selection.frequency_scores.copy()
    base_degree = np.zeros(k, dtype=np.int64)
    for i, j, _sim in base_pairs:
        base_degree[i] += 1
        base_degree[j] += 1

    removal_contrib, removal_degree = _pair_vectors(non_pp_pairs, k, p)
    addition_contrib, addition_degree = _pair_vectors(unselected_pp_pairs, k, p)

    q_max = min(q_removal, q_addition)
    audit_rows: List[SwapAuditRow] = []
    total_examined = 0
    largest_fully_exhausted = 0

    def exact_fallback_success(removal_combo, addition_combo) -> Optional[dict]:
        remove_set = {(pr[0], pr[1]) for pr in removal_combo}
        remaining = [pr for pr in base_pairs if (pr[0], pr[1]) not in remove_set]
        candidate = remaining + list(addition_combo)
        selection = compute_frequency_and_selection(candidate, k, m_poison, p)
        return removal_outcome(selection, is_poison_arr, m_poison)

    for r in range(1, q_max + 1):
        removal_idx_combos = list(itertools.combinations(range(q_removal), r))
        addition_idx_combos = list(itertools.combinations(range(q_addition), r))
        n_removal = len(removal_idx_combos)
        n_addition = len(addition_idx_combos)
        total_this_r = n_removal * n_addition

        if total_this_r > hard_combination_cap:
            return CertifiedSwapResult(
                identity=identity,
                certified_min_swap_count=None,
                minimum_certified=False,
                largest_fully_exhausted_r=largest_fully_exhausted,
                n_combinations_examined=total_examined,
                certified_lower_bound=largest_fully_exhausted + 1,
                successful_upper_bound=None,
                audit_rows=audit_rows,
            )

        removal_idx_matrix = np.array(removal_idx_combos, dtype=np.int64) if n_removal else np.zeros((0, r), dtype=np.int64)
        addition_idx_matrix = np.array(addition_idx_combos, dtype=np.int64) if n_addition else np.zeros((0, r), dtype=np.int64)
        removal_freq_deltas = removal_contrib[removal_idx_matrix].sum(axis=1) if n_removal else np.zeros((0, k))
        removal_deg_deltas = removal_degree[removal_idx_matrix].sum(axis=1) if n_removal else np.zeros((0, k), dtype=np.int64)
        addition_freq_deltas = addition_contrib[addition_idx_matrix].sum(axis=1) if n_addition else np.zeros((0, k))
        addition_deg_deltas = addition_degree[addition_idx_matrix].sum(axis=1) if n_addition else np.zeros((0, k), dtype=np.int64)

        success_found = False
        found_removal_combo = None
        found_addition_combo = None
        examined_this_r = 0

        for ri in range(n_removal):
            freq_after_removal = base_freq - removal_freq_deltas[ri]
            deg_after_removal = base_degree - removal_deg_deltas[ri]

            freq_final = freq_after_removal[None, :] + addition_freq_deltas  # (n_addition, k)
            deg_final = deg_after_removal[None, :] + addition_deg_deltas  # (n_addition, k)
            present = deg_final > 0

            poison_present_ok = present[:, is_poison_arr].all(axis=1)
            clean_present_any = present[:, ~is_poison_arr].any(axis=1) if (~is_poison_arr).any() else np.zeros(n_addition, dtype=bool)

            poison_scores = freq_final[:, is_poison_arr]
            poison_present_mask = present[:, is_poison_arr]
            min_poison_score = np.where(poison_present_mask, poison_scores, np.inf).min(axis=1)

            if (~is_poison_arr).any():
                clean_scores = freq_final[:, ~is_poison_arr]
                clean_present_mask = present[:, ~is_poison_arr]
                max_clean_score = np.where(clean_present_mask, clean_scores, -np.inf).max(axis=1)
            else:
                max_clean_score = np.full(n_addition, -np.inf)

            margin = min_poison_score - max_clean_score
            quick_success = poison_present_ok & (~clean_present_any | (margin > tie_epsilon))
            ambiguous = poison_present_ok & clean_present_any & (np.abs(margin) <= tie_epsilon) & ~quick_success

            examined_this_r += n_addition
            removal_combo = tuple(non_pp_pairs[idx] for idx in removal_idx_combos[ri])

            hit_ai = None
            for ai in range(n_addition):
                if ambiguous[ai]:
                    addition_combo = tuple(unselected_pp_pairs[idx] for idx in addition_idx_combos[ai])
                    exact_outcome = exact_fallback_success(removal_combo, addition_combo)
                    if exact_outcome["success"]:
                        hit_ai = ai
                        break
                elif quick_success[ai]:
                    hit_ai = ai
                    break
            if hit_ai is not None:
                success_found = True
                found_removal_combo = removal_combo
                found_addition_combo = tuple(unselected_pp_pairs[idx] for idx in addition_idx_combos[hit_ai])
                break

        total_examined += examined_this_r
        audit_rows.append(
            SwapAuditRow(
                r=r,
                n_removal_combos=n_removal,
                n_addition_combos=n_addition,
                total_candidates=total_this_r,
                exhaustive=True,
                combinations_examined=examined_this_r,
                success_found=success_found,
            )
        )

        if success_found:
            outcome = exact_fallback_success(found_removal_combo, found_addition_combo)
            return CertifiedSwapResult(
                identity=identity,
                certified_min_swap_count=r,
                minimum_certified=True,
                largest_fully_exhausted_r=r,
                n_combinations_examined=total_examined,
                removed_pairs=list(found_removal_combo),
                added_pairs=list(found_addition_combo),
                outcome=outcome,
                audit_rows=audit_rows,
            )

        largest_fully_exhausted = r

    # Exhausted every r up to q_max with no success -- should not happen
    # given the Phase-5 full-swap feasibility proof (r=q_max IS the
    # complete-PP swap), but return a well-formed result rather than
    # raising if it somehow does.
    return CertifiedSwapResult(
        identity=identity,
        certified_min_swap_count=None,
        minimum_certified=True,  # exhausted every level, genuinely unreachable via pair swap
        largest_fully_exhausted_r=largest_fully_exhausted,
        n_combinations_examined=total_examined,
        audit_rows=audit_rows,
    )
