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
    """Minimum number of (remove one non-PP selected pair, add one
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
