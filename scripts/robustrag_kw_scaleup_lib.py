#!/usr/bin/env python3
"""Pure, stdlib-only logic for the RobustRAG-KW scale-up
(`scripts/run_robustrag_kw_scaleup_bundle1.py`).

Everything here is deterministic and dependency-free so it can be unit
tested without loading Contriever/flan-t5/distilgpt2 and without making a
single GPT/API call. The orchestration script keeps all heavy model work
and all network I/O; this module only decides *which* cases to spend
budget on and *what shape* the published rows take.

Three responsibilities:

1. `select_candidates` -- the pre-registered, deterministic shortlist rule
   for choosing which of the 18 (mutation_family, query_id) cases get
   RobustRAG-KW answer generation.
2. Output schemas (`*_FIELDS`) -- pinned column orders, so a schema change
   is a visible test failure rather than a silently reshaped CSV.
3. `sweep_grid` -- the aggregation/abstention configuration grid, which is
   replayed over already-cached isolated answers at zero API cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Bundle constants.
# ---------------------------------------------------------------------------

#: The three defense-targeted mutation families in `mutation_bundle_1`, in the
#: canonical order used for every deterministic iteration in this module.
MUTATION_FAMILIES: Tuple[str, ...] = (
    "filterrag_targeted",
    "ragdefender_targeted",
    "mlfilterrag_targeted",
)

#: The six bundle query_ids, in the canonical order they appear in
#: `manual_text_mutation_pilot/hotpotqa_50q_k10/selected_queries.csv`.
BUNDLE_QUERY_IDS: Tuple[str, ...] = (
    "5ae224da554299234fd043ee",
    "5aba749055429901930fa7d8",
    "5ae22b8d554299234fd0440f",
    "5a7759fc5542993569682d60",
    "5a8133725542995ce29dcbdb",
    "5a8e068b5542995085b37384",
)

#: The three passage-filtering defenses RobustRAG-KW is compared against.
#: `ml_filterrag_t04` is the pre-registered ML-FilterRAG operating point; the
#: model is never retrained, only its published threshold is read.
FILTER_DEFENSES: Tuple[str, ...] = ("ragdefender", "filterrag_semantic", "ml_filterrag_t04")

#: The 3 cases published in the completed RobustRAG-KW pilot. Used as a
#: regression baseline: their scale-up retrieval must reproduce the published
#: top-10 exactly, and their aggregate verdicts are compared, not overwritten.
PILOT_CASES: Tuple[Tuple[str, str], ...] = (
    ("filterrag_targeted", "5a8e068b5542995085b37384"),
    ("filterrag_targeted", "5ae224da554299234fd043ee"),
    ("filterrag_targeted", "5ae22b8d554299234fd0440f"),
)

ORIGIN_MUTATED_SELF = "mutated_self_query_poison"
ORIGIN_ORIGINAL_SELF = "original_self_query_poison"
ORIGIN_CROSS_QUERY = "cross_query_poison"
ORIGIN_CLEAN = "clean"

#: Shortlist gate: a case is only eligible if the attacker's mutated
#: self-query poison actually survived retrieval this well.
MIN_MUTATED_SELF_RETRIEVED = 4

#: Criterion A: a filtering defense leaves at least this many poisoned
#: passages in the context it hands to the generator.
MIN_RESIDUAL_POISON = 2

#: Criterion B: a filtering defense removes at least this many fewer poisoned
#: passages than it did on the same query's unmutated baseline context.
MIN_REMOVED_POISON_DROP = 2


# ---------------------------------------------------------------------------
# 1. Candidate selection.
# ---------------------------------------------------------------------------

@dataclass
class CaseStats:
    """Everything the shortlist rule is allowed to look at for one
    (mutation_family, query_id) case. Populated by the orchestration script
    from the stage-1 retrieval artifacts; deliberately a plain data holder so
    selection can be tested without any retrieval."""

    family: str
    query_id: str
    n_mutated_self_retrieved: int
    n_retrieved_poison: int
    n_retrieved_clean: int
    #: {defense_name: number of poisoned passages still present after filtering}
    residual_poison_by_defense: Dict[str, int] = field(default_factory=dict)
    #: {defense_name: poisoned passages removed on the mutated context}
    removed_poison_by_defense: Dict[str, int] = field(default_factory=dict)
    #: {defense_name: poisoned passages removed on the unmutated baseline context}
    baseline_removed_poison_by_defense: Dict[str, int] = field(default_factory=dict)


def failure_signature(stats: CaseStats, defenses: Sequence[str] = FILTER_DEFENSES) -> Tuple[bool, ...]:
    """Which filtering defenses leave *any* poison behind, as a boolean
    tuple in fixed `defenses` order. Two cases sharing a signature exercise
    the same qualitative defense-failure mode; a case with a signature no
    earlier case had is 'a distinct failure mode worth reporting'."""
    return tuple(stats.residual_poison_by_defense.get(d, 0) >= 1 for d in defenses)


def _criterion_a(stats: CaseStats, defenses: Sequence[str]) -> List[str]:
    return [d for d in defenses if stats.residual_poison_by_defense.get(d, 0) >= MIN_RESIDUAL_POISON]


def _criterion_b(stats: CaseStats, defenses: Sequence[str]) -> List[str]:
    hits = []
    for d in defenses:
        base = stats.baseline_removed_poison_by_defense.get(d)
        mutated = stats.removed_poison_by_defense.get(d)
        if base is None or mutated is None:
            continue
        if (base - mutated) >= MIN_REMOVED_POISON_DROP:
            hits.append(d)
    return hits


def select_candidates(
    cases: Sequence[CaseStats],
    *,
    defenses: Sequence[str] = FILTER_DEFENSES,
    min_mutated_self_retrieved: int = MIN_MUTATED_SELF_RETRIEVED,
) -> List[Dict]:
    """Apply the pre-registered shortlist rule to every case and return one
    decision row per case, in a fully deterministic order.

    A case is shortlisted iff the retrieval gate holds

        n_mutated_self_retrieved >= min_mutated_self_retrieved

    **and** at least one of:

    - **A** a filtering defense leaves >= `MIN_RESIDUAL_POISON` poisoned
      passages in the generator's context;
    - **B** a filtering defense's `removed_poison` drops by >=
      `MIN_REMOVED_POISON_DROP` versus that query's unmutated baseline;
    - **C** the case's `failure_signature` has not been claimed by an
      earlier selected case (a distinct defense-failure mode).

    Criterion C is order-dependent by construction, so cases are always
    evaluated in `(MUTATION_FAMILIES order, BUNDLE_QUERY_IDS order)`; any
    case whose family or query is outside those tuples sorts last, by name.
    Passing the same `cases` in any input order yields identical output.
    """
    def sort_key(c: CaseStats) -> Tuple[int, str, int, str]:
        fam_rank = (MUTATION_FAMILIES.index(c.family)
                    if c.family in MUTATION_FAMILIES else len(MUTATION_FAMILIES))
        qid_rank = (BUNDLE_QUERY_IDS.index(c.query_id)
                    if c.query_id in BUNDLE_QUERY_IDS else len(BUNDLE_QUERY_IDS))
        return (fam_rank, c.family, qid_rank, c.query_id)

    ordered = sorted(cases, key=sort_key)
    claimed_signatures: set = set()
    rows: List[Dict] = []

    for stats in ordered:
        gate = stats.n_mutated_self_retrieved >= min_mutated_self_retrieved
        a_hits = _criterion_a(stats, defenses)
        b_hits = _criterion_b(stats, defenses)
        signature = failure_signature(stats, defenses)
        # C is only meaningful for gate-passing cases: a case that failed the
        # retrieval gate must never claim a signature, or it would suppress a
        # later eligible case that genuinely exhibits that mode.
        c_hit = gate and signature not in claimed_signatures

        selected = bool(gate and (a_hits or b_hits or c_hit))
        if selected:
            claimed_signatures.add(signature)

        reasons: List[str] = []
        if not gate:
            reasons.append(
                f"gate_failed(mutated_self_retrieved={stats.n_mutated_self_retrieved}"
                f"<{min_mutated_self_retrieved})"
            )
        else:
            if a_hits:
                reasons.append("A:residual_poison>=%d[%s]" % (MIN_RESIDUAL_POISON, "|".join(a_hits)))
            if b_hits:
                reasons.append("B:removed_poison_drop>=%d[%s]" % (MIN_REMOVED_POISON_DROP, "|".join(b_hits)))
            if c_hit:
                reasons.append("C:new_failure_signature%s" % _signature_str(signature, defenses))
            if not reasons:
                reasons.append("no_criterion_met(signature_already_covered)")

        rows.append({
            "family": stats.family,
            "query_id": stats.query_id,
            "n_mutated_self_retrieved": stats.n_mutated_self_retrieved,
            "n_retrieved_poison": stats.n_retrieved_poison,
            "n_retrieved_clean": stats.n_retrieved_clean,
            "retrieval_gate_pass": gate,
            "criterion_a_residual_poison": bool(a_hits),
            "criterion_a_defenses": "|".join(a_hits),
            "criterion_b_removed_poison_drop": bool(b_hits),
            "criterion_b_defenses": "|".join(b_hits),
            "criterion_c_new_failure_mode": bool(c_hit),
            "failure_signature": _signature_str(signature, defenses),
            "selected": selected,
            "selection_reason": ";".join(reasons),
            **{f"residual_poison_{d}": stats.residual_poison_by_defense.get(d) for d in defenses},
            **{f"removed_poison_{d}": stats.removed_poison_by_defense.get(d) for d in defenses},
            **{f"baseline_removed_poison_{d}": stats.baseline_removed_poison_by_defense.get(d)
               for d in defenses},
        })
    return rows


def _signature_str(signature: Sequence[bool], defenses: Sequence[str]) -> str:
    return "(" + ",".join(f"{d}={'leaks' if s else 'clears'}"
                          for d, s in zip(defenses, signature)) + ")"


# ---------------------------------------------------------------------------
# 2. Aggregation / abstention sweep grid.
# ---------------------------------------------------------------------------

#: Vote-threshold variants swept over cached answers.
SWEEP_VOTE_THRESHOLDS: Tuple[float, ...] = (0.0, 0.5, 0.6, 0.7)
#: Abstain-if-low-margin variants (0.0 disables the rule).
SWEEP_ABSTAIN_THRESHOLDS: Tuple[float, ...] = (0.0, 0.2)
SWEEP_ABSTENTION_POLICIES: Tuple[str, ...] = ("discard_abstentions", "include_abstentions")
SWEEP_NORMALIZATION_MODES: Tuple[str, ...] = ("squad", "token")
SWEEP_AGGREGATION_MODES: Tuple[str, ...] = ("exact", "keyword")


def sweep_grid(
    *,
    vote_thresholds: Sequence[float] = SWEEP_VOTE_THRESHOLDS,
    abstain_thresholds: Sequence[float] = SWEEP_ABSTAIN_THRESHOLDS,
    abstention_policies: Sequence[str] = SWEEP_ABSTENTION_POLICIES,
    normalization_modes: Sequence[str] = SWEEP_NORMALIZATION_MODES,
    aggregation_modes: Sequence[str] = SWEEP_AGGREGATION_MODES,
) -> List[Dict]:
    """The full cartesian sweep, in a fixed nesting order so the emitted CSV
    row order is stable across runs. Every configuration is replayed over
    already-cached isolated answers -- this function exists so the grid is a
    testable value rather than a nested loop buried in the orchestrator."""
    grid: List[Dict] = []
    for policy in abstention_policies:
        for norm in normalization_modes:
            for agg in aggregation_modes:
                for vt in vote_thresholds:
                    for at in abstain_thresholds:
                        grid.append({
                            "abstention_policy": policy,
                            "normalization_mode": norm,
                            "aggregation_mode": agg,
                            "vote_threshold": float(vt),
                            "abstain_threshold": float(at),
                        })
    return grid


# ---------------------------------------------------------------------------
# 3. Published output schemas (pinned).
# ---------------------------------------------------------------------------

CANDIDATE_SELECTION_FIELDS: Tuple[str, ...] = (
    "family", "query_id",
    "n_mutated_self_retrieved", "n_retrieved_poison", "n_retrieved_clean",
    "retrieval_gate_pass",
    "criterion_a_residual_poison", "criterion_a_defenses",
    "criterion_b_removed_poison_drop", "criterion_b_defenses",
    "criterion_c_new_failure_mode",
    "failure_signature", "selected", "selection_reason",
    "residual_poison_ragdefender", "residual_poison_filterrag_semantic",
    "residual_poison_ml_filterrag_t04",
    "removed_poison_ragdefender", "removed_poison_filterrag_semantic",
    "removed_poison_ml_filterrag_t04",
    "baseline_removed_poison_ragdefender", "baseline_removed_poison_filterrag_semantic",
    "baseline_removed_poison_ml_filterrag_t04",
)

#: Per-isolated-answer diagnostic record. Every field named in the scale-up
#: task is present; ordering is pinned so a reshaped record fails a test.
ISOLATED_ANSWER_FIELDS: Tuple[str, ...] = (
    "query_id", "mutation_family", "bundle_id", "context_type",
    "doc_id", "retrieved_rank",
    "is_clean", "is_poison", "is_self_query_poison", "is_cross_query_poison",
    "raw_answer", "extracted_answer", "normalized_answer",
    "is_abstain",
    "matches_target_wrong_answer_strict", "matches_correct_answer_strict",
    "prompt_sha256", "model_name", "generation_session_id",
    "origin_label", "true_owning_query_id", "true_global_index", "is_mutated",
    "group_index", "cache_hit",
)

VOTE_SUMMARY_FIELDS: Tuple[str, ...] = (
    "family", "query_id", "bundle_id", "context_type",
    "abstention_policy", "normalization_mode", "aggregation_mode",
    "vote_threshold", "abstain_threshold",
    "final_answer", "abstained", "winning_normalized_answer",
    "winning_vote_count", "winning_vote_share", "vote_margin",
    "denominator", "n_groups", "n_abstentions",
    "strict_asr_success", "correct_answer_match",
)

GENERATION_RESULTS_FIELDS: Tuple[str, ...] = (
    "family", "query_id", "bundle_id", "context_type", "question",
    "target_wrong_answer", "correct_answer",
    "n_retrieved_poison", "n_retrieved_clean",
    "n_self_query_poison", "n_cross_query_poison",
    "final_answer", "abstained",
    "strict_asr_success", "contains_target_wrong_answer", "contains_correct_answer",
    "correct_answer_match",
    "n_isolated_calls", "n_cache_hits", "n_abstentions", "abstention_rate",
    "wrong_answer_vote_share", "correct_answer_vote_share", "vote_margin",
    "generation_session_id", "model_name",
)

VS_EXISTING_DEFENSES_FIELDS: Tuple[str, ...] = (
    "family", "query_id", "bundle_id", "context_type",
    "defense_name", "threshold", "defense_family",
    "retrieved_poison_count", "removed_poison", "remaining_poison",
    "residual_poison_fraction",
    "final_answer", "strict_asr_success",
    "contains_target_wrong_answer", "contains_correct_answer",
    "abstained", "source",
)

ABSTENTION_SWEEP_FIELDS: Tuple[str, ...] = (
    "abstention_policy", "normalization_mode", "aggregation_mode",
    "vote_threshold", "abstain_threshold",
    "n_cases", "n_abstained", "abstention_rate",
    "n_strict_asr_success", "strict_asr_rate",
    "n_correct_answer_match", "correct_answer_match_rate",
)

ORIGIN_BREAKDOWN_FIELDS: Tuple[str, ...] = (
    "family", "query_id", "context_type", "origin_group",
    "n_passages", "n_strict_asr_hit", "n_gold_match", "n_abstain", "n_other",
    "rate_strict_asr_hit", "rate_gold_match", "rate_abstain",
)


# ---------------------------------------------------------------------------
# 4. Small shared helpers.
# ---------------------------------------------------------------------------

def origin_group(origin_label: Optional[str]) -> str:
    """Collapse an `origin_label` into the three reporting buckets used by
    every RobustRAG-KW diagnostic: clean / self_query_poison /
    cross_query_poison. `original_self_query_poison` is an anomaly label that
    must never appear under replacement-only budgets, so it is surfaced
    verbatim rather than folded into `self_query_poison`."""
    if origin_label == ORIGIN_CLEAN:
        return "clean"
    if origin_label == ORIGIN_MUTATED_SELF:
        return "self_query_poison"
    if origin_label == ORIGIN_CROSS_QUERY:
        return "cross_query_poison"
    if origin_label == ORIGIN_ORIGINAL_SELF:
        return ORIGIN_ORIGINAL_SELF
    return "unknown"


def rate(numerator: int, denominator: int) -> Optional[float]:
    return (numerator / denominator) if denominator else None
