"""REGIME-C STAGE-2 V2 VALIDATION PASS.

Correction + validation pass over the V1 Regime-C Stage-2 Identification-
Capacity Study (`run_ragdefender_regime_c_stage2.py`). Fixes three V1
issues without altering the V1 population/frequency/pair findings (which
are re-verified, not recomputed differently, here):

1. The pair-swap "exact minimum" CERTIFICATION bug (see
   `ragdefender_regime_c_stage2_lib.certified_minimum_pair_swap_search`
   docstring) -- V1's `minimal_pair_swap_search` could silently skip
   exhaustive search of a SMALLER swap count and still label a LARGER
   count "exact". V2 uses a corrected, properly-certified search.
2. Replaces the vague "PP-weighting/other" ablation bucket with the exact
   graph-theoretic PP-COVERAGE-LIMITED mechanism, backed by a proved and
   computationally-verified theorem (see
   `ragdefender_regime_c_stage2_lib.pp_coverage_analysis`).
3. Corrects causal overclaims in the V1 report text (pure-PP oracle
   causality, `score_overlap` causality) -- handled in the V2 report
   prose, not in any recomputed number.

Still a pure OFFLINE Stage-2 diagnostic: true count M held fixed, Stage 1
never consulted, no retrieval/Stella/generation/API. Reuses
`run_ragdefender_regime_c_stage2.load_regime_c_cases()` (V1's Phase-0
loader, unmodified) to avoid re-deriving population-loading logic; does
NOT overwrite any V1 output file.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ragdefender_regime_c_stage2_lib as lib  # noqa: E402
import run_ragdefender_regime_c_stage2 as v1  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_regime_c_stage2"


class RegimeCStageV2StopCondition(RuntimeError):
    pass


def _check_outputs_do_not_overwrite(paths: List[Path]) -> None:
    existing = [p for p in paths if p.exists()]
    if existing:
        raise RegimeCStageV2StopCondition(f"Refusing to overwrite existing V2 outputs: {existing}")


# ---------------------------------------------------------------------------
# Step 0 -- reload and re-verify the V1 population (must reproduce exactly;
# STOP condition otherwise)
# ---------------------------------------------------------------------------

def load_and_reverify() -> List[dict]:
    cases = v1.load_regime_c_cases()  # already raises on any Phase-0 mismatch
    if len(cases) != 20:
        raise RegimeCStageV2StopCondition(f"Expected 20 cases, found {len(cases)}.")
    n_success = sum(1 for c in cases if c["success"])
    n_failure = len(cases) - n_success
    if not (n_success == 4 and n_failure == 16):
        raise RegimeCStageV2StopCondition(f"Expected 4/16 success/failure, found {n_success}/{n_failure}.")
    return cases


# ---------------------------------------------------------------------------
# Step 2/3 -- corrected pair-swap certification for all 16 failures
# ---------------------------------------------------------------------------

def build_pair_swap_v2_rows(cases: List[dict]):
    rows = []
    audit_rows = []
    for case in cases:
        if case["success"]:
            continue
        qid = case["query_id"]
        k, M, C = case["k"], case["m_poison"], case["c_clean"]
        is_poison = case["is_poison"]

        identity = lib.pair_count_identity(case["top_pairs"], case["matrix"], is_poison, k)
        result = lib.certified_minimum_pair_swap_search(case["top_pairs"], case["matrix"], is_poison, k, M)

        removed_desc = "|".join(f"{i}-{j}:{sim:.4f}" for i, j, sim in result.removed_pairs)
        removed_classes = "|".join(lib.classify_pair(i, j, is_poison) for i, j, _ in result.removed_pairs)
        added_desc = "|".join(f"{i}-{j}:{sim:.4f}" for i, j, sim in result.added_pairs)

        rows.append(
            {
                "query_id": qid,
                "m_poison": M,
                "c_clean": C,
                "n_pairs": case["n_pairs"],
                "q_selected_non_pp": identity["q_selected_non_pp"],
                "q_missing_pp": identity["q_missing_pp"],
                "identity_holds": identity["identity_holds"],
                "certified_min_swap_count": result.certified_min_swap_count,
                "minimum_certified": result.minimum_certified,
                "n_combinations_examined": result.n_combinations_examined,
                "largest_fully_exhausted_r": result.largest_fully_exhausted_r,
                "certified_lower_bound": result.certified_lower_bound,
                "successful_upper_bound": result.successful_upper_bound,
                "removed_non_pp_pairs": removed_desc,
                "removed_non_pp_pair_classes": removed_classes,
                "added_pp_pairs": added_desc,
                "resulting_removed_poison": result.outcome["removed_poison"] if result.outcome else None,
                "resulting_removed_clean": result.outcome["removed_clean"] if result.outcome else None,
                "resulting_residual_poison": result.outcome["residual_poison"] if result.outcome else None,
                "resulting_success": result.outcome["success"] if result.outcome else None,
                "pair_swap_fraction": (
                    result.certified_min_swap_count / case["n_pairs"] if result.certified_min_swap_count is not None else None
                ),
            }
        )

        for audit in result.audit_rows:
            audit_rows.append(
                {
                    "query_id": qid,
                    "r": audit.r,
                    "n_removal_combos": audit.n_removal_combos,
                    "n_addition_combos": audit.n_addition_combos,
                    "total_candidates": audit.total_candidates,
                    "exhaustive": audit.exhaustive,
                    "combinations_examined": audit.combinations_examined,
                    "success_found": audit.success_found,
                }
            )
    return rows, audit_rows


# ---------------------------------------------------------------------------
# Step 5/5A -- PP coverage analysis for all 20 queries
# ---------------------------------------------------------------------------

def build_pp_coverage_v2_rows(cases: List[dict]):
    rows = []
    for case in cases:
        qid = case["query_id"]
        k, M, C = case["k"], case["m_poison"], case["c_clean"]
        is_poison = case["is_poison"]
        cov = lib.pp_coverage_analysis(case["top_pairs"], is_poison, M)

        variants = lib.ablation_variants(case["top_pairs"], is_poison)
        selection_d = lib.compute_frequency_and_selection(variants["D_pp_only"], k, M)
        outcome_d = lib.removal_outcome(selection_d, is_poison, M)

        theorem_holds = cov["pp_vertex_coverage_complete"] == outcome_d["success"]

        rows.append(
            {
                "query_id": qid,
                "m_poison": M,
                "c_clean": C,
                "success_true_count": case["success"],
                "n_poison_vertices": cov["n_poison_vertices"],
                "n_poison_covered_by_PP": cov["n_poison_covered_by_PP"],
                "n_poison_uncovered_by_PP": cov["n_poison_uncovered_by_PP"],
                "uncovered_poison_indices": "|".join(map(str, cov["uncovered_poison_indices"])),
                "min_poison_pp_degree": cov["min_poison_pp_degree"],
                "median_poison_pp_degree": cov["median_poison_pp_degree"],
                "max_poison_pp_degree": cov["max_poison_pp_degree"],
                "pp_vertex_coverage_complete": cov["pp_vertex_coverage_complete"],
                "variant_D_pp_only_success": outcome_d["success"],
                "variant_D_removed_poison": outcome_d["removed_poison"],
                "variant_D_removed_clean": outcome_d["removed_clean"],
                "theorem_holds": theorem_holds,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Step 5B/5C -- mechanism classification for the 16 failures
# ---------------------------------------------------------------------------

def build_mechanism_classification_v2_rows(cases: List[dict]):
    rows = []
    for case in cases:
        if case["success"]:
            continue
        qid = case["query_id"]
        k, M = case["k"], case["m_poison"]
        is_poison = case["is_poison"]
        cov = lib.pp_coverage_analysis(case["top_pairs"], is_poison, M)
        variants = lib.ablation_variants(case["top_pairs"], is_poison)

        outcomes = {}
        for name in ("B_remove_CC", "C_remove_PC", "D_pp_only"):
            sel = lib.compute_frequency_and_selection(variants[name], k, M)
            outcomes[name] = lib.removal_outcome(sel, is_poison, M)

        label = lib.classify_mechanism_v2(cov["pp_vertex_coverage_complete"], outcomes["C_remove_PC"]["success"])

        # Step 5C exclusivity check: does this query ALSO exhibit the other
        # mechanism's defining property? (i.e. is coverage complete AND
        # C fails, or coverage incomplete AND C succeeds anyway -- the
        # latter should be provably impossible per the coverage theorem,
        # since C keeps every PP edge D has plus CC edges, so if D fails
        # due to missing PP degree, C cannot magically cover that vertex
        # either -- verified explicitly, not assumed.)
        exclusive = not (cov["pp_vertex_coverage_complete"] != outcomes["D_pp_only"]["success"])
        overlap_note = ""
        if label == "A. PC-CONTRIBUTION-DRIVEN" and not cov["pp_vertex_coverage_complete"]:
            overlap_note = "UNEXPECTED: PC-driven but coverage incomplete"
        if label == "B. PP-COVERAGE-LIMITED" and outcomes["C_remove_PC"]["success"]:
            overlap_note = "UNEXPECTED: coverage-limited but C_remove_PC alone succeeds"

        rows.append(
            {
                "query_id": qid,
                "m_poison": M,
                "pp_vertex_coverage_complete": cov["pp_vertex_coverage_complete"],
                "n_poison_uncovered_by_PP": cov["n_poison_uncovered_by_PP"],
                "variant_B_remove_CC_success": outcomes["B_remove_CC"]["success"],
                "variant_C_remove_PC_success": outcomes["C_remove_PC"]["success"],
                "variant_D_pp_only_success": outcomes["D_pp_only"]["success"],
                "mechanism_classification": label,
                "coverage_theorem_consistent": exclusive,
                "overlap_note": overlap_note,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> dict:
    out_swap_v2 = OUTPUT_DIR / "regime_c_pair_swap_oracle_v2.csv"
    out_coverage_v2 = OUTPUT_DIR / "regime_c_pp_coverage_v2.csv"
    out_mechanism_v2 = OUTPUT_DIR / "regime_c_mechanism_classification_v2.csv"
    out_swap_audit_v2 = OUTPUT_DIR / "regime_c_pair_swap_search_audit_v2.csv"
    out_report_v2 = OUTPUT_DIR / "REGIME_C_STAGE2_V2_REPORT.md"

    _check_outputs_do_not_overwrite([out_swap_v2, out_coverage_v2, out_mechanism_v2, out_swap_audit_v2, out_report_v2])

    cases = load_and_reverify()
    print(f"Re-verified {len(cases)} Regime-C cases (4 success / 16 failure) -- STOP checks passed.")

    swap_rows, audit_rows = build_pair_swap_v2_rows(cases)
    coverage_rows = build_pp_coverage_v2_rows(cases)
    mechanism_rows = build_mechanism_classification_v2_rows(cases)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(out_swap_v2, swap_rows)
    _write_csv(out_coverage_v2, coverage_rows)
    _write_csv(out_mechanism_v2, mechanism_rows)
    _write_csv(out_swap_audit_v2, audit_rows)

    return {
        "cases": cases,
        "swap_rows": swap_rows,
        "audit_rows": audit_rows,
        "coverage_rows": coverage_rows,
        "mechanism_rows": mechanism_rows,
        "output_paths": {
            "swap_v2": out_swap_v2, "coverage_v2": out_coverage_v2, "mechanism_v2": out_mechanism_v2,
            "swap_audit_v2": out_swap_audit_v2, "report_v2": out_report_v2,
        },
    }


if __name__ == "__main__":
    result = run()
    print(
        f"Wrote {len(result['swap_rows'])} pair-swap-v2 rows, {len(result['coverage_rows'])} coverage rows, "
        f"{len(result['mechanism_rows'])} mechanism rows, {len(result['audit_rows'])} audit rows."
    )
