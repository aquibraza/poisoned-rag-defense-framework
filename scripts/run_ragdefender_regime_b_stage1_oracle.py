"""REGIME-B STAGE-1 BOUNDARY-SENSITIVITY ORACLE.

Uses ONLY the 19 already-saved Regime-B Stella similarity matrices from
`results/diagnostics/ragdefender_expanded_baseline/`. Purely an OFFLINE
matrix/statistic analysis -- no retrieval, no Stella re-encoding, no
generation, no E1/CORAL/MMD, no LLM/API calls, no modification of
`ragdefender_paper` or any historical artifact.

Implements PHASES 1-6 (see `scripts/ragdefender_regime_b_stage1_oracle_lib.py`
for the pure-function library this script calls):

1.  Exact Stage-1 boundary decomposition per passage/query.
1A. Binding-condition classification (median-limited / mean-gated /
    both-limited / ceiling-reached).
1B. Median tie/separation analysis.
1C. Mean-gate candidate analysis.
2.  Success-vs-failure descriptive comparison.
3.  Statistic-space minimal oracle (idealized, not matrix-realizable).
4.  Symmetric similarity-matrix oracle (realizable-geometry perturbation).
5.  Causal check through the UNCHANGED Stage 2.
6.  Primary-question answers (in the written report).

CONCEPTUAL RULE: primary candidate selection in Phases 3/4 NEVER uses
`is_poison` -- ground-truth labels are attached to already-selected
candidates afterward, for interpretation only (see the library module's
docstring and every `*_lib.py` function's own docstring).

Writes ONLY to `results/diagnostics/ragdefender_regime_b_stage1_oracle/`.
Never overwrites any historical artifact under
`results/diagnostics/ragdefender_expanded_baseline/` or
`results/diagnostics/ragdefender_expanded_gate_c/`.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from defense import ragdefender_internals as ri  # noqa: E402
import ragdefender_regime_b_stage1_oracle_lib as lib  # noqa: E402

BASELINE_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_baseline"
OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_regime_b_stage1_oracle"
CEILING = 5  # floor(k/2) for k=10, the Regime-B structural ceiling


class RegimeBOracleStopCondition(RuntimeError):
    """Raised for any of the task's documented STOP conditions."""


def _check_no_overwrite(paths: List[Path]) -> None:
    existing = [p for p in paths if p.exists()]
    if existing:
        raise RegimeBOracleStopCondition(f"Refusing to overwrite existing outputs: {existing}")


# ---------------------------------------------------------------------------
# Load the frozen Regime-B population (read-only)
# ---------------------------------------------------------------------------

def load_regime_b_cases() -> List[dict]:
    per_query_csv = BASELINE_DIR / "expanded_baseline_per_query.csv"
    if not per_query_csv.exists():
        raise RegimeBOracleStopCondition(f"{per_query_csv} not found.")
    with open(per_query_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["regime"] == "B_AT_CEILING"]

    if len(rows) != 19:
        raise RegimeBOracleStopCondition(
            f"Expected exactly 19 Regime-B queries, found {len(rows)}."
        )
    n_success = sum(1 for r in rows if r["zero_residual_poison_success"] == "True")
    n_failure = len(rows) - n_success
    if not (n_success == 5 and n_failure == 14):
        raise RegimeBOracleStopCondition(
            f"Expected 5 successes / 14 failures, found {n_success} successes / {n_failure} failures."
        )

    with open(BASELINE_DIR / "recovered_contexts.json") as f:
        contexts_by_id = {c["query_id"]: c for c in json.load(f)}

    cases = []
    for row in rows:
        qid = row["query_id"]
        ctx = contexts_by_id.get(qid)
        if ctx is None:
            raise RegimeBOracleStopCondition(f"{qid}: not found in recovered_contexts.json.")
        matrix_path = BASELINE_DIR / "similarity" / f"{qid}_stella_similarity_matrix.npy"
        if not matrix_path.exists():
            raise RegimeBOracleStopCondition(f"{qid}: historical matrix missing at {matrix_path}.")
        matrix = np.load(matrix_path)  # read-only load
        is_poison = np.array(ctx["is_poison"], dtype=bool)

        stage1 = ri.concentration_stage1_paper(matrix)
        if stage1.n_adv_estimated != int(row["n_adv"]):
            raise RegimeBOracleStopCondition(
                f"{qid}: recomputed N_adv ({stage1.n_adv_estimated}) != "
                f"historical N_adv ({row['n_adv']}) -- historical matrix does not reproduce "
                "its own recorded baseline result."
            )
        historical_success = row["zero_residual_poison_success"] == "True"
        recomputed_success = stage1.n_adv_estimated >= CEILING
        if historical_success != recomputed_success:
            raise RegimeBOracleStopCondition(
                f"{qid}: recomputed success status ({recomputed_success}) != "
                f"historical ({historical_success})."
            )

        # Historical Stage-2 frequency scores, for Phase 1C diagnostic only.
        freq_scores = [float(x) for x in row["frequency_scores_i"].split("|")]

        cases.append(
            {
                "query_id": qid,
                "matrix": matrix,
                "is_poison": is_poison,
                "stage1": stage1,
                "historical_n_adv": int(row["n_adv"]),
                "historical_success": historical_success,
                "historical_residual_poison": int(row["residual_poison"]),
                "historical_freq_scores": freq_scores,
                "m_poison": int(row["m_poison"]),
            }
        )
    return cases


# ---------------------------------------------------------------------------
# PHASE 1 / 1A / 1B / 1C
# ---------------------------------------------------------------------------

def build_boundary_and_margin_rows(cases: List[dict]):
    boundary_rows = []
    margin_rows = []
    for case in cases:
        qid = case["query_id"]
        stage1 = case["stage1"]
        is_poison = case["is_poison"]
        counts = lib.query_level_counts(stage1)
        binding = lib.classify_binding_condition(counts["n_above_median"], counts["n_and"], CEILING)
        mgap = lib.median_rank_gap_analysis(stage1.s_median)
        mg_candidates = lib.mean_gate_candidates(stage1)

        boundary_rows.append(
            {
                "query_id": qid,
                "historical_n_adv": case["historical_n_adv"],
                "historical_success": case["historical_success"],
                "n_above_mean": counts["n_above_mean"],
                "n_above_median": counts["n_above_median"],
                "n_and": counts["n_and"],
                "binding_classification": binding,
                "median_rank5": mgap["median_rank5"],
                "median_rank6": mgap["median_rank6"],
                "median_gap": mgap["median_gap"],
                "rank5_equals_rank6": mgap["rank5_equals_rank6"],
                "n_ties_1e-8": mgap["tie_counts_by_tolerance"][1e-8],
                "n_ties_1e-6": mgap["tie_counts_by_tolerance"][1e-6],
                "n_ties_1e-4": mgap["tie_counts_by_tolerance"][1e-4],
                "n_mean_gate_candidates": len(mg_candidates),
                "mean_gate_candidate_indices": "|".join(map(str, mg_candidates)),
            }
        )

        margins = lib.passage_margins(stage1)
        for m in margins:
            i = m["index"]
            margin_rows.append(
                {
                    "query_id": qid,
                    "index": i,
                    "s_mean_i": m["s_mean_i"],
                    "s_median_i": m["s_median_i"],
                    "mean_margin_i": m["mean_margin_i"],
                    "median_margin_i": m["median_margin_i"],
                    "above_mean_i": m["above_mean_i"],
                    "above_median_i": m["above_median_i"],
                    "and_flag_i": m["and_flag_i"],
                    "mean_rank_desc": lib.mean_rank_descending(stage1.s_mean, i),
                    "is_mean_gate_candidate": i in mg_candidates,
                    "stage2_historical_frequency_score": case["historical_freq_scores"][i],
                    "is_poison_i_DIAGNOSTIC_ONLY": bool(is_poison[i]),
                }
            )
    return boundary_rows, margin_rows


# ---------------------------------------------------------------------------
# PHASE 2
# ---------------------------------------------------------------------------

def build_phase2_comparison(cases: List[dict]) -> dict:
    success_stats: dict = {k: [] for k in [
        "n_above_mean", "n_above_median", "n_and", "median_gap",
        "min_positive_mean_margin", "closest_negative_mean_margin",
        "min_positive_median_margin", "closest_negative_median_margin",
        "n_near_ties_1e-6", "smallest_shortfall_to_and",
    ]}
    failure_stats = {k: [] for k in success_stats}

    for case in cases:
        row = lib.success_vs_failure_row(case["stage1"], CEILING)
        target = success_stats if case["historical_success"] else failure_stats
        for k in target:
            target[k].append(row[k])

    comparison = {}
    for k in success_stats:
        comparison[k] = {
            "success": lib.descriptive_summary(success_stats[k]),
            "failure": lib.descriptive_summary(failure_stats[k]),
        }
    return comparison


# ---------------------------------------------------------------------------
# PHASE 3
# ---------------------------------------------------------------------------

def build_statistic_oracle_rows(failures: List[dict]):
    rows = []
    per_query_best = {}
    for case in failures:
        qid = case["query_id"]
        is_poison = case["is_poison"]
        results = lib.statistic_space_oracle_for_query(case["stage1"])
        best = lib.best_statistic_oracle_result(results)
        for r in results:
            rows.append(
                {
                    "query_id": qid,
                    "candidate_index": r.candidate_index,
                    "is_poison_DIAGNOSTIC_ONLY": bool(is_poison[r.candidate_index]),
                    "mean_only_delta": r.mean_only_delta,
                    "median_only_delta": r.median_only_delta,
                    "median_only_monotonic": r.median_only_monotonic,
                    "combined_mean_delta": r.combined_mean_delta,
                    "combined_median_delta": r.combined_median_delta,
                    "combined_magnitude": r.combined_magnitude,
                    "resulting_n_adv_mean_only": r.resulting_n_adv_mean_only,
                    "resulting_n_adv_median_only": r.resulting_n_adv_median_only,
                    "resulting_n_adv_combined": r.resulting_n_adv_combined,
                    "sensitivity_class": r.sensitivity_class,
                    "is_best_for_query": (best is not None and r.candidate_index == best.candidate_index),
                }
            )
        per_query_best[qid] = best
    return rows, per_query_best


# ---------------------------------------------------------------------------
# PHASE 4 / 5
# ---------------------------------------------------------------------------

def build_matrix_oracle_rows(failures: List[dict], paths_dir: Path):
    rows = []
    per_query_best = {}
    for case in failures:
        qid = case["query_id"]
        matrix = case["matrix"]
        is_poison = case["is_poison"]
        stage1 = case["stage1"]
        non_and = [i for i in range(len(stage1.s_mean)) if not bool(stage1.adv_flag[i])]

        results = lib.matrix_oracle_for_query(matrix, non_and, target_n_adv=CEILING, alpha_max=2.0)
        best = lib.best_matrix_oracle_result(results)
        per_query_best[qid] = best

        for r in results:
            row = {
                "query_id": qid,
                "candidate_index": r.candidate_index,
                "mode": r.mode,
                "alpha": r.alpha,
                "is_monotonic": r.is_monotonic,
                "achieved_n_adv": r.achieved_n_adv,
                "is_poison_DIAGNOSTIC_ONLY": bool(is_poison[r.candidate_index]),
                "is_best_for_query": (
                    best is not None and r.candidate_index == best.candidate_index and r.mode == best.mode
                ),
            }
            if r.alpha is not None:
                effect = lib.perturbation_effect_size(matrix, r.perturbed_matrix)
                row.update(
                    {
                        "max_abs_off_diag_change": effect["max_abs_off_diag_change"],
                        "mean_abs_off_diag_change": effect["mean_abs_off_diag_change"],
                        "frobenius_norm_diff": effect["frobenius_norm_diff"],
                        "fraction_off_diag_changed": effect["fraction_off_diag_changed"],
                    }
                )
                # PHASE 5 -- causal check through the UNCHANGED Stage 2, for
                # EVERY case that reached N_adv>=CEILING (not just the best).
                check = lib.stage2_causal_check(r.perturbed_matrix, is_poison, n_adv=CEILING)
                row.update(
                    {
                        "post_removed_poison": check["removed_poison"],
                        "post_removed_clean": check["removed_clean"],
                        "post_residual_poison": check["residual_poison"],
                        "post_stage2_label": check["label"],
                        "original_n_adv": stage1.n_adv_estimated,
                        "original_residual_poison": case["historical_residual_poison"],
                    }
                )
            else:
                row.update(
                    {
                        "max_abs_off_diag_change": None,
                        "mean_abs_off_diag_change": None,
                        "frobenius_norm_diff": None,
                        "fraction_off_diag_changed": None,
                        "post_removed_poison": None,
                        "post_removed_clean": None,
                        "post_residual_poison": None,
                        "post_stage2_label": None,
                        "original_n_adv": None,
                        "original_residual_poison": None,
                    }
                )
            rows.append(row)

        if best is not None:
            path_csv = paths_dir / f"{qid}_nadv_vs_alpha.csv"
            with open(path_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["alpha", "n_adv"])
                for alpha, n_adv in best.n_adv_path:
                    writer.writerow([alpha, n_adv])

    return rows, per_query_best


# ---------------------------------------------------------------------------
# CSV writing helper
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
    out_boundary_csv = OUTPUT_DIR / "regime_b_boundary_per_query.csv"
    out_margins_csv = OUTPUT_DIR / "regime_b_passage_margins.csv"
    out_statistic_csv = OUTPUT_DIR / "regime_b_statistic_oracle.csv"
    out_matrix_csv = OUTPUT_DIR / "regime_b_matrix_oracle.csv"
    out_report_md = OUTPUT_DIR / "REGIME_B_STAGE1_ORACLE_REPORT.md"
    paths_dir = OUTPUT_DIR / "paths"

    _check_no_overwrite([out_boundary_csv, out_margins_csv, out_statistic_csv, out_matrix_csv, out_report_md])

    cases = load_regime_b_cases()
    failures = [c for c in cases if not c["historical_success"]]
    successes = [c for c in cases if c["historical_success"]]
    print(f"Loaded {len(cases)} Regime-B cases: {len(successes)} successes, {len(failures)} failures.")

    boundary_rows, margin_rows = build_boundary_and_margin_rows(cases)
    phase2 = build_phase2_comparison(cases)
    statistic_rows, statistic_best = build_statistic_oracle_rows(failures)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths_dir.mkdir(parents=True, exist_ok=True)
    matrix_rows, matrix_best = build_matrix_oracle_rows(failures, paths_dir)

    _write_csv(out_boundary_csv, boundary_rows)
    _write_csv(out_margins_csv, margin_rows)
    _write_csv(out_statistic_csv, statistic_rows)
    _write_csv(out_matrix_csv, matrix_rows)

    return {
        "cases": cases,
        "failures": failures,
        "successes": successes,
        "boundary_rows": boundary_rows,
        "margin_rows": margin_rows,
        "phase2": phase2,
        "statistic_rows": statistic_rows,
        "statistic_best": statistic_best,
        "matrix_rows": matrix_rows,
        "matrix_best": matrix_best,
        "output_paths": {
            "boundary_csv": out_boundary_csv,
            "margins_csv": out_margins_csv,
            "statistic_csv": out_statistic_csv,
            "matrix_csv": out_matrix_csv,
            "report_md": out_report_md,
            "paths_dir": paths_dir,
        },
    }


if __name__ == "__main__":
    result = run()
    print(f"Wrote {len(result['boundary_rows'])} boundary rows, {len(result['margin_rows'])} margin rows, "
          f"{len(result['statistic_rows'])} statistic-oracle rows, {len(result['matrix_rows'])} matrix-oracle rows.")
