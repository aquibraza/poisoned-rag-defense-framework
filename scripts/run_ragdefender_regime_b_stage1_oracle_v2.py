"""REGIME-B STAGE-1 BOUNDARY-SENSITIVITY ORACLE -- V2 CORRECTION PASS.

Corrects a false-negative bug in the V1 Phase-4 alpha search (endpoint-only
reachability check -- see `ragdefender_regime_b_stage1_oracle_lib
._monotonic_or_grid_search` docstring), reruns ALL Phase-4 matrix-space
candidates with the corrected search (BOTH `boost` and `decrease` for
EVERY candidate, not "decrease only if boost fails"), adds PSD/Gram-matrix
validity diagnostics, recomputes Phase 5 from the corrected winners, and
explicitly verifies the mutual-median-match mechanism for all 11 real
MEDIAN-LIMITED failures.

Phase 1-3 methodology and results are NOT recomputed differently here --
this script re-loads the exact same 19 frozen queries via
`run_ragdefender_regime_b_stage1_oracle.load_regime_b_cases` and re-verifies
(does not re-derive) that Phase 1-3 numbers are unchanged, per the task's
STOP conditions.

Uses ONLY the 19 already-saved Regime-B Stella similarity matrices from
`results/diagnostics/ragdefender_expanded_baseline/` (read-only). No
retrieval, no Stella re-encoding, no text mutation, no generation, no
E1/CORAL/MMD, no LLM/API calls, no modification of `ragdefender_paper`.

Writes ONLY new V2 files under
`results/diagnostics/ragdefender_regime_b_stage1_oracle/` -- never
overwrites the V1 CSVs/report (enforced by an explicit no-overwrite guard
in addition to using distinct `_v2` filenames).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from defense import ragdefender_internals as ri  # noqa: E402
import ragdefender_regime_b_stage1_oracle_lib as lib  # noqa: E402
import run_ragdefender_regime_b_stage1_oracle as v1_driver  # noqa: E402

OUTPUT_DIR = v1_driver.OUTPUT_DIR
BASELINE_DIR = v1_driver.BASELINE_DIR
CEILING = v1_driver.CEILING

V1_PROTECTED_FILES = [
    OUTPUT_DIR / "regime_b_matrix_oracle.csv",
    OUTPUT_DIR / "REGIME_B_STAGE1_ORACLE_REPORT.md",
    OUTPUT_DIR / "regime_b_boundary_per_query.csv",
    OUTPUT_DIR / "regime_b_passage_margins.csv",
    OUTPUT_DIR / "regime_b_statistic_oracle.csv",
]


class RegimeBOracleV2StopCondition(RuntimeError):
    """Raised for any of the task's documented V2 STOP conditions."""


def _check_v2_outputs_do_not_overwrite(new_paths: List[Path]) -> None:
    existing = [p for p in new_paths if p.exists()]
    if existing:
        raise RegimeBOracleV2StopCondition(f"Refusing to overwrite existing V2 outputs: {existing}")
    for protected in V1_PROTECTED_FILES:
        if protected in new_paths:
            raise RegimeBOracleV2StopCondition(f"V2 output path collides with a protected V1 file: {protected}")


def _verify_v1_files_untouched_hashes() -> dict:
    import hashlib

    hashes = {}
    for path in V1_PROTECTED_FILES:
        if path.exists():
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


# ---------------------------------------------------------------------------
# Phase 1-3 re-verification (must be unchanged -- STOP condition otherwise)
# ---------------------------------------------------------------------------

def reverify_phase1_3(cases: List[dict]) -> dict:
    if len(cases) != 19:
        raise RegimeBOracleV2StopCondition(f"Expected 19 Regime-B cases, found {len(cases)}.")
    successes = [c for c in cases if c["historical_success"]]
    failures = [c for c in cases if not c["historical_success"]]
    if not (len(successes) == 5 and len(failures) == 14):
        raise RegimeBOracleV2StopCondition(
            f"Expected 5 successes/14 failures, found {len(successes)}/{len(failures)}."
        )

    n_median_limited = 0
    n_mean_gated = 0
    n_both_limited = 0
    for case in failures:
        counts = lib.query_level_counts(case["stage1"])
        binding = lib.classify_binding_condition(counts["n_above_median"], counts["n_and"], CEILING)
        case["binding_classification"] = binding
        if binding == lib.BINDING_MEDIAN_LIMITED:
            n_median_limited += 1
        elif binding == lib.BINDING_MEAN_GATED:
            n_mean_gated += 1
        elif binding == lib.BINDING_BOTH_LIMITED:
            n_both_limited += 1

    if not (n_median_limited == 11 and n_mean_gated == 3 and n_both_limited == 0):
        raise RegimeBOracleV2StopCondition(
            f"Phase 1A decomposition changed: median-limited={n_median_limited}, "
            f"mean-gated={n_mean_gated}, both-limited={n_both_limited} "
            "(expected 11/3/0). This is a documented STOP condition."
        )

    for case in cases:
        if case["stage1"].n_adv_estimated != case["historical_n_adv"]:
            raise RegimeBOracleV2StopCondition(
                f"{case['query_id']}: Phase-1 N_adv no longer reproduces historical baseline."
            )

    return {
        "n": len(cases),
        "n_success": len(successes),
        "n_failure": len(failures),
        "n_median_limited": n_median_limited,
        "n_mean_gated": n_mean_gated,
        "n_both_limited": n_both_limited,
    }


# ---------------------------------------------------------------------------
# STEP 5 -- mutual-median mechanism validation (all 11 real median-limited failures)
# ---------------------------------------------------------------------------

def build_mutual_median_rows(failures: List[dict]) -> List[dict]:
    rows = []
    for case in failures:
        if case.get("binding_classification") != lib.BINDING_MEDIAN_LIMITED:
            continue
        result = lib.mutual_median_validation_for_query(case["matrix"], case["stage1"].s_median)
        rows.append(
            {
                "query_id": case["query_id"],
                "rank5_idx": result["rank5_idx"],
                "rank6_idx": result["rank6_idx"],
                "is_tied": result["is_tied"],
                "tied_value": result["tied_value"],
                "tied_indices": "|".join(map(str, result["tied_indices"])),
                "providers": ";".join(f"{i}:{sorted(v)}" for i, v in result["providers"].items()),
                "mutual_median_match": result["mutual_median_match"],
                "mutual_pairs": ";".join(f"{p[0]}-{p[1]}" for p in result["mutual_pairs"]),
                "shared_matrix_entry": result["shared_matrix_entry"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# STEP 2/2A -- corrected Phase-4 rerun (both modes, all candidates)
# ---------------------------------------------------------------------------

def build_matrix_oracle_v2_rows(failures: List[dict], paths_v2_dir: Path):
    """Reruns Phase 4 with the corrected search for EVERY non-AND candidate
    in EVERY failure, ALWAYS trying both `boost` and `decrease` (V1's
    early-exit-on-boost-success shortcut is removed here)."""
    rows = []
    per_query_results = {}
    for case in failures:
        qid = case["query_id"]
        matrix = case["matrix"]
        is_poison = case["is_poison"]
        stage1 = case["stage1"]
        non_and = [i for i in range(len(stage1.s_mean)) if not bool(stage1.adv_flag[i])]

        results = lib.matrix_oracle_for_query(matrix, non_and, target_n_adv=CEILING, alpha_max=2.0)
        per_query_results[qid] = results

        unconstrained_winner = lib.select_matrix_winner(results, matrix, require_psd=False)
        psd_winner_1e8 = lib.select_matrix_winner(results, matrix, require_psd=True, psd_tol="1e8")

        for r in results:
            row = {
                "query_id": qid,
                "candidate_index": r.candidate_index,
                "mode": r.mode,
                "alpha": r.alpha,
                "reachable": r.reachable,
                "is_monotonic": r.is_monotonic,
                "endpoint_successful": r.endpoint_successful,
                "n_false_to_true_transitions": r.n_false_to_true_transitions,
                "n_true_to_false_transitions": r.n_true_to_false_transitions,
                "n_success_windows": r.n_success_windows,
                "achieved_n_adv": r.achieved_n_adv,
                "verified": r.verified,
                "is_poison_DIAGNOSTIC_ONLY": bool(is_poison[r.candidate_index]),
                "is_unconstrained_winner": (
                    unconstrained_winner is not None
                    and r.candidate_index == unconstrained_winner.candidate_index
                    and r.mode == unconstrained_winner.mode
                ),
                "is_psd_winner_1e8": (
                    psd_winner_1e8 is not None
                    and r.candidate_index == psd_winner_1e8.candidate_index
                    and r.mode == psd_winner_1e8.mode
                ),
            }
            if r.gram is not None:
                row.update(
                    {
                        "min_eigenvalue": r.gram["min_eigenvalue"],
                        "n_negative_eigenvalues": r.gram["n_negative_eigenvalues"],
                        "psd_valid_tol_1e8": r.gram["psd_valid_tol_1e8"],
                        "psd_valid_tol_1e6": r.gram["psd_valid_tol_1e6"],
                        "is_symmetric": r.gram["is_symmetric"],
                        "diag_near_one": r.gram["diag_near_one"],
                    }
                )
            else:
                row.update(
                    {
                        "min_eigenvalue": None,
                        "n_negative_eigenvalues": None,
                        "psd_valid_tol_1e8": None,
                        "psd_valid_tol_1e6": None,
                        "is_symmetric": None,
                        "diag_near_one": None,
                    }
                )
            if r.perturbed_matrix is not None:
                effect = lib.perturbation_effect_size(matrix, r.perturbed_matrix)
                row["frobenius_norm_diff"] = effect["frobenius_norm_diff"]
                row["max_abs_off_diag_change"] = effect["max_abs_off_diag_change"]
            else:
                row["frobenius_norm_diff"] = None
                row["max_abs_off_diag_change"] = None
            rows.append(row)

        # Optional per-candidate N_adv(alpha) path files -- only for the
        # two winners (unconstrained + PSD-valid) to keep file count sane.
        for winner, tag in ((unconstrained_winner, "unconstrained"), (psd_winner_1e8, "psd1e8")):
            if winner is None:
                continue
            path_csv = paths_v2_dir / f"{qid}_{winner.candidate_index}_{winner.mode}_{tag}_nadv_vs_alpha.csv"
            if not path_csv.exists():
                with open(path_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["alpha", "n_adv"])
                    for alpha, n_adv in winner.n_adv_path:
                        writer.writerow([alpha, n_adv])

    return rows, per_query_results


def build_winners_v2_rows(failures: List[dict], per_query_results: dict):
    rows = []
    for case in failures:
        qid = case["query_id"]
        matrix = case["matrix"]
        is_poison = case["is_poison"]
        results = per_query_results[qid]

        unconstrained = lib.select_matrix_winner(results, matrix, require_psd=False)
        psd_1e8 = lib.select_matrix_winner(results, matrix, require_psd=True, psd_tol="1e8")
        psd_1e6 = lib.select_matrix_winner(results, matrix, require_psd=True, psd_tol="1e6")

        def _describe(winner: Optional["lib.MatrixOracleResult"], label: str) -> dict:
            if winner is None:
                return {
                    f"{label}_candidate_index": None,
                    f"{label}_mode": None,
                    f"{label}_alpha": None,
                    f"{label}_is_poison_DIAGNOSTIC_ONLY": None,
                    f"{label}_min_eigenvalue": None,
                }
            return {
                f"{label}_candidate_index": winner.candidate_index,
                f"{label}_mode": winner.mode,
                f"{label}_alpha": winner.alpha,
                f"{label}_is_poison_DIAGNOSTIC_ONLY": bool(is_poison[winner.candidate_index]),
                f"{label}_min_eigenvalue": winner.gram["min_eigenvalue"] if winner.gram else None,
            }

        row = {"query_id": qid}
        row.update(_describe(unconstrained, "unconstrained_winner"))
        row.update(_describe(psd_1e8, "psd_valid_1e8_winner"))
        row.update(_describe(psd_1e6, "psd_valid_1e6_winner"))
        row["winner_changed_when_psd_required_1e8"] = (
            unconstrained is not None
            and psd_1e8 is not None
            and (unconstrained.candidate_index, unconstrained.mode) != (psd_1e8.candidate_index, psd_1e8.mode)
        ) or (unconstrained is not None and psd_1e8 is None)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# STEP 8 -- Phase 5 recompute from corrected winners
# ---------------------------------------------------------------------------

def build_phase5_v2_rows(failures: List[dict], per_query_results: dict):
    rows = []
    for case in failures:
        qid = case["query_id"]
        matrix = case["matrix"]
        is_poison = case["is_poison"]
        results = per_query_results[qid]

        unconstrained = lib.select_matrix_winner(results, matrix, require_psd=False)
        psd_1e8 = lib.select_matrix_winner(results, matrix, require_psd=True, psd_tol="1e8")

        for winner, tag in ((unconstrained, "unconstrained"), (psd_1e8, "psd_valid_1e8")):
            if winner is None:
                rows.append({"query_id": qid, "winner_type": tag, "available": False})
                continue
            check = lib.stage2_causal_check(winner.perturbed_matrix, is_poison, n_adv=CEILING)
            rows.append(
                {
                    "query_id": qid,
                    "winner_type": tag,
                    "available": True,
                    "candidate_index": winner.candidate_index,
                    "mode": winner.mode,
                    "alpha": winner.alpha,
                    "removed_indices": "|".join(map(str, check["removed_indices"])),
                    "removed_poison": check["removed_poison"],
                    "removed_clean": check["removed_clean"],
                    "residual_poison": check["residual_poison"],
                    "residual_clean": check["residual_clean"],
                    "n_pairs": check["n_pairs"],
                    "pp_count": check["pp_count"],
                    "pc_count": check["pc_count"],
                    "cc_count": check["cc_count"],
                    "frequency_scores": "|".join(f"{x:.6f}" for x in check["frequency_scores"]),
                    "label": check["label"],
                }
            )
    return rows


# ---------------------------------------------------------------------------
# CSV writer helper
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
    out_matrix_v2 = OUTPUT_DIR / "regime_b_matrix_oracle_v2.csv"
    out_winners_v2 = OUTPUT_DIR / "regime_b_matrix_winners_v2.csv"
    out_mutual_median = OUTPUT_DIR / "regime_b_mutual_median_validation.csv"
    out_phase5_v2 = OUTPUT_DIR / "regime_b_phase5_psd_comparison_v2.csv"
    out_report_v2 = OUTPUT_DIR / "REGIME_B_STAGE1_ORACLE_V2_REPORT.md"
    paths_v2_dir = OUTPUT_DIR / "paths_v2"

    new_outputs = [out_matrix_v2, out_winners_v2, out_mutual_median, out_phase5_v2, out_report_v2]
    _check_v2_outputs_do_not_overwrite(new_outputs)
    v1_hashes_before = _verify_v1_files_untouched_hashes()

    cases = v1_driver.load_regime_b_cases()
    phase1_3 = reverify_phase1_3(cases)
    failures = [c for c in cases if not c["historical_success"]]
    print(f"Re-verified Phase 1-3: {phase1_3}")

    paths_v2_dir.mkdir(parents=True, exist_ok=True)

    mutual_median_rows = build_mutual_median_rows(failures)
    matrix_v2_rows, per_query_results = build_matrix_oracle_v2_rows(failures, paths_v2_dir)
    winners_v2_rows = build_winners_v2_rows(failures, per_query_results)
    phase5_v2_rows = build_phase5_v2_rows(failures, per_query_results)

    _write_csv(out_matrix_v2, matrix_v2_rows)
    _write_csv(out_winners_v2, winners_v2_rows)
    _write_csv(out_mutual_median, mutual_median_rows)
    _write_csv(out_phase5_v2, phase5_v2_rows)

    v1_hashes_after = _verify_v1_files_untouched_hashes()
    if v1_hashes_before != v1_hashes_after:
        raise RegimeBOracleV2StopCondition("V1 protected file hashes changed during this run -- ABORTING.")

    return {
        "phase1_3": phase1_3,
        "failures": failures,
        "cases": cases,
        "matrix_v2_rows": matrix_v2_rows,
        "winners_v2_rows": winners_v2_rows,
        "mutual_median_rows": mutual_median_rows,
        "phase5_v2_rows": phase5_v2_rows,
        "per_query_results": per_query_results,
        "output_paths": {
            "matrix_v2": out_matrix_v2,
            "winners_v2": out_winners_v2,
            "mutual_median": out_mutual_median,
            "phase5_v2": out_phase5_v2,
            "report_v2": out_report_v2,
            "paths_v2_dir": paths_v2_dir,
        },
    }


if __name__ == "__main__":
    result = run()
    print(
        f"Wrote {len(result['matrix_v2_rows'])} matrix-v2 rows, "
        f"{len(result['winners_v2_rows'])} winner rows, "
        f"{len(result['mutual_median_rows'])} mutual-median rows, "
        f"{len(result['phase5_v2_rows'])} phase5-v2 rows."
    )
