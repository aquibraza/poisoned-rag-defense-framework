"""REGIME-C STAGE-2 IDENTIFICATION-CAPACITY STUDY.

Second mechanism-level stress-testing study (Regime B is closed -- see
`results/diagnostics/ragdefender_regime_b_stage1_oracle/`). This is a
pure OFFLINE Stage-2 diagnostic: the true poison count `M` is supplied and
held FIXED throughout (Stage 1 / `concentration_stage1_paper` /
`adv_flag` are never consulted anywhere in this module). The question
under study is why `stage2_pair_frequency(matrix, n_adv=M, p=2)` fails to
achieve `removed_poison=M, removed_clean=0` on 16/20 above-ceiling
Regime-C queries even though the correct count is already supplied.

Uses ONLY the already-frozen 20-query Regime-C population from
`results/diagnostics/ragdefender_expanded_baseline/` (similarity matrices,
`is_poison` labels) and cross-verifies against
`results/diagnostics/ragdefender_expanded_gate_c/` (independently computed
oracle-count outcomes). No retrieval, no Stella re-encoding, no text
mutation, no generation, no E1/CORAL/MMD, no LLM/API call. Neither
historical directory is ever written to.
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
import ragdefender_regime_c_stage2_lib as lib  # noqa: E402

BASELINE_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_baseline"
GATE_C_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_gate_c"
OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_regime_c_stage2"
CEILING = 5  # floor(k/2) for k=10 -- Regime C is defined as M > CEILING


class RegimeCStageStopCondition(RuntimeError):
    """Raised for any of the task's documented STOP conditions."""


def _check_outputs_do_not_overwrite(paths: List[Path]) -> None:
    existing = [p for p in paths if p.exists()]
    if existing:
        raise RegimeCStageStopCondition(f"Refusing to overwrite existing outputs: {existing}")


# ---------------------------------------------------------------------------
# PHASE 0 -- population freeze / reproduction (STOP-condition checks)
# ---------------------------------------------------------------------------

def load_regime_c_cases() -> List[dict]:
    per_query_csv = BASELINE_DIR / "expanded_baseline_per_query.csv"
    if not per_query_csv.exists():
        raise RegimeCStageStopCondition(f"{per_query_csv} not found.")
    with open(per_query_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["regime"] == "C_ABOVE_CEILING"]

    if len(rows) != 20:
        raise RegimeCStageStopCondition(f"Expected exactly 20 Regime-C queries, found {len(rows)}.")

    gate_c_csv = GATE_C_DIR / "expanded_gate_c_per_query.csv"
    if not gate_c_csv.exists():
        raise RegimeCStageStopCondition(f"{gate_c_csv} not found.")
    with open(gate_c_csv) as f:
        gate_c_by_id = {r["query_id"]: r for r in csv.DictReader(f) if r["regime"] == "C_ABOVE_CEILING"}
    if len(gate_c_by_id) != 20:
        raise RegimeCStageStopCondition(f"Expected 20 Gate-C Regime-C rows, found {len(gate_c_by_id)}.")

    with open(BASELINE_DIR / "recovered_contexts.json") as f:
        contexts_by_id = {c["query_id"]: c for c in json.load(f)}

    cases = []
    n_success = 0
    n_failure = 0
    for row in rows:
        qid = row["query_id"]
        k = int(row["k"])
        m_poison = int(row["m_poison"])
        c_clean = int(row["c_clean"])
        if k != 10:
            raise RegimeCStageStopCondition(f"{qid}: expected k=10, found k={k}.")
        if not (m_poison > CEILING):
            raise RegimeCStageStopCondition(f"{qid}: expected M>{CEILING}, found M={m_poison}.")
        if not (c_clean >= 1):
            raise RegimeCStageStopCondition(f"{qid}: expected C>=1, found C={c_clean}.")
        if m_poison + c_clean != k:
            raise RegimeCStageStopCondition(f"{qid}: M+C != k ({m_poison}+{c_clean} != {k}).")

        gate_row = gate_c_by_id.get(qid)
        if gate_row is None:
            raise RegimeCStageStopCondition(f"{qid}: not found in expanded_gate_c_per_query.csv.")
        if int(gate_row["m_poison"]) != m_poison:
            raise RegimeCStageStopCondition(f"{qid}: m_poison mismatch between baseline and Gate-C.")

        ctx = contexts_by_id.get(qid)
        if ctx is None:
            raise RegimeCStageStopCondition(f"{qid}: not found in recovered_contexts.json.")
        matrix_path = BASELINE_DIR / "similarity" / f"{qid}_stella_similarity_matrix.npy"
        if not matrix_path.exists():
            raise RegimeCStageStopCondition(f"{qid}: historical matrix missing at {matrix_path}.")
        matrix = np.load(matrix_path)  # read-only load
        is_poison = np.array(ctx["is_poison"], dtype=bool)
        if int(is_poison.sum()) != m_poison:
            raise RegimeCStageStopCondition(
                f"{qid}: is_poison label count ({int(is_poison.sum())}) != m_poison ({m_poison})."
            )

        # Recompute the TRUE-COUNT Stage-2 outcome (n_adv=M, holding count
        # fixed) and verify it reproduces Gate-C's independently-recorded
        # oracle-count result exactly -- this is the population-level STOP
        # check.
        top_pairs, all_pairs, n_pairs = lib.stage2_original_top_pairs(matrix, m_poison)
        selection = lib.compute_frequency_and_selection(top_pairs, k, m_poison)
        outcome = lib.removal_outcome(selection, is_poison, m_poison)
        comp = lib.ptop_composition(top_pairs, all_pairs, n_pairs, is_poison, m_poison, c_clean)

        expected = {
            "n_pairs": int(gate_row["oracle_N_pairs"]),
            "n_PP_selected": int(gate_row["oracle_pp_count"]),
            "n_PC_selected": int(gate_row["oracle_pc_count"]),
            "n_CC_selected": int(gate_row["oracle_cc_count"]),
        }
        for key, expected_val in expected.items():
            got = comp[key] if key != "n_pairs" else n_pairs
            if got != expected_val:
                raise RegimeCStageStopCondition(
                    f"{qid}: recomputed {key}={got} != Gate-C {key}={expected_val}."
                )
        for key, gate_key in (
            ("removed_poison", "oracle_removed_poison"),
            ("removed_clean", "oracle_removed_clean"),
            ("residual_poison", "oracle_residual_poison"),
        ):
            if outcome[key] != int(gate_row[gate_key]):
                raise RegimeCStageStopCondition(
                    f"{qid}: recomputed {key}={outcome[key]} != Gate-C {gate_key}={gate_row[gate_key]}."
                )

        historical_label = gate_row["decomposition_label"]
        recomputed_success = outcome["success"]
        if historical_label == "A. COUNT-LIMITED" and not recomputed_success:
            raise RegimeCStageStopCondition(f"{qid}: Gate-C labeled A (success) but recomputation shows failure.")
        if historical_label == "B. COUNT + IDENTIFICATION LIMITED" and recomputed_success:
            raise RegimeCStageStopCondition(f"{qid}: Gate-C labeled B (failure) but recomputation shows success.")

        if recomputed_success:
            n_success += 1
        else:
            n_failure += 1

        cases.append(
            {
                "query_id": qid,
                "matrix": matrix,
                "is_poison": is_poison,
                "m_poison": m_poison,
                "c_clean": c_clean,
                "k": k,
                "top_pairs": top_pairs,
                "all_pairs": all_pairs,
                "n_pairs": n_pairs,
                "selection": selection,
                "outcome": outcome,
                "composition": comp,
                "historical_label": historical_label,
                "success": recomputed_success,
            }
        )

    if not (n_success == 4 and n_failure == 16):
        raise RegimeCStageStopCondition(
            f"Expected 4 successes / 16 failures at true-count Stage 2, found {n_success}/{n_failure}."
        )

    return cases


# ---------------------------------------------------------------------------
# PHASE 1/1A/1B -- per-query composition + boundary rows, PHASE 2/2A -- passage rows
# ---------------------------------------------------------------------------

def build_per_query_rows(cases: List[dict]) -> List[dict]:
    rows = []
    for case in cases:
        qid = case["query_id"]
        M, C, k = case["m_poison"], case["c_clean"], case["k"]
        comp = case["composition"]
        outcome = case["outcome"]
        selection = case["selection"]
        extended = lib.extended_ranking_with_zero_degree(selection, k)

        f_rank_m = extended[M - 1][1]
        f_rank_m1 = extended[M][1] if M < k else None
        removal_margin = (f_rank_m - f_rank_m1) if f_rank_m1 is not None else None
        rank_m_idx = extended[M - 1][0]
        rank_m1_idx = extended[M][0] if M < k else None
        rank_m_label = "poison" if case["is_poison"][rank_m_idx] else "clean"
        rank_m1_label = ("poison" if case["is_poison"][rank_m1_idx] else "clean") if rank_m1_idx is not None else None
        boundary_label = lib.boundary_classification(rank_m_label, rank_m1_label) if rank_m1_label else None

        clean_scores = [selection.frequency_scores[i] for i in range(k) if not case["is_poison"][i]]
        poison_scores = [selection.frequency_scores[i] for i in range(k) if case["is_poison"][i]]
        max_clean = max(clean_scores)
        min_poison = min(poison_scores)
        score_overlap = max_clean - min_poison

        rows.append(
            {
                "query_id": qid,
                "k": k,
                "m_poison": M,
                "c_clean": C,
                "pair_budget_fraction": case["n_pairs"] / lib.n_choose_2(k),
                **comp,
                "removed_poison": outcome["removed_poison"],
                "removed_clean": outcome["removed_clean"],
                "residual_poison": outcome["residual_poison"],
                "success_true_count": outcome["success"],
                "historical_decomposition_label": case["historical_label"],
                "f_rank_M": f_rank_m,
                "f_rank_M_plus_1": f_rank_m1,
                "removal_margin": removal_margin,
                "rank_M_index": rank_m_idx,
                "rank_M_label_DIAGNOSTIC_ONLY": rank_m_label,
                "rank_M_plus_1_index": rank_m1_idx,
                "rank_M_plus_1_label_DIAGNOSTIC_ONLY": rank_m1_label,
                "boundary_classification": boundary_label,
                "max_clean_frequency_score": max_clean,
                "min_poison_frequency_score": min_poison,
                "score_overlap": score_overlap,
                "score_overlap_exists": bool(score_overlap >= 0),
            }
        )
    return rows


def build_pair_rows(cases: List[dict]) -> List[dict]:
    rows = []
    for case in cases:
        qid = case["query_id"]
        is_poison = case["is_poison"]
        n_pairs = case["n_pairs"]
        selected_set = {(i, j) for i, j, _ in case["top_pairs"]}
        for rank, (i, j, sim) in enumerate(case["all_pairs"], start=1):
            cls = lib.classify_pair(i, j, is_poison)
            rows.append(
                {
                    "query_id": qid,
                    "pair_rank": rank,
                    "i": i,
                    "j": j,
                    "similarity": sim,
                    "abs_similarity": abs(sim),
                    "signed_score_contribution": lib.pair_signed_score(sim),
                    "pair_class": cls,
                    "selected": (i, j) in selected_set,
                    "n_pairs_budget": n_pairs,
                }
            )
    return rows


def build_passage_score_rows(cases: List[dict]) -> List[dict]:
    rows = []
    for case in cases:
        qid = case["query_id"]
        k = case["k"]
        is_poison = case["is_poison"]
        selection = case["selection"]
        detail = lib.passage_incident_detail(case["top_pairs"], is_poison, k)
        removed_set = set(selection.selected_indices)
        # Rank by frequency score using the SAME extended ranking (ties
        # broken exactly as production, plus zero-degree passages appended
        # by ascending index -- see `extended_ranking_with_zero_degree`).
        extended = lib.extended_ranking_with_zero_degree(selection, k)
        rank_by_index = {idx: rank for rank, (idx, _score) in enumerate(extended, start=1)}
        for i in range(k):
            d = detail[i]
            rows.append(
                {
                    "query_id": qid,
                    "index": i,
                    "is_poison_DIAGNOSTIC_ONLY": bool(is_poison[i]),
                    "frequency_score": float(selection.frequency_scores[i]),
                    "rank_by_frequency_score": rank_by_index[i],
                    "n_PP_incident": d["n_PP_incident"],
                    "n_PC_incident": d["n_PC_incident"],
                    "n_CC_incident": d["n_CC_incident"],
                    "signed_PP_contribution": d["signed_PP_contribution"],
                    "signed_PC_contribution": d["signed_PC_contribution"],
                    "signed_CC_contribution": d["signed_CC_contribution"],
                    "total_pair_degree": d["degree"],
                    "removed_in_top_M": i in removed_set,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# PHASE 2B -- displacement mapping (failed queries only)
# ---------------------------------------------------------------------------

def build_displacement_rows(cases: List[dict]) -> List[dict]:
    rows = []
    for case in cases:
        if case["success"]:
            continue
        qid = case["query_id"]
        k = case["k"]
        is_poison = case["is_poison"]
        selection = case["selection"]
        outcome = case["outcome"]
        detail = lib.passage_incident_detail(case["top_pairs"], is_poison, k)
        removed_set = set(selection.selected_indices)

        removed_clean_indices = [i for i in removed_set if not is_poison[i]]
        residual_poison_indices = [i for i in range(k) if is_poison[i] and i not in removed_set]

        for c_idx in removed_clean_indices:
            for p_idx in residual_poison_indices:
                f_clean = float(selection.frequency_scores[c_idx])
                f_poison = float(selection.frequency_scores[p_idx])
                clean_incident = [pr for pr in case["top_pairs"] if c_idx in (pr[0], pr[1])]
                poison_incident = [pr for pr in case["top_pairs"] if p_idx in (pr[0], pr[1])]
                largest_clean_pair = max(clean_incident, key=lambda pr: pr[2]) if clean_incident else None
                largest_poison_pair = max(poison_incident, key=lambda pr: pr[2]) if poison_incident else None
                rows.append(
                    {
                        "query_id": qid,
                        "removed_clean_index": c_idx,
                        "residual_poison_index": p_idx,
                        "f_clean": f_clean,
                        "f_poison": f_poison,
                        "score_difference_clean_minus_poison": f_clean - f_poison,
                        "clean_n_PP_incident": detail[c_idx]["n_PP_incident"],
                        "clean_n_PC_incident": detail[c_idx]["n_PC_incident"],
                        "clean_n_CC_incident": detail[c_idx]["n_CC_incident"],
                        "poison_n_PP_incident": detail[p_idx]["n_PP_incident"],
                        "poison_n_PC_incident": detail[p_idx]["n_PC_incident"],
                        "poison_n_CC_incident": detail[p_idx]["n_CC_incident"],
                        "clean_largest_pair": f"{largest_clean_pair[0]}-{largest_clean_pair[1]}:{largest_clean_pair[2]:.4f}"
                        if largest_clean_pair
                        else None,
                        "poison_largest_pair": f"{largest_poison_pair[0]}-{largest_poison_pair[1]}:{largest_poison_pair[2]:.4f}"
                        if largest_poison_pair
                        else None,
                        "clean_signed_positive_total": sum(
                            v for k_ in ("signed_PP_contribution", "signed_PC_contribution", "signed_CC_contribution")
                            for v in [detail[c_idx][k_]] if v > 0
                        ),
                        "clean_signed_negative_total": sum(
                            v for k_ in ("signed_PP_contribution", "signed_PC_contribution", "signed_CC_contribution")
                            for v in [detail[c_idx][k_]] if v < 0
                        ),
                        "poison_signed_positive_total": sum(
                            v for k_ in ("signed_PP_contribution", "signed_PC_contribution", "signed_CC_contribution")
                            for v in [detail[p_idx][k_]] if v > 0
                        ),
                        "poison_signed_negative_total": sum(
                            v for k_ in ("signed_PP_contribution", "signed_PC_contribution", "signed_CC_contribution")
                            for v in [detail[p_idx][k_]] if v < 0
                        ),
                        "epsilon_needed": max(0.0, f_clean - f_poison) + lib.EPS,
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# PHASE 3/3A -- aggregate by M
# ---------------------------------------------------------------------------

def _descriptive(values: List[float]) -> dict:
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"median": None, "iqr": None, "range": None, "n": 0}
    q1, q3 = np.percentile(arr, [25, 75])
    return {"median": float(np.median(arr)), "iqr": float(q3 - q1), "range": float(arr.max() - arr.min()), "n": int(arr.size)}


def build_by_m_rows(per_query_rows: List[dict]) -> List[dict]:
    by_m: dict = {}
    for row in per_query_rows:
        by_m.setdefault(row["m_poison"], []).append(row)

    rows = []
    for m in sorted(by_m):
        group = by_m[m]
        n = len(group)
        n_success = sum(1 for r in group if r["success_true_count"])
        rows.append(
            {
                "m_poison": m,
                "n_pairs_budget": lib.n_choose_2(m),
                "pair_budget_fraction": lib.n_choose_2(m) / lib.n_choose_2(10),
                "n_queries": n,
                "n_success_true_count": n_success,
                "zero_residual_success_rate": n_success / n,
                "mean_removed_poison": float(np.mean([r["removed_poison"] for r in group])),
                "median_removed_poison": float(np.median([r["removed_poison"] for r in group])),
                "mean_removed_clean": float(np.mean([r["removed_clean"] for r in group])),
                "median_removed_clean": float(np.median([r["removed_clean"] for r in group])),
                "mean_residual_poison": float(np.mean([r["residual_poison"] for r in group])),
                "median_residual_poison": float(np.median([r["residual_poison"] for r in group])),
                "mean_n_PP_selected": float(np.mean([r["n_PP_selected"] for r in group])),
                "mean_n_PC_selected": float(np.mean([r["n_PC_selected"] for r in group])),
                "mean_n_CC_selected": float(np.mean([r["n_CC_selected"] for r in group])),
                "pure_pp_ptop_rate": sum(1 for r in group if r["pair_set_pure_pp"]) / n,
                "median_removal_margin": _descriptive([r["removal_margin"] for r in group])["median"],
                "median_pair_cutoff_margin": _descriptive([r["pair_cutoff_margin"] for r in group])["median"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# PHASE 5 -- pure-PP pair-set oracle
# ---------------------------------------------------------------------------

def build_pure_pp_oracle_rows(cases: List[dict]) -> List[dict]:
    rows = []
    for case in cases:
        qid = case["query_id"]
        k, M = case["k"], case["m_poison"]
        is_poison = case["is_poison"]
        pp_pairs = lib.pure_pp_pair_set(case["matrix"], is_poison)
        selection = lib.compute_frequency_and_selection(pp_pairs, k, M)
        outcome = lib.removal_outcome(selection, is_poison, M)
        rows.append(
            {
                "query_id": qid,
                "m_poison": M,
                "was_true_count_failure": not case["success"],
                "n_true_PP_pairs": len(pp_pairs),
                "expected_n_pairs": lib.n_choose_2(M),
                "removed_poison": outcome["removed_poison"],
                "removed_clean": outcome["removed_clean"],
                "residual_poison": outcome["residual_poison"],
                "pure_pp_repairs_failure": bool((not case["success"]) and outcome["success"]),
                "pure_pp_success": outcome["success"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# PHASE 6 -- pair-class ablation (failed queries only)
# ---------------------------------------------------------------------------

def build_ablation_rows(cases: List[dict]) -> List[dict]:
    rows = []
    for case in cases:
        if case["success"]:
            continue
        qid = case["query_id"]
        k, M = case["k"], case["m_poison"]
        is_poison = case["is_poison"]
        variants = lib.ablation_variants(case["top_pairs"], is_poison)
        outcomes = {}
        for name, plist in variants.items():
            selection = lib.compute_frequency_and_selection(plist, k, M)
            outcomes[name] = lib.removal_outcome(selection, is_poison, M)
            outcomes[name]["n_pairs_in_variant"] = len(plist)
            outcomes[name]["max_clean_score"] = (
                max((selection.frequency_scores[i] for i in range(k) if not is_poison[i]), default=None)
            )
            outcomes[name]["min_poison_score"] = (
                min((selection.frequency_scores[i] for i in range(k) if is_poison[i]), default=None)
            )

        driver_label = lib.classify_ablation_driver(
            outcomes["B_remove_CC"]["success"], outcomes["C_remove_PC"]["success"], outcomes["D_pp_only"]["success"]
        )
        row = {"query_id": qid, "m_poison": M, "driver_classification": driver_label}
        for name, o in outcomes.items():
            row.update(
                {
                    f"{name}_n_pairs": o["n_pairs_in_variant"],
                    f"{name}_removed_poison": o["removed_poison"],
                    f"{name}_removed_clean": o["removed_clean"],
                    f"{name}_residual_poison": o["residual_poison"],
                    f"{name}_success": o["success"],
                    f"{name}_max_clean_score": o["max_clean_score"],
                    f"{name}_min_poison_score": o["min_poison_score"],
                }
            )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# PHASE 7/7A -- minimal pair-swap oracle
# ---------------------------------------------------------------------------

def build_pair_swap_rows(cases: List[dict]) -> List[dict]:
    rows = []
    for case in cases:
        if case["success"]:
            continue
        qid = case["query_id"]
        k, M = case["k"], case["m_poison"]
        is_poison = case["is_poison"]
        result = lib.minimal_pair_swap_search(case["top_pairs"], case["matrix"], is_poison, k, M)
        removed_desc = "|".join(f"{i}-{j}:{sim:.4f}" for i, j, sim in result.removed_pairs)
        added_desc = "|".join(f"{i}-{j}:{sim:.4f}" for i, j, sim in result.added_pairs)
        removed_classes = "|".join(lib.classify_pair(i, j, is_poison) for i, j, _ in result.removed_pairs)
        rows.append(
            {
                "query_id": qid,
                "m_poison": M,
                "n_pairs_budget": case["n_pairs"],
                "minimum_pair_swaps": result.swap_count,
                "is_exact": result.is_exact,
                "pair_swap_fraction": (result.swap_count / case["n_pairs"]) if result.swap_count is not None else None,
                "removed_non_pp_pairs": removed_desc,
                "removed_non_pp_pair_classes": removed_classes,
                "added_pp_pairs": added_desc,
                "resulting_removed_poison": result.outcome["removed_poison"] if result.outcome else None,
                "resulting_removed_clean": result.outcome["removed_clean"] if result.outcome else None,
                "resulting_residual_poison": result.outcome["residual_poison"] if result.outcome else None,
                "resulting_success": result.outcome["success"] if result.outcome else None,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# PHASE 8 -- optional score-space epsilon oracle
# ---------------------------------------------------------------------------

def build_score_epsilon_summary(cases: List[dict]) -> List[dict]:
    rows = []
    for case in cases:
        if case["success"]:
            continue
        eps_needed = lib.minimal_score_epsilon(case["selection"].frequency_scores, case["is_poison"], case["m_poison"])
        rows.append({"query_id": case["query_id"], "m_poison": case["m_poison"], "epsilon_needed": eps_needed})
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
    out_per_query = OUTPUT_DIR / "regime_c_per_query.csv"
    out_pairs = OUTPUT_DIR / "regime_c_pairs.csv"
    out_passage_scores = OUTPUT_DIR / "regime_c_passage_scores.csv"
    out_displacements = OUTPUT_DIR / "regime_c_displacements.csv"
    out_by_m = OUTPUT_DIR / "regime_c_by_M.csv"
    out_pure_pp = OUTPUT_DIR / "regime_c_pure_pp_oracle.csv"
    out_ablation = OUTPUT_DIR / "regime_c_pair_class_ablation.csv"
    out_swap = OUTPUT_DIR / "regime_c_pair_swap_oracle.csv"
    out_epsilon = OUTPUT_DIR / "regime_c_score_epsilon_oracle.csv"
    out_report = OUTPUT_DIR / "REGIME_C_STAGE2_REPORT.md"
    paths_dir = OUTPUT_DIR / "paths"

    new_outputs = [
        out_per_query, out_pairs, out_passage_scores, out_displacements, out_by_m,
        out_pure_pp, out_ablation, out_swap, out_epsilon, out_report,
    ]
    _check_outputs_do_not_overwrite(new_outputs)

    cases = load_regime_c_cases()
    print(f"Loaded and verified {len(cases)} Regime-C cases (STOP checks passed).")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths_dir.mkdir(parents=True, exist_ok=True)

    per_query_rows = build_per_query_rows(cases)
    pair_rows = build_pair_rows(cases)
    passage_rows = build_passage_score_rows(cases)
    displacement_rows = build_displacement_rows(cases)
    by_m_rows = build_by_m_rows(per_query_rows)
    pure_pp_rows = build_pure_pp_oracle_rows(cases)
    ablation_rows = build_ablation_rows(cases)
    swap_rows = build_pair_swap_rows(cases)
    epsilon_rows = build_score_epsilon_summary(cases)

    _write_csv(out_per_query, per_query_rows)
    _write_csv(out_pairs, pair_rows)
    _write_csv(out_passage_scores, passage_rows)
    _write_csv(out_displacements, displacement_rows)
    _write_csv(out_by_m, by_m_rows)
    _write_csv(out_pure_pp, pure_pp_rows)
    _write_csv(out_ablation, ablation_rows)
    _write_csv(out_swap, swap_rows)
    _write_csv(out_epsilon, epsilon_rows)

    # Optional per-query path files.
    for case in cases:
        qid = case["query_id"]
        freq_path = paths_dir / f"{qid}_frequency_scores.csv"
        if not freq_path.exists():
            with open(freq_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "is_poison_DIAGNOSTIC_ONLY", "frequency_score"])
                for i in range(case["k"]):
                    writer.writerow([i, bool(case["is_poison"][i]), float(case["selection"].frequency_scores[i])])
        rank_path = paths_dir / f"{qid}_pair_ranking.csv"
        if not rank_path.exists():
            with open(rank_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["pair_rank", "i", "j", "similarity", "pair_class", "selected"])
                selected_set = {(i, j) for i, j, _ in case["top_pairs"]}
                for rank, (i, j, sim) in enumerate(case["all_pairs"], start=1):
                    writer.writerow([rank, i, j, sim, lib.classify_pair(i, j, case["is_poison"]), (i, j) in selected_set])

    return {
        "cases": cases,
        "per_query_rows": per_query_rows,
        "pair_rows": pair_rows,
        "passage_rows": passage_rows,
        "displacement_rows": displacement_rows,
        "by_m_rows": by_m_rows,
        "pure_pp_rows": pure_pp_rows,
        "ablation_rows": ablation_rows,
        "swap_rows": swap_rows,
        "epsilon_rows": epsilon_rows,
        "output_paths": {
            "per_query": out_per_query, "pairs": out_pairs, "passage_scores": out_passage_scores,
            "displacements": out_displacements, "by_m": out_by_m, "pure_pp": out_pure_pp,
            "ablation": out_ablation, "swap": out_swap, "epsilon": out_epsilon, "report": out_report,
            "paths_dir": paths_dir,
        },
    }


if __name__ == "__main__":
    result = run()
    print(
        f"Wrote {len(result['per_query_rows'])} per-query rows, {len(result['pair_rows'])} pair rows, "
        f"{len(result['passage_rows'])} passage rows, {len(result['displacement_rows'])} displacement rows, "
        f"{len(result['ablation_rows'])} ablation rows, {len(result['swap_rows'])} swap rows."
    )
