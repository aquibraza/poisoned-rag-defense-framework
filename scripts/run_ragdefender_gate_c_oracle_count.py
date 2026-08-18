"""Gate C -- ORACLE-COUNT decomposition (Gate-B follow-up STEP 4).

==========================================================================
PURPOSE
==========================================================================
Separate RAGDefender error into:

    1. Stage-1 COUNT ESTIMATION ERROR
    2. Stage-2 IDENTIFICATION ERROR, conditional on a supplied count

Uses ONLY the frozen Stella similarity matrices already saved by Gate B
(`results/diagnostics/ragdefender_gate_b/similarity/*.npy`) and the
already-saved `is_poison`/composition ground truth in
`results/diagnostics/ragdefender_gate_b/gate_b_per_query.csv`. No new
embeddings, no text mutation, no retrieval, no generation, no API calls.

For every one of the 8 Gate-B queries:

    TRUE COUNT:
        N_poison = observed number of poisoned passages in the retrieved
        context (`n_retrieved_poison`, already recorded by Gate B).

    ESTIMATED PIPELINE:
        N_est = Gate-B paper Stage-1 N_adv (recomputed here from the saved
        matrix via the unchanged `concentration_stage1_paper`, and
        cross-checked against Gate B's own saved `n_adv` column).
        Stage2(matrix, N_est).

    ORACLE-COUNT PIPELINE:
        Stage2(matrix, N_poison).

The oracle does NOT alter embeddings, does NOT alter passage text, and
does NOT identify WHICH passages are poison -- it supplies ONLY the
correct NUMBER of poisoned passages to the otherwise-unchanged
paper-faithful `ragdefender_internals.stage2_pair_frequency`. This is a
diagnostic control, not a deployable defense; nothing here is wired into
`defense/defense_runner.py` or `defense/dispatch.py`.

Never writes into `results/diagnostics/ragdefender_gate_a/` or
`results/diagnostics/ragdefender_gate_b/` -- only into
`results/diagnostics/ragdefender_gate_c_oracle_count/`.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from defense import ragdefender_internals  # noqa: E402

GATE_B_DIR = REPO_ROOT / "results/diagnostics/ragdefender_gate_b"
OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_gate_c_oracle_count"

VALID_LABELS = (
    "A. COUNT-LIMITED",
    "B. COUNT + IDENTIFICATION LIMITED",
    "C. IDENTIFICATION LIMITED",
    "D. BASELINE SUCCESS",
)


class GateCStopCondition(RuntimeError):
    """Raised if this script would touch a Gate-A/B artifact, use oracle
    identities instead of only the oracle count, or if the saved Gate-B
    matrices/labels cannot be reproduced exactly."""


def _pipe(values) -> str:
    return "|".join(str(v) for v in values)


def _classify_pair(i: int, j: int, is_poison: np.ndarray) -> str:
    pi, pj = bool(is_poison[i]), bool(is_poison[j])
    if pi and pj:
        return "PP"
    if not pi and not pj:
        return "CC"
    return "PC"


# ---------------------------------------------------------------------------
# Load saved Gate-B inputs (read-only)
# ---------------------------------------------------------------------------

def load_gate_b_cases() -> List[dict]:
    per_query_csv = GATE_B_DIR / "gate_b_per_query.csv"
    if not per_query_csv.exists():
        raise GateCStopCondition(f"Gate-B per-query CSV not found: {per_query_csv}")
    df = pd.read_csv(per_query_csv)

    cases = []
    for _, row in df.iterrows():
        qid = row["query_id"]
        matrix_path = REPO_ROOT / row["similarity_matrix_path"]
        if not matrix_path.exists():
            raise GateCStopCondition(f"{qid}: saved Gate-B similarity matrix missing: {matrix_path}")
        matrix = np.load(matrix_path)
        is_poison = np.array([bool(int(x)) for x in str(row["is_poison_i"]).split("|")])
        if matrix.shape[0] != len(is_poison):
            raise GateCStopCondition(
                f"{qid}: matrix shape {matrix.shape} does not match is_poison length {len(is_poison)}."
            )
        cases.append({
            "query_id": qid,
            "matrix": matrix,
            "is_poison": is_poison,
            "n_retrieved_poison": int(row["n_retrieved_poison"]),
            "n_retrieved_clean": int(row["n_retrieved_clean"]),
            "k": int(row["k"]),
            "gate_b_n_adv": int(row["n_adv"]),
        })
    return cases


# ---------------------------------------------------------------------------
# Core Gate-C computation (also independently unit-testable on synthetic
# matrices, without touching any Gate-B file)
# ---------------------------------------------------------------------------

def _run_stage2_metrics(matrix: np.ndarray, is_poison: np.ndarray, n_adv: int, true_poison_count: int) -> dict:
    stage2 = ragdefender_internals.stage2_pair_frequency(matrix, n_adv=n_adv, p=2.0)
    pair_classes = [_classify_pair(i, j, is_poison) for i, j, _sim in stage2.top_pairs]
    removed_indices = list(stage2.selected_indices)
    removed_poison = int(sum(1 for idx in removed_indices if is_poison[idx]))
    removed_clean = int(sum(1 for idx in removed_indices if not is_poison[idx]))
    residual_poison = true_poison_count - removed_poison

    denom = removed_poison + removed_clean
    removal_precision: Optional[float] = (removed_poison / denom) if denom > 0 else None
    poison_recall: Optional[float] = (removed_poison / true_poison_count) if true_poison_count > 0 else None

    return {
        "n_adv_used": n_adv,
        "n_pairs": stage2.n_pairs,
        "pp_count": pair_classes.count("PP"),
        "pc_count": pair_classes.count("PC"),
        "cc_count": pair_classes.count("CC"),
        "selected_pairs": _pipe(f"{i}-{j}:{sim:.4f}" for i, j, sim in stage2.top_pairs),
        "frequency_scores": _pipe(f"{x:.6f}" for x in stage2.frequency_scores),
        "ranked_passage_order": _pipe(
            idx for idx, _ in sorted(enumerate(stage2.frequency_scores), key=lambda t: t[1], reverse=True)
        ),
        "removed_indices": _pipe(sorted(removed_indices)),
        "removed_poison": removed_poison,
        "removed_clean": removed_clean,
        "residual_poison": residual_poison,
        "removal_precision": removal_precision,
        "poison_recall": poison_recall,
    }


def _classify_decomposition(estimated: dict, oracle: dict) -> str:
    """Priority-ordered decision tree over EXACTLY the four allowed
    labels -- never invents a fifth. See module docstring / Gate C plan
    for the definitions this mirrors verbatim."""
    if estimated["residual_poison"] == 0:
        return "D. BASELINE SUCCESS"
    if estimated["count_error"] == 0:
        # N_est already equals the true poison count, so
        # Stage2(matrix, N_est) == Stage2(matrix, N_poison) by
        # construction -- the oracle pipeline is IDENTICAL to the
        # estimated one here, and by hypothesis (this branch) it still
        # left residual poison. Failure is attributable to Stage 2 alone.
        return "C. IDENTIFICATION LIMITED"
    if oracle["residual_poison"] == 0 and oracle["removed_clean"] == 0:
        # Oracle count removes all poison with no clean-removal cost.
        return "A. COUNT-LIMITED"
    return "B. COUNT + IDENTIFICATION LIMITED"


def run_gate_c_query(case: dict) -> dict:
    matrix = case["matrix"]
    is_poison = case["is_poison"]
    true_poison_count = case["n_retrieved_poison"]
    true_clean_count = case["n_retrieved_clean"]

    stage1 = ragdefender_internals.concentration_stage1_paper(matrix)
    if stage1.n_adv_estimated != case["gate_b_n_adv"]:
        raise GateCStopCondition(
            f"{case['query_id']}: recomputed Stage-1 N_adv ({stage1.n_adv_estimated}) != "
            f"saved Gate-B N_adv ({case['gate_b_n_adv']}) -- saved matrix does not reproduce "
            "Gate B's own recorded result."
        )
    estimated_n_adv = stage1.n_adv_estimated
    count_error = estimated_n_adv - true_poison_count

    estimated = _run_stage2_metrics(matrix, is_poison, n_adv=estimated_n_adv, true_poison_count=true_poison_count)
    estimated["count_error"] = count_error
    estimated["abs_count_error"] = abs(count_error)

    # ORACLE: supplies ONLY the correct COUNT (an integer), never passage
    # identities -- `_run_stage2_metrics` receives `n_adv=true_poison_count`
    # and runs the SAME unchanged `stage2_pair_frequency` over the SAME
    # matrix; it has no access to `is_poison` except (like the estimated
    # pipeline) for scoring the result afterwards.
    oracle = _run_stage2_metrics(matrix, is_poison, n_adv=true_poison_count, true_poison_count=true_poison_count)

    residual_poison_reduction = estimated["residual_poison"] - oracle["residual_poison"]
    oracle_count_fixes_failure = estimated["residual_poison"] > 0 and oracle["residual_poison"] == 0
    oracle_count_introduces_clean_removal = oracle["removed_clean"] > estimated["removed_clean"]

    decomposition_label = _classify_decomposition(estimated, oracle)
    if decomposition_label not in VALID_LABELS:
        raise GateCStopCondition(f"{case['query_id']}: invalid decomposition label {decomposition_label!r}")

    return {
        "query_id": case["query_id"],
        "k": case["k"],
        "true_poison_count": true_poison_count,
        "true_clean_count": true_clean_count,
        "estimated_N_adv": estimated_n_adv,
        "count_error": count_error,
        "abs_count_error": estimated["abs_count_error"],
        "estimated_N_pairs": estimated["n_pairs"],
        "estimated_pp_count": estimated["pp_count"],
        "estimated_pc_count": estimated["pc_count"],
        "estimated_cc_count": estimated["cc_count"],
        "estimated_selected_pairs": estimated["selected_pairs"],
        "estimated_ranked_passage_order": estimated["ranked_passage_order"],
        "estimated_removed_poison": estimated["removed_poison"],
        "estimated_removed_clean": estimated["removed_clean"],
        "estimated_residual_poison": estimated["residual_poison"],
        "estimated_removal_precision": estimated["removal_precision"],
        "estimated_poison_recall": estimated["poison_recall"],
        "oracle_N_pairs": oracle["n_pairs"],
        "oracle_pp_count": oracle["pp_count"],
        "oracle_pc_count": oracle["pc_count"],
        "oracle_cc_count": oracle["cc_count"],
        "oracle_selected_pairs": oracle["selected_pairs"],
        "oracle_ranked_passage_order": oracle["ranked_passage_order"],
        "oracle_removed_poison": oracle["removed_poison"],
        "oracle_removed_clean": oracle["removed_clean"],
        "oracle_residual_poison": oracle["residual_poison"],
        "oracle_removal_precision": oracle["removal_precision"],
        "oracle_poison_recall": oracle["poison_recall"],
        "residual_poison_reduction": residual_poison_reduction,
        "oracle_count_fixes_failure": oracle_count_fixes_failure,
        "oracle_count_introduces_clean_removal": oracle_count_introduces_clean_removal,
        "decomposition_label": decomposition_label,
    }


def run_gate_c() -> List[dict]:
    cases = load_gate_b_cases()
    return [run_gate_c_query(case) for case in cases]


# ---------------------------------------------------------------------------
# Summary + report
# ---------------------------------------------------------------------------

def build_summary(rows: List[dict]) -> dict:
    n = len(rows)
    n_estimated_success = sum(1 for r in rows if r["estimated_residual_poison"] == 0)
    n_oracle_success = sum(1 for r in rows if r["oracle_residual_poison"] == 0)
    n_estimated_failures = n - n_estimated_success
    n_failures_fixed_by_oracle = sum(1 for r in rows if r["oracle_count_fixes_failure"])
    n_clean_removals_introduced = sum(1 for r in rows if r["oracle_count_introduces_clean_removal"])

    label_counts = {label: sum(1 for r in rows if r["decomposition_label"] == label) for label in VALID_LABELS}

    pp_leading_estimated = [r for r in rows if r["estimated_pp_count"] >= max(r["estimated_pc_count"], r["estimated_cc_count"]) and r["estimated_pp_count"] > 0]
    pp_leading_successes = sum(1 for r in pp_leading_estimated if r["estimated_residual_poison"] == 0)
    pp_leading_failures = sum(1 for r in pp_leading_estimated if r["estimated_residual_poison"] > 0)

    return {
        "n_queries": n,
        "n_estimated_pipeline_successes": n_estimated_success,
        "n_oracle_pipeline_successes": n_oracle_success,
        "n_estimated_pipeline_failures": n_estimated_failures,
        "n_failures_fixed_by_oracle_count": n_failures_fixed_by_oracle,
        "n_clean_removals_introduced_by_oracle": n_clean_removals_introduced,
        **{f"n_label_{label.split('.')[0]}": count for label, count in label_counts.items()},
        "n_pp_leading_estimated_cases": len(pp_leading_estimated),
        "n_pp_leading_estimated_successes": pp_leading_successes,
        "n_pp_leading_estimated_failures": pp_leading_failures,
    }


def write_report(rows: List[dict], summary: dict, path: Path) -> None:
    n = summary["n_queries"]
    lines: List[str] = []
    lines.append("# Gate C -- ORACLE-COUNT Decomposition Report")
    lines.append("")
    lines.append(
        "> Gate C separates RAGDefender error on the Gate-B queries into (1) Stage-1 COUNT ESTIMATION "
        "ERROR and (2) Stage-2 IDENTIFICATION ERROR conditional on a supplied count. It uses ONLY the "
        "already-frozen Stella similarity matrices from Gate B. The oracle supplies ONLY the true NUMBER "
        "of poisoned passages, never their identities, to the unchanged paper-faithful Stage-2 procedure. "
        "This is a diagnostic control, not a deployable defense."
    )
    lines.append("")
    lines.append(
        "No retrieval, generation, E1, CORAL, MMD, or LLM/API experiment was run. No Gate A/B artifact "
        "was overwritten."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Queries evaluated: **{n}**")
    lines.append(f"- Estimated-pipeline (Gate-B) successes: **{summary['n_estimated_pipeline_successes']}/{n}**")
    lines.append(f"- Oracle-count-pipeline successes: **{summary['n_oracle_pipeline_successes']}/{n}**")
    lines.append(
        f"- Of the {summary['n_estimated_pipeline_failures']} estimated-pipeline residual-poison failures, "
        f"**{summary['n_failures_fixed_by_oracle_count']}** become zero-residual-poison successes when only "
        "`N_adv` is corrected to the true poison count."
    )
    lines.append(
        f"- Oracle-count Stage 2 introduces additional clean-passage removal (relative to the estimated "
        f"pipeline) on **{summary['n_clean_removals_introduced_by_oracle']}/{n}** queries."
    )
    lines.append("")
    lines.append("### Decomposition label counts")
    lines.append("")
    for label in VALID_LABELS:
        key = f"n_label_{label.split('.')[0]}"
        lines.append(f"- {label}: **{summary[key]}/{n}**")
    lines.append("")

    lines.append("## Answers to the five Gate-C questions")
    lines.append("")
    lines.append(
        f"**1. Of the {summary['n_estimated_pipeline_failures']} Gate-B residual-poison failures, how many "
        "become zero-residual-poison successes when only N_adv is corrected to the true poison count?** "
        f"{summary['n_failures_fixed_by_oracle_count']}/{summary['n_estimated_pipeline_failures']}."
    )
    lines.append("")
    lines.append(
        "**2. Does oracle-count Stage 2 remove any clean passages?** "
        + (
            f"Yes, on {summary['n_clean_removals_introduced_by_oracle']}/{n} queries oracle-count Stage 2 "
            "removes MORE clean passages than the estimated pipeline did."
            if summary["n_clean_removals_introduced_by_oracle"] > 0
            else "No -- on this sample, oracle-count Stage 2 never removes more clean passages than the "
            "estimated pipeline did."
        )
    )
    lines.append("")
    _all_failures_fixed = (
        summary["n_failures_fixed_by_oracle_count"] == summary["n_estimated_pipeline_failures"]
        and summary["n_estimated_pipeline_failures"] > 0
    )
    if _all_failures_fixed:
        _q3_answer = (
            f"In this eight-query diagnostic sample, residual poison was primarily associated with "
            f"underestimation of N_adv; supplying the true poison count to the unchanged Stage-2 procedure "
            f"eliminated {summary['n_failures_fixed_by_oracle_count']}/{summary['n_estimated_pipeline_failures']} "
            "failures. This is provisional evidence from n=8 and must not be overclaimed as a general "
            "property of RAGDefender or of Stage 1."
        )
    else:
        _q3_answer = (
            f"In this eight-query diagnostic sample, {summary['n_failures_fixed_by_oracle_count']}/"
            f"{summary['n_estimated_pipeline_failures']} residual-poison failures were fully explained by "
            "count underestimation alone; the remainder involve some Stage-2 identification error even "
            "conditional on the correct count. This is provisional evidence from n=8."
        )
    lines.append(
        "**3. Is the Gate-B error therefore attributable primarily to Stage-1 count underestimation in "
        "this sample?** " + _q3_answer
    )
    lines.append("")
    lines.append(
        "**4. Does Stage 2 remain accurate conditional on the correct count?** "
        f"n_label_C (\"C. IDENTIFICATION LIMITED\" -- correct count, Stage 2 still fails) = "
        f"{summary['n_label_C']}/{n}."
    )
    lines.append("")
    lines.append(
        "**5. Is top_pair_pp actually useful as a discriminative failure variable, or merely common to "
        "both successes and failures?** "
        f"Of the {summary['n_pp_leading_estimated_cases']} queries whose estimated-pipeline selected-pair "
        f"set is PP-leading, {summary['n_pp_leading_estimated_successes']} are zero-residual-poison "
        f"successes and {summary['n_pp_leading_estimated_failures']} are residual-poison failures -- "
        + (
            "PP-leading geometry occurs in BOTH successes and failures in this sample, so it is NOT "
            "currently a discriminative success/failure variable."
            if summary["n_pp_leading_estimated_successes"] > 0 and summary["n_pp_leading_estimated_failures"] > 0
            else "insufficient variation in this n=8 sample to determine discriminative power either way."
        )
    )
    lines.append("")

    lines.append("## Per-query detail")
    lines.append("")
    lines.append(
        "| query_id | true_poison | N_est | count_error | est. residual_poison | oracle residual_poison | "
        "oracle removed_clean | label |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| `{r['query_id']}` | {r['true_poison_count']} | {r['estimated_N_adv']} | {r['count_error']} | "
            f"{r['estimated_residual_poison']} | {r['oracle_residual_poison']} | {r['oracle_removed_clean']} | "
            f"{r['decomposition_label']} |"
        )
    lines.append("")

    lines.append("## Interpretation constraints (do not exceed)")
    lines.append("")
    lines.append(
        "Allowed: \"In this eight-query diagnostic sample, residual poison was primarily associated with "
        "underestimation of N_adv; supplying the true poison count to the unchanged Stage-2 procedure "
        f"eliminated {summary['n_failures_fixed_by_oracle_count']}/{summary['n_estimated_pipeline_failures']} "
        "failures.\""
    )
    lines.append("")
    lines.append(
        "NOT allowed: \"RAGDefender fails because Stage 1 is fundamentally broken.\" This is an n=8 "
        "diagnostic sample selected via the Gate-A/Gate-B legacy-success-derived population, not a "
        "prospective, unbiased sample -- see the population-expansion plan for the next step required "
        "before any such general claim."
    )
    lines.append("")

    lines.append("## Data files")
    lines.append("")
    lines.append("- `gate_c_per_query.csv` -- every recorded field per query (context, Stage-1 count error, "
                  "estimated-count Stage-2, oracle-count Stage-2, decomposition, exact selected pairs and "
                  "frequency-score rankings under both counts).")
    lines.append("- `gate_c_summary.csv` -- one-row aggregate summary.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _check_no_overwrite(paths: List[Path]) -> None:
    existing = [p for p in paths if p.exists()]
    if existing:
        raise GateCStopCondition(f"Refusing to overwrite existing output artifact(s): {existing}")


def main() -> None:
    out_per_query = OUTPUT_DIR / "gate_c_per_query.csv"
    out_summary = OUTPUT_DIR / "gate_c_summary.csv"
    out_report = OUTPUT_DIR / "GATE_C_ORACLE_COUNT_REPORT.md"
    _check_no_overwrite([out_per_query, out_summary, out_report])

    rows = run_gate_c()
    summary = build_summary(rows)

    out_per_query.parent.mkdir(parents=True, exist_ok=True)
    with open(out_per_query, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(out_summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    write_report(rows, summary, out_report)

    print(f"Gate C complete: {len(rows)} queries evaluated.")
    print(f"Wrote: {out_per_query}")
    print(f"Wrote: {out_summary}")
    print(f"Wrote: {out_report}")


if __name__ == "__main__":
    main()
