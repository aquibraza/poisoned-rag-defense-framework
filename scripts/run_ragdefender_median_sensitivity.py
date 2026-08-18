"""Median-convention SENSITIVITY diagnostic (Gate-B follow-up STEP 3).

==========================================================================
SCOPE -- READ BEFORE INTERPRETING OUTPUT
==========================================================================
The final ACSAC 2025 paper is SILENT on how to break a tie when taking a
median over an EVEN number of values (both the per-passage median over
`k-1` non-self similarities when `k` is odd, and the global
median-of-medians `s_tilde` when the passage count is even). The
production `ragdefender_paper` implementation
(`defense.ragdefender_internals.concentration_stage1_paper`) resolves this
paper-silent gap using the AUTHORITY RULE: it reuses the authors' own
`torch.median` convention (lower-of-two-middle order statistics), NOT
NumPy's default (average-of-two-middle). See that function's docstring
and docs/RAGDEFENDER_FIDELITY_AUDIT_V2.md for the full justification.

THIS SCRIPT DOES NOT CHANGE THAT DECISION. It exists ONLY to measure how
much of an effect the choice would have had, as a documented
reproducibility/limitations diagnostic:

    A. PRIMARY (production, unchanged): lower-of-two-middle median
       = `ragdefender_internals.concentration_stage1_paper` exactly as-is.
    B. SENSITIVITY ONLY (this script, local, never exported): conventional
       statistical median = average of the two middle values for
       even-sized inputs.

Variant B is implemented ENTIRELY inside this script
(`_concentration_stage1_average_median` below) and is never imported by,
or merged into, `defense/ragdefender_internals.py` or
`defense/defense_runner.py`. It must never become a `--defense` option or
a production baseline -- see `TestSensitivityVariantIsIsolated` in the
companion test file for an explicit guard against this.

Uses ONLY the already-saved Gate-B Stella similarity matrices
(`results/diagnostics/ragdefender_gate_b/similarity/*.npy`) and the
already-saved `is_poison`/retrieved-composition ground truth from
`results/diagnostics/ragdefender_gate_b/gate_b_per_query.csv`. Zero new
embeddings, zero retrieval, zero generation, zero API calls. Never writes
into `results/diagnostics/ragdefender_gate_a/` or
`results/diagnostics/ragdefender_gate_b/` -- only into
`results/diagnostics/ragdefender_median_sensitivity/`.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from defense import ragdefender_internals  # noqa: E402

GATE_B_DIR = REPO_ROOT / "results/diagnostics/ragdefender_gate_b"
OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_median_sensitivity"

SOLE_GATE_B_SUCCESS_QID = "5a722b8655429971e9dc9329"


class MedianSensitivityStopCondition(RuntimeError):
    """Raised if this script would touch a Gate-A/B artifact or if the
    saved Gate-B inputs cannot be reproduced exactly."""


def _pipe(values) -> str:
    return "|".join(str(v) for v in values)


# ---------------------------------------------------------------------------
# SENSITIVITY-ONLY variant B: conventional (average-of-two-middle) median.
# Deliberately NOT added to defense/ragdefender_internals.py -- see module
# docstring. Structurally identical to `concentration_stage1_paper` in
# every other respect (self-exclusion, 1/(k-1) mean, AND logic, no flip);
# the ONLY difference is `_average_median_1d` vs. `_torch_style_median_1d`.
# ---------------------------------------------------------------------------

def _average_median_1d(values: np.ndarray) -> float:
    """Conventional statistical median: average of the two middle values
    for an even-length input (NumPy's own `np.median` default) -- the
    SENSITIVITY-ONLY alternative to the production
    `_torch_style_median_1d` lower-of-two-middle convention."""
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _concentration_stage1_average_median(cos_sim_matrix: np.ndarray) -> ragdefender_internals.ConcentrationResultPaper:
    """SENSITIVITY-ONLY: byte-for-byte the same Eq. (3) logic as
    `ragdefender_internals.concentration_stage1_paper`, except every
    median (`s_median_i` and `s_tilde`) uses `_average_median_1d` instead
    of the production `_torch_style_median_1d`. Never used outside this
    diagnostic script."""
    matrix = np.asarray(cos_sim_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"cos_sim_matrix must be square 2D, got shape {matrix.shape}")
    k = matrix.shape[0]
    if k < 2:
        raise ValueError("_concentration_stage1_average_median: need at least 2 passages to exclude self")

    s_mean = np.zeros(k, dtype=np.float64)
    s_median = np.zeros(k, dtype=np.float64)
    for i in range(k):
        off_diag_row = np.delete(matrix[i, :], i)
        s_mean[i] = off_diag_row.mean()
        s_median[i] = _average_median_1d(off_diag_row)

    s_bar = float(s_mean.mean())
    s_tilde = _average_median_1d(s_median)

    above_mean = s_mean > s_bar
    above_median = s_median > s_tilde
    adv_flag = above_mean & above_median

    return ragdefender_internals.ConcentrationResultPaper(
        s_mean=s_mean,
        s_median=s_median,
        s_bar=s_bar,
        s_tilde=s_tilde,
        above_mean=above_mean,
        above_median=above_median,
        adv_flag=adv_flag,
        n_adv_estimated=int(adv_flag.sum()),
    )


# ---------------------------------------------------------------------------
# Load saved Gate-B inputs (read-only)
# ---------------------------------------------------------------------------

def load_gate_b_cases() -> List[dict]:
    per_query_csv = GATE_B_DIR / "gate_b_per_query.csv"
    if not per_query_csv.exists():
        raise MedianSensitivityStopCondition(f"Gate-B per-query CSV not found: {per_query_csv}")
    df = pd.read_csv(per_query_csv)

    cases = []
    for _, row in df.iterrows():
        qid = row["query_id"]
        matrix_path = REPO_ROOT / row["similarity_matrix_path"]
        if not matrix_path.exists():
            raise MedianSensitivityStopCondition(f"{qid}: saved Gate-B similarity matrix missing: {matrix_path}")
        matrix = np.load(matrix_path)
        is_poison = np.array([bool(int(x)) for x in str(row["is_poison_i"]).split("|")])
        if matrix.shape[0] != len(is_poison):
            raise MedianSensitivityStopCondition(
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


def _run_variant(case: dict, stage1) -> dict:
    matrix = case["matrix"]
    is_poison = case["is_poison"]
    stage2 = ragdefender_internals.stage2_pair_frequency(matrix, n_adv=stage1.n_adv_estimated, p=2.0)
    removed_indices = list(stage2.selected_indices)
    removed_poison = int(sum(1 for idx in removed_indices if is_poison[idx]))
    removed_clean = int(sum(1 for idx in removed_indices if not is_poison[idx]))
    residual_poison = case["n_retrieved_poison"] - removed_poison
    residual_clean = case["n_retrieved_clean"] - removed_clean
    return {
        "n_adv": stage1.n_adv_estimated,
        "s_tilde": stage1.s_tilde,
        "adv_flag_indices": sorted(int(i) for i in np.where(stage1.adv_flag)[0]),
        "n_pairs": stage2.n_pairs,
        "removed_indices": sorted(removed_indices),
        "removed_poison": removed_poison,
        "removed_clean": removed_clean,
        "residual_poison": residual_poison,
        "residual_clean": residual_clean,
    }


def run_sensitivity() -> List[dict]:
    cases = load_gate_b_cases()
    rows = []
    for case in cases:
        matrix = case["matrix"]

        stage1_lower = ragdefender_internals.concentration_stage1_paper(matrix)
        # Cross-check: the saved Gate-B N_adv must match recomputing the
        # PRIMARY (lower-of-two-middle) variant on the saved matrix exactly
        # -- confirms this script reproduces Gate B's own inputs faithfully.
        if stage1_lower.n_adv_estimated != case["gate_b_n_adv"]:
            raise MedianSensitivityStopCondition(
                f"{case['query_id']}: recomputed primary N_adv "
                f"({stage1_lower.n_adv_estimated}) != saved Gate-B N_adv ({case['gate_b_n_adv']}) "
                "-- saved matrix does not reproduce Gate B's own recorded result."
            )
        stage1_average = _concentration_stage1_average_median(matrix)

        lower = _run_variant(case, stage1_lower)
        average = _run_variant(case, stage1_average)

        n_adv_changed = lower["n_adv"] != average["n_adv"]
        removal_set_changed = set(lower["removed_indices"]) != set(average["removed_indices"])

        rows.append({
            "query_id": case["query_id"],
            "n_retrieved_poison": case["n_retrieved_poison"],
            "n_retrieved_clean": case["n_retrieved_clean"],
            "n_adv_lower": lower["n_adv"],
            "n_adv_average": average["n_adv"],
            "n_adv_changed": n_adv_changed,
            "s_tilde_lower": stage1_lower.s_tilde,
            "s_tilde_average": stage1_average.s_tilde,
            "and_flag_indices_lower": _pipe(lower["adv_flag_indices"]),
            "and_flag_indices_average": _pipe(average["adv_flag_indices"]),
            "removed_indices_lower": _pipe(lower["removed_indices"]),
            "removed_indices_average": _pipe(average["removed_indices"]),
            "removed_poison_lower": lower["removed_poison"],
            "removed_poison_average": average["removed_poison"],
            "removed_clean_lower": lower["removed_clean"],
            "removed_clean_average": average["removed_clean"],
            "residual_poison_lower": lower["residual_poison"],
            "residual_poison_average": average["residual_poison"],
            "removal_set_changed": removal_set_changed,
        })
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(rows: List[dict], path: Path) -> None:
    by_qid = {r["query_id"]: r for r in rows}
    n_adv_changed_count = sum(1 for r in rows if r["n_adv_changed"])
    removal_changed_count = sum(1 for r in rows if r["removal_set_changed"])

    success_row = by_qid.get(SOLE_GATE_B_SUCCESS_QID)
    success_status_changed = (
        success_row is not None and (success_row["residual_poison_lower"] == 0) != (success_row["residual_poison_average"] == 0)
    )

    n_adv4_became_5 = sum(
        1 for r in rows if r["n_adv_lower"] == 4 and r["n_adv_average"] == 5
    )
    n_adv4_total = sum(1 for r in rows if r["n_adv_lower"] == 4)

    lines: List[str] = []
    lines.append("# Median-Convention SENSITIVITY Report (Gate-B follow-up STEP 3)")
    lines.append("")
    lines.append(
        "> **This is a sensitivity analysis over a paper-silent implementation choice. The primary "
        "`ragdefender_paper` implementation remains the lower-middle torch-style convention selected "
        "under the documented authority rule.** Variant B below (average-of-two-middle) is a diagnostic "
        "only -- it is not, and must never become, a `--defense` option or production baseline."
    )
    lines.append("")
    lines.append(
        "No retrieval, generation, E1, CORAL, or MMD experiment was run. This script only reuses the "
        "already-saved Gate-B Stella similarity matrices; it computes zero new embeddings."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Queries evaluated: {len(rows)}")
    lines.append(f"- Queries whose `N_adv` changes between conventions: **{n_adv_changed_count}/{len(rows)}**")
    lines.append(f"- Queries whose final Stage-2 removal SET changes between conventions: **{removal_changed_count}/{len(rows)}**")
    lines.append(
        f"- Does the sole Gate-B success (`{SOLE_GATE_B_SUCCESS_QID}`) change zero-residual-poison status? "
        f"**{'YES' if success_status_changed else 'NO'}**"
        + (f" (lower: residual_poison={success_row['residual_poison_lower']}, average: residual_poison={success_row['residual_poison_average']})" if success_row is not None else "")
    )
    lines.append(
        f"- Of the {n_adv4_total} queries with `N_adv=4` under the primary (lower) convention, "
        f"**{n_adv4_became_5}** become `N_adv=5` (the true poison count) under the average convention."
    )
    lines.append("")

    if n_adv_changed_count > 0 or removal_changed_count > 0:
        lines.append(
            "**Interpretation: the sensitivity DOES materially change outcomes for at least one query in "
            "this sample -- flagged as an important limitation/reproducibility issue.** The median "
            "tie-break convention is not a purely cosmetic implementation detail on this dataset; "
            "downstream claims that depend on an exact `N_adv`/removal-set value for any specific query "
            "listed below should be treated as convention-dependent, not convention-invariant."
        )
    else:
        lines.append(
            "**Interpretation: the sensitivity does NOT change outcomes for any of the evaluated queries.** "
            "The Gate-B count-underestimation observation is robust to this specific implementation "
            "ambiguity in this evaluated sample."
        )
    lines.append("")

    lines.append("## Per-query detail")
    lines.append("")
    lines.append(
        "| query_id | N_adv (lower/avg) | s_tilde (lower/avg) | removed_poison (lower/avg) | "
        "residual_poison (lower/avg) | removal set changed |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| `{r['query_id']}` | {r['n_adv_lower']}/{r['n_adv_average']} | "
            f"{r['s_tilde_lower']:.6f}/{r['s_tilde_average']:.6f} | "
            f"{r['removed_poison_lower']}/{r['removed_poison_average']} | "
            f"{r['residual_poison_lower']}/{r['residual_poison_average']} | "
            f"{r['removal_set_changed']} |"
        )
    lines.append("")

    lines.append("## Data files")
    lines.append("")
    lines.append("- `median_sensitivity_per_query.csv` -- every recorded field per query for both conventions.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _check_no_overwrite(paths: List[Path]) -> None:
    existing = [p for p in paths if p.exists()]
    if existing:
        raise MedianSensitivityStopCondition(f"Refusing to overwrite existing output artifact(s): {existing}")


def main() -> None:
    out_csv = OUTPUT_DIR / "median_sensitivity_per_query.csv"
    out_report = OUTPUT_DIR / "MEDIAN_SENSITIVITY_REPORT.md"
    _check_no_overwrite([out_csv, out_report])

    rows = run_sensitivity()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    write_report(rows, out_report)

    print(f"Median sensitivity complete: {len(rows)} queries evaluated.")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_report}")


if __name__ == "__main__":
    main()
