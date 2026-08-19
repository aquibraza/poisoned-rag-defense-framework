"""STEP 5 -- Expanded ORACLE-COUNT decomposition over the prospectively
frozen population (STEP 3), reusing the expanded baseline's (STEP 4) saved
Stella similarity matrices and composition.

==========================================================================
SCOPE
==========================================================================
Reads ONLY `results/diagnostics/ragdefender_expanded_baseline/expanded_baseline_per_query.csv`
and its saved `similarity/*.npy` matrices (STEP 4 outputs) -- never
recomputes embeddings, never touches Gate A/B/C's own artifacts
(`results/diagnostics/ragdefender_gate_a/`, `ragdefender_gate_b/`,
`ragdefender_gate_c_oracle_count/`), and never re-derives the query list.

For every one of the 42 frozen queries:

    ESTIMATED PIPELINE:
        N_est = the SAME `N_adv` already recorded by the expanded
        baseline (recomputed here from the saved matrix via the
        unchanged `concentration_stage1_paper`, and cross-checked
        against the baseline's own saved `n_adv` column -- STOP if they
        disagree).
        Stage2(matrix, N_est).

    ORACLE-COUNT PIPELINE:
        Stage2(matrix, M)     where M = observed retrieved poison count.

The oracle supplies ONLY the integer count `M`, never passage identities,
to the unchanged paper-faithful `stage2_pair_frequency`. Diagnostic
control, not a deployable defense; not wired into
`defense/defense_runner.py` or `defense/dispatch.py`.

Adds, beyond the original (n=8) Gate C:
- regime-level (A/B/C/D) aggregation of the decomposition labels (STEP 5,
  "how does decomposition vary by poison-count regime");
- STEP 5B: `delta_N = N_adv - M` distribution, `|delta_N|` distribution,
  a `delta_N` x `residual_poison` cross-tabulation, and the two
  conditional-probability estimates
  (`P(residual_poison>0 | delta_N<0)`, `P(residual_poison>0 | delta_N=0)`),
  reported with raw counts alongside every rate so unstable tiny cells are
  visible, never silently smoothed over.

Writes ONLY to `results/diagnostics/ragdefender_expanded_gate_c/`; refuses
to overwrite existing outputs there, and never overwrites any Gate-A/B/C
or expanded-baseline artifact.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from defense import ragdefender_internals  # noqa: E402

BASELINE_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_baseline"
OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_gate_c"

VALID_LABELS = (
    "A. COUNT-LIMITED",
    "B. COUNT + IDENTIFICATION LIMITED",
    "C. IDENTIFICATION LIMITED",
    "D. BASELINE SUCCESS",
)
REGIME_ORDER = ["A_BELOW_CEILING", "B_AT_CEILING", "C_ABOVE_CEILING", "D_ALL_POISON"]


class ExpandedGateCStopCondition(RuntimeError):
    """Raised if this script would touch a Gate-A/B/C or expanded-baseline
    artifact, use oracle identities instead of only the oracle count, or
    if the saved baseline matrices/labels cannot be reproduced exactly."""


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
# Load saved expanded-baseline inputs (read-only)
# ---------------------------------------------------------------------------

def load_baseline_cases() -> List[dict]:
    per_query_csv = BASELINE_DIR / "expanded_baseline_per_query.csv"
    if not per_query_csv.exists():
        raise ExpandedGateCStopCondition(
            f"Expanded baseline per-query CSV not found: {per_query_csv} -- run "
            "scripts/run_ragdefender_expanded_baseline.py (STEP 4) first."
        )
    df = pd.read_csv(per_query_csv)

    cases = []
    for _, row in df.iterrows():
        qid = row["query_id"]
        matrix_path = REPO_ROOT / row["similarity_matrix_path"]
        if not matrix_path.exists():
            raise ExpandedGateCStopCondition(f"{qid}: saved similarity matrix missing: {matrix_path}")
        matrix = np.load(matrix_path)
        is_poison = np.array([bool(int(x)) for x in str(row["is_poison_i"]).split("|")])
        if matrix.shape[0] != len(is_poison):
            raise ExpandedGateCStopCondition(
                f"{qid}: matrix shape {matrix.shape} does not match is_poison length {len(is_poison)}."
            )
        cases.append({
            "query_id": qid,
            "matrix": matrix,
            "is_poison": is_poison,
            "m_poison": int(row["m_poison"]),
            "c_clean": int(row["c_clean"]),
            "k": int(row["k"]),
            "regime": row["regime"],
            "baseline_n_adv": int(row["n_adv"]),
            "top_pair_pp": bool(row["top_pair_pp"]) if "top_pair_pp" in row else None,
        })
    return cases


# ---------------------------------------------------------------------------
# Core Gate-C computation (independently unit-testable on synthetic
# matrices, without touching any real artifact)
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
        "removed_indices": _pipe(sorted(removed_indices)),
        "removed_poison": removed_poison,
        "removed_clean": removed_clean,
        "residual_poison": residual_poison,
        "removal_precision": removal_precision,
        "poison_recall": poison_recall,
    }


def _classify_decomposition(estimated: dict, oracle: dict) -> str:
    """Priority-ordered decision tree over EXACTLY the four allowed
    labels -- identical logic to the original (n=8) Gate C."""
    if estimated["residual_poison"] == 0:
        return "D. BASELINE SUCCESS"
    if estimated["count_error"] == 0:
        return "C. IDENTIFICATION LIMITED"
    if oracle["residual_poison"] == 0 and oracle["removed_clean"] == 0:
        return "A. COUNT-LIMITED"
    return "B. COUNT + IDENTIFICATION LIMITED"


def run_gate_c_query(case: dict) -> dict:
    matrix = case["matrix"]
    is_poison = case["is_poison"]
    true_poison_count = case["m_poison"]
    true_clean_count = case["c_clean"]

    stage1 = ragdefender_internals.concentration_stage1_paper(matrix)
    if stage1.n_adv_estimated != case["baseline_n_adv"]:
        raise ExpandedGateCStopCondition(
            f"{case['query_id']}: recomputed Stage-1 N_adv ({stage1.n_adv_estimated}) != "
            f"saved expanded-baseline N_adv ({case['baseline_n_adv']}) -- saved matrix does not "
            "reproduce the baseline's own recorded result."
        )
    estimated_n_adv = stage1.n_adv_estimated
    count_error = estimated_n_adv - true_poison_count

    estimated = _run_stage2_metrics(matrix, is_poison, n_adv=estimated_n_adv, true_poison_count=true_poison_count)
    estimated["count_error"] = count_error
    estimated["abs_count_error"] = abs(count_error)

    # ORACLE: supplies ONLY the correct COUNT (an integer, `true_poison_count`
    # = M, the observed retrieved poison count), never passage identities --
    # `_run_stage2_metrics` receives `n_adv=true_poison_count` and runs the
    # SAME unchanged `stage2_pair_frequency` over the SAME matrix.
    oracle = _run_stage2_metrics(matrix, is_poison, n_adv=true_poison_count, true_poison_count=true_poison_count)

    residual_poison_reduction = estimated["residual_poison"] - oracle["residual_poison"]
    oracle_count_fixes_failure = estimated["residual_poison"] > 0 and oracle["residual_poison"] == 0
    oracle_count_introduces_clean_removal = oracle["removed_clean"] > estimated["removed_clean"]

    decomposition_label = _classify_decomposition(estimated, oracle)
    if decomposition_label not in VALID_LABELS:
        raise ExpandedGateCStopCondition(f"{case['query_id']}: invalid decomposition label {decomposition_label!r}")

    return {
        "query_id": case["query_id"],
        "k": case["k"],
        "regime": case["regime"],
        "m_poison": true_poison_count,
        "c_clean": true_clean_count,
        "estimated_N_adv": estimated_n_adv,
        "delta_N": count_error,
        "abs_delta_N": estimated["abs_count_error"],
        "estimated_N_pairs": estimated["n_pairs"],
        "estimated_pp_count": estimated["pp_count"],
        "estimated_pc_count": estimated["pc_count"],
        "estimated_cc_count": estimated["cc_count"],
        "estimated_removed_poison": estimated["removed_poison"],
        "estimated_removed_clean": estimated["removed_clean"],
        "estimated_residual_poison": estimated["residual_poison"],
        "estimated_removal_precision": estimated["removal_precision"],
        "estimated_poison_recall": estimated["poison_recall"],
        "top_pair_pp": case["top_pair_pp"],
        "oracle_N_pairs": oracle["n_pairs"],
        "oracle_pp_count": oracle["pp_count"],
        "oracle_pc_count": oracle["pc_count"],
        "oracle_cc_count": oracle["cc_count"],
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


def run_expanded_gate_c() -> List[dict]:
    cases = load_baseline_cases()
    return [run_gate_c_query(case) for case in cases]


# ---------------------------------------------------------------------------
# Regime-level aggregation (STEP 5, "how does decomposition vary by
# poison-count regime")
# ---------------------------------------------------------------------------

def build_regime_decomposition(rows: List[dict]) -> List[dict]:
    aggregates: List[dict] = []
    for regime in REGIME_ORDER:
        regime_rows = [r for r in rows if r["regime"] == regime]
        n = len(regime_rows)
        agg: Dict = {"regime": regime, "n_queries": n}
        for label in VALID_LABELS:
            key = f"n_label_{label.split('.')[0]}"
            agg[key] = sum(1 for r in regime_rows if r["decomposition_label"] == label)
        if n > 0:
            n_estimated_failures = sum(1 for r in regime_rows if r["estimated_residual_poison"] > 0)
            n_fixed = sum(1 for r in regime_rows if r["oracle_count_fixes_failure"])
            agg["n_estimated_failures"] = n_estimated_failures
            agg["n_failures_fixed_by_oracle_count"] = n_fixed
            agg["fraction_failures_fixed"] = (n_fixed / n_estimated_failures) if n_estimated_failures > 0 else None
            agg["n_oracle_successes"] = sum(1 for r in regime_rows if r["oracle_residual_poison"] == 0)
            agg["n_clean_removals_introduced"] = sum(1 for r in regime_rows if r["oracle_count_introduces_clean_removal"])
        aggregates.append(agg)
    return aggregates


# ---------------------------------------------------------------------------
# STEP 5B -- count error (delta_N) as a primary variable
# ---------------------------------------------------------------------------

def build_count_error_analysis(rows: List[dict]) -> dict:
    delta_ns = [r["delta_N"] for r in rows]
    abs_delta_ns = [r["abs_delta_N"] for r in rows]

    delta_n_distribution: Dict[int, int] = {}
    for d in delta_ns:
        delta_n_distribution[d] = delta_n_distribution.get(d, 0) + 1

    abs_delta_n_distribution: Dict[int, int] = {}
    for d in abs_delta_ns:
        abs_delta_n_distribution[d] = abs_delta_n_distribution.get(d, 0) + 1

    # Cross-tab: delta_N sign bucket (negative / zero / positive) vs.
    # residual_poison (>0 / ==0), reported as raw counts (never as a rate
    # alone) so small cells are visible.
    cross_tab: Dict[str, Dict[str, int]] = {
        "delta_N<0": {"residual_poison>0": 0, "residual_poison==0": 0},
        "delta_N==0": {"residual_poison>0": 0, "residual_poison==0": 0},
        "delta_N>0": {"residual_poison>0": 0, "residual_poison==0": 0},
    }
    for r in rows:
        bucket = "delta_N<0" if r["delta_N"] < 0 else ("delta_N==0" if r["delta_N"] == 0 else "delta_N>0")
        residual_key = "residual_poison>0" if r["estimated_residual_poison"] > 0 else "residual_poison==0"
        cross_tab[bucket][residual_key] += 1

    def _conditional_p(bucket: str) -> Optional[Dict[str, float]]:
        n_bucket = sum(cross_tab[bucket].values())
        if n_bucket == 0:
            return None
        p = cross_tab[bucket]["residual_poison>0"] / n_bucket
        return {"n": n_bucket, "n_residual_positive": cross_tab[bucket]["residual_poison>0"], "p": p}

    return {
        "n": len(rows),
        "delta_n_distribution": dict(sorted(delta_n_distribution.items())),
        "abs_delta_n_distribution": dict(sorted(abs_delta_n_distribution.items())),
        "mean_delta_n": float(np.mean(delta_ns)) if delta_ns else None,
        "median_delta_n": float(np.median(delta_ns)) if delta_ns else None,
        "mean_abs_delta_n": float(np.mean(abs_delta_ns)) if abs_delta_ns else None,
        "median_abs_delta_n": float(np.median(abs_delta_ns)) if abs_delta_ns else None,
        "cross_tab": cross_tab,
        "p_residual_given_delta_n_negative": _conditional_p("delta_N<0"),
        "p_residual_given_delta_n_zero": _conditional_p("delta_N==0"),
        "p_residual_given_delta_n_positive": _conditional_p("delta_N>0"),
    }


# ---------------------------------------------------------------------------
# top_pair_pp association with outcome (Q9), controlling for count error
# ---------------------------------------------------------------------------

def build_top_pair_pp_association(rows: List[dict]) -> dict:
    rows_with_flag = [r for r in rows if r["top_pair_pp"] is not None]
    exact_count_rows = [r for r in rows_with_flag if r["delta_N"] == 0]

    def _rate(subset: List[dict]) -> dict:
        n = len(subset)
        if n == 0:
            return {"n": 0, "n_pp": 0, "n_success_given_pp": 0, "n_success_given_not_pp": 0}
        pp_rows = [r for r in subset if r["top_pair_pp"]]
        not_pp_rows = [r for r in subset if not r["top_pair_pp"]]
        return {
            "n": n,
            "n_pp": len(pp_rows),
            "n_not_pp": len(not_pp_rows),
            "n_success_given_pp": sum(1 for r in pp_rows if r["estimated_residual_poison"] == 0),
            "n_success_given_not_pp": sum(1 for r in not_pp_rows if r["estimated_residual_poison"] == 0),
        }

    return {
        "overall": _rate(rows_with_flag),
        "exact_count_only": _rate(exact_count_rows),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _check_no_overwrite(paths: List[Path]) -> None:
    existing = [p for p in paths if p.exists()]
    if existing:
        raise ExpandedGateCStopCondition(f"Refusing to overwrite existing output artifact(s): {existing}")


def build_summary(rows: List[dict]) -> dict:
    n = len(rows)
    n_estimated_success = sum(1 for r in rows if r["estimated_residual_poison"] == 0)
    n_oracle_success = sum(1 for r in rows if r["oracle_residual_poison"] == 0)
    n_estimated_failures = n - n_estimated_success
    n_failures_fixed_by_oracle = sum(1 for r in rows if r["oracle_count_fixes_failure"])
    n_clean_removals_introduced = sum(1 for r in rows if r["oracle_count_introduces_clean_removal"])
    label_counts = {label: sum(1 for r in rows if r["decomposition_label"] == label) for label in VALID_LABELS}

    return {
        "n_queries": n,
        "n_estimated_pipeline_successes": n_estimated_success,
        "n_oracle_pipeline_successes": n_oracle_success,
        "n_estimated_pipeline_failures": n_estimated_failures,
        "n_failures_fixed_by_oracle_count": n_failures_fixed_by_oracle,
        "n_clean_removals_introduced_by_oracle": n_clean_removals_introduced,
        **{f"n_label_{label.split('.')[0]}": count for label, count in label_counts.items()},
    }


def write_report(
    rows: List[dict],
    summary: dict,
    regime_decomp: List[dict],
    count_error: dict,
    pp_assoc: dict,
    path: Path,
) -> None:
    n = summary["n_queries"]
    lines: List[str] = []
    lines.append("# Expanded Gate-C -- Oracle-Count Decomposition Report (STEP 5)")
    lines.append("")
    lines.append(
        "> Same decomposition as the original (n=8) Gate C, over the 42-query PROSPECTIVELY FROZEN "
        "population (`../ragdefender_expanded_baseline/PROSPECTIVE_POPULATION_FREEZE.md`), reusing its "
        "saved Stella similarity matrices. The oracle supplies ONLY the observed retrieved poison count "
        "M, never passage identities, to the unchanged paper-faithful Stage-2 procedure. Diagnostic "
        "control, not a deployable defense. No retrieval, generation, E1, CORAL, MMD, or LLM/API "
        "experiment was run."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Queries evaluated: **{n}**")
    lines.append(f"- Estimated-pipeline (baseline) successes: **{summary['n_estimated_pipeline_successes']}/{n}**")
    lines.append(f"- Oracle-count-pipeline successes: **{summary['n_oracle_pipeline_successes']}/{n}**")
    lines.append(
        f"- Of the {summary['n_estimated_pipeline_failures']} estimated-pipeline residual-poison failures, "
        f"**{summary['n_failures_fixed_by_oracle_count']}** become zero-residual-poison successes when only "
        "`N_adv` is corrected to the true poison count M."
    )
    lines.append(
        f"- Oracle-count Stage 2 introduces additional clean-passage removal (relative to the estimated "
        f"pipeline) on **{summary['n_clean_removals_introduced_by_oracle']}/{n}** queries."
    )
    lines.append("")
    lines.append("### Decomposition label counts (all regimes combined)")
    lines.append("")
    for label in VALID_LABELS:
        key = f"n_label_{label.split('.')[0]}"
        lines.append(f"- {label}: **{summary[key]}/{n}**")
    lines.append("")

    lines.append("## Answers to the expanded Gate-C questions")
    lines.append("")
    frac_count_limited = (
        summary["n_label_A"] / summary["n_estimated_pipeline_failures"]
        if summary["n_estimated_pipeline_failures"] > 0 else None
    )
    lines.append(
        f"**1. What fraction of baseline failures are COUNT-LIMITED?** "
        + (f"{summary['n_label_A']}/{summary['n_estimated_pipeline_failures']} "
           f"({frac_count_limited:.1%})." if frac_count_limited is not None else "N/A (no baseline failures).")
    )
    lines.append("")
    frac_id_limited = (
        summary["n_label_C"] / summary["n_estimated_pipeline_failures"]
        if summary["n_estimated_pipeline_failures"] > 0 else None
    )
    lines.append(
        f"**2. What fraction remain IDENTIFICATION-LIMITED after supplying the true count?** "
        + (f"{summary['n_label_C']}/{summary['n_estimated_pipeline_failures']} "
           f"({frac_id_limited:.1%}) are labeled C (count already correct, still fails); "
           f"an additional {summary['n_label_B']}/{summary['n_estimated_pipeline_failures']} are labeled B "
           "(count-limited AND still has some identification cost after correction)."
           if frac_id_limited is not None else "N/A.")
    )
    lines.append("")
    lines.append(
        f"**3. Does the original (n=8) Gate-C 7/7 count-limited-failures result replicate prospectively?** "
        + (
            f"Partially/fully -- see the decomposition label counts above "
            f"({summary['n_label_A']}/{summary['n_estimated_pipeline_failures']} of THIS 42-query "
            "population's failures are COUNT-LIMITED). Compare directly against the original n=8 result "
            "(7/7 COUNT-LIMITED) rather than assuming replication; see regime breakdown below for whether "
            "this holds uniformly or varies by regime."
        )
    )
    lines.append("")
    lines.append("**4. How does the decomposition vary by poison-count regime?** See table below.")
    lines.append("")
    lines.append(
        "| regime | n | A (count-limited) | B (count+id) | C (id-limited) | D (baseline success) | "
        "failures fixed / failures |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for agg in regime_decomp:
        if agg["n_queries"] == 0:
            lines.append(f"| {agg['regime']} | 0 | -- | -- | -- | -- | -- |")
            continue
        frac = agg.get("fraction_failures_fixed")
        frac_str = (
            f"{agg['n_failures_fixed_by_oracle_count']}/{agg['n_estimated_failures']} ({frac:.1%})"
            if frac is not None else "N/A (0 failures)"
        )
        lines.append(
            f"| {agg['regime']} | {agg['n_queries']} | {agg['n_label_A']} | {agg['n_label_B']} | "
            f"{agg['n_label_C']} | {agg['n_label_D']} | {frac_str} |"
        )
    lines.append("")

    regime_a = next((a for a in regime_decomp if a["regime"] == "A_BELOW_CEILING"), None)
    lines.append(
        "**5. In the BELOW-CEILING regime, does Stage 1 still underestimate often?** "
        + (
            "N/A -- Regime A (below ceiling) has zero representation in this prospective population "
            "(see `PROSPECTIVE_POPULATION_FREEZE.md`); this question cannot be answered on the current "
            "sample and requires either a different attack configuration or a different dataset."
            if regime_a is None or regime_a["n_queries"] == 0 else
            f"n={regime_a['n_queries']}, {regime_a['n_label_A'] + regime_a['n_label_B']}/{regime_a['n_queries']} "
            "show any count underestimation-linked label."
        )
    )
    lines.append("")
    regime_b = next((a for a in regime_decomp if a["regime"] == "B_AT_CEILING"), None)
    lines.append(
        "**6. At the ceiling M=floor(k/2), how frequently does Eq. (3) reach the true count?** "
        + (
            f"n={regime_b['n_queries']}; {regime_b['n_label_D']}/{regime_b['n_queries']} queries are "
            "baseline successes (N_adv already equals M); the remainder underestimate even at the "
            "ceiling boundary itself."
            if regime_b and regime_b["n_queries"] > 0 else "N/A -- no Regime-B queries in this population."
        )
    )
    lines.append("")
    regime_c = next((a for a in regime_decomp if a["regime"] == "C_ABOVE_CEILING"), None)
    lines.append(
        "**7. Above the ceiling, does the structural count bound translate into residual poison, or can "
        "Stage 2/system behavior compensate in any way?** "
        + (
            f"n={regime_c['n_queries']}; {regime_c['n_label_D']}/{regime_c['n_queries']} are baseline "
            f"successes (impossible by the ceiling proof unless M<=floor(k/2), which contradicts Regime C's "
            "definition -- see the Regime-C invariant check in the expanded baseline; any Regime-C 'success' "
            "here would indicate M was misclassified, not that Stage 2 compensated for the count "
            f"structurally). Failures fixed by oracle count: {regime_c['n_failures_fixed_by_oracle_count']}/"
            f"{regime_c['n_estimated_failures']} -- the oracle SUPPLIES M directly, so this measures whether "
            "Stage 2 can still isolate M poison passages once given the (above-ceiling) count directly, not "
            "whether Eq. (3) itself can reach M (it structurally cannot, per the ceiling proof)."
            if regime_c and regime_c["n_queries"] > 0 else "N/A -- no Regime-C queries in this population."
        )
    )
    lines.append("")
    lines.append(
        f"**8. Does oracle-count Stage 2 introduce clean removals?** "
        f"{summary['n_clean_removals_introduced_by_oracle']}/{n} queries show the oracle-count pipeline "
        "removing MORE clean passages than the estimated pipeline did."
    )
    lines.append("")
    pp_overall = pp_assoc["overall"]
    pp_exact = pp_assoc["exact_count_only"]
    lines.append(
        "**9. Is top_pair_pp associated with outcome after controlling/describing count error?** "
        f"Overall (n={pp_overall['n']}): of {pp_overall['n_pp']} PP-leading queries, "
        f"{pp_overall['n_success_given_pp']} succeed; of {pp_overall['n_not_pp']} non-PP-leading queries, "
        f"{pp_overall['n_success_given_not_pp']} succeed. Restricted to exact-count queries only "
        f"(delta_N==0, n={pp_exact['n']}, i.e. controlling for count error): of {pp_exact['n_pp']} PP-leading, "
        f"{pp_exact['n_success_given_pp']} succeed; of {pp_exact['n_not_pp']} non-PP-leading, "
        f"{pp_exact['n_success_given_not_pp']} succeed. This is a descriptive association only -- **not** "
        "claimed as causal; the exact-count subset is small and any rate there must be read as a raw count, "
        "not a stable probability."
    )
    lines.append("")

    lines.append("## STEP 5B -- Count error (delta_N = N_adv - M) as a primary variable")
    lines.append("")
    lines.append(f"- Mean delta_N: **{count_error['mean_delta_n']:+.3f}**, median: **{count_error['median_delta_n']:+.1f}**")
    lines.append(f"- Mean |delta_N|: **{count_error['mean_abs_delta_n']:.3f}**, median: **{count_error['median_abs_delta_n']:.1f}**")
    lines.append("")
    lines.append("### delta_N distribution")
    lines.append("")
    lines.append("| delta_N | count |")
    lines.append("|---|---|")
    for d, c in count_error["delta_n_distribution"].items():
        lines.append(f"| {d:+d} | {c} |")
    lines.append("")
    lines.append("### |delta_N| distribution")
    lines.append("")
    lines.append("| \\|delta_N\\| | count |")
    lines.append("|---|---|")
    for d, c in count_error["abs_delta_n_distribution"].items():
        lines.append(f"| {d} | {c} |")
    lines.append("")
    lines.append("### Cross-tab: delta_N sign bucket vs. residual_poison (raw counts)")
    lines.append("")
    lines.append("| bucket | residual_poison>0 | residual_poison==0 |")
    lines.append("|---|---|---|")
    for bucket, counts in count_error["cross_tab"].items():
        lines.append(f"| {bucket} | {counts['residual_poison>0']} | {counts['residual_poison==0']} |")
    lines.append("")
    lines.append("### Conditional probabilities (raw counts alongside every rate)")
    lines.append("")
    for label, key in (
        ("P(residual_poison>0 | delta_N<0)", "p_residual_given_delta_n_negative"),
        ("P(residual_poison>0 | delta_N=0)", "p_residual_given_delta_n_zero"),
        ("P(residual_poison>0 | delta_N>0)", "p_residual_given_delta_n_positive"),
    ):
        val = count_error[key]
        if val is None:
            lines.append(f"- {label}: **N/A** (n=0 in this bucket).")
        else:
            lines.append(
                f"- {label}: **{val['p']:.1%}** ({val['n_residual_positive']}/{val['n']}) "
                + ("-- small cell, do not overinterpret." if val["n"] < 5 else "")
            )
    lines.append("")

    lines.append("## Per-query detail")
    lines.append("")
    lines.append(
        "| query_id | regime | M | N_est | delta_N | est. residual_poison | oracle residual_poison | "
        "oracle removed_clean | label |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| `{r['query_id']}` | {r['regime']} | {r['m_poison']} | {r['estimated_N_adv']} | "
            f"{r['delta_N']:+d} | {r['estimated_residual_poison']} | {r['oracle_residual_poison']} | "
            f"{r['oracle_removed_clean']} | {r['decomposition_label']} |"
        )
    lines.append("")

    lines.append("## Interpretation constraints (do not exceed)")
    lines.append("")
    lines.append(
        "Allowed: report the exact fractions above for THIS 42-query prospective population. Compare "
        "against, but do not merge with, the original n=8 Gate-C development-sample result (7/7 "
        "COUNT-LIMITED)."
    )
    lines.append("")
    lines.append(
        "NOT allowed without further controlled comparison: claiming `top_pair_pp` is CAUSALLY related to "
        "outcome (Q9 above is descriptive only); claiming this population's decomposition generalizes "
        "beyond HotpotQA k=10 under this specific N=5-candidate attack configuration; mixing any k=2 "
        "sanity-check result into this k=10 aggregate."
    )
    lines.append("")

    lines.append("## Data files")
    lines.append("")
    lines.append("- `expanded_gate_c_per_query.csv` -- every recorded field per query.")
    lines.append("- `expanded_gate_c_by_regime.csv` -- regime-level decomposition-label aggregates.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    out_per_query = OUTPUT_DIR / "expanded_gate_c_per_query.csv"
    out_by_regime = OUTPUT_DIR / "expanded_gate_c_by_regime.csv"
    out_report = OUTPUT_DIR / "EXPANDED_GATE_C_REPORT.md"
    _check_no_overwrite([out_per_query, out_by_regime, out_report])

    rows = run_expanded_gate_c()
    summary = build_summary(rows)
    regime_decomp = build_regime_decomposition(rows)
    count_error = build_count_error_analysis(rows)
    pp_assoc = build_top_pair_pp_association(rows)

    out_per_query.parent.mkdir(parents=True, exist_ok=True)
    with open(out_per_query, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    regime_fieldnames = sorted({k for agg in regime_decomp for k in agg.keys()})
    with open(out_by_regime, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=regime_fieldnames)
        writer.writeheader()
        for agg in regime_decomp:
            writer.writerow(agg)

    write_report(rows, summary, regime_decomp, count_error, pp_assoc, out_report)

    print(f"Expanded Gate C complete: {len(rows)} queries evaluated.")
    print(f"Wrote: {out_per_query}")
    print(f"Wrote: {out_by_regime}")
    print(f"Wrote: {out_report}")


if __name__ == "__main__":
    main()
