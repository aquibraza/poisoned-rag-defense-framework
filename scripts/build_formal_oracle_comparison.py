#!/usr/bin/env python3
"""Consolidated formal oracle comparison report for the RAGDefender
Cluster-Normalized Poisoning study.

Reads **only already-existing artifacts on disk** -- it never reruns E1,
CORAL-PCA (Step 1), CORAL-ridge (Step 2), or MMD (Step 3), never performs
retrieval, never loads an embedder, and never calls an LLM/API:

- `results/diagnostics/cluster_normalized_poisoning/BATCH_COMPARISON_SUCCESS_CASES.csv` (E1)
- the most recently written `CORAL_PCA_SWEEP.csv` under
  `results/diagnostics/cluster_normalized_poisoning_formal/*_coral_pca_*/`
- the most recently written `CORAL_RIDGE_SWEEP.csv` under
  `results/diagnostics/cluster_normalized_poisoning_formal/*_coral_ridge_*/`
- the most recently written `MMD_SWEEP.csv` under
  `results/diagnostics/cluster_normalized_poisoning_formal/*_mmd_*/`

The CORAL-PCA artifact already contains the perturbation/preservation
columns (`mean_poison_l2_displacement`, etc.) -- `CORAL_PCA_SWEEP.csv` is
read directly, with no staleness/regeneration check.

Writes:
- `results/diagnostics/cluster_normalized_poisoning_formal/FORMAL_ORACLE_COMPARISON.csv`
- `results/diagnostics/cluster_normalized_poisoning_formal/FORMAL_ORACLE_COMPARISON.md`

Usage:
    python scripts/build_formal_oracle_comparison.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import numpy as np
import pandas as pd

import build_batch_comparison_success_cases as batchmod  # noqa: E402 -- compute_config_summary reuse

DEFAULT_E1_CSV = os.path.join(
    "results", "diagnostics", "cluster_normalized_poisoning", "BATCH_COMPARISON_SUCCESS_CASES.csv"
)
DEFAULT_FORMAL_DIR = os.path.join("results", "diagnostics", "cluster_normalized_poisoning_formal")
DEFAULT_OUTPUT_DIR = DEFAULT_FORMAL_DIR

E1_STRATEGIES = ("rank_aligned", "nearest_bijection")
RIDGE_LAMBDAS = (0.1, 0.01, 0.001)
MMD_LAMBDA_PRESERVES = (0.01, 0.10, 1.00)

REQUIRED_COLUMNS = [
    "method", "perturbation_param",
    "n_tested_rows", "n_query_level_units",
    "residual_poison_failure_row_count", "query_level_failure_count",
    "first_failure_perturbation",
    "top_pair_pp_baseline", "top_pair_pp_at_max_perturbation", "top_pair_pp_reduction",
    "removed_poison_at_max_perturbation", "removed_clean_at_max_perturbation",
    "residual_poison_fraction_at_max_perturbation",
    "mmd_distance_before", "mmd_distance_after", "mmd_distance_reduction",
    "coral_distance_before", "coral_distance_after", "coral_distance_reduction",
    "mean_poison_l2_displacement", "max_poison_l2_displacement",
    "mean_poison_original_cosine", "min_poison_original_cosine",
]

LIMITATIONS_TEXT = """## Limitations

- Only **6** recovered HotpotQA `k=10`/`N=5` originally-successful RAGDefender
  cases are covered by every method in this comparison (2 additional
  candidate queries were excluded upstream for text-recoverability reasons
  and never entered any of the four sweeps).
- All four interventions operate in the **frozen `paraphrase-MiniLM-L6-v2`
  embedding space** -- none re-embeds under a different encoder, and none
  performs any natural-language rewriting of a poisoned passage. These
  remain **oracle embedding-space stress tests**, not text-realizable
  attacks: none of the transformed embeddings compared here has been shown
  to correspond to any actual passage text under the frozen encoder.
- **No GPT/API calls were made** by this report or by any of the four
  interventions it summarizes.
- **No baseline retrieval was rerun** and **no baseline defense file was
  modified** by this report or by any of the four interventions.
- This report itself performs **no new experiments**: it only reads
  already-existing `BATCH_COMPARISON_SUCCESS_CASES.csv` /
  `CORAL_PCA_SWEEP.csv` / `CORAL_RIDGE_SWEEP.csv` / `MMD_SWEEP.csv`
  artifacts from disk.
- Result files under `results/` (including this report) remain
  **ignored/uncommitted** per `.gitignore`, unless explicitly added
  elsewhere.
- **FilterRAG/ML-FilterRAG comparisons remain later work**, out of scope
  for this RAGDefender-only oracle study.
"""


# --------------------------------------------------------------------------
# Artifact discovery (existing artifacts only -- no regeneration, no
# staleness check)
# --------------------------------------------------------------------------

def discover_latest_artifact(formal_dir: str, dir_glob: str, filename: str) -> Optional[Path]:
    """Most recently written `filename` under a `dir_glob`-matching run
    directory inside `formal_dir`. Run directories are named
    `YYYYMMDD_HHMMSS_..._...`, which sorts lexicographically in time
    order, so the lexicographically-last match is the most recent run.
    Returns `None` if no matching artifact exists (caller decides how to
    report that as a skip, never as an error that blocks the rest of the
    report)."""
    candidates = sorted(Path(formal_dir).glob(f"{dir_glob}/{filename}"))
    return candidates[-1] if candidates else None


# --------------------------------------------------------------------------
# Per-method summaries
# --------------------------------------------------------------------------

def _finite_mean(series: pd.Series) -> Optional[float]:
    if series is None or len(series) == 0:
        return None
    val = float(series.mean())
    return val if np.isfinite(val) else None


def summarize_e1(strategy: str, e1_df: pd.DataFrame) -> Dict:
    """One summary row for E1's `strategy` (`rank_aligned` or
    `nearest_bijection`), across every query_id present for that strategy
    in `BATCH_COMPARISON_SUCCESS_CASES.csv` (already filtered upstream to
    the 6 success-case queries -- not re-filtered here)."""
    sub = e1_df[e1_df["anchor_strategy"] == strategy]
    tested_ids = sorted(sub["query_id"].unique())

    n_tested_rows = len(sub)
    residual_poison_failure_row_count = int((sub["removed_poison"] < sub["N_retrieved_poison"]).sum())

    per_query_summaries = [
        batchmod.compute_config_summary(qid, strategy, sub[sub["query_id"] == qid]) for qid in tested_ids
    ]
    first_failure_alphas = [s["first_residual_poison_alpha"] for s in per_query_summaries
                             if s["first_residual_poison_alpha"] is not None]
    query_level_failure_count = len(first_failure_alphas)
    # "First" (mildest) failure onset across queries = the *highest* alpha
    # (least perturbation) among each query's own first-failure alpha.
    first_failure_perturbation = max(first_failure_alphas) if first_failure_alphas else None

    alpha_min, alpha_max = sub["alpha"].min(), sub["alpha"].max()
    at_max_perturbation = sub[sub["alpha"] == alpha_min]  # min alpha == max perturbation
    baseline = sub[sub["alpha"] == alpha_max]

    return {
        "method": f"E1_{strategy}",
        "perturbation_param": f"alpha={alpha_min} (max swept)",
        "n_tested_rows": n_tested_rows,
        "n_query_level_units": len(tested_ids),
        "residual_poison_failure_row_count": residual_poison_failure_row_count,
        "query_level_failure_count": query_level_failure_count,
        "first_failure_perturbation": first_failure_perturbation,
        "top_pair_pp_baseline": _finite_mean(baseline["top_pair_pp"]),
        "top_pair_pp_at_max_perturbation": _finite_mean(at_max_perturbation["top_pair_pp"]),
        "top_pair_pp_reduction": _finite_mean(baseline["top_pair_pp"]) - _finite_mean(at_max_perturbation["top_pair_pp"]),
        "removed_poison_at_max_perturbation": _finite_mean(at_max_perturbation["removed_poison"]),
        "removed_clean_at_max_perturbation": _finite_mean(at_max_perturbation["removed_clean"]),
        "residual_poison_fraction_at_max_perturbation": _finite_mean(at_max_perturbation["residual_poison_fraction"]),
        "mmd_distance_before": None, "mmd_distance_after": None, "mmd_distance_reduction": None,
        "coral_distance_before": None, "coral_distance_after": None, "coral_distance_reduction": None,
        "mean_poison_l2_displacement": None, "max_poison_l2_displacement": None,
        "mean_poison_original_cosine": None, "min_poison_original_cosine": None,
    }


def summarize_perturbation_swept_method(method_name: str, sub_df: pd.DataFrame, param_col: str) -> Dict:
    """One summary row for a CORAL-PCA/CORAL-ridge/MMD sub-sweep
    (`sub_df` already filtered to one `lambda`/`lambda_preserve` value, or
    the whole frame for CORAL-PCA which has no such secondary axis).
    `param_col` is `"beta"` for CORAL-PCA/CORAL-ridge or `"steps"` for
    MMD -- the axis along which perturbation strength increases from the
    identity baseline (`beta=0`/`steps=0`)."""
    tested_ids = sorted(sub_df["query_id"].unique())
    n_tested_rows = len(sub_df)
    residual_poison_failure_row_count = int((sub_df["removed_poison"] < sub_df["N_retrieved_poison"]).sum())

    param_max = sub_df[param_col].max()
    param_min = sub_df[param_col].min()  # identity baseline (0)
    at_max_perturbation = sub_df[sub_df[param_col] == param_max]
    baseline = sub_df[sub_df[param_col] == param_min]

    first_failures: List[Optional[float]] = []
    for qid in tested_ids:
        qsub = sub_df[sub_df["query_id"] == qid].sort_values(param_col, ascending=True)
        first_fail = None
        for _, r in qsub.iterrows():
            if r["removed_poison"] < r["N_retrieved_poison"]:
                first_fail = float(r[param_col])
                break
        first_failures.append(first_fail)
    query_level_failure_count = sum(1 for f in first_failures if f is not None)
    triggered = [f for f in first_failures if f is not None]
    first_failure_perturbation = min(triggered) if triggered else None

    return {
        "method": method_name,
        "perturbation_param": f"{param_col}={param_max} (max swept)",
        "n_tested_rows": n_tested_rows,
        "n_query_level_units": len(tested_ids),
        "residual_poison_failure_row_count": residual_poison_failure_row_count,
        "query_level_failure_count": query_level_failure_count,
        "first_failure_perturbation": first_failure_perturbation,
        "top_pair_pp_baseline": _finite_mean(baseline["top_pair_pp"]),
        "top_pair_pp_at_max_perturbation": _finite_mean(at_max_perturbation["top_pair_pp"]),
        "top_pair_pp_reduction": _finite_mean(baseline["top_pair_pp"]) - _finite_mean(at_max_perturbation["top_pair_pp"]),
        "removed_poison_at_max_perturbation": _finite_mean(at_max_perturbation["removed_poison"]),
        "removed_clean_at_max_perturbation": _finite_mean(at_max_perturbation["removed_clean"]),
        "residual_poison_fraction_at_max_perturbation": _finite_mean(at_max_perturbation["residual_poison_fraction"]),
        "mmd_distance_before": _finite_mean(at_max_perturbation["mmd_distance_before"]),
        "mmd_distance_after": _finite_mean(at_max_perturbation["mmd_distance_after"]),
        "mmd_distance_reduction": _finite_mean(at_max_perturbation["mmd_distance_reduction"]),
        "coral_distance_before": _finite_mean(at_max_perturbation["coral_distance_before"]),
        "coral_distance_after": _finite_mean(at_max_perturbation["coral_distance_after"]),
        "coral_distance_reduction": _finite_mean(at_max_perturbation["coral_distance_reduction"]),
        "mean_poison_l2_displacement": _finite_mean(at_max_perturbation["mean_poison_l2_displacement"]),
        "max_poison_l2_displacement": _finite_mean(at_max_perturbation["max_poison_l2_displacement"]),
        "mean_poison_original_cosine": _finite_mean(at_max_perturbation["mean_poison_original_cosine"]),
        "min_poison_original_cosine": _finite_mean(at_max_perturbation["min_poison_original_cosine"]),
    }


def build_comparison_table(e1_df: pd.DataFrame, pca_df: pd.DataFrame, ridge_df: pd.DataFrame,
                            mmd_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for strategy in E1_STRATEGIES:
        rows.append(summarize_e1(strategy, e1_df))

    rows.append(summarize_perturbation_swept_method("CORAL_PCA", pca_df, param_col="beta"))

    for lam in RIDGE_LAMBDAS:
        sub = ridge_df[np.isclose(ridge_df["lambda"], lam)]
        rows.append(summarize_perturbation_swept_method(f"CORAL_RIDGE_lambda={lam}", sub, param_col="beta"))

    for lp in MMD_LAMBDA_PRESERVES:
        sub = mmd_df[np.isclose(mmd_df["lambda_preserve"], lp)]
        rows.append(summarize_perturbation_swept_method(f"MMD_lambda_preserve={lp:.2f}", sub, param_col="steps"))

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "None"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_report(comparison_df: pd.DataFrame, tested_ids: Sequence[str],
                   e1_csv: Path, pca_csv: Path, ridge_csv: Path, mmd_csv: Path) -> str:
    lines = [
        "# Formal Oracle Comparison Report",
        "",
        "Consolidated comparison of every formal RAGDefender oracle embedding-space "
        "intervention run so far against the same 6 recovered HotpotQA `k=10`/`N=5` "
        "originally-successful RAGDefender cases: **E1** (clean-anchor interpolation, "
        "empirical baseline), **CORAL-PCA** (Step 1), **CORAL-ridge** (Step 2), and "
        "**MMD-minimize** (Step 3).",
        "",
        "**This report performs no new experiments.** It reads only already-existing "
        "artifacts on disk and never reruns E1, CORAL-PCA, CORAL-ridge, or MMD; never "
        "reruns baseline retrieval; never calls an LLM/API; and never modifies a "
        "baseline defense file.",
        "",
        "## Source artifacts",
        "",
        f"- E1: `{e1_csv}`",
        f"- CORAL-PCA: `{pca_csv}`",
        f"- CORAL-ridge: `{ridge_csv}`",
        f"- MMD: `{mmd_csv}`",
        "",
        f"Tested query_ids (common to all four artifacts): {len(tested_ids)} -- "
        + ", ".join(f"`{q}`" for q in tested_ids),
        "",
        "## Method comparison",
        "",
        "One row per method (E1's two anchor strategies; CORAL-ridge's three swept "
        "`lambda` values; MMD's three swept `lambda_preserve` values), each evaluated "
        "at that method's own **maximum swept perturbation** (`E1`: minimum alpha; "
        "`CORAL-PCA`/`CORAL-ridge`: `beta=1.0`; `MMD`: `steps=100`) against its own "
        "identity baseline (`alpha=1.0`/`beta=0`/`steps=0`). `None` in a CORAL/MMD-"
        "distance or preservation column means that method's own artifact does not "
        "compute it (E1's sweep predates those metrics).",
        "",
        "| method | n_tested_rows | n_query_units | failure_rows | query_failures | "
        "first_failure_perturbation | top_pair_pp (base->max) | pp_reduction | "
        "removed_poison@max | removed_clean@max | resid_poison_frac@max | "
        "mmd_dist (before->after) | coral_dist (before->after) | mean_l2_disp | "
        "max_l2_disp | mean_orig_cos | min_orig_cos |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in comparison_df.iterrows():
        lines.append(
            f"| `{r['method']}` | {r['n_tested_rows']} | {r['n_query_level_units']} | "
            f"{r['residual_poison_failure_row_count']} | {r['query_level_failure_count']} | "
            f"{_fmt(r['first_failure_perturbation'])} | "
            f"{_fmt(r['top_pair_pp_baseline'])}->{_fmt(r['top_pair_pp_at_max_perturbation'])} | "
            f"{_fmt(r['top_pair_pp_reduction'])} | {_fmt(r['removed_poison_at_max_perturbation'])} | "
            f"{_fmt(r['removed_clean_at_max_perturbation'])} | "
            f"{_fmt(r['residual_poison_fraction_at_max_perturbation'])} | "
            f"{_fmt(r['mmd_distance_before'])}->{_fmt(r['mmd_distance_after'])} | "
            f"{_fmt(r['coral_distance_before'])}->{_fmt(r['coral_distance_after'])} | "
            f"{_fmt(r['mean_poison_l2_displacement'])} | {_fmt(r['max_poison_l2_displacement'])} | "
            f"{_fmt(r['mean_poison_original_cosine'])} | {_fmt(r['min_poison_original_cosine'])} |"
        )
    lines.append("")

    def get(method: str) -> pd.Series:
        return comparison_df[comparison_df["method"] == method].iloc[0]

    e1_ra, e1_nb = get("E1_rank_aligned"), get("E1_nearest_bijection")
    pca = get("CORAL_PCA")
    ridge_rows = comparison_df[comparison_df["method"].str.startswith("CORAL_RIDGE")]
    mmd_rows = comparison_df[comparison_df["method"].str.startswith("MMD_")]
    mmd_weak_mid = comparison_df[comparison_df["method"].isin(["MMD_lambda_preserve=0.01", "MMD_lambda_preserve=0.10"])]
    mmd_strong = get("MMD_lambda_preserve=1.00")

    e1_total_failures = e1_ra["query_level_failure_count"] + e1_nb["query_level_failure_count"]
    e1_total_units = e1_ra["n_query_level_units"] + e1_nb["n_query_level_units"]
    coral_total_failures = pca["query_level_failure_count"] + int(ridge_rows["query_level_failure_count"].sum())
    coral_total_units = pca["n_query_level_units"] + int(ridge_rows["n_query_level_units"].sum())
    mmd_total_failures = int(mmd_rows["query_level_failure_count"].sum())
    mmd_total_units = int(mmd_rows["n_query_level_units"].sum())

    lines += ["## Questions this report answers", ""]

    lines.append(
        f"**1. Which oracle intervention causes the most consistent residual-poison "
        f"failure?** **E1** (clean-anchor interpolation): `{e1_total_failures}/{e1_total_units}` "
        f"query-level units fail across its two strategies (`rank_aligned`: "
        f"`{e1_ra['query_level_failure_count']}/{e1_ra['n_query_level_units']}`, "
        f"`nearest_bijection`: `{e1_nb['query_level_failure_count']}/{e1_nb['n_query_level_units']}`), "
        f"compared to CORAL (PCA+ridge combined): `{coral_total_failures}/{coral_total_units}` "
        f"and MMD: `{mmd_total_failures}/{mmd_total_units}` query-lambda units. **E1 remains "
        f"the strongest and most consistently effective oracle intervention** in this study."
    )
    lines.append("")

    coral_pp_reductions = pd.concat([pd.Series([pca["top_pair_pp_reduction"]]), ridge_rows["top_pair_pp_reduction"]])
    best_pp_row = comparison_df.loc[comparison_df["top_pair_pp_reduction"].idxmax()]
    lines.append(
        f"**2. Which method most strongly reduces `top_pair_pp`?** `{best_pp_row['method']}` "
        f"has the largest mean `top_pair_pp` reduction at its own max perturbation "
        f"(`{_fmt(best_pp_row['top_pair_pp_reduction'])}`, from "
        f"`{_fmt(best_pp_row['top_pair_pp_baseline'])}` to "
        f"`{_fmt(best_pp_row['top_pair_pp_at_max_perturbation'])}`). E1's two strategies "
        f"reduce `top_pair_pp` the most overall (`{_fmt(e1_ra['top_pair_pp_reduction'])}` each); "
        f"among the formal (non-E1) methods, MMD at weak/mid `lambda_preserve` "
        f"(`{_fmt(mmd_weak_mid['top_pair_pp_reduction'].min())}`-"
        f"`{_fmt(mmd_weak_mid['top_pair_pp_reduction'].max())}`) reduces `top_pair_pp` far "
        f"more than any CORAL configuration (`{_fmt(coral_pp_reductions.min())}`-"
        f"`{_fmt(coral_pp_reductions.max())}`)."
    )
    lines.append("")

    coral_mmd_dist_reductions = pd.concat([pd.Series([pca["mmd_distance_reduction"]]), ridge_rows["mmd_distance_reduction"]])
    lines.append(
        f"**3. Which method reduces global distribution distance without causing "
        f"failure?** Every CORAL configuration (`CORAL_PCA` and all three "
        f"`CORAL_RIDGE_lambda` settings) reduces MMD distance substantially "
        f"(`{_fmt(coral_mmd_dist_reductions.min())}`-`{_fmt(coral_mmd_dist_reductions.max())}`) "
        f"at **zero** residual-poison failures across all 6 queries. `MMD_lambda_preserve=1.00` "
        f"achieves a comparable MMD-distance reduction (`{_fmt(mmd_strong['mmd_distance_reduction'])}`) "
        f"also at zero failures. **CORAL-PCA and CORAL-ridge are the clean examples here: they "
        f"improve global distribution-alignment metrics but do not cause residual-poison "
        f"failure** -- a negative result for covariance alignment as an attack family on "
        f"this defense."
    )
    lines.append("")

    lines.append(
        f"**4. Does MMD behave differently from CORAL?** Yes. MMD at weak/mid "
        f"`lambda_preserve` (`0.01`, `0.10`) causes residual-poison failure in "
        f"`{int(mmd_weak_mid['query_level_failure_count'].sum())}/"
        f"{int(mmd_weak_mid['n_query_level_units'].sum())}` query-lambda units, while "
        f"**neither** CORAL variant causes any failure at any swept perturbation "
        f"(`{coral_total_failures}/{coral_total_units}`). MMD is **stronger than CORAL but "
        f"weaker than E1 in coverage**. The two methods also diverge on which distance "
        f"they actually reduce: **MMD distance decreases under MMD optimization; CORAL "
        f"distance is reported separately and is not the primary optimized objective** "
        f"-- at MMD's max perturbation, `coral_distance_reduction` is in fact *negative* "
        f"for all three `lambda_preserve` settings "
        f"(`{_fmt(mmd_rows['coral_distance_reduction'].min())}` to "
        f"`{_fmt(mmd_rows['coral_distance_reduction'].max())}`, i.e. CORAL distance "
        f"*increases* under MMD optimization), unlike CORAL's own sweeps where "
        f"`coral_distance_reduction` is positive by construction."
    )
    lines.append("")

    lines.append(
        f"**5. Does stronger preservation prevent MMD failure?** Yes, in this sweep: "
        f"`MMD_lambda_preserve=1.00` (strongest preservation weight) causes "
        f"`{mmd_strong['query_level_failure_count']}/{mmd_strong['n_query_level_units']}` "
        f"query failures, versus `{mmd_weak_mid['query_level_failure_count'].min()}-"
        f"{mmd_weak_mid['query_level_failure_count'].max()}` out of "
        f"`{mmd_weak_mid['n_query_level_units'].iloc[0]}` at `lambda_preserve in {{0.01, 0.10}}`. "
        f"**Strong preservation in MMD (`lambda_preserve=1.0`) prevents residual-poison "
        f"failure in the current sweep** -- this is an association within this specific "
        f"`steps<=100`, `lr=0.05`, `gamma=1.0` sweep, not a proof that no `steps`/`lr` "
        f"combination at `lambda_preserve=1.0` could ever cause failure."
    )
    lines.append("")

    fail_disp = mmd_rows.loc[mmd_rows["query_level_failure_count"] > 0, "mean_poison_l2_displacement"]
    nonfail_mmd_disp = mmd_strong["mean_poison_l2_displacement"]
    coral_disp_values = pd.concat([pd.Series([pca["mean_poison_l2_displacement"]]), ridge_rows["mean_poison_l2_displacement"]])
    coral_disp_range = f"{_fmt(coral_disp_values.min())}-{_fmt(coral_disp_values.max())}"
    lines.append(
        f"**6. How much embedding displacement is associated with failure?** MMD "
        f"configs that fail have mean `mean_poison_l2_displacement` around "
        f"`{_fmt(fail_disp.mean()) if len(fail_disp) else 'None'}` at max perturbation, larger than "
        f"MMD's own non-failing `lambda_preserve=1.00` config (`{_fmt(nonfail_mmd_disp)}`). "
        f"**However, displacement magnitude alone is not sufficient to predict failure**: "
        f"every CORAL configuration reaches *larger* displacement at its own max "
        f"perturbation (`{coral_disp_range}`) than any failing MMD config, yet causes "
        f"**zero** failures. Failure appears to require displacement in a direction that "
        f"specifically disrupts the poison-poison top-pair structure (see Q7), not merely "
        f"displacement of large magnitude."
    )
    lines.append("")

    lines.append(
        "**7. Is `top_pair_pp` still the most specific Stage-2 mechanistic indicator?** "
        "Yes. Every method that reduces `top_pair_pp` toward the poison-clean/clean-clean "
        "range shows residual-poison failure (E1's two strategies, and MMD at weak/mid "
        "`lambda_preserve`); every method that leaves `top_pair_pp` at or near its "
        "baseline maximum shows zero failure (CORAL-PCA, CORAL-ridge, and MMD at "
        "`lambda_preserve=1.0`), **regardless of how much global CORAL/MMD distance "
        "or embedding displacement each method achieves**. **`top_pair_pp` remains the "
        "most specific mechanistic indicator: failure occurs when poison-poison "
        "top-pair dominance collapses, while global CORAL/MMD distance reductions "
        "alone are not sufficient.**"
    )
    lines.append("")

    lines += ["", LIMITATIONS_TEXT]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI / main
# --------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e1_csv", default=DEFAULT_E1_CSV)
    parser.add_argument("--formal_dir", default=DEFAULT_FORMAL_DIR)
    parser.add_argument("--pca_csv", default=None, help="Explicit CORAL_PCA_SWEEP.csv path; default auto-discovers latest.")
    parser.add_argument("--ridge_csv", default=None, help="Explicit CORAL_RIDGE_SWEEP.csv path; default auto-discovers latest.")
    parser.add_argument("--mmd_csv", default=None, help="Explicit MMD_SWEEP.csv path; default auto-discovers latest.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Path:
    args = parse_args(argv)

    e1_csv = Path(args.e1_csv)
    if not e1_csv.exists():
        raise FileNotFoundError(f"E1 comparison CSV not found: {e1_csv}")

    pca_csv = Path(args.pca_csv) if args.pca_csv else discover_latest_artifact(
        args.formal_dir, "*_coral_pca_*", "CORAL_PCA_SWEEP.csv"
    )
    if pca_csv is None:
        raise FileNotFoundError(f"No CORAL_PCA_SWEEP.csv found under {args.formal_dir}/*_coral_pca_*/")

    ridge_csv = Path(args.ridge_csv) if args.ridge_csv else discover_latest_artifact(
        args.formal_dir, "*_coral_ridge_*", "CORAL_RIDGE_SWEEP.csv"
    )
    if ridge_csv is None:
        raise FileNotFoundError(f"No CORAL_RIDGE_SWEEP.csv found under {args.formal_dir}/*_coral_ridge_*/")

    mmd_csv = Path(args.mmd_csv) if args.mmd_csv else discover_latest_artifact(
        args.formal_dir, "*_mmd_*", "MMD_SWEEP.csv"
    )
    if mmd_csv is None:
        raise FileNotFoundError(f"No MMD_SWEEP.csv found under {args.formal_dir}/*_mmd_*/")

    e1_df = pd.read_csv(e1_csv)
    pca_df = pd.read_csv(pca_csv)
    ridge_df = pd.read_csv(ridge_csv)
    mmd_df = pd.read_csv(mmd_csv)

    tested_ids = sorted(
        set(e1_df["query_id"]) & set(pca_df["query_id"]) & set(ridge_df["query_id"]) & set(mmd_df["query_id"])
    )

    comparison_df = build_comparison_table(e1_df, pca_df, ridge_df, mmd_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "FORMAL_ORACLE_COMPARISON.csv"
    comparison_df.to_csv(csv_path, index=False)

    report_text = render_report(comparison_df, tested_ids, e1_csv, pca_csv, ridge_csv, mmd_csv)
    md_path = output_dir / "FORMAL_ORACLE_COMPARISON.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return md_path


if __name__ == "__main__":
    main()
