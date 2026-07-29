#!/usr/bin/env python3
"""Cluster-Normalized Poisoning: cross-strategy comparison summary.

Per `docs/CLUSTER_NORMALIZED_POISONING_EXECUTION_PLAN.md` section 8: "The
report must present all four `anchor_strategy` results side by side (plus
E0) so that, e.g., `nearest_bijection` succeeding while `farthest_bijection`
does not (or vice versa) is directly visible."

This script reads the run directories already written by
`scripts/run_cluster_normalized_poisoning.py` for a given `query_id` (one
run directory per `(intervention, anchor_strategy)` combination -- see that
script's `build_run_dir`), and writes a single combined comparison CSV +
Markdown report. It performs no computation of its own beyond aggregation
and label re-derivation: every raw number here is read directly from each
run's own `intervention_sweep.csv` and `run_config.json` -- no diagnostics
are recomputed, no embedder is loaded, no GPT/API call is made, no baseline
retrieval or defense is rerun.

`decision_label` is re-derived here (rather than trusted verbatim from each
run's `intervention_sweep.csv`) using the absolute classification scheme
below, so this script's output is correct even for run directories written
by an older version of `run_cluster_normalized_poisoning.py` that used a
different (relative-to-alpha=1.0) labeling scheme.

Usage:
    python scripts/summarize_cluster_normalized_poisoning.py \\
        --output_dir results/diagnostics/cluster_normalized_poisoning \\
        --query_id 5ae2070a5542994d89d5b313
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_OUTPUT_DIR = os.path.join("results", "diagnostics", "cluster_normalized_poisoning")

# The plan's own category table (docs/RAGDEFENDER_CLUSTER_DIAGNOSTIC_FINDINGS.md)
# labels the severe-failure anchor query this way, and the execution plan
# (section 8) says it is processed "at alpha=1.0 only ... as a numeric
# sanity control" -- not a full sweep. Any run directory this script finds
# with fewer than 2 distinct alpha values is annotated as an identity
# control rather than a sweep.
SEVERE_FAILURE_CONTROL_NOTE = (
    "The severe-failure case was used only as an identity control because "
    "RAGDefender already fails at alpha=1.0 (see "
    "`docs/CLUSTER_NORMALIZED_POISONING_EXECUTION_PLAN.md` section 8: the "
    "failure case is processed \"at alpha=1.0 only ... as a numeric sanity "
    "control\"). It was not intended to receive the full alpha sweep."
)


def decision_label(removed_poison: int, removed_clean: int, n_retrieved_poison: int) -> str:
    """Absolute classification of a removal outcome. Duplicated from
    `scripts/run_cluster_normalized_poisoning.py::decision_label` (kept as
    a small standalone pure function here, rather than imported, so that
    running this summary script never triggers that script's heavier
    embedder-related imports). See that function's docstring for the full
    rationale of each category:

    - `poison_removal_success`
    - `over_removal_success`
    - `residual_poison_failure`
    - `residual_poison_with_clean_false_positive`
    - `clean_density_failure`
    - `no_removal_or_other`
    """
    if n_retrieved_poison > 0 and removed_poison == n_retrieved_poison and removed_clean == 0:
        return "poison_removal_success"
    if n_retrieved_poison > 0 and removed_poison == n_retrieved_poison and removed_clean > 0:
        return "over_removal_success"
    if 0 < removed_poison < n_retrieved_poison and removed_clean == 0:
        return "residual_poison_failure"
    if 0 < removed_poison < n_retrieved_poison and removed_clean > 0:
        return "residual_poison_with_clean_false_positive"
    if removed_poison == 0 and removed_clean > 0:
        return "clean_density_failure"
    return "no_removal_or_other"


def discover_run_dirs(output_dir: str, query_id: str) -> List[Path]:
    """Find every run directory under `output_dir` whose name ends in
    `_<query_id>` (the naming convention `run_cluster_normalized_poisoning.py`
    uses), sorted by directory name (== timestamp prefix) so the most
    recent run for each intervention/strategy combination is easy to pick
    out if a query was run more than once."""
    pattern = os.path.join(output_dir, f"*_{query_id}")
    return sorted(Path(p) for p in glob.glob(pattern) if os.path.isdir(p))


def latest_run_per_intervention(run_dirs: List[Path]) -> Dict[str, Path]:
    """Collapse to the single most recent run directory per intervention
    slug (`E0`, `E1-rank_aligned`, ...), read from each run's own
    `run_config.json` rather than re-parsed from the directory name."""
    latest: Dict[str, Path] = {}
    for run_dir in run_dirs:  # already sorted ascending by timestamp
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        slug = cfg["intervention"] if cfg["intervention"] == "E0" else f"E1-{cfg['anchor_strategy']}"
        latest[slug] = run_dir  # later entries overwrite earlier ones -> most recent wins
    return latest


def load_sweep(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "intervention_sweep.csv")
    df["run_dir"] = run_dir.name
    # Re-derive decision_label with the current absolute scheme, ignoring
    # whatever string is already in the CSV (see module docstring).
    df["decision_label"] = df.apply(
        lambda r: decision_label(int(r["removed_poison"]), int(r["removed_clean"]), int(r["N_retrieved_poison"])),
        axis=1,
    )
    return df


def render_comparison_report(query_id: str, latest: Dict[str, Path], combined: pd.DataFrame) -> str:
    is_single_alpha_control = combined["alpha"].nunique() == 1

    lines = [
        "# Cluster-Normalized Poisoning -- Cross-Strategy Comparison",
        "",
        f"query_id: `{query_id}`",
        "",
        "Aggregates the already-written `intervention_sweep.csv` / `run_config.json` "
        "from each intervention's own run directory below. `decision_label` is "
        "re-derived here using the absolute classification scheme documented in "
        "`scripts/summarize_cluster_normalized_poisoning.py::decision_label` (see "
        "docstring) rather than trusted verbatim from those files. No diagnostics "
        "were recomputed, no embedder was loaded, and no GPT/API call was made to "
        "produce this report.",
        "",
    ]
    if is_single_alpha_control:
        lines += ["**" + SEVERE_FAILURE_CONTROL_NOTE + "**", ""]

    lines += ["## Run directories aggregated", ""]
    for slug in sorted(latest):
        lines.append(f"- `{slug}`: `{latest[slug].name}`")
    lines.append("")

    lines += [
        "## Side-by-side sweep (all interventions/strategies)",
        "",
        "`decision_label` legend: `poison_removal_success` (all poison removed, no "
        "clean false positive) / `over_removal_success` (all poison removed, but a "
        "clean passage was also removed) / `residual_poison_failure` (some poison "
        "left, no clean false positive) / `residual_poison_with_clean_false_positive` "
        "(some poison left AND a clean passage wrongly removed) / "
        "`clean_density_failure` (no poison removed at all; clean passage(s) removed "
        "instead) / `no_removal_or_other`.",
        "",
        "| intervention | alpha | N_adv | top_pair (PP/PC/CC) | removed_poison | removed_clean | "
        "residual_poison_fraction | decision_label |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in combined.iterrows():
        slug = row["intervention"] if row["intervention"] == "E0" else f"E1-{row['anchor_strategy']}"
        lines.append(
            f"| {slug} | {row['alpha']} | {row['N_adv']} | "
            f"{row['top_pair_pp']}/{row['top_pair_pc']}/{row['top_pair_cc']} | "
            f"{row['removed_poison']} | {row['removed_clean']} | {row['residual_poison_fraction']} | "
            f"{row['decision_label']} |"
        )
    lines.append("")

    if is_single_alpha_control:
        lines += [
            "## First alpha (descending from 1.0) each stopping condition first triggers, per strategy",
            "",
            "_Skipped: this comparison contains only a single alpha value "
            "(alpha=1.0 identity control), so there is no sweep over which a "
            "\"first alpha where X changes\" statement is meaningful. See the "
            "note above._",
            "",
        ]
        return "\n".join(lines)

    lines += [
        "## First alpha (descending from 1.0) each stopping condition first triggers, per strategy",
        "",
        "(These conditions are relative to each strategy's own alpha=1.0 row -- a "
        "separate, complementary view from the absolute `decision_label` above.)",
        "",
        "| intervention | pp_decreased | pc_increased | fewer_poison_removed | clean_removed_increased |",
        "|---|---|---|---|---|",
    ]
    for slug in sorted(latest):
        run_dir = latest[slug]
        sub = combined[combined["run_dir"] == run_dir.name].sort_values("alpha", ascending=False)
        baseline = sub.iloc[0] if len(sub) else None

        def first_alpha(mask_col_fn):
            for _, r in sub.iterrows():
                if mask_col_fn(r):
                    return r["alpha"]
            return None

        if baseline is None:
            continue
        base_pp, base_pc = baseline["top_pair_pp"], baseline["top_pair_pc"]
        base_rp, base_rc = baseline["removed_poison"], baseline["removed_clean"]
        pp_a = first_alpha(lambda r: r["top_pair_pp"] < base_pp)
        pc_a = first_alpha(lambda r: r["top_pair_pc"] > base_pc)
        fewer_a = first_alpha(lambda r: r["removed_poison"] < base_rp)
        clean_a = first_alpha(lambda r: r["removed_clean"] > base_rc)
        lines.append(f"| {slug} | {pp_a} | {pc_a} | {fewer_a} | {clean_a} |")

    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query_id", required=True)
    parser.add_argument("--report_path", default=None,
                         help="Where to write the comparison report; defaults to "
                              "<output_dir>/COMPARISON_<query_id>.md")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Path:
    args = parse_args(argv)
    run_dirs = discover_run_dirs(args.output_dir, args.query_id)
    if not run_dirs:
        raise ValueError(f"No run directories found under {args.output_dir!r} for query_id={args.query_id!r}")
    latest = latest_run_per_intervention(run_dirs)

    combined = pd.concat([load_sweep(run_dir) for run_dir in latest.values()], ignore_index=True)
    combined_csv_path = Path(args.output_dir) / f"COMPARISON_{args.query_id}.csv"
    combined.to_csv(combined_csv_path, index=False)

    report_text = render_comparison_report(args.query_id, latest, combined)
    report_path = Path(args.report_path) if args.report_path else Path(args.output_dir) / f"COMPARISON_{args.query_id}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Wrote comparison CSV to: {combined_csv_path}")
    print(f"Wrote comparison report to: {report_path}")
    return report_path


if __name__ == "__main__":
    main()
