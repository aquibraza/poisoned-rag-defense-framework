#!/usr/bin/env python3
"""Summarize RAGDefender diagnostic JSONL records into a CSV and a Markdown
report.

Reads every *.jsonl file under --diagnostics_dir (default
results/diagnostics/ragdefender/), aggregates per (dataset, model, defense,
k, N_injected) group, and writes:

  - results/diagnostics/ragdefender_summary.csv   (full column set)
  - results/diagnostics/RAGDEFENDER_DIAGNOSTIC_REPORT.md  (human-readable
    report, detection-quality metrics first, ASR last)

Usage:
    python scripts/summarize_ragdefender_diagnostics.py
    python scripts/summarize_ragdefender_diagnostics.py --diagnostics_dir results/diagnostics/ragdefender
"""
import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DIAGNOSTICS_DIR = "results/diagnostics/ragdefender"
DEFAULT_CSV_OUT = "results/diagnostics/ragdefender_summary.csv"
DEFAULT_REPORT_OUT = "results/diagnostics/RAGDEFENDER_DIAGNOSTIC_REPORT.md"

DIAGNOSTIC_CONTROL_WARNING = (
    "**Oracle removal and random removal are diagnostic controls, not deployable defenses.** "
    "`oracle_remove_all_poison` uses ground-truth attack labels no real defense has access to "
    "at inference time; `random_remove_same_count` removes passages with no signal at all. Both "
    "exist only to bound/contextualize RAGDefender's performance."
)

CSV_COLUMNS = [
    "dataset", "model", "defense", "k", "N_injected", "n_queries",
    "ASR_no_defense", "ASR_with_defense", "ASR_delta",
    "mean_N_retrieved_poison", "mean_N_retrieved_clean", "mean_N_adv_estimated",
    "mean_removed_poison", "mean_removed_clean",
    "mean_poison_recall", "mean_clean_false_positive_rate",
    "mean_residual_poison_fraction", "mean_latency_defense_sec",
]

# Defenses that removal-quality metrics are meaningful for (i.e. actually
# attempt removal); "none" is intentionally excluded from decision-tree logic.
REMOVAL_DEFENSES = {"ragdefender", "ragdefender_original", "oracle_remove_all_poison", "random_remove_same_count"}
RAGDEFENDER_NAMES = {"ragdefender", "ragdefender_original"}


def load_records(diagnostics_dir: str) -> List[Dict]:
    records: List[Dict] = []
    pattern = os.path.join(diagnostics_dir, "*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def _mean(values: List) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return statistics.mean(clean)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def group_key(record: Dict) -> Tuple:
    return (record["dataset"], record["model"], record["defense"], record["k"], record["N_injected"])


def aggregate(records: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in records:
        groups[group_key(r)].append(r)

    summaries = []
    for (dataset, model, defense, k, n_injected), recs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][3], kv[0][2])):
        asr_no_defense = _mean([r.get("asr_no_defense") for r in recs if r.get("asr_no_defense") is not None])
        asr_with_defense = _mean([r.get("asr_with_defense") for r in recs if r.get("asr_with_defense") is not None])
        asr_delta = (
            asr_with_defense - asr_no_defense
            if asr_with_defense is not None and asr_no_defense is not None
            else None
        )
        summaries.append({
            "dataset": dataset,
            "model": model,
            "defense": defense,
            "k": k,
            "N_injected": n_injected,
            "n_queries": len(recs),
            "ASR_no_defense": asr_no_defense,
            "ASR_with_defense": asr_with_defense,
            "ASR_delta": asr_delta,
            "mean_N_retrieved_poison": _mean([r.get("N_retrieved_poison") for r in recs]),
            "mean_N_retrieved_clean": _mean([r.get("N_retrieved_clean") for r in recs]),
            "mean_N_adv_estimated": _mean([r.get("N_adv_estimated_by_ragdefender") for r in recs]),
            "mean_removed_poison": _mean([r.get("removed_poison") for r in recs]),
            "mean_removed_clean": _mean([r.get("removed_clean") for r in recs]),
            "mean_poison_recall": _mean([r.get("poison_recall") for r in recs]),
            "mean_clean_false_positive_rate": _mean([r.get("clean_false_positive_rate") for r in recs]),
            "mean_residual_poison_fraction": _mean([r.get("residual_poison_fraction") for r in recs]),
            "mean_latency_defense_sec": _mean([r.get("latency_defense_sec") for r in recs]),
        })
    return summaries


def write_csv(summaries: List[Dict], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def render_detection_first_tables(summaries: List[Dict]) -> str:
    """Primary table(s): detection-quality metrics first, ASR last, grouped
    by dataset (rows: k x defense)."""
    lines = ["## 1. Detection-quality results (by dataset)", ""]
    by_dataset: Dict[str, List[Dict]] = defaultdict(list)
    for s in summaries:
        by_dataset[s["dataset"]].append(s)

    if not summaries:
        return "\n".join(lines + ["_No diagnostic records found._", ""])

    for dataset in sorted(by_dataset):
        rows = sorted(by_dataset[dataset], key=lambda s: (s["k"], s["defense"]))
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append(
            "| k | defense | n | mean_N_retrieved_poison | mean_N_adv_estimated | "
            "mean_removed_poison | mean_removed_clean | mean_residual_poison_fraction | ASR_with_defense |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for s in rows:
            asr_str = _fmt(s["ASR_with_defense"]) if s["ASR_with_defense"] is not None else "n/a (dry-run)"
            lines.append(
                f"| {s['k']} | {s['defense']} | {s['n_queries']} | "
                f"{_fmt(s['mean_N_retrieved_poison'])} | {_fmt(s['mean_N_adv_estimated'])} | "
                f"{_fmt(s['mean_removed_poison'])} | {_fmt(s['mean_removed_clean'])} | "
                f"{_fmt(s['mean_residual_poison_fraction'])} | {asr_str} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_decision_tree(summaries: List[Dict]) -> str:
    """Programmatic interpretation decision tree, applied to the aggregated
    numbers, rendered as bullets with the actual observed values."""
    lines = ["## 3. Interpretation decision tree", "", DIAGNOSTIC_CONTROL_WARNING, ""]

    by_dataset: Dict[str, List[Dict]] = defaultdict(list)
    for s in summaries:
        by_dataset[s["dataset"]].append(s)

    if not summaries:
        lines.append("_No diagnostic records found; nothing to interpret._")
        return "\n".join(lines)

    for dataset in sorted(by_dataset):
        rows = by_dataset[dataset]
        lines.append(f"### {dataset}")
        lines.append("")

        rd_rows = sorted(
            [s for s in rows if s["defense"] in RAGDEFENDER_NAMES],
            key=lambda s: s["k"],
        )
        none_rows = {s["k"]: s for s in rows if s["defense"] == "none"}
        oracle_rows = sorted([s for s in rows if s["defense"] == "oracle_remove_all_poison"], key=lambda s: s["k"])

        if not rd_rows:
            lines.append("_No ragdefender_original/ragdefender records for this dataset; skipping decision tree._")
            lines.append("")
            continue

        k5 = next((s for s in rd_rows if s["k"] == 5), rd_rows[0])
        higher_k_rows = [s for s in rd_rows if s["k"] > k5["k"]]

        def _metric_for_compare(s: Dict) -> Optional[float]:
            # Prefer ASR when generation ran; fall back to residual poison
            # fraction (available even in --dry_run) otherwise.
            return s["ASR_with_defense"] if s["ASR_with_defense"] is not None else s["mean_residual_poison_fraction"]

        metric_name = "ASR_with_defense" if k5["ASR_with_defense"] is not None else "mean_residual_poison_fraction (ASR unavailable: dry-run)"
        k5_metric = _metric_for_compare(k5)

        improved_at_higher_k = False
        for s in higher_k_rows:
            hk_metric = _metric_for_compare(s)
            if hk_metric is not None and k5_metric is not None and hk_metric < k5_metric:
                improved_at_higher_k = True
                lines.append(
                    f"- **k={k5['k']} vs k={s['k']}** ({metric_name}): "
                    f"{_fmt(k5_metric)} -> {_fmt(hk_metric)}. "
                    f"RAGDefender improves once k > N -> conclude **threat-model mismatch / "
                    f"retrieval-saturation failure** at k={k5['k']} (100%-poisoned context)."
                )
            elif hk_metric is not None and k5_metric is not None:
                lines.append(
                    f"- **k={k5['k']} vs k={s['k']}** ({metric_name}): "
                    f"{_fmt(k5_metric)} -> {_fmt(hk_metric)} (no improvement). "
                    f"RAGDefender still fails once k > N -> flag for **implementation-mismatch investigation**."
                )

        if not higher_k_rows:
            lines.append(f"- Only k={k5['k']} available for ragdefender in this dataset; cannot compare against k > N yet (run more k values).")

        if k5["mean_poison_recall"] is not None:
            recall_high = k5["mean_poison_recall"] >= 0.5
            asr_high = (k5["ASR_with_defense"] or 0) >= 0.3 if k5["ASR_with_defense"] is not None else None
            if recall_high and asr_high:
                lines.append(
                    f"- At k={k5['k']}: mean_poison_recall={_fmt(k5['mean_poison_recall'])} (removes poison) but "
                    f"ASR_with_defense={_fmt(k5['ASR_with_defense'])} stays high -> conclude "
                    f"**residual-poison sensitivity** (even 1 leftover adversarial passage can control generation)."
                )
            elif recall_high and asr_high is None:
                lines.append(
                    f"- At k={k5['k']}: mean_poison_recall={_fmt(k5['mean_poison_recall'])} but ASR is unavailable "
                    f"(dry-run) -> re-run with generation enabled to test residual-poison sensitivity."
                )

        if k5["mean_clean_false_positive_rate"] is not None and k5["mean_clean_false_positive_rate"] >= 0.3:
            lines.append(
                f"- At k={k5['k']}: mean_clean_false_positive_rate={_fmt(k5['mean_clean_false_positive_rate'])} "
                f"-> conclude **false-positive clean-evidence removal** (RAGDefender is deleting clean passages often)."
            )

        for s in oracle_rows:
            oracle_metric = _metric_for_compare(s)
            if oracle_metric is not None and s["ASR_with_defense"] is not None and s["ASR_with_defense"] >= 0.3:
                lines.append(
                    f"- **Oracle check (k={s['k']})**: even oracle_remove_all_poison (ground-truth poison "
                    f"removal) leaves ASR_with_defense={_fmt(s['ASR_with_defense'])} -> conclude the issue lies in "
                    f"**prompt construction, answer matching, or clean-evidence quality**, not detection at all."
                )
            elif s["ASR_with_defense"] is None:
                lines.append(
                    f"- **Oracle check (k={s['k']})**: ASR unavailable (dry-run); re-run with generation enabled "
                    f"to test whether oracle removal alone drops ASR."
                )

        lines.append("")

    return "\n".join(lines)


def render_worst_queries(records: List[Dict], n: int = 10) -> str:
    lines = ["## 4. Worst 10 queries by residual poison fraction", ""]
    scored = [r for r in records if r.get("residual_poison_fraction") is not None]
    scored.sort(key=lambda r: r["residual_poison_fraction"], reverse=True)
    top = scored[:n]
    if not top:
        lines.append("_No records with a defined residual_poison_fraction._")
        return "\n".join(lines) + "\n"

    lines.append("| dataset | k | defense | query_id | residual_poison_fraction | N_retrieved_poison | removed_poison |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in top:
        lines.append(
            f"| {r['dataset']} | {r['k']} | {r['defense']} | {r['query_id']} | "
            f"{_fmt(r['residual_poison_fraction'])} | {r['N_retrieved_poison']} | {r['removed_poison']} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_clean_gt_poison_removals(records: List[Dict]) -> str:
    lines = ["## 5. Cases where RAGDefender removed more clean than poisoned passages", ""]
    bad = [r for r in records if r["defense"] in RAGDEFENDER_NAMES and r["removed_clean"] > r["removed_poison"]]
    if not bad:
        lines.append("_None found (in the currently loaded diagnostic records)._")
        return "\n".join(lines) + "\n"

    lines.append("| dataset | k | query_id | removed_poison | removed_clean |")
    lines.append("|---|---|---|---|---|")
    for r in bad:
        lines.append(f"| {r['dataset']} | {r['k']} | {r['query_id']} | {r['removed_poison']} | {r['removed_clean']} |")
    lines.append("")
    return "\n".join(lines)


def render_comparison_table(summaries: List[Dict]) -> str:
    lines = ["## 6. RAGDefender vs. oracle vs. random removal", "", DIAGNOSTIC_CONTROL_WARNING, ""]
    by_dataset_k: Dict[Tuple[str, int], Dict[str, Dict]] = defaultdict(dict)
    for s in summaries:
        by_dataset_k[(s["dataset"], s["k"])][s["defense"]] = s

    if not by_dataset_k:
        lines.append("_No diagnostic records found._")
        return "\n".join(lines) + "\n"

    lines.append(
        "| dataset | k | defense | mean_residual_poison_fraction | mean_poison_recall | "
        "mean_clean_false_positive_rate | ASR_with_defense |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for (dataset, k), by_defense in sorted(by_dataset_k.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        for defense_name in ["none", "ragdefender_original", "ragdefender", "oracle_remove_all_poison", "random_remove_same_count"]:
            s = by_defense.get(defense_name)
            if s is None:
                continue
            asr_str = _fmt(s["ASR_with_defense"]) if s["ASR_with_defense"] is not None else "n/a (dry-run)"
            lines.append(
                f"| {dataset} | {k} | {defense_name} | {_fmt(s['mean_residual_poison_fraction'])} | "
                f"{_fmt(s['mean_poison_recall'])} | {_fmt(s['mean_clean_false_positive_rate'])} | {asr_str} |"
            )
    lines.append("")
    return "\n".join(lines)


def render_report(summaries: List[Dict], records: List[Dict]) -> str:
    parts = [
        "# RAGDefender Diagnostic Report",
        "",
        "Auto-generated by `scripts/summarize_ragdefender_diagnostics.py`. "
        "See `docs/RAGDEFENDER_DIAGNOSTIC_PLAN.md` for full methodology.",
        "",
        render_detection_first_tables(summaries),
        "## 2. " + DIAGNOSTIC_CONTROL_WARNING,
        "",
        render_decision_tree(summaries),
        render_worst_queries(records, n=10),
        render_clean_gt_poison_removals(records),
        render_comparison_table(summaries),
    ]
    return "\n".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics_dir", default=DEFAULT_DIAGNOSTICS_DIR)
    parser.add_argument("--csv_out", default=DEFAULT_CSV_OUT)
    parser.add_argument("--report_out", default=DEFAULT_REPORT_OUT)
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_records(args.diagnostics_dir)
    summaries = aggregate(records)

    write_csv(summaries, args.csv_out)
    report = render_report(summaries, records)
    parent = os.path.dirname(args.report_out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Loaded {len(records)} diagnostic record(s) from {args.diagnostics_dir}/*.jsonl")
    print(f"Wrote {len(summaries)} summary row(s) to {args.csv_out}")
    print(f"Wrote report to {args.report_out}")


if __name__ == "__main__":
    main()
