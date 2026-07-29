#!/usr/bin/env python3
"""FilterRAG rerun-after-SLM-fix diagnostic driver (detection-only, zero
GPT/API cost).

Context: an earlier full FilterRAG diagnostic run was contaminated by the
MPS/T5 bug described in `docs/FILTERRAG_BASELINE.md` §3.1 -- google/flan-t5-small
produced an empty `SLM_answer` for 100% of passages on MPS, so `filterrag`
silently collapsed into the `filterrag_query_only` ablation. That bug was
fixed (see `_get_local_hf_slm_pipeline()`'s post-load smoke test in
defense/filterrag.py). This script reruns the detection-only diagnostic
*after* that fix, and adds the extra checks needed to positively confirm the
SLM is actually contributing before spending any money on a live GPT rerun:

1. Reuses (does not duplicate or modify) `build_passage_records`,
   `write_score_csv`, `epsilon_sweep`, `write_sweep_csv` from
   `scripts/filterrag_score_inspection.py` -- same retrieval + LM_targeted
   adversarial-injection pipeline, same per-passage full-vs-query-only
   Freq-Density scoring, same epsilon sweep. Nothing about that script or
   about `defense/filterrag.py`/`defense/dispatch.py` is changed.
2. Captures the SLM pipeline's own stdout (device-resolution + smoke-test
   fallback log lines) to record, in `run_config.json`, whether the MPS
   smoke test actually ran and whether it fell back to CPU -- rather than
   assuming a device or reaching into private module state.
3. Computes an aggregate summary (per k) that the base score-inspection
   script doesn't: number of passages scored, non-empty SLM answer
   count/percentage, full-vs-query-only score divergence count/percentage,
   and mean query-only/SLM-backed scores split by poison vs. clean.
4. Renders a single Markdown report with all of the above plus an
   automated (not hand-waved) stop/go recommendation and a *proposed but
   not executed* live-GPT rerun command list.

No GPT/PaLM/OpenAI API call is made anywhere in this script. It does run
the local FilterRAG SLM (google/flan-t5-small by default) -- free/local,
same as `--defense filterrag` would.

Usage:
    python scripts/filterrag_rerun_after_slm_fix.py \\
        --eval_dataset hotpotqa --k_values 5 10 --N 5 --max_queries 10 \\
        --filterrag_slm_model google/flan-t5-small --filterrag_slm_device auto \\
        --epsilons 0.2 0.3 0.4 0.5 0.6 0.7 0.8
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime
from typing import Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)  # matches scripts/filterrag_score_inspection.py's own convention

from scripts.filterrag_score_inspection import (  # noqa: E402
    build_passage_records,
    epsilon_sweep,
    write_score_csv,
    write_sweep_csv,
)

DEFAULT_EPSILONS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
OUT_ROOT_DEFAULT = "results/diagnostics/filterrag_rerun_after_slm_fix"


class _Tee(io.TextIOBase):
    """Writes to the real stdout *and* an in-memory buffer, so the SLM
    pipeline's own device-resolution/smoke-test-fallback prints stay
    visible live in the console while also being capturable afterwards
    (to fill in run_config.json's resolved-device / fallback-triggered
    fields without reaching into defense/filterrag.py's private globals).
    """

    def __init__(self, real_stream):
        self._real = real_stream
        self.buffer = io.StringIO()

    def write(self, s):
        self._real.write(s)
        self.buffer.write(s)
        return len(s)

    def flush(self):
        self._real.flush()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_dataset", default="hotpotqa", choices=["nq", "hotpotqa", "msmarco"])
    parser.add_argument("--eval_model_code", default="contriever")
    parser.add_argument("--split", default="test")
    parser.add_argument("--score_function", default="dot", choices=["dot", "cos_sim"])
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--N", type=int, default=5, help="adv_per_query")
    parser.add_argument("--max_queries", type=int, default=10)
    parser.add_argument("--filterrag_slm_model", default="google/flan-t5-small")
    parser.add_argument("--filterrag_slm_device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
    parser.add_argument("--out_root", default=OUT_ROOT_DEFAULT)
    parser.add_argument("--n_examples_per_category", type=int, default=5)
    parser.add_argument("--example_epsilon", type=float, default=0.20)
    parser.add_argument("--text_preview_chars", type=int, default=150)
    # Decision-rule thresholds (Phase 3), overridable rather than hardcoded.
    parser.add_argument(
        "--min_pct_nonempty_slm", type=float, default=20.0,
        help="Below this pct of non-empty SLM answers, recommend stop-and-debug-SLM.",
    )
    parser.add_argument(
        "--clean_fpr_threshold", type=float, default=0.3,
        help="Max acceptable mean_clean_false_positive_rate for an epsilon to be recommended for live testing.",
    )
    parser.add_argument(
        "--over_removal_fpr_threshold", type=float, default=0.9,
        help="mean_clean_false_positive_rate at/above this is flagged as 'over-removes clean context'.",
    )
    return parser.parse_args()


def git_commit_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def build_out_dir(args, resolved_device: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    k_part = "_".join(f"k{k}" for k in args.k_values)
    model_part = args.filterrag_slm_model.split("/")[-1]
    dirname = f"{timestamp}_{args.eval_dataset}_{k_part}_N{args.N}_{model_part}_{resolved_device}"
    return os.path.join(args.out_root, dirname)


def run_scoring_with_captured_stdout(args):
    """Runs build_passage_records() (unchanged, imported from
    filterrag_score_inspection.py) while teeing stdout so the SLM
    pipeline's device-resolution / smoke-test-fallback log lines can be
    parsed afterwards, without touching any private state in
    defense/filterrag.py.
    """
    tee = _Tee(sys.stdout)
    t0 = time.perf_counter()
    with redirect_stdout(tee):
        rows = build_passage_records(args)
    elapsed = time.perf_counter() - t0
    captured = tee.buffer.getvalue()
    return rows, elapsed, captured


def parse_slm_device_log(captured_stdout: str) -> Dict:
    device_match = re.search(r"\[FilterRAG\] SLM device: (\w+) \(model=([^)]+)\)", captured_stdout)
    fallback_match = re.search(
        r"\[FilterRAG\] --filterrag_slm_device=(\w+) requested but \w+ is not available.*falling back",
        captured_stdout,
    )
    smoke_test_fallback_match = re.search(
        r"\[FilterRAG\] WARNING: SLM device=.*failed a smoke-test generation.*falling back to device='cpu'",
        captured_stdout,
    )
    per_passage_failure_match = re.search(
        r"\[FilterRAG\] WARNING: SLM generation failed for a passage", captured_stdout,
    )
    return {
        "resolved_device": device_match.group(1) if device_match else None,
        "resolved_model_name_from_log": device_match.group(2) if device_match else None,
        "mps_requested_but_unavailable_fallback": bool(fallback_match),
        "mps_smoke_test_failed_fallback_to_cpu": bool(smoke_test_fallback_match),
        "any_per_passage_slm_generation_failure_logged": bool(per_passage_failure_match),
    }


def compute_aggregate_summary(rows: List[Dict]) -> List[Dict]:
    """Per-k (+ an 'all_k' row) aggregate stats not already produced by
    scripts/filterrag_score_inspection.py's epsilon_sweep(): passages
    scored, non-empty SLM answer count/pct, full-vs-query-only divergence
    count/pct, and mean query-only/SLM-backed score split by poison vs.
    clean."""
    def _summarize(subset: List[Dict], k_label) -> Dict:
        n = len(subset)
        n_nonempty = sum(1 for r in subset if (r["slm_answer"] or "").strip())
        n_divergent = sum(1 for r in subset if abs(r["filterrag_score"] - r["query_only_score"]) > 1e-9)
        poison = [r for r in subset if r["is_poison"]]
        clean = [r for r in subset if not r["is_poison"]]

        def _mean(items, key):
            vals = [r[key] for r in items]
            return sum(vals) / len(vals) if vals else None

        return {
            "k": k_label,
            "n_passages_scored": n,
            "n_slm_answers_nonempty": n_nonempty,
            "pct_slm_answers_nonempty": (100.0 * n_nonempty / n) if n else None,
            "n_divergent_full_vs_query_only": n_divergent,
            "pct_divergent_full_vs_query_only": (100.0 * n_divergent / n) if n else None,
            "n_poison": len(poison),
            "n_clean": len(clean),
            "mean_query_only_score_poison": _mean(poison, "query_only_score"),
            "mean_query_only_score_clean": _mean(clean, "query_only_score"),
            "mean_slm_backed_score_poison": _mean(poison, "filterrag_score"),
            "mean_slm_backed_score_clean": _mean(clean, "filterrag_score"),
        }

    summaries = [_summarize(rows, "all_k")]
    for k in sorted({r["k"] for r in rows}):
        summaries.append(_summarize([r for r in rows if r["k"] == k], k))
    return summaries


AGG_COLUMNS = [
    "k", "n_passages_scored", "n_slm_answers_nonempty", "pct_slm_answers_nonempty",
    "n_divergent_full_vs_query_only", "pct_divergent_full_vs_query_only",
    "n_poison", "n_clean",
    "mean_query_only_score_poison", "mean_query_only_score_clean",
    "mean_slm_backed_score_poison", "mean_slm_backed_score_clean",
]


def write_aggregate_csv(agg_rows: List[Dict], path: str) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AGG_COLUMNS)
        writer.writeheader()
        for r in agg_rows:
            writer.writerow(r)
    print(f"Wrote {len(agg_rows)} aggregate summary row(s) to {path}")


def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def recommend_epsilons(sweep_rows: List[Dict], clean_fpr_threshold: float) -> List[Dict]:
    """Across all k values, pick epsilon(s) for `filterrag` with the best
    poison-recall / clean-false-positive-rate trade-off: among epsilons
    whose *worst-case-across-k* mean_clean_false_positive_rate is at/under
    `clean_fpr_threshold` (or has no clean passages to false-positive on at
    all, e.g. k==N), rank by mean (across k) mean_poison_recall descending,
    epsilon ascending as a tiebreaker (prefer the least aggressive filter
    that still gets the recall)."""
    by_epsilon: Dict[float, List[Dict]] = defaultdict(list)
    for r in sweep_rows:
        if r["defense"] == "filterrag":
            by_epsilon[r["epsilon"]].append(r)

    candidates = []
    for epsilon, per_k_rows in by_epsilon.items():
        fprs = [r["mean_clean_false_positive_rate"] for r in per_k_rows if r["mean_clean_false_positive_rate"] is not None]
        recalls = [r["mean_poison_recall"] for r in per_k_rows if r["mean_poison_recall"] is not None]
        worst_fpr = max(fprs) if fprs else None
        mean_recall = sum(recalls) / len(recalls) if recalls else None
        qualifies = worst_fpr is None or worst_fpr <= clean_fpr_threshold
        candidates.append({
            "epsilon": epsilon,
            "worst_case_clean_fpr_across_k": worst_fpr,
            "mean_poison_recall_across_k": mean_recall,
            "qualifies_under_clean_fpr_threshold": qualifies,
        })
    candidates.sort(key=lambda c: (
        not c["qualifies_under_clean_fpr_threshold"],
        -(c["mean_poison_recall_across_k"] or -1),
        c["epsilon"],
    ))
    return candidates


def render_report(
    *, args, run_config: Dict, agg_rows: List[Dict], sweep_rows: List[Dict],
    recommended: List[Dict], out_dir: str,
) -> str:
    lines = [
        "# FilterRAG Rerun After SLM Fix -- Diagnostic Report",
        "",
        f"Generated by `scripts/filterrag_rerun_after_slm_fix.py`. Output directory: `{out_dir}`.",
        "",
        "No GPT/PaLM/OpenAI API calls were made to produce this report -- detection-only, "
        "using cached retrieval results and the local FilterRAG SLM (see run_config.json).",
        "",
        "## 0. Run configuration",
        "",
        f"- dataset={run_config['eval_dataset']}, k_values={run_config['k_values']}, "
        f"N={run_config['N']}, max_queries={run_config['max_queries']}",
        f"- SLM model={run_config['filterrag_slm_model']}, requested device={run_config['filterrag_slm_device_requested']}, "
        f"**resolved device={run_config['slm_device_log']['resolved_device']}**",
        f"- MPS smoke-test failed and fell back to CPU: "
        f"**{run_config['slm_device_log']['mps_smoke_test_failed_fallback_to_cpu']}**",
        f"- Any per-passage SLM generation failure logged (independent of device fallback): "
        f"{run_config['slm_device_log']['any_per_passage_slm_generation_failure_logged']}",
        f"- git commit: {run_config['git_commit']}",
        f"- scoring wall-clock time: {run_config['scoring_elapsed_sec']:.1f}s for "
        f"{run_config['n_passage_rows']} passage-score rows",
        "",
        "## 1-5. Aggregate summary (per k, plus all_k)",
        "",
        "| k | n_passages_scored | n_slm_nonempty | pct_slm_nonempty | n_divergent | pct_divergent | "
        "mean_query_only(poison) | mean_query_only(clean) | mean_slm_backed(poison) | mean_slm_backed(clean) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in agg_rows:
        lines.append(
            f"| {r['k']} | {r['n_passages_scored']} | {r['n_slm_answers_nonempty']} | "
            f"{_fmt(r['pct_slm_answers_nonempty'], 1)}% | {r['n_divergent_full_vs_query_only']} | "
            f"{_fmt(r['pct_divergent_full_vs_query_only'], 1)}% | "
            f"{_fmt(r['mean_query_only_score_poison'])} | {_fmt(r['mean_query_only_score_clean'])} | "
            f"{_fmt(r['mean_slm_backed_score_poison'])} | {_fmt(r['mean_slm_backed_score_clean'])} |"
        )
    lines.append("")

    lines += [
        "## 6. Epsilon sweep (poison recall / clean FPR / residual poison / empty context)",
        "",
        "| defense | k | epsilon | n | mean_poison_recall | mean_clean_FP_rate | "
        "mean_residual_poison_fraction | n_empty_context |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(sweep_rows, key=lambda r: (r["k"], r["epsilon"], r["defense"])):
        lines.append(
            f"| {r['defense']} | {r['k']} | {r['epsilon']:.2f} | {r['n_queries']} | "
            f"{_fmt(r['mean_poison_recall'])} | {_fmt(r['mean_clean_false_positive_rate'])} | "
            f"{_fmt(r['mean_residual_poison_fraction'])} | {r['n_queries_empty_context']} |"
        )
    lines.append("")

    lines += ["## 7. Does epsilon=0.2 over-remove clean passages?", ""]
    eps02_rows = [r for r in sweep_rows if abs(r["epsilon"] - 0.2) < 1e-9 and r["defense"] == "filterrag"]
    over_removal_flagged = False
    for r in sorted(eps02_rows, key=lambda r: r["k"]):
        fpr = r["mean_clean_false_positive_rate"]
        flagged = fpr is not None and fpr >= args.over_removal_fpr_threshold
        over_removal_flagged = over_removal_flagged or flagged
        note = (
            f"**OVER-REMOVES clean context** (>= {args.over_removal_fpr_threshold:.0%} threshold)" if flagged
            else ("no clean passages present at this k (100% poisoned context, cannot assess)" if fpr is None else "within threshold")
        )
        lines.append(f"- k={r['k']}, epsilon=0.20: mean_clean_false_positive_rate={_fmt(fpr)} -> {note}")
    lines.append("")

    lines += ["## 8. Best epsilon candidates (poison recall vs. clean removal trade-off)", ""]
    lines.append(
        f"Ranked by mean poison recall across k among epsilons with worst-case-across-k "
        f"clean false-positive rate <= {args.clean_fpr_threshold:.2f} (or no clean passages to "
        f"false-positive on):"
    )
    lines.append("")
    lines.append("| rank | epsilon | mean_poison_recall (avg across k) | worst-case clean_FPR (across k) | qualifies |")
    lines.append("|---|---|---|---|---|")
    for i, c in enumerate(recommended[:5], start=1):
        lines.append(
            f"| {i} | {c['epsilon']:.2f} | {_fmt(c['mean_poison_recall_across_k'])} | "
            f"{_fmt(c['worst_case_clean_fpr_across_k'])} | {c['qualifies_under_clean_fpr_threshold']} |"
        )
    lines.append("")

    # --- Phase 3: automated stop/go decision -------------------------------------------------
    lines += ["## Phase 3: stop/go decision", ""]
    all_k_row = next(r for r in agg_rows if r["k"] == "all_k")
    pct_divergent = all_k_row["pct_divergent_full_vs_query_only"] or 0.0
    pct_nonempty = all_k_row["pct_slm_answers_nonempty"] or 0.0

    stop_reasons = []
    if pct_divergent <= 0.0:
        stop_reasons.append(
            f"filterrag and filterrag_query_only are still byte-for-byte identical "
            f"(pct_divergent_full_vs_query_only={_fmt(pct_divergent, 1)}%) -> **STOP, debug SLM**."
        )
    if pct_nonempty < args.min_pct_nonempty_slm:
        stop_reasons.append(
            f"SLM answers are mostly empty (pct_slm_answers_nonempty={_fmt(pct_nonempty, 1)}% < "
            f"{args.min_pct_nonempty_slm:.1f}% threshold) -> **STOP, debug SLM**."
        )

    qualifying = [c for c in recommended if c["qualifies_under_clean_fpr_threshold"]]
    if stop_reasons:
        lines.append("**Recommendation: DO NOT proceed to a live GPT rerun yet.**")
        lines.append("")
        for reason in stop_reasons:
            lines.append(f"- {reason}")
    else:
        lines.append(
            f"- SLM is contributing: {_fmt(pct_divergent, 1)}% of passages have a full-vs-query-only "
            f"score divergence, and {_fmt(pct_nonempty, 1)}% of SLM answers are non-empty."
        )
        if over_removal_flagged:
            lines.append(
                f"- epsilon=0.2 over-removes clean context at one or more k -> "
                f"**do not use only epsilon=0.2 for live runs**."
            )
        if qualifying:
            top = qualifying[:2]
            eps_str = ", ".join(f"{c['epsilon']:.2f}" for c in top)
            lines.append(f"- **Recommendation: proceed to a small live GPT rerun, using epsilon in [{eps_str}].**")
        else:
            lines.append(
                "- No epsilon in the sweep meets the clean-false-positive-rate threshold with "
                "useful poison recall -> **do not proceed to live rerun; widen the epsilon sweep "
                "or re-examine the clean_fpr_threshold before spending on GPT calls.**"
            )
    lines.append("")

    # --- Phase 4: proposed (not executed) live rerun plan -------------------------------------
    lines += [
        "## Phase 4: proposed live rerun plan (NOT executed)",
        "",
        "This command is written here for review only; it was **not run** as part of this "
        "diagnostic pass, and must not be run without explicit approval.",
        "",
    ]
    if qualifying:
        eps_list = " ".join(f"{c['epsilon']:.2f}" for c in qualifying[:2])
    else:
        eps_list = f"{args.example_epsilon:.2f}  # placeholder -- no epsilon cleared the threshold; re-examine before using"
    lines.append("```bash")
    lines.append("python scripts/run_ragdefender_k_sweep.py \\")
    lines.append(f"  --datasets {run_config['eval_dataset']} --k_values {' '.join(str(k) for k in run_config['k_values'])} \\")
    lines.append(f"  --N {run_config['N']} --max_queries {run_config['max_queries']} \\")
    lines.append("  --defenses none filterrag_query_only filterrag ragdefender_original \\")
    lines.append(f"  --filterrag_epsilon {eps_list.split()[0]} \\")
    lines.append(f"  --filterrag_slm_model {run_config['filterrag_slm_model']} \\")
    lines.append("  --execute --live_generation")
    lines.append("```")
    lines.append("")
    lines.append(
        "Compare against `none`, `filterrag_query_only`, and the already-available "
        "`ragdefender_original` dry-run diagnostics under `results/diagnostics/ragdefender/` "
        "(same dataset/k/N/max_queries), using strict ASR "
        "(`defense/asr_match.py`, wired into `scripts/summarize_ragdefender_diagnostics.py`)."
    )
    lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()

    rows, elapsed, captured_stdout = run_scoring_with_captured_stdout(args)
    slm_device_log = parse_slm_device_log(captured_stdout)
    resolved_device = slm_device_log["resolved_device"] or "unknown"

    out_dir = build_out_dir(args, resolved_device)
    os.makedirs(out_dir, exist_ok=True)

    score_csv_path = os.path.join(out_dir, "score_inspection.csv")
    write_score_csv(rows, score_csv_path, epsilon_for_removed_col=args.example_epsilon)

    sweep_rows = epsilon_sweep(rows, args.epsilons)
    write_sweep_csv(sweep_rows, os.path.join(out_dir, "epsilon_sweep.csv"))

    agg_rows = compute_aggregate_summary(rows)
    write_aggregate_csv(agg_rows, os.path.join(out_dir, "aggregate_summary.csv"))

    recommended = recommend_epsilons(sweep_rows, args.clean_fpr_threshold)

    run_config = {
        "generated_at": datetime.now().isoformat(),
        "git_commit": git_commit_hash(),
        "eval_dataset": args.eval_dataset,
        "eval_model_code": args.eval_model_code,
        "split": args.split,
        "score_function": args.score_function,
        "k_values": args.k_values,
        "N": args.N,
        "max_queries": args.max_queries,
        "epsilons": args.epsilons,
        "filterrag_slm_model": args.filterrag_slm_model,
        "filterrag_slm_device_requested": args.filterrag_slm_device,
        "slm_device_log": slm_device_log,
        "n_passage_rows": len(rows),
        "scoring_elapsed_sec": elapsed,
        "thresholds": {
            "min_pct_nonempty_slm": args.min_pct_nonempty_slm,
            "clean_fpr_threshold": args.clean_fpr_threshold,
            "over_removal_fpr_threshold": args.over_removal_fpr_threshold,
        },
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"Wrote run_config.json to {out_dir}")

    report = render_report(
        args=args, run_config=run_config, agg_rows=agg_rows, sweep_rows=sweep_rows,
        recommended=recommended, out_dir=out_dir,
    )
    report_path = os.path.join(out_dir, "FILTER_RAG_RERUN_AFTER_SLM_FIX_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote report to {report_path}")

    print()
    print(f"Resolved SLM device: {resolved_device} "
          f"(mps_smoke_test_failed_fallback_to_cpu={slm_device_log['mps_smoke_test_failed_fallback_to_cpu']})")
    all_k = next(r for r in agg_rows if r["k"] == "all_k")
    print(f"Non-empty SLM answers: {all_k['n_slm_answers_nonempty']}/{all_k['n_passages_scored']} "
          f"({_fmt(all_k['pct_slm_answers_nonempty'], 1)}%)")
    print(f"Full-vs-query-only divergence: {all_k['n_divergent_full_vs_query_only']}/{all_k['n_passages_scored']} "
          f"({_fmt(all_k['pct_divergent_full_vs_query_only'], 1)}%)")
    print(f"\nAll outputs written under: {out_dir}/")


if __name__ == "__main__":
    main()
