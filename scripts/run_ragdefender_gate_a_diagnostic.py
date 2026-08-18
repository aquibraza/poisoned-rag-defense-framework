"""Gate A -- RAGDefender Stage-1 logic-isolation diagnostic.

IMPORTANT SCOPE NOTE (read before interpreting output):
This script is a **logic-isolation diagnostic**, NOT a validation of the
FINAL-PAPER RAGDefender implementation. It runs `ragdefender_legacy` Stage 1
(`ragdefender_internals.concentration_stage1`) and `ragdefender_paper` Stage 1
(`ragdefender_internals.concentration_stage1_paper`) on the *same already-saved
MiniLM (paraphrase-MiniLM-L6-v2) cosine similarity matrix* per query, to
isolate the effect of the Stage-1 equation change (OR-vs-AND, diagonal
inclusion, hybrid-vs-median threshold, flip branch) from the effect of the
encoder change (MiniLM -> Stella). The final ACSAC 2025 paper specifies
Stella, not MiniLM -- so this script deliberately does NOT re-encode
anything, makes zero retrieval calls, and zero generation/LLM API calls. See
Gate B (docs/RAGDEFENDER_FIDELITY_AUDIT_V2.md) for the true paper-fidelity
gate that re-encodes with Stella.

For each of the 8 available instrumented HotpotQA k=10/N=5 queries (the
`ragdefender_cluster_viz` diagnostics run under
results/diagnostics/ragdefender_cluster_viz/), this script:

  1. Loads the saved k x k MiniLM cosine similarity matrix and the
     ground-truth poison/clean labels (`passages/{query_id}_passages.csv`).
  2. Runs `ragdefender_internals.concentration_stage1` (legacy: OR logic,
     diagonal-inclusive, hybrid threshold, flip branch) on that matrix.
  3. Runs `ragdefender_internals.concentration_stage1_paper` (paper: AND
     logic, self-excluded, median-of-medians threshold, no flip) on the
     *same* matrix.
  4. Feeds each variant's `n_adv_estimated` into the shared
     `ragdefender_internals.stage2_pair_frequency` (Eq. 4-7, `p=2`).
  5. Records every requested intermediate per query per variant, and a
     per-query legacy-vs-paper diff table.

Outputs (per plan / P0-B instructions):
  results/diagnostics/ragdefender_gate_a/gate_a_per_query.csv
  results/diagnostics/ragdefender_gate_a/gate_a_summary.csv
  results/diagnostics/ragdefender_gate_a/GATE_A_LOGIC_ISOLATION_REPORT.md
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from defense import ragdefender_internals  # noqa: E402

CLUSTER_VIZ_RUN_DIR = (
    REPO_ROOT
    / "results/diagnostics/ragdefender_cluster_viz"
    / "20260722_042137_clusterdiag_hotpotqa_k10_N5_ragdefender-original_"
    "embedder-paraphraseMiniLM_task-multihop_p2"
)
OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_gate_a"

# Case-inspection labels the report must specifically call out (per P0-B
# instructions), derived from stage1_summary.csv / graph_metrics.csv in the
# source diagnostics run (see script docstring for how these were
# identified -- N_adv_estimated_by_recomputed_concentration and
# removed_poison/removed_clean/residual_poison_fraction columns there).
LEGACY_OVERESTIMATION_CASE = "5a722b8655429971e9dc9329"  # legacy N_adv=7 vs 5 true poison (over by 2)
CLEAN_DENSITY_CASE = "5a8cb288554299585d9e3726"  # clean-clean sim ~0.99; legacy removed 0 poison, 3 clean
PREVIOUSLY_SUCCESSFUL_CASES = [
    "5adbf0a255429947ff17385a",
    "5ab56e32554299637185c594",
    "5ab29c24554299449642c932",
    "5ae6050f55429929b0807a5e",
    "5ae2070a5542994d89d5b313",
    "5a722b8655429971e9dc9329",
]


def _pipe(values) -> str:
    return "|".join(str(v) for v in values)


def _load_case(query_id: str) -> Dict:
    sim_path = CLUSTER_VIZ_RUN_DIR / "similarity" / f"{query_id}_similarity_matrix.npy"
    passages_path = CLUSTER_VIZ_RUN_DIR / "passages" / f"{query_id}_passages.csv"
    matrix = np.load(sim_path)
    passages = pd.read_csv(passages_path).sort_values("passage_index").reset_index(drop=True)
    is_poison = passages["is_poison"].astype(bool).to_numpy()
    if matrix.shape[0] != len(is_poison):
        raise ValueError(
            f"{query_id}: similarity matrix shape {matrix.shape} does not match "
            f"passages row count {len(is_poison)}"
        )
    return {
        "query_id": query_id,
        "matrix": matrix,
        "is_poison": is_poison,
        "n_retrieved_poison": int(is_poison.sum()),
        "n_retrieved_clean": int((~is_poison).sum()),
        "k": matrix.shape[0],
    }


def _classify_pair(i: int, j: int, is_poison: np.ndarray) -> str:
    pi, pj = bool(is_poison[i]), bool(is_poison[j])
    if pi and pj:
        return "PP"
    if (pi and not pj) or (pj and not pi):
        return "PC"
    return "CC"


def _run_variant(case: Dict, variant: str) -> Dict:
    matrix = case["matrix"]
    is_poison = case["is_poison"]
    k = case["k"]

    if variant == "legacy":
        stage1 = ragdefender_internals.concentration_stage1(matrix)
        n_adv = stage1.n_adv_estimated
        mean_i = stage1.avg
        median_i = stage1.median
        global_mean_threshold = stage1.avg_avg
        median_threshold = stage1.avg_median
        legacy_hybrid_threshold: Optional[float] = stage1.combined_threshold
        above_mean_flags = stage1.above_avg
        above_median_flags = stage1.above_median
        combine_logic = "OR"
        flipped: Optional[bool] = stage1.flipped
        final_adv_flags = stage1.adv_side_flag
    elif variant == "paper":
        stage1 = ragdefender_internals.concentration_stage1_paper(matrix)
        n_adv = stage1.n_adv_estimated
        mean_i = stage1.s_mean
        median_i = stage1.s_median
        global_mean_threshold = stage1.s_bar
        median_threshold = stage1.s_tilde
        legacy_hybrid_threshold = None
        above_mean_flags = stage1.above_mean
        above_median_flags = stage1.above_median
        combine_logic = "AND"
        flipped = None
        final_adv_flags = stage1.adv_flag
    else:
        raise ValueError(variant)

    stage2 = ragdefender_internals.stage2_pair_frequency(matrix, n_adv=n_adv, p=2.0)

    pair_classes = [_classify_pair(i, j, is_poison) for i, j, _sim in stage2.top_pairs]
    pp_count = pair_classes.count("PP")
    pc_count = pair_classes.count("PC")
    cc_count = pair_classes.count("CC")
    top_pair_pp: Optional[bool] = (pair_classes[0] == "PP") if pair_classes else None

    removed_indices = list(stage2.selected_indices)
    removed_poison = int(sum(1 for idx in removed_indices if is_poison[idx]))
    removed_clean = int(sum(1 for idx in removed_indices if not is_poison[idx]))
    residual_poison = case["n_retrieved_poison"] - removed_poison
    residual_clean = case["n_retrieved_clean"] - removed_clean
    residual_poison_fraction = (
        residual_poison / case["n_retrieved_poison"] if case["n_retrieved_poison"] > 0 else float("nan")
    )

    return {
        "query_id": case["query_id"],
        "variant": variant,
        "k": k,
        "n_retrieved_poison": case["n_retrieved_poison"],
        "n_retrieved_clean": case["n_retrieved_clean"],
        "retrieved_composition": f"{case['n_retrieved_poison']} poison / {case['n_retrieved_clean']} clean of {k}",
        "is_poison_i": _pipe(int(x) for x in is_poison),
        "n_adv_estimated": n_adv,
        "mean_i": _pipe(f"{x:.6f}" for x in mean_i),
        "median_i": _pipe(f"{x:.6f}" for x in median_i),
        "global_mean_threshold": global_mean_threshold,
        "median_threshold": median_threshold,
        "legacy_hybrid_threshold": legacy_hybrid_threshold,
        "combine_logic": combine_logic,
        "above_mean_flags": _pipe(int(x) for x in above_mean_flags),
        "above_median_flags": _pipe(int(x) for x in above_median_flags),
        "flipped": flipped,
        "final_adv_flags": _pipe(int(x) for x in final_adv_flags),
        "final_adv_flag_indices": _pipe(int(i) for i in np.where(final_adv_flags)[0]),
        "n_pairs": stage2.n_pairs,
        "top_pairs": _pipe(f"{i}-{j}:{sim:.4f}" for i, j, sim in stage2.top_pairs),
        "pp_top_pair_count": pp_count,
        "pc_top_pair_count": pc_count,
        "cc_top_pair_count": cc_count,
        "top_pair_pp": top_pair_pp,
        "removed_indices": _pipe(removed_indices),
        "removed_poison": removed_poison,
        "removed_clean": removed_clean,
        "residual_poison": residual_poison,
        "residual_clean": residual_clean,
        "residual_poison_fraction": residual_poison_fraction,
    }


def _summary_row(legacy: Dict, paper: Dict) -> Dict:
    legacy_flag_set = set(int(x) for x in legacy["final_adv_flag_indices"].split("|") if x != "")
    paper_flag_set = set(int(x) for x in paper["final_adv_flag_indices"].split("|") if x != "")
    legacy_removed_set = set(int(x) for x in legacy["removed_indices"].split("|") if x != "")
    paper_removed_set = set(int(x) for x in paper["removed_indices"].split("|") if x != "")

    n_adv_delta = paper["n_adv_estimated"] - legacy["n_adv_estimated"]
    legacy_success = legacy["residual_poison"] == 0
    paper_success = paper["residual_poison"] == 0

    return {
        "query_id": legacy["query_id"],
        "n_retrieved_poison": legacy["n_retrieved_poison"],
        "n_retrieved_clean": legacy["n_retrieved_clean"],
        "n_adv_legacy": legacy["n_adv_estimated"],
        "n_adv_paper": paper["n_adv_estimated"],
        "n_adv_abs_delta": abs(n_adv_delta),
        "n_adv_signed_delta_paper_minus_legacy": n_adv_delta,
        "n_adv_changed": legacy["n_adv_estimated"] != paper["n_adv_estimated"],
        "stage1_flag_set_legacy": _pipe(sorted(legacy_flag_set)),
        "stage1_flag_set_paper": _pipe(sorted(paper_flag_set)),
        "stage1_flag_set_changed": legacy_flag_set != paper_flag_set,
        "n_pairs_legacy": legacy["n_pairs"],
        "n_pairs_paper": paper["n_pairs"],
        "top_pairs_legacy": legacy["top_pairs"],
        "top_pairs_paper": paper["top_pairs"],
        "pair_set_changed": legacy["top_pairs"] != paper["top_pairs"],
        "removed_set_legacy": _pipe(sorted(legacy_removed_set)),
        "removed_set_paper": _pipe(sorted(paper_removed_set)),
        "removal_decision_changed": legacy_removed_set != paper_removed_set,
        "removed_poison_legacy": legacy["removed_poison"],
        "removed_poison_paper": paper["removed_poison"],
        "removed_clean_legacy": legacy["removed_clean"],
        "removed_clean_paper": paper["removed_clean"],
        "residual_poison_fraction_legacy": legacy["residual_poison_fraction"],
        "residual_poison_fraction_paper": paper["residual_poison_fraction"],
        "legacy_success_zero_residual_poison": legacy_success,
        "paper_success_zero_residual_poison": paper_success,
        "success_status_changed": legacy_success != paper_success,
    }


def run_gate_a() -> Tuple[List[Dict], List[Dict]]:
    with open(CLUSTER_VIZ_RUN_DIR / "run_config.json") as f:
        run_config = json.load(f)
    query_ids = run_config["query_ids_processed"]

    per_query_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    for query_id in query_ids:
        case = _load_case(query_id)
        legacy_row = _run_variant(case, "legacy")
        paper_row = _run_variant(case, "paper")
        per_query_rows.append(legacy_row)
        per_query_rows.append(paper_row)
        summary_rows.append(_summary_row(legacy_row, paper_row))

    return per_query_rows, summary_rows


def _write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_pct(n: int, total: int) -> str:
    return f"{n}/{total} ({100.0 * n / total:.1f}%)" if total else "0/0"


def write_report(summary_rows: List[Dict], path: Path) -> None:
    n = len(summary_rows)
    n_adv_changed = sum(1 for r in summary_rows if r["n_adv_changed"])
    abs_deltas = [r["n_adv_abs_delta"] for r in summary_rows]
    mean_abs_delta = sum(abs_deltas) / n
    median_abs_delta = float(np.median(abs_deltas))
    pair_set_changed = sum(1 for r in summary_rows if r["pair_set_changed"])
    removal_changed = sum(1 for r in summary_rows if r["removal_decision_changed"])
    success_changed = sum(1 for r in summary_rows if r["success_status_changed"])

    by_qid = {r["query_id"]: r for r in summary_rows}

    lines: List[str] = []
    lines.append("# Gate A -- RAGDefender Stage-1 Logic-Isolation Diagnostic Report")
    lines.append("")
    lines.append(
        "> **Gate A is a logic-isolation diagnostic using legacy MiniLM geometry. "
        "It is NOT validation of the final-paper RAGDefender implementation, "
        "because the final paper specifies Stella.**"
    )
    lines.append("")
    lines.append(
        "This diagnostic reuses the already-saved MiniLM (`paraphrase-MiniLM-L6-v2`) "
        "cosine similarity matrices from the existing instrumented HotpotQA k=10/N=5 "
        "runs (`results/diagnostics/ragdefender_cluster_viz/`). No embeddings were "
        "recomputed, no retrieval was run, and no generation/LLM API calls were made. "
        "It isolates the effect of the Stage-1 *equation* change "
        "(`ragdefender_legacy`: OR logic, diagonal-inclusive mean/median, a hybrid "
        "threshold, a flip branch -- vs. `ragdefender_paper`: AND logic, "
        "self-excluded 1/(k-1) mean/median, a single median-of-medians threshold, "
        "no flip branch) from the effect of the *encoder* change (MiniLM -> Stella), "
        "by holding the similarity geometry fixed and swapping only the Stage-1 "
        "decision logic. Stage 2 (pair-frequency identification, Eq. 4-7, `p=2`) is "
        "shared and unchanged between variants."
    )
    lines.append("")
    lines.append(
        f"**Queries evaluated:** {n} (all 8 instrumented HotpotQA k=10/N=5 cases with "
        "a saved MiniLM similarity matrix and recoverable passage text; 2 additional "
        "cases were skipped upstream for text-recovery mismatches and have no matrix)."
    )
    lines.append("")
    lines.append("## Summary statistics (legacy vs. paper Stage 1, same matrix)")
    lines.append("")
    lines.append(f"1. **How often `N_adv` changes:** {_fmt_pct(n_adv_changed, n)} of queries.")
    lines.append(
        f"2. **Mean/median absolute change in `N_adv`:** mean = {mean_abs_delta:.2f}, "
        f"median = {median_abs_delta:.2f} (per-query deltas: "
        + ", ".join(f"{r['query_id'][:8]}={r['n_adv_signed_delta_paper_minus_legacy']:+d}" for r in summary_rows)
        + ")."
    )
    lines.append(
        f"3. **How often the Stage-2 pair set changes:** {_fmt_pct(pair_set_changed, n)} of queries "
        "(a change here can follow purely from a different `N_adv` changing `N_pairs = max(1, C(N_adv,2))`, "
        "even when the underlying similarity ranking is identical)."
    )
    lines.append(
        f"4. **How often the removal decision changes:** {_fmt_pct(removal_changed, n)} of queries "
        "(the *set* of passage indices Stage 2 selects for removal differs between variants)."
    )
    lines.append(
        f"5. **Whether legacy failure categories survive paper Stage-1 logic:** "
        f"{_fmt_pct(success_changed, n)} of queries flip their zero-residual-poison "
        "success/failure status when switching Stage-1 logic on the same MiniLM matrix "
        "(see per-case inspection below for which direction each flip goes)."
    )
    lines.append("")
    lines.append("## Note: per-passage median values are numerically identical between variants here")
    lines.append("")
    lines.append(
        "Inspecting `gate_a_per_query.csv`, the per-passage `median_i` vectors and the global "
        "`median_threshold` are identical between `legacy` and `paper` for every one of these 8 "
        "queries -- this is a real mathematical property of this dataset, not a computation bug. "
        "`_torch_style_median_1d` returns the *lower* of the two middle values for an even-length "
        "input. Self-similarity (the diagonal, `S_ii = 1.0`) is always the maximum value in its row "
        "(cosine similarity is bounded by 1), so appending it to the sorted off-diagonal values never "
        "shifts the lower-median index for even `k`: with `k=10`, the paper's self-excluded median "
        "(lower of 2 middle values of 9 sorted off-diagonal entries, index 4) and the legacy's "
        "self-included median (lower of 2 middle values of 10 sorted entries including the "
        "appended-at-the-end diagonal maximum, index 4) land on the exact same array position. "
        "**This means, for this even-`k` dataset, none of the observed `N_adv`/removal differences "
        "come from the median formula itself** -- they come entirely from (a) the mean now excluding "
        "the diagonal with a `1/(k-1)` denominator, (b) AND vs. OR combination, (c) a pure "
        "median-of-medians threshold vs. the legacy hybrid `(avg_median + avg_avg)/2` threshold, and "
        "(d) the removed flip branch. This coincidence would not necessarily hold for odd `k` (e.g. "
        "HotpotQA's nominal `k=2`... though `k=2` cannot use a k-1=1-element median meaningfully) or "
        "for a matrix where a near-duplicate off-diagonal similarity ties or exceeds the diagonal."
    )
    lines.append("")
    lines.append("## Per-case inspection (as requested)")
    lines.append("")

    def _case_block(qid: str, label: str) -> List[str]:
        r = by_qid[qid]
        block = [f"### {label}: `{qid}`", ""]
        block.append(
            f"- Retrieved composition: {r['n_retrieved_poison']} poison / {r['n_retrieved_clean']} clean "
            f"(out of {r['n_retrieved_poison'] + r['n_retrieved_clean']})"
        )
        block.append(f"- `N_adv`: legacy = {r['n_adv_legacy']}, paper = {r['n_adv_paper']} (delta = {r['n_adv_signed_delta_paper_minus_legacy']:+d})")
        block.append(f"- Stage-1 flagged-index set: legacy = {{{r['stage1_flag_set_legacy']}}}, paper = {{{r['stage1_flag_set_paper']}}} -- changed: {r['stage1_flag_set_changed']}")
        block.append(f"- `N_pairs`: legacy = {r['n_pairs_legacy']}, paper = {r['n_pairs_paper']}")
        block.append(f"- Stage-2 removal decision changed: {r['removal_decision_changed']} (legacy removed = {{{r['removed_set_legacy']}}}, paper removed = {{{r['removed_set_paper']}}})")
        block.append(
            f"- Residual poison fraction: legacy = {r['residual_poison_fraction_legacy']:.3f} "
            f"(removed_poison={r['removed_poison_legacy']}, removed_clean={r['removed_clean_legacy']}), "
            f"paper = {r['residual_poison_fraction_paper']:.3f} "
            f"(removed_poison={r['removed_poison_paper']}, removed_clean={r['removed_clean_paper']})"
        )
        block.append(f"- Zero-residual-poison success status changed: {r['success_status_changed']}")
        block.append("")
        return block

    lines.extend(_case_block(LEGACY_OVERESTIMATION_CASE, "Legacy N_adv over-estimation case"))
    lines.extend(_case_block(CLEAN_DENSITY_CASE, "Clean-density case (legacy known-failure control)"))
    lines.append("### Previously-successful cases (six-case cohort)")
    lines.append("")
    lines.append(
        "These are the six queries previously identified as `ragdefender_legacy` "
        "zero-residual-poison successes on this same MiniLM geometry "
        "(`results/diagnostics/cluster_normalized_poisoning/BATCH_COMPARISON_SUCCESS_CASES.md`). "
        "Per correction 3 (§8a of the fidelity-correction plan), whether these cases "
        "remain successful under paper Stage-1 logic on this MiniLM matrix does **not** "
        "by itself establish paper fidelity -- Gate B (Stella re-encoding) is the actual "
        "fidelity gate and may identify a different case population."
    )
    lines.append("")
    lines.append("| query_id | legacy N_adv | paper N_adv | legacy residual poison frac | paper residual poison frac | success status changed |")
    lines.append("|---|---|---|---|---|---|")
    for qid in PREVIOUSLY_SUCCESSFUL_CASES:
        r = by_qid[qid]
        lines.append(
            f"| `{qid}` | {r['n_adv_legacy']} | {r['n_adv_paper']} | "
            f"{r['residual_poison_fraction_legacy']:.3f} | {r['residual_poison_fraction_paper']:.3f} | "
            f"{r['success_status_changed']} |"
        )
    lines.append("")

    n_still_success = sum(1 for qid in PREVIOUSLY_SUCCESSFUL_CASES if not by_qid[qid]["success_status_changed"])
    lines.append(
        f"**{n_still_success} of {len(PREVIOUSLY_SUCCESSFUL_CASES)}** previously-successful cases keep "
        "zero residual poison under paper Stage-1 logic on this (MiniLM) matrix; "
        f"**{len(PREVIOUSLY_SUCCESSFUL_CASES) - n_still_success}** flip status. This is a "
        "logic-only observation on legacy geometry -- see the scope note above and Gate B "
        "for the actual fidelity determination."
    )
    lines.append("")
    lines.append("## Full per-query comparison table")
    lines.append("")
    lines.append("| query_id | poison/clean | N_adv (L) | N_adv (P) | Δ | N_pairs (L) | N_pairs (P) | pair set Δ | removal Δ | resid.poison.frac (L) | resid.poison.frac (P) | success Δ |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary_rows:
        lines.append(
            f"| `{r['query_id']}` | {r['n_retrieved_poison']}/{r['n_retrieved_clean']} | "
            f"{r['n_adv_legacy']} | {r['n_adv_paper']} | {r['n_adv_signed_delta_paper_minus_legacy']:+d} | "
            f"{r['n_pairs_legacy']} | {r['n_pairs_paper']} | {r['pair_set_changed']} | "
            f"{r['removal_decision_changed']} | {r['residual_poison_fraction_legacy']:.3f} | "
            f"{r['residual_poison_fraction_paper']:.3f} | {r['success_status_changed']} |"
        )
    lines.append("")
    lines.append("## Data files")
    lines.append("")
    lines.append("- `gate_a_per_query.csv` -- one row per (query_id, variant) with every requested intermediate "
                  "(per-passage `mean_i`/`median_i` vectors, thresholds, Stage-1 boolean flags, Stage-2 `N_pairs`, "
                  "top-pair PP/PC/CC counts, `top_pair_pp`, removed/residual counts) for both `legacy` and `paper` rows.")
    lines.append("- `gate_a_summary.csv` -- one row per query_id, legacy-vs-paper diff table backing the statistics above.")
    lines.append("")
    lines.append("## Scope reminder")
    lines.append("")
    lines.append(
        "This report does not modify any manuscript claim. Per instructions, no oracle "
        "(E1), CORAL, or MMD reruns were performed, and Gate B (Stella re-encoding) was "
        "not run as part of this diagnostic."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    per_query_rows, summary_rows = run_gate_a()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(per_query_rows, OUTPUT_DIR / "gate_a_per_query.csv")
    _write_csv(summary_rows, OUTPUT_DIR / "gate_a_summary.csv")
    write_report(summary_rows, OUTPUT_DIR / "GATE_A_LOGIC_ISOLATION_REPORT.md")
    print(f"Gate A complete: {len(summary_rows)} queries evaluated.")
    print(f"Wrote: {OUTPUT_DIR / 'gate_a_per_query.csv'}")
    print(f"Wrote: {OUTPUT_DIR / 'gate_a_summary.csv'}")
    print(f"Wrote: {OUTPUT_DIR / 'GATE_A_LOGIC_ISOLATION_REPORT.md'}")


if __name__ == "__main__":
    main()
