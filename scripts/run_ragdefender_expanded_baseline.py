"""STEP 4 -- Expanded paper-faithful baseline (`ragdefender_paper` + Stella,
HotpotQA k=10) over the prospectively frozen population (STEP 3).

==========================================================================
SCOPE
==========================================================================
Reads ONLY the frozen population written by
`scripts/build_ragdefender_expanded_population.py`
(`results/diagnostics/ragdefender_expanded_baseline/recovered_contexts.json`)
-- never re-derives or re-selects the query list. For each of the 42
frozen queries: encodes the already-recovered passage texts with Stella,
computes paper-faithful Stage 1 (`concentration_stage1_paper`) and Stage 2
(`stage2_pair_frequency`), and records every requested field (context,
Stage-1 vectors/flags/count-error metrics, Stage-2 pair/removal detail,
outcome).

This is a BASELINE DEFENSE EVALUATION ONLY -- no oracle intervention.
Zero retrieval, zero generation, zero E1/CORAL/MMD, zero LLM/API calls.
Writes ONLY to `results/diagnostics/ragdefender_expanded_baseline/`, and
refuses to overwrite `PROSPECTIVE_POPULATION_FREEZE.md`,
`prospective_population.csv`, or `recovered_contexts.json` (STEP 3's
artifacts) -- appends new files (`expanded_baseline_per_query.csv`,
`expanded_baseline_by_regime.csv`, `EXPANDED_BASELINE_REPORT.md`,
`similarity/`, `embeddings/`) alongside them.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from defense import defense_runner, ragdefender_internals  # noqa: E402
import ragdefender_expanded_population_lib as poplib  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_baseline"
STELLA_MODEL_NAME = "dunzhang/stella_en_1.5B_v5"
REQUESTED_DEVICE = "cpu"


class ExpandedBaselineStopCondition(RuntimeError):
    pass


def _pipe(values) -> str:
    return "|".join(str(v) for v in values)


def _classify_pair(i: int, j: int, is_poison: np.ndarray) -> str:
    pi, pj = bool(is_poison[i]), bool(is_poison[j])
    if pi and pj:
        return "PP"
    if pi != pj:
        return "PC"
    return "CC"


# ---------------------------------------------------------------------------
# Frozen-population loading (read-only; never re-derives the list)
# ---------------------------------------------------------------------------

def load_frozen_population() -> List[dict]:
    path = OUTPUT_DIR / "recovered_contexts.json"
    if not path.exists():
        raise ExpandedBaselineStopCondition(
            f"{path} not found -- run scripts/build_ragdefender_expanded_population.py (STEP 3) first."
        )
    with open(path) as f:
        contexts = json.load(f)
    if not contexts:
        raise ExpandedBaselineStopCondition("Frozen population is empty.")
    return contexts


# ---------------------------------------------------------------------------
# Stella loading (identical pattern to Gate B)
# ---------------------------------------------------------------------------

def load_stella_model():
    cfg = defense_runner.DefenseConfig(ragdefender_version="paper", device=REQUESTED_DEVICE)
    try:
        s_model = defense_runner._get_s_model(cfg)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        raise ExpandedBaselineStopCondition(f"Stella failed to load: {exc!r}") from exc
    actual_device = str(s_model.device)
    if actual_device != REQUESTED_DEVICE:
        raise ExpandedBaselineStopCondition(
            f"Requested device {REQUESTED_DEVICE!r} != actual device {actual_device!r}."
        )
    return s_model, actual_device


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

def run_baseline_query(ctx: dict, s_model, st_util, output_dir: Path) -> dict:
    query_id = ctx["query_id"]
    texts = ctx["texts"]
    is_poison = np.array(ctx["is_poison"], dtype=bool)
    k = ctx["k"]
    m_poison = ctx["m_poison"]
    c_clean = ctx["c_clean"]

    try:
        embeddings = s_model.encode(texts, convert_to_tensor=True)
    except Exception as exc:  # noqa: BLE001
        raise ExpandedBaselineStopCondition(f"{query_id}: Stella failed to encode passages: {exc!r}") from exc

    import torch as _torch

    if not bool(_torch.isfinite(embeddings).all()):
        raise ExpandedBaselineStopCondition(f"{query_id}: non-finite Stella embeddings observed.")

    matrix = st_util.cos_sim(embeddings, embeddings).cpu().numpy().astype(np.float64)
    embeddings_np = embeddings.cpu().numpy().astype(np.float64)

    np.save(output_dir / "embeddings" / f"{query_id}_stella_embeddings.npy", embeddings_np)
    np.save(output_dir / "similarity" / f"{query_id}_stella_similarity_matrix.npy", matrix)

    stage1 = ragdefender_internals.concentration_stage1_paper(matrix)
    stage2 = ragdefender_internals.stage2_pair_frequency(matrix, n_adv=stage1.n_adv_estimated, p=2.0)

    n_adv = stage1.n_adv_estimated
    count_error = n_adv - m_poison
    if n_adv > k // 2:
        raise ExpandedBaselineStopCondition(
            f"{query_id}: n_adv={n_adv} exceeds structural ceiling floor(k/2)={k // 2} -- theorem/impl "
            "mismatch, STOP."
        )

    mean_margin = stage1.s_mean - stage1.s_bar
    median_margin = stage1.s_median - stage1.s_tilde
    min_abs_mean_margin = float(np.min(np.abs(mean_margin)))
    min_abs_median_margin = float(np.min(np.abs(median_margin)))

    pair_classes = [_classify_pair(i, j, is_poison) for i, j, _sim in stage2.top_pairs]
    pp_count = pair_classes.count("PP")
    pc_count = pair_classes.count("PC")
    cc_count = pair_classes.count("CC")
    top_pair_label = pair_classes[0] if pair_classes else None

    removed_indices = list(stage2.selected_indices)
    removed_poison = int(sum(1 for idx in removed_indices if is_poison[idx]))
    removed_clean = int(sum(1 for idx in removed_indices if not is_poison[idx]))
    residual_poison = m_poison - removed_poison
    residual_clean = c_clean - removed_clean

    denom = removed_poison + removed_clean
    removal_precision = (removed_poison / denom) if denom > 0 else None
    poison_removal_recall = (removed_poison / m_poison) if m_poison > 0 else None
    clean_removed_bool = removed_clean > 0

    return {
        # IDENTITY
        "query_id": query_id,
        # CONTEXT
        "k": k,
        "m_poison": m_poison,
        "c_clean": c_clean,
        "rho": ctx["rho"],
        "ceiling": ctx["ceiling"],
        "regime": ctx["regime"],
        # STAGE 1
        "s_mean_i": _pipe(f"{x:.6f}" for x in stage1.s_mean),
        "s_median_i": _pipe(f"{x:.6f}" for x in stage1.s_median),
        "s_bar": stage1.s_bar,
        "s_tilde": stage1.s_tilde,
        "above_mean_flags": _pipe(int(x) for x in stage1.above_mean),
        "above_median_flags": _pipe(int(x) for x in stage1.above_median),
        "and_flags": _pipe(int(x) for x in stage1.adv_flag),
        "n_adv": n_adv,
        "count_error": count_error,
        "abs_count_error": abs(count_error),
        "count_underestimated": count_error < 0,
        "count_overestimated": count_error > 0,
        "count_exact": count_error == 0,
        "min_abs_mean_margin": min_abs_mean_margin,
        "min_abs_median_margin": min_abs_median_margin,
        # STAGE 2
        "n_pairs": stage2.n_pairs,
        "selected_pairs": _pipe(f"{i}-{j}:{sim:.4f}" for i, j, sim in stage2.top_pairs),
        "selected_pair_classes": _pipe(pair_classes),
        "pp_count": pp_count,
        "pc_count": pc_count,
        "cc_count": cc_count,
        "top_pair_label": top_pair_label,
        "top_pair_pp": top_pair_label == "PP",
        "top_pair_pc": top_pair_label == "PC",
        "top_pair_cc": top_pair_label == "CC",
        "frequency_scores_i": _pipe(f"{x:.6f}" for x in stage2.frequency_scores),
        "removed_indices": _pipe(removed_indices),
        "removed_poison": removed_poison,
        "removed_clean": removed_clean,
        "residual_poison": residual_poison,
        "residual_clean": residual_clean,
        "removal_precision": removal_precision,
        "poison_removal_recall": poison_removal_recall,
        "clean_removed": clean_removed_bool,
        # OUTCOME
        "zero_residual_poison_success": residual_poison == 0,
        "is_poison_i": _pipe(int(x) for x in is_poison),
        "embeddings_path": str((output_dir / "embeddings" / f"{query_id}_stella_embeddings.npy").relative_to(REPO_ROOT)),
        "similarity_matrix_path": str((output_dir / "similarity" / f"{query_id}_stella_similarity_matrix.npy").relative_to(REPO_ROOT)),
    }


# ---------------------------------------------------------------------------
# Regime aggregation
# ---------------------------------------------------------------------------

REGIME_ORDER = ["A_BELOW_CEILING", "B_AT_CEILING", "C_ABOVE_CEILING", "D_ALL_POISON"]


def _mean(values: List[float]):
    return float(np.mean(values)) if values else None


def _median(values: List[float]):
    return float(np.median(values)) if values else None


def build_regime_aggregates(rows: List[dict]) -> List[dict]:
    aggregates: List[dict] = []
    for regime in REGIME_ORDER:
        regime_rows = [r for r in rows if r["regime"] == regime]
        n = len(regime_rows)
        agg = {"regime": regime, "n_queries": n}
        if n == 0:
            aggregates.append(agg)
            continue

        true_counts = [r["m_poison"] for r in regime_rows]
        n_advs = [r["n_adv"] for r in regime_rows]
        count_errors = [r["count_error"] for r in regime_rows]
        precisions = [r["removal_precision"] for r in regime_rows if r["removal_precision"] is not None]
        recalls = [r["poison_removal_recall"] for r in regime_rows if r["poison_removal_recall"] is not None]

        agg.update({
            "mean_true_poison_count": _mean(true_counts),
            "median_true_poison_count": _median(true_counts),
            "mean_n_adv": _mean(n_advs),
            "median_n_adv": _median(n_advs),
            "exact_count_rate": sum(1 for r in regime_rows if r["count_exact"]) / n,
            "undercount_rate": sum(1 for r in regime_rows if r["count_underestimated"]) / n,
            "overcount_rate": sum(1 for r in regime_rows if r["count_overestimated"]) / n,
            "mean_signed_count_error": _mean(count_errors),
            "zero_residual_poison_rate": sum(1 for r in regime_rows if r["zero_residual_poison_success"]) / n,
            "mean_removal_precision": _mean(precisions),
            "mean_poison_removal_recall": _mean(recalls),
            "clean_removal_rate": sum(1 for r in regime_rows if r["clean_removed"]) / n,
        })
        aggregates.append(agg)

        if regime == "C_ABOVE_CEILING":
            for r in regime_rows:
                if not (r["n_adv"] <= r["ceiling"] < r["m_poison"]):
                    raise ExpandedBaselineStopCondition(
                        f"{r['query_id']}: Regime C invariant n_adv <= floor(k/2) < M FAILED "
                        f"(n_adv={r['n_adv']}, ceiling={r['ceiling']}, m_poison={r['m_poison']})."
                    )
    return aggregates


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _check_no_overwrite(paths: List[Path]) -> None:
    existing = [p for p in paths if p.exists()]
    if existing:
        raise ExpandedBaselineStopCondition(f"Refusing to overwrite existing output artifact(s): {existing}")


def _write_csv(rows: List[dict], path: Path) -> None:
    """Fieldnames are the UNION of keys across all rows (first-seen order),
    not merely `rows[0].keys()` -- e.g. an empty Regime-A aggregate (fewer
    keys than a populated regime) can legitimately be the first row in
    `regime_aggregates`, and `rows[0].keys()` alone would then omit keys
    only present on later, non-empty regime rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(rows: List[dict], regime_aggregates: List[dict], dependency_info: dict, path: Path) -> None:
    n = len(rows)
    n_success = sum(1 for r in rows if r["zero_residual_poison_success"])
    n_exact = sum(1 for r in rows if r["count_exact"])
    n_under = sum(1 for r in rows if r["count_underestimated"])
    n_over = sum(1 for r in rows if r["count_overestimated"])

    lines: List[str] = []
    lines.append("# Expanded Paper-Faithful Baseline Report (STEP 4)")
    lines.append("")
    lines.append(
        "> `ragdefender_paper` + Stella, HotpotQA k=10, over the 42-query PROSPECTIVELY FROZEN population "
        "(see `PROSPECTIVE_POPULATION_FREEZE.md`). Baseline defense evaluation only -- no oracle "
        "intervention. No retrieval, generation, E1, CORAL, MMD, or LLM/API experiment was run. No new "
        "retrieval was run."
    )
    lines.append("")
    lines.append("## Dependency / environment record")
    lines.append("")
    for k, v in dependency_info.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append(f"**Queries evaluated:** {n} (all 42 frozen queries).")
    lines.append("")

    lines.append("## Headline results")
    lines.append("")
    lines.append(f"- Zero-residual-poison success rate: **{n_success}/{n}** ({n_success / n:.1%})")
    lines.append(f"- Exact-count rate: **{n_exact}/{n}** ({n_exact / n:.1%})")
    lines.append(f"- Undercount rate: **{n_under}/{n}** ({n_under / n:.1%})")
    lines.append(f"- Overcount rate: **{n_over}/{n}** ({n_over / n:.1%})")
    lines.append("")

    lines.append("## Results by poison-count regime (STEP 4B)")
    lines.append("")
    lines.append(
        "| regime | n | mean true poison | median true poison | mean N_adv | median N_adv | exact-count rate "
        "| undercount rate | mean signed count error | zero-residual-poison rate | mean removal precision "
        "| mean poison recall | clean-removal rate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for agg in regime_aggregates:
        if agg["n_queries"] == 0:
            lines.append(f"| {agg['regime']} | 0 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| {agg['regime']} | {agg['n_queries']} | {agg['mean_true_poison_count']:.2f} | "
            f"{agg['median_true_poison_count']:.1f} | {agg['mean_n_adv']:.2f} | {agg['median_n_adv']:.1f} | "
            f"{agg['exact_count_rate']:.1%} | {agg['undercount_rate']:.1%} | "
            f"{agg['mean_signed_count_error']:+.2f} | {agg['zero_residual_poison_rate']:.1%} | "
            + (f"{agg['mean_removal_precision']:.2f}" if agg["mean_removal_precision"] is not None else "N/A")
            + " | "
            + (f"{agg['mean_poison_removal_recall']:.2f}" if agg["mean_poison_removal_recall"] is not None else "N/A")
            + f" | {agg['clean_removal_rate']:.1%} |"
        )
    lines.append("")

    regime_c = next((a for a in regime_aggregates if a["regime"] == "C_ABOVE_CEILING"), None)
    if regime_c and regime_c["n_queries"] > 0:
        lines.append(
            f"**Regime C invariant check (n_adv <= floor(k/2) < M):** verified for all "
            f"{regime_c['n_queries']} Regime-C queries -- no violation (would have raised "
            "`ExpandedBaselineStopCondition` and aborted this run if it had failed)."
        )
        lines.append("")

    lines.append("## Per-query detail")
    lines.append("")
    lines.append("| query_id | regime | M | C | N_adv | count_error | residual_poison | removal_precision | success |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| `{r['query_id']}` | {r['regime']} | {r['m_poison']} | {r['c_clean']} | {r['n_adv']} | "
            f"{r['count_error']:+d} | {r['residual_poison']} | "
            + (f"{r['removal_precision']:.2f}" if r["removal_precision"] is not None else "N/A")
            + f" | {r['zero_residual_poison_success']} |"
        )
    lines.append("")

    lines.append("## Data files")
    lines.append("")
    lines.append("- `expanded_baseline_per_query.csv` -- every requested field per query.")
    lines.append("- `expanded_baseline_by_regime.csv` -- regime-level aggregates (STEP 4B).")
    lines.append("- `similarity/{query_id}_stella_similarity_matrix.npy` -- full k x k Stella similarity matrix.")
    lines.append("- `embeddings/{query_id}_stella_embeddings.npy` -- raw Stella passage embeddings.")
    lines.append("")
    lines.append(
        "## Scope reminder\n\nThis is a baseline defense evaluation only; no oracle intervention. "
        "See `results/diagnostics/ragdefender_expanded_gate_c/` (STEP 5) for the oracle-count "
        "decomposition over this SAME frozen population and its saved matrices."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    import sentence_transformers
    import torch as _torch
    import transformers

    dependency_info = {
        "stella_model": STELLA_MODEL_NAME,
        "transformers_version": transformers.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": _torch.__version__,
        "requested_device": REQUESTED_DEVICE,
    }

    out_per_query_csv = OUTPUT_DIR / "expanded_baseline_per_query.csv"
    out_by_regime_csv = OUTPUT_DIR / "expanded_baseline_by_regime.csv"
    out_report_md = OUTPUT_DIR / "EXPANDED_BASELINE_REPORT.md"
    _check_no_overwrite([out_per_query_csv, out_by_regime_csv, out_report_md])

    (OUTPUT_DIR / "similarity").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "embeddings").mkdir(parents=True, exist_ok=True)

    contexts = load_frozen_population()
    s_model, actual_device = load_stella_model()
    dependency_info["actual_device"] = actual_device
    _, st_util = defense_runner._lazy_st()  # noqa: SLF001

    rows = []
    for i, ctx in enumerate(contexts):
        print(f"[{i + 1}/{len(contexts)}] {ctx['query_id']} (k={ctx['k']}, M={ctx['m_poison']}, "
              f"C={ctx['c_clean']}, regime={ctx['regime']})...")
        rows.append(run_baseline_query(ctx, s_model, st_util, OUTPUT_DIR))

    regime_aggregates = build_regime_aggregates(rows)

    _write_csv(rows, out_per_query_csv)
    _write_csv(regime_aggregates, out_by_regime_csv)
    write_report(rows, regime_aggregates, dependency_info, out_report_md)

    print(f"Expanded baseline complete: {len(rows)} queries evaluated on device={actual_device}.")
    print(f"Wrote: {out_per_query_csv}")
    print(f"Wrote: {out_by_regime_csv}")
    print(f"Wrote: {out_report_md}")


if __name__ == "__main__":
    main()
