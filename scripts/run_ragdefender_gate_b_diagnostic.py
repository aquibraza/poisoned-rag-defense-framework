"""Gate B -- RAGDefender true paper-fidelity gate (Stella re-encoding).

==========================================================================
SCOPE (read before interpreting output)
==========================================================================
Gate B evaluates the FINAL-PAPER-FAITHFUL RAGDefender implementation
(`ragdefender_paper`: self-excluded AND-logic Stage 1, shared Stage 2,
Stella embedder) on our FIXED HotpotQA k=10/N=5 stress contexts.

It is NOT a reproduction of the published HotpotQA performance numbers,
because the final paper's nominal HotpotQA setting is k=2, not k=10.

It establishes:
  - final-paper Stage-1 arithmetic on real passage text;
  - final-paper Stage-2 arithmetic on real passage text;
  - Stella-derived internal passage embeddings/similarity geometry;
  - the behavior of that implementation on our EXISTING fixed retrieved
    contexts (no new retrieval).

This script performs ZERO retrieval, ZERO generation, ZERO E1/CORAL/MMD
experiments, and ZERO GPT/OpenAI/Anthropic API calls. The only "new"
computation relative to Gate A is loading Stella and encoding the same 8
queries' already-fixed retrieved passage texts.

==========================================================================
INPUT POPULATION
==========================================================================
The same 8 recoverable instrumented HotpotQA k=10/N=5 query contexts used
by Gate A (results/diagnostics/ragdefender_cluster_viz/20260722_042137_.../).
Passage text is recovered from the paired query_results/*.json record's
`input_prompt_no_defense` field, via the exact same
`scripts.visualize_ragdefender_clusters.recover_pre_defense_texts` /
`load_query_results_index` functions Gate A's source data was built with --
i.e. this script re-derives the identical texts, it does not invent a new
retrieval. The two upstream queries lacking reliable saved text
(text-recovery mismatches) are NOT attempted here either.

==========================================================================
STOP CONDITIONS (see module-level `GateBStopCondition`)
==========================================================================
This script raises `GateBStopCondition` and writes NO CSV/report output if:
  - Stella fails to encode any query;
  - non-finite embeddings are observed;
  - the requested device does not match the actual device Stella loaded on;
  - the recovered passage count/labels do not match the saved Gate-A
    passages.csv context for that query_id;
  - any Gate-B output path already exists (refuses to overwrite);
  - the production compatibility shim
    (`defense_runner._apply_stella_dynamic_cache_compat_shim`) fails.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from defense import defense_runner, ragdefender_internals  # noqa: E402
import visualize_ragdefender_clusters as viz  # noqa: E402

GATE_A_RUN_DIR = (
    REPO_ROOT
    / "results/diagnostics/ragdefender_cluster_viz"
    / "20260722_042137_clusterdiag_hotpotqa_k10_N5_ragdefender-original_"
    "embedder-paraphraseMiniLM_task-multihop_p2"
)
GATE_A_OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_gate_a"
OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_gate_b"

STELLA_MODEL_NAME = "dunzhang/stella_en_1.5B_v5"
REQUESTED_DEVICE = "cpu"

LEGACY_OVERESTIMATION_CASE = "5a722b8655429971e9dc9329"
CLEAN_DENSITY_CASE = "5a8cb288554299585d9e3726"
PREVIOUSLY_SUCCESSFUL_CASES = [
    "5adbf0a255429947ff17385a",
    "5ab56e32554299637185c594",
    "5ab29c24554299449642c932",
    "5ae6050f55429929b0807a5e",
    "5ae2070a5542994d89d5b313",
    "5a722b8655429971e9dc9329",
]


class GateBStopCondition(RuntimeError):
    """Raised for any of the documented STOP conditions. Callers must not
    proceed to write output CSVs/reports if this is raised."""


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
# Input recovery + STOP-condition consistency check against Gate A
# ---------------------------------------------------------------------------

def _load_gate_a_context() -> Tuple[dict, Dict[str, np.ndarray]]:
    """Load run_config.json + per-query is_poison ground truth from the
    already-saved Gate-A/cluster-viz passages CSVs, purely for the
    consistency check below -- never written to."""
    with open(GATE_A_RUN_DIR / "run_config.json") as f:
        run_config = json.load(f)
    gate_a_is_poison: Dict[str, np.ndarray] = {}
    for qid in run_config["query_ids_processed"]:
        passages = pd.read_csv(GATE_A_RUN_DIR / "passages" / f"{qid}_passages.csv")
        passages = passages.sort_values("passage_index").reset_index(drop=True)
        gate_a_is_poison[qid] = passages["is_poison"].astype(bool).to_numpy()
    return run_config, gate_a_is_poison


def _recover_case_texts(query_id: str, diagnostics_records: List[dict], query_results_index: dict) -> dict:
    rec = next((r for r in diagnostics_records if r["query_id"] == query_id), None)
    if rec is None:
        raise GateBStopCondition(f"{query_id}: no diagnostics record found -- cannot recover context.")

    qr = query_results_index.get(query_id)
    texts = viz.recover_pre_defense_texts(qr)
    if texts is None:
        raise GateBStopCondition(
            f"{query_id}: passage text not recoverable (query_results record missing or "
            "input_prompt_no_defense absent) -- this query must not be attempted per the "
            "'do not fabricate/reconstruct' instruction."
        )

    doc_ids = list(rec["retrieved_doc_ids"])
    is_poison = np.array([bool(x) for x in rec["retrieved_is_poison"]])
    k = len(doc_ids)

    if len(texts) != k:
        raise GateBStopCondition(
            f"{query_id}: recovered {len(texts)} passage texts, expected k={k} "
            "(rank-order parse mismatch) -- STOP per passage-count-mismatch condition."
        )

    return {
        "query_id": query_id,
        "texts": texts,
        "doc_ids": doc_ids,
        "is_poison": is_poison,
        "k": k,
        "n_retrieved_poison": int(is_poison.sum()),
        "n_retrieved_clean": int((~is_poison).sum()),
    }


def _verify_matches_gate_a(case: dict, gate_a_is_poison: Dict[str, np.ndarray]) -> None:
    qid = case["query_id"]
    if qid not in gate_a_is_poison:
        raise GateBStopCondition(f"{qid}: not present in the saved Gate-A passages context.")
    expected = gate_a_is_poison[qid]
    if len(expected) != case["k"]:
        raise GateBStopCondition(
            f"{qid}: passage count mismatch vs. saved Gate-A context "
            f"(Gate B recovered k={case['k']}, Gate A had k={len(expected)})."
        )
    if not np.array_equal(expected, case["is_poison"]):
        raise GateBStopCondition(
            f"{qid}: is_poison labels do not match the saved Gate-A context "
            f"(Gate B={case['is_poison'].tolist()}, Gate A={expected.tolist()})."
        )


# ---------------------------------------------------------------------------
# Stella loading with explicit STOP-condition checks
# ---------------------------------------------------------------------------

def _load_stella_model():
    cfg = defense_runner.DefenseConfig(ragdefender_version="paper", device=REQUESTED_DEVICE)
    try:
        s_model = defense_runner._get_s_model(cfg)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any load/shim failure must STOP
        raise GateBStopCondition(
            f"Stella failed to load (production compatibility shim or model load failure): {exc!r}"
        ) from exc

    actual_device = str(s_model.device)
    if actual_device != REQUESTED_DEVICE:
        raise GateBStopCondition(
            f"Requested device {REQUESTED_DEVICE!r} != actual device {actual_device!r} -- STOP per "
            "device-mismatch condition."
        )
    return s_model, cfg, actual_device


# ---------------------------------------------------------------------------
# Per-query Gate-B computation
# ---------------------------------------------------------------------------

def _geometry_stats(matrix: np.ndarray, is_poison: np.ndarray) -> Dict:
    k = matrix.shape[0]
    buckets: Dict[str, List[float]] = {"PP": [], "PC": [], "CC": []}
    for i in range(k):
        for j in range(i + 1, k):
            buckets[_classify_pair(i, j, is_poison)].append(float(matrix[i, j]))
    out = {}
    for label in ("PP", "PC", "CC"):
        vals = buckets[label]
        out[f"mean_{label.lower()}_similarity"] = float(np.mean(vals)) if vals else None
        out[f"max_{label.lower()}_similarity"] = float(np.max(vals)) if vals else None
    return out


def _classify_query(
    removed_poison: int,
    removed_clean: int,
    residual_poison: int,
    top_pair_label: Optional[str],
    pp_count: int,
    pc_count: int,
    cc_count: int,
) -> List[str]:
    """Descriptive, multi-label, from-scratch mechanism classification under
    `ragdefender_paper` -- deliberately NOT forced into the old legacy
    taxonomy. Rules are heuristic and documented here (not a formal
    definition); borderline cases should still be inspected manually.
    A query may receive more than one label.

    IMPORTANT (Gate-B follow-up STEP 2 correction): this function takes
    ONLY Stage-2 evidence (selected-pair PP/PC/CC composition, the actual
    removed/residual counts) as input. It must NEVER be given the Stage-1
    concentration AND-flags (`ConcentrationResultPaper.adv_flag`) or any
    quantity derived from them (e.g. "how many Stage-1-flagged passages
    are poison vs. clean"). Stage 1's flags are COUNT-ESTIMATION INDICATOR
    FLAGS ONLY -- they determine N_adv (a single integer), not which
    specific passages get removed; Stage 2 makes that decision
    independently via its own frequency-score ranking over ALL k passages,
    which can (and in this codebase's Gate-B run, does) select a different
    index set than the one Stage 1 flagged. Treating Stage-1 flags as if
    they were "the predicted adversarial subset" was the specific taxonomy
    bug this correction removes; the previous implementation of this
    function took `is_poison`/`adv_flag` and inferred a
    "clean-density / clean-top-pair failure" label whenever more
    Stage-1-flagged passages were clean than poison -- that inference path
    has been deleted, not merely renamed. A "clean-density / clean-top-pair"
    descriptor is now assigned ONLY from actual Stage-2 evidence: the
    single highest-similarity selected pair is CC, CC pairs are the
    plurality of the selected pair set, or Stage 2 actually removed a
    clean passage (which is also equivalent here to "clean passages
    dominate the top `N_adv` frequency-score ranking," since the removed
    set IS that top-`N_adv` ranking by construction)."""
    labels: List[str] = []

    if residual_poison == 0:
        labels.append("zero-residual-poison success")
    else:
        labels.append("residual-poison failure")

    if removed_clean > 0:
        labels.append("clean over-removal")

    cc_is_plurality = cc_count > pp_count and cc_count > pc_count
    if top_pair_label == "CC" or cc_is_plurality or removed_clean > removed_poison:
        labels.append("clean-density / clean-top-pair failure")

    n_categories_present = sum(1 for c in (pp_count, pc_count, cc_count) if c > 0)
    if n_categories_present >= 2 and residual_poison > 0:
        labels.append("mixed-pair failure")

    if pp_count == 0 and pc_count == 0 and cc_count == 0:
        labels.append("other / inspect manually (N_pairs too small to classify by composition)")

    return labels


def run_gate_b_query(case: dict, s_model, st_util, output_dir: Path) -> dict:
    query_id = case["query_id"]
    texts = case["texts"]
    is_poison = case["is_poison"]
    k = case["k"]

    try:
        embeddings = s_model.encode(texts, convert_to_tensor=True)
    except Exception as exc:  # noqa: BLE001
        raise GateBStopCondition(f"{query_id}: Stella failed to encode passages: {exc!r}") from exc

    import torch as _torch

    if not bool(_torch.isfinite(embeddings).all()):
        raise GateBStopCondition(f"{query_id}: non-finite Stella embeddings observed.")

    matrix = st_util.cos_sim(embeddings, embeddings).cpu().numpy().astype(np.float64)
    embeddings_np = embeddings.cpu().numpy().astype(np.float64)

    np.save(output_dir / "embeddings" / f"{query_id}_stella_embeddings.npy", embeddings_np)
    np.save(output_dir / "similarity" / f"{query_id}_stella_similarity_matrix.npy", matrix)

    stage1 = ragdefender_internals.concentration_stage1_paper(matrix)
    stage2 = ragdefender_internals.stage2_pair_frequency(matrix, n_adv=stage1.n_adv_estimated, p=2.0)

    mean_margin = stage1.s_mean - stage1.s_bar
    median_margin = stage1.s_median - stage1.s_tilde
    min_abs_mean_margin = float(np.min(np.abs(mean_margin)))
    min_abs_median_margin = float(np.min(np.abs(median_margin)))
    min_abs_threshold_margin = min(min_abs_mean_margin, min_abs_median_margin)

    pair_classes = [_classify_pair(i, j, is_poison) for i, j, _sim in stage2.top_pairs]
    pp_count = pair_classes.count("PP")
    pc_count = pair_classes.count("PC")
    cc_count = pair_classes.count("CC")
    top_pair_label = pair_classes[0] if pair_classes else None

    removed_indices = list(stage2.selected_indices)
    removed_poison = int(sum(1 for idx in removed_indices if is_poison[idx]))
    removed_clean = int(sum(1 for idx in removed_indices if not is_poison[idx]))
    residual_poison = case["n_retrieved_poison"] - removed_poison
    residual_clean = case["n_retrieved_clean"] - removed_clean
    residual_poison_fraction = (
        residual_poison / case["n_retrieved_poison"] if case["n_retrieved_poison"] > 0 else float("nan")
    )

    # STEP 2 correction: classification uses ONLY Stage-2 evidence
    # (removed/residual counts, selected-pair PP/PC/CC composition) --
    # never `stage1.adv_flag` or `is_poison[stage1.adv_flag]`. See
    # `_classify_query`'s docstring for why that Stage-1-flag-based
    # inference was removed, not merely relabeled.
    classification = _classify_query(
        removed_poison, removed_clean, residual_poison, top_pair_label, pp_count, pc_count, cc_count,
    )

    geometry = _geometry_stats(matrix, is_poison)

    # NOTE on the four `*flag*` fields below (STEP 2 taxonomy correction):
    # `above_mean_flags`/`above_median_flags`/`final_and_flags`/
    # `final_adv_flag_indices` are Stage-1 COUNT-ESTIMATION INDICATOR
    # FLAGS. Their only formal role is determining `n_adv` (a single
    # integer, via `sum(final_and_flags)`). They are NOT a predicted
    # adversarial-passage subset -- Stage 2 independently decides WHICH
    # `n_adv` passages actually get removed (`removed_indices` below,
    # ranked by `frequency_scores_i`, which can be, and on this dataset
    # sometimes is, a different index set than `final_adv_flag_indices`).
    # Kept here as useful intermediate diagnostics only; do not use them
    # to derive a mechanism-level classification -- see `_classify_query`.
    return {
        "query_id": query_id,
        "k": k,
        "n_retrieved_poison": case["n_retrieved_poison"],
        "n_retrieved_clean": case["n_retrieved_clean"],
        **geometry,
        "s_mean_i": _pipe(f"{x:.6f}" for x in stage1.s_mean),
        "s_median_i": _pipe(f"{x:.6f}" for x in stage1.s_median),
        "s_bar": stage1.s_bar,
        "s_tilde": stage1.s_tilde,
        "above_mean_flags": _pipe(int(x) for x in stage1.above_mean),
        "above_median_flags": _pipe(int(x) for x in stage1.above_median),
        "final_and_flags": _pipe(int(x) for x in stage1.adv_flag),
        "final_adv_flag_indices": _pipe(int(i) for i in np.where(stage1.adv_flag)[0]),
        "n_adv": stage1.n_adv_estimated,
        "mean_margin_i": _pipe(f"{x:.6f}" for x in mean_margin),
        "median_margin_i": _pipe(f"{x:.6f}" for x in median_margin),
        "min_abs_mean_margin": min_abs_mean_margin,
        "min_abs_median_margin": min_abs_median_margin,
        "min_abs_threshold_margin": min_abs_threshold_margin,
        "n_pairs": stage2.n_pairs,
        "selected_pairs": _pipe(f"{i}-{j}:{sim:.4f}" for i, j, sim in stage2.top_pairs),
        "selected_pair_classes": _pipe(pair_classes),
        "pp_top_pair_count": pp_count,
        "pc_top_pair_count": pc_count,
        "cc_top_pair_count": cc_count,
        "top_pair_pp": top_pair_label == "PP",
        "top_pair_pc": top_pair_label == "PC",
        "top_pair_cc": top_pair_label == "CC",
        "frequency_scores_i": _pipe(f"{x:.6f}" for x in stage2.frequency_scores),
        "ranked_passage_order": _pipe(
            idx for idx, _ in sorted(enumerate(stage2.frequency_scores), key=lambda t: t[1], reverse=True)
        ),
        "removed_indices": _pipe(removed_indices),
        "removed_poison": removed_poison,
        "removed_clean": removed_clean,
        "residual_poison": residual_poison,
        "residual_clean": residual_clean,
        "residual_poison_fraction": residual_poison_fraction,
        "classification": _pipe(classification),
        "is_poison_i": _pipe(int(x) for x in is_poison),
        "embeddings_path": str((output_dir / "embeddings" / f"{query_id}_stella_embeddings.npy").relative_to(REPO_ROOT)),
        "similarity_matrix_path": str((output_dir / "similarity" / f"{query_id}_stella_similarity_matrix.npy").relative_to(REPO_ROOT)),
    }


# ---------------------------------------------------------------------------
# Three-way comparison (Legacy / Gate A / Gate B)
# ---------------------------------------------------------------------------

def _gate_a_pair_composition(row: pd.Series) -> Tuple[bool, bool, bool]:
    """Recompute top_pair_pc/top_pair_cc from Gate A's saved `top_pairs` +
    `is_poison_i` string columns (Gate A only stored `top_pair_pp`)."""
    top_pairs_str = row["top_pairs"]
    is_poison_str = row["is_poison_i"]
    if not isinstance(top_pairs_str, str) or not top_pairs_str:
        return False, False, False
    is_poison = np.array([bool(int(x)) for x in is_poison_str.split("|")])
    first_pair = top_pairs_str.split("|")[0]
    ij_part = first_pair.split(":")[0]
    i_str, j_str = ij_part.split("-")
    label = _classify_pair(int(i_str), int(j_str), is_poison)
    return label == "PP", label == "PC", label == "CC"


def build_comparison_rows(gate_a_per_query_path: Path, gate_b_rows: List[dict]) -> List[dict]:
    gate_a_df = pd.read_csv(gate_a_per_query_path)
    gate_b_by_qid = {row["query_id"]: row for row in gate_b_rows}

    comparison_rows: List[dict] = []
    for query_id in gate_b_by_qid:
        legacy_row = gate_a_df[(gate_a_df["query_id"] == query_id) & (gate_a_df["variant"] == "legacy")].iloc[0]
        gate_a_row = gate_a_df[(gate_a_df["query_id"] == query_id) & (gate_a_df["variant"] == "paper")].iloc[0]
        gate_b_row = gate_b_by_qid[query_id]

        legacy_pp, legacy_pc, legacy_cc = _gate_a_pair_composition(legacy_row)
        gatea_pp, gatea_pc, gatea_cc = _gate_a_pair_composition(gate_a_row)

        row = {
            "query_id": query_id,
            "n_retrieved_poison": int(legacy_row["n_retrieved_poison"]),
            "n_retrieved_clean": int(legacy_row["n_retrieved_clean"]),

            "n_adv_legacy": int(legacy_row["n_adv_estimated"]),
            "n_adv_gate_a": int(gate_a_row["n_adv_estimated"]),
            "n_adv_gate_b": int(gate_b_row["n_adv"]),
            "n_adv_delta_legacy_to_gate_a": int(gate_a_row["n_adv_estimated"]) - int(legacy_row["n_adv_estimated"]),
            "n_adv_delta_gate_a_to_gate_b": int(gate_b_row["n_adv"]) - int(gate_a_row["n_adv_estimated"]),

            "n_pairs_legacy": int(legacy_row["n_pairs"]),
            "n_pairs_gate_a": int(gate_a_row["n_pairs"]),
            "n_pairs_gate_b": int(gate_b_row["n_pairs"]),

            "top_pair_pp_legacy": bool(legacy_pp),
            "top_pair_pp_gate_a": bool(gatea_pp),
            "top_pair_pp_gate_b": bool(gate_b_row["top_pair_pp"]),
            "top_pair_pc_legacy": bool(legacy_pc),
            "top_pair_pc_gate_a": bool(gatea_pc),
            "top_pair_pc_gate_b": bool(gate_b_row["top_pair_pc"]),
            "top_pair_cc_legacy": bool(legacy_cc),
            "top_pair_cc_gate_a": bool(gatea_cc),
            "top_pair_cc_gate_b": bool(gate_b_row["top_pair_cc"]),

            "removed_poison_legacy": int(legacy_row["removed_poison"]),
            "removed_poison_gate_a": int(gate_a_row["removed_poison"]),
            "removed_poison_gate_b": int(gate_b_row["removed_poison"]),
            "removed_clean_legacy": int(legacy_row["removed_clean"]),
            "removed_clean_gate_a": int(gate_a_row["removed_clean"]),
            "removed_clean_gate_b": int(gate_b_row["removed_clean"]),
            "residual_poison_legacy": int(legacy_row["residual_poison"]),
            "residual_poison_gate_a": int(gate_a_row["residual_poison"]),
            "residual_poison_gate_b": int(gate_b_row["residual_poison"]),
        }
        comparison_rows.append(row)
    return comparison_rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _check_no_overwrite(paths: List[Path]) -> None:
    existing = [p for p in paths if p.exists()]
    if existing:
        raise GateBStopCondition(
            f"Refusing to overwrite existing Gate-B output artifact(s): {existing} -- STOP per "
            "no-overwrite-historical-artifact condition."
        )


def _write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(
    per_query_rows: List[dict],
    comparison_rows: List[dict],
    dependency_info: dict,
    path: Path,
) -> None:
    by_qid = {r["query_id"]: r for r in per_query_rows}
    comp_by_qid = {r["query_id"]: r for r in comparison_rows}

    successes = [qid for qid, r in by_qid.items() if r["residual_poison"] == 0]
    failures = [qid for qid, r in by_qid.items() if r["residual_poison"] > 0]

    n_adv_deltas_a_to_b = [r["n_adv_delta_gate_a_to_gate_b"] for r in comparison_rows]
    n_adv_changed_a_to_b = sum(1 for d in n_adv_deltas_a_to_b if d != 0)

    pp_pc_cc_changed = sum(
        1 for r in comparison_rows
        if (r["top_pair_pp_gate_a"], r["top_pair_pc_gate_a"], r["top_pair_cc_gate_a"])
        != (r["top_pair_pp_gate_b"], r["top_pair_pc_gate_b"], r["top_pair_cc_gate_b"])
    )

    legacy_success_survivors = [
        qid for qid in PREVIOUSLY_SUCCESSFUL_CASES if qid in by_qid and by_qid[qid]["residual_poison"] == 0
    ]

    lines: List[str] = []
    lines.append("# Gate B -- RAGDefender Stella Paper-Fidelity Report")
    lines.append("")
    lines.append(
        "> Gate B evaluates the FINAL-PAPER-FAITHFUL RAGDefender implementation "
        "(`ragdefender_paper`) on our FIXED HotpotQA k=10/N=5 stress contexts. It is NOT "
        "a reproduction of the published HotpotQA performance numbers, because the final "
        "paper's nominal HotpotQA setting is k=2. No retrieval, generation, E1, CORAL, or "
        "MMD experiment was run."
    )
    lines.append("")
    lines.append("## Dependency / environment record")
    lines.append("")
    for k, v in dependency_info.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append(f"**Queries evaluated:** {len(per_query_rows)} of 8 (the same 8 recoverable instrumented "
                  "HotpotQA k=10/N=5 contexts used by Gate A; the 2 upstream text-recovery-mismatch queries "
                  "were not attempted).")
    lines.append("")

    lines.append("## Answers to the required Gate-B report questions")
    lines.append("")
    lines.append(
        f"**1. How many of the 8 queries are zero-residual-poison successes under paper-faithful "
        f"RAGDefender + Stella?** {len(successes)} of {len(by_qid)}: "
        + ", ".join(f"`{q}`" for q in successes) + "."
    )
    lines.append("")
    mean_delta = float(np.mean(n_adv_deltas_a_to_b))
    median_delta = float(np.median(n_adv_deltas_a_to_b))
    lines.append(
        f"**2. How does Stella change `N_adv` relative to Gate A?** `N_adv` changed for "
        f"{n_adv_changed_a_to_b}/{len(comparison_rows)} queries when moving from Gate A (MiniLM + paper "
        f"Stage 1) to Gate B (Stella + paper Stage 1); mean signed delta = {mean_delta:+.2f}, "
        f"median = {median_delta:+.2f} (per-query: "
        + ", ".join(f"{r['query_id'][:8]}={r['n_adv_delta_gate_a_to_gate_b']:+d}" for r in comparison_rows)
        + ")."
    )
    lines.append("")
    lines.append(
        f"**3. How does Stella change PP/PC/CC top-pair composition relative to Gate A?** The top-pair "
        f"class (PP vs. PC vs. CC) changed for {pp_pc_cc_changed}/{len(comparison_rows)} queries between "
        "Gate A and Gate B (see `gate_b_comparison.csv` for the full per-query breakdown)."
    )
    lines.append("")
    cd_row = by_qid.get(CLEAN_DENSITY_CASE)
    if cd_row is not None:
        lines.append(
            f"**4. Does the previously observed clean-density failure survive Stella?** Case "
            f"`{CLEAN_DENSITY_CASE}`: residual_poison = {cd_row['residual_poison']}, "
            f"residual_poison_fraction = {cd_row['residual_poison_fraction']:.3f}, "
            f"top-pair class = {'PP' if cd_row['top_pair_pp'] else ('PC' if cd_row['top_pair_pc'] else 'CC')}, "
            f"classification = {cd_row['classification']}. "
            + ("**Yes, it survives under Stella + paper logic too.**" if cd_row["residual_poison"] > 0
               else "**No -- it is resolved under Stella + paper logic.**")
        )
    else:
        lines.append("**4. Does the previously observed clean-density failure survive Stella?** N/A -- case not in this run.")
    lines.append("")
    lines.append(
        f"**5. Do any of the old six legacy success cases remain successes?** "
        f"{len(legacy_success_survivors)} of {len(PREVIOUSLY_SUCCESSFUL_CASES)}: "
        + (", ".join(f"`{q}`" for q in legacy_success_survivors) if legacy_success_survivors else "none")
        + "."
    )
    lines.append("")
    all_min_mean_margins = sorted(
        ((qid, r["min_abs_mean_margin"]) for qid, r in by_qid.items()), key=lambda t: t[1]
    )
    closest_mean_qid, closest_mean_margin = all_min_mean_margins[0]
    median_margin_is_always_zero = all(r["min_abs_median_margin"] == 0.0 for r in by_qid.values())
    lines.append(
        "**6. Are any Stage-1 decisions numerically close to the paper thresholds?** "
        + (
            "**Caveat on the median margin (read before the mean margin below):** "
            "`min_abs_median_margin` is exactly `0.0000` for **all 8/8** queries. This is **not** a "
            "coincidental knife-edge finding -- it is a deterministic consequence of the even-count "
            "median tie-break rule this codebase uses (`_torch_style_median_1d`, matching `torch.median`: "
            "for an even number of values it returns the *lower* of the two middle order statistics, "
            "which is always one of the actual input values, never an interpolated one). Because our "
            "k=10 stress regime always has an even count of per-passage medians `{s_median_i}`, `s_tilde` "
            "is *guaranteed* to exactly equal one specific passage's `s_median_i` on every query -- this "
            "is structural to using an even k, not evidence of a fragile per-query measurement, and would "
            "not necessarily recur at the paper's own odd/even-agnostic nominal k=2 setting. "
            if median_margin_is_always_zero else ""
        )
        + f"The informative signal is therefore the **mean** margin: the smallest `min_abs_mean_margin` "
        f"across all 8 queries is **{closest_mean_margin:.4f}** (query `{closest_mean_qid}`) -- see the "
        f"Threshold-Margin table below for all 8. "
        + ("This is small enough (<0.001) to flag as a genuinely numerically borderline mean-threshold "
           "decision for that query."
           if closest_mean_margin < 0.001 else
           "None of the 8 queries have a mean-margin small enough (<0.001) to call numerically borderline.")
    )
    lines.append("")
    lines.append(
        "**7. Does the evidence still support pursuing `top_pair_pp` as the primary mechanism-level "
        "variable for the next oracle experiment?** "
        f"Answered conservatively: under Stella + paper Stage 1, the top pair is PP for "
        f"{sum(1 for r in by_qid.values() if r['top_pair_pp'])}/{len(by_qid)} queries "
        f"(vs. {sum(1 for r in comparison_rows if r['top_pair_pp_gate_a'])}/{len(comparison_rows)} under "
        f"Gate A and {sum(1 for r in comparison_rows if r['top_pair_pp_legacy'])}/{len(comparison_rows)} under "
        "legacy). `top_pair_pp` remains a directionally plausible signal -- it is still the dominant top-pair "
        "outcome across all three variants on this population -- but Gate B alone (n=8, one fixed k=10/N=5 "
        "geometry, no oracle intervention yet run on the Stella-based population) does **not** establish it as "
        "a validated \"operative assumption\" of the defense. It should be treated as the leading *candidate* "
        "mechanism-level variable to test causally (via the still-pending E1/CORAL/MMD reruns on the "
        "re-identified population below), not as a confirmed finding."
    )
    lines.append("")

    lines.append("## Success-case re-identification (paper-faithful, Stella-based)")
    lines.append("")
    lines.append(
        "**Per instructions: E1/CORAL/MMD are NOT automatically run on this population. This section only "
        "identifies it.**"
    )
    lines.append("")
    lines.append(f"**{len(successes)} of {len(by_qid)}** queries satisfy `residual_poison == 0` under "
                  "`ragdefender_paper` + Stella:")
    lines.append("")
    lines.append("| query_id | removed_poison | removed_clean | N_adv | top_pair_pp |")
    lines.append("|---|---|---|---|---|")
    for qid in successes:
        r = by_qid[qid]
        lines.append(f"| `{qid}` | {r['removed_poison']} | {r['removed_clean']} | {r['n_adv']} | {r['top_pair_pp']} |")
    lines.append("")
    lines.append(f"**{len(failures)} of {len(by_qid)}** queries do NOT satisfy `residual_poison == 0`:")
    lines.append("")
    lines.append("| query_id | residual_poison | classification |")
    lines.append("|---|---|---|")
    for qid in failures:
        r = by_qid[qid]
        lines.append(f"| `{qid}` | {r['residual_poison']} | {r['classification']} |")
    lines.append("")

    lines.append("## Threshold-margin findings (numerically borderline decisions)")
    lines.append("")
    lines.append(
        "**Read the median-margin column with the Q6 caveat above in mind**: it is `0.0000` for all 8 "
        "queries by construction (even-count median tie-break), not because every query independently "
        "landed on a knife-edge. The **mean-margin** column is the informative one; rows are sorted by it."
    )
    lines.append("")
    lines.append("| query_id | min\\|mean margin\\| | min\\|median margin\\| (structurally ~0 for even k) |")
    lines.append("|---|---|---|")
    for qid, _ in all_min_mean_margins:
        r = by_qid[qid]
        lines.append(
            f"| `{qid}` | {r['min_abs_mean_margin']:.4f} | {r['min_abs_median_margin']:.4f} |"
        )
    lines.append("")

    lines.append("## Three-way comparison (Legacy / Gate A / Gate B)")
    lines.append("")
    lines.append(
        "Interpretation guide (do not infer causality beyond this controlled decomposition): "
        "**Legacy -> Gate A** isolates the Stage-1 *logic* effect on fixed MiniLM geometry; "
        "**Gate A -> Gate B** isolates the Stella *encoder/geometry* effect while holding paper Stage-1/2 "
        "logic fixed."
    )
    lines.append("")
    lines.append("| query_id | N_adv (L/A/B) | N_pairs (L/A/B) | top_pair (L/A/B) | removed_poison (L/A/B) | residual_poison (L/A/B) |")
    lines.append("|---|---|---|---|---|---|")
    for r in comparison_rows:
        def _tp(prefix):
            if r[f"top_pair_pp_{prefix}"]:
                return "PP"
            if r[f"top_pair_pc_{prefix}"]:
                return "PC"
            if r[f"top_pair_cc_{prefix}"]:
                return "CC"
            return "-"
        lines.append(
            f"| `{r['query_id']}` | {r['n_adv_legacy']}/{r['n_adv_gate_a']}/{r['n_adv_gate_b']} | "
            f"{r['n_pairs_legacy']}/{r['n_pairs_gate_a']}/{r['n_pairs_gate_b']} | "
            f"{_tp('legacy')}/{_tp('gate_a')}/{_tp('gate_b')} | "
            f"{r['removed_poison_legacy']}/{r['removed_poison_gate_a']}/{r['removed_poison_gate_b']} | "
            f"{r['residual_poison_legacy']}/{r['residual_poison_gate_a']}/{r['residual_poison_gate_b']} |"
        )
    lines.append("")

    lines.append("## Data files")
    lines.append("")
    lines.append("- `gate_b_per_query.csv` -- one row per query with every requested Gate-B field "
                  "(geometry, Stage-1 vectors/margins, Stage-2 pair/frequency detail, outcome, classification).")
    lines.append(
        "  - **Taxonomy note:** `above_mean_flags`/`above_median_flags`/`final_and_flags`/"
        "`final_adv_flag_indices` are Stage-1 **count-estimation indicator flags only** (they determine "
        "`n_adv`, a single integer). They are NOT a predicted adversarial-passage subset and were never "
        "used to derive `classification` -- `classification` is computed solely from Stage-2 evidence "
        "(`removed_indices`, `selected_pair_classes`, outcome counts). See `_classify_query`'s docstring."
    )
    lines.append("- `gate_b_comparison.csv` -- the three-way (Legacy/Gate A/Gate B) comparison table.")
    lines.append("- `similarity/{query_id}_stella_similarity_matrix.npy` -- full k x k Stella cosine-similarity matrix per query.")
    lines.append("- `embeddings/{query_id}_stella_embeddings.npy` -- raw Stella passage embeddings per query.")
    lines.append("")
    lines.append("## Scope reminder")
    lines.append("")
    lines.append(
        "No retrieval, generation, E1, CORAL, or MMD experiment was run. This report does not modify any "
        "manuscript claim; the re-identified success population above is recorded for review only."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import sentence_transformers
    import torch as _torch
    import transformers

    dependency_info = {
        "stella_model": STELLA_MODEL_NAME,
        "stella_revision": "7817065102fd9e1b031fe874e910c01f40b2f001",
        "transformers_version": transformers.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": _torch.__version__,
        "requested_device": REQUESTED_DEVICE,
    }

    out_per_query_csv = OUTPUT_DIR / "gate_b_per_query.csv"
    out_comparison_csv = OUTPUT_DIR / "gate_b_comparison.csv"
    out_report_md = OUTPUT_DIR / "GATE_B_STELLA_FIDELITY_REPORT.md"
    _check_no_overwrite([out_per_query_csv, out_comparison_csv, out_report_md])

    (OUTPUT_DIR / "similarity").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "embeddings").mkdir(parents=True, exist_ok=True)

    run_config, gate_a_is_poison = _load_gate_a_context()
    diagnostics_records = viz._read_jsonl(run_config["diagnostics_jsonl"])  # noqa: SLF001
    query_results_index = viz.load_query_results_index(run_config["query_results_dir"])

    cases = []
    for qid in run_config["query_ids_processed"]:
        case = _recover_case_texts(qid, diagnostics_records, query_results_index)
        _verify_matches_gate_a(case, gate_a_is_poison)
        cases.append(case)

    s_model, cfg, actual_device = _load_stella_model()
    dependency_info["actual_device"] = actual_device
    _, st_util = defense_runner._lazy_st()  # noqa: SLF001

    per_query_rows = [run_gate_b_query(case, s_model, st_util, OUTPUT_DIR) for case in cases]

    comparison_rows = build_comparison_rows(GATE_A_OUTPUT_DIR / "gate_a_per_query.csv", per_query_rows)

    _write_csv(per_query_rows, out_per_query_csv)
    _write_csv(comparison_rows, out_comparison_csv)
    write_report(per_query_rows, comparison_rows, dependency_info, out_report_md)

    print(f"Gate B complete: {len(per_query_rows)} queries evaluated on device={actual_device}.")
    print(f"Wrote: {out_per_query_csv}")
    print(f"Wrote: {out_comparison_csv}")
    print(f"Wrote: {out_report_md}")


if __name__ == "__main__":
    main()
