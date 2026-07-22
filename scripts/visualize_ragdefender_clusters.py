#!/usr/bin/env python3
"""RAGDefender cluster/concentration diagnostics -- faithful internals recompute.

Recomputes, per query, exactly the RAGDefender Stage-1 concentration and
Stage-2 pair-frequency internals `defense/defense_runner.py::apply_defense`
(the code path `ragdefender_original` actually runs) uses -- same embedder
(`paraphrase-MiniLM-L6-v2` by default, imported via `defense_runner._lazy_st`
so it is *literally* the same import path, not just an equivalent one),
same `sentence_transformers.util.cos_sim`, same OR/combined-threshold/flip
concentration logic, same `Counter`-based pair-frequency Stage-2 scoring
(see `defense/ragdefender_internals.py` for the faithfulness notes) -- and
cross-checks the recomputed `N_adv` / removed indices against the real
`N_adv_estimated_by_ragdefender` / `removed_doc_ids` already recorded in an
existing diagnostics JSONL.

This script never calls an LLM/API, never modifies `defense/defense_runner.py`,
`defense/dispatch.py`, `main.py`, or any existing `results/` artifact, and
never imports the `ragdefender` PyPI/GitHub package. Passage text is recovered
from the paired `query_results/*.json` record's `input_prompt_no_defense`
field -- only available for *live* (non `--dry_run`) runs; queries where that
is unavailable are skipped and reported, never silently guessed.

Every invocation creates one new timestamped run directory under
`--output_dir` (default `results/diagnostics/ragdefender_cluster_viz/`):

    <YYYYMMDD_HHMMSS>_clusterdiag_<dataset>_k<k>_N<N>_<defense>_
        embedder-<short_embedder>_task-<task_type>_p<p>/

containing `run_config.json`, `manifest.json`, `stage1_summary.csv`,
`stage2_summary.csv`, `graph_metrics.csv`, `passages/<query_id>_passages.csv`,
`similarity/<query_id>_similarity_matrix.{csv,npy}` (+ `_reordered.csv`),
`plots/<query_id>_*.png` (skipped under `--no_plots` or if matplotlib is
unavailable), and its own `RAGDEFENDER_CLUSTER_VISUALIZATION_REPORT.md`
(this run's data only -- no rescanning of other run directories).

Usage:
    python scripts/visualize_ragdefender_clusters.py \\
        --diagnostics_jsonl results/diagnostics/ragdefender_smoke_live_10q/hotpotqa-...-defense-ragdefender_original.jsonl \\
        --query_results_dir results/query_results/ragdefender_smoke_live_10q \\
        --no_plots --max_queries 1
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

import fix_sentence_transformers  # noqa: E402,F401 -- required before any sentence_transformers import, matches main.py's compat patch

from defense.defense_runner import _lazy_st  # noqa: SLF001 -- intentional reuse of the exact ST/util import path ragdefender_original itself uses
from defense.diagnostics import read_jsonl
from defense.ragdefender_internals import (
    ConcentrationResult,
    Stage2Result,
    concentration_stage1,
    stage2_pair_frequency,
)

DEFAULT_OUTPUT_DIR = os.path.join("results", "diagnostics", "ragdefender_cluster_viz")
DEFAULT_EMBEDDER = "paraphrase-MiniLM-L6-v2"
DEFAULT_STAGE2_P = 2.0

EMBEDDER_SHORT_NAMES = {
    "paraphrase-MiniLM-L6-v2": "paraphraseMiniLM",
    "sentence-transformers/all-MiniLM-L6-v2": "allMiniLM",
}

PASSAGE_COLUMNS = [
    "query_id", "passage_index", "passage_id", "rank", "is_poison", "removed_by_ragdefender",
    "retrieval_score", "text_preview", "concentration_mean_similarity", "concentration_median_similarity",
    "above_global_mean", "above_combined_median_threshold", "concentration_final_flag", "stage2_frequency_score",
    "nearest_neighbor_index", "nearest_neighbor_is_poison", "nearest_neighbor_similarity",
    "mean_similarity_to_poison", "mean_similarity_to_clean", "x_pca", "y_pca",
]

STAGE1_COLUMNS = [
    "query_id", "k", "N_retrieved_poison", "N_retrieved_clean", "global_mean_similarity", "global_median_similarity",
    "combined_median_threshold", "concentration_flag_count", "result_flipped",
    "N_adv_estimated_by_recomputed_concentration", "N_adv_estimated_in_diagnostics", "agreement_with_diagnostics",
]

STAGE2_COLUMNS = [
    "query_id", "N_adv", "N_pairs", "top_pairs", "top_pair_similarities", "stage2_frequency_scores",
    "selected_indices_recomputed", "removed_indices_in_diagnostics", "agreement_with_diagnostics",
]

GRAPH_COLUMNS = [
    "query_id", "k", "mean_poison_poison_similarity", "mean_poison_clean_similarity", "mean_clean_clean_similarity",
    "max_poison_poison_similarity", "nearest_neighbor_poison_ratio_mean", "poison_neighborhood_entropy",
    "clean_neighborhood_entropy", "removed_poison", "removed_clean", "residual_poison_fraction",
    "asr_no_defense_strict", "asr_with_defense_strict",
]

SEVERE_QUERY_ID = "5a8cb288554299585d9e3726"


# --------------------------------------------------------------------------
# Naming / slug helpers (run-directory naming only -- not RAGDefender math)
# --------------------------------------------------------------------------

def slugify_embedder(name: str) -> str:
    if name in EMBEDDER_SHORT_NAMES:
        return EMBEDDER_SHORT_NAMES[name]
    slug = "".join(ch for ch in name if ch.isalnum())
    return slug[:24] if slug else "embedder"


def slugify_defense(name: str) -> str:
    return (name or "unknown").replace("_", "-")


def format_p(p: float) -> str:
    return str(int(p)) if float(p).is_integer() else str(p)


def build_run_dir(output_dir: str, dataset: str, k: int, n_injected: int, defense: str,
                   embedder: str, task_type: str, p: float) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    embedder_short = slugify_embedder(embedder)
    defense_slug = slugify_defense(defense)
    task_slug = (task_type or "unknown").replace("_", "")
    name = (
        f"{ts}_clusterdiag_{dataset}_k{k}_N{n_injected}_{defense_slug}_"
        f"embedder-{embedder_short}_task-{task_slug}_p{format_p(p)}"
    )
    run_dir = Path(output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    for sub in ("passages", "similarity", "plots", "logs"):
        (run_dir / sub).mkdir()
    return run_dir


# --------------------------------------------------------------------------
# Text recovery from query_results (see module docstring / plan for why this
# is the only faithful source of retrieved passage text)
# --------------------------------------------------------------------------

def recover_pre_defense_texts(query_result_record: Optional[Dict]) -> Optional[List[str]]:
    """Recover the exact pre-defense, rank-ordered retrieved passage texts
    from a `query_results/*.json` record's `input_prompt_no_defense` field.

    Returns None if the record is missing or has no such field (e.g. a
    --dry_run query_results file, where no generation -- and therefore no
    prompt -- was ever produced). Callers must treat None as "not
    recoverable" and skip the query, never guess/reconstruct text another way.
    """
    if not query_result_record:
        return None
    prompt = query_result_record.get("input_prompt_no_defense")
    if not prompt or "Contexts: " not in prompt or "\n\nQuery:" not in prompt:
        return None
    body = prompt.split("Contexts: ", 1)[1].split("\n\nQuery:", 1)[0]
    # src/prompts.py's MULTIPLE_PROMPT template leaves a trailing space on the
    # last context line ("[context] \n\nQuery:") -- strip every line.
    return [line.strip() for line in body.split("\n")]


def load_query_results_index(query_results_dir: str) -> Dict[str, Dict]:
    """Load every `*.json` file in `query_results_dir` (main.py's
    `save_results` format: `[{"iter_0": [...]}]`) and index records by their
    `id` field."""
    index: Dict[str, Dict] = {}
    pattern = os.path.join(query_results_dir, "*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for iter_block in data:
            for _, recs in iter_block.items():
                for rec in recs:
                    if "id" in rec:
                        index[rec["id"]] = rec
    return index


# --------------------------------------------------------------------------
# Embedding / similarity -- reuses defense_runner's exact private import path
# --------------------------------------------------------------------------

def load_embedder(name: str):
    SentenceTransformer, _ = _lazy_st()
    return SentenceTransformer(name)


def encode_texts(model, texts: Sequence[str]):
    return model.encode(list(texts), convert_to_tensor=True)


def cos_sim_from_embeddings(embeddings) -> np.ndarray:
    _, st_util = _lazy_st()
    sim = st_util.cos_sim(embeddings, embeddings)
    return np.asarray(sim.cpu().numpy(), dtype=np.float64)


def compute_pca_coords(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """2D PCA projection of one query's own k embeddings (RAGDefender's
    embedder, not an arbitrary external one). Purely a visualization aid --
    k is small (5-10), so this is a low-rank, per-query projection, never a
    globally comparable embedding space."""
    from sklearn.decomposition import PCA

    k = embeddings.shape[0]
    if k < 2:
        zeros = np.zeros(k)
        return zeros, zeros.copy()
    n_components = max(1, min(2, k - 1, embeddings.shape[1]))
    coords = PCA(n_components=n_components).fit_transform(embeddings)
    x = coords[:, 0]
    y = coords[:, 1] if coords.shape[1] > 1 else np.zeros(k)
    return x, y


def _umap_available() -> bool:
    try:
        import umap  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# Per-passage helpers
# --------------------------------------------------------------------------

def _nearest_neighbor(sim_matrix: np.ndarray, i: int) -> Tuple[Optional[int], Optional[float]]:
    k = sim_matrix.shape[0]
    if k <= 1:
        return None, None
    row = sim_matrix[i].copy()
    row[i] = -np.inf
    nn_idx = int(np.argmax(row))
    return nn_idx, float(sim_matrix[i, nn_idx])


def _mean_similarity_to_group(sim_matrix: np.ndarray, i: int, is_poison: Sequence[bool], want_poison: bool) -> Optional[float]:
    k = sim_matrix.shape[0]
    vals = [sim_matrix[i, j] for j in range(k) if j != i and bool(is_poison[j]) == want_poison]
    return float(np.mean(vals)) if vals else None


def build_passage_rows(rec: Dict, texts: List[str], sim_matrix: np.ndarray, stage1: ConcentrationResult,
                        stage2: Stage2Result, x_pca: np.ndarray, y_pca: np.ndarray, removed_set: set) -> List[Dict]:
    doc_ids = list(rec["retrieved_doc_ids"])
    is_poison = [bool(x) for x in rec["retrieved_is_poison"]]
    k = len(texts)
    rows = []
    for i in range(k):
        nn_idx, nn_sim = _nearest_neighbor(sim_matrix, i)
        text = texts[i]
        rows.append({
            "query_id": rec["query_id"],
            "passage_index": i,
            "passage_id": doc_ids[i],
            "rank": i + 1,
            "is_poison": is_poison[i],
            "removed_by_ragdefender": doc_ids[i] in removed_set,
            "retrieval_score": None,  # not recoverable from existing artifacts -- documented limitation
            "text_preview": (text[:117] + "...") if len(text) > 120 else text,
            "concentration_mean_similarity": float(stage1.avg[i]),
            "concentration_median_similarity": float(stage1.median[i]),
            "above_global_mean": bool(stage1.above_avg[i]),
            "above_combined_median_threshold": bool(stage1.above_median[i]),
            "concentration_final_flag": bool(stage1.adv_side_flag[i]),
            "stage2_frequency_score": float(stage2.frequency_scores[i]) if i < len(stage2.frequency_scores) else 0.0,
            "nearest_neighbor_index": nn_idx,
            "nearest_neighbor_is_poison": bool(is_poison[nn_idx]) if nn_idx is not None else None,
            "nearest_neighbor_similarity": nn_sim,
            "mean_similarity_to_poison": _mean_similarity_to_group(sim_matrix, i, is_poison, True),
            "mean_similarity_to_clean": _mean_similarity_to_group(sim_matrix, i, is_poison, False),
            "x_pca": float(x_pca[i]),
            "y_pca": float(y_pca[i]),
        })
    return rows


# --------------------------------------------------------------------------
# Similarity matrix persistence
# --------------------------------------------------------------------------

def save_similarity_matrix(run_dir: Path, qid: str, doc_ids: List[str], is_poison: List[bool],
                            removed_flags: List[bool], sim_matrix: np.ndarray) -> Tuple[Path, Path, Path, List[int]]:
    sim_dir = run_dir / "similarity"
    labels = [f"{i}:{doc_ids[i]}" for i in range(len(doc_ids))]
    pd.DataFrame(sim_matrix, index=labels, columns=labels).to_csv(sim_dir / f"{qid}_similarity_matrix.csv")
    csv_path = sim_dir / f"{qid}_similarity_matrix.csv"

    npy_path = sim_dir / f"{qid}_similarity_matrix.npy"
    np.save(npy_path, sim_matrix)

    # Group order: poison+removed, poison+kept, clean+removed, clean+kept (stable within each group).
    order = sorted(
        range(len(doc_ids)),
        key=lambda i: (0 if is_poison[i] else 1, 0 if removed_flags[i] else 1, i),
    )
    reordered = sim_matrix[np.ix_(order, order)]
    reordered_labels = [
        f"{i}:{'P' if is_poison[i] else 'C'}{'R' if removed_flags[i] else 'K'}:{doc_ids[i]}" for i in order
    ]
    reordered_path = sim_dir / f"{qid}_similarity_matrix_reordered.csv"
    pd.DataFrame(reordered, index=reordered_labels, columns=reordered_labels).to_csv(reordered_path)

    return csv_path, npy_path, reordered_path, order


# --------------------------------------------------------------------------
# Stage1 / Stage2 / graph-metrics row builders
# --------------------------------------------------------------------------

def build_stage1_row(rec: Dict, stage1: ConcentrationResult) -> Dict:
    diag_n_adv = rec.get("N_adv_estimated_by_ragdefender")
    agreement = (diag_n_adv is not None) and (stage1.n_adv_estimated == diag_n_adv)
    return {
        "query_id": rec["query_id"],
        "k": rec["k"],
        "N_retrieved_poison": rec.get("N_retrieved_poison"),
        "N_retrieved_clean": rec.get("N_retrieved_clean"),
        "global_mean_similarity": stage1.avg_avg,
        "global_median_similarity": stage1.avg_median,
        "combined_median_threshold": stage1.combined_threshold,
        "concentration_flag_count": int(stage1.raw_or_flag.sum()),
        "result_flipped": stage1.flipped,
        "N_adv_estimated_by_recomputed_concentration": stage1.n_adv_estimated,
        "N_adv_estimated_in_diagnostics": diag_n_adv,
        "agreement_with_diagnostics": agreement,
    }


def build_stage2_row(rec: Dict, n_adv: int, stage2: Stage2Result, removed_indices_in_diagnostics: List[int]) -> Dict:
    agreement = set(stage2.selected_indices) == set(removed_indices_in_diagnostics)
    return {
        "query_id": rec["query_id"],
        "N_adv": n_adv,
        "N_pairs": stage2.n_pairs,
        "top_pairs": json.dumps([[int(i), int(j), round(float(s), 6)] for i, j, s in stage2.top_pairs]),
        "top_pair_similarities": json.dumps([round(float(s), 6) for _, _, s in stage2.top_pairs]),
        "stage2_frequency_scores": json.dumps([round(float(s), 6) for s in stage2.frequency_scores]),
        "selected_indices_recomputed": json.dumps(list(stage2.selected_indices)),
        "removed_indices_in_diagnostics": json.dumps(list(removed_indices_in_diagnostics)),
        "agreement_with_diagnostics": agreement,
    }


def _shannon_entropy_binary(labels: List[bool]) -> Optional[float]:
    if not labels:
        return None
    p1 = sum(1 for l in labels if l) / len(labels)
    p0 = 1.0 - p1
    ent = 0.0
    for p in (p0, p1):
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def compute_graph_metrics(rec: Dict, sim_matrix: np.ndarray, is_poison: List[bool], removed_flags: List[bool]) -> Dict:
    k = sim_matrix.shape[0]
    pp, pc, cc = [], [], []
    for i in range(k):
        for j in range(i + 1, k):
            v = sim_matrix[i, j]
            if is_poison[i] and is_poison[j]:
                pp.append(v)
            elif is_poison[i] != is_poison[j]:
                pc.append(v)
            else:
                cc.append(v)

    nn_is_poison = []
    for i in range(k):
        nn_idx, _ = _nearest_neighbor(sim_matrix, i)
        nn_is_poison.append(bool(is_poison[nn_idx]) if nn_idx is not None else False)
    nn_poison_ratio = float(np.mean(nn_is_poison)) if nn_is_poison else None
    poison_nn_labels = [nn_is_poison[i] for i in range(k) if is_poison[i]]
    clean_nn_labels = [nn_is_poison[i] for i in range(k) if not is_poison[i]]

    return {
        "query_id": rec["query_id"],
        "k": k,
        "mean_poison_poison_similarity": float(np.mean(pp)) if pp else None,
        "mean_poison_clean_similarity": float(np.mean(pc)) if pc else None,
        "mean_clean_clean_similarity": float(np.mean(cc)) if cc else None,
        "max_poison_poison_similarity": float(np.max(pp)) if pp else None,
        "nearest_neighbor_poison_ratio_mean": nn_poison_ratio,
        "poison_neighborhood_entropy": _shannon_entropy_binary(poison_nn_labels),
        "clean_neighborhood_entropy": _shannon_entropy_binary(clean_nn_labels),
        "removed_poison": rec.get("removed_poison"),
        "removed_clean": rec.get("removed_clean"),
        "residual_poison_fraction": rec.get("residual_poison_fraction"),
        "asr_no_defense_strict": rec.get("asr_no_defense_strict"),
        "asr_with_defense_strict": rec.get("asr_with_defense_strict"),
    }


# --------------------------------------------------------------------------
# Plots (Agg backend; every function is best-effort and never raises past a
# missing optional dependency -- callers check for a None return)
# --------------------------------------------------------------------------

def _configure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        print(f"[visualize_ragdefender_clusters] matplotlib unavailable ({exc}); continuing with plots disabled.")
        return None


def plot_pca_scatter(plt, run_dir: Path, qid: str, x_pca: np.ndarray, y_pca: np.ndarray,
                      is_poison: List[bool], removed_flags: List[bool]) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(x_pca)):
        color = "tab:red" if is_poison[i] else "tab:blue"
        if removed_flags[i]:
            # 'x' is an unfilled marker -- matplotlib ignores edgecolors on it,
            # so use linewidths alone to make removed points visually heavier.
            ax.scatter(x_pca[i], y_pca[i], c=color, marker="x", s=140, linewidths=3, zorder=3)
        else:
            ax.scatter(x_pca[i], y_pca[i], c=color, marker="o", s=140, edgecolors="none", zorder=3)
        ax.annotate(str(i), (x_pca[i], y_pca[i]), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_title(f"PCA scatter -- {qid}\nred=poison blue=clean, x=removed o=kept")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    path = run_dir / "plots" / f"{qid}_pca_scatter.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_heatmap(plt, run_dir: Path, qid: str, sim_matrix: np.ndarray, labels: List[str],
                  filename: str, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim_matrix, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title(title)
    path = run_dir / "plots" / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_similarity_distribution(plt, run_dir: Path, qid: str, sim_matrix: np.ndarray, is_poison: List[bool]) -> Path:
    k = sim_matrix.shape[0]
    pp, pc, cc = [], [], []
    for i in range(k):
        for j in range(i + 1, k):
            v = sim_matrix[i, j]
            if is_poison[i] and is_poison[j]:
                pp.append(v)
            elif is_poison[i] != is_poison[j]:
                pc.append(v)
            else:
                cc.append(v)
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(-1, 1, 25)
    if pp:
        ax.hist(pp, bins=bins, alpha=0.5, label=f"poison-poison (n={len(pp)})", color="tab:red")
    if pc:
        ax.hist(pc, bins=bins, alpha=0.5, label=f"poison-clean (n={len(pc)})", color="tab:purple")
    if cc:
        ax.hist(cc, bins=bins, alpha=0.5, label=f"clean-clean (n={len(cc)})", color="tab:blue")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("pair count")
    ax.set_title(f"Similarity distribution -- {qid}")
    ax.legend(fontsize=8)
    path = run_dir / "plots" / f"{qid}_similarity_distribution.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_stage2_pairgraph(plt, run_dir: Path, qid: str, k: int, is_poison: List[bool],
                           removed_flags: List[bool], stage2: Stage2Result) -> Optional[Path]:
    try:
        import networkx as nx
    except ImportError:
        return None

    g = nx.Graph()
    g.add_nodes_from(range(k))
    for x, y, sim in stage2.top_pairs:
        g.add_edge(x, y, weight=sim)
    pos = nx.circular_layout(g)

    scores = np.asarray(stage2.frequency_scores, dtype=np.float64)
    if scores.size and scores.max() > scores.min():
        norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        norm = np.zeros_like(scores)
    sizes = 300 + norm * 2700
    colors = ["tab:red" if is_poison[i] else "tab:blue" for i in range(k)]
    edgecolors = ["black" if removed_flags[i] else "none" for i in range(k)]
    linewidths = [2.5 if removed_flags[i] else 0.5 for i in range(k)]

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.5)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes, node_color=colors,
                            edgecolors=edgecolors, linewidths=linewidths)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8)
    ax.set_title(f"Stage-2 top-pair graph -- {qid}\nsize=frequency score, red=poison blue=clean, thick border=removed")
    ax.axis("off")
    path = run_dir / "plots" / f"{qid}_stage2_pairgraph.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Severe-query narrative (data-driven -- no hardcoded conclusions)
# --------------------------------------------------------------------------

def build_query_narrative(rec: Dict, stage1: ConcentrationResult, stage2: Stage2Result, n_adv: int,
                           doc_ids: List[str], is_poison: List[bool], removed_indices_in_diagnostics: List[int]) -> str:
    poison_idx = [i for i, p in enumerate(is_poison) if p]
    freq_rounded = [round(float(s), 4) for s in stage2.frequency_scores]
    lines = [
        f"Diagnostics record: `N_adv_estimated_by_ragdefender={rec.get('N_adv_estimated_by_ragdefender')}`, "
        f"`removed_doc_ids={rec.get('removed_doc_ids')}`, `poison_recall={rec.get('poison_recall')}`, "
        f"`residual_poison_fraction={rec.get('residual_poison_fraction')}`.",
        "",
        f"Recomputed Stage 1: avg_avg={stage1.avg_avg:.4f}, avg_median={stage1.avg_median:.4f}, "
        f"combined_threshold={stage1.combined_threshold:.4f}, raw_or_flag_count={int(stage1.raw_or_flag.sum())}, "
        f"flipped={stage1.flipped}, N_adv_estimated={stage1.n_adv_estimated}.",
        "",
        f"Recomputed Stage 2 (N_adv={n_adv}, N_pairs={stage2.n_pairs}): "
        f"top_pairs={[(i, j, round(float(s), 4)) for i, j, s in stage2.top_pairs]}, "
        f"frequency_scores(by rank index)={freq_rounded}, selected_indices={stage2.selected_indices}.",
        "",
        f"Poison passages are at rank indices {poison_idx} (doc_ids "
        f"{[doc_ids[i] for i in poison_idx]}); diagnostics removed rank indices "
        f"{removed_indices_in_diagnostics} (doc_ids {[doc_ids[i] for i in removed_indices_in_diagnostics]}).",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# run_config.json / manifest.json helpers
# --------------------------------------------------------------------------

def _try_version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def _run_git(args: List[str], cwd: str) -> Optional[str]:
    try:
        out = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def git_commit(repo_dir: str) -> Optional[str]:
    return _run_git(["rev-parse", "HEAD"], repo_dir)


def git_status_short(repo_dir: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "status", "--short"], cwd=repo_dir, capture_output=True, text=True, timeout=10)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def ragdefender_clone_info(repo_root: str) -> Tuple[Optional[str], Optional[str]]:
    clone_dir = os.path.join(repo_root, "RAGDefender")
    if not os.path.isdir(os.path.join(clone_dir, ".git")):
        return None, None
    commit = git_commit(clone_dir)
    tags_raw = _run_git(["tag", "--points-at", "HEAD"], clone_dir)
    tag = next((t for t in (tags_raw or "").splitlines() if t), None)
    return commit, tag


def build_run_config(args: argparse.Namespace, dataset: str, k: int, n_injected: int, defense: str,
                      processed_ids: List[str], skipped_ids: List[str], skip_reasons: Dict[str, str],
                      embedder_short: str, plots_enabled: bool, plots_skip_reason: Optional[str]) -> Dict:
    ragdefender_commit, ragdefender_tag = ragdefender_clone_info(REPO_ROOT)
    return {
        "timestamp": datetime.now().isoformat(),
        "run_type": "ragdefender_cluster_diagnostics",
        "dataset": dataset,
        "k": k,
        "N": n_injected,
        "defense": defense,
        "diagnostics_jsonl": os.path.abspath(args.diagnostics_jsonl),
        "query_results_dir": os.path.abspath(args.query_results_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "query_ids_processed": processed_ids,
        "query_ids_skipped": skipped_ids,
        "skip_reasons": skip_reasons,
        "task_type": args.task_type,
        "embedder": args.embedder,
        "embedder_short": embedder_short,
        "p": args.stage2_p,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": _try_version("torch"),
        "sentence_transformers_version": _try_version("sentence_transformers"),
        "sklearn_version": _try_version("sklearn"),
        "pandas_version": _try_version("pandas"),
        "matplotlib_version": _try_version("matplotlib"),
        "git_commit": git_commit(REPO_ROOT),
        "git_status_short": git_status_short(REPO_ROOT),
        "ragdefender_clone_commit": ragdefender_commit,
        "ragdefender_clone_tag": ragdefender_tag,
        "ragdefender_package_imported": False,
        "gpt_or_api_calls_made": False,
        "raw_embeddings_saved": False,
        "also_umap_requested": args.also_umap,
        "also_umap_available": _umap_available(),
        "plots_enabled": plots_enabled,
        "plots_skip_reason": plots_skip_reason,
        "argv": sys.argv,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def render_report(run_dir: Path, args: argparse.Namespace, dataset: str, k: int, n_injected: int, defense: str,
                   processed_ids: List[str], skipped_ids: List[str], skip_reasons: Dict[str, str],
                   detail_by_qid: Dict[str, Dict]) -> str:
    lines = [
        "# RAGDefender Cluster Diagnostics Report",
        "",
        f"Run directory: `{run_dir.name}`",
        "",
        f"- dataset: `{dataset}`, k={k}, N={n_injected}, defense=`{defense}`",
        f"- embedder: `{args.embedder}`, task_type: `{args.task_type}`, stage2 p={args.stage2_p}",
        f"- queries processed: {len(processed_ids)}, skipped: {len(skipped_ids)}",
        "",
        "## Implementation fidelity",
        "",
        "Existing baseline behavior (`ragdefender_original`) is reproduced by "
        "`defense/defense_runner.py`, which `defense/ragdefender_internals.py` reimplements "
        "side-by-side for this diagnostic (same embedder, same "
        "`sentence_transformers.util.cos_sim` call, same OR/combined-threshold/flip "
        "concentration logic, same pair-frequency Stage-2 scoring -- see "
        "`docs/RAGDEFENDER_VERSION_AUDIT.md` for the full comparison). The official "
        "`ragdefender` v0.2.0 package (`ragdefender.grouping.concentration."
        "ConcentrationBasedGrouping`, `ragdefender.identification.topk.IdentifyAdversarial`) is "
        "useful for comparison/documentation only -- it was **not** imported or run for this "
        "diagnostic (`ragdefender_package_imported=false` in `run_config.json`). "
        "**Recomputation is only claimed to agree with existing diagnostics where the "
        "cross-check below actually passes for this run's queries** -- agreement is not "
        "assumed a priori and any mismatch is reported, not hidden.",
        "",
        "## Stage 1 / Stage 2 agreement (this run)",
        "",
        "| query_id | N_adv (diagnostics) | N_adv (recomputed) | stage1 agree | stage2 agree | failure_stage |",
        "|---|---|---|---|---|---|",
    ]
    for qid in processed_ids:
        d = detail_by_qid[qid]
        lines.append(
            f"| {qid} | {d['diag_n_adv']} | {d['recomputed_n_adv']} | {d['stage1_agree']} | "
            f"{d['stage2_agree']} | {d['failure_stage']} |"
        )
    lines.append("")

    if skipped_ids:
        lines.append("## Skipped queries")
        lines.append("")
        for qid in skipped_ids:
            lines.append(f"- `{qid}`: {skip_reasons.get(qid)}")
        lines.append("")

    if SEVERE_QUERY_ID in detail_by_qid:
        lines.append(f"## Query `{SEVERE_QUERY_ID}`")
        lines.append("")
        lines.append(detail_by_qid[SEVERE_QUERY_ID]["narrative"])
        lines.append("")

    lines.extend([
        "## Future normalized/adaptive poison objective (qualitative note)",
        "",
        "The per-passage signals exposed here -- Stage-1 threshold margins "
        "(`concentration_mean_similarity`/`concentration_median_similarity` vs. "
        "`combined_median_threshold`) and Stage-2 frequency-score gaps between the last "
        "selected and first non-selected index -- are continuous quantities a future adaptive "
        "attacker could try to minimize (stay just below threshold / just below the selection "
        "cutoff) instead of only maximizing retrieval score. **No adaptive attack is "
        "implemented in this task**; this is only a note on what a future normalized objective "
        "could be built from.",
        "",
        "## Limitations",
        "",
        "- `retrieval_score` in `passages/<query_id>_passages.csv` is always `null` -- not "
        "recoverable from existing diagnostics/query_results artifacts.",
        "- PCA is fit per-query on that query's own k embeddings only (k is small, e.g. 5-10), "
        "so the 2D projection is a low-rank, per-query visualization aid, not a globally "
        "comparable embedding space.",
        "- Only live (`--dry_run False`) query_results with a recoverable "
        "`input_prompt_no_defense` were processed; dry-run-only diagnostics are skipped and "
        "listed above, never guessed.",
        "- The `ragdefender` PyPI/GitHub package was not imported or run "
        "(`ragdefender_package_imported=false`).",
        "- No GPT/API calls were made (`gpt_or_api_calls_made=false`).",
    ])
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI / main
# --------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostics_jsonl", required=True, help="Path to one ragdefender_original diagnostics JSONL.")
    parser.add_argument("--query_results_dir", required=True, help="Dir containing the matching query_results/*.json.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query_id", default=None, help="Process only this query_id.")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--task_type", default="multi_hop", choices=["multi_hop", "single_hop"])
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--stage2_p", type=float, default=DEFAULT_STAGE2_P)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--also_umap", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Path:
    args = parse_args(argv)

    records = read_jsonl(args.diagnostics_jsonl)
    if not records:
        raise ValueError(f"No records found in {args.diagnostics_jsonl}")
    if args.query_id:
        records = [r for r in records if r["query_id"] == args.query_id]
        if not records:
            raise ValueError(f"query_id {args.query_id!r} not found in {args.diagnostics_jsonl}")
    if args.max_queries is not None:
        records = records[: args.max_queries]
    if not records:
        raise ValueError("No records left to process after applying --query_id/--max_queries.")

    first = records[0]
    dataset, k, n_injected, defense = first["dataset"], first["k"], first["N_injected"], first["defense"]

    query_results_index = load_query_results_index(args.query_results_dir)
    run_dir = build_run_dir(args.output_dir, dataset, k, n_injected, defense, args.embedder, args.task_type, args.stage2_p)

    manifest: Dict[str, List[str]] = {"config": [], "csv": [], "matrices": [], "plots": [], "report": [], "logs": []}
    log_lines: List[str] = [f"visualize_ragdefender_clusters.py run started: {datetime.now().isoformat()}"]

    plots_skip_reason = None
    if args.no_plots:
        plt = None
        plots_skip_reason = "--no_plots passed"
    else:
        plt = _configure_matplotlib()
        if plt is None:
            plots_skip_reason = "matplotlib import failed (missing dependency)"
    plots_enabled = plt is not None

    model = load_embedder(args.embedder)

    stage1_rows: List[Dict] = []
    stage2_rows: List[Dict] = []
    graph_rows: List[Dict] = []
    processed_ids: List[str] = []
    skipped_ids: List[str] = []
    skip_reasons: Dict[str, str] = {}
    detail_by_qid: Dict[str, Dict] = {}

    for rec in records:
        qid = rec["query_id"]
        qr = query_results_index.get(qid)
        texts = recover_pre_defense_texts(qr)
        expected_k = len(rec["retrieved_doc_ids"])

        if texts is None:
            reason = (
                "query_results record not found for this query_id"
                if qr is None else
                "input_prompt_no_defense missing (likely a --dry_run query_results file)"
            )
            skipped_ids.append(qid)
            skip_reasons[qid] = reason
            log_lines.append(f"SKIP {qid}: {reason}")
            continue
        if len(texts) != expected_k:
            reason = f"recovered {len(texts)} passage texts, expected {expected_k} (rank-order parse mismatch)"
            skipped_ids.append(qid)
            skip_reasons[qid] = reason
            log_lines.append(f"SKIP {qid}: {reason}")
            continue

        doc_ids = list(rec["retrieved_doc_ids"])
        is_poison = [bool(x) for x in rec["retrieved_is_poison"]]
        removed_set = set(rec.get("removed_doc_ids") or [])
        removed_flags = [d in removed_set for d in doc_ids]
        removed_indices_in_diagnostics = [i for i, d in enumerate(doc_ids) if d in removed_set]

        embeddings_t = encode_texts(model, texts)
        sim_matrix = cos_sim_from_embeddings(embeddings_t)
        embeddings_np = np.asarray(embeddings_t.cpu().numpy(), dtype=np.float64)

        stage1 = concentration_stage1(sim_matrix)
        stage2 = stage2_pair_frequency(sim_matrix, stage1.n_adv_estimated, p=args.stage2_p)

        diag_n_adv = rec.get("N_adv_estimated_by_ragdefender")
        stage1_agree = (diag_n_adv is not None) and (stage1.n_adv_estimated == diag_n_adv)
        stage2_agree = set(stage2.selected_indices) == set(removed_indices_in_diagnostics)
        if diag_n_adv is not None and not stage1_agree:
            failure_stage = "stage1_mismatch"
        elif not stage2_agree:
            failure_stage = "stage2_mismatch"
        else:
            failure_stage = "match"

        log_lines.append(
            f"{qid}: diag_n_adv={diag_n_adv} recomputed_n_adv={stage1.n_adv_estimated} "
            f"stage1_agree={stage1_agree} stage2_agree={stage2_agree} failure_stage={failure_stage}"
        )

        x_pca, y_pca = compute_pca_coords(embeddings_np)

        passage_rows = build_passage_rows(rec, texts, sim_matrix, stage1, stage2, x_pca, y_pca, removed_set)
        passages_path = run_dir / "passages" / f"{qid}_passages.csv"
        pd.DataFrame(passage_rows, columns=PASSAGE_COLUMNS).to_csv(passages_path, index=False)
        manifest["csv"].append(str(passages_path.relative_to(run_dir)))

        csv_path, npy_path, reordered_path, order = save_similarity_matrix(
            run_dir, qid, doc_ids, is_poison, removed_flags, sim_matrix
        )
        manifest["matrices"].extend(str(p.relative_to(run_dir)) for p in (csv_path, npy_path, reordered_path))

        stage1_rows.append(build_stage1_row(rec, stage1))
        stage2_rows.append(build_stage2_row(rec, stage1.n_adv_estimated, stage2, removed_indices_in_diagnostics))
        graph_rows.append(compute_graph_metrics(rec, sim_matrix, is_poison, removed_flags))

        detail_by_qid[qid] = {
            "diag_n_adv": diag_n_adv,
            "recomputed_n_adv": stage1.n_adv_estimated,
            "stage1_agree": stage1_agree,
            "stage2_agree": stage2_agree,
            "failure_stage": failure_stage,
            "narrative": build_query_narrative(
                rec, stage1, stage2, stage1.n_adv_estimated, doc_ids, is_poison, removed_indices_in_diagnostics
            ),
        }

        if plt is not None:
            labels = [f"{i}:{'P' if is_poison[i] else 'C'}{'R' if removed_flags[i] else 'K'}" for i in range(len(doc_ids))]
            reordered_matrix = sim_matrix[np.ix_(order, order)]
            reordered_labels = [labels[i] for i in order]
            plot_paths = [
                plot_pca_scatter(plt, run_dir, qid, x_pca, y_pca, is_poison, removed_flags),
                plot_heatmap(plt, run_dir, qid, sim_matrix, labels, f"{qid}_similarity_heatmap.png",
                             f"Similarity heatmap (original rank order) -- {qid}"),
                plot_heatmap(plt, run_dir, qid, reordered_matrix, reordered_labels, f"{qid}_similarity_heatmap_reordered.png",
                             f"Similarity heatmap (grouped poison/clean, removed/kept) -- {qid}"),
                plot_similarity_distribution(plt, run_dir, qid, sim_matrix, is_poison),
                plot_stage2_pairgraph(plt, run_dir, qid, len(doc_ids), is_poison, removed_flags, stage2),
            ]
            for p in plot_paths:
                if p is not None:
                    manifest["plots"].append(str(p.relative_to(run_dir)))
                else:
                    log_lines.append(f"{qid}: stage2 pair-graph plot skipped (networkx not importable)")

        processed_ids.append(qid)

    stage1_csv = run_dir / "stage1_summary.csv"
    pd.DataFrame(stage1_rows, columns=STAGE1_COLUMNS).to_csv(stage1_csv, index=False)
    manifest["csv"].append(str(stage1_csv.relative_to(run_dir)))

    stage2_csv = run_dir / "stage2_summary.csv"
    pd.DataFrame(stage2_rows, columns=STAGE2_COLUMNS).to_csv(stage2_csv, index=False)
    manifest["csv"].append(str(stage2_csv.relative_to(run_dir)))

    graph_csv = run_dir / "graph_metrics.csv"
    pd.DataFrame(graph_rows, columns=GRAPH_COLUMNS).to_csv(graph_csv, index=False)
    manifest["csv"].append(str(graph_csv.relative_to(run_dir)))

    if args.also_umap:
        if _umap_available():
            log_lines.append("--also_umap requested and umap-learn is importable, but per-query UMAP is not yet "
                              "implemented in this version; skipping (documented gap, not a crash).")
        else:
            log_lines.append("--also_umap requested but umap-learn is not installed; skipping (never a hard failure).")

    embedder_short = slugify_embedder(args.embedder)
    run_config = build_run_config(
        args, dataset, k, n_injected, defense, processed_ids, skipped_ids, skip_reasons,
        embedder_short, plots_enabled, plots_skip_reason,
    )
    run_config_path = run_dir / "run_config.json"
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)
    manifest["config"].append(str(run_config_path.relative_to(run_dir)))

    report_text = render_report(
        run_dir, args, dataset, k, n_injected, defense, processed_ids, skipped_ids, skip_reasons, detail_by_qid
    )
    report_path = run_dir / "RAGDEFENDER_CLUSTER_VISUALIZATION_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    manifest["report"].append(str(report_path.relative_to(run_dir)))

    log_path = run_dir / "logs" / "run.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    manifest["logs"].append(str(log_path.relative_to(run_dir)))

    manifest_path = run_dir / "manifest.json"
    manifest["config"].append(str(manifest_path.relative_to(run_dir)))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote RAGDefender cluster diagnostics run to: {run_dir}")
    print(f"Processed {len(processed_ids)} query(ies), skipped {len(skipped_ids)}.")
    return run_dir


if __name__ == "__main__":
    main()
