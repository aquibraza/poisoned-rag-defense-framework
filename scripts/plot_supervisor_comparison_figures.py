#!/usr/bin/env python3
"""Supervisor-facing comparison figures for the text-mutation / ML-FilterRAG
results that followed the E1 oracle briefing.

Produces two families of plots:

1. **CSV-only comparison charts** (no model load): cross-defense heatmap,
   full-retrieval before/after bars, ML-FilterRAG feature-oracle recall
   curve, and Freq-Density / ML-probability mechanism bars.
2. **E1-style RAGDefender graphs** (loads `paraphrase-MiniLM-L6-v2` only):
   side-by-side Stage-2 top-pair graphs and PCA scatters for two
   FilterRAG-targeted text-mutation cases -- one where the poison clique
   breaks (Schmeichel), one where it stays intact while ML-FilterRAG
   fails (Gibson). Same visual language as
   `scripts/visualize_ragdefender_clusters.py` /
   `scripts/run_cluster_normalized_poisoning.py` (red=poison, blue=clean,
   thick border / x = removed).

Never calls an LLM/API, never retrains, never reruns retrieval. Writes
PNGs under `--out_dir` (default `docs/figures/supervisor_briefing/`).

Usage:
    python scripts/plot_supervisor_comparison_figures.py
    python scripts/plot_supervisor_comparison_figures.py --skip_pairgraphs
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402

DEFAULT_OUT_DIR = os.path.join("docs", "figures", "supervisor_briefing")
DEFAULT_PILOT_DIR = os.path.join("manual_text_mutation_pilot", "hotpotqa_50q_k10")
DPI = 160

FAMILY_LABELS = {
    "ragdefender_targeted": "Aimed at\nRAGDefender",
    "filterrag_targeted": "Aimed at\nFilterRAG",
    "mlfilterrag_targeted": "Aimed at\nML-FilterRAG",
}
FAMILY_ORDER = ("ragdefender_targeted", "filterrag_targeted", "mlfilterrag_targeted")
DEFENSE_ORDER = ("ragdefender", "filterrag", "ml_filterrag")
DEFENSE_LABELS = {
    "ragdefender": "RAGDefender",
    "filterrag": "FilterRAG",
    "ml_filterrag": "ML-FilterRAG",
}
QUERY_SHORT = {
    "5ae224da554299234fd043ee": "Gibson / gin",
    "5ae22b8d554299234fd0440f": "Schmeichel / IFFHS",
    "5a8e068b5542995085b37384": "Ferocactus / Silene",
    "5a7759fc5542993569682d60": "Teide / Garajonay",
    "5aba749055429901930fa7d8": "Menges / Avakian",
    "5a8133725542995ce29dcbdb": "Roth / Childers",
}

# Held-out TEST split, threshold=0.4, from FEATURE_ORACLE_REPORT.md
# (the CSV lives under gitignored results/). Alpha=1.0 is unmodified poison.
ORACLE_RECALL_T04 = {
    "alpha": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    "nearest_clean_bijection": [
        0.9492, 0.9492, 0.9068, 0.8729, 0.8220, 0.8051, 0.6780, 0.5339, 0.3814, 0.3220, 0.1864,
    ],
    "same_query_clean_centroid": [
        0.9492, 0.9407, 0.9068, 0.9153, 0.8559, 0.8136, 0.6864, 0.5000, 0.4661, 0.3220, 0.3390,
    ],
    "clean_centroid": [
        0.9492, 0.9492, 0.9153, 0.9322, 0.9407, 0.9237, 0.8559, 0.7119, 0.7797, 0.9661, 1.0000,
    ],
}

# Two FilterRAG-targeted cases that reuse the E1 slide language.
PAIRGRAPH_CASES = (
    {
        "query_id": "5ae22b8d554299234fd0440f",
        "slug": "schmeichel",
        "file_prefix": "05",
        "pca_prefix": "06",
        "title_left": "Original poison (before rewrite)",
        "title_right": "FilterRAG-targeted rewrite",
        "caption": "Poison clique breaks: top-pair PP 10 -> 3; RAGDefender removes 5 -> 3.",
    },
    {
        "query_id": "5ae224da554299234fd043ee",
        "slug": "gibson",
        "file_prefix": "07",
        "pca_prefix": "08",
        "title_left": "Original poison (before rewrite)",
        "title_right": "FilterRAG-targeted rewrite",
        "caption": "Clique stays (PP=10); RAGDefender still removes all 5. ML-FilterRAG goes 5 -> 0.",
    },
)


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def _setup_style(plt) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_cross_defense_heatmap(plt, rows: List[Dict[str, str]], out_path: Path) -> None:
    """Mean delta removed_poison: mutation family (rows) x defense (cols)."""
    matrix = np.zeros((len(FAMILY_ORDER), len(DEFENSE_ORDER)))
    for i, fam in enumerate(FAMILY_ORDER):
        for j, defense in enumerate(DEFENSE_ORDER):
            match = [r for r in rows if r["family"] == fam and r["defense"] == defense]
            if not match:
                matrix[i, j] = np.nan
            else:
                matrix[i, j] = _f(match[0], "mean_delta_removed_poison")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    vmax = max(2.2, float(np.nanmax(np.abs(matrix))))
    im = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DEFENSE_ORDER)))
    ax.set_xticklabels([DEFENSE_LABELS[d] for d in DEFENSE_ORDER])
    ax.set_yticks(range(len(FAMILY_ORDER)))
    ax.set_yticklabels([FAMILY_LABELS[f].replace("\n", " ") for f in FAMILY_ORDER])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=11,
                    color="white" if abs(val) >= 0.9 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean change in poisoned passages removed (of 5)")
    ax.set_title("Cross-defense effect of targeted text rewrites\nnegative = defense got weaker")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_full_retrieval_bars(plt, rows: List[Dict[str, str]], out_path: Path) -> None:
    """Baseline vs mutated removed_poison for the 3 full-retrieval queries."""
    query_ids = [
        "5a8e068b5542995085b37384",
        "5ae224da554299234fd043ee",
        "5ae22b8d554299234fd0440f",
    ]
    defenses = [
        ("ragdefender_removed_poison", "RAGDefender"),
        ("filterrag_removed_poison", "FilterRAG"),
        ("ml_removed_poison_t04", "ML-FilterRAG"),
    ]
    by_qid = {}
    for r in rows:
        by_qid.setdefault(r["query_id"], {})[r["condition"]] = r

    n_q = len(query_ids)
    n_d = len(defenses)
    x = np.arange(n_q)
    width = 0.12
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    colors_base = ["#4C78A8", "#F58518", "#54A24B"]
    colors_mut = ["#9EC4E4", "#F2C39A", "#B5D8B0"]
    for d_i, (col, label) in enumerate(defenses):
        base_vals = [_f(by_qid[qid]["baseline_recomputed"], col) for qid in query_ids]
        mut_vals = [_f(by_qid[qid]["mutated"], col) for qid in query_ids]
        offset = (d_i - (n_d - 1) / 2) * (2 * width + 0.02)
        bars_b = ax.bar(x + offset - width / 2, base_vals, width, color=colors_base[d_i],
                        label=f"{label} original")
        bars_m = ax.bar(x + offset + width / 2, mut_vals, width, color=colors_mut[d_i],
                        label=f"{label} rewritten", hatch="//", edgecolor=colors_base[d_i], linewidth=0.6)
        for bar, val in list(zip(bars_b, base_vals)) + list(zip(bars_m, mut_vals)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                    f"{int(val)}", ha="center", va="bottom", fontsize=7, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([QUERY_SHORT[qid] for qid in query_ids])
    ax.set_ylabel("Poisoned passages removed")
    ax.set_ylim(0, 7)
    ax.set_title("Full Contriever rerun: original vs FilterRAG-targeted rewrite\nall 5 mutated poisons still retrieved at ranks 1-5")
    ax.legend(ncols=2, fontsize=8, loc="upper right")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_oracle_recall_curve(plt, out_path: Path) -> None:
    alphas = ORACLE_RECALL_T04["alpha"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(alphas, ORACLE_RECALL_T04["nearest_clean_bijection"],
            marker="o", label="Nearest-clean bijection", color="#C44E52")
    ax.plot(alphas, ORACLE_RECALL_T04["same_query_clean_centroid"],
            marker="s", label="Same-query clean centroid", color="#4C78A8")
    ax.plot(alphas, ORACLE_RECALL_T04["clean_centroid"],
            marker="^", label="Global clean centroid", color="#54A24B")
    ax.axhline(0.9, color="#888888", linestyle="--", linewidth=0.8, label="recall = 0.9")
    ax.axvline(0.4, color="#888888", linestyle=":", linewidth=0.8)
    ax.set_xlabel("alpha (1.0 = original poison features, 0.0 = fully clean-like)")
    ax.set_ylabel("Poison recall")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.02, 1.02)
    ax.invert_xaxis()
    ax.set_title("ML-FilterRAG-top-k feature-space oracle (HotpotQA test, t=0.4)\nheld-out queries; clean rows never modified")
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_mechanism_bars(plt, score_rows: List[Dict[str, str]], delta_rows: List[Dict[str, str]],
                         baseline_rows: List[Dict[str, str]], out_path: Path) -> None:
    """Freq-Density and ML poison probability: original vs FilterRAG-targeted rewrite."""
    baseline_by_qid = {r["query_id"]: r for r in baseline_rows}
    mutated = [r for r in score_rows if r["family"] == "filterrag_targeted"]
    mutated.sort(key=lambda r: QUERY_SHORT.get(r["query_id"], r["query_id"]))
    qids = [r["query_id"] for r in mutated]
    labels = [QUERY_SHORT.get(q, q[:8]) for q in qids]

    fd_base, fd_mut, ml_base, ml_mut = [], [], [], []
    for r in mutated:
        qid = r["query_id"]
        if qid in baseline_by_qid:
            b = baseline_by_qid[qid]
            fd_base.append(_f(b, "filterrag_mean_freq_density_poison"))
            ml_base.append(_f(b, "ml_mean_poison_probability"))
        else:
            drow = next(d for d in delta_rows if d["query_id"] == qid and d["family"] == "filterrag_targeted")
            fd_base.append(_f(r, "filterrag_mean_freq_density_poison") - _f(drow, "delta_filterrag_mean_freq_density_poison"))
            ml_base.append(_f(r, "ml_mean_poison_probability") - _f(drow, "delta_ml_mean_poison_probability"))
        fd_mut.append(_f(r, "filterrag_mean_freq_density_poison"))
        ml_mut.append(_f(r, "ml_mean_poison_probability"))

    x = np.arange(len(qids))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))

    axes[0].bar(x - width / 2, fd_base, width, color="#4C78A8", label="Original poison")
    axes[0].bar(x + width / 2, fd_mut, width, color="#F2C39A", edgecolor="#F58518",
                hatch="//", label="FilterRAG-targeted rewrite")
    axes[0].axhline(0.2, color="#C44E52", linestyle="--", linewidth=1, label="FilterRAG epsilon = 0.2")
    axes[0].set_ylabel("Mean Freq-Density (poison passages)")
    axes[0].set_title("Keyword-density signal")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].legend(fontsize=7)

    axes[1].bar(x - width / 2, ml_base, width, color="#4C78A8", label="Original poison")
    axes[1].bar(x + width / 2, ml_mut, width, color="#F2C39A", edgecolor="#F58518",
                hatch="//", label="FilterRAG-targeted rewrite")
    axes[1].axhline(0.4, color="#C44E52", linestyle="--", linewidth=1, label="ML threshold = 0.4")
    axes[1].set_ylabel("Mean predicted poison probability")
    axes[1].set_title("ML-FilterRAG score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=7)

    fig.suptitle("Why FilterRAG-targeted rewrites work: the statistical signal drops, the false claim stays",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def _draw_pairgraph(ax, nx, k: int, is_poison: Sequence[bool], removed_flags: Sequence[bool],
                     stage2, title: str) -> None:
    g = nx.Graph()
    g.add_nodes_from(range(k))
    for x, y, sim in stage2.top_pairs:
        g.add_edge(x, y, weight=sim)
    pos = nx.circular_layout(g, scale=1.0)
    scores = np.asarray(stage2.frequency_scores, dtype=np.float64)
    if scores.size and scores.max() > scores.min():
        norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        norm = np.zeros_like(scores)
    sizes = 280 + norm * 2200
    colors = ["#C44E52" if is_poison[i] else "#4C78A8" for i in range(k)]
    edgecolors = ["black" if removed_flags[i] else "#dddddd" for i in range(k)]
    linewidths = [2.8 if removed_flags[i] else 0.6 for i in range(k)]
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.45, width=1.2, edge_color="#666666")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes, node_color=colors,
                           edgecolors=edgecolors, linewidths=linewidths)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8, font_color="#222222")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    n_poison_removed = sum(1 for i in range(k) if is_poison[i] and removed_flags[i])
    n_poison = sum(1 for i in range(k) if is_poison[i])
    n_clean_removed = sum(1 for i in range(k) if (not is_poison[i]) and removed_flags[i])
    caught_all = n_poison_removed == n_poison
    mark = "all poison removed" if caught_all else "residual poison"
    color = "#2CA02C" if caught_all else "#C44E52"
    ax.text(
        0.5, -0.08,
        f"{mark}  |  poison {n_poison_removed}/{n_poison}, clean {n_clean_removed}",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color=color,
    )


def _draw_pca(ax, xy: np.ndarray, is_poison: Sequence[bool], removed_flags: Sequence[bool],
               title: str) -> None:
    for i in range(len(xy)):
        color = "#C44E52" if is_poison[i] else "#4C78A8"
        if removed_flags[i]:
            ax.scatter(xy[i, 0], xy[i, 1], c=color, marker="x", s=120, linewidths=2.4, zorder=3)
        else:
            ax.scatter(xy[i, 0], xy[i, 1], c=color, marker="o", s=90, edgecolors="none", zorder=3)
        ax.annotate(str(i), (xy[i, 0], xy[i, 1]), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color="#333333")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")


def _ragdefender_snapshot(passages) -> Dict:
    """Embed + Stage-1/2 + dispatch removal, returning plot inputs."""
    from defense import defense_runner
    from defense.dispatch import run_defense
    from defense.passages import removed_passages
    from defense.ragdefender_internals import concentration_stage1, stage2_pair_frequency
    from sentence_transformers import util as st_util

    query = ""  # RAGDefender Stage-1/2 does not use the query text
    kept, _diag = run_defense(
        "ragdefender_original", query, passages, "hotpotqa",
        device="cpu", gpu_id=0, top_k=None,
    )
    removed = removed_passages(passages, kept)
    removed_ids = {p.doc_id for p in removed}

    texts = [p.text for p in passages]
    is_poison = [p.is_poison for p in passages]
    removed_flags = [p.doc_id in removed_ids for p in passages]
    cfg = defense_runner.DefenseConfig(device="cpu")
    s_model = defense_runner._get_s_model(cfg)  # noqa: SLF001
    embeddings = s_model.encode(texts, convert_to_tensor=True)
    sim = st_util.cos_sim(embeddings, embeddings).cpu().numpy()
    z = np.asarray(embeddings.cpu().numpy(), dtype=np.float64)
    stage1 = concentration_stage1(sim)
    stage2 = stage2_pair_frequency(sim, stage1.n_adv_estimated)
    return {
        "k": len(passages),
        "is_poison": is_poison,
        "removed_flags": removed_flags,
        "stage2": stage2,
        "z": z,
        "sim": sim,
    }


def plot_pairgraph_and_pca_cases(plt, nx, pca_fn, pilot_dir: str, out_dir: Path) -> List[Path]:
    import run_normalized_targeted_mutation_bundle_1_eval as norm_eval
    import run_text_mutation_fixed_context_eval as base_eval

    poison_by_query = base_eval.load_mutation_input_passages(
        os.path.join(pilot_dir, "mutation_input_passages.csv")
    )
    clean_by_query = base_eval.load_clean_context_passages(
        os.path.join(pilot_dir, "clean_context_passages.csv")
    )
    family_path = os.path.join(
        pilot_dir, "mutation_bundle_1", "normalized", "filterrag_targeted.normalized.jsonl"
    )
    family = norm_eval.load_normalized_family(family_path, "filterrag_targeted", "filterrag")

    written: List[Path] = []
    for case in PAIRGRAPH_CASES:
        qid = case["query_id"]
        original = base_eval.build_original_context(poison_by_query[qid], clean_by_query[qid])
        bundle = norm_eval.family_record_to_bundle(family[qid])
        mutated = base_eval.build_mutated_context(original, poison_by_query[qid], bundle)
        before = _ragdefender_snapshot(original)
        after = _ragdefender_snapshot(mutated)

        fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.4))
        _draw_pairgraph(axes[0], nx, before["k"], before["is_poison"], before["removed_flags"],
                        before["stage2"], case["title_left"])
        _draw_pairgraph(axes[1], nx, after["k"], after["is_poison"], after["removed_flags"],
                        after["stage2"], case["title_right"])
        fig.suptitle(
            f"Stage-2 top-pair graph -- {QUERY_SHORT[qid]}\n"
            f"red=poison, blue=clean, size=frequency score, thick border=removed\n{case['caption']}",
            fontsize=11,
        )
        fig.tight_layout()
        pair_path = out_dir / f"{case['file_prefix']}_pairgraph_{case['slug']}_before_after.png"
        fig.savefig(pair_path, dpi=DPI)
        plt.close(fig)
        written.append(pair_path)

        z_all = np.vstack([before["z"], after["z"]])
        xy_all = pca_fn(z_all, n_components=2)
        n = before["k"]
        xy_before, xy_after = xy_all[:n], xy_all[n:]
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0))
        _draw_pca(axes[0], xy_before, before["is_poison"], before["removed_flags"],
                  case["title_left"])
        _draw_pca(axes[1], xy_after, after["is_poison"], after["removed_flags"],
                  case["title_right"])
        fig.suptitle(
            f"PCA of RAGDefender embeddings -- {QUERY_SHORT[qid]}\n"
            f"red=poison, blue=clean, x=removed, o=kept  |  {case['caption']}",
            fontsize=11,
        )
        fig.tight_layout()
        pca_path = out_dir / f"{case['pca_prefix']}_pca_{case['slug']}_before_after.png"
        fig.savefig(pca_path, dpi=DPI)
        plt.close(fig)
        written.append(pca_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--pilot_dir", default=DEFAULT_PILOT_DIR)
    parser.add_argument("--skip_pairgraphs", action="store_true",
                        help="Skip MiniLM embed + top-pair/PCA figures.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_style(plt)

    summary_path = os.path.join(
        args.pilot_dir, "mutation_bundle_1", "evaluation_normalized",
        "normalized_targeted_family_summary_by_defense.csv",
    )
    scores_path = os.path.join(
        args.pilot_dir, "mutation_bundle_1", "evaluation_normalized",
        "normalized_targeted_family_bundle_scores.csv",
    )
    deltas_path = os.path.join(
        args.pilot_dir, "mutation_bundle_1", "evaluation_normalized",
        "normalized_targeted_family_bundle_deltas.csv",
    )
    retrieval_path = os.path.join(
        args.pilot_dir, "mutation_bundle_1", "full_retrieval_pilot",
        "full_retrieval_defense_scores.csv",
    )
    baseline_path = os.path.join(
        args.pilot_dir, "mutation_bundle_0", "evaluation",
        "fixed_context_baseline_by_query.csv",
    )

    written: List[Path] = []

    p = out_dir / "01_cross_defense_heatmap.png"
    plot_cross_defense_heatmap(plt, _read_csv(summary_path), p)
    written.append(p)

    p = out_dir / "02_full_retrieval_before_after.png"
    plot_full_retrieval_bars(plt, _read_csv(retrieval_path), p)
    written.append(p)

    p = out_dir / "03_ml_feature_oracle_recall_vs_alpha.png"
    plot_oracle_recall_curve(plt, p)
    written.append(p)

    p = out_dir / "04_mechanism_freq_density_and_ml_prob.png"
    plot_mechanism_bars(plt, _read_csv(scores_path), _read_csv(deltas_path),
                        _read_csv(baseline_path), p)
    written.append(p)

    if not args.skip_pairgraphs:
        try:
            import networkx as nx
            from sklearn.decomposition import PCA

            def pca_fn(z, n_components=2):
                return PCA(n_components=n_components, random_state=0).fit_transform(z)

            written.extend(plot_pairgraph_and_pca_cases(plt, nx, pca_fn, args.pilot_dir, out_dir))
        except ImportError as exc:
            print(f"Skipping pairgraph/PCA figures ({exc}).")

    print("Wrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
