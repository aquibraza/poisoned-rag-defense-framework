#!/usr/bin/env python3
"""Cluster-Normalized Poisoning: oracle embedding-space stress test runner.

Implements the execution plan in
`docs/CLUSTER_NORMALIZED_POISONING_EXECUTION_PLAN.md`: for one
`(query_id, intervention, anchor_strategy)` combination, this script

1. re-encodes the query's exact retrieved passage text (recovered from
   `query_results/*.json`, the same way
   `scripts/visualize_ragdefender_clusters.py::recover_pre_defense_texts`
   does),
2. splits the embeddings into poison/clean sub-blocks,
3. applies an oracle intervention (E0 clean-centroid interpolation, or E1
   clean-anchor interpolation with one of four anchor strategies) to the
   poison sub-block *only*, across a sweep of `alpha` values,
4. recomputes the cosine matrix and feeds it into the **unmodified**
   `defense/ragdefender_internals.py::concentration_stage1` /
   `stage2_pair_frequency`,
5. writes every artifact `docs/CLUSTER_NORMALIZED_POISONING_EXECUTION_PLAN.md`
   section 7 specifies.

This script never imports or calls `defense/defense_runner.py`,
`defense/dispatch.py`, `defense/filterrag.py`, or `main.py`, never performs
generation, and never makes a network/LLM/API call (the embedder is loaded
fully offline from the local sentence-transformers cache -- see
`_force_offline_env` below). Retrieval membership (`k`, which doc_ids were
retrieved) is never changed; only the poisoned passages' *embeddings* are
transformed, purely to test RAGDefender's own similarity-based decision.

Usage:
    python scripts/run_cluster_normalized_poisoning.py \\
        --diagnostics_jsonl results/diagnostics/ragdefender_smoke_live_10q/hotpotqa-...-defense-ragdefender_original.jsonl \\
        --query_results_dir results/query_results/ragdefender_smoke_live_10q \\
        --query_id 5ae2070a5542994d89d5b313 \\
        --intervention E1 --anchor_strategy nearest_bijection
"""
from __future__ import annotations

import argparse
import json
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
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def _force_offline_env() -> None:
    """Guarantee the embedder load below can never reach the network:
    forces sentence-transformers/huggingface_hub to use the local cache
    only. Set as early as possible, before any sentence_transformers
    import happens (via `visualize_ragdefender_clusters`'s lazy import)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_force_offline_env()

import numpy as np
import pandas as pd

try:
    import fix_sentence_transformers  # noqa: F401 -- optional compat patch for older sentence_transformers
except ImportError:
    pass

import visualize_ragdefender_clusters as viz  # noqa: E402 -- sys.path set up above

from defense.cluster_normalized_poisoning import (  # noqa: E402
    ANCHOR_STRATEGIES,
    anchor_interpolate,
    centroid_interpolate,
    recombine_poison_clean,
    resolve_anchor_permutation,
    split_poison_clean,
)
from defense.ragdefender_internals import (  # noqa: E402
    ConcentrationResult,
    Stage2Result,
    concentration_stage1,
    stage2_pair_frequency,
)

DEFAULT_OUTPUT_DIR = os.path.join("results", "diagnostics", "cluster_normalized_poisoning")
DEFAULT_EMBEDDER = "paraphrase-MiniLM-L6-v2"
DEFAULT_STAGE2_P = 2.0
DEFAULT_ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
DEFAULT_RANDOM_SEED = 12
INTERVENTIONS = ("E0", "E1")


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------

def intervention_slug(intervention: str, anchor_strategy: Optional[str]) -> str:
    if intervention == "E0":
        return "E0"
    return f"E1-{anchor_strategy}"


def build_run_dir(output_dir: str, dataset: str, k: int, n_injected: int,
                   intervention: str, anchor_strategy: Optional[str], query_id: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = intervention_slug(intervention, anchor_strategy)
    name = f"{ts}_oracle_{dataset}_k{k}_N{n_injected}_{slug}_{query_id}"
    run_dir = Path(output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "similarity_matrices").mkdir()
    (run_dir / "plots").mkdir()
    return run_dir


# --------------------------------------------------------------------------
# Metrics snapshot (before/after), independent of any single query_results
# `rec` dict -- computed purely from a cosine matrix + is_poison + a
# Stage2Result's selection, so it applies equally to the original state and
# every transformed alpha.
# --------------------------------------------------------------------------

def _similarity_group_stats(sim_matrix: np.ndarray, is_poison: Sequence[bool]) -> Dict[str, Optional[float]]:
    k = sim_matrix.shape[0]
    pp, pc, cc = [], [], []
    for i in range(k):
        for j in range(i + 1, k):
            v = float(sim_matrix[i, j])
            if is_poison[i] and is_poison[j]:
                pp.append(v)
            elif is_poison[i] != is_poison[j]:
                pc.append(v)
            else:
                cc.append(v)
    return {
        "mean_poison_poison_similarity": float(np.mean(pp)) if pp else None,
        "mean_poison_clean_similarity": float(np.mean(pc)) if pc else None,
        "mean_clean_clean_similarity": float(np.mean(cc)) if cc else None,
        "max_poison_poison_similarity": float(np.max(pp)) if pp else None,
    }


def _top_pair_mix(top_pairs: List[Tuple[int, int, float]], is_poison: Sequence[bool]) -> Tuple[int, int, int]:
    pp = pc = cc = 0
    for i, j, _ in top_pairs:
        if is_poison[i] and is_poison[j]:
            pp += 1
        elif is_poison[i] != is_poison[j]:
            pc += 1
        else:
            cc += 1
    return pp, pc, cc


def _neighborhood_entropies(sim_matrix: np.ndarray, is_poison: Sequence[bool]) -> Dict[str, Optional[float]]:
    k = sim_matrix.shape[0]
    nn_is_poison = []
    for i in range(k):
        nn_idx, _ = viz._nearest_neighbor(sim_matrix, i)
        nn_is_poison.append(bool(is_poison[nn_idx]) if nn_idx is not None else False)
    nn_poison_ratio = float(np.mean(nn_is_poison)) if nn_is_poison else None
    poison_nn_labels = [nn_is_poison[i] for i in range(k) if is_poison[i]]
    clean_nn_labels = [nn_is_poison[i] for i in range(k) if not is_poison[i]]
    return {
        "nearest_neighbor_poison_ratio_mean": nn_poison_ratio,
        "poison_neighborhood_entropy": viz._shannon_entropy_binary(poison_nn_labels),
        "clean_neighborhood_entropy": viz._shannon_entropy_binary(clean_nn_labels),
    }


def compute_metrics_snapshot(sim_matrix: np.ndarray, is_poison: Sequence[bool],
                              stage1: ConcentrationResult, stage2: Stage2Result) -> Dict:
    k = sim_matrix.shape[0]
    n_poison = sum(1 for p in is_poison if p)
    n_clean = k - n_poison
    removed_poison = sum(1 for idx in stage2.selected_indices if is_poison[idx])
    removed_clean = sum(1 for idx in stage2.selected_indices if not is_poison[idx])
    residual_poison_fraction = ((n_poison - removed_poison) / n_poison) if n_poison else None
    pp, pc, cc = _top_pair_mix(stage2.top_pairs, is_poison)

    row: Dict = {
        "k": k,
        "N_retrieved_poison": n_poison,
        "N_retrieved_clean": n_clean,
        "N_adv": stage1.n_adv_estimated,
        "top_pair_pp": pp,
        "top_pair_pc": pc,
        "top_pair_cc": cc,
        "removed_poison": removed_poison,
        "removed_clean": removed_clean,
        "residual_poison_fraction": residual_poison_fraction,
        "selected_indices": json.dumps(list(stage2.selected_indices)),
    }
    row.update(_similarity_group_stats(sim_matrix, is_poison))
    row.update(_neighborhood_entropies(sim_matrix, is_poison))
    return row


def decision_label(removed_poison: int, removed_clean: int, n_retrieved_poison: int) -> str:
    """Absolute classification of RAGDefender's post-transform removal
    outcome for a single alpha -- **not** relative to the alpha=1.0
    baseline of this run. This mirrors the category scheme already used in
    `docs/RAGDEFENDER_CLUSTER_DIAGNOSTIC_FINDINGS.md` (which labels
    `5a8cb288554299585d9e3726` as `clean_density_failure`), so a label
    means the same thing whether it comes from the original untransformed
    diagnostics or from a transformed alpha in this sweep:

    - `poison_removal_success`: all retrieved poison removed, no clean
      passage removed (`removed_poison == n_retrieved_poison and
      removed_clean == 0`).
    - `over_removal_success`: all retrieved poison removed, but at least
      one clean passage was also removed (`removed_poison ==
      n_retrieved_poison and removed_clean > 0`).
    - `residual_poison_failure`: some but not all poison removed, and no
      clean false positive (`0 < removed_poison < n_retrieved_poison and
      removed_clean == 0`).
    - `residual_poison_with_clean_false_positive`: some but not all
      poison removed, *and* a clean passage was incorrectly removed too
      (`0 < removed_poison < n_retrieved_poison and removed_clean > 0`).
    - `clean_density_failure`: no poison removed at all, but clean
      passage(s) were removed instead (`removed_poison == 0 and
      removed_clean > 0`).
    - `no_removal_or_other`: nothing removed, or any other configuration
      not covered above (e.g. `n_retrieved_poison == 0`).
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


# --------------------------------------------------------------------------
# Plots (best-effort; never raises past a missing optional dependency)
# --------------------------------------------------------------------------

def _plot_pairgraph(plt, path: Path, title: str, k: int, is_poison: Sequence[bool],
                     removed_flags: Sequence[bool], stage2: Stage2Result) -> Optional[Path]:
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
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_similarity_distribution_before_after(plt, path: Path, qid: str,
                                                 sim_before: np.ndarray, sim_after: np.ndarray,
                                                 alpha_after: float, is_poison: Sequence[bool]) -> Path:
    def _pp_pc_cc(sim_matrix):
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
        return pp, pc, cc

    pp0, pc0, cc0 = _pp_pc_cc(sim_before)
    pp1, pc1, cc1 = _pp_pc_cc(sim_after)
    bins = np.linspace(-1, 1, 25)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, pp, pc, cc, title in (
        (axes[0], pp0, pc0, cc0, f"Before (original) -- {qid}"),
        (axes[1], pp1, pc1, cc1, f"After alpha={alpha_after} -- {qid}"),
    ):
        if pp:
            ax.hist(pp, bins=bins, alpha=0.5, label=f"poison-poison (n={len(pp)})", color="tab:red")
        if pc:
            ax.hist(pc, bins=bins, alpha=0.5, label=f"poison-clean (n={len(pc)})", color="tab:purple")
        if cc:
            ax.hist(cc, bins=bins, alpha=0.5, label=f"clean-clean (n={len(cc)})", color="tab:blue")
        ax.set_xlabel("cosine similarity")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("pair count")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# run_config.json helpers (mirrors visualize_ragdefender_clusters.py's
# conventions)
# --------------------------------------------------------------------------

def _run_git(args: List[str], cwd: str) -> Optional[str]:
    try:
        out = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_status_short(repo_dir: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "status", "--short"], cwd=repo_dir, capture_output=True, text=True, timeout=10)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def build_run_config(args: argparse.Namespace, dataset: str, k: int, n_injected: int, defense: str,
                      query_id: str, anchor_assignment, plots_enabled: bool,
                      plots_skip_reason: Optional[str]) -> Dict:
    pi_pairs = None
    pi_is_bijection = None
    anchor_strategy_out = None
    random_seed_out = None
    objective_value = None
    if anchor_assignment is not None:
        pi_pairs = [[i, c] for i, c in enumerate(anchor_assignment.pi)]
        pi_is_bijection = anchor_assignment.is_bijection
        anchor_strategy_out = anchor_assignment.strategy
        random_seed_out = anchor_assignment.random_seed
        objective_value = anchor_assignment.objective_value

    return {
        "timestamp": datetime.now().isoformat(),
        "run_type": "cluster_normalized_poisoning_oracle",
        "dataset": dataset,
        "k": k,
        "N_injected": n_injected,
        "defense": defense,
        "query_id": query_id,
        "diagnostics_jsonl": os.path.abspath(args.diagnostics_jsonl),
        "query_results_dir": os.path.abspath(args.query_results_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "intervention": args.intervention,
        "anchor_strategy": anchor_strategy_out,
        "random_seed": random_seed_out,
        "pi": pi_pairs,
        "pi_is_bijection": pi_is_bijection,
        "pi_objective_value": objective_value,
        "alphas": args.alphas,
        "intervention_level": "embedding",
        "embedder": args.embedder,
        "stage2_p": args.stage2_p,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": viz._try_version("torch"),
        "sentence_transformers_version": viz._try_version("sentence_transformers"),
        "sklearn_version": viz._try_version("sklearn"),
        "pandas_version": viz._try_version("pandas"),
        "matplotlib_version": viz._try_version("matplotlib"),
        "git_commit": _run_git(["rev-parse", "HEAD"], REPO_ROOT),
        "git_status_short": git_status_short(REPO_ROOT),
        "ragdefender_package_imported": False,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        "oracle_constraints": {
            "retrieval_membership_fixed": True,
            "generator_text_fixed": True,
            "transform_scope": "ragdefender_similarity_decision_only",
            "claims_text_realizable_attack": False,
            "gpt_or_api_calls_made": False,
            "baseline_files_modified": [],
            "anchor_strategy": anchor_strategy_out,
            "random_seed": random_seed_out,
            "pi_is_bijection": pi_is_bijection,
        },
        "plots_enabled": plots_enabled,
        "plots_skip_reason": plots_skip_reason,
        "argv": sys.argv,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

LIMITATIONS_TEXT = """## Limitations (verbatim, per the execution plan section 10)

- This is an **oracle embedding-space diagnostic**. `Z_poison` is transformed
  directly; no natural-language rewrite of any poisoned passage is performed
  or implied.
- It **does not prove natural-language realizability**. A finding that the
  intervention flips RAGDefender's decision says only that the *defense*
  depends on a geometric assumption that is fragile under a controlled
  representation change -- it does not show the transform is reachable by
  rewriting the poisoned passage's text under the frozen
  `paraphrase-MiniLM-L6-v2` encoder.
- **Text-space mutation is a later phase**, out of scope for this run.
- **FilterRAG and ML-FilterRAG comparisons come after** the RAGDefender
  oracle study and are not part of this run.
- **Centroid interpolation (E0) may increase poison-poison similarity**
  instead of reducing it; it is only a sanity baseline.
- **Clean-anchor interpolation (E1) is still an oracle embedding
  intervention, not a text-realizable attack.**
- **Alpha values below 0.5 may be geometrically extreme** and must not be
  interpreted as plausible natural-language passage rewrites.
- **`nearest_bijection`/`farthest_bijection`'s brute-force permutation
  search is `O(N!)`**, tractable here only because `N_poison` is small.
"""


def render_report(run_dir: Path, args: argparse.Namespace, query_id: str, dataset: str, k: int,
                   n_injected: int, original_row: Dict, sweep_rows: List[Dict],
                   anchor_assignment, stopping_alphas: Dict[str, Optional[float]]) -> str:
    slug = intervention_slug(args.intervention, args.anchor_strategy)
    lines = [
        "# Cluster-Normalized Poisoning Oracle Report",
        "",
        f"Run directory: `{run_dir.name}`",
        "",
        f"- query_id: `{query_id}`, dataset: `{dataset}`, k={k}, N_injected={n_injected}",
        f"- intervention: `{slug}`",
        f"- embedder: `{args.embedder}` (offline load; `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`)",
        "- **No GPT/API calls made. No baseline defense modified. No baseline sweep rerun.**",
        "",
    ]
    if anchor_assignment is not None:
        lines += [
            "## Anchor assignment (E1)",
            "",
            f"- strategy: `{anchor_assignment.strategy}`",
            f"- pi (poison_local_index -> clean_local_index): "
            f"{[[i, c] for i, c in enumerate(anchor_assignment.pi)]}",
            f"- is_bijection: `{anchor_assignment.is_bijection}`",
            f"- objective_value (sum cos at chosen pi, nearest/farthest_bijection only): "
            f"{anchor_assignment.objective_value}",
            "",
        ]

    lines += [
        "## Original (untransformed) state",
        "",
        f"- N_adv={original_row['N_adv']}, top_pair_mix (PP/PC/CC) = "
        f"{original_row['top_pair_pp']}/{original_row['top_pair_pc']}/{original_row['top_pair_cc']}",
        f"- removed_poison={original_row['removed_poison']}, removed_clean={original_row['removed_clean']}, "
        f"residual_poison_fraction={original_row['residual_poison_fraction']}",
        f"- mean_poison_poison_similarity={original_row['mean_poison_poison_similarity']}, "
        f"mean_poison_clean_similarity={original_row['mean_poison_clean_similarity']}, "
        f"mean_clean_clean_similarity={original_row['mean_clean_clean_similarity']}",
        "",
        "## Alpha sweep",
        "",
        "Note: `decision_label` below is an **absolute** classification of each "
        "alpha's removal outcome (see `decision_label()` in this script) -- it is "
        "not relative to the alpha=1.0 row of this table.",
        "",
        "| alpha | N_adv | top_pair (PP/PC/CC) | removed_poison | removed_clean | "
        "residual_poison_fraction | decision_label |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sweep_rows:
        lines.append(
            f"| {row['alpha']} | {row['N_adv']} | {row['top_pair_pp']}/{row['top_pair_pc']}/{row['top_pair_cc']} | "
            f"{row['removed_poison']} | {row['removed_clean']} | {row['residual_poison_fraction']} | "
            f"{row['decision_label']} |"
        )

    lines += [
        "",
        "## First alpha (descending from 1.0) at which each stopping condition first triggers",
        "",
        "(These conditions are relative to this run's own alpha=1.0 row -- a "
        "separate, complementary view from the absolute `decision_label` above.)",
        "",
    ]
    condition_text = {
        "pp_decreased": "Poison-poison top-pair count strictly decreases from the alpha=1.0 value",
        "pc_increased": "Poison-clean top-pair count strictly increases from the alpha=1.0 value",
        "fewer_poison_removed": "RAGDefender removes strictly fewer poisoned passages than at alpha=1.0",
        "clean_removed_increased": "A clean passage is newly selected for removal (removed_clean increases)",
    }
    for key, desc in condition_text.items():
        val = stopping_alphas.get(key)
        if val is None:
            lines.append(f"- **{key}** ({desc}): not triggered by any alpha in this sweep.")
        else:
            lines.append(f"- **{key}** ({desc}): first triggered at alpha={val}.")

    lines += ["", LIMITATIONS_TEXT]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI / main
# --------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostics_jsonl", required=True)
    parser.add_argument("--query_results_dir", required=True)
    parser.add_argument("--query_id", required=True)
    parser.add_argument("--intervention", required=True, choices=INTERVENTIONS)
    parser.add_argument("--anchor_strategy", default=None, choices=ANCHOR_STRATEGIES,
                         help="Required when --intervention E1.")
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--stage2_p", type=float, default=DEFAULT_STAGE2_P)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args(argv)
    if args.intervention == "E1" and args.anchor_strategy is None:
        parser.error("--anchor_strategy is required when --intervention E1")
    return args


def main(argv: Optional[List[str]] = None) -> Path:
    args = parse_args(argv)

    records = viz._read_jsonl(args.diagnostics_jsonl)
    matches = [r for r in records if r["query_id"] == args.query_id]
    if not matches:
        raise ValueError(f"query_id {args.query_id!r} not found in {args.diagnostics_jsonl}")
    rec = matches[0]
    dataset, k, n_injected, defense = rec["dataset"], rec["k"], rec["N_injected"], rec["defense"]

    query_results_index = viz.load_query_results_index(args.query_results_dir)
    qr = query_results_index.get(args.query_id)
    texts = viz.recover_pre_defense_texts(qr)
    expected_k = len(rec["retrieved_doc_ids"])
    if texts is None or len(texts) != expected_k:
        reason = (
            "query_results record not found for this query_id" if qr is None else
            "input_prompt_no_defense missing or recovered passage count mismatch"
        )
        raise ValueError(f"Cannot recover exact passage text for {args.query_id!r}: {reason}")

    doc_ids = list(rec["retrieved_doc_ids"])
    is_poison = [bool(x) for x in rec["retrieved_is_poison"]]

    run_dir = build_run_dir(args.output_dir, dataset, k, n_injected, args.intervention, args.anchor_strategy,
                             args.query_id)
    manifest: Dict[str, List[str]] = {"config": [], "csv": [], "matrices": [], "plots": [], "report": []}

    plots_skip_reason = None
    if args.no_plots:
        plt = None
        plots_skip_reason = "--no_plots passed"
    else:
        plt = viz._configure_matplotlib()
        if plt is None:
            plots_skip_reason = "matplotlib import failed (missing dependency)"
    plots_enabled = plt is not None

    model = viz.load_embedder(args.embedder)
    embeddings_t = viz.encode_texts(model, texts)
    sim_before = viz.cos_sim_from_embeddings(embeddings_t)
    z = np.asarray(embeddings_t.cpu().numpy(), dtype=np.float64)

    stage1_before = concentration_stage1(sim_before)
    stage2_before = stage2_pair_frequency(sim_before, stage1_before.n_adv_estimated, p=args.stage2_p)
    original_row = compute_metrics_snapshot(sim_before, is_poison, stage1_before, stage2_before)

    original_metrics_path = run_dir / "original_metrics.csv"
    pd.DataFrame([{
        "query_id": args.query_id, "dataset": dataset, **original_row,
        "decision_label": decision_label(original_row["removed_poison"], original_row["removed_clean"],
                                          original_row["N_retrieved_poison"]),
    }]).to_csv(original_metrics_path, index=False)
    manifest["csv"].append(str(original_metrics_path.relative_to(run_dir)))

    z_poison, z_clean, poison_idx, clean_idx = split_poison_clean(z, is_poison)
    if args.intervention == "E1" and len(poison_idx) != len(clean_idx):
        raise ValueError(
            f"E1 requires N_poison == N_clean for a bijective assignment; got "
            f"N_poison={len(poison_idx)}, N_clean={len(clean_idx)} for query_id={args.query_id!r}."
        )

    anchor_assignment = None
    if args.intervention == "E1":
        anchor_assignment = resolve_anchor_permutation(
            z_poison, z_clean, args.anchor_strategy, random_seed=args.random_seed
        )

    np.save(run_dir / "similarity_matrices" / "original_M.npy", sim_before)
    manifest["matrices"].append(str((run_dir / "similarity_matrices" / "original_M.npy").relative_to(run_dir)))

    removed_before = [doc_ids[i] in {doc_ids[j] for j in stage2_before.selected_indices} for i in range(k)]
    if plots_enabled:
        p = _plot_pairgraph(
            plt, run_dir / "plots" / "pairgraph_before.png",
            f"Stage-2 top-pair graph (before) -- {args.query_id}\nred=poison blue=clean, thick border=removed",
            k, is_poison, removed_before, stage2_before,
        )
        if p is not None:
            manifest["plots"].append(str(p.relative_to(run_dir)))

    stage1_rows: List[Dict] = [{
        "alpha": None, "avg_avg": stage1_before.avg_avg, "avg_median": stage1_before.avg_median,
        "combined_threshold": stage1_before.combined_threshold,
        "concentration_flag_count": int(stage1_before.raw_or_flag.sum()),
        "result_flipped": stage1_before.flipped, "N_adv": stage1_before.n_adv_estimated,
        "matches_original_N_adv": True,
    }]
    stage2_rows: List[Dict] = [{
        "alpha": None, "N_pairs": stage2_before.n_pairs,
        "top_pairs": json.dumps([[int(i), int(j), round(float(s), 6)] for i, j, s in stage2_before.top_pairs]),
        "frequency_scores": json.dumps([round(float(s), 6) for s in stage2_before.frequency_scores]),
        "selected_indices": json.dumps(list(stage2_before.selected_indices)),
        "removed_poison": original_row["removed_poison"], "removed_clean": original_row["removed_clean"],
        "matches_original_selected_indices": True,
    }]
    normalized_rows: List[Dict] = []
    sweep_rows: List[Dict] = []

    # `baseline_*` below are used ONLY for the relative "stopping condition"
    # tracking (first alpha where X changes from this run's own alpha=1.0
    # row) -- NOT for `decision_label`, which is an absolute classification
    # (see `decision_label()` docstring).
    baseline_removed_poison = original_row["removed_poison"]
    baseline_removed_clean = original_row["removed_clean"]
    baseline_top_pair_pp = original_row["top_pair_pp"]
    baseline_top_pair_pc = original_row["top_pair_pc"]

    stopping_alphas: Dict[str, Optional[float]] = {
        "pp_decreased": None, "pc_increased": None,
        "fewer_poison_removed": None, "clean_removed_increased": None,
    }

    alphas_sorted = sorted(args.alphas, reverse=True)
    for alpha in alphas_sorted:
        if args.intervention == "E0":
            z_poison_prime = centroid_interpolate(z_poison, z_clean, alpha)
        else:
            z_poison_prime = anchor_interpolate(z_poison, z_clean, anchor_assignment.pi, alpha)
        z_prime = recombine_poison_clean(z_poison_prime, z_clean, poison_idx, clean_idx, k)

        import torch
        embeddings_prime_t = torch.tensor(z_prime, dtype=torch.float32)
        sim_after = viz.cos_sim_from_embeddings(embeddings_prime_t)

        stage1_after = concentration_stage1(sim_after)
        stage2_after = stage2_pair_frequency(sim_after, stage1_after.n_adv_estimated, p=args.stage2_p)
        row = compute_metrics_snapshot(sim_after, is_poison, stage1_after, stage2_after)
        label = decision_label(row["removed_poison"], row["removed_clean"], row["N_retrieved_poison"])

        cond_pp = row["top_pair_pp"] < baseline_top_pair_pp
        cond_pc = row["top_pair_pc"] > baseline_top_pair_pc
        cond_fewer = row["removed_poison"] < baseline_removed_poison
        cond_clean = row["removed_clean"] > baseline_removed_clean
        if cond_pp and stopping_alphas["pp_decreased"] is None:
            stopping_alphas["pp_decreased"] = alpha
        if cond_pc and stopping_alphas["pc_increased"] is None:
            stopping_alphas["pc_increased"] = alpha
        if cond_fewer and stopping_alphas["fewer_poison_removed"] is None:
            stopping_alphas["fewer_poison_removed"] = alpha
        if cond_clean and stopping_alphas["clean_removed_increased"] is None:
            stopping_alphas["clean_removed_increased"] = alpha

        matches_original_n_adv = stage1_after.n_adv_estimated == stage1_before.n_adv_estimated
        matches_original_selection = set(stage2_after.selected_indices) == set(stage2_before.selected_indices)

        np.save(run_dir / "similarity_matrices" / f"transformed_M_alpha{alpha}.npy", sim_after)
        manifest["matrices"].append(
            str((run_dir / "similarity_matrices" / f"transformed_M_alpha{alpha}.npy").relative_to(run_dir))
        )

        removed_after = [doc_ids[i] in {doc_ids[j] for j in stage2_after.selected_indices} for i in range(k)]
        if plots_enabled:
            p = _plot_pairgraph(
                plt, run_dir / "plots" / f"pairgraph_after_alpha{alpha}.png",
                f"Stage-2 top-pair graph (after, alpha={alpha}) -- {args.query_id}\n"
                f"red=poison blue=clean, thick border=removed",
                k, is_poison, removed_after, stage2_after,
            )
            if p is not None:
                manifest["plots"].append(str(p.relative_to(run_dir)))

        stage1_rows.append({
            "alpha": alpha, "avg_avg": stage1_after.avg_avg, "avg_median": stage1_after.avg_median,
            "combined_threshold": stage1_after.combined_threshold,
            "concentration_flag_count": int(stage1_after.raw_or_flag.sum()),
            "result_flipped": stage1_after.flipped, "N_adv": stage1_after.n_adv_estimated,
            "matches_original_N_adv": matches_original_n_adv,
        })
        stage2_rows.append({
            "alpha": alpha, "N_pairs": stage2_after.n_pairs,
            "top_pairs": json.dumps([[int(i), int(j), round(float(s), 6)] for i, j, s in stage2_after.top_pairs]),
            "frequency_scores": json.dumps([round(float(s), 6) for s in stage2_after.frequency_scores]),
            "selected_indices": json.dumps(list(stage2_after.selected_indices)),
            "removed_poison": row["removed_poison"], "removed_clean": row["removed_clean"],
            "matches_original_selected_indices": matches_original_selection,
        })
        normalized_rows.append({"alpha": alpha, **row, "decision_label": label})
        sweep_rows.append({
            "alpha": alpha, "intervention": args.intervention, "anchor_strategy": args.anchor_strategy,
            "random_seed": args.random_seed if args.anchor_strategy == "random" else None,
            **row, "decision_label": label,
            "cond_pp_decreased": cond_pp, "cond_pc_increased": cond_pc,
            "cond_fewer_poison_removed": cond_fewer, "cond_clean_removed_increased": cond_clean,
        })

    if plots_enabled:
        min_alpha = min(alphas_sorted)
        z_poison_prime_min = (
            centroid_interpolate(z_poison, z_clean, min_alpha) if args.intervention == "E0"
            else anchor_interpolate(z_poison, z_clean, anchor_assignment.pi, min_alpha)
        )
        z_prime_min = recombine_poison_clean(z_poison_prime_min, z_clean, poison_idx, clean_idx, k)
        import torch
        sim_min = viz.cos_sim_from_embeddings(torch.tensor(z_prime_min, dtype=torch.float32))
        p = _plot_similarity_distribution_before_after(
            plt, run_dir / "plots" / "similarity_distribution_before_after.png",
            args.query_id, sim_before, sim_min, min_alpha, is_poison,
        )
        manifest["plots"].append(str(p.relative_to(run_dir)))

    normalized_metrics_path = run_dir / "normalized_metrics.csv"
    pd.DataFrame(normalized_rows).to_csv(normalized_metrics_path, index=False)
    manifest["csv"].append(str(normalized_metrics_path.relative_to(run_dir)))

    sweep_path = run_dir / "intervention_sweep.csv"
    pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)
    manifest["csv"].append(str(sweep_path.relative_to(run_dir)))

    stage1_path = run_dir / "stage1_before_after.csv"
    pd.DataFrame(stage1_rows).to_csv(stage1_path, index=False)
    manifest["csv"].append(str(stage1_path.relative_to(run_dir)))

    stage2_path = run_dir / "stage2_before_after.csv"
    pd.DataFrame(stage2_rows).to_csv(stage2_path, index=False)
    manifest["csv"].append(str(stage2_path.relative_to(run_dir)))

    run_config = build_run_config(args, dataset, k, n_injected, defense, args.query_id, anchor_assignment,
                                   plots_enabled, plots_skip_reason)
    run_config_path = run_dir / "run_config.json"
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)
    manifest["config"].append(str(run_config_path.relative_to(run_dir)))

    report_text = render_report(run_dir, args, args.query_id, dataset, k, n_injected, original_row, sweep_rows,
                                 anchor_assignment, stopping_alphas)
    report_path = run_dir / "CLUSTER_NORMALIZED_POISONING_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    manifest["report"].append(str(report_path.relative_to(run_dir)))

    manifest_path = run_dir / "manifest.json"
    manifest["config"].append(str(manifest_path.relative_to(run_dir)))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote cluster-normalized-poisoning oracle run to: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
