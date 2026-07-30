#!/usr/bin/env python3
"""Cluster-Normalized Poisoning -- CORAL PCA/subspace oracle intervention.

**Step 1 only** of the CORAL/MMD oracle intervention plan (see
`docs/CORAL_MMD_ORACLE_INTERVENTION_PLAN.md` and
`.cursor/plans/coral_mmd_oracle_intervention_plan_31a5d179.plan.md`): runs
`defense.coral_mmd_intervention.coral_pca_transform` (PCA/subspace
CORAL-style covariance alignment) across a `beta` sweep, for the same 6
HotpotQA k=10/N=5 originally-successful RAGDefender cases already tested by
E1 in `scripts/build_batch_comparison_success_cases.py`.

Full-dimensional ridge-regularized CORAL and the MMD-minimizing optimizer
are **not implemented by this script** (deferred to later steps of that
plan).

Like `scripts/run_cluster_normalized_poisoning.py`, this script:

- never imports or calls `defense/defense_runner.py`, `defense/dispatch.py`,
  `defense/filterrag.py`, or `main.py`,
- never performs generation and never makes a network/LLM/API call (the
  embedder is loaded fully offline from the local sentence-transformers
  cache),
- never changes retrieval membership (`k`, which doc_ids were retrieved) --
  only the poisoned passages' *embeddings* are transformed,
- **never reruns E1** -- the E1 comparison section of the report is read
  directly from `results/diagnostics/cluster_normalized_poisoning/
  BATCH_COMPARISON_SUCCESS_CASES.csv`, an artifact already on disk from a
  prior run of `scripts/build_batch_comparison_success_cases.py`.

Usage:
    python scripts/run_coral_pca_oracle_intervention.py
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
    """Guarantee the embedder load below can never reach the network. Set
    as early as possible, before any sentence_transformers import happens
    (via `visualize_ragdefender_clusters`'s lazy import). Mirrors
    `run_cluster_normalized_poisoning.py::_force_offline_env` exactly."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_force_offline_env()

import numpy as np
import pandas as pd
import torch

import visualize_ragdefender_clusters as viz  # noqa: E402
import run_cluster_normalized_poisoning as cnp_run  # noqa: E402 -- compute_metrics_snapshot, decision_label
import build_batch_comparison_success_cases as batchmod  # noqa: E402

from defense.cluster_normalized_poisoning import (  # noqa: E402
    recombine_poison_clean,
    split_poison_clean,
)
from defense.coral_mmd_intervention import (  # noqa: E402
    compute_preservation_metrics,
    coral_pca_transform,
)
from defense.distribution_metrics import (  # noqa: E402
    DEFAULT_MMD_GAMMA,
    coral_distance_from_gram,
    mmd_rbf_distance_from_gram,
    slice_gram_blocks,
)
from defense.ragdefender_internals import (  # noqa: E402
    concentration_stage1,
    stage2_pair_frequency,
)

DEFAULT_OUTPUT_DIR = os.path.join("results", "diagnostics", "cluster_normalized_poisoning_formal")
DEFAULT_E1_OUTPUT_DIR = batchmod.DEFAULT_OUTPUT_DIR  # results/diagnostics/cluster_normalized_poisoning
DEFAULT_EMBEDDER = "paraphrase-MiniLM-L6-v2"
DEFAULT_STAGE2_P = 2.0
DEFAULT_BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_EPS = 1e-8
E1_COMPARISON_STRATEGIES = ("rank_aligned", "nearest_bijection")

INTERVENTION_LABEL = "CORAL_PCA"

LIMITATIONS_TEXT = """## Limitations

- **This is Step 1 of the CORAL/MMD oracle intervention plan only.** Only
  the PCA/subspace CORAL-style covariance-alignment transform is
  implemented and run here. Full-dimensional ridge-regularized CORAL and
  the MMD-minimizing gradient-based oracle optimizer (Steps 2/3 of
  `docs/CORAL_MMD_ORACLE_INTERVENTION_PLAN.md`) are **not implemented or
  run by this script**.
- **This remains an oracle embedding-space stress test.** `Z_poison` is
  transformed directly; no natural-language rewrite of any poisoned
  passage is performed or implied. It does not prove natural-language
  realizability under the frozen `paraphrase-MiniLM-L6-v2` encoder.
- **E1 is not replaced and was not rerun.** The E1 comparison section below
  is read directly from an already-existing artifact
  (`BATCH_COMPARISON_SUCCESS_CASES.csv`) written by a prior run of
  `scripts/build_batch_comparison_success_cases.py`. E1 remains the
  empirical oracle baseline; CORAL-PCA is a formal follow-up compared
  against it, not a superseding result.
- **`beta` and `alpha` run in opposite perturbation directions.** E1's
  `alpha` sweep descends from `1.0` (no perturbation) to `0.3` (maximum
  perturbation); this script's `beta` sweep **ascends** from `0.0` (no
  perturbation, identity) to `1.0` (maximum perturbation, pure CORAL
  recoloring). `first_residual_poison_beta` below is therefore the
  **smallest** beta (ascending) at which residual-poison failure first
  occurs -- the direct analogue of E1's `first_residual_poison_alpha`
  being the **largest** alpha (descending) at which it first occurs.
- **Small-sample rank deficiency.** With `n_poison = n_clean = 5` points in
  `d = 384` dimensions, the mean-centered covariance of each group has
  rank `<= 4`. The PCA/subspace transform restricts CORAL's whiten/recolor
  operation to the top `rank <= min(n_poison-1, n_clean-1)` singular
  directions of each group's own centered data (computed via SVD of the
  centered data matrix, never a naive/ridge inverse of the full `384x384`
  covariance) -- at the default rank this is an exact decomposition of the
  centered data (no signal is discarded), not an approximation.
  `defense/coral_mmd_intervention.py`'s module docstring has the full
  derivation and the reason a naive full-dimensional (unregularized or
  ridge-regularized) CORAL is invalid at this sample size.
- **Alpha/beta values causing failure may be geometrically extreme** and
  must not be interpreted as plausible natural-language passage rewrites.
- **FilterRAG and ML-FilterRAG comparisons come after** the RAGDefender
  oracle study and are not part of this run.
- **No GPT/API calls were made. No baseline retrieval was rerun. No
  baseline defense file was modified.**
"""


# --------------------------------------------------------------------------
# Query discovery (reuses build_batch_comparison_success_cases.py verbatim)
# --------------------------------------------------------------------------

def discover_tested_query_ids(records: List[Dict], records_by_id: Dict[str, Dict], qr_index: Dict,
                               dataset: str, k: int, n_injected: int,
                               exclude_query_ids: Sequence[str]) -> Tuple[List[str], Dict[str, str]]:
    identified_ids = batchmod.discover_success_case_ids(records, dataset, k, n_injected, exclude_query_ids)
    tested_ids: List[str] = []
    excluded: Dict[str, str] = {}
    for qid in identified_ids:
        ok, recovered_len, expected_k = batchmod.check_text_recoverable(qr_index, records_by_id[qid])
        if not ok:
            excluded[qid] = (
                f"text recovery mismatch: recovered {recovered_len} line(s), expected {expected_k}"
            )
            continue
        tested_ids.append(qid)
    return tested_ids, excluded


# --------------------------------------------------------------------------
# Per-query CORAL-PCA sweep
# --------------------------------------------------------------------------

def process_query(qid: str, rec: Dict, texts: List[str], model, betas: Sequence[float],
                   rank: Optional[int], eps: float, stage2_p: float, mmd_gamma: float,
                   run_dir: Path, manifest: Dict[str, List[str]]) -> Tuple[Dict, str, List[Dict]]:
    dataset, k = rec["dataset"], rec["k"]
    doc_ids = list(rec["retrieved_doc_ids"])
    is_poison = [bool(x) for x in rec["retrieved_is_poison"]]

    embeddings_t = viz.encode_texts(model, texts)
    sim_before = viz.cos_sim_from_embeddings(embeddings_t)
    z = np.asarray(embeddings_t.cpu().numpy(), dtype=np.float64)

    stage1_before = concentration_stage1(sim_before)
    stage2_before = stage2_pair_frequency(sim_before, stage1_before.n_adv_estimated, p=stage2_p)
    original_row = cnp_run.compute_metrics_snapshot(sim_before, is_poison, stage1_before, stage2_before)
    original_label = cnp_run.decision_label(
        original_row["removed_poison"], original_row["removed_clean"], original_row["N_retrieved_poison"]
    )

    z_poison, z_clean, poison_idx, clean_idx = split_poison_clean(z, is_poison)
    if len(poison_idx) < 2 or len(clean_idx) < 2:
        raise ValueError(
            f"CORAL-PCA requires >= 2 poison and >= 2 clean rows; got "
            f"N_poison={len(poison_idx)}, N_clean={len(clean_idx)} for query_id={qid!r}."
        )

    g_pp0, g_pc0, g_cc0 = slice_gram_blocks(sim_before, poison_idx, clean_idx)
    coral_before = coral_distance_from_gram(g_pp0, g_pc0, g_cc0)
    mmd_before = mmd_rbf_distance_from_gram(g_pp0, g_pc0, g_cc0, gamma=mmd_gamma)

    original_matrix_path = run_dir / "similarity_matrices" / f"{qid}_original_M.npy"
    np.save(original_matrix_path, sim_before)
    manifest["matrices"].append(str(original_matrix_path.relative_to(run_dir)))

    rows: List[Dict] = []
    for beta in betas:
        result = coral_pca_transform(z_poison, z_clean, beta, rank=rank, eps=eps)
        z_prime = recombine_poison_clean(result.z_poison_final, z_clean, poison_idx, clean_idx, k)

        embeddings_prime_t = torch.tensor(z_prime, dtype=torch.float32)
        sim_after = viz.cos_sim_from_embeddings(embeddings_prime_t)

        stage1_after = concentration_stage1(sim_after)
        stage2_after = stage2_pair_frequency(sim_after, stage1_after.n_adv_estimated, p=stage2_p)
        row = cnp_run.compute_metrics_snapshot(sim_after, is_poison, stage1_after, stage2_after)
        label = cnp_run.decision_label(row["removed_poison"], row["removed_clean"], row["N_retrieved_poison"])

        g_pp, g_pc, g_cc = slice_gram_blocks(sim_after, poison_idx, clean_idx)
        coral_after = coral_distance_from_gram(g_pp, g_pc, g_cc)
        mmd_after = mmd_rbf_distance_from_gram(g_pp, g_pc, g_cc, gamma=mmd_gamma)

        preservation = compute_preservation_metrics(z_poison, result.z_poison_final)

        matrix_path = run_dir / "similarity_matrices" / f"{qid}_beta{beta}_M.npy"
        np.save(matrix_path, sim_after)
        manifest["matrices"].append(str(matrix_path.relative_to(run_dir)))

        rows.append({
            "query_id": qid, "dataset": dataset, "intervention": INTERVENTION_LABEL,
            "beta": beta, "rank": result.rank,
            "coral_distance_before": coral_before, "coral_distance_after": coral_after,
            "coral_distance_reduction": coral_before - coral_after,
            "mmd_distance_before": mmd_before, "mmd_distance_after": mmd_after,
            "mmd_distance_reduction": mmd_before - mmd_after,
            "mean_poison_l2_displacement": preservation.mean_l2_displacement,
            "max_poison_l2_displacement": preservation.max_l2_displacement,
            "mean_poison_original_cosine": preservation.mean_original_cosine,
            "min_poison_original_cosine": preservation.min_original_cosine,
            **row,
            "decision_label": label,
        })

    return original_row, original_label, rows


def first_residual_poison_beta(sub: pd.DataFrame) -> Optional[float]:
    """First beta, **ascending** from 0.0 (least perturbation), at which
    `removed_poison < N_retrieved_poison` first occurs -- the CORAL-PCA
    analogue of `build_batch_comparison_success_cases.compute_config_summary`'s
    `first_residual_poison_alpha`, but scanning in the opposite (ascending)
    direction since beta=0 is identity here, not beta=1 (see
    `LIMITATIONS_TEXT`)."""
    sub_sorted = sub.sort_values("beta", ascending=True)
    for _, r in sub_sorted.iterrows():
        if r["removed_poison"] < r["N_retrieved_poison"]:
            return float(r["beta"])
    return None


# --------------------------------------------------------------------------
# E1 comparison (reads already-written artifacts only; never reruns E1)
# --------------------------------------------------------------------------

def load_e1_comparison(e1_output_dir: str, tested_ids: Sequence[str]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    csv_path = Path(e1_output_dir) / "BATCH_COMPARISON_SUCCESS_CASES.csv"
    if not csv_path.exists():
        return None, f"E1 comparison skipped: {csv_path} not found (E1 batch was not run/found on disk)."
    e1_df = pd.read_csv(csv_path)
    rows = []
    for qid in tested_ids:
        for strategy in E1_COMPARISON_STRATEGIES:
            sub = e1_df[(e1_df["query_id"] == qid) & (e1_df["anchor_strategy"] == strategy)]
            if sub.empty:
                continue
            summary = batchmod.compute_config_summary(qid, strategy, sub)
            rows.append(summary)
    if not rows:
        return None, f"E1 comparison skipped: no matching rows for tested query_ids in {csv_path}."
    return pd.DataFrame(rows), None


# --------------------------------------------------------------------------
# run_config.json / manifest.json
# --------------------------------------------------------------------------

def _run_git(args_list: List[str], cwd: str) -> Optional[str]:
    try:
        out = subprocess.run(["git"] + args_list, cwd=cwd, capture_output=True, text=True, timeout=10)
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
                      tested_ids: Sequence[str], excluded: Dict[str, str],
                      per_query_rank: Dict[str, int]) -> Dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "run_type": "coral_pca_oracle_intervention",
        "plan_step": "step_1_coral_pca_subspace_only",
        "dataset": dataset,
        "k": k,
        "N_injected": n_injected,
        "defense": defense,
        "query_ids_tested": list(tested_ids),
        "query_ids_excluded": excluded,
        "per_query_resolved_rank": per_query_rank,
        "diagnostics_jsonl": os.path.abspath(args.diagnostics_jsonl),
        "query_results_dir": os.path.abspath(args.query_results_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "e1_output_dir": os.path.abspath(args.e1_output_dir),
        "intervention": INTERVENTION_LABEL,
        "coral_variant": "pca_subspace",
        "betas": args.betas,
        "rank_requested": args.rank,
        "eigenvalue_floor": args.eps,
        "mmd_gamma": args.mmd_gamma,
        "intervention_level": "embedding",
        "embedder": args.embedder,
        "stage2_p": args.stage2_p,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": viz._try_version("torch"),
        "sentence_transformers_version": viz._try_version("sentence_transformers"),
        "sklearn_version": viz._try_version("sklearn"),
        "pandas_version": viz._try_version("pandas"),
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
            "e1_rerun": False,
            "full_ridge_coral_implemented": False,
            "mmd_optimizer_implemented": False,
        },
        "argv": sys.argv,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def render_report(tested_ids: Sequence[str], excluded: Dict[str, str], sweep_df: pd.DataFrame,
                   summary_rows: List[Dict], e1_df: Optional[pd.DataFrame], e1_skip_reason: Optional[str],
                   dataset: str, k: int, n_injected: int, betas: Sequence[float], run_dir: Path) -> str:
    lines = [
        "# CORAL PCA/Subspace Oracle Intervention Report (Step 1)",
        "",
        f"Run directory: `{run_dir.name}`",
        "",
        f"dataset=`{dataset}`, k={k}, N_injected={n_injected}, intervention=`{INTERVENTION_LABEL}` "
        f"(PCA/subspace variant only), betas swept = {betas}.",
        "",
        "**No GPT/API calls were made. Baseline retrieval was not rerun. No baseline defense file "
        "was modified. E1 was not rerun** (comparison section reads an already-existing E1 batch "
        "artifact). All claims below are oracle embedding-space stress-test findings, not evidence "
        "of a text-realizable attack -- see Limitations.",
        "",
        "## Tested queries",
        "",
        f"- **{len(tested_ids)}** query(ies) tested (same success-case discovery/text-recoverability "
        "gate as `scripts/build_batch_comparison_success_cases.py`).",
        f"- **{len(excluded)}** excluded.",
        "",
        "| query_id | status |",
        "|---|---|",
    ]
    for qid in tested_ids:
        lines.append(f"| `{qid}` | tested |")
    for qid, reason in excluded.items():
        lines.append(f"| `{qid}` | **excluded** -- {reason} |")
    lines.append("")

    lines += [
        "## Per-(query, beta) CORAL-PCA sweep",
        "",
        "`decision_label` is the absolute classification from "
        "`scripts/run_cluster_normalized_poisoning.py::decision_label` (reused unmodified).",
        "",
        "| query_id | beta | rank | coral_before | coral_after | mmd_before | mmd_after | "
        "top_pair (PP/PC/CC) | N_adv | removed_poison | removed_clean | residual_poison_fraction | "
        "decision_label |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in sweep_df.sort_values(["query_id", "beta"]).iterrows():
        lines.append(
            f"| `{r['query_id']}` | {r['beta']} | {r['rank']} | {r['coral_distance_before']:.4f} | "
            f"{r['coral_distance_after']:.4f} | {r['mmd_distance_before']:.4f} | "
            f"{r['mmd_distance_after']:.4f} | {r['top_pair_pp']}/{r['top_pair_pc']}/{r['top_pair_cc']} | "
            f"{r['N_adv']} | {r['removed_poison']} | {r['removed_clean']} | "
            f"{r['residual_poison_fraction']} | {r['decision_label']} |"
        )
    lines.append("")

    lines += [
        "## Perturbation / preservation metrics",
        "",
        "How far CORAL-PCA moves each query's poison embeddings away from their "
        "original (untransformed) position, and how well the original direction is "
        "preserved, as `beta` increases. Both the original and transformed poison "
        "embeddings are L2-normalized before these are computed (the same "
        "unit-sphere representation RAGDefender's cosine-similarity decision "
        "operates on), so `beta=0.0` gives **exact** identity "
        "(`mean/max_poison_l2_displacement == 0.0`, `mean/min_poison_original_cosine == 1.0`) "
        "-- see `defense/coral_mmd_intervention.py::compute_preservation_metrics`. "
        "These are the metrics that make CORAL-PCA comparable, on perturbation "
        "magnitude, to E1, later ridge-CORAL, and MMD.",
        "",
        "| query_id | beta | mean_poison_l2_displacement | max_poison_l2_displacement | "
        "mean_poison_original_cosine | min_poison_original_cosine |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in sweep_df.sort_values(["query_id", "beta"]).iterrows():
        lines.append(
            f"| `{r['query_id']}` | {r['beta']} | {r['mean_poison_l2_displacement']:.4f} | "
            f"{r['max_poison_l2_displacement']:.4f} | {r['mean_poison_original_cosine']:.4f} | "
            f"{r['min_poison_original_cosine']:.4f} |"
        )
    lines.append("")
    beta_max = max(betas)
    beta_max_rows = sweep_df[sweep_df["beta"] == beta_max]
    if not beta_max_rows.empty:
        lines.append(
            f"At `beta={beta_max}` (maximum swept perturbation) across the {len(beta_max_rows)} "
            f"tested queries: `mean_poison_l2_displacement` ranges "
            f"[{beta_max_rows['mean_poison_l2_displacement'].min():.4f}, "
            f"{beta_max_rows['mean_poison_l2_displacement'].max():.4f}], "
            f"`max_poison_l2_displacement` ranges "
            f"[{beta_max_rows['max_poison_l2_displacement'].min():.4f}, "
            f"{beta_max_rows['max_poison_l2_displacement'].max():.4f}], and "
            f"`mean_poison_original_cosine` ranges "
            f"[{beta_max_rows['mean_poison_original_cosine'].min():.4f}, "
            f"{beta_max_rows['mean_poison_original_cosine'].max():.4f}] -- see the per-query "
            "summary and the E1 comparison below for how this perturbation magnitude relates to "
            "RAGDefender's decision outcome."
        )
        lines.append("")

    lines += [
        "## Per-query summary: first beta (ascending from 0.0) causing residual-poison failure",
        "",
        "See `LIMITATIONS_TEXT` note below on why this scans **ascending** from beta=0.0 "
        "(identity), the opposite direction from E1's alpha sweep.",
        "",
        f"| query_id | first_residual_poison_beta | baseline_decision_label | "
        f"final_decision_label (beta={max(betas)}) |",
        "|---|---|---|---|",
    ]
    n_failed = 0
    for row in summary_rows:
        if row["first_residual_poison_beta"] is not None:
            n_failed += 1
        lines.append(
            f"| `{row['query_id']}` | {row['first_residual_poison_beta']} | "
            f"`{row['baseline_decision_label']}` | `{row['final_decision_label']}` |"
        )
    lines.append("")
    lines.append(
        f"**{n_failed} / {len(summary_rows)}** tested queries show residual-poison failure under "
        f"CORAL-PCA within the swept betas (`{betas}`)."
    )
    lines.append("")

    lines += [
        "## Comparison against the existing E1 batch (read-only; E1 not rerun)",
        "",
    ]
    if e1_df is None:
        lines.append(f"_{e1_skip_reason}_")
        lines.append("")
    else:
        lines += [
            f"Reference strategies: `{', '.join(E1_COMPARISON_STRATEGIES)}` (E1's own "
            "`first_residual_poison_alpha`, descending from alpha=1.0, read directly from "
            "`results/diagnostics/cluster_normalized_poisoning/BATCH_COMPARISON_SUCCESS_CASES.csv`).",
            "",
            "| query_id | E1 strategy | E1 first_residual_poison_alpha | CORAL-PCA first_residual_poison_beta |",
            "|---|---|---|---|",
        ]
        beta_by_qid = {row["query_id"]: row["first_residual_poison_beta"] for row in summary_rows}
        for _, r in e1_df.sort_values(["query_id", "strategy"]).iterrows():
            lines.append(
                f"| `{r['query_id']}` | `{r['strategy']}` | {r['first_residual_poison_alpha']} | "
                f"{beta_by_qid.get(r['query_id'])} |"
            )
        lines.append("")
        lines.append(
            "Both columns report the perturbation strength (in each method's own, opposite-direction "
            "units) at which RAGDefender's Stage-2 selection first leaves some retrieved poison "
            "un-removed; they are not on a directly convertible numeric scale (E1's `alpha` is a "
            "convex-interpolation weight toward a single clean anchor per poison point, while "
            "`beta` mixes toward a jointly whitened/recolored CORAL target) -- this table is a "
            "qualitative side-by-side, not a claim of equivalent perturbation magnitude."
        )
        lines.append("")

    lines += ["", LIMITATIONS_TEXT]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI / main
# --------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostics_jsonl", default=batchmod.DEFAULT_DIAGNOSTICS_JSONL)
    parser.add_argument("--query_results_dir", default=batchmod.DEFAULT_QUERY_RESULTS_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--e1_output_dir", default=DEFAULT_E1_OUTPUT_DIR)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n_injected", type=int, default=5)
    parser.add_argument("--exclude_query_id", action="append", default=["5a8cb288554299585d9e3726"])
    parser.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    parser.add_argument("--rank", type=int, default=None,
                         help="Subspace rank; default None resolves per-query to "
                              "min(n_poison-1, n_clean-1).")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--mmd_gamma", type=float, default=DEFAULT_MMD_GAMMA)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--stage2_p", type=float, default=DEFAULT_STAGE2_P)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Path:
    args = parse_args(argv)

    records = viz._read_jsonl(args.diagnostics_jsonl)
    records_by_id = {r["query_id"]: r for r in records}
    qr_index = viz.load_query_results_index(args.query_results_dir)

    tested_ids, excluded = discover_tested_query_ids(
        records, records_by_id, qr_index, args.dataset, args.k, args.n_injected, args.exclude_query_id
    )
    if not tested_ids:
        raise ValueError("No tested query_ids found -- check --diagnostics_jsonl/--query_results_dir.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{ts}_coral_pca_{args.dataset}_k{args.k}_N{args.n_injected}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "similarity_matrices").mkdir()
    manifest: Dict[str, List[str]] = {"config": [], "csv": [], "matrices": [], "report": []}

    model = viz.load_embedder(args.embedder)

    all_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    per_query_rank: Dict[str, int] = {}
    for qid in tested_ids:
        rec = records_by_id[qid]
        texts = viz.recover_pre_defense_texts(qr_index.get(qid))
        original_row, original_label, rows = process_query(
            qid, rec, texts, model, args.betas, args.rank, args.eps, args.stage2_p, args.mmd_gamma,
            run_dir, manifest,
        )
        per_query_rank[qid] = rows[0]["rank"] if rows else None
        all_rows.extend(rows)

        sub_df = pd.DataFrame(rows)
        summary_rows.append({
            "query_id": qid,
            "baseline_decision_label": original_label,
            "final_decision_label": sub_df.sort_values("beta", ascending=False).iloc[0]["decision_label"],
            "first_residual_poison_beta": first_residual_poison_beta(sub_df),
        })

    sweep_df = pd.DataFrame(all_rows)
    sweep_csv_path = run_dir / "CORAL_PCA_SWEEP.csv"
    sweep_df.to_csv(sweep_csv_path, index=False)
    manifest["csv"].append(str(sweep_csv_path.relative_to(run_dir)))

    e1_df, e1_skip_reason = load_e1_comparison(args.e1_output_dir, tested_ids)

    report_text = render_report(
        tested_ids, excluded, sweep_df, summary_rows, e1_df, e1_skip_reason,
        args.dataset, args.k, args.n_injected, args.betas, run_dir,
    )
    report_path = run_dir / "CORAL_PCA_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    manifest["report"].append(str(report_path.relative_to(run_dir)))

    defense = records_by_id[tested_ids[0]]["defense"]
    run_config = build_run_config(
        args, args.dataset, args.k, args.n_injected, defense, tested_ids, excluded, per_query_rank
    )
    run_config_path = run_dir / "run_config.json"
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)
    manifest["config"].append(str(run_config_path.relative_to(run_dir)))

    manifest_path = run_dir / "manifest.json"
    manifest["config"].append(str(manifest_path.relative_to(run_dir)))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Tested {len(tested_ids)} queries x {len(args.betas)} betas = {len(sweep_df)} CORAL-PCA rows.")
    print(f"Wrote CORAL-PCA oracle run to: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
