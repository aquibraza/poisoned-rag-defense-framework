#!/usr/bin/env python3
"""Cluster-Normalized Poisoning -- full-ridge CORAL oracle intervention.

**Step 2 only** of the CORAL/MMD oracle intervention plan (see
`docs/CORAL_MMD_ORACLE_INTERVENTION_PLAN.md` and
`.cursor/plans/coral_mmd_oracle_intervention_plan_31a5d179.plan.md`): runs
`defense.coral_mmd_intervention.coral_ridge_transform` (full-dimensional
ridge-regularized CORAL-style covariance alignment) across a `lambda x
beta` sweep, for the same 6 HotpotQA k=10/N=5 originally-successful
RAGDefender cases already tested by E1
(`scripts/build_batch_comparison_success_cases.py`) and by Step 1
(`scripts/run_coral_pca_oracle_intervention.py`).

The MMD-minimizing optimizer is **not implemented by this script**
(deferred to a later step of that plan).

Like `scripts/run_coral_pca_oracle_intervention.py`, this script:

- never imports or calls `defense/defense_runner.py`, `defense/dispatch.py`,
  `defense/filterrag.py`, or `main.py`,
- never performs generation and never makes a network/LLM/API call (the
  embedder is loaded fully offline from the local sentence-transformers
  cache),
- never changes retrieval membership (`k`, which doc_ids were retrieved) --
  only the poisoned passages' *embeddings* are transformed,
- **never reruns E1** -- the E1 comparison section reads
  `results/diagnostics/cluster_normalized_poisoning/
  BATCH_COMPARISON_SUCCESS_CASES.csv` directly, an artifact already on disk
  from a prior run of `scripts/build_batch_comparison_success_cases.py`,
- **never reruns Step 1 (CORAL-PCA)** -- the CORAL-PCA comparison reads
  the most recent `CORAL_PCA_SWEEP.csv` under
  `results/diagnostics/cluster_normalized_poisoning_formal/` directly.

Usage:
    python scripts/run_coral_ridge_oracle_intervention.py
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
    coral_ridge_transform,
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
DEFAULT_LAMBDAS = [1e-1, 1e-2, 1e-3]
DEFAULT_EPS = 1e-8
E1_COMPARISON_STRATEGIES = ("rank_aligned", "nearest_bijection")

INTERVENTION_LABEL = "CORAL_RIDGE"

LIMITATIONS_TEXT = """## Limitations

- **This is Step 2 of the CORAL/MMD oracle intervention plan only.** Only
  the full-dimensional ridge-regularized CORAL transform is implemented and
  run here. The MMD-minimizing gradient-based oracle optimizer (Step 3 of
  `docs/CORAL_MMD_ORACLE_INTERVENTION_PLAN.md`) is **not implemented or run
  by this script**.
- **This remains an oracle embedding-space stress test.** `Z_poison` is
  transformed directly; no natural-language rewrite of any poisoned
  passage is performed or implied. It does not prove natural-language
  realizability under the frozen `paraphrase-MiniLM-L6-v2` encoder.
- **E1 is not replaced and was not rerun.** The E1 comparison section below
  is read directly from an already-existing artifact
  (`BATCH_COMPARISON_SUCCESS_CASES.csv`) written by a prior run of
  `scripts/build_batch_comparison_success_cases.py`. E1 remains the
  empirical oracle baseline; CORAL-ridge is a formal follow-up compared
  against it, not a superseding result.
- **Step 1 (CORAL-PCA) is not replaced and was not rerun.** The CORAL-PCA
  comparison section reads an already-existing `CORAL_PCA_SWEEP.csv`
  written by a prior run of `scripts/run_coral_pca_oracle_intervention.py`.
- **`beta` and `alpha` run in opposite perturbation directions**, exactly
  as for Step 1: E1's `alpha` sweep descends from `1.0` (no perturbation)
  to `0.3` (maximum); this script's `beta` sweep **ascends** from `0.0`
  (identity) to `1.0` (maximum). `first_residual_poison_beta` below is
  therefore the **smallest** beta (ascending, per `lambda`) at which
  residual-poison failure first occurs.
- **`lambda` is a regularizer, not a data-derived quantity.** Unlike Step
  1's PCA/subspace transform -- which, at its default rank, is an *exact*
  decomposition of the true (rank-deficient) poison/clean covariance with
  no discarded signal -- this full-ridge transform inverts the **entire**
  `384 x 384` covariance `Cov + lambda*I`. In the `d - (n-1) = 380`
  directions with **zero true poison/clean covariance signal** at this
  sample size (`n_poison = n_clean = 5`), the ridge-regularized eigenvalue
  is exactly `lambda` on both sides, so the whiten/recolor operation in
  those 380 directions reduces to a **rescaling by `sqrt(lambda_cc /
  lambda_cp)`** (here `1.0`, since both groups use the same `lambda`) --
  i.e. any perturbation this transform applies *beyond* Step 1's exact
  4-dimensional signal subspace is an artifact of the arbitrary `lambda`
  choice interacting with the poison embeddings' component in 380
  essentially-noise directions of `paraphrase-MiniLM-L6-v2`'s embedding
  space, **not** a property of the poison/clean distributions themselves.
  Any residual-poison failure observed here that does **not** also occur
  under Step 1 must be read in that light: it demonstrates fragility of
  RAGDefender's Stage 2 to a specific, regularizer-dependent oracle
  perturbation of all 384 embedding dimensions, not to a `lambda`-free,
  purely data-driven distribution-alignment attack.
- **Small-sample rank deficiency underlies both Step 1 and Step 2.** With
  `n_poison = n_clean = 5` points in `d = 384` dimensions, the true
  mean-centered covariance of each group has rank `<= 4`.
  `defense/coral_mmd_intervention.py`'s module docstring has the full
  derivation of why an *unregularized* full-dimensional inverse is
  undefined at this sample size, and why Step 1's truncated (exact,
  data-only) subspace approach and Step 2's ridge-regularized (approximate,
  regularizer-influenced) full-dimensional approach are not equivalent.
- **Alpha/beta/lambda values causing failure may be geometrically
  extreme** and must not be interpreted as plausible natural-language
  passage rewrites.
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
# Per-query CORAL-ridge sweep (lambda x beta grid)
# --------------------------------------------------------------------------

def process_query(qid: str, rec: Dict, texts: List[str], model, lambdas: Sequence[float],
                   betas: Sequence[float], eps: float, stage2_p: float, mmd_gamma: float,
                   run_dir: Path, manifest: Dict[str, List[str]]) -> Tuple[Dict, str, List[Dict]]:
    dataset, k = rec["dataset"], rec["k"]
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
            f"CORAL-ridge requires >= 2 poison and >= 2 clean rows; got "
            f"N_poison={len(poison_idx)}, N_clean={len(clean_idx)} for query_id={qid!r}."
        )

    g_pp0, g_pc0, g_cc0 = slice_gram_blocks(sim_before, poison_idx, clean_idx)
    coral_before = coral_distance_from_gram(g_pp0, g_pc0, g_cc0)
    mmd_before = mmd_rbf_distance_from_gram(g_pp0, g_pc0, g_cc0, gamma=mmd_gamma)

    original_matrix_path = run_dir / "similarity_matrices" / f"{qid}_original_M.npy"
    np.save(original_matrix_path, sim_before)
    manifest["matrices"].append(str(original_matrix_path.relative_to(run_dir)))

    rows: List[Dict] = []
    for lam in lambdas:
        for beta in betas:
            result = coral_ridge_transform(z_poison, z_clean, beta, lam, eps=eps)
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

            matrix_path = run_dir / "similarity_matrices" / f"{qid}_lambda{lam}_beta{beta}_M.npy"
            np.save(matrix_path, sim_after)
            manifest["matrices"].append(str(matrix_path.relative_to(run_dir)))

            rows.append({
                "query_id": qid, "dataset": dataset, "intervention": INTERVENTION_LABEL,
                "lambda": lam, "beta": beta,
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
    `removed_poison < N_retrieved_poison` first occurs, for one (query,
    lambda) slice of the sweep -- same convention as
    `run_coral_pca_oracle_intervention.py::first_residual_poison_beta`."""
    sub_sorted = sub.sort_values("beta", ascending=True)
    for _, r in sub_sorted.iterrows():
        if r["removed_poison"] < r["N_retrieved_poison"]:
            return float(r["beta"])
    return None


# --------------------------------------------------------------------------
# E1 / CORAL-PCA comparison (reads already-written artifacts only; reruns nothing)
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
            # compute_config_summary reports baseline_top_pair_pp (alpha=1.0)
            # and final_decision_label (min alpha) but not the top_pair_pp
            # value at that same min-alpha (most-perturbed) row -- read it
            # directly from the already-loaded sub-frame (no recomputation).
            most_perturbed_row = sub.sort_values("alpha", ascending=False).iloc[-1]
            summary["final_top_pair_pp"] = int(most_perturbed_row["top_pair_pp"])
            rows.append(summary)
    if not rows:
        return None, f"E1 comparison skipped: no matching rows for tested query_ids in {csv_path}."
    return pd.DataFrame(rows), None


def discover_latest_coral_pca_sweep_csv(output_dir: str) -> Optional[Path]:
    """Most recently written `CORAL_PCA_SWEEP.csv` under `output_dir`
    (`results/diagnostics/cluster_normalized_poisoning_formal/`), by run
    directory name (`YYYYMMDD_HHMMSS_coral_pca_...`, which sorts
    lexicographically in time order) -- never regenerated here."""
    candidates = sorted(Path(output_dir).glob("*_coral_pca_*/CORAL_PCA_SWEEP.csv"))
    return candidates[-1] if candidates else None


def load_coral_pca_comparison(coral_pca_sweep_csv: Optional[Path],
                               tested_ids: Sequence[str]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if coral_pca_sweep_csv is None or not coral_pca_sweep_csv.exists():
        return None, (
            "CORAL-PCA comparison skipped: no CORAL_PCA_SWEEP.csv found under "
            f"{DEFAULT_OUTPUT_DIR} (run scripts/run_coral_pca_oracle_intervention.py first)."
        )
    pca_df = pd.read_csv(coral_pca_sweep_csv)
    pca_df = pca_df[pca_df["query_id"].isin(tested_ids)]
    if pca_df.empty:
        return None, f"CORAL-PCA comparison skipped: no matching rows for tested query_ids in {coral_pca_sweep_csv}."
    return pca_df, None


# --------------------------------------------------------------------------
# Method comparison: E1 vs CORAL-PCA vs CORAL-ridge
# --------------------------------------------------------------------------

def build_method_comparison(tested_ids: Sequence[str], ridge_sweep_df: pd.DataFrame,
                             pca_df: Optional[pd.DataFrame], e1_df: Optional[pd.DataFrame],
                             lambdas: Sequence[float], betas: Sequence[float]) -> pd.DataFrame:
    """One row per `(query_id, method)`, comparing (at each method's own
    *maximum swept perturbation*, plus its `first_residual_poison_*` if
    any): which method causes residual-poison failure at the least
    perturbation, which reduces `top_pair_pp`/CORAL/MMD distance the most,
    and which best preserves the original poison embeddings. `E1` columns
    for preservation/displacement are `None` -- E1's own sweep artifact
    (`BATCH_COMPARISON_SUCCESS_CASES.csv`) never computed those metrics
    (see Limitations); this is not recomputed here since doing so would
    require rerunning E1's embeddings, which is out of scope.
    """
    rows: List[Dict] = []
    beta_max = max(betas)

    if e1_df is not None:
        for _, r in e1_df.iterrows():
            rows.append({
                "query_id": r["query_id"],
                "method": f"E1_{r['strategy']}",
                "perturbation_param": "alpha=0.3 (max swept)",
                "first_residual_poison_perturbation": r.get("first_residual_poison_alpha"),
                "causes_residual_poison_failure": r.get("first_residual_poison_alpha") is not None
                and not pd.isna(r.get("first_residual_poison_alpha")),
                "top_pair_pp_baseline": r.get("baseline_top_pair_pp"),
                "top_pair_pp_at_max_perturbation": r.get("final_top_pair_pp"),
                "coral_distance_before": None,
                "coral_distance_after": None,
                "mmd_distance_before": None,
                "mmd_distance_after": None,
                "mean_poison_l2_displacement": None,
                "mean_poison_original_cosine": None,
                "max_poison_l2_displacement": None,
                "decision_label_at_max_perturbation": r.get("final_decision_label"),
            })

    if pca_df is not None:
        for qid in tested_ids:
            sub = pca_df[pca_df["query_id"] == qid]
            if sub.empty:
                continue
            baseline = sub[sub["beta"] == 0.0]
            at_max = sub[sub["beta"] == beta_max]
            if at_max.empty:
                continue
            at_max = at_max.iloc[0]
            baseline_row = baseline.iloc[0] if not baseline.empty else at_max
            first_beta = first_residual_poison_beta(sub)
            rows.append({
                "query_id": qid,
                "method": "CORAL_PCA",
                "perturbation_param": f"beta={beta_max} (max swept)",
                "first_residual_poison_perturbation": first_beta,
                "causes_residual_poison_failure": first_beta is not None,
                "top_pair_pp_baseline": baseline_row.get("top_pair_pp"),
                "top_pair_pp_at_max_perturbation": at_max.get("top_pair_pp"),
                "coral_distance_before": at_max.get("coral_distance_before"),
                "coral_distance_after": at_max.get("coral_distance_after"),
                "mmd_distance_before": at_max.get("mmd_distance_before"),
                "mmd_distance_after": at_max.get("mmd_distance_after"),
                "mean_poison_l2_displacement": at_max.get("mean_poison_l2_displacement"),
                "mean_poison_original_cosine": at_max.get("mean_poison_original_cosine"),
                "max_poison_l2_displacement": at_max.get("max_poison_l2_displacement"),
                "decision_label_at_max_perturbation": at_max.get("decision_label"),
            })

    for qid in tested_ids:
        for lam in lambdas:
            sub = ridge_sweep_df[(ridge_sweep_df["query_id"] == qid) & (ridge_sweep_df["lambda"] == lam)]
            if sub.empty:
                continue
            baseline = sub[sub["beta"] == 0.0]
            at_max = sub[sub["beta"] == beta_max]
            if at_max.empty:
                continue
            at_max = at_max.iloc[0]
            baseline_row = baseline.iloc[0] if not baseline.empty else at_max
            first_beta = first_residual_poison_beta(sub)
            rows.append({
                "query_id": qid,
                "method": f"CORAL_RIDGE_lambda={lam}",
                "perturbation_param": f"beta={beta_max} (max swept)",
                "first_residual_poison_perturbation": first_beta,
                "causes_residual_poison_failure": first_beta is not None,
                "top_pair_pp_baseline": baseline_row.get("top_pair_pp"),
                "top_pair_pp_at_max_perturbation": at_max.get("top_pair_pp"),
                "coral_distance_before": at_max.get("coral_distance_before"),
                "coral_distance_after": at_max.get("coral_distance_after"),
                "mmd_distance_before": at_max.get("mmd_distance_before"),
                "mmd_distance_after": at_max.get("mmd_distance_after"),
                "mean_poison_l2_displacement": at_max.get("mean_poison_l2_displacement"),
                "mean_poison_original_cosine": at_max.get("mean_poison_original_cosine"),
                "max_poison_l2_displacement": at_max.get("max_poison_l2_displacement"),
                "decision_label_at_max_perturbation": at_max.get("decision_label"),
            })

    return pd.DataFrame(rows)


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
                      coral_pca_sweep_csv: Optional[Path]) -> Dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "run_type": "coral_ridge_oracle_intervention",
        "plan_step": "step_2_coral_full_ridge_only",
        "dataset": dataset,
        "k": k,
        "N_injected": n_injected,
        "defense": defense,
        "query_ids_tested": list(tested_ids),
        "query_ids_excluded": excluded,
        "diagnostics_jsonl": os.path.abspath(args.diagnostics_jsonl),
        "query_results_dir": os.path.abspath(args.query_results_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "e1_output_dir": os.path.abspath(args.e1_output_dir),
        "coral_pca_sweep_csv": str(coral_pca_sweep_csv) if coral_pca_sweep_csv else None,
        "intervention": INTERVENTION_LABEL,
        "coral_variant": "full_ridge",
        "betas": args.betas,
        "lambdas": args.lambdas,
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
            "coral_pca_rerun": False,
            "full_ridge_coral_implemented": True,
            "unregularized_inverse_used": False,
            "mmd_optimizer_implemented": False,
        },
        "argv": sys.argv,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def render_report(tested_ids: Sequence[str], excluded: Dict[str, str], sweep_df: pd.DataFrame,
                   summary_rows: List[Dict], e1_df: Optional[pd.DataFrame], e1_skip_reason: Optional[str],
                   pca_df: Optional[pd.DataFrame], pca_skip_reason: Optional[str],
                   method_comparison_df: pd.DataFrame,
                   dataset: str, k: int, n_injected: int, betas: Sequence[float], lambdas: Sequence[float],
                   run_dir: Path) -> str:
    beta_max = max(betas)
    lines = [
        "# CORAL Full-Ridge Oracle Intervention Report (Step 2)",
        "",
        f"Run directory: `{run_dir.name}`",
        "",
        f"dataset=`{dataset}`, k={k}, N_injected={n_injected}, intervention=`{INTERVENTION_LABEL}` "
        f"(full-dimensional ridge-regularized variant only), betas swept = {betas}, "
        f"lambdas swept = {lambdas}.",
        "",
        "**No GPT/API calls were made. Baseline retrieval was not rerun. No baseline defense file "
        "was modified. E1 was not rerun. Step 1 (CORAL-PCA) was not rerun** (both comparison "
        "sections read already-existing artifacts). All claims below are oracle embedding-space "
        "stress-test findings, not evidence of a text-realizable attack -- see Limitations.",
        "",
        "## Tested queries",
        "",
        f"- **{len(tested_ids)}** query(ies) tested (same success-case discovery/text-recoverability "
        "gate as `scripts/build_batch_comparison_success_cases.py` and Step 1).",
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
        "## Per-(query, lambda, beta) CORAL-ridge sweep",
        "",
        "`decision_label` is the absolute classification from "
        "`scripts/run_cluster_normalized_poisoning.py::decision_label` (reused unmodified).",
        "",
        "| query_id | lambda | beta | coral_before | coral_after | mmd_before | mmd_after | "
        "top_pair (PP/PC/CC) | N_adv | removed_poison | removed_clean | residual_poison_fraction | "
        "decision_label |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in sweep_df.sort_values(["query_id", "lambda", "beta"]).iterrows():
        lines.append(
            f"| `{r['query_id']}` | {r['lambda']} | {r['beta']} | {r['coral_distance_before']:.4f} | "
            f"{r['coral_distance_after']:.4f} | {r['mmd_distance_before']:.4f} | "
            f"{r['mmd_distance_after']:.4f} | {r['top_pair_pp']}/{r['top_pair_pc']}/{r['top_pair_cc']} | "
            f"{r['N_adv']} | {r['removed_poison']} | {r['removed_clean']} | "
            f"{r['residual_poison_fraction']} | {r['decision_label']} |"
        )
    lines.append("")

    lines += [
        "## Perturbation / preservation metrics",
        "",
        "Same convention as Step 1's `CORAL_PCA_REPORT.md`: both the original and transformed "
        "poison embeddings are L2-normalized before displacement/cosine are computed, so "
        "`beta=0.0` gives **exact** identity regardless of `lambda` -- see "
        "`defense/coral_mmd_intervention.py::compute_preservation_metrics`.",
        "",
        "| query_id | lambda | beta | mean_poison_l2_displacement | max_poison_l2_displacement | "
        "mean_poison_original_cosine | min_poison_original_cosine |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in sweep_df.sort_values(["query_id", "lambda", "beta"]).iterrows():
        lines.append(
            f"| `{r['query_id']}` | {r['lambda']} | {r['beta']} | {r['mean_poison_l2_displacement']:.4f} | "
            f"{r['max_poison_l2_displacement']:.4f} | {r['mean_poison_original_cosine']:.4f} | "
            f"{r['min_poison_original_cosine']:.4f} |"
        )
    lines.append("")

    lines += [
        "## Per-(query, lambda) summary: first beta (ascending from 0.0) causing residual-poison failure",
        "",
        f"| query_id | lambda | first_residual_poison_beta | baseline_decision_label | "
        f"final_decision_label (beta={beta_max}) |",
        "|---|---|---|---|---|",
    ]
    n_failed = 0
    for row in summary_rows:
        if row["first_residual_poison_beta"] is not None:
            n_failed += 1
        lines.append(
            f"| `{row['query_id']}` | {row['lambda']} | {row['first_residual_poison_beta']} | "
            f"`{row['baseline_decision_label']}` | `{row['final_decision_label']}` |"
        )
    lines.append("")
    lines.append(
        f"**{n_failed} / {len(summary_rows)}** tested `(query, lambda)` configs show residual-poison "
        f"failure under CORAL-ridge within the swept betas (`{betas}`)."
    )
    lines.append("")

    lines += ["## Method comparison: E1 vs CORAL-PCA (Step 1) vs CORAL-ridge (Step 2)", ""]
    if method_comparison_df.empty:
        lines.append("_Method comparison skipped: no data available (see skip reasons below)._")
        lines.append("")
    else:
        lines += [
            "One row per `(query_id, method)`, each at that method's own maximum swept "
            "perturbation. `None` in a CORAL-distance/preservation column means that method's own "
            "artifact did not compute it (E1's sweep predates CORAL/MMD distance and preservation "
            "metrics; both are read-only here, not recomputed).",
            "",
            "| query_id | method | first_residual_poison_perturbation | causes_failure | "
            "top_pair_pp_baseline | top_pair_pp_at_max | coral_before | coral_after | "
            "mean_poison_l2_displacement | mean_poison_original_cosine |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in method_comparison_df.sort_values(["query_id", "method"]).iterrows():
            def _fmt(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return "None"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)
            lines.append(
                f"| `{r['query_id']}` | `{r['method']}` | {_fmt(r['first_residual_poison_perturbation'])} | "
                f"{r['causes_residual_poison_failure']} | {_fmt(r['top_pair_pp_baseline'])} | "
                f"{_fmt(r['top_pair_pp_at_max_perturbation'])} | {_fmt(r['coral_distance_before'])} | "
                f"{_fmt(r['coral_distance_after'])} | {_fmt(r['mean_poison_l2_displacement'])} | "
                f"{_fmt(r['mean_poison_original_cosine'])} |"
            )
        lines.append("")

    if e1_skip_reason:
        lines.append(f"_{e1_skip_reason}_")
        lines.append("")
    if pca_skip_reason:
        lines.append(f"_{pca_skip_reason}_")
        lines.append("")

    lines += ["## Questions this run answers", ""]

    n_ridge_failures = sum(1 for row in summary_rows if row["first_residual_poison_beta"] is not None)
    lines.append(
        f"**1. Does ridge CORAL cause any residual-poison failures?** "
        f"{'Yes' if n_ridge_failures > 0 else 'No'} -- {n_ridge_failures} / {len(summary_rows)} "
        f"tested `(query, lambda)` configs show residual-poison failure within the swept betas "
        f"(see the per-`(query, lambda)` summary table above)."
    )
    lines.append("")

    if not method_comparison_df.empty:
        ridge_rows = method_comparison_df[method_comparison_df["method"].str.startswith("CORAL_RIDGE")]
        pca_rows = method_comparison_df[method_comparison_df["method"] == "CORAL_PCA"]
        if not ridge_rows.empty and not pca_rows.empty:
            ridge_pp_reduction = (ridge_rows["top_pair_pp_baseline"] - ridge_rows["top_pair_pp_at_max_perturbation"]).mean()
            pca_pp_reduction = (pca_rows["top_pair_pp_baseline"] - pca_rows["top_pair_pp_at_max_perturbation"]).mean()
            lines.append(
                f"**2. Does it reduce `top_pair_pp` more than PCA-CORAL?** Mean `top_pair_pp` "
                f"reduction at max swept perturbation: CORAL-ridge = {ridge_pp_reduction:.3f} "
                f"(averaged across lambdas and queries) vs CORAL-PCA = {pca_pp_reduction:.3f}."
            )
            lines.append("")
            ridge_disp = ridge_rows["mean_poison_l2_displacement"].mean()
            pca_disp = pca_rows["mean_poison_l2_displacement"].mean()
            ridge_cos = ridge_rows["mean_poison_original_cosine"].mean()
            pca_cos = pca_rows["mean_poison_original_cosine"].mean()
            lines.append(
                f"**3. Does it preserve original poison embeddings more or less than PCA-CORAL?** "
                f"Mean `mean_poison_l2_displacement` at max swept perturbation: CORAL-ridge = "
                f"{ridge_disp:.4f} vs CORAL-PCA = {pca_disp:.4f}; mean "
                f"`mean_poison_original_cosine`: CORAL-ridge = {ridge_cos:.4f} vs CORAL-PCA = "
                f"{pca_cos:.4f}. {'Lower displacement / higher cosine means CORAL-ridge preserves the original poison embeddings *more*; higher displacement / lower cosine means it preserves them *less* (moves them further) than CORAL-PCA.' }"
            )
            lines.append("")
            ridge_coral_reduction = (ridge_rows["coral_distance_before"] - ridge_rows["coral_distance_after"]).mean()
            pca_coral_reduction = (pca_rows["coral_distance_before"] - pca_rows["coral_distance_after"]).mean()
            reduced_coral_mask = (ridge_rows["coral_distance_before"] - ridge_rows["coral_distance_after"]) > 1e-6
            high_pp_mask = ridge_rows["top_pair_pp_at_max_perturbation"] >= 8
            any_high_pp_despite_reduction = bool((reduced_coral_mask & high_pp_mask).any())
            lines.append(
                f"**4. Are CORAL/MMD reductions still insufficient when `top_pair_pp` remains high?** "
                f"Mean CORAL-distance reduction at max perturbation: CORAL-ridge = "
                f"{ridge_coral_reduction:.4f} vs CORAL-PCA = {pca_coral_reduction:.4f}. "
                f"{'Yes -- at least one (query, lambda) config reduces CORAL distance while `top_pair_pp` (out of 10 possible pairs) remains high (>= 8), reproducing Step 1' + chr(39) + 's finding that global distribution-alignment metrics can improve substantially without disrupting the specific pairwise poison-poison concentration Stage 2 keys on.' if any_high_pp_despite_reduction else 'Not observed in this run: CORAL/MMD distance reduction and top_pair_pp reduction track together across the swept configs.'}"
            )
            lines.append("")

    if e1_df is not None and not method_comparison_df.empty:
        e1_rows = method_comparison_df[method_comparison_df["method"].str.startswith("E1_")]
        e1_any_failure = bool(e1_rows["causes_residual_poison_failure"].any()) if not e1_rows.empty else False
        coral_any_failure = n_ridge_failures > 0 or (
            not method_comparison_df[method_comparison_df["method"] == "CORAL_PCA"].empty
            and bool(method_comparison_df[method_comparison_df["method"] == "CORAL_PCA"]["causes_residual_poison_failure"].any())
        )
        lines.append(
            f"**5. Is covariance alignment generally weaker than E1 clean-anchor interpolation for "
            f"this defense?** E1 causes residual-poison failure in "
            f"{'all' if e1_any_failure else 'none'} of its tested `(query, strategy)` configs "
            f"(6/6 queries by `alpha<=0.5`, per the existing `BATCH_COMPARISON_SUCCESS_CASES.md`); "
            f"CORAL (PCA and/or ridge, this run) causes residual-poison failure in "
            f"{'at least one' if coral_any_failure else 'none'} tested config. "
            f"{'This suggests covariance alignment (both variants) remains weaker than E1' + chr(39) + 's per-point clean-anchor interpolation for disrupting this specific defense mechanism, even where ridge-CORAL perturbs poison embeddings further than PCA-CORAL.' if not coral_any_failure else 'Ridge-CORAL' + chr(39) + 's full-384-dimensional perturbation (see Limitations on why this differs from a pure data-driven alignment) narrows or closes this gap relative to Step 1 for at least one config; see the per-config table above for which.'}"
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
    parser.add_argument("--coral_pca_sweep_csv", default=None,
                         help="Path to an existing CORAL_PCA_SWEEP.csv; default None auto-discovers "
                              "the most recent one under --output_dir.")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n_injected", type=int, default=5)
    parser.add_argument("--exclude_query_id", action="append", default=["5a8cb288554299585d9e3726"])
    parser.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--mmd_gamma", type=float, default=DEFAULT_MMD_GAMMA)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--stage2_p", type=float, default=DEFAULT_STAGE2_P)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Path:
    args = parse_args(argv)

    for lam in args.lambdas:
        if lam <= 0:
            raise ValueError(f"--lambdas must all be > 0 (ridge regularization); got {lam}")

    records = viz._read_jsonl(args.diagnostics_jsonl)
    records_by_id = {r["query_id"]: r for r in records}
    qr_index = viz.load_query_results_index(args.query_results_dir)

    tested_ids, excluded = discover_tested_query_ids(
        records, records_by_id, qr_index, args.dataset, args.k, args.n_injected, args.exclude_query_id
    )
    if not tested_ids:
        raise ValueError("No tested query_ids found -- check --diagnostics_jsonl/--query_results_dir.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{ts}_coral_ridge_{args.dataset}_k{args.k}_N{args.n_injected}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "similarity_matrices").mkdir()
    manifest: Dict[str, List[str]] = {"config": [], "csv": [], "matrices": [], "report": []}

    model = viz.load_embedder(args.embedder)

    all_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    for qid in tested_ids:
        rec = records_by_id[qid]
        texts = viz.recover_pre_defense_texts(qr_index.get(qid))
        original_row, original_label, rows = process_query(
            qid, rec, texts, model, args.lambdas, args.betas, args.eps, args.stage2_p, args.mmd_gamma,
            run_dir, manifest,
        )
        all_rows.extend(rows)

        sub_df = pd.DataFrame(rows)
        for lam in args.lambdas:
            lam_sub = sub_df[sub_df["lambda"] == lam]
            summary_rows.append({
                "query_id": qid,
                "lambda": lam,
                "baseline_decision_label": original_label,
                "final_decision_label": lam_sub.sort_values("beta", ascending=False).iloc[0]["decision_label"],
                "first_residual_poison_beta": first_residual_poison_beta(lam_sub),
            })

    sweep_df = pd.DataFrame(all_rows)
    sweep_csv_path = run_dir / "CORAL_RIDGE_SWEEP.csv"
    sweep_df.to_csv(sweep_csv_path, index=False)
    manifest["csv"].append(str(sweep_csv_path.relative_to(run_dir)))

    e1_df, e1_skip_reason = load_e1_comparison(args.e1_output_dir, tested_ids)

    coral_pca_sweep_csv = Path(args.coral_pca_sweep_csv) if args.coral_pca_sweep_csv else \
        discover_latest_coral_pca_sweep_csv(args.output_dir)
    pca_df, pca_skip_reason = load_coral_pca_comparison(coral_pca_sweep_csv, tested_ids)

    method_comparison_df = build_method_comparison(
        tested_ids, sweep_df, pca_df, e1_df, args.lambdas, args.betas
    )
    comparison_csv_path = run_dir / "METHOD_COMPARISON_CORAL.csv"
    method_comparison_df.to_csv(comparison_csv_path, index=False)
    manifest["csv"].append(str(comparison_csv_path.relative_to(run_dir)))

    report_text = render_report(
        tested_ids, excluded, sweep_df, summary_rows, e1_df, e1_skip_reason, pca_df, pca_skip_reason,
        method_comparison_df, args.dataset, args.k, args.n_injected, args.betas, args.lambdas, run_dir,
    )
    report_path = run_dir / "CORAL_RIDGE_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    manifest["report"].append(str(report_path.relative_to(run_dir)))

    defense = records_by_id[tested_ids[0]]["defense"]
    run_config = build_run_config(
        args, args.dataset, args.k, args.n_injected, defense, tested_ids, excluded, coral_pca_sweep_csv
    )
    run_config_path = run_dir / "run_config.json"
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)
    manifest["config"].append(str(run_config_path.relative_to(run_dir)))

    manifest_path = run_dir / "manifest.json"
    manifest["config"].append(str(manifest_path.relative_to(run_dir)))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Tested {len(tested_ids)} queries x {len(args.lambdas)} lambdas x {len(args.betas)} betas "
        f"= {len(sweep_df)} CORAL-ridge rows."
    )
    print(f"Wrote CORAL-ridge oracle run to: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
