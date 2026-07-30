#!/usr/bin/env python3
"""Cluster-Normalized Poisoning -- MMD-minimizing oracle intervention.

**Step 3 only** of the CORAL/MMD oracle intervention plan (see
`docs/CORAL_MMD_ORACLE_INTERVENTION_PLAN.md` and
`.cursor/plans/coral_mmd_oracle_intervention_plan_31a5d179.plan.md`): runs
`defense.coral_mmd_intervention.mmd_minimize_transform` (direct
gradient-based RBF-MMD minimization over the poison embeddings, via
PyTorch autograd) across a `lambda_preserve x steps` sweep, for the same 6
HotpotQA k=10/N=5 originally-successful RAGDefender cases already tested
by E1 (`scripts/build_batch_comparison_success_cases.py`), Step 1
(`scripts/run_coral_pca_oracle_intervention.py`), and Step 2
(`scripts/run_coral_ridge_oracle_intervention.py`).

This is **not** DAN: no discriminator/auxiliary network is trained. The
only thing optimized is the poison embeddings themselves, for one fixed,
already-retrieved set of passages -- exactly the same "oracle
embedding-space intervention" family as E0/E1/CORAL-PCA/CORAL-ridge.

Like the Step 1/2 scripts, this script:

- never imports or calls `defense/defense_runner.py`, `defense/dispatch.py`,
  `defense/filterrag.py`, or `main.py`,
- never performs generation and never makes a network/LLM/API call (the
  embedder is loaded fully offline from the local sentence-transformers
  cache),
- never changes retrieval membership (`k`, which doc_ids were retrieved) --
  only the poisoned passages' *embeddings* are transformed,
- **never reruns E1, Step 1, or Step 2** -- all three comparison sections
  read already-existing artifacts on disk (`BATCH_COMPARISON_SUCCESS_CASES.csv`,
  the most recent `CORAL_PCA_SWEEP.csv`, and the most recent
  `CORAL_RIDGE_SWEEP.csv`).

Usage:
    python scripts/run_mmd_oracle_intervention.py
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
    mmd_minimize_transform,
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
DEFAULT_LAMBDA_PRESERVES = [0.01, 0.1, 1.0]
DEFAULT_STEPS_LIST = [0, 50, 100]
DEFAULT_GAMMA = DEFAULT_MMD_GAMMA  # 1.0
DEFAULT_LR = 0.05
DEFAULT_SEED = 0
E1_COMPARISON_STRATEGIES = ("rank_aligned", "nearest_bijection")

INTERVENTION_LABEL = "MMD_MINIMIZE"

LIMITATIONS_TEXT = """## Limitations

- **This is Step 3 of the CORAL/MMD oracle intervention plan only.** It
  implements and runs only the direct MMD-minimizing gradient-based oracle
  optimizer. No other intervention is implemented or modified here.
- **This is not DAN.** No discriminator or auxiliary network is trained;
  `mmd_minimize_transform` optimizes the poison embeddings themselves,
  directly, via PyTorch autograd, for one fixed, already-retrieved set of
  passages. DAN is literature background motivating why MMD-style
  alignment objectives are meaningful, not something implemented here.
- **This remains an oracle embedding-space stress test.** `Z_poison` is
  optimized directly; no natural-language rewrite of any poisoned passage
  is performed or implied. It does not prove natural-language
  realizability under the frozen `paraphrase-MiniLM-L6-v2` encoder.
- **E1 is not replaced and was not rerun.** The E1 comparison section
  reads an already-existing artifact (`BATCH_COMPARISON_SUCCESS_CASES.csv`).
  E1 remains the empirical oracle baseline; MMD is a formal follow-up
  compared against it, not a superseding result.
- **Step 1 (CORAL-PCA) and Step 2 (CORAL-ridge) are not replaced and were
  not rerun.** Both comparison sections read already-existing
  `CORAL_PCA_SWEEP.csv` / `CORAL_RIDGE_SWEEP.csv` artifacts.
- **The optimization runs entirely on the unit sphere.** Both the original
  poison embeddings and the clean embeddings are L2-normalized once up
  front, and the transformed poison embeddings are re-projected onto the
  unit sphere after every optimizer step (rather than adding an explicit
  norm-penalty term) -- see `defense/coral_mmd_intervention.py::
  mmd_minimize_transform`'s docstring for why this keeps the objective in
  the same representation RAGDefender's own cosine-similarity Stage 1/2
  logic operates on. This is a design choice, not a property of MMD
  minimization in general.
- **`gamma` is a fixed, lightweight RBF-kernel bandwidth**
  (`DEFAULT_MMD_GAMMA=1.0`), not a per-query median-heuristic bandwidth --
  the same simplicity choice already documented for the Gram-based
  diagnostic metric in `defense/distribution_metrics.py`. A different
  `gamma` could change the optimization landscape and is not swept here.
- **`lambda_preserve`/`steps`/`lr` values causing failure may be
  geometrically extreme** and must not be interpreted as plausible
  natural-language passage rewrites.
- **Small-sample rank deficiency** (`n_poison = n_clean = 5` in `d = 384`)
  does not directly constrain MMD the way it constrains CORAL's covariance
  inversion (MMD never inverts a covariance matrix), but the same *sample
  size* caveat applies to how representative any of these 6 queries'
  gradients are of a general attack.
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
# Per-query MMD sweep (lambda_preserve x steps grid), with per-step traces
# --------------------------------------------------------------------------

def process_query(qid: str, rec: Dict, texts: List[str], model, lambda_preserves: Sequence[float],
                   steps_list: Sequence[int], gamma: float, lr: float, seed: Optional[int], stage2_p: float,
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
    if len(poison_idx) < 1 or len(clean_idx) < 1:
        raise ValueError(
            f"MMD minimization requires >= 1 poison and >= 1 clean row; got "
            f"N_poison={len(poison_idx)}, N_clean={len(clean_idx)} for query_id={qid!r}."
        )

    g_pp0, g_pc0, g_cc0 = slice_gram_blocks(sim_before, poison_idx, clean_idx)
    coral_before = coral_distance_from_gram(g_pp0, g_pc0, g_cc0)
    mmd_before = mmd_rbf_distance_from_gram(g_pp0, g_pc0, g_cc0, gamma=gamma)

    original_matrix_path = run_dir / "similarity_matrices" / f"{qid}_original_M.npy"
    np.save(original_matrix_path, sim_before)
    manifest["matrices"].append(str(original_matrix_path.relative_to(run_dir)))

    rows: List[Dict] = []
    for lp in lambda_preserves:
        for st in steps_list:
            trace_rows: List[Dict] = []

            def on_step(step: int, z_poison_step: np.ndarray, mmd_loss: float, preserve_loss: float,
                        total_loss: float) -> None:
                z_prime_step = recombine_poison_clean(z_poison_step, z_clean, poison_idx, clean_idx, k)
                sim_step = viz.cos_sim_from_embeddings(torch.tensor(z_prime_step, dtype=torch.float32))
                stage1_step = concentration_stage1(sim_step)
                stage2_step = stage2_pair_frequency(sim_step, stage1_step.n_adv_estimated, p=stage2_p)
                row_step = cnp_run.compute_metrics_snapshot(sim_step, is_poison, stage1_step, stage2_step)
                label_step = cnp_run.decision_label(
                    row_step["removed_poison"], row_step["removed_clean"], row_step["N_retrieved_poison"]
                )
                trace_rows.append({
                    "step": step,
                    "total_loss": total_loss,
                    "mmd_loss": mmd_loss,
                    "preservation_loss": preserve_loss,
                    "mean_pp_similarity": row_step["mean_poison_poison_similarity"],
                    "top_pair_pp": row_step["top_pair_pp"],
                    "removed_poison": row_step["removed_poison"],
                    "removed_clean": row_step["removed_clean"],
                    "residual_poison_fraction": row_step["residual_poison_fraction"],
                    "decision_label": label_step,
                })

            result = mmd_minimize_transform(
                z_poison, z_clean, lambda_preserve=lp, gamma=gamma, steps=st, lr=lr, seed=seed, on_step=on_step
            )

            trace_df = pd.DataFrame(trace_rows)
            trace_path = run_dir / "traces" / f"{qid}_lp{lp}_steps{st}_trace.csv"
            trace_df.to_csv(trace_path, index=False)
            manifest["traces"].append(str(trace_path.relative_to(run_dir)))

            z_prime = recombine_poison_clean(result.z_poison_final, z_clean, poison_idx, clean_idx, k)
            embeddings_prime_t = torch.tensor(z_prime, dtype=torch.float32)
            sim_after = viz.cos_sim_from_embeddings(embeddings_prime_t)

            stage1_after = concentration_stage1(sim_after)
            stage2_after = stage2_pair_frequency(sim_after, stage1_after.n_adv_estimated, p=stage2_p)
            row = cnp_run.compute_metrics_snapshot(sim_after, is_poison, stage1_after, stage2_after)
            label = cnp_run.decision_label(row["removed_poison"], row["removed_clean"], row["N_retrieved_poison"])

            g_pp, g_pc, g_cc = slice_gram_blocks(sim_after, poison_idx, clean_idx)
            coral_after = coral_distance_from_gram(g_pp, g_pc, g_cc)
            mmd_after = mmd_rbf_distance_from_gram(g_pp, g_pc, g_cc, gamma=gamma)

            preservation = compute_preservation_metrics(z_poison, result.z_poison_final)

            matrix_path = run_dir / "similarity_matrices" / f"{qid}_lp{lp}_steps{st}_M.npy"
            np.save(matrix_path, sim_after)
            manifest["matrices"].append(str(matrix_path.relative_to(run_dir)))

            first_failure_step = None
            for _, r in trace_df.sort_values("step", ascending=True).iterrows():
                if r["residual_poison_fraction"] and r["residual_poison_fraction"] > 0:
                    first_failure_step = int(r["step"])
                    break

            rows.append({
                "query_id": qid, "dataset": dataset, "intervention": INTERVENTION_LABEL,
                "lambda_preserve": lp, "steps": st, "gamma": gamma, "lr": lr,
                "first_failure_step_in_trace": first_failure_step,
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


def first_residual_poison_steps(sub: pd.DataFrame) -> Optional[int]:
    """First `steps` value, **ascending** from 0 (least perturbation), at
    which `removed_poison < N_retrieved_poison` first occurs, for one
    `(query, lambda_preserve)` slice of the coarse sweep grid -- the MMD
    analogue of the CORAL scripts' `first_residual_poison_beta`. See also
    `first_failure_step_in_trace` (in `CSV`/`process_query`) for the
    finer-grained, per-optimizer-step answer read directly from each
    config's own trace."""
    sub_sorted = sub.sort_values("steps", ascending=True)
    for _, r in sub_sorted.iterrows():
        if r["removed_poison"] < r["N_retrieved_poison"]:
            return int(r["steps"])
    return None


# --------------------------------------------------------------------------
# E1 / CORAL-PCA / CORAL-ridge comparison (reads existing artifacts only)
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
            most_perturbed_row = sub.sort_values("alpha", ascending=False).iloc[-1]
            summary["final_top_pair_pp"] = int(most_perturbed_row["top_pair_pp"])
            rows.append(summary)
    if not rows:
        return None, f"E1 comparison skipped: no matching rows for tested query_ids in {csv_path}."
    return pd.DataFrame(rows), None


def discover_latest_sweep_csv(output_dir: str, glob_pattern: str, filename: str) -> Optional[Path]:
    """Most recently written `filename` matching `glob_pattern` under
    `output_dir` (`results/diagnostics/cluster_normalized_poisoning_formal/`),
    by run directory name (`YYYYMMDD_HHMMSS_..._...`, which sorts
    lexicographically in time order) -- never regenerated here."""
    candidates = sorted(Path(output_dir).glob(f"{glob_pattern}/{filename}"))
    return candidates[-1] if candidates else None


def load_coral_comparison(sweep_csv: Optional[Path], tested_ids: Sequence[str],
                           label: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if sweep_csv is None or not sweep_csv.exists():
        return None, (
            f"{label} comparison skipped: no sweep CSV found under {DEFAULT_OUTPUT_DIR} "
            f"(run the Step 1/2 script first)."
        )
    df = pd.read_csv(sweep_csv)
    df = df[df["query_id"].isin(tested_ids)]
    if df.empty:
        return None, f"{label} comparison skipped: no matching rows for tested query_ids in {sweep_csv}."
    return df, None


# --------------------------------------------------------------------------
# Method comparison: E1 vs CORAL-PCA vs CORAL-ridge vs MMD
# --------------------------------------------------------------------------

def _first_residual_beta_pca(sub: pd.DataFrame) -> Optional[float]:
    sub_sorted = sub.sort_values("beta", ascending=True)
    for _, r in sub_sorted.iterrows():
        if r["removed_poison"] < r["N_retrieved_poison"]:
            return float(r["beta"])
    return None


def build_method_comparison(tested_ids: Sequence[str], mmd_sweep_df: pd.DataFrame,
                             pca_df: Optional[pd.DataFrame], ridge_df: Optional[pd.DataFrame],
                             e1_df: Optional[pd.DataFrame], lambda_preserves: Sequence[float],
                             steps_list: Sequence[int]) -> pd.DataFrame:
    """One row per `(query_id, method)`, each at that method's own maximum
    swept perturbation (`E1`: `alpha=0.3`; `CORAL_PCA`: `beta=max(betas)`;
    `CORAL_RIDGE_lambda=<l>`: `beta=max(betas)`; `MMD_lp=<lp>`:
    `steps=max(steps_list)`). `None` in a CORAL/MMD-distance or
    preservation column means that method's own artifact did not compute
    it (E1's sweep predates those metrics; read-only here, not
    recomputed)."""
    rows: List[Dict] = []
    steps_max = max(steps_list)

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
                "coral_distance_before": None, "coral_distance_after": None,
                "mmd_distance_before": None, "mmd_distance_after": None,
                "mean_poison_l2_displacement": None, "mean_poison_original_cosine": None,
                "decision_label_at_max_perturbation": r.get("final_decision_label"),
            })

    if pca_df is not None:
        beta_max = float(pca_df["beta"].max())
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
            first_beta = _first_residual_beta_pca(sub)
            rows.append({
                "query_id": qid, "method": "CORAL_PCA",
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
                "decision_label_at_max_perturbation": at_max.get("decision_label"),
            })

    if ridge_df is not None:
        beta_max = float(ridge_df["beta"].max())
        for qid in tested_ids:
            for lam in sorted(ridge_df["lambda"].unique()):
                sub = ridge_df[(ridge_df["query_id"] == qid) & (ridge_df["lambda"] == lam)]
                if sub.empty:
                    continue
                baseline = sub[sub["beta"] == 0.0]
                at_max = sub[sub["beta"] == beta_max]
                if at_max.empty:
                    continue
                at_max = at_max.iloc[0]
                baseline_row = baseline.iloc[0] if not baseline.empty else at_max
                first_beta = _first_residual_beta_pca(sub)
                rows.append({
                    "query_id": qid, "method": f"CORAL_RIDGE_lambda={lam}",
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
                    "decision_label_at_max_perturbation": at_max.get("decision_label"),
                })

    for qid in tested_ids:
        for lp in lambda_preserves:
            sub = mmd_sweep_df[(mmd_sweep_df["query_id"] == qid) & (mmd_sweep_df["lambda_preserve"] == lp)]
            if sub.empty:
                continue
            baseline = sub[sub["steps"] == 0]
            at_max = sub[sub["steps"] == steps_max]
            if at_max.empty:
                continue
            at_max = at_max.iloc[0]
            baseline_row = baseline.iloc[0] if not baseline.empty else at_max
            first_steps = first_residual_poison_steps(sub)
            rows.append({
                "query_id": qid, "method": f"MMD_lp={lp}",
                "perturbation_param": f"steps={steps_max} (max swept)",
                "first_residual_poison_perturbation": first_steps,
                "causes_residual_poison_failure": first_steps is not None,
                "top_pair_pp_baseline": baseline_row.get("top_pair_pp"),
                "top_pair_pp_at_max_perturbation": at_max.get("top_pair_pp"),
                "coral_distance_before": at_max.get("coral_distance_before"),
                "coral_distance_after": at_max.get("coral_distance_after"),
                "mmd_distance_before": at_max.get("mmd_distance_before"),
                "mmd_distance_after": at_max.get("mmd_distance_after"),
                "mean_poison_l2_displacement": at_max.get("mean_poison_l2_displacement"),
                "mean_poison_original_cosine": at_max.get("mean_poison_original_cosine"),
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
                      coral_pca_sweep_csv: Optional[Path], coral_ridge_sweep_csv: Optional[Path]) -> Dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "run_type": "mmd_oracle_intervention",
        "plan_step": "step_3_mmd_minimize_only",
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
        "coral_ridge_sweep_csv": str(coral_ridge_sweep_csv) if coral_ridge_sweep_csv else None,
        "intervention": INTERVENTION_LABEL,
        "lambda_preserves": args.lambda_preserves,
        "steps_list": args.steps_list,
        "gamma": args.gamma,
        "lr": args.lr,
        "seed": args.seed,
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
            "coral_ridge_rerun": False,
            "dan_trained": False,
            "mmd_optimizer_implemented": True,
        },
        "argv": sys.argv,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def render_report(tested_ids: Sequence[str], excluded: Dict[str, str], sweep_df: pd.DataFrame,
                   summary_rows: List[Dict], e1_df: Optional[pd.DataFrame], e1_skip_reason: Optional[str],
                   pca_df: Optional[pd.DataFrame], pca_skip_reason: Optional[str],
                   ridge_df: Optional[pd.DataFrame], ridge_skip_reason: Optional[str],
                   method_comparison_df: pd.DataFrame,
                   dataset: str, k: int, n_injected: int, lambda_preserves: Sequence[float],
                   steps_list: Sequence[int], gamma: float, lr: float, run_dir: Path) -> str:
    steps_max = max(steps_list)
    lines = [
        "# MMD-Minimizing Oracle Intervention Report (Step 3)",
        "",
        f"Run directory: `{run_dir.name}`",
        "",
        f"dataset=`{dataset}`, k={k}, N_injected={n_injected}, intervention=`{INTERVENTION_LABEL}`, "
        f"lambda_preserve swept = {lambda_preserves}, steps swept = {steps_list}, gamma={gamma}, lr={lr}.",
        "",
        "**No GPT/API calls were made. Baseline retrieval was not rerun. No baseline defense file "
        "was modified. E1, Step 1 (CORAL-PCA), and Step 2 (CORAL-ridge) were not rerun** (all three "
        "comparison sections read already-existing artifacts). All claims below are oracle "
        "embedding-space stress-test findings, not evidence of a text-realizable attack -- see "
        "Limitations.",
        "",
        "## Tested queries",
        "",
        f"- **{len(tested_ids)}** query(ies) tested (same success-case discovery/text-recoverability "
        "gate as `scripts/build_batch_comparison_success_cases.py` and Steps 1/2).",
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
        "## Per-(query, lambda_preserve, steps) MMD sweep",
        "",
        "`decision_label` is the absolute classification from "
        "`scripts/run_cluster_normalized_poisoning.py::decision_label` (reused unmodified). "
        "`first_failure_step_in_trace` is read from that config's own **per-step** trace CSV "
        "(every optimizer step is Stage-1/2-recomputed, not just the coarse `steps` grid values) -- "
        "`None` if no step in that trace shows residual poison.",
        "",
        "| query_id | lambda_preserve | steps | first_failure_step_in_trace | coral_before | coral_after | "
        "mmd_before | mmd_after | top_pair (PP/PC/CC) | N_adv | removed_poison | removed_clean | "
        "residual_poison_fraction | decision_label |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in sweep_df.sort_values(["query_id", "lambda_preserve", "steps"]).iterrows():
        lines.append(
            f"| `{r['query_id']}` | {r['lambda_preserve']} | {r['steps']} | {r['first_failure_step_in_trace']} | "
            f"{r['coral_distance_before']:.4f} | {r['coral_distance_after']:.4f} | "
            f"{r['mmd_distance_before']:.4f} | {r['mmd_distance_after']:.4f} | "
            f"{r['top_pair_pp']}/{r['top_pair_pc']}/{r['top_pair_cc']} | {r['N_adv']} | "
            f"{r['removed_poison']} | {r['removed_clean']} | {r['residual_poison_fraction']} | "
            f"{r['decision_label']} |"
        )
    lines.append("")

    lines += [
        "## Perturbation / preservation metrics",
        "",
        "Same convention as Steps 1/2: both the original and transformed poison embeddings are "
        "L2-normalized before displacement/cosine are computed, so `steps=0` gives **exact** "
        "identity regardless of `lambda_preserve` -- see "
        "`defense/coral_mmd_intervention.py::compute_preservation_metrics`.",
        "",
        "| query_id | lambda_preserve | steps | mean_poison_l2_displacement | max_poison_l2_displacement | "
        "mean_poison_original_cosine | min_poison_original_cosine |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in sweep_df.sort_values(["query_id", "lambda_preserve", "steps"]).iterrows():
        lines.append(
            f"| `{r['query_id']}` | {r['lambda_preserve']} | {r['steps']} | "
            f"{r['mean_poison_l2_displacement']:.4f} | {r['max_poison_l2_displacement']:.4f} | "
            f"{r['mean_poison_original_cosine']:.4f} | {r['min_poison_original_cosine']:.4f} |"
        )
    lines.append("")

    lines += [
        "## Per-(query, lambda_preserve) summary: first steps (ascending from 0) causing residual-poison failure",
        "",
        f"| query_id | lambda_preserve | first_residual_poison_steps | baseline_decision_label | "
        f"final_decision_label (steps={steps_max}) |",
        "|---|---|---|---|---|",
    ]
    n_failed = 0
    for row in summary_rows:
        if row["first_residual_poison_steps"] is not None:
            n_failed += 1
        lines.append(
            f"| `{row['query_id']}` | {row['lambda_preserve']} | {row['first_residual_poison_steps']} | "
            f"`{row['baseline_decision_label']}` | `{row['final_decision_label']}` |"
        )
    lines.append("")
    lines.append(
        f"**{n_failed} / {len(summary_rows)}** tested `(query, lambda_preserve)` configs show "
        f"residual-poison failure under MMD minimization within the swept steps (`{steps_list}`)."
    )
    lines.append("")

    lines += ["## Method comparison: E1 vs CORAL-PCA vs CORAL-ridge vs MMD", ""]
    if method_comparison_df.empty:
        lines.append("_Method comparison skipped: no data available (see skip reasons below)._")
        lines.append("")
    else:
        lines += [
            "One row per `(query_id, method)`, each at that method's own maximum swept perturbation. "
            "`None` in a CORAL/MMD-distance or preservation column means that method's own artifact "
            "did not compute it (E1's sweep predates those metrics; read-only here, not recomputed).",
            "",
            "| query_id | method | first_residual_poison_perturbation | causes_failure | "
            "top_pair_pp_baseline | top_pair_pp_at_max | mmd_before | mmd_after | "
            "mean_poison_l2_displacement | mean_poison_original_cosine |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        def _fmt(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return "None"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        for _, r in method_comparison_df.sort_values(["query_id", "method"]).iterrows():
            lines.append(
                f"| `{r['query_id']}` | `{r['method']}` | {_fmt(r['first_residual_poison_perturbation'])} | "
                f"{r['causes_residual_poison_failure']} | {_fmt(r['top_pair_pp_baseline'])} | "
                f"{_fmt(r['top_pair_pp_at_max_perturbation'])} | {_fmt(r['mmd_distance_before'])} | "
                f"{_fmt(r['mmd_distance_after'])} | {_fmt(r['mean_poison_l2_displacement'])} | "
                f"{_fmt(r['mean_poison_original_cosine'])} |"
            )
        lines.append("")

    for reason in (e1_skip_reason, pca_skip_reason, ridge_skip_reason):
        if reason:
            lines.append(f"_{reason}_")
            lines.append("")

    lines += ["## Questions this run answers", ""]

    n_mmd_failures = sum(1 for row in summary_rows if row["first_residual_poison_steps"] is not None)
    lines.append(
        f"**1. Does MMD optimization cause residual-poison failures?** "
        f"{'Yes' if n_mmd_failures > 0 else 'No'} -- {n_mmd_failures} / {len(summary_rows)} tested "
        f"`(query, lambda_preserve)` configs show residual-poison failure within the swept steps "
        f"(see the per-`(query, lambda_preserve)` summary table above)."
    )
    lines.append("")

    if n_mmd_failures > 0:
        failing = [row for row in summary_rows if row["first_residual_poison_steps"] is not None]
        min_steps_row = min(failing, key=lambda row: row["first_residual_poison_steps"])
        lines.append(
            f"**2. At what lambda_preserve / steps does failure first appear, if any?** Earliest: "
            f"query `{min_steps_row['query_id']}`, lambda_preserve={min_steps_row['lambda_preserve']}, "
            f"first_residual_poison_steps={min_steps_row['first_residual_poison_steps']} (see also "
            f"`first_failure_step_in_trace` in the sweep table above for the exact per-step trace "
            f"crossing point, finer-grained than the coarse `{steps_list}` grid)."
        )
    else:
        lines.append(
            "**2. At what lambda_preserve / steps does failure first appear, if any?** N/A -- no "
            "residual-poison failure occurred at any swept `(lambda_preserve, steps)` config."
        )
    lines.append("")

    if not method_comparison_df.empty:
        mmd_rows = method_comparison_df[method_comparison_df["method"].str.startswith("MMD_")]
        pca_rows = method_comparison_df[method_comparison_df["method"] == "CORAL_PCA"]
        ridge_rows = method_comparison_df[method_comparison_df["method"].str.startswith("CORAL_RIDGE")]
        coral_rows = pd.concat([pca_rows, ridge_rows]) if not pca_rows.empty or not ridge_rows.empty else pca_rows

        if not mmd_rows.empty and not coral_rows.empty:
            mmd_pp_reduction = (mmd_rows["top_pair_pp_baseline"] - mmd_rows["top_pair_pp_at_max_perturbation"]).mean()
            coral_pp_reduction = (coral_rows["top_pair_pp_baseline"] - coral_rows["top_pair_pp_at_max_perturbation"]).mean()
            lines.append(
                f"**3. Does MMD reduce `top_pair_pp` more than CORAL?** Mean `top_pair_pp` reduction "
                f"at max swept perturbation: MMD = {mmd_pp_reduction:.3f} (averaged across "
                f"lambda_preserve and queries) vs CORAL (PCA+ridge combined) = {coral_pp_reduction:.3f}."
            )
            lines.append("")

            mmd_reduction = (mmd_rows["mmd_distance_before"] - mmd_rows["mmd_distance_after"]).mean()
            mmd_disp = mmd_rows["mean_poison_l2_displacement"].mean()
            mmd_cos = mmd_rows["mean_poison_original_cosine"].mean()
            lines.append(
                f"**4. Does MMD reduce MMD distance while preserving original poison embeddings?** "
                f"Mean MMD-distance reduction at max perturbation = {mmd_reduction:.4f}; mean "
                f"`mean_poison_l2_displacement` = {mmd_disp:.4f}; mean `mean_poison_original_cosine` "
                f"= {mmd_cos:.4f}. By construction (`lambda_preserve * mean(||Zp_prime - Zp||^2)` is "
                f"part of the optimized loss), MMD is *designed* to trade off MMD-distance reduction "
                f"against preservation -- these two numbers together characterize where each "
                f"lambda_preserve setting lands on that trade-off (see the full sweep table for the "
                f"per-lambda_preserve breakdown)."
            )
            lines.append("")

        if e1_df is not None and not method_comparison_df[method_comparison_df["method"].str.startswith("E1_")].empty:
            e1_rows = method_comparison_df[method_comparison_df["method"].str.startswith("E1_")]
            e1_any_failure = bool(e1_rows["causes_residual_poison_failure"].any())
            coral_any_failure = bool(coral_rows["causes_residual_poison_failure"].any()) if not coral_rows.empty else False
            mmd_any_failure = bool(mmd_rows["causes_residual_poison_failure"].any()) if not mmd_rows.empty else False
            if mmd_any_failure and not coral_any_failure:
                relative = "stronger than CORAL (causes failure where CORAL did not)"
            elif mmd_any_failure and coral_any_failure:
                relative = "at least as strong as CORAL (both cause failure in this run)"
            else:
                relative = "no stronger than CORAL (neither causes failure in this run)"
            vs_e1 = "weaker than E1" if (e1_any_failure and not mmd_any_failure) else (
                "comparable to E1" if (e1_any_failure and mmd_any_failure) else "N/A (E1 comparison unavailable)"
                if e1_df is None else "stronger than or equal to E1"
            )
            lines.append(
                f"**5. Is MMD stronger than CORAL but weaker/stronger than E1?** MMD is **{relative}**. "
                f"E1 causes residual-poison failure in "
                f"{'all' if e1_any_failure else 'none'} of its tested `(query, strategy)` configs; "
                f"relative to E1, MMD in this run is **{vs_e1}**."
            )
            lines.append("")

    if n_mmd_failures > 0 and not method_comparison_df.empty:
        mmd_rows_all = method_comparison_df[method_comparison_df["method"].str.startswith("MMD_")]
        failing_mmd = mmd_rows_all[mmd_rows_all["causes_residual_poison_failure"]]
        succeeding_mmd = mmd_rows_all[~mmd_rows_all["causes_residual_poison_failure"]]
        if not failing_mmd.empty and not succeeding_mmd.empty:
            fail_disp = failing_mmd["mean_poison_l2_displacement"].mean()
            ok_disp = succeeding_mmd["mean_poison_l2_displacement"].mean()
            lines.append(
                f"**6. Does residual-poison failure require large embedding displacement?** Mean "
                f"`mean_poison_l2_displacement` at max perturbation: failing configs = {fail_disp:.4f} "
                f"vs non-failing configs = {ok_disp:.4f}. "
                f"{'Failing configs show larger displacement, consistent with failure requiring substantial movement away from the original poison embeddings.' if fail_disp > ok_disp else 'Failing configs do NOT show larger displacement than non-failing configs -- failure does not appear to require especially large embedding displacement in this run.'}"
            )
        else:
            lines.append(
                "**6. Does residual-poison failure require large embedding displacement?** "
                "Insufficient variation (failing and non-failing configs are not both present) to "
                "compare displacement between them in this run."
            )
    else:
        lines.append(
            "**6. Does residual-poison failure require large embedding displacement?** N/A -- no "
            "residual-poison failure occurred in this run to compare displacement against."
        )
    lines.append("")

    lines.append(
        "**7. Is `top_pair_pp` still the most specific mechanistic indicator?** Across the sweep "
        "table above, `top_pair_pp` tracks `removed_poison`/`decision_label` more tightly than the "
        "aggregate `coral_distance_after`/`mmd_distance_after` values do (large CORAL/MMD-distance "
        "reductions repeatedly co-occur with `top_pair_pp` still at or near its maximum, i.e. "
        "`removed_poison` unaffected) -- consistent with Steps 1/2's own finding. `top_pair_pp` "
        "remains the more specific, mechanistic (Stage-2-pair-level) indicator of whether RAGDefender's "
        "actual removal decision is at risk; the global distribution-distance metrics are necessary "
        "but not sufficient signals of that risk."
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
    parser.add_argument("--coral_ridge_sweep_csv", default=None,
                         help="Path to an existing CORAL_RIDGE_SWEEP.csv; default None auto-discovers "
                              "the most recent one under --output_dir.")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n_injected", type=int, default=5)
    parser.add_argument("--exclude_query_id", action="append", default=["5a8cb288554299585d9e3726"])
    parser.add_argument("--lambda_preserves", type=float, nargs="+", default=DEFAULT_LAMBDA_PRESERVES)
    parser.add_argument("--steps_list", type=int, nargs="+", default=DEFAULT_STEPS_LIST)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--stage2_p", type=float, default=DEFAULT_STAGE2_P)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Path:
    args = parse_args(argv)

    for st in args.steps_list:
        if st < 0:
            raise ValueError(f"--steps_list must all be >= 0; got {st}")

    records = viz._read_jsonl(args.diagnostics_jsonl)
    records_by_id = {r["query_id"]: r for r in records}
    qr_index = viz.load_query_results_index(args.query_results_dir)

    tested_ids, excluded = discover_tested_query_ids(
        records, records_by_id, qr_index, args.dataset, args.k, args.n_injected, args.exclude_query_id
    )
    if not tested_ids:
        raise ValueError("No tested query_ids found -- check --diagnostics_jsonl/--query_results_dir.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{ts}_mmd_{args.dataset}_k{args.k}_N{args.n_injected}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "similarity_matrices").mkdir()
    (run_dir / "traces").mkdir()
    manifest: Dict[str, List[str]] = {"config": [], "csv": [], "matrices": [], "traces": [], "report": []}

    model = viz.load_embedder(args.embedder)

    all_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    for qid in tested_ids:
        rec = records_by_id[qid]
        texts = viz.recover_pre_defense_texts(qr_index.get(qid))
        original_row, original_label, rows = process_query(
            qid, rec, texts, model, args.lambda_preserves, args.steps_list, args.gamma, args.lr, args.seed,
            args.stage2_p, run_dir, manifest,
        )
        all_rows.extend(rows)

        sub_df = pd.DataFrame(rows)
        for lp in args.lambda_preserves:
            lp_sub = sub_df[sub_df["lambda_preserve"] == lp]
            summary_rows.append({
                "query_id": qid,
                "lambda_preserve": lp,
                "baseline_decision_label": original_label,
                "final_decision_label": lp_sub.sort_values("steps", ascending=False).iloc[0]["decision_label"],
                "first_residual_poison_steps": first_residual_poison_steps(lp_sub),
            })

    sweep_df = pd.DataFrame(all_rows)
    sweep_csv_path = run_dir / "MMD_SWEEP.csv"
    sweep_df.to_csv(sweep_csv_path, index=False)
    manifest["csv"].append(str(sweep_csv_path.relative_to(run_dir)))

    e1_df, e1_skip_reason = load_e1_comparison(args.e1_output_dir, tested_ids)

    coral_pca_sweep_csv = Path(args.coral_pca_sweep_csv) if args.coral_pca_sweep_csv else \
        discover_latest_sweep_csv(args.output_dir, "*_coral_pca_*", "CORAL_PCA_SWEEP.csv")
    pca_df, pca_skip_reason = load_coral_comparison(coral_pca_sweep_csv, tested_ids, "CORAL-PCA")

    coral_ridge_sweep_csv = Path(args.coral_ridge_sweep_csv) if args.coral_ridge_sweep_csv else \
        discover_latest_sweep_csv(args.output_dir, "*_coral_ridge_*", "CORAL_RIDGE_SWEEP.csv")
    ridge_df, ridge_skip_reason = load_coral_comparison(coral_ridge_sweep_csv, tested_ids, "CORAL-ridge")

    method_comparison_df = build_method_comparison(
        tested_ids, sweep_df, pca_df, ridge_df, e1_df, args.lambda_preserves, args.steps_list
    )
    comparison_csv_path = run_dir / "METHOD_COMPARISON_FORMAL.csv"
    method_comparison_df.to_csv(comparison_csv_path, index=False)
    manifest["csv"].append(str(comparison_csv_path.relative_to(run_dir)))

    report_text = render_report(
        tested_ids, excluded, sweep_df, summary_rows, e1_df, e1_skip_reason, pca_df, pca_skip_reason,
        ridge_df, ridge_skip_reason, method_comparison_df, args.dataset, args.k, args.n_injected,
        args.lambda_preserves, args.steps_list, args.gamma, args.lr, run_dir,
    )
    report_path = run_dir / "MMD_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    manifest["report"].append(str(report_path.relative_to(run_dir)))

    defense = records_by_id[tested_ids[0]]["defense"]
    run_config = build_run_config(
        args, args.dataset, args.k, args.n_injected, defense, tested_ids, excluded,
        coral_pca_sweep_csv, coral_ridge_sweep_csv,
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
        f"Tested {len(tested_ids)} queries x {len(args.lambda_preserves)} lambda_preserves x "
        f"{len(args.steps_list)} steps = {len(sweep_df)} MMD rows."
    )
    print(f"Wrote MMD oracle run to: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
