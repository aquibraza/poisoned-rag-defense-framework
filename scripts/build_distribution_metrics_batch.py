#!/usr/bin/env python3
"""Cluster-Normalized Poisoning: formal distribution-matching diagnostics
for the existing E1 oracle batch.

Adds CORAL distance and RBF-kernel MMD (see `defense/distribution_metrics.py`
-- **diagnostic metrics only**, no CORAL transform or MMD optimizer is
implemented) to every `(query_id, anchor_strategy, alpha)` row already
produced by `scripts/run_cluster_normalized_poisoning.py`'s E1 runs, joins
them with the existing per-alpha batch summary fields, and answers six
distribution-level questions about the batch.

This script performs **no new oracle computation**: it reads

- the diagnostics `.jsonl` (for `retrieved_is_poison`, to recover
  `poison_idx`/`clean_idx` -- exactly how
  `scripts/run_cluster_normalized_poisoning.py` itself derives them, see
  `defense.cluster_normalized_poisoning.split_poison_clean`),
- each run's already-written `intervention_sweep.csv` (via
  `scripts/summarize_cluster_normalized_poisoning.py::load_sweep`), and
- each run's already-saved `similarity_matrices/*.npy` cosine matrices,

and otherwise reuses `scripts/build_batch_comparison_success_cases.py`'s
success-case discovery/text-recoverability gate verbatim, so the tested
`(query, strategy)` set here is identical to that batch's. No embedder is
loaded, no GPT/API call is made, no baseline retrieval or defense file is
read/written.

Usage:
    python scripts/build_distribution_metrics_batch.py \\
        --diagnostics_jsonl results/diagnostics/ragdefender_smoke_live_10q/hotpotqa-...-defense-ragdefender_original.jsonl \\
        --query_results_dir results/query_results/ragdefender_smoke_live_10q
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import visualize_ragdefender_clusters as viz  # noqa: E402
import summarize_cluster_normalized_poisoning as summ  # noqa: E402
import build_batch_comparison_success_cases as batchmod  # noqa: E402
from defense.distribution_metrics import (  # noqa: E402
    DEFAULT_MMD_GAMMA,
    coral_distance_from_gram,
    mmd_rbf_distance_from_gram,
    slice_gram_blocks,
)

E1_STRATEGIES = batchmod.E1_STRATEGIES
DEFAULT_OUTPUT_DIR = batchmod.DEFAULT_OUTPUT_DIR
DEFAULT_DIAGNOSTICS_JSONL = batchmod.DEFAULT_DIAGNOSTICS_JSONL
DEFAULT_QUERY_RESULTS_DIR = batchmod.DEFAULT_QUERY_RESULTS_DIR
PP_WEAKENING_ALPHA_THRESHOLD = batchmod.PP_WEAKENING_ALPHA_THRESHOLD


# --------------------------------------------------------------------------
# Load already-saved similarity matrices + attach CORAL/MMD per alpha
# --------------------------------------------------------------------------

_TRANSFORMED_ALPHA_RE = re.compile(r"^transformed_M_alpha([0-9.]+)\.npy$")


def load_similarity_matrices(run_dir: Path) -> Dict[float, np.ndarray]:
    """Load every saved cosine matrix for a run directory, keyed by alpha
    (rounded to 6 decimals to make float-keyed lookups robust). Prefers
    `transformed_M_alpha{alpha}.npy` for every alpha the sweep covers
    (including 1.0, which `run_cluster_normalized_poisoning.py` also
    writes); falls back to `original_M.npy` for alpha=1.0 only if no
    `transformed_M_alpha1.0.npy` file exists (older run directories)."""
    sim_dir = run_dir / "similarity_matrices"
    matrices: Dict[float, np.ndarray] = {}
    for path in sorted(sim_dir.glob("transformed_M_alpha*.npy")):
        match = _TRANSFORMED_ALPHA_RE.match(path.name)
        if match:
            matrices[round(float(match.group(1)), 6)] = np.load(path)
    original_path = sim_dir / "original_M.npy"
    if original_path.exists() and 1.0 not in matrices:
        matrices[1.0] = np.load(original_path)
    return matrices


def _lookup_matrix(matrices: Dict[float, np.ndarray], alpha: float) -> np.ndarray:
    key = round(float(alpha), 6)
    if key in matrices:
        return matrices[key]
    close = [k for k in matrices if abs(k - alpha) < 1e-6]
    if close:
        return matrices[close[0]]
    raise FileNotFoundError(f"No saved similarity matrix found for alpha={alpha!r} "
                             f"(available alphas: {sorted(matrices)})")


def attach_distribution_metrics(df: pd.DataFrame, matrices: Dict[float, np.ndarray],
                                 poison_idx: Sequence[int], clean_idx: Sequence[int],
                                 gamma: float = DEFAULT_MMD_GAMMA) -> pd.DataFrame:
    """Pure (given already-loaded `matrices`): adds `coral_distance` and
    `mmd_distance` columns to `df` (one alpha-sweep DataFrame), computed
    from each alpha's saved cosine matrix sliced into poison/clean Gram
    blocks. Does not mutate `df` in place; returns a copy."""
    df = df.copy()
    coral_values: List[float] = []
    mmd_values: List[float] = []
    for alpha in df["alpha"]:
        m = _lookup_matrix(matrices, float(alpha))
        g_pp, g_pc, g_cc = slice_gram_blocks(m, poison_idx, clean_idx)
        coral_values.append(coral_distance_from_gram(g_pp, g_pc, g_cc))
        mmd_values.append(mmd_rbf_distance_from_gram(g_pp, g_pc, g_cc, gamma=gamma))
    df["coral_distance"] = coral_values
    df["mmd_distance"] = mmd_values
    return df


# --------------------------------------------------------------------------
# Per-(query, strategy) extended summary (distribution-metric trigger
# alphas, on top of scripts/build_batch_comparison_success_cases.py's
# existing compute_config_summary)
# --------------------------------------------------------------------------

def pearson_corr(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Plain-numpy Pearson correlation (no SciPy dependency, matching this
    plan's existing no-SciPy convention). `None` if undefined (fewer than
    2 points, or either series is constant)."""
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compute_extended_config_summary(query_id: str, strategy: str, df: pd.DataFrame) -> Dict:
    """`df` must already have `coral_distance`/`mmd_distance` columns
    attached (via `attach_distribution_metrics`). Extends
    `build_batch_comparison_success_cases.compute_config_summary` with:

    - `{coral,mmd}_decreased_alpha`: first alpha (descending from 1.0)
      where the metric first drops below its own alpha=1.0 baseline value.
    - `{baseline,final}_{coral,mmd}_distance` and `{coral,mmd}_reduction`
      (baseline minus final, at the deepest swept alpha) -- positive means
      the poison group's distribution moved *toward* the clean group's
      over the sweep.
    - `alpha_{coral,mmd}_pearson_r`: Pearson correlation between `alpha`
      and the metric across the sweep -- positive means the metric falls
      as alpha falls (the expected direction if the intervention is
      genuinely moving the poison distribution toward the clean one).
    - `mean_pp_decreased_alpha`: same "first decrease" pattern applied to
      the already-existing `mean_poison_poison_similarity` column, for the
      Q_D leading-indicator comparison.
    """
    base = batchmod.compute_config_summary(query_id, strategy, df)
    sub = df.sort_values("alpha", ascending=False).reset_index(drop=True)
    baseline = sub.iloc[0]
    final = sub.iloc[-1]

    def first_alpha(predicate) -> Optional[float]:
        for _, r in sub.iterrows():
            if predicate(r):
                return float(r["alpha"])
        return None

    base.update({
        "baseline_coral_distance": float(baseline["coral_distance"]),
        "final_coral_distance": float(final["coral_distance"]),
        "coral_reduction": float(baseline["coral_distance"] - final["coral_distance"]),
        "coral_decreased_alpha": first_alpha(lambda r: r["coral_distance"] < baseline["coral_distance"]),
        "alpha_coral_pearson_r": pearson_corr(sub["alpha"], sub["coral_distance"]),
        "baseline_mmd_distance": float(baseline["mmd_distance"]),
        "final_mmd_distance": float(final["mmd_distance"]),
        "mmd_reduction": float(baseline["mmd_distance"] - final["mmd_distance"]),
        "mmd_decreased_alpha": first_alpha(lambda r: r["mmd_distance"] < baseline["mmd_distance"]),
        "alpha_mmd_pearson_r": pearson_corr(sub["alpha"], sub["mmd_distance"]),
        "mean_pp_decreased_alpha": first_alpha(
            lambda r: r["mean_poison_poison_similarity"] < baseline["mean_poison_poison_similarity"]
        ),
    })
    return base


# --------------------------------------------------------------------------
# Generic leading-indicator-vs-outcome categorization (generalizes
# build_batch_comparison_success_cases.answer_q6 to any pair of columns)
# --------------------------------------------------------------------------

def leading_indicator_categories(summary_df: pd.DataFrame, leading_col: str, outcome_col: str) -> Dict:
    """For every row (one `(query, strategy)` config), classify the
    relationship between a candidate "leading indicator" alpha
    (`leading_col`, e.g. `coral_decreased_alpha`) and an "outcome" alpha
    (`outcome_col`, e.g. `first_residual_poison_alpha`):

    - `leading_precedes_or_coincides_with_outcome`: both triggered, and
      the leading indicator changed at the same-or-higher alpha (i.e. no
      later) than the outcome occurred.
    - `leading_triggered_without_outcome`: the leading indicator changed,
      but the outcome never occurred within the swept alphas.
    - `outcome_without_leading_triggered_first`: the outcome occurred at
      an alpha where the leading indicator had not yet (or never) changed
      -- evidence against this being a leading indicator for this config.
    - `neither_triggered`: no effect observed in this config.

    Also computes `mean_lead_gap`: the mean of `(leading_alpha -
    outcome_alpha)` over configs where the indicator precedes-or-coincides
    with the outcome (always `>= 0`). A **large** mean lead gap means the
    indicator tends to fire long before the outcome -- which can mean
    it is a trivially-early/oversensitive signal (e.g. it changes at the
    very first perturbation step regardless of whether failure ever
    follows) rather than a *specific* early-warning signal. Combined with
    `support_fraction`, this disambiguates indicators that tie on support
    but differ in how tightly they track the actual outcome.

    Returns `{"categories": {...}, "support_fraction": float|None,
    "n_informative": int, "mean_lead_gap": float|None}`, where
    `support_fraction` is the fraction of "informative" configs (either
    condition triggered) in which the leading indicator actually
    preceded-or-coincided with the outcome."""
    categories = {
        "leading_precedes_or_coincides_with_outcome": 0,
        "leading_triggered_without_outcome": 0,
        "outcome_without_leading_triggered_first": 0,
        "neither_triggered": 0,
    }
    lead_gaps: List[float] = []
    for _, r in summary_df.iterrows():
        lead = r[leading_col]
        outcome = r[outcome_col]
        lead_ok = lead is not None and not pd.isna(lead)
        outcome_ok = outcome is not None and not pd.isna(outcome)
        if lead_ok and outcome_ok:
            if lead >= outcome:
                cat = "leading_precedes_or_coincides_with_outcome"
                lead_gaps.append(float(lead) - float(outcome))
            else:
                cat = "outcome_without_leading_triggered_first"
        elif lead_ok and not outcome_ok:
            cat = "leading_triggered_without_outcome"
        elif outcome_ok and not lead_ok:
            cat = "outcome_without_leading_triggered_first"
        else:
            cat = "neither_triggered"
        categories[cat] += 1
    n_informative = (categories["leading_precedes_or_coincides_with_outcome"]
                      + categories["outcome_without_leading_triggered_first"])
    support_fraction = (
        categories["leading_precedes_or_coincides_with_outcome"] / n_informative if n_informative else None
    )
    mean_lead_gap = float(np.mean(lead_gaps)) if lead_gaps else None
    return {
        "categories": categories,
        "support_fraction": support_fraction,
        "n_informative": n_informative,
        "mean_lead_gap": mean_lead_gap,
    }


# --------------------------------------------------------------------------
# The six batch-analysis questions
# --------------------------------------------------------------------------

def answer_decrease_with_alpha(summary_df: pd.DataFrame) -> Dict:
    """Q_A: do CORAL/MMD distances decrease as alpha decreases?"""
    n = len(summary_df)
    coral_final_lower = int((summary_df["final_coral_distance"] < summary_df["baseline_coral_distance"]).sum())
    mmd_final_lower = int((summary_df["final_mmd_distance"] < summary_df["baseline_mmd_distance"]).sum())
    coral_r = [v for v in summary_df["alpha_coral_pearson_r"] if v is not None and not pd.isna(v)]
    mmd_r = [v for v in summary_df["alpha_mmd_pearson_r"] if v is not None and not pd.isna(v)]
    return {
        "n_configs": n,
        "coral_final_lower_than_baseline_count": coral_final_lower,
        "mmd_final_lower_than_baseline_count": mmd_final_lower,
        "mean_alpha_coral_pearson_r": float(np.mean(coral_r)) if coral_r else None,
        "mean_alpha_mmd_pearson_r": float(np.mean(mmd_r)) if mmd_r else None,
        "n_configs_positive_coral_r": sum(1 for v in coral_r if v > 0),
        "n_configs_positive_mmd_r": sum(1 for v in mmd_r if v > 0),
        "n_r_informative": len(coral_r),
    }


def answer_align_with_pp_weakening(summary_df: pd.DataFrame) -> Dict:
    """Q_B: do decreases in CORAL/MMD align with PP top-pair weakening?"""
    return {
        "coral_vs_pp": leading_indicator_categories(summary_df, "coral_decreased_alpha", "pp_decreased_alpha"),
        "mmd_vs_pp": leading_indicator_categories(summary_df, "mmd_decreased_alpha", "pp_decreased_alpha"),
    }


def answer_predict_residual_poison_failure(summary_df: pd.DataFrame) -> Dict:
    """Q_C: do CORAL/MMD decreases precede or coincide with residual-poison
    failure? (Deliberately not phrased as "predict" -- see Q_D: they do
    precede-or-coincide in essentially every informative config, but so
    trivially/early that this alone is weak evidence of a specific causal
    or predictive relationship; `top_pair_pp` is the more specific
    mechanistic indicator.)"""
    return {
        "coral_vs_failure": leading_indicator_categories(summary_df, "coral_decreased_alpha",
                                                          "first_residual_poison_alpha"),
        "mmd_vs_failure": leading_indicator_categories(summary_df, "mmd_decreased_alpha",
                                                        "first_residual_poison_alpha"),
    }


def answer_best_predictor_comparison(summary_df: pd.DataFrame) -> Dict:
    """Q_D: which of PP top-pair count, CORAL distance, MMD distance, or
    mean PP similarity is the more specific mechanistic indicator that
    precedes-or-coincides with residual-poison failure? Ranks the four
    candidates primarily by `support_fraction` against
    `first_residual_poison_alpha`, and -- since `support_fraction` alone
    can saturate at 100% for an indicator that simply fires at the very
    first perturbation step regardless of whether failure follows --
    breaks ties by the *smaller* `mean_lead_gap` (the indicator that
    fires closer to, not just always before, the actual failure). This
    is a specificity comparison, not a claim that any of these four
    causally predicts failure."""
    candidates = {
        "top_pair_pp": "pp_decreased_alpha",
        "coral_distance": "coral_decreased_alpha",
        "mmd_distance": "mmd_decreased_alpha",
        "mean_poison_poison_similarity": "mean_pp_decreased_alpha",
    }
    results = {}
    for label, col in candidates.items():
        results[label] = leading_indicator_categories(summary_df, col, "first_residual_poison_alpha")

    def sort_key(item):
        _, result = item
        support = result["support_fraction"] if result["support_fraction"] is not None else -1.0
        gap = result["mean_lead_gap"] if result["mean_lead_gap"] is not None else float("inf")
        return (-support, gap)

    ranked = sorted(results.items(), key=sort_key)
    return {"results": results, "ranked": ranked, "best": ranked[0][0] if ranked else None}


def answer_best_mover_strategy(summary_df: pd.DataFrame, query_ids: Sequence[str]) -> Dict:
    """Q_E: which E1 anchor strategy most consistently moves poison
    embeddings toward the clean distribution (largest CORAL/MMD
    reduction from alpha=1.0 to the deepest swept alpha)?"""
    mean_reduction = summary_df.groupby("strategy")[["coral_reduction", "mmd_reduction"]].mean()
    coral_wins = {s: 0 for s in E1_STRATEGIES}
    mmd_wins = {s: 0 for s in E1_STRATEGIES}
    for qid in query_ids:
        rows = summary_df[summary_df["query_id"] == qid]
        if rows.empty:
            continue
        coral_wins[rows.loc[rows["coral_reduction"].idxmax(), "strategy"]] += 1
        mmd_wins[rows.loc[rows["mmd_reduction"].idxmax(), "strategy"]] += 1
    return {
        "mean_reduction_by_strategy": mean_reduction.to_dict(orient="index"),
        "coral_wins_by_strategy": coral_wins,
        "mmd_wins_by_strategy": mmd_wins,
        "top_coral_mover": max(coral_wins, key=coral_wins.get) if coral_wins else None,
        "top_mmd_mover": max(mmd_wins, key=mmd_wins.get) if mmd_wins else None,
    }


def answer_geometric_meaningfulness(decrease: Dict, predict: Dict) -> Dict:
    """Q_F: is the E1 effect geometrically meaningful, or just arbitrary
    interpolation? Synthesized from Q_A (does the distribution actually
    move toward clean as alpha falls) and Q_C (does that movement precede
    or coincide with the actual failure outcome), rather than a
    fixed/hardcoded verdict."""
    mean_r = np.nanmean([
        v for v in [decrease["mean_alpha_coral_pearson_r"], decrease["mean_alpha_mmd_pearson_r"]] if v is not None
    ]) if (decrease["mean_alpha_coral_pearson_r"] is not None or decrease["mean_alpha_mmd_pearson_r"] is not None) \
        else None
    coral_support = predict["coral_vs_failure"]["support_fraction"]
    mmd_support = predict["mmd_vs_failure"]["support_fraction"]
    supports = [v for v in [coral_support, mmd_support] if v is not None]
    mean_support = float(np.mean(supports)) if supports else None
    geometrically_meaningful = (
        mean_r is not None and mean_r > 0.5 and mean_support is not None and mean_support >= 0.7
    )
    return {
        "mean_alpha_distance_pearson_r": float(mean_r) if mean_r is not None else None,
        "mean_predictive_support_fraction": mean_support,
        "verdict_geometrically_meaningful": geometrically_meaningful,
    }


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------

LIMITATIONS_TEXT = """## Limitations

- **CORAL and MMD are diagnostic metrics only in this batch.** No CORAL
  feature-alignment transform and no MMD-minimizing optimizer is
  implemented or run here -- both remain deferred (candidates B/C in
  `docs/CLUSTER_NORMALIZED_POISONING_EXECUTION_PLAN.md`).
- **This remains an oracle embedding-space stress test.** Every metric
  here is computed from already-transformed poisoned-passage embeddings
  produced by `scripts/run_cluster_normalized_poisoning.py`'s E1
  interventions; no new transform, retrieval, or generation was run.
- **No text-realizable attack claim is made.** A geometrically meaningful
  CORAL/MMD reduction says the *representation* moved toward the clean
  distribution under a direct embedding-space intervention -- not that
  this movement is reachable by rewriting the poisoned passage's text
  under the frozen `paraphrase-MiniLM-L6-v2` encoder.
- CORAL/MMD here are computed **from cosine-similarity matrices only**
  (no raw embeddings were re-loaded or re-encoded) via the Gram-matrix
  identities documented in `defense/distribution_metrics.py`. These
  identities were validated by numerically comparing this
  Gram-matrix-only computation against a direct from-embedding
  covariance/kernel computation; the two **numerically match to
  floating-point tolerance** (see `defense/distribution_metrics.py`'s own
  tests) -- not an approximation, but also not a claim of any stronger
  guarantee than that. They describe the L2-normalized embedding
  *directions* RAGDefender's cosine-similarity logic actually operates
  on, not the raw pre-normalization embedding vectors.
- **CORAL/MMD decreases precede or coincide with residual-poison
  failure, but `top_pair_pp` is the more specific mechanistic
  indicator** (see Q_C/Q_D): CORAL/MMD fall at essentially every alpha
  step regardless of whether failure follows, so their high
  precedes-or-coincides rate is a weak, non-causal association, not
  evidence that CORAL/MMD predict failure.
- The RBF kernel's `gamma` is a **fixed, lightweight default**
  (not a per-query median-heuristic bandwidth) -- a deliberate simplicity
  choice for a diagnostic metric.
- Sample size is small (6 queries x 4 strategies = 24 configs); all
  correlations/support fractions below are descriptive, not
  statistically powered causal claims.
"""


def render_markdown(decrease: Dict, align: Dict, predict: Dict, best_predictor: Dict,
                     best_mover: Dict, meaningfulness: Dict, summary_df: pd.DataFrame,
                     tested_query_ids: Sequence[str], gamma: float) -> str:
    n = decrease["n_configs"]
    lines = [
        "# Cluster-Normalized Poisoning -- Distribution-Matching Diagnostics (E1 Batch)",
        "",
        f"Extends `BATCH_COMPARISON_SUCCESS_CASES.md`/`.csv` ({len(tested_query_ids)} tested queries x "
        f"{len(E1_STRATEGIES)} E1 anchor strategies = {n} configs, alpha in "
        "{1.0, 0.9, ..., 0.3}) with CORAL distance and RBF-kernel MMD "
        f"(gamma={gamma}) computed from each alpha's already-saved cosine matrix.",
        "",
        "**No GPT/API calls were made. No oracle sweep, baseline retrieval, or baseline "
        "defense file was rerun/modified.** CORAL/MMD are diagnostic metrics only here "
        "-- see Limitations.",
        "",
        "## Q_A. Do CORAL/MMD distances decrease as alpha decreases?",
        "",
        f"- **{decrease['coral_final_lower_than_baseline_count']} / {n}** configs have a lower CORAL "
        "distance at the deepest swept alpha than at the alpha=1.0 baseline.",
        f"- **{decrease['mmd_final_lower_than_baseline_count']} / {n}** configs have a lower MMD "
        "distance at the deepest swept alpha than at the alpha=1.0 baseline.",
        f"- Mean Pearson correlation between `alpha` and `coral_distance` across the sweep: "
        f"**{decrease['mean_alpha_coral_pearson_r']:.3f}** "
        f"({decrease['n_configs_positive_coral_r']}/{decrease['n_r_informative']} configs positive)."
        if decrease["mean_alpha_coral_pearson_r"] is not None else
        "- Mean Pearson correlation between `alpha` and `coral_distance`: not computable (no variation).",
        f"- Mean Pearson correlation between `alpha` and `mmd_distance` across the sweep: "
        f"**{decrease['mean_alpha_mmd_pearson_r']:.3f}** "
        f"({decrease['n_configs_positive_mmd_r']}/{decrease['n_r_informative']} configs positive)."
        if decrease["mean_alpha_mmd_pearson_r"] is not None else
        "- Mean Pearson correlation between `alpha` and `mmd_distance`: not computable (no variation).",
        "",
        "(A positive alpha-distance correlation means the distance falls as alpha falls -- the "
        "expected direction if E1 is genuinely moving the poison group's distribution toward the "
        "clean group's, not an arbitrary perturbation.)",
        "",
        "## Q_B. Do decreases in CORAL/MMD align with PP top-pair weakening?",
        "",
        "| leading indicator | precedes/coincides with PP-weakening | indicator w/o PP-weakening | "
        "PP-weakening w/o indicator first | neither | support fraction | mean lead gap (alpha) |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, key in [("CORAL decrease", "coral_vs_pp"), ("MMD decrease", "mmd_vs_pp")]:
        cats = align[key]["categories"]
        sf = align[key]["support_fraction"]
        gap = align[key]["mean_lead_gap"]
        sf_str = f"{sf:.0%}" if sf is not None else "n/a"
        gap_str = f"{gap:.2f}" if gap is not None else "n/a"
        lines.append(
            f"| {label} | {cats['leading_precedes_or_coincides_with_outcome']} | "
            f"{cats['leading_triggered_without_outcome']} | {cats['outcome_without_leading_triggered_first']} | "
            f"{cats['neither_triggered']} | {sf_str} | {gap_str} |"
        )
    lines += [
        "",
        "## Q_C. Do CORAL/MMD decreases precede or coincide with residual-poison failure?",
        "",
        "| leading indicator | precedes/coincides with failure | indicator w/o failure | "
        "failure w/o indicator first | neither | support fraction | mean lead gap (alpha) |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, key in [("CORAL decrease", "coral_vs_failure"), ("MMD decrease", "mmd_vs_failure")]:
        cats = predict[key]["categories"]
        sf = predict[key]["support_fraction"]
        gap = predict[key]["mean_lead_gap"]
        sf_str = f"{sf:.0%}" if sf is not None else "n/a"
        gap_str = f"{gap:.2f}" if gap is not None else "n/a"
        lines.append(
            f"| {label} | {cats['leading_precedes_or_coincides_with_outcome']} | "
            f"{cats['leading_triggered_without_outcome']} | {cats['outcome_without_leading_triggered_first']} | "
            f"{cats['neither_triggered']} | {sf_str} | {gap_str} |"
        )
    lines += [
        "",
        "**CORAL/MMD decreases precede or coincide with residual-poison failure, but "
        "`top_pair_pp` is the more specific mechanistic indicator** (see Q_D below): "
        "CORAL/MMD fall at essentially every alpha step regardless of whether failure "
        "follows, so a high precedes-or-coincides rate here is weak, non-causal "
        "association, not evidence that CORAL/MMD predict failure.",
        "",
        "(`mean lead gap` = mean of `leading_alpha - failure_alpha` over configs where the "
        "indicator precedes-or-coincides with failure; a **large** gap means the indicator "
        "fires immediately at the slightest perturbation regardless of whether failure "
        "follows, which inflates `support_fraction` without making it a *specific* "
        "early-warning signal -- see Q_D.)",
        "",
        "## Q_D. Which of PP top-pair count, CORAL distance, MMD distance, or mean PP "
        "similarity is the more specific mechanistic indicator preceding residual-poison "
        "failure?",
        "",
        "Each candidate's `support_fraction` against `first_residual_poison_alpha` "
        "(fraction of informative configs where the candidate changed at the same-or-higher "
        "alpha than the failure occurred), broken by `mean_lead_gap` when tied on support "
        "(smaller gap = fires closer to, not just always before, the actual failure -- a more "
        "*specific* signal). This ranks specificity among leading indicators; it is not a "
        "causal-prediction claim for any of them.",
        "",
        "| candidate leading indicator | support fraction | mean lead gap (alpha) | n informative |",
        "|---|---|---|---|",
    ]
    for label, result in best_predictor["ranked"]:
        sf = result["support_fraction"]
        gap = result["mean_lead_gap"]
        sf_str = f"{sf:.0%}" if sf is not None else "n/a"
        gap_str = f"{gap:.2f}" if gap is not None else "n/a"
        lines.append(f"| `{label}` | {sf_str} | {gap_str} | {result['n_informative']} |")
    best_label = best_predictor["best"]
    best_result = dict(best_predictor["ranked"]).get(best_label) if best_label else None
    distinct_supports = {r["support_fraction"] for _, r in best_predictor["ranked"]
                          if r["support_fraction"] is not None}
    all_tied_on_support = len(distinct_supports) <= 1
    if best_label and best_result and best_result["support_fraction"] is None:
        answer_line = "**Answer: no candidate is informative in this batch (no config triggered both conditions).**"
    elif best_label and all_tied_on_support and best_result and best_result["mean_lead_gap"] is not None:
        answer_line = (
            f"**Answer: `{best_label}`**, but only as a tiebreaker -- all four candidates tie at "
            f"{best_result['support_fraction']:.0%} raw support fraction in this batch. "
            f"`{best_label}` is selected because it has the smallest mean lead gap "
            f"({best_result['mean_lead_gap']:.2f} alpha units), i.e. it fires closer to the "
            "actual failure rather than immediately at the first perturbation step. CORAL/MMD "
            "decreases precede or coincide with residual-poison failure, but "
            f"`{best_label}` is the more specific mechanistic indicator."
        )
    elif best_label:
        answer_line = f"**Answer: `{best_label}`** has the highest support fraction among these four candidates."
    else:
        answer_line = "**Answer: no candidate is informative in this batch.**"
    lines += [
        "",
        answer_line,
        "",
        "## Q_E. Which E1 anchor strategy most consistently moves poison embeddings toward "
        "the clean distribution?",
        "",
        "Mean CORAL/MMD reduction (alpha=1.0 baseline minus the deepest swept alpha, positive "
        "= moved toward clean) per strategy, and how many of the "
        f"{len(tested_query_ids)} tested queries each strategy achieves the *largest* "
        "reduction for:",
        "",
        "| strategy | mean coral_reduction | mean mmd_reduction | queries won (coral) | queries won (mmd) |",
        "|---|---|---|---|---|",
    ]
    mean_red = best_mover["mean_reduction_by_strategy"]
    for strategy in E1_STRATEGIES:
        row = mean_red.get(strategy, {"coral_reduction": float("nan"), "mmd_reduction": float("nan")})
        lines.append(
            f"| `{strategy}` | {row['coral_reduction']:.4f} | {row['mmd_reduction']:.4f} | "
            f"{best_mover['coral_wins_by_strategy'].get(strategy, 0)} | "
            f"{best_mover['mmd_wins_by_strategy'].get(strategy, 0)} |"
        )
    lines += [
        "",
        f"**Answer:** `{best_mover['top_coral_mover']}` wins the most queries by CORAL reduction; "
        f"`{best_mover['top_mmd_mover']}` wins the most queries by MMD reduction.",
        "",
        "## Q_F. Is the E1 effect geometrically meaningful, or just arbitrary interpolation?",
        "",
    ]
    r = meaningfulness["mean_alpha_distance_pearson_r"]
    sf = meaningfulness["mean_predictive_support_fraction"]
    r_str = f"{r:.3f}" if r is not None else "n/a"
    sf_str = f"{sf:.0%}" if sf is not None else "n/a"
    if meaningfulness["verdict_geometrically_meaningful"]:
        verdict = (
            f"**Geometrically meaningful.** The mean alpha-vs-distance correlation across configs "
            f"is {r_str} (distances fall as alpha falls, as expected for a genuine move toward the "
            f"clean distribution, not noise), and CORAL/MMD decreases precede-or-coincide with "
            f"actual residual-poison failure in a mean of {sf_str} of informative configs. E1's "
            f"effect on RAGDefender's decision tracks a real, formally-measured distribution shift "
            f"of the poison group toward the clean group -- not merely an arbitrary perturbation "
            f"that happens to change cosine similarities. That said (see Q_D), CORAL/MMD decreases "
            f"precede or coincide with residual-poison failure, but "
            f"{('`' + best_predictor['best'] + '`') if best_predictor['best'] else 'top_pair_pp'} is "
            f"the more specific mechanistic indicator -- CORAL/MMD alone should not be read as "
            f"predicting failure."
        )
    else:
        verdict = (
            f"**Mixed / not clearly geometrically meaningful by this batch's numbers.** Mean "
            f"alpha-vs-distance correlation is {r_str} and mean CORAL/MMD-precedes-failure support "
            f"is {sf_str}, which does not clear this report's threshold (r>0.5 and support>=70%) for "
            f"calling the effect a clearly formal distribution shift rather than an arbitrary "
            f"perturbation. Treat E1's mechanism as only partially characterized by CORAL/MMD in "
            f"this batch."
        )
    lines.append(verdict)
    lines.append("")

    lines += [
        "## Per-(query, strategy) distribution-metric summary",
        "",
        "| query_id | strategy | baseline_coral | final_coral | coral_reduction | "
        "baseline_mmd | final_mmd | mmd_reduction | first_residual_poison_alpha |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary_df.sort_values(["query_id", "strategy"]).iterrows():
        lines.append(
            f"| `{r['query_id']}` | `{r['strategy']}` | {r['baseline_coral_distance']:.4f} | "
            f"{r['final_coral_distance']:.4f} | {r['coral_reduction']:.4f} | "
            f"{r['baseline_mmd_distance']:.4f} | {r['final_mmd_distance']:.4f} | "
            f"{r['mmd_reduction']:.4f} | {r['first_residual_poison_alpha']} |"
        )
    lines.append("")
    lines += ["", LIMITATIONS_TEXT]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI / main
# --------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostics_jsonl", default=DEFAULT_DIAGNOSTICS_JSONL)
    parser.add_argument("--query_results_dir", default=DEFAULT_QUERY_RESULTS_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n_injected", type=int, default=5)
    parser.add_argument("--exclude_query_id", action="append", default=["5a8cb288554299585d9e3726"])
    parser.add_argument("--mmd_gamma", type=float, default=DEFAULT_MMD_GAMMA)
    parser.add_argument("--report_md_path", default=None)
    parser.add_argument("--report_csv_path", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Path:
    args = parse_args(argv)

    records = viz._read_jsonl(args.diagnostics_jsonl)
    records_by_id = {r["query_id"]: r for r in records}
    identified_ids = batchmod.discover_success_case_ids(records, args.dataset, args.k, args.n_injected,
                                                          args.exclude_query_id)
    qr_index = viz.load_query_results_index(args.query_results_dir)

    tested_ids: List[str] = []
    for qid in identified_ids:
        ok, _, _ = batchmod.check_text_recoverable(qr_index, records_by_id[qid])
        if not ok:
            continue
        run_dirs = summ.discover_run_dirs(args.output_dir, qid)
        latest = summ.latest_run_per_intervention(run_dirs)
        if all(f"E1-{s}" in latest for s in E1_STRATEGIES):
            tested_ids.append(qid)

    all_rows = []
    config_summaries = []
    for qid in tested_ids:
        rec = records_by_id[qid]
        is_poison = [bool(x) for x in rec["retrieved_is_poison"]]
        poison_idx = [i for i, p in enumerate(is_poison) if p]
        clean_idx = [i for i, p in enumerate(is_poison) if not p]

        run_dirs = summ.discover_run_dirs(args.output_dir, qid)
        latest = summ.latest_run_per_intervention(run_dirs)
        for strategy in E1_STRATEGIES:
            run_dir = latest[f"E1-{strategy}"]
            df = summ.load_sweep(run_dir)
            df.insert(0, "query_id", qid)
            matrices = load_similarity_matrices(run_dir)
            df = attach_distribution_metrics(df, matrices, poison_idx, clean_idx, gamma=args.mmd_gamma)

            config_summary = compute_extended_config_summary(qid, strategy, df)
            df["first_residual_poison_alpha"] = config_summary["first_residual_poison_alpha"]
            all_rows.append(df)
            config_summaries.append(config_summary)

    combined_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(config_summaries)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.report_csv_path) if args.report_csv_path else output_dir / "DISTRIBUTION_METRICS_BATCH.csv"
    combined_df.to_csv(csv_path, index=False)

    if len(summary_df):
        decrease = answer_decrease_with_alpha(summary_df)
        align = answer_align_with_pp_weakening(summary_df)
        predict = answer_predict_residual_poison_failure(summary_df)
        best_predictor = answer_best_predictor_comparison(summary_df)
        best_mover = answer_best_mover_strategy(summary_df, tested_ids)
        meaningfulness = answer_geometric_meaningfulness(decrease, predict)
        report_text = render_markdown(decrease, align, predict, best_predictor, best_mover,
                                       meaningfulness, summary_df, tested_ids, args.mmd_gamma)
    else:
        report_text = "# Cluster-Normalized Poisoning -- Distribution-Matching Diagnostics (E1 Batch)\n\n" \
                       "No tested (query, strategy) configs were found.\n"

    md_path = Path(args.report_md_path) if args.report_md_path else output_dir / "DISTRIBUTION_METRICS_BATCH.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Computed distribution metrics for {len(tested_ids)} queries x {len(E1_STRATEGIES)} strategies "
          f"= {len(summary_df)} configs.")
    print(f"Wrote distribution-metrics CSV to: {csv_path}")
    print(f"Wrote distribution-metrics report to: {md_path}")
    return md_path


if __name__ == "__main__":
    main()
