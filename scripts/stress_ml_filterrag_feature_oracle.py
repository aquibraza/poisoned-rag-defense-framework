#!/usr/bin/env python3
"""ML-FilterRAG-top-k **feature-space oracle stress test**.

Asks a narrow, offline question: *if a poisoned passage's four
ML-FilterRAG-top-k classifier features (`freq_density_score`,
`matched_freq_sum`, `perplexity`, `slm_answer_logprob` -- or whatever
`feature_names` the loaded classifier artifact actually stores) were
linearly interpolated toward a "clean-looking" feature vector, at what
point does the trained classifier stop flagging it as poison?*

This is a **feature-space oracle**, not an attack: `z_prime = alpha *
z_poison + (1 - alpha) * z_target` is applied directly to already-computed
feature *numbers* in an existing `features.csv`. It never asks "does some
real poisoned passage *text* exist whose extracted features equal
`z_prime`?" -- see the two explicit limitations below (also restated in
`FEATURE_ORACLE_REPORT.md`).

**Hard constraints (verified by `tests/test_stress_ml_filterrag_feature_oracle.py`):**
- Never imports/modifies `defense/ml_filterrag.py`, `defense/filterrag.py`,
  `defense/cluster_normalized_poisoning.py`, or any RAGDefender module --
  this script only *reads* `MLFilterRAGClassifier.load()`'s already-trained
  artifact and an already-written feature CSV.
- No GPT/API import, no `llm.query()` call, anywhere in this file.
- No retrieval is rerun (no BEIR corpus, no embedder, no `Attacker`).
- No passage text is read, generated, or rewritten -- this script never
  touches the CSV's `slm_answer` column or any text field; it operates
  purely on the numeric `feature_names` columns and the `is_poison`/`k`/
  `query_id`/`split` bookkeeping columns.

**Method** (see module docstring sections below for each target strategy):
For every `(strategy, alpha)` combination in the sweep, every row with
`is_poison == True` has its feature vector replaced by
`z_prime = alpha * z_poison + (1 - alpha) * z_target` (clean rows are
*never* modified, for any alpha/strategy), the trained classifier
re-predicts every row's label from `z_prime`, and detection-quality
metrics (poison recall, clean false-positive rate, residual poison
count/fraction, feature displacement) are recomputed from those fresh
predictions.

Usage:
    python scripts/stress_ml_filterrag_feature_oracle.py \\
        --features_csv results/diagnostics/ml_filterrag_dataset_hotpotqa_50q/features.csv \\
        --model_path models/ml_filterrag/hotpotqa_50q_mlfilterrag_topk_rf.joblib \\
        --out_dir results/diagnostics/ml_filterrag_feature_oracle
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from defense.ml_filterrag import DEFAULT_FEATURE_NAMES, DEFAULT_THRESHOLD, MLFilterRAGClassifier  # noqa: E402

DEFAULT_FEATURES_CSV = "results/diagnostics/ml_filterrag_dataset_hotpotqa_50q/features.csv"
DEFAULT_MODEL_PATH = "models/ml_filterrag/hotpotqa_50q_mlfilterrag_topk_rf.joblib"
DEFAULT_OUT_DIR = "results/diagnostics/ml_filterrag_feature_oracle"

# Exactly the sweep requested: 1.0 down to 0.0 in steps of 0.1, in this order
# (order matters for "first alpha at which recall drops below X" -- see
# first_break_alphas()).
DEFAULT_ALPHA_SWEEP: Tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)

CLEAN_CENTROID = "clean_centroid"
SAME_K_CLEAN_CENTROID = "same_k_clean_centroid"
SAME_QUERY_CLEAN_CENTROID = "same_query_clean_centroid"
NEAREST_CLEAN_BIJECTION = "nearest_clean_bijection"
ALL_STRATEGIES: Tuple[str, ...] = (
    CLEAN_CENTROID,
    SAME_K_CLEAN_CENTROID,
    SAME_QUERY_CLEAN_CENTROID,
    NEAREST_CLEAN_BIJECTION,
)

DEFAULT_RECALL_BREAK_THRESHOLDS: Tuple[float, ...] = (0.90, 0.80, 0.50)

# nearest_clean_bijection: groups (poison_local, clean_pool) with both sides
# no larger than this are solved by brute-force itertools.permutations (an
# exact minimum-total-distance bijection when sizes are equal), mirroring
# defense/cluster_normalized_poisoning.py's own brute-force bound for its
# `nearest_bijection` anchor strategy -- 8! = 40320 is comfortably tractable.
# Larger/unequal groups fall back to deterministic greedy nearest-neighbor
# matching (see `_match_group_nearest`).
BRUTE_FORCE_MAX_GROUP_SIZE = 8


def _safe_div(numerator, denominator) -> Optional[float]:
    return (numerator / denominator) if denominator else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features_dataframe(path: str):
    """Read `features.csv` and normalize `is_poison` to a real bool column
    (CSV round-trips it as the strings "True"/"False" unless pandas already
    inferred a bool dtype) -- mirrors
    `scripts/train_ml_filterrag.py::load_rows()`'s exact normalization."""
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(path)
    if "is_poison" not in df.columns:
        raise ValueError(f"--features_csv {path!r} has no 'is_poison' column -- not a valid ML-FilterRAG feature CSV.")
    if df["is_poison"].dtype != bool:
        df["is_poison"] = df["is_poison"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
    return df


def resolve_feature_names(classifier: MLFilterRAGClassifier, override: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    """Default: exactly `classifier.feature_names` (the artifact's own
    trained-on feature order) -- per the task's "use only the classifier's
    artifact.feature_names by default". `override`, if given, is used
    verbatim instead (an explicit, reportable deviation from the artifact)."""
    if override:
        return tuple(override)
    return tuple(classifier.feature_names)


def feature_matrix(df, feature_names: Sequence[str]):
    """Fixed-column-order float matrix, mirroring
    `defense.ml_filterrag.features_to_matrix()` (never imported directly,
    since that function only accepts a list-of-dicts, not a DataFrame) --
    raises `KeyError` immediately if a feature_names entry isn't a CSV
    column, never a silent 0.0."""
    import numpy as np  # noqa: PLC0415

    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise KeyError(f"--features_csv is missing required feature column(s) {missing} for feature_names={list(feature_names)!r}")
    return df[list(feature_names)].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Target-vector strategies
# ---------------------------------------------------------------------------

def _group_key_series(df, column: str):
    """Return `df[column]` if present, else a constant single-group key
    (`0` for every row) with a printed note -- lets every strategy degrade
    gracefully to "treat the whole dataset as one group" on a synthetic/
    minimal CSV that lacks that column, instead of crashing."""
    if column in df.columns:
        return df[column], True
    print(f"[stress_ml_filterrag_feature_oracle] WARNING: no {column!r} column in --features_csv; "
          f"treating the whole dataset as a single group for strategies that use it.")
    import pandas as pd  # noqa: PLC0415

    return pd.Series([0] * len(df), index=df.index), False


def build_targets_clean_centroid(df, X, feature_names: Sequence[str]) -> Tuple["object", Dict]:
    """Strategy 1: `clean_centroid` -- every poison row's target is the
    single global centroid (mean feature vector) of every clean
    (`is_poison == False`) row in the CSV."""
    import numpy as np  # noqa: PLC0415

    is_clean = ~df["is_poison"].to_numpy(dtype=bool)
    if not is_clean.any():
        raise ValueError("clean_centroid strategy requires at least one is_poison==False row in --features_csv.")
    centroid = X[is_clean].mean(axis=0)
    targets = np.tile(centroid, (len(df), 1))
    meta = {
        "strategy": CLEAN_CENTROID,
        "global_clean_centroid": {name: float(v) for name, v in zip(feature_names, centroid)},
        "n_clean_rows_used": int(is_clean.sum()),
    }
    return targets, meta


def build_targets_same_k_clean_centroid(df, X, feature_names: Sequence[str]) -> Tuple["object", Dict]:
    """Strategy 2: `same_k_clean_centroid` -- target is the clean centroid
    computed only from rows sharing this row's `k` value. Falls back to the
    global clean centroid (with an explicit, counted warning) for any `k`
    group that itself has zero clean rows -- never crashes, never silently
    reuses an unrelated `k` group's centroid."""
    import numpy as np  # noqa: PLC0415

    is_clean = df["is_poison"].to_numpy(dtype=bool) == False  # noqa: E712 -- explicit bool comparison for clarity
    k_series, has_k_col = _group_key_series(df, "k")
    global_centroid, global_meta = build_targets_clean_centroid(df, X, feature_names)
    global_centroid_vec = global_centroid[0]

    targets = np.zeros_like(X)
    centroids_by_k: Dict = {}
    fallback_k_values: List = []
    for k_value in sorted(k_series.unique(), key=lambda v: (str(type(v)), v)):
        row_mask = (k_series == k_value).to_numpy()
        clean_mask = row_mask & is_clean
        if clean_mask.any():
            centroid = X[clean_mask].mean(axis=0)
        else:
            centroid = global_centroid_vec
            fallback_k_values.append(k_value)
        centroids_by_k[k_value] = {name: float(v) for name, v in zip(feature_names, centroid)}
        targets[row_mask] = centroid
    meta = {
        "strategy": SAME_K_CLEAN_CENTROID,
        "has_k_column": has_k_col,
        "centroids_by_k": {str(k): v for k, v in centroids_by_k.items()},
        "k_values_with_no_clean_rows_fell_back_to_global": [str(k) for k in fallback_k_values],
    }
    return targets, meta


def build_targets_same_query_clean_centroid(df, X, feature_names: Sequence[str]) -> Tuple["object", Dict]:
    """Strategy 3: `same_query_clean_centroid` -- target is the clean
    centroid within this row's `query_id`, if that `query_id` has at least
    one clean row; otherwise falls back to that row's `same_k_clean_centroid`
    target (which itself falls back to the global clean centroid if even
    that `k` group has zero clean rows). If `query_id` is not a column at
    all, every row falls back to `same_k_clean_centroid` outright."""
    import numpy as np  # noqa: PLC0415

    same_k_targets, same_k_meta = build_targets_same_k_clean_centroid(df, X, feature_names)

    if "query_id" not in df.columns:
        print("[stress_ml_filterrag_feature_oracle] WARNING: no 'query_id' column in --features_csv; "
              "same_query_clean_centroid falls back to same_k_clean_centroid for every row.")
        meta = {
            "strategy": SAME_QUERY_CLEAN_CENTROID,
            "has_query_id_column": False,
            "fallback_reason": "no query_id column present",
            "n_rows_fell_back_to_same_k": int(len(df)),
        }
        return same_k_targets, meta

    is_clean = df["is_poison"].to_numpy(dtype=bool) == False  # noqa: E712
    query_series = df["query_id"]

    targets = np.array(same_k_targets, copy=True)
    centroids_by_query: Dict = {}
    fell_back_query_ids: List = []
    for query_id in sorted(query_series.dropna().unique(), key=str):
        row_mask = (query_series == query_id).to_numpy()
        clean_mask = row_mask & is_clean
        if clean_mask.any():
            centroid = X[clean_mask].mean(axis=0)
            targets[row_mask] = centroid
            centroids_by_query[query_id] = {name: float(v) for name, v in zip(feature_names, centroid)}
        else:
            # Leave targets[row_mask] as same_k_targets (already assigned
            # above) -- this *is* the documented fallback.
            fell_back_query_ids.append(query_id)

    meta = {
        "strategy": SAME_QUERY_CLEAN_CENTROID,
        "has_query_id_column": True,
        "centroids_by_query_id": centroids_by_query,
        "query_ids_with_no_clean_rows_fell_back_to_same_k": [str(q) for q in fell_back_query_ids],
        "n_query_ids_fell_back": len(fell_back_query_ids),
        "same_k_meta": same_k_meta,
    }
    return targets, meta


def _pairwise_l2(a, b):
    import numpy as np  # noqa: PLC0415

    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def _brute_force_match(poison_positions: Sequence[int], clean_positions: Sequence[int], X) -> Dict[int, int]:
    """Exact minimum-total-L2-distance bijection between
    `poison_positions` and `clean_positions` (equal length, both
    `<= BRUTE_FORCE_MAX_GROUP_SIZE`) via `itertools.permutations` -- mirrors
    `defense/cluster_normalized_poisoning.py::_brute_force_best_permutation`'s
    own brute-force pattern (never imported from there; independently
    reimplemented here for this script's own row-pairing use case). Ties
    are broken deterministically by `itertools.permutations`' own
    (lexicographic, index-order) enumeration order: the first
    minimum-cost permutation encountered wins."""
    n = len(poison_positions)
    dist = _pairwise_l2(X[list(poison_positions)], X[list(clean_positions)])
    best_perm = None
    best_cost = math.inf
    for perm in itertools.permutations(range(n)):
        cost = sum(dist[i, perm[i]] for i in range(n))
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return {poison_positions[i]: clean_positions[best_perm[i]] for i in range(n)}


def _match_group_nearest(poison_positions: Sequence[int], clean_positions: Sequence[int], X) -> Dict[int, int]:
    """Pair every poison row position with a clean row position, nearby in
    `X`'s feature space:

    - If both sides are non-empty and small enough
      (`<= BRUTE_FORCE_MAX_GROUP_SIZE`) and equal in size, solve the exact
      minimum-total-distance bijection by brute force (`_brute_force_match`).
    - Otherwise, deterministic greedy nearest-neighbor matching: process
      poison rows in ascending row-position order; for each, pick the
      closest still-"available" clean row (ties broken by the lowest clean
      row position -- a fixed, deterministic tie-breaker), then remove that
      clean row from the available pool. If the available pool empties
      before every poison row is matched (more poison rows than clean rows
      in this group), it is refilled with the *full* clean pool and
      matching continues (so every poison row always gets a real nearby
      clean row, never `None`, at the cost of some clean rows being reused).
    """
    import numpy as np  # noqa: PLC0415

    n_p, n_c = len(poison_positions), len(clean_positions)
    if n_p == 0:
        return {}
    if n_c == 0:
        raise ValueError("_match_group_nearest requires at least one clean row position to pair against.")

    if n_p == n_c and n_p <= BRUTE_FORCE_MAX_GROUP_SIZE:
        return _brute_force_match(poison_positions, clean_positions, X)

    ordered_poison = sorted(poison_positions)
    clean_pool = sorted(clean_positions)
    available = list(clean_pool)
    assignment: Dict[int, int] = {}
    for p_pos in ordered_poison:
        if not available:
            available = list(clean_pool)
        dists = np.array([np.linalg.norm(X[p_pos] - X[c_pos]) for c_pos in available])
        best_idx = int(np.argmin(dists))
        # np.argmin already returns the first (lowest-position, since
        # `available` is sorted ascending) minimum on ties.
        chosen = available.pop(best_idx)
        assignment[p_pos] = chosen
    return assignment


def build_targets_nearest_clean_bijection(df, X, feature_names: Sequence[str], seed: int = 12) -> Tuple["object", Dict]:
    """Strategy 4: `nearest_clean_bijection` -- each poison row is paired
    with a nearby clean row in feature space, and that clean row's own
    feature vector is the target (not a centroid).

    Grouping/escalation (documented design choice -- the task spec does not
    pin down a grouping scope): rows are grouped by `k` (falling back to a
    single whole-dataset group if no `k` column exists). If a `k` group has
    zero clean rows locally, its poison rows escalate to the *global* clean
    pool instead (mirrors the same escalation idea used by
    `same_k_clean_centroid`/`same_query_clean_centroid`, applied here to a
    pool of candidate clean *rows* rather than a single centroid). Within
    each `(poison rows, resolved clean pool)` pairing, matching itself is
    resolved by `_match_group_nearest()` (brute-force for small equal-size
    groups, deterministic greedy otherwise). `seed` is accepted for
    interface/reporting symmetry with this repo's other seeded splits
    (default `12`, matching `defense.ml_filterrag.query_level_train_test_split`'s
    default) but does not affect the outcome: every tie is broken by a
    fixed row-position order, not randomness.
    """
    import numpy as np  # noqa: PLC0415

    is_poison = df["is_poison"].to_numpy(dtype=bool)
    is_clean = ~is_poison
    if not is_clean.any():
        raise ValueError("nearest_clean_bijection strategy requires at least one is_poison==False row in --features_csv.")
    global_clean_positions = list(np.nonzero(is_clean)[0])

    k_series, has_k_col = _group_key_series(df, "k")

    targets = np.array(X, copy=True)  # clean rows' "targets" are never read; poison rows get overwritten below
    escalated_k_values: List = []
    group_info: List[Dict] = []

    for k_value in sorted(k_series.unique(), key=lambda v: (str(type(v)), v)):
        row_mask = (k_series == k_value).to_numpy()
        poison_positions = list(np.nonzero(row_mask & is_poison)[0])
        if not poison_positions:
            continue
        local_clean_positions = list(np.nonzero(row_mask & is_clean)[0])
        if local_clean_positions:
            clean_pool = local_clean_positions
            escalated = False
        else:
            clean_pool = global_clean_positions
            escalated = True
            escalated_k_values.append(k_value)

        assignment = _match_group_nearest(poison_positions, clean_pool, X)
        for p_pos, c_pos in assignment.items():
            targets[p_pos] = X[c_pos]
        group_info.append({
            "k": str(k_value),
            "n_poison": len(poison_positions),
            "n_clean_pool": len(clean_pool),
            "escalated_to_global_clean_pool": escalated,
            "matched_via_brute_force": len(poison_positions) == len(clean_pool) and len(poison_positions) <= BRUTE_FORCE_MAX_GROUP_SIZE,
        })

    meta = {
        "strategy": NEAREST_CLEAN_BIJECTION,
        "seed": seed,
        "has_k_column": has_k_col,
        "brute_force_max_group_size": BRUTE_FORCE_MAX_GROUP_SIZE,
        "k_values_escalated_to_global_clean_pool": [str(k) for k in escalated_k_values],
        "groups": group_info,
    }
    return targets, meta


def build_targets(strategy: str, df, X, feature_names: Sequence[str], seed: int = 12) -> Tuple["object", Dict]:
    if strategy == CLEAN_CENTROID:
        return build_targets_clean_centroid(df, X, feature_names)
    if strategy == SAME_K_CLEAN_CENTROID:
        return build_targets_same_k_clean_centroid(df, X, feature_names)
    if strategy == SAME_QUERY_CLEAN_CENTROID:
        return build_targets_same_query_clean_centroid(df, X, feature_names)
    if strategy == NEAREST_CLEAN_BIJECTION:
        return build_targets_nearest_clean_bijection(df, X, feature_names, seed=seed)
    raise ValueError(f"Unknown strategy {strategy!r}; expected one of {ALL_STRATEGIES}")


# ---------------------------------------------------------------------------
# Interpolation + classification
# ---------------------------------------------------------------------------

def interpolate_poison_features(X, targets, is_poison, alpha: float):
    """`z_prime = alpha * z_poison + (1 - alpha) * z_target` for every
    `is_poison == True` row; every other row is returned byte-for-byte
    unchanged (clean rows are never modified, for any alpha/strategy)."""
    modified = X.copy()
    if is_poison.any():
        modified[is_poison] = alpha * X[is_poison] + (1.0 - alpha) * targets[is_poison]
    return modified


def classify(classifier: MLFilterRAGClassifier, X, threshold: float):
    proba = classifier.predict_proba(X)
    pred = (proba >= threshold).astype(int)
    return pred, proba


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_alpha_metrics(
    *, X_original, X_modified, is_poison, pred, feature_names: Sequence[str],
) -> Dict:
    """Detection-quality + displacement metrics for one (strategy, alpha)
    combination. `poison_recall`/`clean_false_positive_rate`/
    `residual_poison_count`/`residual_poison_fraction` mirror
    `defense/diagnostics.py::build_diagnostic_record()`'s exact
    definitions (poison_recall = removed_poison / retrieved_poison;
    residual_poison_fraction = residual_poison / (residual_poison +
    residual_clean)), applied over this CSV's rows instead of one query's
    retrieved passages."""
    import numpy as np  # noqa: PLC0415

    is_clean = ~is_poison
    n_poison = int(is_poison.sum())
    n_clean = int(is_clean.sum())

    removed_poison = int(pred[is_poison].sum()) if n_poison else 0
    removed_clean = int(pred[is_clean].sum()) if n_clean else 0
    residual_poison_count = n_poison - removed_poison
    residual_clean_count = n_clean - removed_clean

    poison_recall = _safe_div(removed_poison, n_poison)
    clean_fpr = _safe_div(removed_clean, n_clean)
    residual_poison_fraction = _safe_div(residual_poison_count, residual_poison_count + residual_clean_count)

    metrics: Dict = {
        "n_rows": int(len(pred)),
        "n_poison": n_poison,
        "n_clean": n_clean,
        "removed_poison": removed_poison,
        "removed_clean": removed_clean,
        "poison_recall": poison_recall,
        "clean_false_positive_rate": clean_fpr,
        "residual_poison_count": residual_poison_count,
        "residual_clean_count": residual_clean_count,
        "residual_poison_fraction": residual_poison_fraction,
    }

    if n_poison:
        displacement = np.linalg.norm(X_modified[is_poison] - X_original[is_poison], axis=1)
        metrics["mean_poison_l2_displacement"] = float(displacement.mean())
        metrics["max_poison_l2_displacement"] = float(displacement.max())
        abs_change = np.abs(X_modified[is_poison] - X_original[is_poison])
        for i, name in enumerate(feature_names):
            metrics[f"mean_abs_change__{name}"] = float(abs_change[:, i].mean())
    else:
        metrics["mean_poison_l2_displacement"] = None
        metrics["max_poison_l2_displacement"] = None
        for name in feature_names:
            metrics[f"mean_abs_change__{name}"] = None

    return metrics


def first_break_alphas(
    alpha_to_recall: Dict[float, Optional[float]], thresholds: Sequence[float] = DEFAULT_RECALL_BREAK_THRESHOLDS,
) -> Dict[float, Optional[float]]:
    """For each `threshold`, the first alpha (processed in descending order,
    i.e. starting from `alpha=1.0` -- no modification -- down to
    `alpha=0.0`) at which `poison_recall < threshold`. `None` if the recall
    never drops below that threshold anywhere in the sweep."""
    ordered_alphas = sorted(alpha_to_recall.keys(), reverse=True)
    result: Dict[float, Optional[float]] = {t: None for t in thresholds}
    for alpha in ordered_alphas:
        recall = alpha_to_recall[alpha]
        if recall is None:
            continue
        for t in thresholds:
            if result[t] is None and recall < t:
                result[t] = alpha
    return result


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------

def run_sweep(
    df, classifier: MLFilterRAGClassifier, feature_names: Sequence[str], *, threshold: float,
    alphas: Sequence[float] = DEFAULT_ALPHA_SWEEP, strategies: Sequence[str] = ALL_STRATEGIES,
    seed: int = 12, recall_break_thresholds: Sequence[float] = DEFAULT_RECALL_BREAK_THRESHOLDS,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Runs every `(strategy, alpha)` combination; returns
    `(sweep_rows, strategy_meta)`:
    - `sweep_rows`: one dict per `(strategy, alpha)`, ready to write as a
      CSV row (includes `first_break_alpha__{threshold}` columns, computed
      once per strategy after that strategy's full alpha sweep).
    - `strategy_meta`: per-strategy target-construction bookkeeping (from
      `build_targets()`), for `run_config.json`/the report's provenance.
    """
    X = feature_matrix(df, feature_names)
    is_poison = df["is_poison"].to_numpy(dtype=bool)

    sweep_rows: List[Dict] = []
    strategy_meta: Dict[str, Dict] = {}

    for strategy in strategies:
        targets, meta = build_targets(strategy, df, X, feature_names, seed=seed)
        strategy_meta[strategy] = meta

        alpha_to_recall: Dict[float, Optional[float]] = {}
        per_alpha_rows: List[Dict] = []
        for alpha in alphas:
            X_modified = interpolate_poison_features(X, targets, is_poison, alpha)
            pred, proba = classify(classifier, X_modified, threshold)
            metrics = compute_alpha_metrics(
                X_original=X, X_modified=X_modified, is_poison=is_poison,
                pred=pred, feature_names=feature_names,
            )
            alpha_to_recall[alpha] = metrics["poison_recall"]
            row = {"strategy": strategy, "alpha": alpha}
            row.update(metrics)
            per_alpha_rows.append(row)

        breaks = first_break_alphas(alpha_to_recall, recall_break_thresholds)
        for row in per_alpha_rows:
            for t in recall_break_thresholds:
                row[f"first_break_alpha__{t}"] = breaks[t]
        strategy_meta[strategy]["first_break_alphas"] = {str(t): breaks[t] for t in recall_break_thresholds}
        sweep_rows.extend(per_alpha_rows)

    return sweep_rows, strategy_meta


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_sweep_csv(rows: List[Dict], path: str, feature_names: Sequence[str], recall_break_thresholds: Sequence[float]) -> None:
    import csv

    fieldnames = [
        "strategy", "alpha", "n_rows", "n_poison", "n_clean", "removed_poison", "removed_clean",
        "poison_recall", "clean_false_positive_rate", "residual_poison_count", "residual_clean_count",
        "residual_poison_fraction", "mean_poison_l2_displacement", "max_poison_l2_displacement",
    ] + [f"mean_abs_change__{name}" for name in feature_names] + [
        f"first_break_alpha__{t}" for t in recall_break_thresholds
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"Wrote {len(rows)} sweep row(s) to {path}")


def write_run_config(path: str, **config) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Wrote run config to {path}")


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def summarize_best_worst_strategy(strategy_meta: Dict[str, Dict], recall_break_thresholds: Sequence[float]) -> Dict:
    """"Best" (for an attacker) = the strategy whose poison recall breaches
    the *lowest* recall_break_threshold at the *highest* alpha (i.e. the
    smallest feature-space nudge already causes evasion) -- ranked by the
    0.50 threshold's first_break_alpha (highest alpha wins; "never breaks"
    ranks last). "Worst" (for an attacker; most robust defense) is the
    opposite end of that same ranking."""
    key_threshold = min(recall_break_thresholds)  # the strictest/lowest bar an attacker must clear
    ranked = []
    for strategy, meta in strategy_meta.items():
        alpha = meta["first_break_alphas"].get(str(key_threshold))
        # None ("never breaks 0.50") sorts as worse-for-attacker than any real alpha.
        sort_key = (alpha is None, -(alpha if alpha is not None else 0.0))
        ranked.append((sort_key, strategy, alpha))
    ranked.sort(key=lambda t: t[0])
    best = ranked[0]
    worst = ranked[-1]
    return {
        "ranking_criterion": f"first_break_alpha at recall < {key_threshold} (higher alpha = more effective evasion)",
        "best_for_attacker_strategy": best[1],
        "best_for_attacker_first_break_alpha": best[2],
        "worst_for_attacker_strategy": worst[1],
        "worst_for_attacker_first_break_alpha": worst[2],
        "ranking": [{"strategy": s, "first_break_alpha": a} for _, s, a in ranked],
    }


def write_report_md(
    path: str, *, model_path: str, features_csv: str, feature_names: Sequence[str], threshold: float,
    n_rows: int, n_poison: int, n_clean: int, split_counts: Optional[Dict], alphas: Sequence[float],
    strategies: Sequence[str], sweep_rows: List[Dict], strategy_meta: Dict[str, Dict],
    recall_break_thresholds: Sequence[float],
) -> None:
    best_worst = summarize_best_worst_strategy(strategy_meta, recall_break_thresholds)

    lines = [
        "# ML-FilterRAG-top-k Feature-Space Oracle Stress Test",
        "",
        "**Offline, detection-only oracle stress test.** No GPT/API call, no `llm.query()` "
        "call, no retrieval rerun, no passage text read/generated/rewritten anywhere in this "
        "run -- see the Limitations section below.",
        "",
        "## Inputs",
        "",
        f"- Model artifact: `{model_path}`",
        f"- Feature CSV: `{features_csv}`",
        f"- Feature names used (from the classifier artifact unless `--feature_names` was "
        f"passed explicitly): `{list(feature_names)}`",
        f"- Classification threshold used: `{threshold}`",
        f"- Rows: {n_rows} total ({n_poison} poison / {n_clean} clean)",
    ]
    if split_counts is not None:
        lines.append(
            f"- `split` column present: {split_counts} (this sweep evaluates every row "
            "regardless of split, unless `--split_filter` restricted it -- see run_config.json)"
        )
    else:
        lines.append("- No `split` column present in `--features_csv`; train/test status not applicable.")
    lines += [
        f"- Alpha sweep: `{list(alphas)}`",
        f"- Strategies: `{list(strategies)}`",
        f"- Recall break thresholds checked: `{list(recall_break_thresholds)}`",
        "",
        "## Method",
        "",
        "For every `(strategy, alpha)` pair, every `is_poison == True` row's feature vector is "
        "replaced by `z_prime = alpha * z_poison + (1 - alpha) * z_target` (clean rows are never "
        "modified), the trained classifier re-predicts every row's label from `z_prime` at the "
        "threshold above, and detection-quality metrics are recomputed from those fresh "
        "predictions -- `poison_recall`/`clean_false_positive_rate`/`residual_poison_fraction` "
        "use the exact same definitions as `defense/diagnostics.py::build_diagnostic_record()`.",
        "",
        "## Results by strategy and alpha",
        "",
        "| strategy | alpha | poison_recall | clean_fpr | residual_poison_count | "
        "residual_poison_fraction | mean_l2_disp | max_l2_disp |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sweep_rows:
        lines.append(
            f"| {row['strategy']} | {row['alpha']} | {_fmt(row['poison_recall'])} | "
            f"{_fmt(row['clean_false_positive_rate'])} | {row['residual_poison_count']} | "
            f"{_fmt(row['residual_poison_fraction'])} | {_fmt(row['mean_poison_l2_displacement'])} | "
            f"{_fmt(row['max_poison_l2_displacement'])} |"
        )
    lines.append("")

    lines += ["## First-break alpha table", "", "| strategy | " + " | ".join(f"recall < {t}" for t in recall_break_thresholds) + " |",
               "|---|" + "---|" * len(recall_break_thresholds)]
    for strategy in strategies:
        breaks = strategy_meta[strategy]["first_break_alphas"]
        cells = [_fmt(breaks.get(str(t))) if breaks.get(str(t)) is not None else "never" for t in recall_break_thresholds]
        lines.append(f"| {strategy} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "## Best/worst strategy summary",
        "",
        f"- Ranking criterion: {best_worst['ranking_criterion']}",
        f"- Most effective evasion (worst for the defense): `{best_worst['best_for_attacker_strategy']}` "
        f"(first breaks at alpha={_fmt(best_worst['best_for_attacker_first_break_alpha'])})",
        f"- Least effective evasion (most robust defense): `{best_worst['worst_for_attacker_strategy']}` "
        f"(first breaks at alpha={_fmt(best_worst['worst_for_attacker_first_break_alpha'])})",
        "",
        "| rank | strategy | first_break_alpha |",
        "|---|---|---|",
    ]
    for i, entry in enumerate(best_worst["ranking"]):
        lines.append(f"| {i + 1} | {entry['strategy']} | {_fmt(entry['first_break_alpha'])} |")
    lines.append("")

    lines += [
        "## Limitations",
        "",
        "- **Feature-space oracle only, not a text-realizable attack.** `z_prime = alpha * "
        "z_poison + (1 - alpha) * z_target` is applied directly to already-extracted feature "
        "*numbers*; this script never asks (and cannot answer) whether any real poisoned "
        "passage *text* exists whose `freq_density_score`/`matched_freq_sum`/`perplexity`/"
        "`slm_answer_logprob` would actually equal `z_prime` -- some interpolated feature "
        "combinations may be text-unrealizable (e.g. a `slm_answer_logprob` implying a "
        "different SLM answer than the passage's actual `matched_freq_sum`/`freq_density_score` "
        "would produce for that same passage/query pair).",
        "- **ML-FilterRAG-top-k, not the paper's full top-s Algorithm 2.** This repo's harness "
        "retrieves `top_k` directly (no oversized `top-s` candidate pool filtered down to "
        "`top-k`); see `docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md` sections 1, 9, 10. Every "
        "result above should be reported as \"ML-FilterRAG-top-k\", never bare \"ML-FilterRAG\".",
        "- No GPT/API call was made. No `llm.query()` call was made. No retrieval was rerun. No "
        "passage text was read, generated, or rewritten.",
        "",
    ]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote report to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features_csv", default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--feature_names", nargs="+", default=None,
        help="Override feature_names; default: use the loaded classifier artifact's own "
             "feature_names (paper-aligned DEFAULT_FEATURE_NAMES for a paper-aligned artifact).",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Default: the artifact's own threshold_default.")
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHA_SWEEP))
    parser.add_argument("--strategies", nargs="+", default=list(ALL_STRATEGIES), choices=list(ALL_STRATEGIES))
    parser.add_argument("--seed", type=int, default=12, help="Deterministic tie-break/reporting seed (see build_targets_nearest_clean_bijection docstring).")
    parser.add_argument(
        "--recall_break_thresholds", nargs="+", type=float, default=list(DEFAULT_RECALL_BREAK_THRESHOLDS),
    )
    parser.add_argument(
        "--split_filter", default="all", choices=["all", "train", "test"],
        help="Restrict the stress test to rows with this 'split' value, if a 'split' column "
             "exists (default: 'all' rows, train+test combined).",
    )
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    print("[stress_ml_filterrag_feature_oracle] No GPT/API call will be made; no `llm.query()` "
          "call will be made; no retrieval will be rerun; no passage text is read/generated/rewritten.")

    t0 = time.perf_counter()
    df = load_features_dataframe(args.features_csv)

    split_counts = None
    if "split" in df.columns:
        split_counts = df["split"].value_counts().to_dict()
        if args.split_filter != "all":
            df = df[df["split"] == args.split_filter].reset_index(drop=True)
            print(f"[stress_ml_filterrag_feature_oracle] restricted to split=={args.split_filter!r}: {len(df)} row(s)")
    elif args.split_filter != "all":
        raise ValueError("--split_filter was given, but --features_csv has no 'split' column.")

    classifier = MLFilterRAGClassifier.load(args.model_path)
    feature_names = resolve_feature_names(classifier, args.feature_names)
    threshold = classifier.threshold_default if args.threshold is None else args.threshold

    n_poison = int(df["is_poison"].sum())
    n_clean = int(len(df) - n_poison)
    print(
        f"[stress_ml_filterrag_feature_oracle] {len(df)} row(s) ({n_poison} poison / {n_clean} clean); "
        f"feature_names={list(feature_names)}; threshold={threshold}"
    )

    sweep_rows, strategy_meta = run_sweep(
        df, classifier, feature_names, threshold=threshold, alphas=args.alphas, strategies=args.strategies,
        seed=args.seed, recall_break_thresholds=args.recall_break_thresholds,
    )
    print(f"[stress_ml_filterrag_feature_oracle] sweep completed in {time.perf_counter() - t0:.1f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    write_sweep_csv(
        sweep_rows, os.path.join(args.out_dir, "FEATURE_ORACLE_SWEEP.csv"), feature_names, args.recall_break_thresholds,
    )
    write_report_md(
        os.path.join(args.out_dir, "FEATURE_ORACLE_REPORT.md"),
        model_path=args.model_path, features_csv=args.features_csv, feature_names=feature_names,
        threshold=threshold, n_rows=len(df), n_poison=n_poison, n_clean=n_clean, split_counts=split_counts,
        alphas=args.alphas, strategies=args.strategies, sweep_rows=sweep_rows, strategy_meta=strategy_meta,
        recall_break_thresholds=args.recall_break_thresholds,
    )
    write_run_config(
        os.path.join(args.out_dir, "run_config.json"),
        features_csv=os.path.abspath(args.features_csv), model_path=os.path.abspath(args.model_path),
        feature_names=list(feature_names), threshold=threshold, alphas=list(args.alphas),
        strategies=list(args.strategies), seed=args.seed, recall_break_thresholds=list(args.recall_break_thresholds),
        split_filter=args.split_filter, n_rows=len(df), n_poison=n_poison, n_clean=n_clean,
        split_counts=split_counts, strategy_meta=strategy_meta,
        no_gpt_api_calls_made=True, no_live_generation_through_llm_query=True, retrieval_rerun=False,
        status="ML-FilterRAG-top-k feature-space oracle stress test (not a text-realizable attack; "
               "see FEATURE_ORACLE_REPORT.md Limitations)",
    )
    print(f"[stress_ml_filterrag_feature_oracle] wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
