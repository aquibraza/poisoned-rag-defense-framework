#!/usr/bin/env python3
"""Cluster-Normalized Poisoning: batch comparison across all originally
successful RAGDefender cases.

Per the user's batch-experiment request (see chat), and staying inside
`docs/CLUSTER_NORMALIZED_POISONING_EXECUTION_PLAN.md`'s scope: this script

1. identifies every HotpotQA k=10/N=5 query in a diagnostics `.jsonl` where
   baseline RAGDefender **originally succeeded**
   (`removed_poison == N_retrieved_poison and residual_poison_fraction ==
   0.0`), excluding any explicitly-excluded control queries,
2. checks, for each candidate, whether its exact retrieved-passage text is
   recoverable (`recover_pre_defense_texts` returning exactly `k` lines) --
   candidates that are not recoverable are excluded and reported by name
   and reason, never silently dropped or guessed at,
3. aggregates the **already-written** `intervention_sweep.csv` files from
   `scripts/run_cluster_normalized_poisoning.py`'s E1 runs (all four
   anchor strategies) for every recoverable, tested query,
4. answers the batch report's seven questions purely from that aggregated
   table.

This script performs no embedding, no cosine/Stage-1/Stage-2
recomputation, and no GPT/API call of its own -- it only reads
`run_config.json` / `intervention_sweep.csv` files that
`run_cluster_normalized_poisoning.py` already wrote (that script is the one
that does the actual oracle recomputation; see
`docs/CLUSTER_NORMALIZED_POISONING_EXECUTION_PLAN.md`). It also never reads
or modifies any baseline defense/retrieval code or file.

Aggregation semantics (revised): `decision_label` changing from the
alpha=1.0 baseline is **not** always a defense failure -- for a baseline
that over-removes (`over_removal_success`, i.e. it already removed some
clean passages as false positives at alpha=1.0), a transition to
`poison_removal_success` under a transformed alpha is a *clean-FP
improvement*, not a failure. Two alphas are therefore tracked separately
per `(query, strategy)`:

- `first_residual_poison_alpha`: first alpha (descending from 1.0) where
  `removed_poison < N_retrieved_poison`. This is the actual defense-failure
  condition for this oracle stress test (poison survives into the
  post-defense passage set) and is what Q3/Q4/Q5/Q7 are based on.
- `first_label_change_alpha`: first alpha where the absolute
  `decision_label` changes at all from the baseline label, kept only as a
  secondary diagnostic (it also fires on non-failure transitions such as
  `over_removal_success` -> `poison_removal_success`).

Usage:
    python scripts/build_batch_comparison_success_cases.py \\
        --diagnostics_jsonl results/diagnostics/ragdefender_smoke_live_10q/hotpotqa-...-defense-ragdefender_original.jsonl \\
        --query_results_dir results/query_results/ragdefender_smoke_live_10q \\
        --exclude_query_id 5a8cb288554299585d9e3726
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import visualize_ragdefender_clusters as viz  # noqa: E402
import summarize_cluster_normalized_poisoning as summ  # noqa: E402  (decision_label, discover_run_dirs, ...)

DEFAULT_OUTPUT_DIR = os.path.join("results", "diagnostics", "cluster_normalized_poisoning")
DEFAULT_DIAGNOSTICS_JSONL = os.path.join(
    "results", "diagnostics", "ragdefender_smoke_live_10q",
    "hotpotqa-contriever-gpt4-Top10--M10x1-adv-LM_targeted-dot-5-10-defense-ragdefender_original.jsonl",
)
DEFAULT_QUERY_RESULTS_DIR = os.path.join("results", "query_results", "ragdefender_smoke_live_10q")
E1_STRATEGIES = ("rank_aligned", "nearest_bijection", "farthest_bijection", "random")
PP_WEAKENING_ALPHA_THRESHOLD = 0.6
RESIDUAL_POISON_ALPHA_THRESHOLD = 0.5
DEEPEST_ALPHA = 0.3


# --------------------------------------------------------------------------
# Step 1: identify originally-successful cases (pure function; no I/O)
# --------------------------------------------------------------------------

def discover_success_case_ids(records: Sequence[Dict], dataset: str, k: int, n_injected: int,
                               exclude_query_ids: Sequence[str]) -> List[str]:
    """Return every `query_id` in `records` matching `dataset`/`k`/`N_injected`
    where baseline RAGDefender originally removed *all* retrieved poison and
    no poison remained (`removed_poison == N_retrieved_poison and
    residual_poison_fraction == 0.0`), excluding `exclude_query_ids`.
    Order-preserving, first-match-wins if `query_id` repeats."""
    exclude = set(exclude_query_ids)
    out: List[str] = []
    for r in records:
        if r.get("dataset") != dataset or r.get("k") != k or r.get("N_injected") != n_injected:
            continue
        qid = r.get("query_id")
        if qid is None or qid in exclude:
            continue
        n_poison = r.get("N_retrieved_poison")
        if n_poison is None or n_poison <= 0:
            continue
        if r.get("removed_poison") == n_poison and r.get("residual_poison_fraction") == 0.0:
            out.append(qid)
    return out


# --------------------------------------------------------------------------
# Step 2: text-recoverability gate (matches run_cluster_normalized_poisoning.py's
# own gate exactly, so "tested" here means "would not have raised there")
# --------------------------------------------------------------------------

def check_text_recoverable(qr_index: Dict, rec: Dict) -> Tuple[bool, Optional[int], int]:
    qr = qr_index.get(rec["query_id"])
    texts = viz.recover_pre_defense_texts(qr)
    expected_k = len(rec["retrieved_doc_ids"])
    recovered_len = None if texts is None else len(texts)
    return recovered_len == expected_k, recovered_len, expected_k


# --------------------------------------------------------------------------
# Step 3/4: per-(query,strategy) derived summary + the seven questions
# (pure functions operating on already-loaded DataFrames; no I/O)
# --------------------------------------------------------------------------

def compute_config_summary(query_id: str, strategy: str, df: pd.DataFrame) -> Dict:
    """`df` = one (query_id, strategy)'s alpha-sweep rows (as written by
    `run_cluster_normalized_poisoning.py`, with `decision_label` already
    re-derived by `summarize_cluster_normalized_poisoning.load_sweep`).
    Returns the alpha (descending from 1.0) at which each condition first
    triggers, or `None` if it never triggers within the swept alphas.

    `first_residual_poison_alpha` is the actual defense-failure condition
    (`removed_poison < N_retrieved_poison`, an absolute condition -- not
    relative to this config's own baseline). `first_label_change_alpha` is
    relative to this config's alpha=1.0 baseline label and is kept only as
    a secondary diagnostic: it also fires on non-failure transitions, e.g.
    `over_removal_success` -> `poison_removal_success` is a clean-FP
    *improvement*, not a defense failure, but still counts as a label
    change."""
    sub = df.sort_values("alpha", ascending=False).reset_index(drop=True)
    baseline = sub.iloc[0]

    def first_alpha(predicate) -> Optional[float]:
        for _, r in sub.iterrows():
            if predicate(r):
                return float(r["alpha"])
        return None

    return {
        "query_id": query_id,
        "strategy": strategy,
        "baseline_alpha": float(baseline["alpha"]),
        "baseline_decision_label": baseline["decision_label"],
        "baseline_top_pair_pp": int(baseline["top_pair_pp"]),
        "baseline_removed_poison": int(baseline["removed_poison"]),
        "baseline_removed_clean": int(baseline["removed_clean"]),
        "n_retrieved_poison": int(baseline["N_retrieved_poison"]),
        "pp_decreased_alpha": first_alpha(lambda r: r["top_pair_pp"] < baseline["top_pair_pp"]),
        "pc_increased_alpha": first_alpha(lambda r: r["top_pair_pc"] > baseline["top_pair_pc"]),
        "fewer_poison_removed_alpha": first_alpha(lambda r: r["removed_poison"] < baseline["removed_poison"]),
        "clean_removed_increased_alpha": first_alpha(lambda r: r["removed_clean"] > baseline["removed_clean"]),
        "first_residual_poison_alpha": first_alpha(lambda r: r["removed_poison"] < r["N_retrieved_poison"]),
        "first_label_change_alpha": first_alpha(lambda r: r["decision_label"] != baseline["decision_label"]),
        "final_decision_label": sub.iloc[-1]["decision_label"],
        "min_alpha_swept": float(sub["alpha"].min()),
    }


def answer_q1(tested_ids: Sequence[str], identified_ids: Sequence[str],
              excluded: Dict[str, str]) -> Dict:
    return {
        "identified_count": len(identified_ids),
        "tested_count": len(tested_ids),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "tested_query_ids": list(tested_ids),
    }


def _any_all_counts(summary_df: pd.DataFrame, query_ids: Sequence[str], trigger_col: str,
                     threshold: float) -> Tuple[int, int, Dict[str, int]]:
    """For each query, count how many of its 4 strategies trigger `trigger_col`
    (an alpha value or NaN/None) at an alpha `<= threshold`. Returns
    (any_count, all_count, {query_id: n_strategies_triggered})."""
    per_query: Dict[str, int] = {}
    for qid in query_ids:
        rows = summary_df[summary_df["query_id"] == qid]
        n_triggered = sum(
            1 for v in rows[trigger_col] if v is not None and not pd.isna(v) and v <= threshold
        )
        per_query[qid] = n_triggered
    any_count = sum(1 for n in per_query.values() if n >= 1)
    all_count = sum(1 for n in per_query.values() if n == 4)
    return any_count, all_count, per_query


def answer_q2(summary_df: pd.DataFrame, query_ids: Sequence[str]) -> Dict:
    any_count, all_count, per_query = _any_all_counts(
        summary_df, query_ids, "pp_decreased_alpha", PP_WEAKENING_ALPHA_THRESHOLD
    )
    return {"any_strategy_count": any_count, "all_strategies_count": all_count, "per_query": per_query}


def answer_q3(summary_df: pd.DataFrame, query_ids: Sequence[str]) -> Dict:
    """Q3 uses `first_residual_poison_alpha` (actual defense failure --
    poison survives into the post-defense set), not label change."""
    any_count, all_count, per_query = _any_all_counts(
        summary_df, query_ids, "first_residual_poison_alpha", RESIDUAL_POISON_ALPHA_THRESHOLD
    )
    return {"any_strategy_count": any_count, "all_strategies_count": all_count, "per_query": per_query}


def answer_q4(summary_df: pd.DataFrame, query_ids: Sequence[str]) -> Dict:
    """Q4 uses `first_residual_poison_alpha` (actual defense failure), not
    label change."""
    any_le, all_le, per_query_le = _any_all_counts(
        summary_df, query_ids, "first_residual_poison_alpha", DEEPEST_ALPHA
    )
    # Stricter reading: needed the *deepest* swept alpha specifically (did
    # not already fail at any higher alpha in the sweep).
    per_query_exact: Dict[str, int] = {}
    for qid in query_ids:
        rows = summary_df[summary_df["query_id"] == qid]
        n_exact = sum(
            1 for v in rows["first_residual_poison_alpha"]
            if v is not None and not pd.isna(v) and v == DEEPEST_ALPHA
        )
        per_query_exact[qid] = n_exact
    any_exact = sum(1 for n in per_query_exact.values() if n >= 1)
    all_exact = sum(1 for n in per_query_exact.values() if n == 4)
    return {
        "cumulative_le_0_3": {"any_strategy_count": any_le, "all_strategies_count": all_le,
                               "per_query": per_query_le},
        "exactly_at_0_3_only": {"any_strategy_count": any_exact, "all_strategies_count": all_exact,
                                 "per_query": per_query_exact},
    }


def answer_q5(summary_df: pd.DataFrame, query_ids: Sequence[str]) -> Dict:
    """Per query, the strategy (or tied strategies) whose
    `first_residual_poison_alpha` is the *highest* (== causes actual
    residual-poison failure with the least perturbation == "earliest" in
    the descending alpha sweep) wins that query. Tallied across all tested
    queries; queries where no strategy ever causes residual-poison failure
    contribute to `never_broke_count` instead."""
    wins = {s: 0 for s in E1_STRATEGIES}
    never_broke_count = 0
    per_query_winner: Dict[str, List[str]] = {}
    for qid in query_ids:
        rows = summary_df[summary_df["query_id"] == qid]
        broke = rows[rows["first_residual_poison_alpha"].notna()]
        if broke.empty:
            never_broke_count += 1
            per_query_winner[qid] = []
            continue
        max_alpha = broke["first_residual_poison_alpha"].max()
        winners = sorted(broke[broke["first_residual_poison_alpha"] == max_alpha]["strategy"].tolist())
        per_query_winner[qid] = winners
        for s in winners:
            wins[s] += 1
    ranked = sorted(wins.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "wins_by_strategy": wins, "ranked": ranked, "never_broke_count": never_broke_count,
        "per_query_winner": per_query_winner,
        "top_strategy": ranked[0][0] if ranked and ranked[0][1] > 0 else None,
    }


def answer_q6(summary_df: pd.DataFrame) -> Dict:
    """Does a PP top-pair decrease predict/precede *actual residual-poison
    failure* (`first_residual_poison_alpha`), across all tested
    (query, strategy) configs? Four mutually exclusive categories per
    config:

    - `pp_precedes_or_coincides_with_residual_poison_failure`: both
      triggered, and PP decreased at the same-or-higher alpha (i.e. no
      later) than residual-poison failure occurred.
    - `pp_decreased_without_residual_poison_failure`: PP weakened in the
      top-pair graph, but residual-poison failure never occurred within
      the swept alphas -- PP weakening alone was not sufficient.
    - `residual_poison_failure_without_pp_decrease_first`: residual-poison
      failure occurred at an alpha where PP had *not yet* (or never)
      decreased -- evidence against PP-decrease being a leading indicator
      for this config.
    - `neither_triggered`: no effect observed in this config within the
      swept alphas.
    """
    categories = {
        "pp_precedes_or_coincides_with_residual_poison_failure": 0,
        "pp_decreased_without_residual_poison_failure": 0,
        "residual_poison_failure_without_pp_decrease_first": 0,
        "neither_triggered": 0,
    }
    details = []
    for _, r in summary_df.iterrows():
        pp_a = r["pp_decreased_alpha"]
        rp_a = r["first_residual_poison_alpha"]
        pp_triggered = pp_a is not None and not pd.isna(pp_a)
        rp_triggered = rp_a is not None and not pd.isna(rp_a)
        if pp_triggered and rp_triggered:
            cat = "pp_precedes_or_coincides_with_residual_poison_failure" if pp_a >= rp_a else \
                "residual_poison_failure_without_pp_decrease_first"
        elif pp_triggered and not rp_triggered:
            cat = "pp_decreased_without_residual_poison_failure"
        elif rp_triggered and not pp_triggered:
            cat = "residual_poison_failure_without_pp_decrease_first"
        else:
            cat = "neither_triggered"
        categories[cat] += 1
        details.append({"query_id": r["query_id"], "strategy": r["strategy"], "category": cat,
                         "pp_decreased_alpha": pp_a, "first_residual_poison_alpha": rp_a})
    n_informative = categories["pp_precedes_or_coincides_with_residual_poison_failure"] + \
        categories["residual_poison_failure_without_pp_decrease_first"]
    support_fraction = (
        categories["pp_precedes_or_coincides_with_residual_poison_failure"] / n_informative
        if n_informative else None
    )
    return {"categories": categories, "details": details, "support_fraction_of_informative_configs": support_fraction}


def answer_q7(summary_df: pd.DataFrame, query_ids: Sequence[str]) -> Dict:
    """Per query, how many of its 4 strategies show *actual residual-poison
    failure* (`first_residual_poison_alpha`) by alpha<=0.5 -- a
    distribution across queries reveals whether the effect is broadly
    consistent (most queries cluster near the same count) or concentrated
    in one or a few queries (high variance, some at 0)."""
    _, _, per_query = _any_all_counts(summary_df, query_ids, "first_residual_poison_alpha",
                                       RESIDUAL_POISON_ALPHA_THRESHOLD)
    counts = list(per_query.values())
    n = len(counts)
    mean = sum(counts) / n if n else 0.0
    variance = sum((c - mean) ** 2 for c in counts) / n if n else 0.0
    distribution: Dict[int, int] = {}
    for c in counts:
        distribution[c] = distribution.get(c, 0) + 1
    return {
        "per_query_n_strategies_with_residual_poison_failure_by_0_5": per_query,
        "mean_strategies_with_residual_poison_failure": mean,
        "variance": variance,
        "distribution_of_counts": distribution,  # {n_strategies_failing: n_queries}
        "queries_with_zero_strategies_with_residual_poison_failure": sum(1 for c in counts if c == 0),
        "queries_with_all_four_strategies_with_residual_poison_failure": sum(1 for c in counts if c == 4),
    }


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------

LABEL_CHANGE_VS_FAILURE_NOTE = (
    "**Note on label change vs. actual defense failure:** `first_label_change_alpha` "
    "(a secondary diagnostic below) fires on *any* change in the absolute "
    "`decision_label`, including non-failure transitions. In particular, a "
    "transition from `over_removal_success` to `poison_removal_success` is a "
    "**clean-FP improvement, not a defense failure** (the baseline was already "
    "removing an extra clean passage as a false positive; the transformed alpha "
    "stops doing that while still removing all poison). Q3/Q4/Q5/Q7 below use "
    "`first_residual_poison_alpha` instead (`removed_poison < N_retrieved_poison`), "
    "which is the actual defense-failure condition for this oracle stress test and "
    "is unaffected by that kind of improvement."
)

LIMITATIONS_TEXT = """## Limitations (verbatim, per the execution plan section 10)

- This batch is an **oracle embedding-space diagnostic** across multiple
  queries. Every poisoned passage's embedding is transformed directly; no
  natural-language rewrite of any poisoned passage is performed or implied
  for any query in this batch.
- It **does not prove natural-language realizability** for any query. A
  finding that an intervention flips RAGDefender's decision says only that
  the *defense* depends on a geometric assumption that is fragile under a
  controlled representation change for that query -- not that the
  transform is reachable by rewriting the poisoned passage's text under the
  frozen `paraphrase-MiniLM-L6-v2` encoder.
- **Text-space mutation is a later phase**, out of scope for this batch.
- **FilterRAG and ML-FilterRAG comparisons come after** the RAGDefender
  oracle study and are not part of this batch.
- Only **E1 clean-anchor interpolation** is covered by this batch (all four
  anchor strategies). CORAL/MMD/DAN-style distribution-matching
  interventions (candidates B/C/D in the execution plan) are not run here.
- **Alpha values below 0.5 may be geometrically extreme** and must not be
  interpreted as plausible natural-language passage rewrites.
- Two originally-successful candidates were **excluded from this batch**
  because their exact retrieved-passage text could not be recovered from
  `input_prompt_no_defense` without guessing (see the exclusions table
  below) -- this is a data-recoverability limitation of this batch, not a
  finding about those queries' RAGDefender behavior.
- A label change from `over_removal_success` to `poison_removal_success` is
  a clean-FP improvement, not a defense failure -- see the note above Q3.
"""


def render_markdown(q1: Dict, q2: Dict, q3: Dict, q4: Dict, q5: Dict, q6: Dict, q7: Dict,
                     summary_df: pd.DataFrame, dataset: str, k: int, n_injected: int) -> str:
    lines = [
        "# Cluster-Normalized Poisoning -- Batch Comparison (Originally Successful Cases)",
        "",
        f"dataset=`{dataset}`, k={k}, N_injected={n_injected}, intervention=`E1` "
        f"(all four anchor strategies), alphas swept = 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3.",
        "",
        "**No GPT/API calls were made. Baseline retrieval was not rerun. No baseline "
        "defense file was modified.** All claims below are oracle embedding-space "
        "stress-test findings, not evidence of a text-realizable attack -- see "
        "Limitations.",
        "",
        "## Q1. How many originally successful RAGDefender cases were tested?",
        "",
        f"- **{q1['identified_count']}** cases matched the success criterion "
        f"(`removed_poison == N_retrieved_poison and residual_poison_fraction == 0.0`) "
        f"for `dataset={dataset}, k={k}, N_injected={n_injected}`, after excluding the "
        f"known failed control(s).",
        f"- **{q1['excluded_count']}** of those were excluded from this batch because "
        f"their exact passage text could not be recovered (see below).",
        f"- **{q1['tested_count']}** cases were actually tested (all four E1 anchor "
        f"strategies, full alpha sweep, each).",
        "",
        "| query_id | status |",
        "|---|---|",
    ]
    for qid in q1["tested_query_ids"]:
        lines.append(f"| `{qid}` | tested |")
    for qid, reason in q1["excluded"].items():
        lines.append(f"| `{qid}` | **excluded** -- {reason} |")
    lines.append("")

    lines += [
        "## Q2. How many show PP top-pair weakening by alpha <= "
        f"{PP_WEAKENING_ALPHA_THRESHOLD}?",
        "",
        f"- **{q2['any_strategy_count']} / {q1['tested_count']}** tested queries show "
        f"`top_pair_pp` decrease from the alpha=1.0 baseline by alpha "
        f"<= {PP_WEAKENING_ALPHA_THRESHOLD} under **at least one** E1 strategy.",
        f"- **{q2['all_strategies_count']} / {q1['tested_count']}** show it under "
        f"**all four** E1 strategies.",
        "",
        "| query_id | strategies triggering PP-weakening by alpha<=" +
        f"{PP_WEAKENING_ALPHA_THRESHOLD} (out of 4) |",
        "|---|---|",
    ]
    for qid, n in q2["per_query"].items():
        lines.append(f"| `{qid}` | {n} |")
    lines.append("")

    lines += ["", LABEL_CHANGE_VS_FAILURE_NOTE, ""]

    lines += [
        "## Q3. How many show actual residual-poison failure?",
        "",
        "\"Actual residual-poison failure\" = `first_residual_poison_alpha` (first alpha, "
        "descending from 1.0, where `removed_poison < N_retrieved_poison`, i.e. some "
        "retrieved poison survives into the post-defense passage set) occurs by alpha "
        f"<= {RESIDUAL_POISON_ALPHA_THRESHOLD}. This is **not** the same as any "
        "`decision_label` change -- see the note above.",
        "",
        f"- **{q3['any_strategy_count']} / {q1['tested_count']}** tested queries show "
        f"actual residual-poison failure by alpha <= {RESIDUAL_POISON_ALPHA_THRESHOLD} under "
        f"**at least one** E1 strategy.",
        f"- **{q3['all_strategies_count']} / {q1['tested_count']}** show it under "
        f"**all four** E1 strategies.",
        "",
        "| query_id | strategies with residual-poison failure by alpha<=" +
        f"{RESIDUAL_POISON_ALPHA_THRESHOLD} (out of 4) |",
        "|---|---|",
    ]
    for qid, n in q3["per_query"].items():
        lines.append(f"| `{qid}` | {n} |")
    lines.append("")

    lines += [
        f"## Q4. How many require alpha <= {DEEPEST_ALPHA}?",
        "",
        f"Phrased precisely: how many configs **first break** (`first_residual_poison_alpha`) "
        f"only at the deepest tested alpha, alpha={DEEPEST_ALPHA} -- i.e. residual-poison "
        f"failure did **not** already occur at any higher swept alpha (1.0 down to "
        f"{DEEPEST_ALPHA + 0.1:.1f}), and only appears once the sweep reaches its floor. "
        f"This is the headline answer below. A secondary, cumulative "
        f"reading -- residual-poison failure occurring *at or below* alpha={DEEPEST_ALPHA} "
        f"(a superset of Q3's <= {RESIDUAL_POISON_ALPHA_THRESHOLD} count, since "
        f"{DEEPEST_ALPHA} < {RESIDUAL_POISON_ALPHA_THRESHOLD}) -- is also reported for "
        f"completeness.",
        "",
        f"- **First break only at the deepest tested alpha ({DEEPEST_ALPHA})**: "
        f"**{q4['exactly_at_0_3_only']['any_strategy_count']} / {q1['tested_count']}** "
        f"queries (>=1 strategy), **{q4['exactly_at_0_3_only']['all_strategies_count']} / "
        f"{q1['tested_count']}** (all 4 strategies).",
        f"- _(secondary, cumulative reading, alpha <= {DEEPEST_ALPHA})_: "
        f"**{q4['cumulative_le_0_3']['any_strategy_count']} / {q1['tested_count']}** queries "
        f"(>=1 strategy), **{q4['cumulative_le_0_3']['all_strategies_count']} / "
        f"{q1['tested_count']}** (all 4 strategies).",
        "",
    ]

    lines += [
        "## Q5. Which E1 anchor strategy causes residual-poison failure at the highest alpha?",
        "",
        "\"Highest alpha\" = triggers actual residual-poison failure "
        "(`first_residual_poison_alpha`) at the *highest* alpha (least perturbation "
        "needed -- i.e. \"earliest\" in the descending sweep). Per query, the "
        "strategy(ies) achieving that highest alpha win; ties are shared.",
        "",
        "| strategy | queries won (caused residual-poison failure at the highest alpha) |",
        "|---|---|",
    ]
    for strategy, wins in q5["ranked"]:
        lines.append(f"| `{strategy}` | {wins} |")
    lines.append(f"| _(never caused residual-poison failure within the sweep)_ | {q5['never_broke_count']} |")
    lines.append("")
    if q5["top_strategy"]:
        lines.append(f"**Answer: `{q5['top_strategy']}`** causes residual-poison failure at the "
                      "highest alpha most often in this batch.")
    else:
        lines.append("**Answer: no strategy caused residual-poison failure within the swept alphas "
                      "for any tested query.**")
    lines.append("")

    lines += [
        "## Q6. Does PP top-pair reduction predict actual residual-poison failure?",
        "",
        "Compares `pp_decreased_alpha` versus `first_residual_poison_alpha` (not label "
        "change) across all tested `(query, strategy)` configs (`" +
        f"{q1['tested_count'] * len(E1_STRATEGIES)}` total):",
        "",
        "| category | count |",
        "|---|---|",
    ]
    for cat, count in q6["categories"].items():
        lines.append(f"| `{cat}` | {count} |")
    lines.append("")
    if q6["support_fraction_of_informative_configs"] is not None:
        lines.append(
            f"Of the configs where at least one of {{PP-weakening, residual-poison-failure}} "
            f"was observed, PP-weakening preceded-or-coincided with residual-poison failure in "
            f"**{q6['support_fraction_of_informative_configs']:.0%}** of them."
        )
    else:
        lines.append("No config showed either PP-weakening or residual-poison failure; no verdict possible.")
    lines.append("")

    lines += [
        "## Q7. Is the effect consistent across queries, or mostly one-query-specific?",
        "",
        "Per query, number of the 4 E1 strategies (out of 4) showing actual "
        f"residual-poison failure (`first_residual_poison_alpha`) by alpha <= "
        f"{RESIDUAL_POISON_ALPHA_THRESHOLD}:",
        "",
        "| query_id | strategies with residual-poison failure (out of 4) |",
        "|---|---|",
    ]
    for qid, n in q7["per_query_n_strategies_with_residual_poison_failure_by_0_5"].items():
        lines.append(f"| `{qid}` | {n} |")
    lines += [
        "",
        f"- Mean strategies-with-residual-poison-failure across tested queries: "
        f"**{q7['mean_strategies_with_residual_poison_failure']:.2f} / 4** (variance={q7['variance']:.2f}).",
        f"- Queries with **zero** strategies showing residual-poison failure by "
        f"alpha<={RESIDUAL_POISON_ALPHA_THRESHOLD}: "
        f"**{q7['queries_with_zero_strategies_with_residual_poison_failure']} / {q1['tested_count']}**.",
        f"- Queries with **all four** strategies showing residual-poison failure by "
        f"alpha<={RESIDUAL_POISON_ALPHA_THRESHOLD}: "
        f"**{q7['queries_with_all_four_strategies_with_residual_poison_failure']} / {q1['tested_count']}**.",
        "",
    ]
    zero = q7["queries_with_zero_strategies_with_residual_poison_failure"]
    four = q7["queries_with_all_four_strategies_with_residual_poison_failure"]
    total = q1["tested_count"]
    if total and (zero + four) >= total * 0.7 and zero > 0 and four > 0:
        verdict = ("**Bimodal / query-dependent**: most tested queries fall at one extreme "
                   "(0/4 or 4/4 strategies showing residual-poison failure) rather than a uniform "
                   "middle ground -- the effect's *presence* depends heavily on which query is being "
                   "tested, even though when it is present it tends to be strategy-agnostic (all four "
                   "E1 strategies agree).")
    elif total and four / total >= 0.7:
        verdict = ("**Broadly consistent**: most tested queries show residual-poison failure under "
                   "all four E1 strategies by this alpha threshold.")
    elif total and zero / total >= 0.7:
        verdict = ("**Mostly absent**: most tested queries show no residual-poison failure under any "
                   "E1 strategy by this alpha threshold within this sweep.")
    else:
        verdict = "**Mixed**: no single pattern (all-consistent, all-absent, or clean bimodal split) dominates."
    lines.append(verdict)
    lines.append("")

    lines += [
        "## Per-(query, strategy) summary table",
        "",
        "| query_id | strategy | baseline_label | pp_decreased_alpha | "
        "first_residual_poison_alpha | clean_removed_increased_alpha | first_label_change_alpha "
        "(secondary) | final_decision_label |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary_df.sort_values(["query_id", "strategy"]).iterrows():
        lines.append(
            f"| `{r['query_id']}` | `{r['strategy']}` | `{r['baseline_decision_label']}` | "
            f"{r['pp_decreased_alpha']} | {r['first_residual_poison_alpha']} | "
            f"{r['clean_removed_increased_alpha']} | {r['first_label_change_alpha']} | "
            f"`{r['final_decision_label']}` |"
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
    parser.add_argument("--exclude_query_id", action="append", default=["5a8cb288554299585d9e3726"],
                         help="Repeatable. Defaults to the known severe-failure control.")
    parser.add_argument("--report_md_path", default=None)
    parser.add_argument("--report_csv_path", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Path:
    args = parse_args(argv)

    records = viz._read_jsonl(args.diagnostics_jsonl)
    records_by_id = {r["query_id"]: r for r in records}
    identified_ids = discover_success_case_ids(records, args.dataset, args.k, args.n_injected,
                                                args.exclude_query_id)

    qr_index = viz.load_query_results_index(args.query_results_dir)
    excluded: Dict[str, str] = {}
    tested_ids: List[str] = []
    for qid in identified_ids:
        ok, recovered_len, expected_k = check_text_recoverable(qr_index, records_by_id[qid])
        if not ok:
            excluded[qid] = (
                f"text recovery mismatch: recovered {recovered_len} line(s), expected {expected_k} "
                f"(embedded newline inside a retrieved passage's raw text most likely splits it into "
                f"more lines than `k`; see `visualize_ragdefender_clusters.recover_pre_defense_texts`'s "
                f"no-guessing policy)"
            )
            continue
        run_dirs = summ.discover_run_dirs(args.output_dir, qid)
        latest = summ.latest_run_per_intervention(run_dirs)
        missing = [f"E1-{s}" for s in E1_STRATEGIES if f"E1-{s}" not in latest]
        if missing:
            excluded[qid] = f"missing run directories for: {', '.join(missing)}"
            continue
        tested_ids.append(qid)

    config_summaries = []
    all_sweep_rows = []
    for qid in tested_ids:
        run_dirs = summ.discover_run_dirs(args.output_dir, qid)
        latest = summ.latest_run_per_intervention(run_dirs)
        for strategy in E1_STRATEGIES:
            slug = f"E1-{strategy}"
            df = summ.load_sweep(latest[slug])
            df.insert(0, "query_id", qid)
            all_sweep_rows.append(df)
            config_summaries.append(compute_config_summary(qid, strategy, df))

    combined_sweep_df = pd.concat(all_sweep_rows, ignore_index=True) if all_sweep_rows else pd.DataFrame()
    summary_df = pd.DataFrame(config_summaries)

    q1 = answer_q1(tested_ids, identified_ids, excluded)
    q2 = answer_q2(summary_df, tested_ids) if tested_ids else {"any_strategy_count": 0, "all_strategies_count": 0, "per_query": {}}
    q3 = answer_q3(summary_df, tested_ids) if tested_ids else {"any_strategy_count": 0, "all_strategies_count": 0, "per_query": {}}
    q4 = answer_q4(summary_df, tested_ids) if tested_ids else {
        "cumulative_le_0_3": {"any_strategy_count": 0, "all_strategies_count": 0, "per_query": {}},
        "exactly_at_0_3_only": {"any_strategy_count": 0, "all_strategies_count": 0, "per_query": {}},
    }
    q5 = answer_q5(summary_df, tested_ids) if tested_ids else {
        "wins_by_strategy": {s: 0 for s in E1_STRATEGIES}, "ranked": [], "never_broke_count": 0,
        "per_query_winner": {}, "top_strategy": None,
    }
    q6 = answer_q6(summary_df) if len(summary_df) else {
        "categories": {}, "details": [], "support_fraction_of_informative_configs": None,
    }
    q7 = answer_q7(summary_df, tested_ids) if tested_ids else {
        "per_query_n_strategies_with_residual_poison_failure_by_0_5": {},
        "mean_strategies_with_residual_poison_failure": 0.0, "variance": 0.0,
        "distribution_of_counts": {}, "queries_with_zero_strategies_with_residual_poison_failure": 0,
        "queries_with_all_four_strategies_with_residual_poison_failure": 0,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.report_csv_path) if args.report_csv_path else output_dir / "BATCH_COMPARISON_SUCCESS_CASES.csv"
    combined_sweep_df.to_csv(csv_path, index=False)

    md_path = Path(args.report_md_path) if args.report_md_path else output_dir / "BATCH_COMPARISON_SUCCESS_CASES.md"
    report_text = render_markdown(q1, q2, q3, q4, q5, q6, q7, summary_df, args.dataset, args.k, args.n_injected)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Identified {q1['identified_count']} success cases; tested {q1['tested_count']}; "
          f"excluded {q1['excluded_count']}.")
    print(f"Wrote batch CSV to: {csv_path}")
    print(f"Wrote batch report to: {md_path}")
    return md_path


if __name__ == "__main__":
    main()
