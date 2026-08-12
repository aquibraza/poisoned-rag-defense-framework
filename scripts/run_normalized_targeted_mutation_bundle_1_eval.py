#!/usr/bin/env python3
"""Re-runs the fixed-context, cross-defense evaluation of the 3 defense-
*targeted* GPT mutation families under
`manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/`, this time
consuming the **normalized** JSONL files produced by
`scripts/audit_normalize_mutation_bundle_1.py`
(`mutation_bundle_1/normalized/*.normalized.jsonl`) instead of the raw
GPT-authored family files, and compares every metric against the previous
(`scripts/run_targeted_mutation_bundle_1_eval.py`) run's output to confirm
whether canonical-`doc_id` normalization changed anything.

**Fixed retrieval only** -- identical guarantees to
`scripts/run_text_mutation_fixed_context_eval.py` (imported and reused
directly, never reimplemented, for every context-reconstruction/defense-
scoring primitive): this script never reruns retrieval, never trains or
retrains any model, never calls an LLM/GPT/API, and never calls
`llm.query()`. It only:

1. reconstructs each selected query's *exact* original k=10 retrieved
   context from already-exported pilot artifacts
   (`mutation_input_passages.csv` + `clean_context_passages.csv`);
2. for each of the 3 *normalized* mutation-family JSONL files, replaces
   *only* the 5 poisoned passages' text (matched by `poison_slot`) with
   that family's already-normalized `mutated_text`, leaving every clean
   passage and k=10 membership untouched;
3. re-scores that fixed passage list with the same already-trained defense
   scorers `run_text_mutation_fixed_context_eval.py` uses (RAGDefender,
   semantic-mode FilterRAG, ML-FilterRAG-top-k) -- imported and reused
   unmodified, not copied;
4. diffs this run's per-(family, query_id) metrics against the previous
   raw-family-file run's already-written
   `mutation_bundle_1/evaluation/targeted_family_bundle_scores.csv`.

Expected outcome (documented, not assumed): the previous run already
resolved every mutated passage's *identity* by `(query_id, poison_slot)`
against `mutation_input_passages.csv` -- never by the raw family file's own
(possibly wrong) `doc_id` -- so normalizing `doc_id` should not change any
score. This script verifies that claim empirically rather than asserting it.

Usage:
    python scripts/run_normalized_targeted_mutation_bundle_1_eval.py \\
        --pilot_dir manual_text_mutation_pilot/hotpotqa_50q_k10 \\
        --normalized_dir manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/normalized \\
        --previous_eval_dir manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/evaluation \\
        --out_dir manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/evaluation_normalized
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Belt-and-suspenders: forced offline before importing anything that may
# touch huggingface_hub/sentence-transformers, mirroring the two scripts
# this one reuses.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import run_text_mutation_fixed_context_eval as base_eval  # noqa: E402
import run_targeted_mutation_bundle_1_eval as prev_eval  # noqa: E402

DEFAULT_PILOT_DIR = base_eval.DEFAULT_PILOT_DIR
DEFAULT_BUNDLE_DIR = os.path.join(DEFAULT_PILOT_DIR, "mutation_bundle_1")
DEFAULT_NORMALIZED_DIR = os.path.join(DEFAULT_BUNDLE_DIR, "normalized")
DEFAULT_PREVIOUS_EVAL_DIR = os.path.join(DEFAULT_BUNDLE_DIR, "evaluation")
DEFAULT_ML_MODEL_PATH = base_eval.DEFAULT_ML_MODEL_PATH

DEFENSE_NAMES = prev_eval.DEFENSE_NAMES  # ("ragdefender", "filterrag", "ml_filterrag")

# Each normalized family's expected filename + intended target defense.
# Family keys are identical to run_targeted_mutation_bundle_1_eval.FAMILY_SPECS's
# so the previous run's CSV rows (keyed by "family") join cleanly against this
# run's rows.
NORMALIZED_FAMILY_SPECS: Dict[str, Dict] = {
    "ragdefender_targeted": {
        "normalized_filename": "ragdefender_targeted.normalized.jsonl",
        "intended_defense": "ragdefender",
    },
    "filterrag_targeted": {
        "normalized_filename": "filterrag_targeted.normalized.jsonl",
        "intended_defense": "filterrag",
    },
    "mlfilterrag_targeted": {
        "normalized_filename": "mlfilterrag_targeted.normalized.jsonl",
        "intended_defense": "ml_filterrag",
    },
}


# ---------------------------------------------------------------------------
# 1. Loading + validating the normalized JSONL files (strict: these are
#    expected to already be clean, so any schema violation here is treated
#    as a real bug, not a soft "flag").
# ---------------------------------------------------------------------------

def load_normalized_family(path: str, family_key: str, intended_defense: str) -> Dict[str, Dict]:
    """Parses one `*.normalized.jsonl` file (one JSON object per line) into
    `{query_id: {"query_id", "question", "target_wrong_answer",
    "selection_role", "passages": [{"poison_slot", "doc_id",
    "source_file_doc_id", "text"}, ...]}}`. Raises `ValueError` naming the
    offending line/query on any schema violation (missing query_id, wrong
    passage count, duplicate/missing/invalid `poison_slot`, missing
    canonical `doc_id`, empty `mutated_text`)."""
    out: Dict[str, Dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("query_id")
            if not qid or not isinstance(qid, str):
                raise ValueError(f"{path}: line {line_no}: missing/invalid 'query_id'.")

            passages = rec.get("mutated_passages")
            if not isinstance(passages, list) or len(passages) != 5:
                raise ValueError(
                    f"{path}: query_id={qid!r} has "
                    f"{len(passages) if isinstance(passages, list) else 'no'} mutated_passages "
                    "(expected exactly 5)."
                )

            parsed_passages: List[Dict] = []
            seen_slots = set()
            for p in passages:
                slot_raw = p.get("poison_slot")
                if slot_raw is None:
                    raise ValueError(f"{path}: query_id={qid!r} has a passage missing 'poison_slot'.")
                try:
                    slot = int(slot_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}: query_id={qid!r} has a non-integer poison_slot={slot_raw!r}.") from exc
                if slot in seen_slots:
                    raise ValueError(f"{path}: query_id={qid!r} has a duplicate poison_slot={slot!r}.")
                seen_slots.add(slot)

                doc_id = p.get("doc_id")
                if not doc_id or not isinstance(doc_id, str):
                    raise ValueError(f"{path}: query_id={qid!r} poison_slot={slot!r} is missing a canonical 'doc_id'.")

                text = p.get("mutated_text")
                if not text or not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{path}: query_id={qid!r} poison_slot={slot!r} has an empty 'mutated_text'.")

                parsed_passages.append({
                    "poison_slot": slot,
                    "doc_id": doc_id,
                    "source_file_doc_id": p.get("source_file_doc_id"),
                    "text": text,
                })

            if sorted(seen_slots) != [0, 1, 2, 3, 4]:
                raise ValueError(
                    f"{path}: query_id={qid!r} has poison_slot values {sorted(seen_slots)!r} "
                    "(expected exactly [0, 1, 2, 3, 4])."
                )
            parsed_passages.sort(key=lambda x: x["poison_slot"])

            out[qid] = {
                "query_id": qid,
                "question": rec.get("question"),
                "target_wrong_answer": rec.get("target_wrong_answer"),
                "selection_role": rec.get("selection_role", ""),
                "family": rec.get("family", family_key),
                "intended_defense": rec.get("intended_defense", intended_defense),
                "passages": parsed_passages,
            }
    if not out:
        raise ValueError(f"{path}: no records parsed (empty file?).")
    return out


def validate_canonical_doc_id(
    family_key: str, records: Dict[str, Dict], poison_by_query: Dict[str, List[Dict]]
) -> None:
    """Asserts every normalized passage's `doc_id` matches the authoritative
    `(query_id, poison_slot) -> doc_id` mapping in `mutation_input_passages.csv`.
    Unlike `audit_normalize_mutation_bundle_1.py`'s permissive audit of the
    *raw* family files (where a mismatch is an expected, reported data-quality
    flag), a mismatch *here* -- in files that `scripts/audit_normalize_mutation_bundle_1.py`
    already claims to have normalized -- indicates a real normalization bug
    and is treated as fatal. Raises `ValueError` listing every mismatch found."""
    mismatches: List[Dict] = []
    for qid, rec in records.items():
        doc_by_slot = {int(r["poison_slot"]): r["doc_id"] for r in poison_by_query.get(qid, [])}
        for p in rec["passages"]:
            expected = doc_by_slot.get(p["poison_slot"])
            if expected is None:
                raise ValueError(
                    f"{family_key}: query_id={qid!r} poison_slot={p['poison_slot']!r} has no "
                    "corresponding row in mutation_input_passages.csv."
                )
            if p["doc_id"] != expected:
                mismatches.append({
                    "family": family_key, "query_id": qid, "poison_slot": p["poison_slot"],
                    "normalized_doc_id": p["doc_id"], "csv_doc_id": expected,
                })
    if mismatches:
        raise ValueError(
            f"{family_key}: {len(mismatches)} normalized passage(s) carry a doc_id that does NOT "
            f"match mutation_input_passages.csv (normalization bug): {mismatches!r}"
        )


def family_record_to_bundle(rec: Dict) -> Dict:
    """Adapts a parsed normalized-family record into the
    `{"mutated_passages": [...]}` shape `base_eval.build_mutated_context`
    expects (`poison_rank` == `poison_slot`, `mutated_text` == `text`)."""
    return {
        "mutated_passages": [
            {"poison_rank": p["poison_slot"], "mutated_text": p["text"]} for p in rec["passages"]
        ]
    }


# ---------------------------------------------------------------------------
# 2. Comparison against the previous (raw-family-file) mutation_bundle_1 run.
# ---------------------------------------------------------------------------

COMPARISON_TOLERANCE = 1e-9


def _coerce_csv_value(v: Optional[str]):
    if v is None or v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f


def load_previous_bundle_scores(path: str) -> Optional[Dict[tuple, Dict]]:
    """Loads the previous run's `targeted_family_bundle_scores.csv` (from
    the raw-family-file evaluation) into `{(family, query_id): row}`, with
    numeric-looking values coerced to float/bool. Returns `None` (never
    raises) if the file is absent, so the comparison is explicitly marked
    unavailable rather than silently skipped or fabricated."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[tuple, Dict] = {}
    for row in rows:
        coerced = {k: _coerce_csv_value(v) for k, v in row.items()}
        out[(row["family"], row["query_id"])] = coerced
    return out


COMPARISON_SUMMARY_FIELDS = [
    "family", "query_id", "previous_run_found",
    "n_metrics_compared", "n_metrics_changed", "any_metric_changed",
    "max_abs_diff", "changed_metric_names",
]


def build_comparison_rows(
    new_bundle_rows: Sequence[Dict], previous_by_key: Optional[Dict[tuple, Dict]],
) -> List[Dict]:
    """One row per (family, query_id) in the new (normalized) run, with a
    `{metric}_previous` / `{metric}_new` / `{metric}_changed` triplet for
    *every* metric in `base_eval._NUMERIC_METRIC_KEYS` (29 metrics -- the
    full set scored by both runs), plus a summary
    (`n_metrics_changed`, `any_metric_changed`, `max_abs_diff`,
    `changed_metric_names`). `changed` uses a `1e-9` absolute-difference
    tolerance; `None` vs a real value always counts as changed."""
    metric_keys = list(base_eval._NUMERIC_METRIC_KEYS)  # noqa: SLF001 -- shared module constant, not private state
    rows: List[Dict] = []
    for row in new_bundle_rows:
        key = (row["family"], row["query_id"])
        prev = previous_by_key.get(key) if previous_by_key is not None else None
        out: Dict = {
            "family": row["family"], "query_id": row["query_id"],
            "previous_run_found": prev is not None,
        }
        changed_metrics: List[str] = []
        max_abs_diff = 0.0
        for mk in metric_keys:
            new_v = row.get(mk)
            prev_v = prev.get(mk) if prev is not None else None
            if prev is None:
                changed = None
            elif new_v is None and prev_v is None:
                changed = False
            elif new_v is None or prev_v is None:
                changed = True
            else:
                diff = abs(float(new_v) - float(prev_v))
                changed = diff > COMPARISON_TOLERANCE
                max_abs_diff = max(max_abs_diff, diff)
            out[f"{mk}_previous"] = prev_v
            out[f"{mk}_new"] = new_v
            out[f"{mk}_changed"] = changed
            if changed:
                changed_metrics.append(mk)
        out["n_metrics_compared"] = len(metric_keys) if prev is not None else 0
        out["n_metrics_changed"] = len(changed_metrics)
        out["any_metric_changed"] = bool(changed_metrics) if prev is not None else None
        out["changed_metric_names"] = ";".join(changed_metrics)
        out["max_abs_diff"] = max_abs_diff if prev is not None else None
        rows.append(out)
    return rows


def comparison_fields() -> List[str]:
    metric_keys = list(base_eval._NUMERIC_METRIC_KEYS)  # noqa: SLF001
    fields = list(COMPARISON_SUMMARY_FIELDS)
    for mk in metric_keys:
        fields += [f"{mk}_previous", f"{mk}_new", f"{mk}_changed"]
    return fields


def resolve_first_pilot_deltas_path(pilot_dir: str) -> str:
    """The first (generic gpt_b01/b02/b03) pilot's outputs now live under
    `mutation_bundle_0/evaluation/` (moved there in a later reorg); the
    older `evaluation/` path is kept as a fallback for robustness. Neither
    location is required to exist -- `prev_eval.load_first_pilot_deltas`
    already handles a missing file by returning `None`."""
    candidates = [
        os.path.join(pilot_dir, "mutation_bundle_0", "evaluation", "mutation_bundle_deltas.csv"),
        os.path.join(pilot_dir, "evaluation", "mutation_bundle_deltas.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


# ---------------------------------------------------------------------------
# 3. Report.
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    return base_eval._fmt(v)  # noqa: SLF001 -- shared formatting helper, not private state


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(statistics.fmean(values)) if values else None


def build_report(
    *, pilot_dir: str, normalized_dir: str, previous_eval_dir: str, out_dir: str,
    ml_model_path: str, query_ids: Sequence[str], bundle_rows: Sequence[Dict],
    delta_rows: Sequence[Dict], summary_rows: Sequence[Dict], cross_matrix: Sequence[Dict],
    comparison_rows: Sequence[Dict], previous_available: bool, selected_queries: Dict[str, Dict],
    first_pilot_deltas: Optional[List[Dict]],
) -> str:
    lines: List[str] = []
    lines.append("# Normalized Targeted Mutation Bundle 1 -- Fixed-Context Cross-Defense Re-Evaluation Report")
    lines.append("")
    lines.append(
        "Re-run of the fixed-context, cross-defense evaluation of the 3 defense-*targeted* GPT "
        f"mutation families, this time consuming the **normalized** JSONL files in "
        f"`{os.path.relpath(normalized_dir, REPO_ROOT)}/` (canonical `doc_id` substituted by "
        "`scripts/audit_normalize_mutation_bundle_1.py`) instead of the raw GPT-authored family "
        "files. Retrieval membership/order is identical between the baseline and every mutated "
        "context for a given query; only the 5 poisoned passages' text differs per family; clean "
        "passages are byte-identical everywhere."
    )
    lines.append("")
    lines.append("## Artifact paths used")
    lines.append("")
    for key, spec in NORMALIZED_FAMILY_SPECS.items():
        lines.append(
            f"- `{os.path.relpath(os.path.join(normalized_dir, spec['normalized_filename']), REPO_ROOT)}` "
            f"({key}, intended target: {spec['intended_defense']})"
        )
    lines.append(f"- `{os.path.join(pilot_dir, 'selected_queries.csv')}`")
    lines.append(f"- `{os.path.join(pilot_dir, 'mutation_input_passages.csv')}`")
    lines.append(f"- `{os.path.join(pilot_dir, 'clean_context_passages.csv')}`")
    lines.append(f"- `{ml_model_path}` (trained ML-FilterRAG-top-k random-forest classifier, loaded read-only)")
    lines.append(
        f"- `{os.path.relpath(os.path.join(previous_eval_dir, 'targeted_family_bundle_scores.csv'), REPO_ROOT)}` "
        "(previous raw-family-file run, for comparison only -- not re-scored)"
    )
    lines.append("")

    lines.append("## Queries evaluated")
    lines.append("")
    for qid in query_ids:
        q = selected_queries.get(qid, {})
        lines.append(
            f"- `{qid}` ({q.get('selection_role', 'n/a')}) -- {q.get('question', 'n/a')} "
            f"(target wrong answer: {q.get('target_wrong_answer', 'n/a')})"
        )
    lines.append("")

    lines.append("## Summary by family x defense")
    lines.append("")
    lines.append("| family | intended target | defense | mean removed_poison | mean delta_removed_poison | mean residual_poison_fraction |")
    lines.append("|---|---|---|---:|---:|---:|")
    for r in summary_rows:
        marker = " **(target)**" if r["is_intended_target"] else ""
        lines.append(
            f"| `{r['family']}` | {r['intended_target_defense']} | {r['defense']}{marker} | "
            f"{_fmt(r['mean_removed_poison'])} | {_fmt(r['mean_delta_removed_poison'])} | "
            f"{_fmt(r['mean_residual_poison_fraction'])} |"
        )
    lines.append("")

    lines.append("## Cross-defense failure matrix")
    lines.append("")
    lines.append("| family | intended target | RAGDefender delta (weakened?) | FilterRAG delta (weakened?) | ML-FilterRAG delta (weakened?) | any cross-defense failure |")
    lines.append("|---|---|---|---|---|---|")
    for r in cross_matrix:
        lines.append(
            f"| `{r['family']}` | {r['intended_target_defense']} | "
            f"{_fmt(r['ragdefender_mean_delta_removed_poison'])} ({r['ragdefender_weakened']}) | "
            f"{_fmt(r['filterrag_mean_delta_removed_poison'])} ({r['filterrag_weakened']}) | "
            f"{_fmt(r['ml_filterrag_mean_delta_removed_poison'])} ({r['ml_filterrag_weakened']}) | "
            f"{r['any_cross_defense_failure']} |"
        )
    lines.append("")

    # ---- Comparison vs. previous run --------------------------------------
    lines.append("## Comparison vs. the previous (raw-family-file) mutation_bundle_1 run")
    lines.append("")
    if not previous_available:
        lines.append(
            f"Previous run's `{os.path.relpath(os.path.join(previous_eval_dir, 'targeted_family_bundle_scores.csv'), REPO_ROOT)}` "
            "was not found -- comparison unavailable; no comparison was fabricated."
        )
        lines.append("")
        total_compared = 0
        total_changed = 0
        rows_with_any_change = []
    else:
        total_compared = sum(r["n_metrics_compared"] for r in comparison_rows)
        total_changed = sum(r["n_metrics_changed"] for r in comparison_rows)
        rows_with_any_change = [r for r in comparison_rows if r["any_metric_changed"]]
        lines.append(
            f"Compared **all {len(base_eval._NUMERIC_METRIC_KEYS)} numeric metrics** across all "
            f"{len(comparison_rows)} (family, query_id) bundle rows ({total_compared} total metric "
            f"comparisons): **{total_changed} changed** (tolerance {COMPARISON_TOLERANCE:g})."
        )
        lines.append("")
        lines.append("| family | query_id | n_metrics_changed | max_abs_diff | changed_metric_names |")
        lines.append("|---|---|---:|---:|---|")
        for r in comparison_rows:
            lines.append(
                f"| `{r['family']}` | `{r['query_id']}` | {r['n_metrics_changed']} | "
                f"{_fmt(r['max_abs_diff'])} | {r['changed_metric_names'] or '(none)'} |"
            )
        lines.append("")

    lines.append("## Answers")
    lines.append("")
    if not previous_available:
        lines.append(
            "**1. Did the normalized evaluation reproduce the previous targeted evaluation?** "
            "Unknown -- the previous run's score CSV was not found, so no comparison could be made."
        )
        lines.append(
            "**2. Did any metric or removal decision change after canonical doc_id normalization?** "
            "Unknown for the same reason."
        )
    else:
        reproduced = total_changed == 0
        lines.append(
            f"**1. Did the normalized evaluation reproduce the previous targeted evaluation?** "
            f"{'Yes -- exact reproduction' if reproduced else 'No'}: {total_changed} of "
            f"{total_compared} metric comparisons differed (across {len(comparison_rows)} bundle "
            f"rows x {len(base_eval._NUMERIC_METRIC_KEYS)} metrics each)."
        )
        lines.append("")
        if reproduced:
            change_summary = "No."
        else:
            changed_bundle_labels = ", ".join(f"`{r['family']}`/`{r['query_id']}`" for r in rows_with_any_change)
            change_summary = f"Yes -- {len(rows_with_any_change)} bundle row(s) changed: {changed_bundle_labels}."
        lines.append(
            f"**2. Did any metric or removal decision change after canonical doc_id normalization?** "
            f"{change_summary} "
            "This is expected: both the previous (raw-family-file) run and this normalized run "
            "resolve mutated-passage *identity* by `(query_id, poison_slot)` against "
            "`mutation_input_passages.csv` when substituting text into the fixed k=10 context "
            "(see `run_text_mutation_fixed_context_eval.build_mutated_context`) -- neither run ever "
            "uses a family file's own `doc_id` for that substitution. Normalizing `doc_id` therefore "
            "changes only metadata carried alongside the mutated text, not which passage's text is "
            "mutated or how it is scored."
        )
    lines.append("")

    # ---- Q3: best family per defense (not just each family's own target). --
    summary_by_family_defense = {(r["family"], r["defense"]): r for r in summary_rows}
    lines.append("**3. Which mutation family best attacked each defense?**")
    lines.append("")
    for defense in DEFENSE_NAMES:
        candidates = [
            (family, summary_by_family_defense[(family, defense)])
            for family in NORMALIZED_FAMILY_SPECS
        ]
        best_family, best_row = min(
            candidates,
            key=lambda kv: (kv[1]["mean_delta_removed_poison"] if kv[1]["mean_delta_removed_poison"] is not None else 0.0),
        )
        lines.append(
            f"- {defense}: `{best_family}` (mean delta_removed_poison = "
            f"{_fmt(best_row['mean_delta_removed_poison'])}; all 3: "
            + ", ".join(
                f"`{family}`={_fmt(summary_by_family_defense[(family, defense)]['mean_delta_removed_poison'])}"
                for family in NORMALIZED_FAMILY_SPECS
            ) + ")."
        )
    lines.append("")

    def _largest_drop_for_defense(delta_key: str) -> Optional[Dict]:
        candidates = [r for r in delta_rows if r.get(delta_key) is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r[delta_key])

    largest_drop_ragdefender = _largest_drop_for_defense("delta_ragdefender_removed_poison")
    largest_drop_filterrag = _largest_drop_for_defense("delta_filterrag_removed_poison")
    largest_drop_ml = _largest_drop_for_defense("delta_ml_removed_poison_t04")

    lines.append("**4. Which individual bundle caused the largest removed_poison drop for each defense?**")
    lines.append("")
    for label, row, key in (
        ("RAGDefender", largest_drop_ragdefender, "delta_ragdefender_removed_poison"),
        ("FilterRAG", largest_drop_filterrag, "delta_filterrag_removed_poison"),
        ("ML-FilterRAG (t=0.4)", largest_drop_ml, "delta_ml_removed_poison_t04"),
    ):
        if row is None:
            lines.append(f"- {label}: n/a (no non-null deltas).")
        else:
            lines.append(f"- {label}: `{row['query_id']}` / `{row['family']}` (delta_removed_poison = {_fmt(row[key])}).")
    lines.append("")

    ranked = sorted(
        bundle_rows,
        key=lambda r: (
            r["ragdefender_removed_poison"] + r["filterrag_removed_poison"] + r["ml_removed_poison_t04"],
            r["ml_mean_poison_probability"] if r["ml_mean_poison_probability"] is not None else 1.0,
        ),
    )
    top2 = ranked[:2]
    lines.append("**5. Which 1-2 bundles are the best candidates for a full retrieval rerun?**")
    lines.append("")
    for i, row in enumerate(top2, start=1):
        lines.append(
            f"{i}. `{row['query_id']}` / `{row['family']}` -- combined removal across the three "
            f"defenses: RAGDefender={row['ragdefender_removed_poison']}, "
            f"FilterRAG={row['filterrag_removed_poison']}, ML-FilterRAG t0.4={row['ml_removed_poison_t04']} "
            f"(out of {row['N_retrieved_poison']} retrieved poison passages); mean ML poison "
            f"probability={_fmt(row['ml_mean_poison_probability'])}."
        )
    lines.append("")

    if first_pilot_deltas is None:
        lines.append(
            "**6. Are the results stronger than the first generic mutation pilot?** "
            "Comparison unavailable -- the first pilot's `mutation_bundle_deltas.csv` was not found "
            "at either expected path; no comparison was fabricated."
        )
    else:
        first_removed = [d["delta_removed_poison"] for d in first_pilot_deltas if d.get("delta_removed_poison") is not None]
        first_proba = [d["delta_mean_poison_probability"] for d in first_pilot_deltas if d.get("delta_mean_poison_probability") is not None]
        this_removed = [d["delta_ml_removed_poison_t04"] for d in delta_rows if d.get("delta_ml_removed_poison_t04") is not None]
        this_proba = [d["delta_ml_mean_poison_probability"] for d in delta_rows if d.get("delta_ml_mean_poison_probability") is not None]
        first_mean_removed = _mean(first_removed)
        this_mean_removed = _mean(this_removed)
        first_mean_proba = _mean(first_proba)
        this_mean_proba = _mean(this_proba)
        stronger = (
            this_mean_removed is not None and first_mean_removed is not None and this_mean_removed < first_mean_removed
        )
        lines.append(
            f"**6. Are the results stronger than the first generic mutation pilot?** "
            f"Mean delta_removed_poison (ML-FilterRAG t=0.4): first pilot (generic gpt_b01/b02/b03, "
            f"4 primary queries) = {_fmt(first_mean_removed)}, this normalized targeted pilot (3 "
            f"families, {len(query_ids)} queries) = {_fmt(this_mean_removed)}. Mean "
            f"delta_mean_poison_probability: first pilot = {_fmt(first_mean_proba)}, this pilot = "
            f"{_fmt(this_mean_proba)}. "
            f"{'Yes -- the targeted families achieved a more negative (larger) mean reduction in ML-FilterRAG removed_poison than the first generic pilot.' if stronger else 'No / marginal -- the targeted families did not achieve a larger mean reduction in ML-FilterRAG removed_poison than the first generic pilot on this comparison basis.'} "
            "(Identical conclusion to the previous raw-family-file run, as expected -- see Q1/Q2.)"
        )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Each family here is a *single* rewrite per poison_slot (one bundle per query/family), "
        "so per-family statistics are means over 6 queries, not over multiple independent bundle "
        "attempts per query -- identical scope to the previous raw-family-file run."
    )
    lines.append(
        "- This comparison is against the previous run's *scores*, computed independently in this "
        "process (fresh model loads, same `device=cpu` determinism setting); it is not a claim that "
        "the two runs share literal in-memory state, only that they produce numerically identical "
        "output given identical inputs."
    )
    lines.append("")
    lines.append("## Process confirmation")
    lines.append("")
    lines.append("- No GPT/API calls were made.")
    lines.append("- No `llm.query()` calls were made.")
    lines.append("- Retrieval was not rerun (k=10 membership reconstructed verbatim from existing pilot CSV artifacts).")
    lines.append("- No model was trained or retrained (every model loaded read-only for inference).")
    lines.append("- No defense code (`defense/*.py`) was modified; every defense function used here is called unmodified via `scripts/run_text_mutation_fixed_context_eval.py`.")
    lines.append("- Only text substitution on already-normalized mutation family files was applied; no mutations or normalizations were generated by this script.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Orchestration.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot_dir", default=os.path.join(REPO_ROOT, DEFAULT_PILOT_DIR))
    parser.add_argument("--normalized_dir", default=None)
    parser.add_argument("--previous_eval_dir", default=None)
    parser.add_argument("--ml_model_path", default=os.path.join(REPO_ROOT, DEFAULT_ML_MODEL_PATH))
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    pilot_dir = args.pilot_dir
    bundle_dir = os.path.join(pilot_dir, "mutation_bundle_1")
    normalized_dir = args.normalized_dir or os.path.join(bundle_dir, "normalized")
    previous_eval_dir = args.previous_eval_dir or os.path.join(bundle_dir, "evaluation")
    out_dir = args.out_dir or os.path.join(bundle_dir, "evaluation_normalized")

    selected_queries = base_eval.load_selected_queries(os.path.join(pilot_dir, "selected_queries.csv"))
    poison_by_query = base_eval.load_mutation_input_passages(os.path.join(pilot_dir, "mutation_input_passages.csv"))
    clean_by_query = base_eval.load_clean_context_passages(os.path.join(pilot_dir, "clean_context_passages.csv"))

    family_records: Dict[str, Dict[str, Dict]] = {}
    for family_key, spec in NORMALIZED_FAMILY_SPECS.items():
        path = os.path.join(normalized_dir, spec["normalized_filename"])
        records = load_normalized_family(path, family_key, spec["intended_defense"])
        validate_canonical_doc_id(family_key, records, poison_by_query)
        family_records[family_key] = records
        print(f"[run_normalized_targeted_mutation_bundle_1_eval] {family_key}: loaded {len(records)} normalized query record(s), canonical doc_id verified.")

    query_ids = sorted(set.union(*(set(r) for r in family_records.values())))
    for qid in query_ids:
        if qid not in selected_queries or qid not in poison_by_query or qid not in clean_by_query:
            raise ValueError(f"query_id={qid!r} from a normalized family file is missing required pilot rows.")
    for family_key, records in family_records.items():
        missing = set(query_ids) - set(records)
        if missing:
            print(f"[run_normalized_targeted_mutation_bundle_1_eval] WARNING: family={family_key!r} is missing query_id(s) {sorted(missing)!r}; skipping those for this family.")

    print(f"[run_normalized_targeted_mutation_bundle_1_eval] loading models (device={base_eval.DEVICE})...")
    models = base_eval.load_models(args.ml_model_path)

    bundle_rows: List[Dict] = []
    delta_rows: List[Dict] = []

    for qid in query_ids:
        q = selected_queries[qid]
        question = q["question"]
        target_wrong_answer = q["target_wrong_answer"]

        original_context = base_eval.build_original_context(poison_by_query[qid], clean_by_query[qid])
        print(f"[run_normalized_targeted_mutation_bundle_1_eval] scoring baseline for {qid} ({len(original_context)} passages)...")
        baseline_metrics = base_eval.score_context(question, original_context, models)

        for family_key, spec in NORMALIZED_FAMILY_SPECS.items():
            rec = family_records[family_key].get(qid)
            if rec is None:
                continue

            bundle = family_record_to_bundle(rec)
            mutated_context = base_eval.build_mutated_context(original_context, poison_by_query[qid], bundle)
            base_eval.assert_same_k10_membership(original_context, mutated_context)
            for orig, mut in zip(original_context, mutated_context):
                if not orig.is_poison and orig.text != mut.text:
                    raise AssertionError(
                        f"{qid}/{family_key}: clean passage doc_id={orig.doc_id!r} text changed "
                        "(must remain unchanged)."
                    )

            print(f"[run_normalized_targeted_mutation_bundle_1_eval] scoring {qid} / {family_key}...")
            bundle_metrics = base_eval.score_context(question, mutated_context, models)
            bundle_rows.append({
                "query_id": qid, "k": 10, "family": family_key, "bundle_id": family_key,
                "intended_target_defense": spec["intended_defense"],
                "selection_role": q.get("selection_role", ""),
                "question": question, "target_wrong_answer": target_wrong_answer,
                **bundle_metrics,
            })

            deltas = base_eval.compute_deltas(baseline_metrics, bundle_metrics)
            delta_rows.append({
                "query_id": qid, "k": 10, "family": family_key, "bundle_id": family_key,
                "intended_target_defense": spec["intended_defense"],
                **deltas,
            })

    summary_rows: List[Dict] = []
    for family_key, spec in NORMALIZED_FAMILY_SPECS.items():
        family_bundle_rows = [r for r in bundle_rows if r["family"] == family_key]
        family_delta_rows = [d for d in delta_rows if d["family"] == family_key]
        for defense in DEFENSE_NAMES:
            summary_rows.append(
                prev_eval.summarize_family_defense(family_key, spec["intended_defense"], defense, family_bundle_rows, family_delta_rows)
            )
    cross_matrix = prev_eval.build_cross_defense_failure_matrix(summary_rows)

    previous_scores_path = os.path.join(previous_eval_dir, "targeted_family_bundle_scores.csv")
    previous_by_key = load_previous_bundle_scores(previous_scores_path)
    comparison_rows = build_comparison_rows(bundle_rows, previous_by_key)

    first_pilot_deltas = prev_eval.load_first_pilot_deltas(resolve_first_pilot_deltas_path(pilot_dir))

    baseline_fields = [
        "query_id", "k", "selection_role", "question", "target_wrong_answer",
        "N_retrieved_poison", "N_retrieved_clean",
    ] + list(base_eval._NUMERIC_METRIC_KEYS)
    bundle_fields = ["family", "bundle_id", "intended_target_defense"] + baseline_fields
    delta_fields = (
        ["query_id", "k", "family", "bundle_id", "intended_target_defense"]
        + [f"delta_{k}" for k in base_eval._NUMERIC_METRIC_KEYS]
        + list(base_eval.DELTA_ALIASES.keys())
    )

    base_eval.write_csv(os.path.join(out_dir, "normalized_targeted_family_bundle_scores.csv"), bundle_fields, bundle_rows)
    base_eval.write_csv(os.path.join(out_dir, "normalized_targeted_family_bundle_deltas.csv"), delta_fields, delta_rows)
    base_eval.write_csv(os.path.join(out_dir, "normalized_targeted_family_summary_by_defense.csv"), prev_eval.SUMMARY_FIELDS, summary_rows)
    base_eval.write_csv(os.path.join(out_dir, "normalized_cross_defense_failure_matrix.csv"), prev_eval.CROSS_MATRIX_FIELDS, cross_matrix)
    base_eval.write_csv(os.path.join(out_dir, "normalized_vs_previous_comparison.csv"), comparison_fields(), comparison_rows)

    report = build_report(
        pilot_dir=pilot_dir, normalized_dir=normalized_dir, previous_eval_dir=previous_eval_dir,
        out_dir=out_dir, ml_model_path=args.ml_model_path, query_ids=query_ids,
        bundle_rows=bundle_rows, delta_rows=delta_rows, summary_rows=summary_rows,
        cross_matrix=cross_matrix, comparison_rows=comparison_rows,
        previous_available=previous_by_key is not None, selected_queries=selected_queries,
        first_pilot_deltas=first_pilot_deltas,
    )
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "NORMALIZED_TARGETED_MUTATION_BUNDLE_1_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    total_changed = sum(r["n_metrics_changed"] for r in comparison_rows) if previous_by_key is not None else None
    print(
        f"[run_normalized_targeted_mutation_bundle_1_eval] wrote {len(bundle_rows)} bundle row(s), "
        f"{len(delta_rows)} delta row(s), {len(summary_rows)} summary row(s), "
        f"{len(comparison_rows)} comparison row(s) to {out_dir}"
    )
    print(f"[run_normalized_targeted_mutation_bundle_1_eval] metrics changed vs previous run: {total_changed}")
    print(
        f"[run_normalized_targeted_mutation_bundle_1_eval] SLM generation calls: "
        f"{models.memo_slm_answer_fn.calls} (cache hits: {models.memo_slm_answer_fn.cache_hits})"
    )
    print(
        f"[run_normalized_targeted_mutation_bundle_1_eval] LM perplexity calls: "
        f"{models.memo_causal_scorer.calls} (cache hits: {models.memo_causal_scorer.cache_hits})"
    )


if __name__ == "__main__":
    main()
