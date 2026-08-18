#!/usr/bin/env python3
"""Fixed-context, cross-defense evaluation of the v2 RAGDefender-targeted GPT
mutation family (`ragdefender-v2_semantic_support.jsonl.txt`) in
`manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/`, scored
against all three defenses (RAGDefender, semantic FilterRAG epsilon=0.2,
ML-FilterRAG-top-k at t in {0.35, 0.4, 0.5}) and compared against the v1
`ragdefender_targeted` family's already-published scores.

**Fixed retrieval only** -- identical guarantees to
`scripts/run_text_mutation_fixed_context_eval.py` and
`scripts/run_targeted_mutation_bundle_1_eval.py` (both reused directly, not
reimplemented, for every defense-scoring/context-reconstruction primitive):
this script never reruns retrieval, never trains/retrains any model, never
calls an LLM/GPT/API, and never calls `llm.query()`. It only:

1. reconstructs each of the 6 selected queries' *exact* original k=10
   retrieved context from already-exported pilot artifacts
   (`mutation_input_passages.csv` + `clean_context_passages.csv`);
2. replaces *only* the 5 poisoned passages' text (matched by `poison_slot`)
   with the v2 file's `rewritten_text`, leaving every clean passage and
   k=10 membership untouched;
3. re-scores that fixed passage list against all 3 defenses with the same
   already-trained defense scorers `run_text_mutation_fixed_context_eval.py`
   uses -- imported and reused unmodified, not copied;
4. loads the v1 `ragdefender_targeted` family's *already-computed* scores
   from `evaluation/targeted_family_bundle_scores.csv` (never re-scored) to
   build a per-query v2-vs-v1 comparison, and reports each row's `delta_*`
   vs. its own freshly-computed baseline for this run.

**Output is written to a brand-new directory
(`evaluation_ragdefender_v2/`) and never mutates any existing file** under
`evaluation/` or `evaluation_normalized/` -- this is a comparison run, not a
replacement of the prior results.

Usage:
    python scripts/run_ragdefender_v2_semantic_support_eval.py \\
        --pilot_dir manual_text_mutation_pilot/hotpotqa_50q_k10 \\
        --bundle_dir manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1 \\
        --out_dir manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/evaluation_ragdefender_v2
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import run_text_mutation_fixed_context_eval as base_eval  # noqa: E402
import run_targeted_mutation_bundle_1_eval as v1_eval  # noqa: E402

DEFAULT_PILOT_DIR = base_eval.DEFAULT_PILOT_DIR
DEFAULT_BUNDLE_DIR = os.path.join(DEFAULT_PILOT_DIR, "mutation_bundle_1")
DEFAULT_ML_MODEL_PATH = base_eval.DEFAULT_ML_MODEL_PATH
DEFAULT_V2_FILENAME = "ragdefender-v2_semantic_support.jsonl.txt"
DEFAULT_V1_SCORES_PATH = os.path.join(DEFAULT_BUNDLE_DIR, "evaluation", "targeted_family_bundle_scores.csv")

V2_SPEC: Dict = {
    "filename": DEFAULT_V2_FILENAME,
    "intended_defense": "ragdefender",
    "passage_list_keys": ("rewritten_passages",),
    "text_field_keys": ("rewritten_text",),
}
V1_FAMILY_KEY = "ragdefender_targeted"
V2_FAMILY_KEY = "ragdefender_v2_semantic_support"
DEFENSE_NAMES = ("ragdefender", "filterrag", "ml_filterrag")


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(statistics.fmean(values)) if values else None


def _fmt(v) -> str:
    return base_eval._fmt(v)  # noqa: SLF001 -- shared formatting helper, not private state


# ---------------------------------------------------------------------------
# 1. Load the v1 family's already-computed scores (never re-scored).
# ---------------------------------------------------------------------------

def load_v1_scores(path: str) -> Dict[str, Dict]:
    """Reads the already-computed `targeted_family_bundle_scores.csv` from
    the v1 targeted-mutation evaluation, restricted to
    `family == "ragdefender_targeted"`, keyed by `query_id`. Returns `{}` if
    the file is absent or has no matching rows -- never fabricates a
    comparison row. Numeric-looking string values are coerced to float;
    everything else (including `""`) is left as `None`/string."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, Dict] = {}
    for r in rows:
        if r.get("family") != V1_FAMILY_KEY:
            continue
        coerced: Dict = {}
        for k, v in r.items():
            if v in ("", None):
                coerced[k] = None
            else:
                try:
                    coerced[k] = float(v)
                except ValueError:
                    coerced[k] = v
        out[r["query_id"]] = coerced
    return out


# ---------------------------------------------------------------------------
# 2. Comparison rows: v2 fresh score vs. v1's already-published score, both
#    also compared against v2's own freshly-computed baseline.
# ---------------------------------------------------------------------------

COMPARISON_METRIC_KEYS = (
    "ragdefender_removed_poison", "ragdefender_top_pair_pp", "ragdefender_residual_poison_fraction",
    "filterrag_removed_poison", "filterrag_mean_freq_density_poison",
    "ml_removed_poison_t04", "ml_mean_poison_probability",
)

COMPARISON_FIELDS = (
    ["query_id", "question", "target_wrong_answer"]
    + [f"baseline_{k}" for k in COMPARISON_METRIC_KEYS]
    + [f"v1_{k}" for k in COMPARISON_METRIC_KEYS]
    + [f"v2_{k}" for k in COMPARISON_METRIC_KEYS]
    + [f"delta_v1_vs_baseline_{k}" for k in COMPARISON_METRIC_KEYS]
    + [f"delta_v2_vs_baseline_{k}" for k in COMPARISON_METRIC_KEYS]
    + [f"delta_v2_vs_v1_{k}" for k in COMPARISON_METRIC_KEYS]
    + ["v2_improved_over_v1_on_ragdefender"]
)


def build_comparison_row(
    qid: str, question: str, target_wrong_answer: str,
    baseline_metrics: Dict, v1_row: Optional[Dict], v2_row: Dict,
) -> Dict:
    row: Dict = {"query_id": qid, "question": question, "target_wrong_answer": target_wrong_answer}
    for k in COMPARISON_METRIC_KEYS:
        b = baseline_metrics.get(k)
        v1 = v1_row.get(k) if v1_row is not None else None
        v2 = v2_row.get(k)
        row[f"baseline_{k}"] = b
        row[f"v1_{k}"] = v1
        row[f"v2_{k}"] = v2
        row[f"delta_v1_vs_baseline_{k}"] = (v1 - b) if (v1 is not None and b is not None) else None
        row[f"delta_v2_vs_baseline_{k}"] = (v2 - b) if (v2 is not None and b is not None) else None
        row[f"delta_v2_vs_v1_{k}"] = (v2 - v1) if (v2 is not None and v1 is not None) else None
    # "improved" for RAGDefender means v2 removed *fewer* poison passages
    # than v1 (i.e. evaded more) -- lower removed_poison / lower top_pair_pp
    # is the attacker's goal for its own intended target.
    d_removed = row["delta_v2_vs_v1_ragdefender_removed_poison"]
    row["v2_improved_over_v1_on_ragdefender"] = bool(d_removed is not None and d_removed < 0)
    return row


# ---------------------------------------------------------------------------
# 3. Report.
# ---------------------------------------------------------------------------

def build_report(
    *, pilot_dir: str, bundle_dir: str, v2_path: str, v1_scores_path: str, ml_model_path: str,
    query_ids: Sequence[str], baseline_rows: Sequence[Dict], v2_bundle_rows: Sequence[Dict],
    v2_delta_rows: Sequence[Dict], comparison_rows: Sequence[Dict], doc_id_mismatches: Sequence[Dict],
    selected_queries: Dict[str, Dict], v1_available: bool,
) -> str:
    lines: List[str] = []
    lines.append("# RAGDefender v2 (Semantic Support) Mutation -- Fixed-Context Cross-Defense Evaluation Report")
    lines.append("")
    lines.append(
        "Fixed-retrieval, cross-defense evaluation of the **re-iterated** RAGDefender-targeted GPT "
        f"mutation family in `{os.path.relpath(v2_path, REPO_ROOT)}` "
        "(`mutation_family: ragdefender_semantic_support_v2`), scored against **all three** defenses "
        "(RAGDefender, semantic FilterRAG epsilon=0.2, ML-FilterRAG-top-k at t in {0.35, 0.4, 0.5}) and "
        "compared against the v1 `ragdefender_targeted` family's already-published scores. Retrieval "
        "membership/order is identical between the baseline and every mutated context for a given "
        "query; only the 5 poisoned passages' text differs; clean passages are byte-identical "
        "everywhere. **This is a separate, additive comparison run -- no file under "
        f"`{os.path.relpath(os.path.join(bundle_dir, 'evaluation'), REPO_ROOT)}/` or "
        f"`{os.path.relpath(os.path.join(bundle_dir, 'evaluation_normalized'), REPO_ROOT)}/` was read "
        "for scoring purposes (only `targeted_family_bundle_scores.csv` was read, read-only, for the "
        "v1 comparison columns) or modified by this script.**"
    )
    lines.append("")
    lines.append("## Artifact paths used")
    lines.append("")
    lines.append(f"- `{os.path.relpath(v2_path, REPO_ROOT)}` (v2 mutation family, intended target: ragdefender)")
    lines.append(
        f"- `{os.path.relpath(v1_scores_path, REPO_ROOT)}` (v1 `ragdefender_targeted` scores, "
        f"read-only, for comparison only -- {'found' if v1_available else 'NOT FOUND -- v1 comparison columns are blank'})"
    )
    lines.append(f"- `{os.path.join(pilot_dir, 'selected_queries.csv')}`")
    lines.append(f"- `{os.path.join(pilot_dir, 'mutation_input_passages.csv')}`")
    lines.append(f"- `{os.path.join(pilot_dir, 'clean_context_passages.csv')}`")
    lines.append(f"- `{ml_model_path}` (trained ML-FilterRAG-top-k random-forest classifier, loaded read-only)")
    lines.append("")

    if doc_id_mismatches:
        lines.append("## Data-integrity note: v2 file `doc_id` mismatches")
        lines.append("")
        lines.append(
            f"{len(doc_id_mismatches)} passage(s) in the v2 file have a `doc_id` field that does not "
            "match `mutation_input_passages.csv`'s authoritative `(query_id, poison_slot) -> doc_id` "
            "mapping. Passage identity is resolved by `poison_slot` against that CSV (never by the "
            "file's own `doc_id`), so this does not affect which passage's text was mutated:"
        )
        lines.append("")
        lines.append("| query_id | poison_slot | file doc_id | csv doc_id |")
        lines.append("|---|---:|---|---|")
        for m in doc_id_mismatches:
            lines.append(f"| `{m['query_id']}` | {m['poison_slot']} | `{m['file_doc_id']}` | `{m['csv_doc_id']}` |")
        lines.append("")
    else:
        lines.append(
            "## Data-integrity note: v2 file `doc_id` mismatches\n\nNone -- every v2 passage's "
            "`doc_id` matches `mutation_input_passages.csv`'s authoritative "
            "`(query_id, poison_slot) -> doc_id` mapping.\n"
        )

    lines.append("## Queries evaluated")
    lines.append("")
    for qid in query_ids:
        q = selected_queries.get(qid, {})
        lines.append(
            f"- `{qid}` ({q.get('selection_role', 'n/a')}) -- {q.get('question', 'n/a')} "
            f"(target wrong answer: {q.get('target_wrong_answer', 'n/a')})"
        )
    lines.append("")

    lines.append("## v2 vs. baseline vs. v1 -- per query, all 3 defenses")
    lines.append("")
    lines.append(
        "| query_id | RAGDefender removed_poison (base/v1/v2) | RAGDefender top_pair_pp (base/v1/v2) | "
        "FilterRAG removed_poison (base/v1/v2) | ML-FilterRAG t0.4 removed_poison (base/v1/v2) | "
        "ML mean_poison_probability (base/v1/v2) | v2 improved vs v1 on RAGDefender? |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in comparison_rows:
        lines.append(
            f"| `{r['query_id']}` | "
            f"{_fmt(r['baseline_ragdefender_removed_poison'])}/{_fmt(r['v1_ragdefender_removed_poison'])}/{_fmt(r['v2_ragdefender_removed_poison'])} | "
            f"{_fmt(r['baseline_ragdefender_top_pair_pp'])}/{_fmt(r['v1_ragdefender_top_pair_pp'])}/{_fmt(r['v2_ragdefender_top_pair_pp'])} | "
            f"{_fmt(r['baseline_filterrag_removed_poison'])}/{_fmt(r['v1_filterrag_removed_poison'])}/{_fmt(r['v2_filterrag_removed_poison'])} | "
            f"{_fmt(r['baseline_ml_removed_poison_t04'])}/{_fmt(r['v1_ml_removed_poison_t04'])}/{_fmt(r['v2_ml_removed_poison_t04'])} | "
            f"{_fmt(r['baseline_ml_mean_poison_probability'])}/{_fmt(r['v1_ml_mean_poison_probability'])}/{_fmt(r['v2_ml_mean_poison_probability'])} | "
            f"{r['v2_improved_over_v1_on_ragdefender']} |"
        )
    lines.append("")

    # ---- Aggregate answers -------------------------------------------------
    mean_v2_removed = _mean([r["ragdefender_removed_poison"] for r in v2_bundle_rows])
    mean_v1_removed = _mean([r["v1_ragdefender_removed_poison"] for r in comparison_rows])
    mean_v2_delta_removed = _mean([d["delta_ragdefender_removed_poison"] for d in v2_delta_rows])
    mean_v1_delta_removed = _mean([r["delta_v1_vs_baseline_ragdefender_removed_poison"] for r in comparison_rows])

    n_improved = sum(1 for r in comparison_rows if r["v2_improved_over_v1_on_ragdefender"])
    any_v2_weakened_ragdefender = any(
        d.get("delta_ragdefender_removed_poison") is not None and d["delta_ragdefender_removed_poison"] < 0
        for d in v2_delta_rows
    )

    cross_weakened = {}
    for defense, delta_key in (
        ("filterrag", "delta_filterrag_removed_poison"),
        ("ml_filterrag", "delta_ml_removed_poison_t04"),
    ):
        vals = [d[delta_key] for d in v2_delta_rows if d.get(delta_key) is not None]
        cross_weakened[defense] = _mean(vals)

    best_row = min(
        v2_bundle_rows,
        key=lambda r: (
            r["ragdefender_removed_poison"] + r["filterrag_removed_poison"] + r["ml_removed_poison_t04"],
            r["ml_mean_poison_probability"] if r["ml_mean_poison_probability"] is not None else 1.0,
        ),
    )

    lines.append("## Answers")
    lines.append("")
    lines.append(
        f"**1. Did v2 weaken RAGDefender (its intended target) at fixed retrieval?** "
        f"{'Yes' if any_v2_weakened_ragdefender else 'No'} -- mean delta_ragdefender_removed_poison "
        f"(v2 vs. its own baseline) = {_fmt(mean_v2_delta_removed)} across {len(v2_delta_rows)} queries "
        f"(v1's equivalent mean delta was {_fmt(mean_v1_delta_removed)})."
    )
    lines.append("")
    lines.append(
        f"**2. Is v2 stronger against RAGDefender than v1?** "
        f"v2 improved over v1 (removed fewer poison passages) on {n_improved}/{len(comparison_rows)} "
        f"queries. Mean RAGDefender removed_poison: v1={_fmt(mean_v1_removed)}, v2={_fmt(mean_v2_removed)} "
        f"({'v2 is stronger (lower mean removal)' if (mean_v2_removed is not None and mean_v1_removed is not None and mean_v2_removed < mean_v1_removed) else ('v1 is stronger or tied' if (mean_v2_removed is not None and mean_v1_removed is not None and mean_v2_removed >= mean_v1_removed) else 'comparison unavailable')})."
    )
    lines.append("")
    lines.append(
        f"**3. Did v2 unexpectedly weaken the other two defenses?** "
        f"Mean delta_removed_poison (v2 vs. its own baseline): FilterRAG={_fmt(cross_weakened.get('filterrag'))}, "
        f"ML-FilterRAG t=0.4={_fmt(cross_weakened.get('ml_filterrag'))}."
    )
    lines.append("")
    lines.append(
        f"**4. Which query is the best candidate for a full-retrieval rerun of v2?** "
        f"`{best_row['query_id']}` -- lowest combined removal across the three defenses "
        f"(RAGDefender={best_row['ragdefender_removed_poison']}, FilterRAG={best_row['filterrag_removed_poison']}, "
        f"ML-FilterRAG t0.4={best_row['ml_removed_poison_t04']}, out of {best_row['N_retrieved_poison']} "
        "retrieved poison passages)."
    )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- v1 comparison columns rely on `targeted_family_bundle_scores.csv` exactly as already "
        "published by the prior evaluation run (not re-scored here); if that file is regenerated with "
        "different model/library versions, the v1 columns in this report would reflect whatever that "
        "file contains at the time this script is run."
    )
    lines.append(
        "- All models (SLM, LM, RAGDefender embedder, ML-FilterRAG classifier) run on "
        f"device=`{base_eval.DEVICE}` for determinism, matching prior mutation_bundle_1 evaluations; "
        "baseline metrics here are a fresh, independent re-scoring of the fixed passages for this "
        "evaluation's own internal reference, not a claim of bit-identical reproduction of any other run."
    )
    lines.append("")
    lines.append("## Process confirmation")
    lines.append("")
    lines.append("- No GPT/API calls were made.")
    lines.append("- No `llm.query()` calls were made.")
    lines.append("- Retrieval was not rerun (k=10 membership reconstructed verbatim from existing pilot CSV artifacts).")
    lines.append("- No model was trained or retrained (every model loaded read-only for inference).")
    lines.append("- No defense code (`defense/*.py`) was modified; every defense function used here is called unmodified via `scripts/run_text_mutation_fixed_context_eval.py`.")
    lines.append(
        "- No existing file under `evaluation/` or `evaluation_normalized/` was modified; "
        "`targeted_family_bundle_scores.csv` was only read, never written to; all new output goes to "
        "a separate `evaluation_ragdefender_v2/` directory."
    )
    lines.append("- Only text substitution on the already-provided v2 mutation file was applied; no mutations were generated by this script.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Orchestration.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot_dir", default=os.path.join(REPO_ROOT, DEFAULT_PILOT_DIR))
    parser.add_argument("--bundle_dir", default=None)
    parser.add_argument("--v2_filename", default=DEFAULT_V2_FILENAME)
    parser.add_argument("--v1_scores_path", default=None)
    parser.add_argument("--ml_model_path", default=os.path.join(REPO_ROOT, DEFAULT_ML_MODEL_PATH))
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    pilot_dir = args.pilot_dir
    bundle_dir = args.bundle_dir or os.path.join(pilot_dir, "mutation_bundle_1")
    out_dir = args.out_dir or os.path.join(bundle_dir, "evaluation_ragdefender_v2")
    v2_path = os.path.join(bundle_dir, args.v2_filename)
    v1_scores_path = args.v1_scores_path or os.path.join(bundle_dir, "evaluation", "targeted_family_bundle_scores.csv")

    # Guard: never allow this script to write into the existing results dirs.
    for forbidden in (
        os.path.join(bundle_dir, "evaluation"),
        os.path.join(bundle_dir, "evaluation_normalized"),
    ):
        if os.path.abspath(out_dir) == os.path.abspath(forbidden):
            raise ValueError(
                f"--out_dir must not be the existing '{forbidden}' directory; this script only "
                "writes new, separate comparison output."
            )

    selected_queries = base_eval.load_selected_queries(os.path.join(pilot_dir, "selected_queries.csv"))
    poison_by_query = base_eval.load_mutation_input_passages(os.path.join(pilot_dir, "mutation_input_passages.csv"))
    clean_by_query = base_eval.load_clean_context_passages(os.path.join(pilot_dir, "clean_context_passages.csv"))

    v2_records = v1_eval.parse_family_file(v2_path, V2_SPEC)
    doc_id_mismatches = v1_eval.check_doc_id_consistency(V2_FAMILY_KEY, v2_records, poison_by_query)
    if doc_id_mismatches:
        print(f"[run_ragdefender_v2_semantic_support_eval] WARNING: {len(doc_id_mismatches)} v2 doc_id mismatch(es) detected (see report); scoring used poison_slot-based identity regardless.")

    v1_scores_by_qid = load_v1_scores(v1_scores_path)
    v1_available = bool(v1_scores_by_qid)
    if not v1_available:
        print(f"[run_ragdefender_v2_semantic_support_eval] WARNING: no v1 '{V1_FAMILY_KEY}' rows found at {v1_scores_path!r}; v1 comparison columns will be blank.")

    query_ids = sorted(v2_records.keys())
    for qid in query_ids:
        if qid not in selected_queries or qid not in poison_by_query or qid not in clean_by_query:
            raise ValueError(f"query_id={qid!r} from the v2 mutation file is missing required pilot rows.")

    print(f"[run_ragdefender_v2_semantic_support_eval] loading models (device={base_eval.DEVICE})...")
    models = base_eval.load_models(args.ml_model_path)

    baseline_rows: List[Dict] = []
    v2_bundle_rows: List[Dict] = []
    v2_delta_rows: List[Dict] = []
    comparison_rows: List[Dict] = []

    for qid in query_ids:
        q = selected_queries[qid]
        question = q["question"]
        target_wrong_answer = q["target_wrong_answer"]

        original_context = base_eval.build_original_context(poison_by_query[qid], clean_by_query[qid])
        print(f"[run_ragdefender_v2_semantic_support_eval] scoring baseline for {qid} ({len(original_context)} passages)...")
        baseline_metrics = base_eval.score_context(question, original_context, models)
        baseline_rows.append({
            "query_id": qid, "k": 10, "selection_role": q.get("selection_role", ""),
            "question": question, "target_wrong_answer": target_wrong_answer,
            **baseline_metrics,
        })

        rec = v2_records[qid]
        bundle = v1_eval.family_record_to_bundle(rec)
        mutated_context = base_eval.build_mutated_context(original_context, poison_by_query[qid], bundle)
        base_eval.assert_same_k10_membership(original_context, mutated_context)
        for orig, mut in zip(original_context, mutated_context):
            if not orig.is_poison and orig.text != mut.text:
                raise AssertionError(
                    f"{qid}/{V2_FAMILY_KEY}: clean passage doc_id={orig.doc_id!r} text changed "
                    "(must remain unchanged)."
                )

        print(f"[run_ragdefender_v2_semantic_support_eval] scoring {qid} / {V2_FAMILY_KEY}...")
        v2_metrics = base_eval.score_context(question, mutated_context, models)
        v2_row = {
            "query_id": qid, "k": 10, "family": V2_FAMILY_KEY, "bundle_id": V2_FAMILY_KEY,
            "intended_target_defense": V2_SPEC["intended_defense"],
            "selection_role": q.get("selection_role", ""),
            "question": question, "target_wrong_answer": target_wrong_answer,
            **v2_metrics,
        }
        v2_bundle_rows.append(v2_row)

        deltas = base_eval.compute_deltas(baseline_metrics, v2_metrics)
        v2_delta_rows.append({
            "query_id": qid, "k": 10, "family": V2_FAMILY_KEY, "bundle_id": V2_FAMILY_KEY,
            "intended_target_defense": V2_SPEC["intended_defense"],
            **deltas,
        })

        v1_row = v1_scores_by_qid.get(qid)
        comparison_rows.append(
            build_comparison_row(qid, question, target_wrong_answer, baseline_metrics, v1_row, v2_metrics)
        )

    baseline_fields = [
        "query_id", "k", "selection_role", "question", "target_wrong_answer",
        "N_retrieved_poison", "N_retrieved_clean",
    ] + list(base_eval._NUMERIC_METRIC_KEYS)
    v2_bundle_fields = ["family", "bundle_id", "intended_target_defense"] + baseline_fields
    v2_delta_fields = (
        ["query_id", "k", "family", "bundle_id", "intended_target_defense"]
        + [f"delta_{k}" for k in base_eval._NUMERIC_METRIC_KEYS]
        + list(base_eval.DELTA_ALIASES.keys())
    )

    os.makedirs(out_dir, exist_ok=True)
    base_eval.write_csv(os.path.join(out_dir, "ragdefender_v2_bundle_scores.csv"), v2_bundle_fields, v2_bundle_rows)
    base_eval.write_csv(os.path.join(out_dir, "ragdefender_v2_bundle_deltas.csv"), v2_delta_fields, v2_delta_rows)
    base_eval.write_csv(os.path.join(out_dir, "ragdefender_v2_vs_v1_comparison.csv"), list(COMPARISON_FIELDS), comparison_rows)

    report = build_report(
        pilot_dir=pilot_dir, bundle_dir=bundle_dir, v2_path=v2_path, v1_scores_path=v1_scores_path,
        ml_model_path=args.ml_model_path, query_ids=query_ids, baseline_rows=baseline_rows,
        v2_bundle_rows=v2_bundle_rows, v2_delta_rows=v2_delta_rows, comparison_rows=comparison_rows,
        doc_id_mismatches=doc_id_mismatches, selected_queries=selected_queries, v1_available=v1_available,
    )
    report_path = os.path.join(out_dir, "RAGDEFENDER_V2_SEMANTIC_SUPPORT_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(
        f"[run_ragdefender_v2_semantic_support_eval] wrote {len(baseline_rows)} baseline row(s), "
        f"{len(v2_bundle_rows)} v2 bundle row(s), {len(v2_delta_rows)} v2 delta row(s), "
        f"{len(comparison_rows)} comparison row(s) to {out_dir}"
    )
    print(
        f"[run_ragdefender_v2_semantic_support_eval] SLM generation calls: "
        f"{models.memo_slm_answer_fn.calls} (cache hits: {models.memo_slm_answer_fn.cache_hits})"
    )
    print(
        f"[run_ragdefender_v2_semantic_support_eval] LM perplexity calls: "
        f"{models.memo_causal_scorer.calls} (cache hits: {models.memo_causal_scorer.cache_hits})"
    )


if __name__ == "__main__":
    main()
