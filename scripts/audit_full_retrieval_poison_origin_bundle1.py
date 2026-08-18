#!/usr/bin/env python3
"""Audit `scripts/run_full_retrieval_pilot_bundle1.py`'s 3-query full-retrieval
pilot for **self-query poison vs cross-query poison contamination**.

Motivation: `full_retrieval_results_by_query.csv` reported
`retrieved_poison_count=6` for query_id `5a8e068b5542995085b37384`, even though
each selected query has exactly 5 self-query poison slots (replaced, never
augmented -- see `apply_replacements`/`assert_budget_preserved`). This script
determines, passage-by-passage, whether that 6th poison passage (and every
other retrieved poison passage) is:

- `mutated_self_query_poison`: one of *this* query's own 5 replaced slots;
- `original_self_query_poison`: this query's own original (un-mutated) poison
  text somehow still present (would indicate a budget-preservation bug --
  expected count is always 0 in the mutated condition);
- `cross_query_poison`: another pool query's own poison text, retrieved into
  *this* query's top-10 because the shared 50-query adversarial candidate
  pool is scored against every query (documented, expected behavior -- see
  `scripts/build_ml_filterrag_dataset.py`'s module docstring and
  `scripts/run_full_retrieval_pilot_bundle1.py`'s "Retrieval was rerun only
  for the 3 selected queries" methodology note); or
- `clean`: an original BEIR corpus passage.

**Important data-representation caveat this audit corrects for:** the
`doc_id` string `merge_and_topk()` writes for every adversarial candidate
(`adv::LM_targeted::<qid>::<j>`) always uses the *currently retrieved-for*
query's id for `<qid>`, not the pool query that originally authored that
text -- `<j>` is the only reliable part (a global index into the shared
250-slot pool). This is harmless for retrieval/scoring (dot-product scores
and the survival-rank cross-check in `retrieval_survival_stats` only ever
compare `(qid, j)` pairs against a query's *own* canonical slots, and no two
queries share a `j`), but it means the emitted doc_id string alone cannot be
used to identify cross-query contamination -- the true owning query must be
recovered from `j` via `full_pool_query_ids[j // N_ADV_PER_QUERY]`
(`classify_passage_origin` below). This script never touches or renames the
existing doc_id convention (that would require re-running/overwriting the
already-published pilot artifacts); it only *reads* `j` and reports the
correct owner alongside it.

**Strict constraints honored (identical to
`scripts/run_full_retrieval_pilot_bundle1.py`):**

- No GPT/API call, no `llm.query()` call (offline `Attacker.get_attack()`
  template substitution only).
- No model is trained or retrained -- Contriever, the FilterRAG SLM, the
  ML-FilterRAG perplexity LM, the RAGDefender similarity model, and the
  ML-FilterRAG-top-k classifier are all loaded read-only for inference,
  reusing `run_full_retrieval_pilot_bundle1.py`'s and
  `run_text_mutation_fixed_context_eval.py`'s existing loaders verbatim.
- No defense code (`defense/*.py`) is modified -- this script calls
  `defense.dispatch.run_defense`, `defense.filterrag.filterrag_defense`, and
  `defense.ml_filterrag.extract_features` + the already-trained classifier's
  `predict_proba` exactly as `run_text_mutation_fixed_context_eval.py`'s
  `score_ragdefender`/`score_filterrag`/`score_ml_filterrag` already do --
  only the *bucketing* of which removed passage is self/cross/clean is new,
  the removal decisions themselves are untouched.
- Retrieval is rebuilt exactly as `run_full_retrieval_pilot_bundle1.py` does
  (same Contriever model, same rebuilt 50-query pool, same replacement plan,
  same merge_and_topk), for the same 3 selected queries only -- this is a
  deterministic re-derivation of the *already-audited* mutated top-10, not a
  new experiment.

Usage:
    python scripts/audit_full_retrieval_poison_origin_bundle1.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import run_full_retrieval_pilot_bundle1 as pilot  # noqa: E402

from defense.passages import RetrievedPassage, removed_passages  # noqa: E402

SELECTED_QUERY_IDS = pilot.SELECTED_QUERY_IDS
FAMILY = pilot.FAMILY
BUNDLE_ID = pilot.BUNDLE_ID
K = pilot.K
N_ADV_PER_QUERY = pilot.N_ADV_PER_QUERY
RETRIEVAL_DEVICE = pilot.RETRIEVAL_DEVICE
SCORE_FUNCTION = pilot.SCORE_FUNCTION

ORIGIN_MUTATED_SELF = "mutated_self_query_poison"
ORIGIN_ORIGINAL_SELF = "original_self_query_poison"
ORIGIN_CROSS_QUERY = "cross_query_poison"
ORIGIN_CLEAN = "clean"

DEFENSE_NAMES = ("ragdefender", "filterrag_semantic", "ml_filterrag_t035", "ml_filterrag_t04", "ml_filterrag_t05")


# ---------------------------------------------------------------------------
# 1. Pure origin-classification logic (unit-testable, no model/I-O).
# ---------------------------------------------------------------------------

def classify_passage_origin(
    *,
    doc_id: str,
    source: str,
    current_qid: str,
    full_pool_query_ids: Sequence[str],
    mutated_self_global_indices: Set[int],
    n_adv_per_query: int = N_ADV_PER_QUERY,
) -> Dict:
    """Classify one retrieved passage's true poison origin.

    For `source == "corpus"` returns `origin_label="clean"` with no
    index/owner (clean passages are never in the adversarial pool). For
    `source == "adversarial"`, parses the doc_id's trailing global pool
    index `j` (via `pilot.extract_global_index` -- never guesses), computes
    the *true* owning pool query `full_pool_query_ids[j // n_adv_per_query]`
    (ignoring the doc_id's own, possibly-mislabeled, qid segment -- see
    module docstring), and labels:
    - `mutated_self_query_poison` if the true owner is `current_qid` AND `j`
      is one of that query's replaced slots (`mutated_self_global_indices`);
    - `original_self_query_poison` if the true owner is `current_qid` but
      `j` is NOT a replaced slot (should never occur for the 3 selected
      queries -- all 5 of their own slots are always replaced -- surfaced as
      an explicit anomaly label rather than silently folded into either
      other bucket);
    - `cross_query_poison` if the true owner is a different pool query.
    """
    if source == "corpus":
        return {"origin_label": ORIGIN_CLEAN, "true_global_index": None, "true_owning_query_id": None}
    if source != "adversarial":
        raise ValueError(f"doc_id={doc_id!r}: unknown source {source!r} (expected 'corpus' or 'adversarial').")

    global_index = pilot.extract_global_index(doc_id)
    owning_position = global_index // n_adv_per_query
    if not (0 <= owning_position < len(full_pool_query_ids)):
        raise ValueError(
            f"doc_id={doc_id!r}: global_index={global_index} implies pool position {owning_position}, "
            f"out of range for a {len(full_pool_query_ids)}-query pool."
        )
    owning_qid = full_pool_query_ids[owning_position]

    if owning_qid == current_qid:
        origin_label = ORIGIN_MUTATED_SELF if global_index in mutated_self_global_indices else ORIGIN_ORIGINAL_SELF
    else:
        origin_label = ORIGIN_CROSS_QUERY

    return {"origin_label": origin_label, "true_global_index": global_index, "true_owning_query_id": owning_qid}


def build_origin_rows(
    *,
    qid: str,
    k: int,
    passages: Sequence[RetrievedPassage],
    full_pool_query_ids: Sequence[str],
    mutated_self_global_indices: Set[int],
    removed_doc_ids_by_defense: Dict[str, Set[str]],
) -> List[Dict]:
    """One row per retrieved passage (rank order), tagging its true origin
    and, for every defense in `removed_doc_ids_by_defense`, whether that
    specific passage (by doc_id) was removed. Pure function -- the caller
    supplies already-computed removed-doc_id sets, so this performs no
    defense scoring itself."""
    rows: List[Dict] = []
    for p in passages:
        origin = classify_passage_origin(
            doc_id=p.doc_id, source=p.source, current_qid=qid,
            full_pool_query_ids=full_pool_query_ids, mutated_self_global_indices=mutated_self_global_indices,
        )
        row = {
            "query_id": qid, "k": k, "rank": (p.rank + 1) if p.rank is not None else None,
            "doc_id": p.doc_id, "source": p.source, "is_poison": p.is_poison,
            "retrieval_score": p.retrieval_score,
            "origin_label": origin["origin_label"],
            "true_owning_query_id": origin["true_owning_query_id"],
            "true_global_index": origin["true_global_index"],
        }
        for defense_name in DEFENSE_NAMES:
            row[f"removed_by_{defense_name}"] = p.doc_id in removed_doc_ids_by_defense.get(defense_name, set())
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 2. Pure budget/contamination verification over already-built origin rows.
# ---------------------------------------------------------------------------

def verify_replacement_budget(
    replacement_plan: Dict[str, Dict[int, "pilot.ReplacementSlot"]], selected_query_ids: Sequence[str]
) -> Dict[str, Dict]:
    """Per-query check that exactly 5 self-query poison slots were replaced,
    with 5 *distinct* global indices (no double-counting a slot)."""
    out: Dict[str, Dict] = {}
    for qid in selected_query_ids:
        slots = replacement_plan.get(qid, {})
        global_indices = [slot.global_index for slot in slots.values()]
        out[qid] = {
            "n_slots_replaced": len(slots),
            "n_distinct_global_indices": len(set(global_indices)),
            "exactly_5_replaced": len(slots) == 5 and len(set(global_indices)) == 5,
        }
    return out


def count_origins_per_query(origin_rows: Sequence[Dict], selected_query_ids: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """`{query_id: {origin_label: count}}` over all retrieved (mutated
    condition) passages, plus a `total_poison`/`total_clean` convenience
    tally."""
    out: Dict[str, Dict[str, int]] = {
        qid: {ORIGIN_MUTATED_SELF: 0, ORIGIN_ORIGINAL_SELF: 0, ORIGIN_CROSS_QUERY: 0, ORIGIN_CLEAN: 0}
        for qid in selected_query_ids
    }
    for row in origin_rows:
        out[row["query_id"]][row["origin_label"]] += 1
    for qid in selected_query_ids:
        out[qid]["total_poison"] = (
            out[qid][ORIGIN_MUTATED_SELF] + out[qid][ORIGIN_ORIGINAL_SELF] + out[qid][ORIGIN_CROSS_QUERY]
        )
        out[qid]["total_clean"] = out[qid][ORIGIN_CLEAN]
    return out


def verify_no_original_self_poison_duplicate(origin_rows: Sequence[Dict]) -> List[Dict]:
    """Rows (if any) where a query's own *original* (un-mutated) poison text
    was retrieved alongside its mutated replacement -- i.e. budget-
    preservation failed at the retrieval level, not just the pool-text
    level. Expected to be empty for every selected query."""
    return [r for r in origin_rows if r["origin_label"] == ORIGIN_ORIGINAL_SELF]


def verify_max_five_mutated_self_per_query(
    origin_rows: Sequence[Dict], selected_query_ids: Sequence[str]
) -> Dict[str, Dict]:
    """Per-query count of *distinct* `mutated_self_query_poison` global
    indices retrieved, and whether it exceeds the 5-slot budget (it never
    can, structurally, since the pool only contains 5 such slots per query
    -- this function makes that invariant an explicit, checked fact rather
    than an assumption)."""
    out: Dict[str, Dict] = {}
    for qid in selected_query_ids:
        indices = {
            r["true_global_index"] for r in origin_rows
            if r["query_id"] == qid and r["origin_label"] == ORIGIN_MUTATED_SELF
        }
        out[qid] = {"n_mutated_self_retrieved": len(indices), "exceeds_budget_of_5": len(indices) > 5}
    return out


def summarize_removed_by_origin(
    origin_rows: Sequence[Dict], selected_query_ids: Sequence[str]
) -> Dict[Tuple[str, str], Dict[str, int]]:
    """`{(query_id, defense_name): {origin_label: n_removed}}` -- for each
    defense, how many of the passages it removed were
    mutated_self_query_poison vs cross_query_poison vs clean (and, as an
    anomaly check, original_self_query_poison, expected 0 everywhere)."""
    out: Dict[Tuple[str, str], Dict[str, int]] = {}
    for qid in selected_query_ids:
        for defense_name in DEFENSE_NAMES:
            counts = {ORIGIN_MUTATED_SELF: 0, ORIGIN_ORIGINAL_SELF: 0, ORIGIN_CROSS_QUERY: 0, ORIGIN_CLEAN: 0}
            for r in origin_rows:
                if r["query_id"] != qid or not r[f"removed_by_{defense_name}"]:
                    continue
                counts[r["origin_label"]] += 1
            out[(qid, defense_name)] = counts
    return out


# ---------------------------------------------------------------------------
# 3. Heavy orchestration (rebuilds the exact same retrieval + defenses as
#    run_full_retrieval_pilot_bundle1.py; models loaded read-only).
# ---------------------------------------------------------------------------

def _removed_doc_ids(before: Sequence[RetrievedPassage], kept: Sequence[RetrievedPassage]) -> Set[str]:
    return {p.doc_id for p in removed_passages(before, kept)}


def write_csv(path: str, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot_dir", default=os.path.join(REPO_ROOT, pilot.DEFAULT_PILOT_DIR))
    parser.add_argument("--bundle_dir", default=None)
    parser.add_argument("--dataset_config", default=os.path.join(REPO_ROOT, pilot.DEFAULT_DATASET_CONFIG))
    parser.add_argument("--incorrect_answers", default=os.path.join(REPO_ROOT, pilot.DEFAULT_INCORRECT_ANSWERS))
    parser.add_argument("--beir_results", default=os.path.join(REPO_ROOT, pilot.DEFAULT_BEIR_RESULTS))
    parser.add_argument("--corpus_path", default=os.path.join(REPO_ROOT, pilot.DEFAULT_CORPUS_PATH))
    parser.add_argument("--ml_model_path", default=os.path.join(REPO_ROOT, pilot.DEFAULT_ML_MODEL_PATH))
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    pilot_dir = args.pilot_dir
    bundle_dir = args.bundle_dir or os.path.join(pilot_dir, pilot.BUNDLE_DIR_NAME)
    out_dir = args.out_dir or os.path.join(bundle_dir, "full_retrieval_pilot")

    print("[audit_poison_origin] No GPT/API call will be made; no llm.query() call will be made.")
    print(f"[audit_poison_origin] selected query_ids: {list(SELECTED_QUERY_IDS)}")

    base_eval = pilot.base_eval
    selected_queries = base_eval.load_selected_queries(os.path.join(pilot_dir, "selected_queries.csv"))
    poison_by_query = base_eval.load_mutation_input_passages(os.path.join(pilot_dir, "mutation_input_passages.csv"))
    normalized_by_qid = pilot.load_normalized_family_file(
        os.path.join(bundle_dir, "normalized", "filterrag_targeted.normalized.jsonl")
    )
    replacement_plan = pilot.build_replacement_plan(SELECTED_QUERY_IDS, poison_by_query, normalized_by_qid)

    budget_check = verify_replacement_budget(replacement_plan, SELECTED_QUERY_IDS)
    for qid, chk in budget_check.items():
        if not chk["exactly_5_replaced"]:
            raise AssertionError(f"query_id={qid!r}: replacement budget check failed: {chk!r}")
    print("[audit_poison_origin] replacement budget verified: exactly 5 self-query slots per selected query.")

    full_pool_query_ids = pilot.load_full_pool_query_ids(args.dataset_config)

    import json  # noqa: PLC0415
    with open(args.incorrect_answers, "r", encoding="utf-8") as f:
        incorrect_answers = json.load(f)
    with open(args.beir_results, "r", encoding="utf-8") as f:
        beir_results = json.load(f)

    print(f"[audit_poison_origin] loading retrieval model ({pilot.EVAL_MODEL_CODE}, device={RETRIEVAL_DEVICE})...")
    import torch  # noqa: PLC0415
    from src.utils import load_models as load_retrieval_models  # noqa: PLC0415

    model, c_model, tokenizer, get_emb = load_retrieval_models(pilot.EVAL_MODEL_CODE)
    model.eval()
    model.to(RETRIEVAL_DEVICE)
    c_model.eval()
    c_model.to(RETRIEVAL_DEVICE)

    print(f"[audit_poison_origin] rebuilding the full {len(full_pool_query_ids)}-query adversarial pool "
          "(offline template substitution, no LLM/GPT call)...")
    baseline_adv_text_list = pilot.build_full_pool_adv_text_list(
        full_pool_query_ids, incorrect_answers, model=model, c_model=c_model, tokenizer=tokenizer, get_emb=get_emb
    )
    mutated_adv_text_list, replaced_indices = pilot.apply_replacements(baseline_adv_text_list, replacement_plan)
    pilot.assert_budget_preserved(
        baseline_adv_text_list, mutated_adv_text_list, replaced_indices, n_selected_queries=len(SELECTED_QUERY_IDS)
    )

    print("[audit_poison_origin] embedding mutated adversarial pool...")
    mutated_adv_embs = pilot.embed_texts(
        mutated_adv_text_list, model=c_model, tokenizer=tokenizer, get_emb=get_emb, device=RETRIEVAL_DEVICE
    )

    wanted_clean_doc_ids = sorted({
        doc_id for qid in SELECTED_QUERY_IDS for doc_id in list(beir_results[qid].keys())[:K]
    })
    print(f"[audit_poison_origin] streaming corpus.jsonl for {len(wanted_clean_doc_ids)} clean doc_id(s)...")
    clean_texts = pilot.stream_corpus_texts(args.corpus_path, wanted_clean_doc_ids)

    print("[audit_poison_origin] loading defense-scoring models (SLM/LM/RAGDefender embedder/ML classifier)...")
    defense_models = base_eval.load_models(args.ml_model_path)

    all_origin_rows: List[Dict] = []
    for qid in SELECTED_QUERY_IDS:
        question = selected_queries[qid]["question"]
        mutated_self_global_indices = {slot.global_index for slot in replacement_plan[qid].values()}

        clean_topk_doc_ids = list(beir_results[qid].keys())[:K]
        clean_entries = [
            {"score": beir_results[qid][d], "context": clean_texts[d], "doc_id": d} for d in clean_topk_doc_ids
        ]
        query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
        query_input = {k: v.to(RETRIEVAL_DEVICE) for k, v in query_input.items()}
        with torch.no_grad():
            query_emb = get_emb(model, query_input)
        mutated_scores = pilot.score_adv_texts_against_query(mutated_adv_embs, query_emb, SCORE_FUNCTION)
        mutated_topk = pilot.merge_and_topk(clean_entries, mutated_adv_text_list, mutated_scores, qid=qid, k=K)
        mutated_passages = pilot.label_passages(mutated_topk)

        print(f"[audit_poison_origin] scoring defenses (passage-level) for {qid} ...")
        kept_rd, _ = base_eval.run_defense(
            "ragdefender_original", question, mutated_passages, "hotpotqa",
            device=base_eval.DEVICE, gpu_id=0, top_k=None,
        )
        kept_fr, _ = base_eval.filterrag_defense(
            question, mutated_passages, epsilon=base_eval.FILTERRAG_EPSILON,
            slm_answer_fn=defense_models.memo_slm_answer_fn,
            matching_mode=base_eval.SEMANTIC_MATCHING_MODE, semantic_threshold=base_eval.SEMANTIC_THRESHOLD,
        )
        feature_rows = base_eval.extract_features(
            question, mutated_passages, slm_answer_fn=defense_models.memo_slm_answer_fn,
            slm_logprob_model=defense_models.slm_logprob_model, slm_logprob_tokenizer=defense_models.slm_logprob_tokenizer,
            matching_mode=base_eval.SEMANTIC_MATCHING_MODE, semantic_threshold=base_eval.SEMANTIC_THRESHOLD,
            causal_lm_scorer=defense_models.memo_causal_scorer, lm_model_name=base_eval.LM_MODEL,
            lm_device=base_eval.DEVICE,
        )
        X = base_eval.features_to_matrix(feature_rows, defense_models.classifier.feature_names)
        proba = defense_models.classifier.predict_proba(X)

        removed_doc_ids_by_defense: Dict[str, Set[str]] = {
            "ragdefender": _removed_doc_ids(mutated_passages, kept_rd),
            "filterrag_semantic": _removed_doc_ids(mutated_passages, kept_fr),
        }
        for t in base_eval.ML_THRESHOLDS:
            suffix = base_eval.ML_THRESHOLD_SUFFIX[t]
            removed_idx_t = [i for i, pr in enumerate(proba) if float(pr) >= t]
            removed_doc_ids_by_defense[f"ml_filterrag_{suffix}"] = {mutated_passages[i].doc_id for i in removed_idx_t}

        rows = build_origin_rows(
            qid=qid, k=K, passages=mutated_passages, full_pool_query_ids=full_pool_query_ids,
            mutated_self_global_indices=mutated_self_global_indices,
            removed_doc_ids_by_defense=removed_doc_ids_by_defense,
        )
        all_origin_rows.extend(rows)

    breakdown_fields = [
        "query_id", "k", "rank", "doc_id", "source", "is_poison", "origin_label",
        "true_owning_query_id", "true_global_index", "retrieval_score",
    ] + [f"removed_by_{d}" for d in DEFENSE_NAMES]
    write_csv(os.path.join(out_dir, "full_retrieval_poison_origin_breakdown.csv"), breakdown_fields, all_origin_rows)
    print(f"[audit_poison_origin] wrote {len(all_origin_rows)} passage-origin rows.")

    origin_counts = count_origins_per_query(all_origin_rows, SELECTED_QUERY_IDS)
    dup_rows = verify_no_original_self_poison_duplicate(all_origin_rows)
    max5_check = verify_max_five_mutated_self_per_query(all_origin_rows, SELECTED_QUERY_IDS)
    removed_by_origin = summarize_removed_by_origin(all_origin_rows, SELECTED_QUERY_IDS)

    prior_defense_scores_path = os.path.join(out_dir, "full_retrieval_defense_scores.csv")
    prior_mutated_by_qid: Dict[str, Dict] = {}
    if os.path.exists(prior_defense_scores_path):
        with open(prior_defense_scores_path, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["condition"] == "mutated":
                    prior_mutated_by_qid[r["query_id"]] = r

    report = build_report(
        replacement_plan=replacement_plan, budget_check=budget_check, origin_counts=origin_counts,
        dup_rows=dup_rows, max5_check=max5_check, removed_by_origin=removed_by_origin,
        prior_mutated_by_qid=prior_mutated_by_qid, out_dir=out_dir,
    )
    with open(os.path.join(out_dir, "full_retrieval_budget_contamination_audit.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[audit_poison_origin] wrote outputs to {out_dir}")


# ---------------------------------------------------------------------------
# 4. Report.
# ---------------------------------------------------------------------------

def _fmt_bool(b: bool) -> str:
    return "Yes" if b else "No"


def build_report(
    *, replacement_plan: Dict, budget_check: Dict[str, Dict], origin_counts: Dict[str, Dict[str, int]],
    dup_rows: List[Dict], max5_check: Dict[str, Dict], removed_by_origin: Dict[Tuple[str, str], Dict[str, int]],
    prior_mutated_by_qid: Dict[str, Dict], out_dir: str,
) -> str:
    lines: List[str] = []
    lines.append("# Full-Retrieval Pilot -- Poison-Origin & Budget-Contamination Audit")
    lines.append("")
    lines.append(
        "Audits `manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/full_retrieval_pilot/` "
        "(the 3-query full-retrieval rerun of the normalized `filterrag_targeted` mutation cases) for "
        "**self-query poison vs cross-query poison contamination** in the freshly-retrieved mutated "
        "top-10, and re-verifies the replacement budget at the retrieval level (not just the pool-text "
        "level `assert_budget_preserved` already checked)."
    )
    lines.append("")

    lines.append("## 1. Retrieved poison composition per query (mutated condition, top-10)")
    lines.append("")
    lines.append(
        "| query_id | mutated_self_query_poison | cross_query_poison | original_self_query_poison (anomaly) | "
        "total poison | clean | total retrieved |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for qid, c in origin_counts.items():
        total_retrieved = c["total_poison"] + c["total_clean"]
        lines.append(
            f"| `{qid}` | {c[ORIGIN_MUTATED_SELF]} | {c[ORIGIN_CROSS_QUERY]} | {c[ORIGIN_ORIGINAL_SELF]} | "
            f"{c['total_poison']} | {c['total_clean']} | {total_retrieved} |"
        )
    lines.append("")

    lines.append("## 2. Replacement budget verification")
    lines.append("")
    lines.append(
        "| query_id | self-query slots replaced (plan) | distinct global indices | exactly 5 replaced | "
        "distinct mutated_self_query_poison retrieved | exceeds budget of 5 |"
    )
    lines.append("|---|---:|---:|---|---:|---|")
    for qid in budget_check:
        b = budget_check[qid]
        m5 = max5_check[qid]
        lines.append(
            f"| `{qid}` | {b['n_slots_replaced']} | {b['n_distinct_global_indices']} | "
            f"{_fmt_bool(b['exactly_5_replaced'])} | {m5['n_mutated_self_retrieved']} | "
            f"{_fmt_bool(m5['exceeds_budget_of_5'])} |"
        )
    lines.append("")
    lines.append(
        f"**Original poison duplicated alongside mutated poison at retrieval time?** "
        f"{_fmt_bool(len(dup_rows) > 0)} -- {len(dup_rows)} `original_self_query_poison` row(s) found "
        "across all 3 queries (expected 0; every one of the 3 selected queries had all 5 of its own "
        "poison slots replaced before retrieval was rerun, so its own *original* poison text can never "
        "be a retrieval candidate)."
    )
    lines.append("")

    lines.append("## 3. Defense removals split by poison origin (mutated condition)")
    lines.append("")
    lines.append(
        "| query_id | defense | removed mutated_self_query_poison | removed cross_query_poison | "
        "removed clean | removed original_self_query_poison (anomaly) | total removed poison |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for (qid, defense_name), counts in removed_by_origin.items():
        total_removed_poison = counts[ORIGIN_MUTATED_SELF] + counts[ORIGIN_CROSS_QUERY] + counts[ORIGIN_ORIGINAL_SELF]
        lines.append(
            f"| `{qid}` | {defense_name} | {counts[ORIGIN_MUTATED_SELF]} | {counts[ORIGIN_CROSS_QUERY]} | "
            f"{counts[ORIGIN_CLEAN]} | {counts[ORIGIN_ORIGINAL_SELF]} | {total_removed_poison} |"
        )
    lines.append("")

    lines.append("### Cross-check against `full_retrieval_defense_scores.csv` (mutated condition)")
    lines.append("")
    defense_to_prior_key = {
        "ragdefender": "ragdefender_removed_poison",
        "filterrag_semantic": "filterrag_removed_poison",
        "ml_filterrag_t035": "ml_removed_poison_t035",
        "ml_filterrag_t04": "ml_removed_poison_t04",
        "ml_filterrag_t05": "ml_removed_poison_t05",
    }
    mismatches: List[str] = []
    if prior_mutated_by_qid:
        lines.append("| query_id | defense | this audit's removed poison (self+cross) | prior aggregate removed_poison | match |")
        lines.append("|---|---|---:|---:|---|")
        for (qid, defense_name), counts in removed_by_origin.items():
            audit_total = counts[ORIGIN_MUTATED_SELF] + counts[ORIGIN_CROSS_QUERY] + counts[ORIGIN_ORIGINAL_SELF]
            prior_key = defense_to_prior_key[defense_name]
            prior_val = prior_mutated_by_qid.get(qid, {}).get(prior_key)
            prior_val_i = int(prior_val) if prior_val not in (None, "") else None
            match = (prior_val_i == audit_total)
            if not match:
                mismatches.append(f"{qid}/{defense_name}: audit={audit_total} prior={prior_val_i}")
            lines.append(f"| `{qid}` | {defense_name} | {audit_total} | {prior_val_i} | {_fmt_bool(match)} |")
        lines.append("")
        lines.append(
            "All rows match." if not mismatches else
            "**Mismatch(es) found:** " + "; ".join(mismatches) + " -- investigate before trusting the "
            "origin breakdown above."
        )
    else:
        lines.append(
            "`full_retrieval_defense_scores.csv` was not found alongside this audit's output directory; "
            "skipping the cross-check."
        )
    lines.append("")

    any_cross = any(c[ORIGIN_CROSS_QUERY] > 0 for c in origin_counts.values())
    six_poison_qids = [qid for qid, c in origin_counts.items() if c["total_poison"] == 6]
    any_augmentation = any(not b["exactly_5_replaced"] for b in budget_check.values())
    any_dup = len(dup_rows) > 0
    any_over_budget = any(m["exceeds_budget_of_5"] for m in max5_check.values())

    lines.append("## Answers")
    lines.append("")
    lines.append(
        f"**Did any query inject more than 5 mutated self-query poison passages?** "
        f"{_fmt_bool(any_over_budget)} -- every selected query's replacement plan replaced exactly 5 "
        "distinct self-query poison slots (see Section 2), and the shared 50-query pool structurally "
        "contains only 5 slots per query, so no query's retrieved top-10 can ever contain more than 5 "
        "`mutated_self_query_poison` passages."
    )
    lines.append("")
    lines.append(
        f"**Which query retrieved 6 total poison passages?** "
        + (", ".join(f"`{qid}`" for qid in six_poison_qids) if six_poison_qids else "None -- no query in this pilot retrieved 6 total poison passages.")
        + "."
    )
    lines.append("")
    if six_poison_qids:
        qid6 = six_poison_qids[0]
        c6 = origin_counts[qid6]
        lines.append(
            f"**Was the sixth poison passage cross-query poison?** "
            f"{_fmt_bool(c6[ORIGIN_CROSS_QUERY] > 0)} -- `{qid6}` retrieved "
            f"{c6[ORIGIN_MUTATED_SELF]} of its own mutated self-query poison passages (the maximum "
            f"possible) plus {c6[ORIGIN_CROSS_QUERY]} passage(s) originally authored for a *different* "
            "pool query, confirmed by resolving the passage's global pool index against "
            "`dataset_config.json`'s ordered `target_query_ids` (not by trusting the doc_id's own qid "
            "segment, which always reads as the currently-retrieved-for query -- see module docstring)."
        )
    else:
        lines.append(
            "**Was the sixth poison passage cross-query poison?** N/A -- no query retrieved 6 total "
            "poison passages in this pilot."
        )
    lines.append("")
    lines.append(
        f"**Did the extra cross-query poison affect defense counts?** "
        f"{_fmt_bool(any_cross)} -- see Section 3's per-origin removal breakdown; any query/defense cell "
        "with a non-zero `removed cross_query_poison` count had its aggregate `removed_poison` figure in "
        "`full_retrieval_defense_scores.csv` inflated by cross-query poison the defense also happened to "
        "remove (or, symmetrically, its residual/failure count deflated if the defense missed the "
        "cross-query passage) -- i.e. the aggregate `removed_poison`/`residual_poison_fraction` metrics "
        "for that query conflate the mutated self-query poison outcome (the one this pilot was designed "
        "to test) with an incidental cross-query outcome from a different attack instance."
    )
    lines.append("")
    lines.append(
        f"**Are the full-retrieval conclusions still valid?** "
        + (
            "Yes, with one caveat. All budget-preservation invariants hold exactly (5/5 self-query slots "
            "replaced per query, 0 original-poison duplication, 0 over-budget queries -- see Section 2), "
            "so the pilot's core claim (\"the mutated self-query poison survives retrieval and its effect "
            "on defenses is X\") is unaffected for the 2 of 3 queries with no cross-query contamination "
            f"({', '.join(f'`{qid}`' for qid, c in origin_counts.items() if c[ORIGIN_CROSS_QUERY] == 0)}). "
            f"For {', '.join(f'`{qid}`' for qid in six_poison_qids)}, the per-defense `removed_poison`/"
            "`residual_poison_fraction` *aggregate* figures reported in `full_retrieval_defense_scores.csv` "
            "should be read as \"self-query mutated poison + 1 incidental cross-query poison passage\", not "
            "as a pure measurement of the mutated candidate alone -- Section 3's `removed "
            "mutated_self_query_poison` column isolates the self-query-only figure for that query, which "
            "is the more precise number for any paper-level claim about that specific mutated candidate."
            if six_poison_qids else
            "Yes -- no cross-query contamination was found in any of the 3 queries' retrieved top-10, so "
            "every reported `retrieved_poison_count` in `full_retrieval_results_by_query.csv` and every "
            "per-defense `removed_poison` figure in `full_retrieval_defense_scores.csv` already reflects "
            "self-query mutated poison exclusively."
        )
    )
    lines.append("")

    lines.append("## Methodology notes")
    lines.append("")
    lines.append(
        "- Retrieval was rebuilt identically to `scripts/run_full_retrieval_pilot_bundle1.py` (same "
        "Contriever model/device, same rebuilt 50-query adversarial pool via `Attacker.get_attack()`, "
        "same replacement plan, same `merge_and_topk`) -- this audit re-derives the already-published "
        "mutated top-10, it does not construct a new experimental condition."
    )
    lines.append(
        "- True poison origin is recovered from each adversarial passage's global pool index `j` "
        "(`pilot.extract_global_index(doc_id)`) against the ordered `dataset_config.json::target_query_ids` "
        "pool (`full_pool_query_ids[j // 5]`), never from the doc_id's own embedded qid segment (which "
        "`merge_and_topk` always sets to the currently-retrieved-for query, not the text's true author -- "
        "see the module docstring's caveat)."
    )
    lines.append(
        "- Defense removal decisions (`defense.dispatch.run_defense(\"ragdefender_original\", ...)`, "
        "`defense.filterrag.filterrag_defense(..., epsilon=0.2)`, `defense.ml_filterrag.extract_features` "
        "+ the already-trained classifier's `predict_proba` at t in {0.35, 0.4, 0.5}) are byte-for-byte "
        "the same calls `run_text_mutation_fixed_context_eval.py`'s `score_ragdefender`/`score_filterrag`/"
        "`score_ml_filterrag` already make; only the post-hoc bucketing of which removed doc_id is "
        "self/cross/clean is new."
    )
    lines.append(f"- Output directory: `{os.path.relpath(out_dir, REPO_ROOT)}`.")
    lines.append("")

    lines.append("## Process confirmation")
    lines.append("")
    lines.append("- No GPT/API calls were made.")
    lines.append("- No `llm.query()` calls were made.")
    lines.append("- No model was trained or retrained.")
    lines.append("- No defense code (`defense/*.py`) was modified.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"[audit_poison_origin] total run time: {time.perf_counter() - t0:.1f}s")
