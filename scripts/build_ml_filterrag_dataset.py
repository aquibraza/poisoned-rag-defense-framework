#!/usr/bin/env python3
"""Build a labeled ML-FilterRAG-top-k feature dataset from this repo's
existing ground-truth `is_poison` labels.

Reproduces the *exact* same retrieval + adversarial-injection pipeline
`main.py`/`scripts/filterrag_score_inspection.py` use (same corpus/queries/
qrels, same BEIR top-k results, same `Attacker.get_attack()` adv texts, same
embedding-similarity re-ranking) for a fixed set of target queries, sweeping
multiple `k`/`attack_method` values, and computes the full
`defense.ml_filterrag.extract_features()` feature dict for every retrieved
passage.

**No GPT/API call is ever made.** `Attacker.get_attack()`'s `LM_targeted`
adversarial texts are pre-generated offline (read from
`results/adv_targeted_results/`, see `src/attack.py`); the FilterRAG SLM
(`google/flan-t5-small` by default) and the ML-FilterRAG perplexity LM
(`distilgpt2` by default) are both small local HF models run entirely
offline, exactly like `--defense filterrag`/`--defense ml_filterrag` would.
No `llm.query()` call, no live generation through any LLM.

**Ground-truth-only labels**: every row's `is_poison` label comes from the
`source`/`is_poison` keys attached at passage-construction time (attacker
ground truth), never inferred from passage text -- see
`defense/passages.py`.

**Leakage-safe query-level split** (see
`docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md` section 2): the train/test split
is computed once, per dataset, over the *set* of `query_id`s used by this
run -- before any passage row is generated -- via
`defense.ml_filterrag.query_level_train_test_split()`. Every row for a given
`query_id`, across every `k`/`attack_method` swept here, lands in exactly
one split. `defense.ml_filterrag.assert_no_query_id_leakage()` is asserted
before writing any output.

Usage:
    python scripts/build_ml_filterrag_dataset.py \\
        --eval_dataset hotpotqa --k_values 5 10 --N 5 --max_queries 20 \\
        --out_dir results/diagnostics/ml_filterrag_dataset_hotpotqa
"""
import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)  # load_beir_datasets() resolves ./datasets relative to cwd, like main.py

import torch  # noqa: E402

import fix_sentence_transformers  # noqa: E402,F401 -- match main.py's compat patch

from src.utils import load_beir_datasets, load_json, load_models  # noqa: E402
from src.attack import Attacker  # noqa: E402

from defense.filterrag import DEFAULT_SEMANTIC_THRESHOLD, VALID_MATCHING_MODES, local_hf_slm_answer_fn  # noqa: E402
from defense.ml_filterrag import (  # noqa: E402
    ALL_FEATURE_NAMES,
    DEFAULT_LM_MODEL,
    assert_no_query_id_leakage,
    extract_features,
    get_causal_lm_scorer,
    get_slm_model_and_tokenizer,
    query_level_train_test_split,
)
from defense.passages import label_passages  # noqa: E402

CSV_COLUMNS = (
    ["dataset", "attack", "k", "N", "query_id", "doc_id", "is_poison", "split"]
    + list(ALL_FEATURE_NAMES)
    + ["slm_answer"]
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_dataset", default="hotpotqa", choices=["nq", "hotpotqa", "msmarco"])
    parser.add_argument("--eval_model_code", default="contriever")
    parser.add_argument("--split", default="test")
    parser.add_argument("--score_function", default="dot", choices=["dot", "cos_sim"])
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--N", type=int, default=5, help="adv_per_query")
    parser.add_argument(
        "--attack_methods", nargs="+", default=["LM_targeted"],
        help=(
            "Attack method(s) whose offline-generated adversarial texts to inject "
            "(via Attacker.get_attack(), zero GPT/API calls -- see src/attack.py). "
            "'hotflip' additionally requires CUDA (hardcoded in Attacker.hotflip()) "
            "and is not exercised by this script's defaults."
        ),
    )
    parser.add_argument("--max_queries", type=int, default=20)
    parser.add_argument("--filterrag_slm_model", default="google/flan-t5-small")
    parser.add_argument("--filterrag_slm_device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--ml_filterrag_matching_mode", default="semantic", choices=list(VALID_MATCHING_MODES))
    parser.add_argument("--ml_filterrag_semantic_threshold", type=float, default=DEFAULT_SEMANTIC_THRESHOLD)
    parser.add_argument("--ml_filterrag_lm_model", default=DEFAULT_LM_MODEL)
    parser.add_argument(
        "--test_fraction", type=float, default=0.2,
        help="Fraction of query_ids assigned to the test split (query-level, see module docstring).",
    )
    parser.add_argument("--split_seed", type=int, default=12)
    parser.add_argument("--out_dir", default="results/diagnostics/ml_filterrag_dataset")
    return parser.parse_args()


def build_feature_rows(args) -> List[Dict]:
    """Reproduce main.py's retrieval + injection pipeline for the first
    `--max_queries` target queries, for every `(attack_method, k)` combo,
    and compute the full ML-FilterRAG feature dict for every retrieved
    passage. Returns a flat list of per-passage dicts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ml_filterrag_dataset] retrieval device: {device}")

    corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
    incorrect_answers = list(load_json(f"results/adv_targeted_results/{args.eval_dataset}.json").values())

    beir_results_path = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}.json"
    if args.score_function == "cos_sim":
        beir_results_path = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-cos.json"
    with open(beir_results_path, "r") as f:
        results = json.load(f)

    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    model.eval()
    model.to(device)
    c_model.eval()
    c_model.to(device)

    target_queries_idx = list(range(args.max_queries))
    query_ids = [incorrect_answers[i]["id"] for i in target_queries_idx]

    # Query-level split, computed ONCE per dataset, before any row exists --
    # see module docstring / docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md sec 2.
    train_ids, test_ids = query_level_train_test_split(
        query_ids, test_fraction=args.test_fraction, seed=args.split_seed
    )
    assert_no_query_id_leakage(train_ids, test_ids)
    split_by_qid = {qid: "train" for qid in train_ids}
    split_by_qid.update({qid: "test" for qid in test_ids})
    print(f"[ml_filterrag_dataset] query-level split: {len(train_ids)} train / {len(test_ids)} test query_ids")

    print(f"[ml_filterrag_dataset] loading FilterRAG SLM ({args.filterrag_slm_model})...")
    slm_answer_fn = local_hf_slm_answer_fn(args.filterrag_slm_model, device=args.filterrag_slm_device)
    slm_model, slm_tokenizer = get_slm_model_and_tokenizer(args.filterrag_slm_model, device=args.filterrag_slm_device)
    print(f"[ml_filterrag_dataset] loading perplexity LM ({args.ml_filterrag_lm_model})...")
    causal_lm_scorer = get_causal_lm_scorer(args.ml_filterrag_lm_model, device="cpu")

    max_k = max(args.k_values)
    all_rows: List[Dict] = []

    for attack_method in args.attack_methods:
        attack_t0 = time.perf_counter()

        class _AttackArgs:
            pass

        _AttackArgs.attack_method = attack_method
        _AttackArgs.adv_per_query = args.N
        _AttackArgs.eval_dataset = args.eval_dataset
        attacker = Attacker(_AttackArgs(), model=model, c_model=c_model, tokenizer=tokenizer, get_emb=get_emb)

        target_queries = [None] * len(target_queries_idx)
        for iter_idx, i in enumerate(target_queries_idx):
            top1_idx = list(results[incorrect_answers[i]["id"]].keys())[0]
            top1_score = results[incorrect_answers[i]["id"]][top1_idx]
            target_queries[iter_idx] = {
                "query": incorrect_answers[i]["question"],
                "top1_score": top1_score,
                "id": incorrect_answers[i]["id"],
            }

        # Offline/templated for LM_targeted -- no LLM call, see src/attack.py.
        adv_text_groups = attacker.get_attack(target_queries)
        adv_text_list = sum(adv_text_groups, [])
        adv_input = tokenizer(adv_text_list, padding=True, truncation=True, return_tensors="pt")
        adv_input = {key: value.to(device) for key, value in adv_input.items()}
        with torch.no_grad():
            adv_embs = get_emb(c_model, adv_input)

        for iter_idx, i in enumerate(target_queries_idx):
            qid = incorrect_answers[i]["id"]
            question = incorrect_answers[i]["question"]

            # Retrieve enough corpus candidates for the largest k up front,
            # then merge with adversarial texts and sort *once* -- taking a
            # prefix of length k from this single sorted list for every k
            # in --k_values is equivalent to independently sorting+slicing
            # at each k (same scores, same tie-breaking), just without
            # redundant recomputation.
            topk_idx = list(results[qid].keys())[:max_k]
            merged_results = [
                {
                    "score": results[qid][idx], "context": corpus[idx]["text"], "doc_id": idx,
                    "source": "corpus", "is_poison": False,
                }
                for idx in topk_idx
            ]

            query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
            query_input = {key: value.to(device) for key, value in query_input.items()}
            with torch.no_grad():
                query_emb = get_emb(model, query_input)
            for j in range(len(adv_text_list)):
                adv_emb = adv_embs[j, :].unsqueeze(0)
                if args.score_function == "dot":
                    adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
                else:
                    adv_sim = torch.cosine_similarity(adv_emb, query_emb).cpu().item()
                merged_results.append({
                    "score": adv_sim, "context": adv_text_list[j],
                    "doc_id": f"adv::{attack_method}::{qid}::{j}",
                    "source": "adversarial", "is_poison": True,
                })
            merged_results = sorted(merged_results, key=lambda x: float(x["score"]), reverse=True)

            for k in args.k_values:
                topk_results = merged_results[:k]
                passages = label_passages(topk_results)

                feature_rows = extract_features(
                    question, passages,
                    slm_answer_fn=slm_answer_fn,
                    slm_logprob_model=slm_model,
                    slm_logprob_tokenizer=slm_tokenizer,
                    matching_mode=args.ml_filterrag_matching_mode,
                    semantic_threshold=args.ml_filterrag_semantic_threshold,
                    causal_lm_scorer=causal_lm_scorer,
                )

                for p, row in zip(passages, feature_rows):
                    assert p.doc_id == row["doc_id"]
                    out_row = {
                        "dataset": args.eval_dataset,
                        "attack": attack_method,
                        "k": k,
                        "N": args.N,
                        "query_id": qid,
                        "doc_id": p.doc_id,
                        "is_poison": p.is_poison,
                        "split": split_by_qid[qid],
                        "slm_answer": row["slm_answer"],
                    }
                    out_row.update({name: row[name] for name in ALL_FEATURE_NAMES})
                    all_rows.append(out_row)

        print(
            f"[ml_filterrag_dataset] attack={attack_method}: scored {len(target_queries_idx)} queries "
            f"x {len(args.k_values)} k-values in {time.perf_counter() - attack_t0:.1f}s"
        )

    # Re-assert on the actual rows written (per §2: downstream consumers
    # must never trust a single 'split' boolean blindly without being able
    # to re-derive/verify it).
    written_train_ids = {r["query_id"] for r in all_rows if r["split"] == "train"}
    written_test_ids = {r["query_id"] for r in all_rows if r["split"] == "test"}
    assert_no_query_id_leakage(written_train_ids, written_test_ids)

    return all_rows


def write_dataset_csv(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} feature row(s) to {path}")


def write_config_json(args, rows: List[Dict], path: str) -> None:
    """Provenance record: exact feature-extractor config (matching mode,
    semantic threshold, SLM/LM model names, split seed) used to build this
    dataset -- so downstream scripts/humans can verify/re-derive the split
    and reproduce the run."""
    train_ids = sorted({r["query_id"] for r in rows if r["split"] == "train"})
    test_ids = sorted({r["query_id"] for r in rows if r["split"] == "test"})
    config = {
        "status": "ML-FilterRAG-top-k dataset (not the paper's top-s pipeline; see "
                   "docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md sec 1/9/10)",
        "eval_dataset": args.eval_dataset,
        "eval_model_code": args.eval_model_code,
        "split": args.split,
        "score_function": args.score_function,
        "k_values": args.k_values,
        "N": args.N,
        "attack_methods": args.attack_methods,
        "max_queries": args.max_queries,
        "filterrag_slm_model": args.filterrag_slm_model,
        "filterrag_slm_device": args.filterrag_slm_device,
        "ml_filterrag_matching_mode": args.ml_filterrag_matching_mode,
        "ml_filterrag_semantic_threshold": args.ml_filterrag_semantic_threshold,
        "ml_filterrag_lm_model": args.ml_filterrag_lm_model,
        "test_fraction": args.test_fraction,
        "split_seed": args.split_seed,
        "n_rows": len(rows),
        "n_train_query_ids": len(train_ids),
        "n_test_query_ids": len(test_ids),
        "train_query_ids": train_ids,
        "test_query_ids": test_ids,
        "no_gpt_api_calls_made": True,
        "no_live_generation_through_llm_query": True,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote dataset config to {path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[ml_filterrag_dataset] No GPT/API call will be made; no llm.query() call will be made.")

    t0 = time.perf_counter()
    rows = build_feature_rows(args)
    print(f"[ml_filterrag_dataset] total build time: {time.perf_counter() - t0:.1f}s for {len(rows)} row(s)")

    write_dataset_csv(rows, os.path.join(args.out_dir, "features.csv"))
    write_config_json(args, rows, os.path.join(args.out_dir, "dataset_config.json"))


if __name__ == "__main__":
    main()
