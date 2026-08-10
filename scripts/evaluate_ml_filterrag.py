#!/usr/bin/env python3
"""Evaluate a trained ML-FilterRAG-top-k classifier against threshold
FilterRAG (semantic mode), on the same retrieved/injected passage sets.

Reuses the exact same offline retrieval + adversarial-injection harness as
`scripts/build_ml_filterrag_dataset.py`/`main.py` (same corpus/queries/
qrels, same BEIR top-k results, same offline `Attacker.get_attack()` adv
texts, same embedding-similarity re-ranking), then calls
`defense.dispatch.run_defense(...)` **twice per query** -- once with
`--defense ml_filterrag` (the trained artifact), once with `--defense
filterrag --filterrag_matching_mode semantic --filterrag_semantic_threshold
0.6` (the paper-faithful threshold-FilterRAG baseline) -- and computes the
identical detection-quality metrics `defense/diagnostics.py` already knows
how to derive from passage lists alone (poison recall, clean false-positive
rate, residual poison fraction), with **no LLM generation** (every
diagnostic field that would require generation is left `None`, exactly
matching `--dry_run` runs of `main.py`).

**Dry-run only. No GPT/API call is ever made. No `llm.query()` call is ever
made. No live generation is run** -- `--dry_run False` is refused with a
clear `NotImplementedError` (live-generation ASR comparison is out of scope
for this MVP script; see `docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md` sec 6).

**Held-out evaluation**: pass `--held_out_config
results/diagnostics/ml_filterrag_dataset_<...>/dataset_config.json` (written
by `scripts/build_ml_filterrag_dataset.py`) to restrict evaluation strictly
to that dataset's `test_query_ids` -- i.e. `query_id`s never used to train
the classifier being evaluated. This reconstructs the *exact same*
adversarial candidate pool (`target_query_ids`, the dataset's full
`--max_queries` query set, not just `test_query_ids`) that
`build_ml_filterrag_dataset.py` used, before restricting the actual
scoring/metrics to just the held-out subset -- see
`resolve_query_id_pools()`'s docstring for why this distinction matters:
`Attacker.get_attack()` batches every pool query together and any pool
query's adversarial text can end up in *any other* pool query's retrieved
top-k, so evaluating with a narrower pool than the one used to build the
dataset silently changes which passages are retrieved/labeled poison for
the held-out queries, relative to that dataset's own `features.csv`. Older
`dataset_config.json` files (predating `target_query_ids`) are handled by
reconstructing the pool from their `max_queries` field, with a warning.
Without `--held_out_config`, this script prints an explicit warning and
simply evaluates the first `--max_queries` queries, with **no guarantee**
they were excluded from any particular model's training set -- the caller
is responsible for that guarantee in that case.

Usage:
    python scripts/evaluate_ml_filterrag.py \\
        --eval_dataset hotpotqa \\
        --ml_filterrag_model_path models/ml_filterrag/hotpotqa_LM_targeted_random_forest_TIMESTAMP.joblib \\
        --held_out_config results/diagnostics/ml_filterrag_dataset_hotpotqa/dataset_config.json \\
        --out_dir results/diagnostics/ml_filterrag_eval_hotpotqa
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)  # load_beir_datasets() resolves ./datasets relative to cwd, like main.py

import torch  # noqa: E402

import fix_sentence_transformers  # noqa: E402,F401 -- match main.py's compat patch

from src.utils import load_beir_datasets, load_json, load_models  # noqa: E402
from src.attack import Attacker  # noqa: E402

from defense.diagnostics import build_diagnostic_record, timer  # noqa: E402
from defense.dispatch import run_defense  # noqa: E402
from defense.filterrag import DEFAULT_EPSILON, DEFAULT_SEMANTIC_THRESHOLD  # noqa: E402
from defense.ml_filterrag import DEFAULT_LM_MODEL, DEFAULT_THRESHOLD, VALID_MODEL_TYPES  # noqa: E402,F401
from defense.passages import label_passages  # noqa: E402


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean-like string, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_dataset", default="hotpotqa", choices=["nq", "hotpotqa", "msmarco"])
    parser.add_argument("--eval_model_code", default="contriever")
    parser.add_argument("--split", default="test")
    parser.add_argument("--score_function", default="dot", choices=["dot", "cos_sim"])
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--N", type=int, default=5, help="adv_per_query")
    parser.add_argument("--attack_methods", nargs="+", default=["LM_targeted"])
    parser.add_argument("--max_queries", type=int, default=20)
    parser.add_argument(
        "--held_out_config", default=None,
        help="Path to a dataset_config.json written by scripts/build_ml_filterrag_dataset.py; "
             "if given, evaluation is restricted to its 'test_query_ids' (genuinely held out "
             "from that dataset's training split). Strongly recommended.",
    )

    parser.add_argument("--ml_filterrag_model_path", required=True)
    parser.add_argument("--ml_filterrag_threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--ml_filterrag_matching_mode", default="semantic", choices=["exact", "semantic"])
    parser.add_argument("--ml_filterrag_semantic_threshold", type=float, default=DEFAULT_SEMANTIC_THRESHOLD)
    parser.add_argument("--ml_filterrag_lm_model", default=DEFAULT_LM_MODEL)

    parser.add_argument("--filterrag_slm_model", default="google/flan-t5-small")
    parser.add_argument("--filterrag_slm_device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--filterrag_matching_mode", default="semantic", choices=["exact", "semantic"],
        help="Baseline threshold-FilterRAG's matching mode -- default 'semantic' (paper-faithful, "
             "see docs/FILTERRAG_FIDELITY_AUDIT.md), matching this script's ml_filterrag default.",
    )
    parser.add_argument("--filterrag_semantic_threshold", type=float, default=DEFAULT_SEMANTIC_THRESHOLD)
    parser.add_argument("--filterrag_epsilon", type=float, default=DEFAULT_EPSILON)

    parser.add_argument(
        "--dry_run", type=_str2bool, default=True,
        help="Must be True -- this MVP script never runs live generation/llm.query() "
             "(--dry_run False raises NotImplementedError).",
    )
    parser.add_argument("--out_dir", default="results/diagnostics/ml_filterrag_eval")
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def resolve_query_id_pools(args, incorrect_answers: List[Dict]) -> Tuple[List[str], List[str]]:
    """Return `(full_pool_query_ids, eval_query_ids)`, both ordered lists of
    `query_id` strings (not indices).

    `full_pool_query_ids` is the *exact* adversarial candidate pool that
    must be fed to `Attacker.get_attack()` -- see
    `scripts/build_ml_filterrag_dataset.py::build_feature_rows()`'s
    docstring: `Attacker.get_attack(target_queries)` is called once per
    pool with every pool query, and every resulting adversarial text is
    then scored against *every* pool query's own embedding, so a query's
    retrieved top-k composition depends on which *other* queries share its
    pool. Reconstructing this pool as just the held-out subset (as an
    earlier version of this function did) silently changes which
    passages are retrieved/labeled for those held-out queries relative to
    the dataset that was actually built -- see
    docs/ML_FILTERRAG_FIDELITY notes in `docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md`
    and the `--held_out_config` module docstring above.

    `eval_query_ids` is the (usually much smaller) subset of
    `full_pool_query_ids` this run actually scores/reports metrics for --
    the held-out `test_query_ids` when `--held_out_config` is given, else
    the full pool itself (unchanged from before: with no config, there is
    no train/test distinction to make).
    """
    if args.held_out_config:
        with open(args.held_out_config, "r", encoding="utf-8") as f:
            held_out_cfg = json.load(f)
        test_query_ids = set(held_out_cfg.get("test_query_ids", []))
        if not test_query_ids:
            raise ValueError(f"--held_out_config {args.held_out_config!r} has an empty 'test_query_ids' list.")

        full_pool_query_ids = held_out_cfg.get("target_query_ids")
        if full_pool_query_ids is None:
            max_queries_in_config = held_out_cfg.get("max_queries")
            if max_queries_in_config is None:
                raise ValueError(
                    f"--held_out_config {args.held_out_config!r} has neither 'target_query_ids' "
                    "nor 'max_queries' -- cannot reconstruct the exact adversarial candidate pool "
                    "used to build that dataset (see resolve_query_id_pools() docstring for why "
                    "this matters -- a held-out subset alone is NOT equivalent). Rebuild the "
                    "dataset with the current scripts/build_ml_filterrag_dataset.py (which writes "
                    "'target_query_ids'), then re-point --held_out_config at the new "
                    "dataset_config.json."
                )
            print(
                f"[evaluate_ml_filterrag] WARNING: --held_out_config {args.held_out_config!r} "
                "predates 'target_query_ids' -- reconstructing the full candidate pool as the "
                f"first max_queries={max_queries_in_config} queries from "
                f"results/adv_targeted_results/{args.eval_dataset}.json, exactly as the dataset "
                "builder did at build time. If that file has changed since the dataset was built, "
                "this reconstruction will NOT match and retrieved-passage counts may still "
                "disagree with the dataset's features.csv -- rebuild the dataset if in doubt."
            )
            full_pool_query_ids = [ia["id"] for ia in incorrect_answers[:max_queries_in_config]]

        missing_from_pool = test_query_ids - set(full_pool_query_ids)
        if missing_from_pool:
            raise ValueError(
                f"--held_out_config {args.held_out_config!r} is inconsistent: {len(missing_from_pool)} "
                f"'test_query_ids' entr(y/ies) are not present in its own 'target_query_ids' "
                f"(or the max_queries-based reconstruction of them): {sorted(missing_from_pool)[:5]}... "
                "-- rebuild the dataset."
            )

        id_to_idx = {ia["id"]: i for i, ia in enumerate(incorrect_answers)}
        missing = [qid for qid in full_pool_query_ids if qid not in id_to_idx]
        if missing:
            print(
                f"[evaluate_ml_filterrag] WARNING: {len(missing)} candidate-pool query_id(s) from "
                f"--held_out_config not found in this dataset's incorrect_answers file, skipped: "
                f"{missing[:5]}..."
            )
        full_pool_query_ids = [qid for qid in full_pool_query_ids if qid in id_to_idx]

        eval_query_ids = [qid for qid in full_pool_query_ids if qid in test_query_ids]
        if args.max_queries and len(eval_query_ids) > args.max_queries:
            eval_query_ids = eval_query_ids[: args.max_queries]

        print(
            f"[evaluate_ml_filterrag] reconstructed a {len(full_pool_query_ids)}-query-id adversarial "
            f"candidate pool (matching the dataset build); evaluating {len(eval_query_ids)} held-out "
            f"test query_id(s) from it (genuinely excluded from that dataset's training split)."
        )
        return full_pool_query_ids, eval_query_ids

    print(
        "[evaluate_ml_filterrag] WARNING: no --held_out_config given -- evaluating the first "
        f"--max_queries={args.max_queries} queries with NO verification that they were excluded "
        "from the ml_filterrag model's training set. Pass --held_out_config for a genuine "
        "held-out comparison."
    )
    ids = [ia["id"] for ia in incorrect_answers[: args.max_queries]]
    return ids, ids


def _pooled_metrics(records: List[Dict]) -> Dict:
    """Pool poison/clean counts across every record (rather than averaging
    per-query ratios, which is undefined/misleading whenever a query has
    zero retrieved poison or zero retrieved clean passages)."""
    n_retrieved_poison = sum(r["N_retrieved_poison"] for r in records)
    n_retrieved_clean = sum(r["N_retrieved_clean"] for r in records)
    n_removed_poison = sum(r["removed_poison"] for r in records)
    n_removed_clean = sum(r["removed_clean"] for r in records)
    residual_poison = sum(r["residual_poison_count"] for r in records)
    residual_clean = sum(r["residual_clean_count"] for r in records)
    latencies = [r["latency_defense_sec"] for r in records if r["latency_defense_sec"] is not None]

    def _safe_div(n, d):
        return (n / d) if d > 0 else None

    return {
        "n_queries_x_k": len(records),
        "n_retrieved_poison": n_retrieved_poison,
        "n_retrieved_clean": n_retrieved_clean,
        "n_removed_poison": n_removed_poison,
        "n_removed_clean": n_removed_clean,
        "poison_recall_pooled": _safe_div(n_removed_poison, n_retrieved_poison),
        "clean_false_positive_rate_pooled": _safe_div(n_removed_clean, n_retrieved_clean),
        "residual_poison_fraction_pooled": _safe_div(residual_poison, residual_poison + residual_clean),
        "mean_latency_defense_sec": (sum(latencies) / len(latencies)) if latencies else None,
    }


def run_evaluation(args) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[evaluate_ml_filterrag] retrieval device: {device}")

    corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
    incorrect_answers = list(load_json(f"results/adv_targeted_results/{args.eval_dataset}.json").values())

    beir_results_path = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}.json"
    if args.score_function == "cos_sim":
        beir_results_path = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-cos.json"
    with open(beir_results_path, "r") as f:
        results = json.load(f)

    full_pool_query_ids, eval_query_ids = resolve_query_id_pools(args, incorrect_answers)
    if not eval_query_ids:
        raise ValueError("No target queries resolved -- check --held_out_config / --max_queries.")

    id_to_idx = {ia["id"]: i for i, ia in enumerate(incorrect_answers)}
    pool_idx = [id_to_idx[qid] for qid in full_pool_query_ids]
    eval_idx = [id_to_idx[qid] for qid in eval_query_ids]

    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    model.eval()
    model.to(device)
    c_model.eval()
    c_model.to(device)

    ml_records: List[Dict] = []
    filterrag_records: List[Dict] = []

    for attack_method in args.attack_methods:
        class _AttackArgs:
            pass

        _AttackArgs.attack_method = attack_method
        _AttackArgs.adv_per_query = args.N
        _AttackArgs.eval_dataset = args.eval_dataset
        attacker = Attacker(_AttackArgs(), model=model, c_model=c_model, tokenizer=tokenizer, get_emb=get_emb)

        # Attacker.get_attack() must see the *full* candidate pool (every
        # query_id used to build the held-out dataset), not just the
        # queries we're about to score/report on below -- see
        # resolve_query_id_pools()'s docstring for why a narrower pool
        # would silently change retrieved-passage composition.
        target_queries = [None] * len(pool_idx)
        for iter_idx, i in enumerate(pool_idx):
            top1_idx = list(results[incorrect_answers[i]["id"]].keys())[0]
            top1_score = results[incorrect_answers[i]["id"]][top1_idx]
            target_queries[iter_idx] = {
                "query": incorrect_answers[i]["question"],
                "top1_score": top1_score,
                "id": incorrect_answers[i]["id"],
            }

        adv_text_groups = attacker.get_attack(target_queries)
        adv_text_list = sum(adv_text_groups, [])
        adv_input = tokenizer(adv_text_list, padding=True, truncation=True, return_tensors="pt")
        adv_input = {key: value.to(device) for key, value in adv_input.items()}
        with torch.no_grad():
            adv_embs = get_emb(c_model, adv_input)

        # ...but only the held-out subset is actually scored/reported on.
        max_k = max(args.k_values)
        for i in eval_idx:
            qid = incorrect_answers[i]["id"]
            question = incorrect_answers[i]["question"]

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
                retrieved_passages = label_passages(topk_results)

                with timer() as t_ml:
                    kept_ml, diag_ml = run_defense(
                        "ml_filterrag", question, retrieved_passages, args.eval_dataset,
                        query_id=qid,
                        ml_filterrag_model_path=args.ml_filterrag_model_path,
                        ml_filterrag_threshold=args.ml_filterrag_threshold,
                        ml_filterrag_matching_mode=args.ml_filterrag_matching_mode,
                        ml_filterrag_semantic_threshold=args.ml_filterrag_semantic_threshold,
                        ml_filterrag_lm_model=args.ml_filterrag_lm_model,
                        filterrag_slm_model=args.filterrag_slm_model,
                        filterrag_slm_device=args.filterrag_slm_device,
                    )
                ml_records.append(build_diagnostic_record(
                    query_id=qid, dataset=args.eval_dataset, model=args.eval_model_code,
                    attack=attack_method, defense="ml_filterrag", k=k, N_injected=args.N,
                    retrieved_passages=retrieved_passages, kept_passages=kept_ml,
                    N_adv_estimated_by_ragdefender=diag_ml.get("N_adv_estimated_by_ragdefender"),
                    latency_defense_sec=t_ml["elapsed_sec"], notes=diag_ml.get("notes", ""),
                ))

                with timer() as t_fr:
                    kept_fr, diag_fr = run_defense(
                        "filterrag", question, retrieved_passages, args.eval_dataset,
                        query_id=qid,
                        filterrag_epsilon=args.filterrag_epsilon,
                        filterrag_slm_model=args.filterrag_slm_model,
                        filterrag_slm_device=args.filterrag_slm_device,
                        filterrag_matching_mode=args.filterrag_matching_mode,
                        filterrag_semantic_threshold=args.filterrag_semantic_threshold,
                    )
                filterrag_records.append(build_diagnostic_record(
                    query_id=qid, dataset=args.eval_dataset, model=args.eval_model_code,
                    attack=attack_method, defense="filterrag", k=k, N_injected=args.N,
                    retrieved_passages=retrieved_passages, kept_passages=kept_fr,
                    N_adv_estimated_by_ragdefender=diag_fr.get("N_adv_estimated_by_ragdefender"),
                    latency_defense_sec=t_fr["elapsed_sec"], notes=diag_fr.get("notes", ""),
                ))

    return {"ml_filterrag": ml_records, "filterrag": filterrag_records}


def write_jsonl(records: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} record(s) to {path}")


def write_comparison_report(path: str, *, args, ml_metrics: Dict, filterrag_metrics: Dict) -> None:
    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = [
        "# ML-FilterRAG-top-k vs. Threshold FilterRAG (semantic) -- Detection-Quality Comparison",
        "",
        "**Status: ML-FilterRAG-top-k (MVP)** -- this repo's harness retrieves `top_k` "
        "directly, not the paper's `top-s -> filter -> top-k` pipeline. See "
        "`docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md`.",
        "",
        f"- Dataset: `{args.eval_dataset}` (split=`{args.split}`, model_code=`{args.eval_model_code}`)",
        f"- Attack method(s): `{args.attack_methods}`, N (adv_per_query)=`{args.N}`, k_values=`{args.k_values}`",
        f"- ML-FilterRAG model: `{args.ml_filterrag_model_path}` "
        f"(threshold=`{args.ml_filterrag_threshold}`, matching_mode=`{args.ml_filterrag_matching_mode}`, "
        f"semantic_threshold=`{args.ml_filterrag_semantic_threshold}`, lm_model=`{args.ml_filterrag_lm_model}`)",
        f"- Baseline: threshold FilterRAG, matching_mode=`{args.filterrag_matching_mode}`, "
        f"semantic_threshold=`{args.filterrag_semantic_threshold}`, epsilon=`{args.filterrag_epsilon}`",
        f"- Held-out config: `{args.held_out_config}`" if args.held_out_config else
        "- **No --held_out_config given -- held-out status vs. the ml_filterrag model's training "
        "set is NOT verified by this run.**",
        "- Dry-run: no LLM generation, no `llm.query()`, no GPT/API call made.",
        "",
        "## Pooled detection-quality metrics",
        "",
        "| Metric | ML-FilterRAG-top-k | Threshold FilterRAG (semantic) |",
        "|---|---|---|",
    ]
    metric_keys = [
        ("n_queries_x_k", "N (query, k) combos evaluated"),
        ("n_retrieved_poison", "Total retrieved poison passages"),
        ("n_retrieved_clean", "Total retrieved clean passages"),
        ("n_removed_poison", "Total removed poison passages"),
        ("n_removed_clean", "Total removed clean passages"),
        ("poison_recall_pooled", "Poison recall (pooled)"),
        ("clean_false_positive_rate_pooled", "Clean false-positive rate (pooled)"),
        ("residual_poison_fraction_pooled", "Residual poison fraction (pooled)"),
        ("mean_latency_defense_sec", "Mean defense latency (sec)"),
    ]
    for key, label in metric_keys:
        lines.append(f"| {label} | {_fmt(ml_metrics[key])} | {_fmt(filterrag_metrics[key])} |")
    lines.append("")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote comparison report to {path}")


def main():
    args = parse_args()
    if not args.dry_run:
        raise NotImplementedError(
            "--dry_run False is not implemented -- scripts/evaluate_ml_filterrag.py never runs "
            "live generation or llm.query() (see module docstring / "
            "docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md sec 6)."
        )
    print("[evaluate_ml_filterrag] No GPT/API call will be made; no llm.query() call will be made (dry-run only).")

    t0 = time.perf_counter()
    all_records = run_evaluation(args)
    print(f"[evaluate_ml_filterrag] total eval time: {time.perf_counter() - t0:.1f}s")

    ml_metrics = _pooled_metrics(all_records["ml_filterrag"])
    filterrag_metrics = _pooled_metrics(all_records["filterrag"])

    run_name = args.run_name or f"{args.eval_dataset}_ml_filterrag_vs_filterrag_{time.strftime('%Y%m%d_%H%M%S')}"
    write_jsonl(all_records["ml_filterrag"], os.path.join(args.out_dir, f"{run_name}_ml_filterrag.jsonl"))
    write_jsonl(all_records["filterrag"], os.path.join(args.out_dir, f"{run_name}_filterrag_baseline.jsonl"))
    write_comparison_report(
        os.path.join(args.out_dir, f"{run_name}_COMPARISON_REPORT.md"),
        args=args, ml_metrics=ml_metrics, filterrag_metrics=filterrag_metrics,
    )

    print(json.dumps({"ml_filterrag": ml_metrics, "filterrag": filterrag_metrics}, indent=2))


if __name__ == "__main__":
    main()
