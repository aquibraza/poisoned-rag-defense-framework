#!/usr/bin/env python3
"""FilterRAG score-inspection / epsilon-calibration tool.

The diagnostic sweep in scripts/run_ragdefender_k_sweep.py showed that
`filterrag` and `filterrag_query_only` made byte-for-byte identical removal
decisions at epsilon=0.2 on the HotpotQA 10-query set -- i.e. the SLM step
wasn't contributing anything yet. This script exists to find out *why*,
without spending any additional GPT-4 calls or re-running the SLM more than
once per passage:

1. Reproduces the *exact* same retrieval + LM_targeted adversarial-injection
   pipeline main.py uses (same corpus/queries/qrels, same BEIR top-k results,
   same `Attacker.get_attack()` templated adv texts, same embedding
   similarity re-ranking) for a fixed set of target queries -- entirely
   independent of defense/dispatch.py, so this never calls run_defense() and
   never needs to guess an epsilon up front.
2. For every retrieved passage at every requested k, computes BOTH the full
   (SLM-backed) Freq-Density score and the query-only ablation score (see
   defense/filterrag.py), plus the SLM-generated answer text and an
   "answer-only" Freq-Density score that isolates just the SLM answer's
   token contribution (excludes the query's own tokens) -- this is the
   number that should differ from query_only_score if the SLM step is doing
   anything at all.
3. Writes the full per-passage score table to CSV.
4. Sweeps a list of candidate epsilon thresholds *directly on the already-
   computed scores* (no re-running of retrieval/SLM per epsilon) and reports
   the same detection-quality metrics defense/diagnostics.py computes
   (mean_removed_poison, mean_removed_clean, mean_poison_recall,
   mean_clean_false_positive_rate, mean_residual_poison_fraction) plus a
   count of queries left with an empty final context, for both `filterrag`
   and `filterrag_query_only` at each epsilon -- so `full` vs `query_only`
   can be compared epsilon-by-epsilon.
5. Prints a handful of illustrative examples (poisoned-and-removed,
   clean-falsely-removed, clean-correctly-kept) at epsilon=0.2.

This makes zero live GPT-4/PaLM/etc. calls (LM_targeted's adversarial texts
are pre-generated, read from results/adv_targeted_results/; see
src/attack.py:Attacker.get_attack). It does run the local FilterRAG SLM
(google/flan-t5-small by default) once per retrieved passage, same as
`--defense filterrag` would.

Usage:
    python scripts/filterrag_score_inspection.py \
        --eval_dataset hotpotqa --k_values 5 10 --N 5 --max_queries 10 \
        --out_dir results/diagnostics/filterrag_calibration_10q
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)  # load_beir_datasets() resolves ./datasets relative to cwd, like main.py

import torch  # noqa: E402

import fix_sentence_transformers  # noqa: E402,F401 -- match main.py's compat patch

from src.utils import load_beir_datasets, load_json, load_models  # noqa: E402
from src.attack import Attacker  # noqa: E402

from defense.filterrag import _tokenize, freq_density, local_hf_slm_answer_fn, score_passages  # noqa: E402
from defense.passages import label_passages  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_dataset", default="hotpotqa", choices=["nq", "hotpotqa", "msmarco"])
    parser.add_argument("--eval_model_code", default="contriever")
    parser.add_argument("--split", default="test")
    parser.add_argument("--score_function", default="dot", choices=["dot", "cos_sim"])
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--N", type=int, default=5, help="adv_per_query")
    parser.add_argument("--max_queries", type=int, default=10)
    parser.add_argument("--filterrag_slm_model", default="google/flan-t5-small")
    parser.add_argument("--filterrag_slm_device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--epsilons", nargs="+", type=float, default=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    )
    parser.add_argument("--out_dir", default="results/diagnostics/filterrag_calibration_10q")
    parser.add_argument("--n_examples_per_category", type=int, default=5)
    parser.add_argument("--example_epsilon", type=float, default=0.20)
    parser.add_argument("--text_preview_chars", type=int, default=150)
    return parser.parse_args()


def build_passage_records(args) -> List[Dict]:
    """Reproduce main.py's retrieval + LM_targeted injection pipeline for
    the first `--max_queries` target queries (matching main.py's default
    --random_targets False, iter=0 behavior), and compute FilterRAG scores
    for every retrieved passage at every requested k.

    Returns a flat list of per-passage dicts (one per (k, query, passage)).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[calibration] retrieval device: {device}")

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

    class _AttackArgs:
        attack_method = "LM_targeted"
        adv_per_query = args.N
        eval_dataset = args.eval_dataset

    attacker = Attacker(_AttackArgs(), model=model, c_model=c_model, tokenizer=tokenizer, get_emb=get_emb)

    target_queries_idx = list(range(args.max_queries))  # matches main.py's default (random_targets=False, iter=0)
    target_queries = [incorrect_answers[idx]["question"] for idx in target_queries_idx]
    for iter_idx, i in enumerate(target_queries_idx):
        top1_idx = list(results[incorrect_answers[i]["id"]].keys())[0]
        top1_score = results[incorrect_answers[i]["id"]][top1_idx]
        target_queries[iter_idx] = {
            "query": target_queries[iter_idx], "top1_score": top1_score, "id": incorrect_answers[i]["id"],
        }

    adv_text_groups = attacker.get_attack(target_queries)  # offline/templated -- no LLM call, see module docstring
    adv_text_list = sum(adv_text_groups, [])
    adv_input = tokenizer(adv_text_list, padding=True, truncation=True, return_tensors="pt")
    adv_input = {key: value.to(device) for key, value in adv_input.items()}
    with torch.no_grad():
        adv_embs = get_emb(c_model, adv_input)

    print(f"[calibration] loading FilterRAG SLM ({args.filterrag_slm_model})...")
    slm_answer_fn = local_hf_slm_answer_fn(args.filterrag_slm_model, device=args.filterrag_slm_device)

    all_rows: List[Dict] = []
    for k in args.k_values:
        k_t0 = time.perf_counter()
        for iter_idx, i in enumerate(target_queries_idx):
            qid = incorrect_answers[i]["id"]
            question = incorrect_answers[i]["question"]

            topk_idx = list(results[qid].keys())[:k]
            topk_results = [
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
                topk_results.append({
                    "score": adv_sim, "context": adv_text_list[j], "doc_id": f"adv::{qid}::{j}",
                    "source": "adversarial", "is_poison": True,
                })
            topk_results = sorted(topk_results, key=lambda x: float(x["score"]), reverse=True)[:k]

            passages = label_passages(topk_results)
            query_only_scores = score_passages(question, passages, slm_answer_fn=None)
            full_scores = score_passages(question, passages, slm_answer_fn=slm_answer_fn)

            for rank, p in enumerate(passages):
                fs = full_scores[rank]
                qs = query_only_scores[rank]
                assert fs["doc_id"] == p.doc_id == qs["doc_id"]
                slm_answer = fs["slm_answer"]
                answer_only_score = freq_density(p.text, _tokenize(slm_answer)) if slm_answer else 0.0
                all_rows.append({
                    "dataset": args.eval_dataset,
                    "k": k,
                    "query_id": qid,
                    "rank": rank,
                    "doc_id": p.doc_id,
                    "is_poison": p.is_poison,
                    "filterrag_score": fs["freq_density_score"],
                    "query_only_score": qs["freq_density_score"],
                    "slm_answer": slm_answer,
                    "answer_token_overlap_score": answer_only_score,
                    "text_preview": p.text[: args.text_preview_chars].replace("\n", " "),
                })
        print(f"[calibration] k={k}: scored {len(target_queries_idx)} queries in {time.perf_counter() - k_t0:.1f}s")

    return all_rows


def write_score_csv(rows: List[Dict], path: str, epsilon_for_removed_col: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = [
        "dataset", "k", "query_id", "rank", "doc_id", "is_poison",
        "filterrag_score", "query_only_score", "slm_answer", "answer_token_overlap_score",
        f"removed_at_epsilon_{epsilon_for_removed_col:.2f}_full",
        f"removed_at_epsilon_{epsilon_for_removed_col:.2f}_query_only",
        "text_preview",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                **r,
                f"removed_at_epsilon_{epsilon_for_removed_col:.2f}_full": r["filterrag_score"] >= epsilon_for_removed_col,
                f"removed_at_epsilon_{epsilon_for_removed_col:.2f}_query_only": r["query_only_score"] >= epsilon_for_removed_col,
            })
    print(f"Wrote {len(rows)} per-passage score row(s) to {path}")


def _mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def epsilon_sweep(rows: List[Dict], epsilons: Sequence[float]) -> List[Dict]:
    """Compute the same detection-quality metrics defense/diagnostics.py
    does (poison_recall, clean_false_positive_rate, residual_poison_fraction)
    per (k, defense, epsilon), directly from already-computed scores -- no
    re-running of retrieval or the SLM."""
    by_k_query: Dict[Tuple[int, str], List[Dict]] = defaultdict(list)
    for r in rows:
        by_k_query[(r["k"], r["query_id"])].append(r)

    summary_rows: List[Dict] = []
    for defense, score_key in (("filterrag", "filterrag_score"), ("filterrag_query_only", "query_only_score")):
        for k in sorted({r["k"] for r in rows}):
            keys = [key for key in by_k_query if key[0] == k]
            for epsilon in epsilons:
                removed_poison_list, removed_clean_list = [], []
                poison_recall_list, clean_fpr_list, residual_frac_list = [], [], []
                n_empty_context = 0
                for key in keys:
                    passages = by_k_query[key]
                    n_poison = sum(1 for p in passages if p["is_poison"])
                    n_clean = len(passages) - n_poison
                    removed = [p for p in passages if p[score_key] >= epsilon]
                    removed_poison = sum(1 for p in removed if p["is_poison"])
                    removed_clean = len(removed) - removed_poison
                    residual_poison = n_poison - removed_poison
                    residual_clean = n_clean - removed_clean

                    removed_poison_list.append(removed_poison)
                    removed_clean_list.append(removed_clean)
                    poison_recall_list.append(removed_poison / n_poison if n_poison > 0 else None)
                    clean_fpr_list.append(removed_clean / n_clean if n_clean > 0 else None)
                    denom = residual_poison + residual_clean
                    residual_frac_list.append(residual_poison / denom if denom > 0 else None)
                    if denom == 0:
                        n_empty_context += 1

                summary_rows.append({
                    "defense": defense,
                    "k": k,
                    "epsilon": epsilon,
                    "n_queries": len(keys),
                    "mean_removed_poison": _mean(removed_poison_list),
                    "mean_removed_clean": _mean(removed_clean_list),
                    "mean_poison_recall": _mean(poison_recall_list),
                    "mean_clean_false_positive_rate": _mean(clean_fpr_list),
                    "mean_residual_poison_fraction": _mean(residual_frac_list),
                    "n_queries_empty_context": n_empty_context,
                })
    return summary_rows


def write_sweep_csv(summary_rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = [
        "defense", "k", "epsilon", "n_queries", "mean_removed_poison", "mean_removed_clean",
        "mean_poison_recall", "mean_clean_false_positive_rate", "mean_residual_poison_fraction",
        "n_queries_empty_context",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(f"Wrote {len(summary_rows)} epsilon-sweep summary row(s) to {path}")


def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_sweep_markdown(summary_rows: List[Dict], path: str) -> None:
    lines = [
        "# FilterRAG epsilon calibration sweep",
        "",
        "Computed directly from cached per-passage scores (no re-running of retrieval/SLM per epsilon).",
        "",
        "| defense | k | epsilon | n | mean_removed_poison | mean_removed_clean | mean_poison_recall | "
        "mean_clean_FP_rate | mean_residual_poison_fraction | n_empty_context |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(summary_rows, key=lambda r: (r["k"], r["epsilon"], r["defense"])):
        lines.append(
            f"| {r['defense']} | {r['k']} | {r['epsilon']:.2f} | {r['n_queries']} | "
            f"{_fmt(r['mean_removed_poison'])} | {_fmt(r['mean_removed_clean'])} | "
            f"{_fmt(r['mean_poison_recall'])} | {_fmt(r['mean_clean_false_positive_rate'])} | "
            f"{_fmt(r['mean_residual_poison_fraction'])} | {r['n_queries_empty_context']} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote epsilon-sweep report to {path}")


def write_examples_markdown(rows: List[Dict], path: str, epsilon: float, n_per_category: int) -> None:
    # k=10 has both clean and poison passages present; k=5 (100% poison in
    # this attack setup) has no clean passages, so it can't illustrate
    # false-positive/true-negative examples.
    k10_rows = [r for r in rows if r["k"] == 10]

    poisoned_removed = [r for r in k10_rows if r["is_poison"] and r["filterrag_score"] >= epsilon]
    clean_removed = [r for r in k10_rows if not r["is_poison"] and r["filterrag_score"] >= epsilon]
    clean_kept = [r for r in k10_rows if not r["is_poison"] and r["filterrag_score"] < epsilon]

    lines = [
        "# FilterRAG score-inspection examples",
        "",
        f"epsilon={epsilon:.2f}, k=10, dataset=hotpotqa. All examples use the full "
        "(SLM-backed) `filterrag_score`.",
        "",
    ]

    def _section(title: str, examples: List[Dict]):
        lines.append(f"## {title} (showing up to {n_per_category})")
        lines.append("")
        if not examples:
            lines.append("_None found in this sample._")
            lines.append("")
            return
        for r in examples[:n_per_category]:
            lines.append(f"- **query_id={r['query_id']}** rank={r['rank']} is_poison={r['is_poison']}")
            lines.append(f"  - filterrag_score={_fmt(r['filterrag_score'])}, query_only_score={_fmt(r['query_only_score'])}, "
                          f"answer_token_overlap_score={_fmt(r['answer_token_overlap_score'])}")
            lines.append(f"  - slm_answer: {r['slm_answer']!r}")
            lines.append(f"  - text preview: {r['text_preview']!r}")
        lines.append("")

    _section("Poisoned passages correctly removed", poisoned_removed)
    _section("Clean passages falsely removed", clean_removed)
    _section("Clean passages correctly kept", clean_kept)

    diverging = sorted(
        k10_rows, key=lambda r: abs(r["filterrag_score"] - r["query_only_score"]), reverse=True
    )
    diverging = [r for r in diverging if abs(r["filterrag_score"] - r["query_only_score"]) > 1e-9]
    lines.append(f"## Largest full-vs-query_only score divergence (showing up to {n_per_category})")
    lines.append("")
    lines.append(f"{len(diverging)}/{len(k10_rows)} passages at k=10 have any nonzero divergence at all "
                  "(i.e. the SLM answer's tokens contributed something the query's own tokens didn't).")
    lines.append("")
    if not diverging:
        lines.append("_None found in this sample._")
        lines.append("")
    else:
        for r in diverging[:n_per_category]:
            lines.append(f"- **query_id={r['query_id']}** rank={r['rank']} is_poison={r['is_poison']}")
            lines.append(f"  - filterrag_score={_fmt(r['filterrag_score'])}, query_only_score={_fmt(r['query_only_score'])}, "
                          f"delta={_fmt(r['filterrag_score'] - r['query_only_score'])}, "
                          f"answer_token_overlap_score={_fmt(r['answer_token_overlap_score'])}")
            lines.append(f"  - slm_answer: {r['slm_answer']!r}")
            lines.append(f"  - text preview: {r['text_preview']!r}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote examples report to {path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.perf_counter()
    rows = build_passage_records(args)
    print(f"[calibration] total scoring time: {time.perf_counter() - t0:.1f}s for {len(rows)} passage-score rows")

    score_csv_path = os.path.join(args.out_dir, "filterrag_score_inspection.csv")
    write_score_csv(rows, score_csv_path, epsilon_for_removed_col=args.example_epsilon)

    sweep_rows = epsilon_sweep(rows, args.epsilons)
    write_sweep_csv(sweep_rows, os.path.join(args.out_dir, "filterrag_epsilon_sweep.csv"))
    write_sweep_markdown(sweep_rows, os.path.join(args.out_dir, "FILTERRAG_EPSILON_SWEEP_REPORT.md"))

    write_examples_markdown(
        rows, os.path.join(args.out_dir, "FILTERRAG_SCORE_EXAMPLES.md"),
        epsilon=args.example_epsilon, n_per_category=args.n_examples_per_category,
    )


if __name__ == "__main__":
    main()
