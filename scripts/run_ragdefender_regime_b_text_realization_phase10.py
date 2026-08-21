"""PHASE 10 -- L4 RETRIEVAL-PRESERVING CHECK.

Runs retrieval ONLY for the L3-successful variant(s) whose target
candidate is attacker-controlled poison (`candidate_is_poison == True`),
per `phase10_l4_eligible_variants.json` (written by
`run_ragdefender_regime_b_text_realization_phase6_9.py`). For the Regime-B
study this is exactly ONE variant: query `5ae22b8d554299234fd0440f`,
mutation `R5` (candidate_index=6, poison_slot mapping to global adv-pool
index 36).

Reuses the EXISTING paper-faithful Contriever retrieval + poison-pool
plumbing from `scripts/run_full_retrieval_pilot_bundle1.py` (imported, not
duplicated) -- same offline template-based `LM_targeted` attack pool
construction (`Attacker.get_attack`, no LLM/API call), same dot-product
Contriever scoring, same `merge_and_topk`/`label_passages` merge logic.

Only the ONE eligible poison slot's text is replaced (this study's own R5
rewrite, not the unrelated `filterrag_targeted` mutation family that
script's `main()` uses for its own 3 selected queries) -- attack budget
(5 poison/query in the shared 50-query pool) is otherwise preserved
exactly, and every other pool query's poison text is untouched.

If (and only if) the rewritten poison passage survives into the fresh
top-10 with the intended composition, reruns the UNCHANGED paper-faithful
`ragdefender_paper` (Stella-based `concentration_stage1_paper` +
`stage2_pair_frequency`) on the newly-retrieved context.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np

import run_full_retrieval_pilot_bundle1 as pilot  # noqa: E402
from defense.passages import label_passages  # noqa: E402
from defense import ragdefender_internals as ri  # noqa: E402
import run_ragdefender_regime_b_text_realization as drv  # noqa: E402

OUTPUT_DIR = drv.OUTPUT_DIR
K = 10
DATASET_CONFIG = REPO_ROOT / "results/diagnostics/ml_filterrag_dataset_hotpotqa_50q/dataset_config.json"
INCORRECT_ANSWERS = REPO_ROOT / "results/adv_targeted_results/hotpotqa.json"
BEIR_RESULTS = REPO_ROOT / "results/beir_results/hotpotqa-contriever.json"
CORPUS_PATH = REPO_ROOT / "datasets/hotpotqa/corpus.jsonl"


class Phase10StopCondition(RuntimeError):
    pass


def load_eligible_variants():
    path = OUTPUT_DIR / "phase10_l4_eligible_variants.json"
    with open(path) as f:
        return json.load(f)


def load_rewrite_text(query_id: str, mutation_id: str) -> str:
    for name in ("rewrite_bank.jsonl", "rewrite_bank_round2.jsonl"):
        p = OUTPUT_DIR / name
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                row = json.loads(line)
                if row["query_id"] == query_id and row["mutation_id"] == mutation_id:
                    return row["rewritten_text"], row["original_text"]
    raise Phase10StopCondition(f"Rewrite text not found for {query_id}/{mutation_id}.")


def main() -> None:
    eligible = load_eligible_variants()
    if not eligible:
        print("PHASE 10: 0 L4-eligible variants (no poison-target FULL L3 realization). Nothing to run.")
        with open(OUTPUT_DIR / "regime_b_text_realization_retrieval.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "mutation_id", "candidate_is_poison", "full_realization", "note"])
        return

    cases = drv.load_frozen_failures()
    cases_by_qid = {c["query_id"]: c for c in cases}

    with open(REPO_ROOT / "results/diagnostics/ragdefender_expanded_baseline/recovered_contexts.json") as f:
        recovered_contexts = {c["query_id"]: c for c in json.load(f)}

    full_pool_query_ids = pilot.load_full_pool_query_ids(str(DATASET_CONFIG))
    with open(INCORRECT_ANSWERS) as f:
        incorrect_answers = json.load(f)
    with open(BEIR_RESULTS) as f:
        beir_results = json.load(f)

    rows_out = []
    for variant in eligible:
        qid = variant["query_id"]
        mutation_id = variant["mutation_id"]
        case = cases_by_qid[qid]
        idx = case["candidate_index"]
        ctx = recovered_contexts[qid]
        original_doc_id = ctx["doc_ids"][idx]
        global_index = pilot.extract_global_index(original_doc_id)

        rewritten_text, original_text_from_bank = load_rewrite_text(qid, mutation_id)
        if original_text_from_bank != case["texts"][idx]:
            raise Phase10StopCondition(f"{qid}/{mutation_id}: rewrite-bank original_text mismatch vs frozen context.")

        if qid not in full_pool_query_ids:
            raise Phase10StopCondition(f"{qid} is not part of the 50-query adversarial pool.")
        if qid not in beir_results:
            raise Phase10StopCondition(f"{qid} missing from {BEIR_RESULTS}.")

        print(f"[phase10] {qid}/{mutation_id}: target doc_id={original_doc_id} global_index={global_index}")

        print("[phase10] loading Contriever retrieval model (offline)...")
        import torch  # noqa: PLC0415
        from src.utils import load_models as load_retrieval_models  # noqa: PLC0415

        model, c_model, tokenizer, get_emb = load_retrieval_models(pilot.EVAL_MODEL_CODE)
        model.eval().to(pilot.RETRIEVAL_DEVICE)
        c_model.eval().to(pilot.RETRIEVAL_DEVICE)

        print(f"[phase10] rebuilding the {len(full_pool_query_ids)}-query adversarial pool (offline template, no LLM)...")
        baseline_adv_text_list = pilot.build_full_pool_adv_text_list(
            full_pool_query_ids, incorrect_answers, model=model, c_model=c_model, tokenizer=tokenizer, get_emb=get_emb
        )
        if baseline_adv_text_list[global_index] != case["texts"][idx]:
            raise Phase10StopCondition(
                f"{qid}: rebuilt adv-pool text at global_index={global_index} does not match the frozen "
                "original candidate text -- pool reconstruction drift."
            )

        mutated_adv_text_list = list(baseline_adv_text_list)
        mutated_adv_text_list[global_index] = rewritten_text
        replaced_indices = [global_index]
        # Attack-budget check: exactly one index changed, pool size unchanged.
        if len(mutated_adv_text_list) != len(baseline_adv_text_list):
            raise Phase10StopCondition("Pool size changed -- must not augment the corpus.")
        n_changed = sum(1 for a, b in zip(baseline_adv_text_list, mutated_adv_text_list) if a != b)
        if n_changed != 1:
            raise Phase10StopCondition(f"Expected exactly 1 changed pool text, found {n_changed}.")

        print("[phase10] embedding baseline + mutated adversarial pools with Contriever...")
        baseline_adv_embs = pilot.embed_texts(
            baseline_adv_text_list, model=c_model, tokenizer=tokenizer, get_emb=get_emb, device=pilot.RETRIEVAL_DEVICE
        )
        mutated_adv_embs = pilot.embed_texts(
            mutated_adv_text_list, model=c_model, tokenizer=tokenizer, get_emb=get_emb, device=pilot.RETRIEVAL_DEVICE
        )

        question = incorrect_answers[qid]["question"]
        query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
        query_input = {k: v.to(pilot.RETRIEVAL_DEVICE) for k, v in query_input.items()}
        with torch.no_grad():
            query_emb = get_emb(model, query_input)

        baseline_scores = pilot.score_adv_texts_against_query(baseline_adv_embs, query_emb, pilot.SCORE_FUNCTION)
        mutated_scores = pilot.score_adv_texts_against_query(mutated_adv_embs, query_emb, pilot.SCORE_FUNCTION)

        clean_topk_doc_ids = list(beir_results[qid].keys())[:K]
        print(f"[phase10] streaming corpus.jsonl for {len(clean_topk_doc_ids)} clean doc_id(s)...")
        clean_texts = pilot.stream_corpus_texts(str(CORPUS_PATH), clean_topk_doc_ids)
        clean_entries = [
            {"score": beir_results[qid][d], "context": clean_texts[d], "doc_id": d} for d in clean_topk_doc_ids
        ]

        baseline_topk = pilot.merge_and_topk(clean_entries, baseline_adv_text_list, baseline_scores, qid=qid, k=K)
        mutated_topk = pilot.merge_and_topk(clean_entries, mutated_adv_text_list, mutated_scores, qid=qid, k=K)

        baseline_passages = label_passages(baseline_topk)
        mutated_passages = label_passages(mutated_topk)

        target_doc_id = pilot._adv_doc_id(qid, global_index)  # noqa: SLF001
        baseline_rank = next((p.rank + 1 for p in baseline_passages if p.doc_id == target_doc_id), None)
        mutated_rank = next((p.rank + 1 for p in mutated_passages if p.doc_id == target_doc_id), None)
        still_retrieved = mutated_rank is not None

        n_poison_mutated = sum(1 for p in mutated_passages if p.is_poison)
        n_clean_mutated = len(mutated_passages) - n_poison_mutated
        topk_membership_changed = sorted(p.doc_id for p in baseline_passages) != sorted(p.doc_id for p in mutated_passages)

        print(
            f"[phase10] original_rank={baseline_rank} rewritten_rank={mutated_rank} "
            f"still_retrieved={still_retrieved} total_poison_M_after_rewrite={n_poison_mutated} "
            f"clean_count={n_clean_mutated} topk_membership_changed={topk_membership_changed}"
        )

        row = {
            "query_id": qid,
            "mutation_id": mutation_id,
            "candidate_is_poison": True,
            "full_realization": True,
            "target_doc_id": target_doc_id,
            "original_poison_rank": baseline_rank,
            "rewritten_poison_rank": mutated_rank,
            "still_retrieved_in_topk": still_retrieved,
            "topk_membership_changed": topk_membership_changed,
            "total_retrieved_poison_M_after_rewrite": n_poison_mutated,
            "clean_count_after_rewrite": n_clean_mutated,
        }

        if still_retrieved:
            # Build the new 10-passage context (doc_id-sorted, matching
            # recovered_contexts.json's own convention) and rerun the
            # UNCHANGED paper-faithful Stage 1 + Stage 2 with Stella.
            sorted_passages = sorted(mutated_passages, key=lambda p: p.doc_id)
            new_texts = [p.text for p in sorted_passages]
            new_is_poison = [p.is_poison for p in sorted_passages]

            s_model, st_util, _ = drv.load_stella_model()
            matrix = drv.encode_matrix(s_model, st_util, new_texts)
            stage1 = ri.concentration_stage1_paper(matrix)
            stage2 = ri.stage2_pair_frequency(matrix, n_adv=stage1.n_adv_estimated, p=2.0)
            removed_poison = sum(1 for i in stage2.selected_indices if new_is_poison[i])
            removed_clean = len(stage2.selected_indices) - removed_poison
            residual_poison = n_poison_mutated - removed_poison

            row.update(
                {
                    "l4_stage1_n_adv": stage1.n_adv_estimated,
                    "l4_stage2_removed_poison": removed_poison,
                    "l4_stage2_removed_clean": removed_clean,
                    "l4_stage2_residual_poison": residual_poison,
                }
            )
            print(
                f"[phase10] L4 rerun: N_adv={stage1.n_adv_estimated} removed_poison={removed_poison} "
                f"removed_clean={removed_clean} residual_poison={residual_poison}"
            )
        else:
            row.update(
                {
                    "l4_stage1_n_adv": None,
                    "l4_stage2_removed_poison": None,
                    "l4_stage2_removed_clean": None,
                    "l4_stage2_residual_poison": None,
                }
            )

        rows_out.append(row)

    out_path = OUTPUT_DIR / "regime_b_text_realization_retrieval.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
