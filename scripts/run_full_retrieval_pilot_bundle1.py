#!/usr/bin/env python3
"""Full-retrieval rerun of the 3 strongest normalized `filterrag_targeted`
mutation cases from `manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/`,
for query_ids `5a8e068b5542995085b37384`, `5ae224da554299234fd043ee`,
`5ae22b8d554299234fd0440f`.

Unlike `scripts/run_text_mutation_fixed_context_eval.py` /
`scripts/run_targeted_mutation_bundle_1_eval.py` (which never rerun
retrieval and reconstruct the fixed k=10 context verbatim from CSV
artifacts), this script actually **reruns dense retrieval** (Contriever,
dot-product scoring, exactly matching `scripts/build_ml_filterrag_dataset.py`
/ `scripts/evaluate_ml_filterrag.py` / `main.py`'s own retrieval+injection
pipeline) against a corpus variant in which each selected query's 5 original
poisoned passages have been **replaced** (not augmented) by that query's 5
normalized `filterrag_targeted` mutated passages, to test whether the
mutated poison still survives into the fresh top-10 and, if so, whether it
still weakens the three defenses.

**Strict constraints honored:**

- No GPT/API call is ever made, no `llm.query()` call is ever made (the
  `LM_targeted` "attack" is a pure offline string-template substitution --
  see `src/attack.py::Attacker.get_attack` -- not a live model call).
- No model is trained or retrained: Contriever, the FilterRAG SLM
  (`google/flan-t5-small`), the ML-FilterRAG perplexity LM (`distilgpt2`),
  the RAGDefender similarity model (`paraphrase-MiniLM-L6-v2`), and the
  ML-FilterRAG-top-k random-forest classifier
  (`models/ml_filterrag/hotpotqa_50q_mlfilterrag_topk_rf.joblib`) are all
  loaded read-only for inference.
- `top_k=10` only.
- Attack budget preserved: exactly 5 poisoned passages per selected query,
  before and after mutation -- the 5 mutated passages *replace* the 5
  original poisoned passages in the shared 50-query adversarial candidate
  pool (see `build_replacement_plan`/`apply_replacements`); the pool is
  never augmented with both original and mutated poison for the same query
  in this primary run.
- Retrieval is rerun only for the 3 selected queries (`SELECTED_QUERY_IDS`);
  the other 47 pool queries' own poison text is left untouched and is never
  scored/reported on here (it exists only so the shared adversarial
  candidate pool has the same composition as the original 50-query dataset
  build -- see module docstring of `scripts/build_ml_filterrag_dataset.py`
  for why a query's retrieved top-k can include another pool query's
  adversarial text).
- No defense code (`defense/*.py`) is modified; every defense function used
  here (`defense.dispatch.run_defense`, `defense.filterrag.filterrag_defense`,
  `defense.ml_filterrag.extract_features`, `defense.ragdefender_internals`)
  is imported and reused unmodified via
  `scripts/run_text_mutation_fixed_context_eval.py`'s existing
  `score_context()` (same function the fixed-context pilots already use).

**Efficiency note (does not change retrieval semantics):** rather than
loading BEIR's `GenericDataLoader` (which parses the full ~5.2M-passage
`datasets/hotpotqa/corpus.jsonl` into memory), this script streams that same
file once and extracts text only for the small set of already-known clean
top-k doc_ids taken from the existing precomputed
`results/beir_results/hotpotqa-contriever.json` (the same file
`build_ml_filterrag_dataset.py`/`evaluate_ml_filterrag.py` read from) --
mathematically identical clean-passage candidates and scores, just without
loading BEIR's qrels/queries or the ~5.2M irrelevant passages for a 3-query
pilot.

**Mutated-passage embedding semantics:** the normalized `mutated_text` field
(see `manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/normalized/`)
is, by design (see `Mutation_Prompts.rtf`'s FilterRAG-targeted prompt,
"minimize lexical and semantic overlap with the question wording"), the
*complete* replacement passage text with no question prefix. This script
embeds it exactly as `scripts/run_text_mutation_fixed_context_eval.py`'s
`build_mutated_context()` already treats it for defense scoring: as the
full passage text, not re-concatenated with the `question + "."` prefix
`Attacker.get_attack()` prepends to the *original* LM_targeted poison. This
is a deliberate, documented modeling choice (attacker publishes a
self-contained rewritten passage), not an oversight.

Usage:
    python scripts/run_full_retrieval_pilot_bundle1.py \\
        --pilot_dir manual_text_mutation_pilot/hotpotqa_50q_k10 \\
        --out_dir manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/full_retrieval_pilot
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import run_text_mutation_fixed_context_eval as base_eval  # noqa: E402

from defense.passages import RetrievedPassage, label_passages  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed pilot configuration -- the 3 selected normalized filterrag_targeted
# cases named explicitly in the task.
# ---------------------------------------------------------------------------

SELECTED_QUERY_IDS: Tuple[str, ...] = (
    "5a8e068b5542995085b37384",
    "5ae224da554299234fd043ee",
    "5ae22b8d554299234fd0440f",
)
FAMILY = "filterrag_targeted"
BUNDLE_ID = "filterrag_targeted"
BUNDLE_DIR_NAME = "mutation_bundle_1"

EVAL_DATASET = "hotpotqa"
EVAL_MODEL_CODE = "contriever"
SCORE_FUNCTION = "dot"
ATTACK_METHOD = "LM_targeted"
N_ADV_PER_QUERY = 5
K = 10
RETRIEVAL_DEVICE = "cpu"  # deterministic; matches base_eval.DEVICE for defense scoring

DEFAULT_PILOT_DIR = base_eval.DEFAULT_PILOT_DIR
DEFAULT_BUNDLE_DIR = os.path.join(DEFAULT_PILOT_DIR, BUNDLE_DIR_NAME)
DEFAULT_ML_MODEL_PATH = base_eval.DEFAULT_ML_MODEL_PATH
DEFAULT_DATASET_CONFIG = "results/diagnostics/ml_filterrag_dataset_hotpotqa_50q/dataset_config.json"
DEFAULT_INCORRECT_ANSWERS = f"results/adv_targeted_results/{EVAL_DATASET}.json"
DEFAULT_BEIR_RESULTS = f"results/beir_results/{EVAL_DATASET}-{EVAL_MODEL_CODE}.json"
DEFAULT_CORPUS_PATH = f"datasets/{EVAL_DATASET}/corpus.jsonl"
DEFAULT_FIXED_CONTEXT_SCORES = os.path.join(
    DEFAULT_BUNDLE_DIR, "evaluation_normalized", "normalized_targeted_family_bundle_scores.csv"
)


def _adv_doc_id(qid: str, global_idx: int) -> str:
    return f"adv::{ATTACK_METHOD}::{qid}::{global_idx}"


# ---------------------------------------------------------------------------
# 1. Pure parsing / bookkeeping (no model, no I/O beyond simple file reads;
#    unit-testable without any heavy dependency).
# ---------------------------------------------------------------------------

def load_full_pool_query_ids(dataset_config_path: str) -> List[str]:
    """The exact, ordered 50-query adversarial candidate pool
    `scripts/build_ml_filterrag_dataset.py` used to build
    `results/diagnostics/ml_filterrag_dataset_hotpotqa_50q/` -- pool
    position `p` (0-indexed) contributes global adv-text indices
    `[p*N .. p*N+N-1]` to the flat `adv_text_list` (see
    `build_full_pool_adv_text_list`). Required so the 3 selected queries'
    canonical `adv::LM_targeted::<qid>::<j>` doc_ids resolve to the exact
    same `j` they did when `mutation_input_passages.csv` was built."""
    with open(dataset_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    pool = cfg.get("target_query_ids")
    if not pool:
        raise ValueError(f"{dataset_config_path}: missing/empty 'target_query_ids'.")
    return list(pool)


def extract_global_index(doc_id: str) -> int:
    """Parse the trailing `::<int>` off an `adv::LM_targeted::<qid>::<j>`
    doc_id. Raises ValueError on any other shape (never silently guesses)."""
    parts = doc_id.split("::")
    if len(parts) < 4 or parts[0] != "adv":
        raise ValueError(f"doc_id={doc_id!r} does not look like an adv::... doc_id.")
    try:
        return int(parts[-1])
    except ValueError as exc:
        raise ValueError(f"doc_id={doc_id!r} has a non-integer trailing index.") from exc


@dataclass
class NormalizedSlot:
    poison_slot: int
    doc_id: str
    source_file_doc_id: Optional[str]
    mutated_text: str


@dataclass
class NormalizedRecord:
    query_id: str
    family: str
    question: str
    target_wrong_answer: str
    slots: Dict[int, NormalizedSlot] = field(default_factory=dict)


def load_normalized_family_file(path: str) -> Dict[str, NormalizedRecord]:
    """Parse `normalized/filterrag_targeted.normalized.jsonl` (one compact
    JSON object per line, see module docstring of
    `scripts/audit_normalize_mutation_bundle_1.py`). Returns
    `{query_id: NormalizedRecord}`. Raises on malformed rows (missing
    query_id, wrong slot count, duplicate/missing poison_slot, empty
    mutated_text) -- never silently drops/invents data."""
    out: Dict[str, NormalizedRecord] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("query_id")
            if not qid or not isinstance(qid, str):
                raise ValueError(f"{path}: line {line_no} missing a valid 'query_id'.")
            mutated_passages = rec.get("mutated_passages")
            if not isinstance(mutated_passages, list) or len(mutated_passages) != 5:
                raise ValueError(
                    f"{path}: query_id={qid!r} has "
                    f"{len(mutated_passages) if isinstance(mutated_passages, list) else 'no'} "
                    "mutated_passages (expected exactly 5)."
                )
            slots: Dict[int, NormalizedSlot] = {}
            for m in mutated_passages:
                slot = m.get("poison_slot")
                if slot is None:
                    raise ValueError(f"{path}: query_id={qid!r} has a passage missing 'poison_slot'.")
                slot = int(slot)
                if slot in slots:
                    raise ValueError(f"{path}: query_id={qid!r} has a duplicate poison_slot={slot!r}.")
                text = m.get("mutated_text")
                if not text or not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{path}: query_id={qid!r} poison_slot={slot!r} has empty mutated_text.")
                doc_id = m.get("doc_id")
                if not doc_id:
                    raise ValueError(f"{path}: query_id={qid!r} poison_slot={slot!r} is missing 'doc_id'.")
                slots[slot] = NormalizedSlot(
                    poison_slot=slot, doc_id=doc_id,
                    source_file_doc_id=m.get("source_file_doc_id"), mutated_text=text,
                )
            if sorted(slots) != [0, 1, 2, 3, 4]:
                raise ValueError(f"{path}: query_id={qid!r} has poison_slot values {sorted(slots)!r}.")
            out[qid] = NormalizedRecord(
                query_id=qid, family=rec.get("family", ""), question=rec.get("question", ""),
                target_wrong_answer=rec.get("target_wrong_answer", ""), slots=slots,
            )
    return out


@dataclass
class ReplacementSlot:
    query_id: str
    poison_slot: int
    original_doc_id: str
    global_index: int
    original_poison_text: str
    mutated_text: str
    source_file_doc_id: Optional[str]
    mutation_family: str
    bundle_id: str
    bundle_dir: str


def build_replacement_plan(
    selected_query_ids: Sequence[str],
    poison_by_query: Dict[str, List[Dict]],
    normalized_by_qid: Dict[str, NormalizedRecord],
    *,
    expected_family: str = FAMILY,
    bundle_id: str = BUNDLE_ID,
    bundle_dir: str = BUNDLE_DIR_NAME,
) -> Dict[str, Dict[int, ReplacementSlot]]:
    """Cross-check + merge `mutation_input_passages.csv` (authoritative
    query_id/poison_slot/doc_id/original_poison_text mapping) against the
    normalized mutation family file's `mutated_text`, for exactly the 3
    selected queries. Raises immediately (never silently degrades) if:
    - a selected query_id is missing from either source;
    - a query does not have exactly 5 poison rows / 5 mutated slots;
    - the normalized record's family isn't `expected_family`;
    - a normalized slot's `doc_id` doesn't match the CSV's canonical
      `doc_id` for that `(query_id, poison_slot)` (metadata-mapping
      preservation check)."""
    plan: Dict[str, Dict[int, ReplacementSlot]] = {}
    for qid in selected_query_ids:
        if qid not in poison_by_query:
            raise ValueError(f"query_id={qid!r} is missing from mutation_input_passages.csv.")
        if qid not in normalized_by_qid:
            raise ValueError(f"query_id={qid!r} is missing from the normalized {expected_family} file.")
        rows = poison_by_query[qid]
        if len(rows) != 5:
            raise ValueError(f"query_id={qid!r} has {len(rows)} poison row(s) in the CSV (expected 5).")
        norm = normalized_by_qid[qid]
        if norm.family != expected_family:
            raise ValueError(
                f"query_id={qid!r}: normalized record family={norm.family!r} != expected {expected_family!r}."
            )
        doc_id_by_slot = {int(r["poison_slot"]): r["doc_id"] for r in rows}
        text_by_slot = {int(r["poison_slot"]): r["original_poison_text"] for r in rows}
        if sorted(doc_id_by_slot) != [0, 1, 2, 3, 4]:
            raise ValueError(f"query_id={qid!r}: CSV poison_slot values {sorted(doc_id_by_slot)!r} != 0..4.")

        slots: Dict[int, ReplacementSlot] = {}
        for slot in range(5):
            csv_doc_id = doc_id_by_slot[slot]
            norm_slot = norm.slots.get(slot)
            if norm_slot is None:
                raise ValueError(f"query_id={qid!r}: normalized file missing poison_slot={slot}.")
            if norm_slot.doc_id != csv_doc_id:
                raise ValueError(
                    f"query_id={qid!r} poison_slot={slot}: normalized doc_id={norm_slot.doc_id!r} "
                    f"!= canonical CSV doc_id={csv_doc_id!r} -- doc_id/poison_slot mapping is not "
                    "preserved; refusing to guess."
                )
            slots[slot] = ReplacementSlot(
                query_id=qid, poison_slot=slot, original_doc_id=csv_doc_id,
                global_index=extract_global_index(csv_doc_id),
                original_poison_text=text_by_slot[slot], mutated_text=norm_slot.mutated_text,
                source_file_doc_id=norm_slot.source_file_doc_id,
                mutation_family=expected_family, bundle_id=bundle_id, bundle_dir=bundle_dir,
            )
        plan[qid] = slots
    return plan


def apply_replacements(
    baseline_adv_text_list: Sequence[str],
    replacement_plan: Dict[str, Dict[int, ReplacementSlot]],
) -> Tuple[List[str], List[int]]:
    """Return `(mutated_adv_text_list, replaced_global_indices)`: a *copy*
    of `baseline_adv_text_list` with only the plan's global indices
    overwritten by their `mutated_text` -- every other pool query's poison
    text (and every clean corpus passage, which this list never contains)
    is untouched. Before overwriting, asserts the baseline text at that
    index is byte-identical to the plan's own `original_poison_text` (a
    strong cross-check that the freshly-rebuilt 50-query pool lines up
    exactly with the archived pilot CSVs, i.e. that no pool-reconstruction
    drift silently occurred)."""
    mutated = list(baseline_adv_text_list)
    replaced_indices: List[int] = []
    for qid, slots in replacement_plan.items():
        for slot, r in slots.items():
            if r.global_index < 0 or r.global_index >= len(mutated):
                raise IndexError(
                    f"query_id={qid!r} poison_slot={slot}: global_index={r.global_index} out of range "
                    f"for a pool of {len(mutated)} adv texts."
                )
            actual = baseline_adv_text_list[r.global_index]
            if actual != r.original_poison_text:
                raise AssertionError(
                    f"query_id={qid!r} poison_slot={slot} global_index={r.global_index}: "
                    "freshly rebuilt adv-text pool does not match mutation_input_passages.csv's "
                    f"original_poison_text.\n  rebuilt={actual!r}\n  csv={r.original_poison_text!r}"
                )
            mutated[r.global_index] = r.mutated_text
            replaced_indices.append(r.global_index)
    return mutated, sorted(replaced_indices)


def assert_budget_preserved(
    baseline_adv_text_list: Sequence[str],
    mutated_adv_text_list: Sequence[str],
    replaced_indices: Sequence[int],
    *,
    n_selected_queries: int,
) -> None:
    """Attack-budget invariants, asserted (never merely printed):
    - pool size unchanged (no augmentation, no truncation);
    - exactly `n_selected_queries * 5` indices were replaced (5 per query,
      no more, no fewer);
    - every replaced index's text actually changed (mutated != original --
      i.e. the original text is gone, not duplicated alongside the
      mutated text at the same slot);
    - every *other* index is byte-identical to baseline (no accidental
      collateral edits to the other 47 pool queries' own poison text)."""
    if len(baseline_adv_text_list) != len(mutated_adv_text_list):
        raise AssertionError(
            f"Pool size changed: baseline={len(baseline_adv_text_list)}, "
            f"mutated={len(mutated_adv_text_list)} -- corpus must not be augmented."
        )
    expected_n_replaced = n_selected_queries * 5
    if len(replaced_indices) != expected_n_replaced:
        raise AssertionError(
            f"Expected exactly {expected_n_replaced} replaced indices "
            f"({n_selected_queries} queries x 5 slots), got {len(replaced_indices)}."
        )
    if len(set(replaced_indices)) != len(replaced_indices):
        raise AssertionError("Duplicate global indices in replaced_indices -- budget would be inflated.")
    replaced_set = set(replaced_indices)
    for idx in replaced_set:
        if mutated_adv_text_list[idx] == baseline_adv_text_list[idx]:
            raise AssertionError(f"Index {idx} was supposed to be replaced but text is unchanged.")
    for idx in range(len(baseline_adv_text_list)):
        if idx not in replaced_set and mutated_adv_text_list[idx] != baseline_adv_text_list[idx]:
            raise AssertionError(f"Index {idx} was NOT in the replacement plan but its text changed.")


# ---------------------------------------------------------------------------
# 2. Retrieval merge/rank -- pure function over embeddings (unit-testable
#    with small fake tensors; no model loaded here).
# ---------------------------------------------------------------------------

def merge_and_topk(
    clean_entries: Sequence[Dict],
    adv_text_list: Sequence[str],
    adv_scores: Sequence[float],
    *,
    qid: str,
    k: int,
    attack_method: str = ATTACK_METHOD,
) -> List[Dict]:
    """Merge precomputed clean-corpus candidates (`clean_entries`, each
    `{"score","context","doc_id","source":"corpus","is_poison":False}`)
    with the adversarial pool (`adv_text_list`, one already-computed
    similarity score per text in `adv_scores`, same convention `main.py`/
    `scripts/build_ml_filterrag_dataset.py` use: `doc_id =
    f"adv::{attack_method}::{qid}::{j}"`, `source="adversarial"`,
    `is_poison=True`), sort descending by score, and return the top `k`.

    Pure/deterministic given its inputs -- contains no model call, so it is
    unit-testable with synthetic scores instead of real Contriever
    embeddings."""
    if len(adv_text_list) != len(adv_scores):
        raise ValueError("adv_text_list and adv_scores must be the same length.")
    merged: List[Dict] = [dict(e, source="corpus", is_poison=False) for e in clean_entries]
    for j, (text, score) in enumerate(zip(adv_text_list, adv_scores)):
        merged.append({
            "score": float(score), "context": text, "doc_id": _adv_doc_id(qid, j),
            "source": "adversarial", "is_poison": True,
        })
    merged.sort(key=lambda x: float(x["score"]), reverse=True)
    return merged[:k]


# ---------------------------------------------------------------------------
# 3. Retrieval survival stats (pure).
# ---------------------------------------------------------------------------

def retrieval_survival_stats(
    topk_passages: Sequence[RetrievedPassage], canonical_poison_doc_ids: Sequence[str]
) -> Dict:
    canonical = list(canonical_poison_doc_ids)
    doc_id_to_rank1 = {p.doc_id: p.rank + 1 for p in topk_passages if p.rank is not None}
    survived_ranks = sorted(doc_id_to_rank1[d] for d in canonical if d in doc_id_to_rank1)
    n_poison = sum(1 for p in topk_passages if p.is_poison)
    n_clean = len(topk_passages) - n_poison
    n_budget = len(canonical)
    return {
        "retrieved_poison_count": n_poison,
        "retrieved_clean_count": n_clean,
        "canonical_poison_survived_count": len(survived_ranks),
        "canonical_poison_survived_ranks": ";".join(str(r) for r in survived_ranks),
        "mean_poison_retrieval_rank": (
            float(statistics.fmean(survived_ranks)) if survived_ranks else None
        ),
        "retrieval_survival_rate": (len(survived_ranks) / n_budget) if n_budget else None,
        "all_5_poison_survive": len(survived_ranks) == n_budget and n_budget > 0,
    }


# ---------------------------------------------------------------------------
# 4. Heavy orchestration (models, corpus, real embeddings) -- main() only.
# ---------------------------------------------------------------------------

def stream_corpus_texts(corpus_path: str, wanted_doc_ids: Sequence[str]) -> Dict[str, str]:
    """Single streaming pass over `corpus.jsonl` extracting `text` only for
    `wanted_doc_ids` (a small, already-known set of clean top-k doc_ids
    from `results/beir_results/...`), instead of loading the full
    ~5.2M-passage corpus into memory via BEIR's GenericDataLoader. Uses a
    cheap substring pre-check before `json.loads` per line to keep this
    fast on a 2GB+ file."""
    wanted = set(wanted_doc_ids)
    found: Dict[str, str] = {}
    if not wanted:
        return found
    needles = [f'"_id": "{d}"' for d in wanted]
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not any(needle in line for needle in needles):
                continue
            row = json.loads(line)
            doc_id = row.get("_id")
            if doc_id in wanted:
                found[doc_id] = row.get("text", "")
                if len(found) == len(wanted):
                    break
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"{corpus_path}: could not find text for doc_id(s): {sorted(missing)}")
    return found


def build_full_pool_adv_text_list(
    full_pool_query_ids: Sequence[str], incorrect_answers: Dict[str, Dict], *,
    model, c_model, tokenizer, get_emb,
) -> List[str]:
    """Reproduce `Attacker.get_attack()`'s offline, template-based
    (no LLM/GPT call) construction of the flat 50-query x N=5 adversarial
    text pool, in the exact pool order `load_full_pool_query_ids` returns
    -- i.e. `adv_text_list[p*N + j]` is pool-position-`p`'s `j`-th LM_targeted
    text, matching the `adv::LM_targeted::<qid>::<global_index>` doc_id
    convention used throughout this pilot."""
    from src.attack import Attacker  # noqa: PLC0415 -- heavy import kept out of module scope

    class _AttackArgs:
        pass

    _AttackArgs.attack_method = ATTACK_METHOD
    _AttackArgs.adv_per_query = N_ADV_PER_QUERY
    _AttackArgs.eval_dataset = EVAL_DATASET
    attacker = Attacker(_AttackArgs(), model=model, c_model=c_model, tokenizer=tokenizer, get_emb=get_emb)

    target_queries = [
        {"query": incorrect_answers[qid]["question"], "top1_score": 0.0, "id": qid}
        for qid in full_pool_query_ids
    ]
    adv_text_groups = attacker.get_attack(target_queries)
    return sum(adv_text_groups, [])


def embed_texts(texts: Sequence[str], *, model, tokenizer, get_emb, device: str, batch_size: int = 32):
    import torch  # noqa: PLC0415

    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            chunks.append(get_emb(model, enc).cpu())
    return torch.cat(chunks, dim=0)


def score_adv_texts_against_query(adv_embs, query_emb, score_function: str) -> List[float]:
    import torch  # noqa: PLC0415

    if score_function == "dot":
        sims = torch.mm(adv_embs, query_emb.T).squeeze(-1)
    else:
        sims = torch.cosine_similarity(adv_embs, query_emb)
    return [float(v) for v in sims]


def write_csv(path: str, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def write_jsonl(path: str, rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_candidate_inputs_rows(replacement_plan: Dict[str, Dict[int, ReplacementSlot]]) -> List[Dict]:
    rows = []
    for qid, slots in replacement_plan.items():
        for slot in sorted(slots):
            r = slots[slot]
            rows.append({
                "query_id": r.query_id, "poison_slot": r.poison_slot,
                "original_doc_id": r.original_doc_id, "global_index": r.global_index,
                "mutation_family": r.mutation_family, "bundle_id": r.bundle_id,
                "bundle_dir": r.bundle_dir, "source_file_doc_id": r.source_file_doc_id,
                "original_poison_text": r.original_poison_text, "mutated_text": r.mutated_text,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot_dir", default=os.path.join(REPO_ROOT, DEFAULT_PILOT_DIR))
    parser.add_argument("--bundle_dir", default=None)
    parser.add_argument("--dataset_config", default=os.path.join(REPO_ROOT, DEFAULT_DATASET_CONFIG))
    parser.add_argument("--incorrect_answers", default=os.path.join(REPO_ROOT, DEFAULT_INCORRECT_ANSWERS))
    parser.add_argument("--beir_results", default=os.path.join(REPO_ROOT, DEFAULT_BEIR_RESULTS))
    parser.add_argument("--corpus_path", default=os.path.join(REPO_ROOT, DEFAULT_CORPUS_PATH))
    parser.add_argument("--fixed_context_scores", default=os.path.join(REPO_ROOT, DEFAULT_FIXED_CONTEXT_SCORES))
    parser.add_argument("--ml_model_path", default=os.path.join(REPO_ROOT, DEFAULT_ML_MODEL_PATH))
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    pilot_dir = args.pilot_dir
    bundle_dir = args.bundle_dir or os.path.join(pilot_dir, BUNDLE_DIR_NAME)
    out_dir = args.out_dir or os.path.join(bundle_dir, "full_retrieval_pilot")

    print("[full_retrieval_pilot] No GPT/API call will be made; no llm.query() call will be made.")
    print(f"[full_retrieval_pilot] selected query_ids: {list(SELECTED_QUERY_IDS)}")

    selected_queries = base_eval.load_selected_queries(os.path.join(pilot_dir, "selected_queries.csv"))
    poison_by_query = base_eval.load_mutation_input_passages(os.path.join(pilot_dir, "mutation_input_passages.csv"))
    clean_by_query = base_eval.load_clean_context_passages(os.path.join(pilot_dir, "clean_context_passages.csv"))
    normalized_by_qid = load_normalized_family_file(
        os.path.join(bundle_dir, "normalized", "filterrag_targeted.normalized.jsonl")
    )

    for qid in SELECTED_QUERY_IDS:
        if qid not in selected_queries:
            raise ValueError(f"query_id={qid!r} is not present in selected_queries.csv")

    replacement_plan = build_replacement_plan(SELECTED_QUERY_IDS, poison_by_query, normalized_by_qid)
    write_jsonl(
        os.path.join(out_dir, "full_retrieval_candidate_inputs.jsonl"),
        build_candidate_inputs_rows(replacement_plan),
    )
    print(f"[full_retrieval_pilot] wrote candidate-input rows for {len(replacement_plan)} query(ies)")

    full_pool_query_ids = load_full_pool_query_ids(args.dataset_config)
    for qid in SELECTED_QUERY_IDS:
        if qid not in full_pool_query_ids:
            raise ValueError(f"query_id={qid!r} is not part of the 50-query adversarial pool in {args.dataset_config!r}")

    with open(args.incorrect_answers, "r", encoding="utf-8") as f:
        incorrect_answers = json.load(f)
    with open(args.beir_results, "r", encoding="utf-8") as f:
        beir_results = json.load(f)
    for qid in SELECTED_QUERY_IDS:
        if qid not in beir_results:
            raise ValueError(f"query_id={qid!r} missing from {args.beir_results!r}")

    print(f"[full_retrieval_pilot] loading retrieval model ({EVAL_MODEL_CODE}, device={RETRIEVAL_DEVICE})...")
    import torch  # noqa: PLC0415
    from src.utils import load_models as load_retrieval_models  # noqa: PLC0415

    model, c_model, tokenizer, get_emb = load_retrieval_models(EVAL_MODEL_CODE)
    model.eval()
    model.to(RETRIEVAL_DEVICE)
    c_model.eval()
    c_model.to(RETRIEVAL_DEVICE)

    print(f"[full_retrieval_pilot] rebuilding the full {len(full_pool_query_ids)}-query adversarial pool "
          "(offline template substitution, no LLM/GPT call)...")
    baseline_adv_text_list = build_full_pool_adv_text_list(
        full_pool_query_ids, incorrect_answers, model=model, c_model=c_model, tokenizer=tokenizer, get_emb=get_emb
    )
    assert len(baseline_adv_text_list) == len(full_pool_query_ids) * N_ADV_PER_QUERY

    mutated_adv_text_list, replaced_indices = apply_replacements(baseline_adv_text_list, replacement_plan)
    assert_budget_preserved(
        baseline_adv_text_list, mutated_adv_text_list, replaced_indices,
        n_selected_queries=len(SELECTED_QUERY_IDS),
    )
    print(f"[full_retrieval_pilot] replaced {len(replaced_indices)} of {len(baseline_adv_text_list)} pool "
          "adv texts (attack budget preserved; no augmentation).")

    print("[full_retrieval_pilot] embedding baseline + mutated adversarial pools...")
    baseline_adv_embs = embed_texts(baseline_adv_text_list, model=c_model, tokenizer=tokenizer, get_emb=get_emb, device=RETRIEVAL_DEVICE)
    mutated_adv_embs = embed_texts(mutated_adv_text_list, model=c_model, tokenizer=tokenizer, get_emb=get_emb, device=RETRIEVAL_DEVICE)

    wanted_clean_doc_ids = sorted({
        doc_id for qid in SELECTED_QUERY_IDS for doc_id in list(beir_results[qid].keys())[:K]
    })
    print(f"[full_retrieval_pilot] streaming corpus.jsonl for {len(wanted_clean_doc_ids)} clean doc_id(s)...")
    clean_texts = stream_corpus_texts(args.corpus_path, wanted_clean_doc_ids)

    print("[full_retrieval_pilot] loading defense-scoring models (SLM/LM/RAGDefender embedder/ML classifier)...")
    defense_models = base_eval.load_models(args.ml_model_path)

    results_by_query_rows: List[Dict] = []
    defense_score_rows: List[Dict] = []

    for qid in SELECTED_QUERY_IDS:
        question = selected_queries[qid]["question"]
        target_wrong_answer = selected_queries[qid]["target_wrong_answer"]
        canonical_poison_doc_ids = [replacement_plan[qid][slot].original_doc_id for slot in range(5)]

        clean_topk_doc_ids = list(beir_results[qid].keys())[:K]
        clean_entries = [
            {"score": beir_results[qid][d], "context": clean_texts[d], "doc_id": d}
            for d in clean_topk_doc_ids
        ]

        query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
        query_input = {k: v.to(RETRIEVAL_DEVICE) for k, v in query_input.items()}
        with torch.no_grad():
            query_emb = get_emb(model, query_input)

        baseline_scores = score_adv_texts_against_query(baseline_adv_embs, query_emb, SCORE_FUNCTION)
        mutated_scores = score_adv_texts_against_query(mutated_adv_embs, query_emb, SCORE_FUNCTION)

        baseline_topk = merge_and_topk(clean_entries, baseline_adv_text_list, baseline_scores, qid=qid, k=K)
        mutated_topk = merge_and_topk(clean_entries, mutated_adv_text_list, mutated_scores, qid=qid, k=K)

        baseline_passages = label_passages(baseline_topk)
        mutated_passages = label_passages(mutated_topk)

        baseline_survival = retrieval_survival_stats(baseline_passages, canonical_poison_doc_ids)
        mutated_survival = retrieval_survival_stats(mutated_passages, canonical_poison_doc_ids)

        archived_poison_rows = sorted(poison_by_query[qid], key=lambda r: int(r["retrieved_rank"]))
        archived_poison_doc_ids_by_rank = [r["doc_id"] for r in archived_poison_rows]
        archived_clean_rows = sorted(clean_by_query[qid], key=lambda r: int(r["retrieved_rank"]))
        recomputed_baseline_doc_ids_ordered = [p.doc_id for p in baseline_passages]
        # Archived rows are ordered by retrieved_rank across poison+clean combined;
        # rebuild that combined order for a fair comparison.
        archived_by_rank = sorted(
            archived_poison_rows + archived_clean_rows, key=lambda r: int(r["retrieved_rank"])
        )
        archived_topk_doc_ids_by_rank = [r["doc_id"] for r in archived_by_rank]
        baseline_reproduced_exactly = recomputed_baseline_doc_ids_ordered == archived_topk_doc_ids_by_rank

        results_by_query_rows.append({
            "query_id": qid, "k": K, "question": question, "target_wrong_answer": target_wrong_answer,
            "retrieved_poison_count": mutated_survival["retrieved_poison_count"],
            "retrieved_clean_count": mutated_survival["retrieved_clean_count"],
            "canonical_poison_survived_count": mutated_survival["canonical_poison_survived_count"],
            "canonical_poison_survived_ranks": mutated_survival["canonical_poison_survived_ranks"],
            "mean_poison_retrieval_rank": mutated_survival["mean_poison_retrieval_rank"],
            "retrieval_survival_rate": mutated_survival["retrieval_survival_rate"],
            "all_5_poison_survive": mutated_survival["all_5_poison_survive"],
            "baseline_recomputed_retrieved_poison_count": baseline_survival["retrieved_poison_count"],
            "baseline_recomputed_retrieved_clean_count": baseline_survival["retrieved_clean_count"],
            "baseline_recomputed_reproduces_archived_topk_exactly": baseline_reproduced_exactly,
            "archived_baseline_retrieved_poison_count": len(archived_poison_doc_ids_by_rank),
            "archived_baseline_retrieved_clean_count": len(archived_clean_rows),
        })

        for condition, passages in (("baseline_recomputed", baseline_passages), ("mutated", mutated_passages)):
            print(f"[full_retrieval_pilot] scoring defenses: {qid} / {condition} ...")
            metrics = base_eval.score_context(question, passages, defense_models)
            row = {
                "query_id": qid, "k": K, "family": FAMILY, "bundle_id": BUNDLE_ID,
                "condition": condition, "question": question,
                "target_wrong_answer": target_wrong_answer, **metrics,
            }
            defense_score_rows.append(row)

    results_fields = [
        "query_id", "k", "question", "target_wrong_answer",
        "retrieved_poison_count", "retrieved_clean_count",
        "canonical_poison_survived_count", "canonical_poison_survived_ranks",
        "mean_poison_retrieval_rank", "retrieval_survival_rate", "all_5_poison_survive",
        "baseline_recomputed_retrieved_poison_count", "baseline_recomputed_retrieved_clean_count",
        "baseline_recomputed_reproduces_archived_topk_exactly",
        "archived_baseline_retrieved_poison_count", "archived_baseline_retrieved_clean_count",
    ]
    write_csv(os.path.join(out_dir, "full_retrieval_results_by_query.csv"), results_fields, results_by_query_rows)

    defense_fields = (
        ["query_id", "k", "family", "bundle_id", "condition", "question", "target_wrong_answer"]
        + ["N_retrieved_poison", "N_retrieved_clean"]
        + list(base_eval._NUMERIC_METRIC_KEYS)
    )
    write_csv(os.path.join(out_dir, "full_retrieval_defense_scores.csv"), defense_fields, defense_score_rows)

    fixed_ctx_rows_by_qid: Dict[str, Dict] = {}
    if os.path.exists(args.fixed_context_scores):
        with open(args.fixed_context_scores, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["family"] == FAMILY and r["query_id"] in SELECTED_QUERY_IDS:
                    fixed_ctx_rows_by_qid[r["query_id"]] = r

    mutated_by_qid = {r["query_id"]: r for r in defense_score_rows if r["condition"] == "mutated"}
    vs_fixed_rows: List[Dict] = []
    compare_keys = [
        "N_retrieved_poison", "N_retrieved_clean",
        "ragdefender_removed_poison", "ragdefender_removed_clean", "ragdefender_top_pair_pp",
        "filterrag_removed_poison", "filterrag_removed_clean",
        "ml_removed_poison_t035", "ml_removed_poison_t04", "ml_removed_poison_t05",
        "ml_mean_poison_probability",
    ]
    for qid in SELECTED_QUERY_IDS:
        fr = fixed_ctx_rows_by_qid.get(qid, {})
        fr_row = mutated_by_qid[qid]
        out_row = {"query_id": qid}
        for key in compare_keys:
            full_val = fr_row.get(key)
            fixed_val = fr.get(key)
            try:
                fixed_val_f = float(fixed_val) if fixed_val not in (None, "") else None
            except ValueError:
                fixed_val_f = None
            out_row[f"full_retrieval_{key}"] = full_val
            out_row[f"fixed_context_{key}"] = fixed_val_f
            out_row[f"delta_{key}"] = (
                (full_val - fixed_val_f) if (full_val is not None and fixed_val_f is not None) else None
            )
        vs_fixed_rows.append(out_row)
    vs_fixed_fields = ["query_id"]
    for key in compare_keys:
        vs_fixed_fields += [f"full_retrieval_{key}", f"fixed_context_{key}", f"delta_{key}"]
    write_csv(os.path.join(out_dir, "full_retrieval_vs_fixed_context.csv"), vs_fixed_fields, vs_fixed_rows)

    report = build_report(
        results_by_query_rows=results_by_query_rows, defense_score_rows=defense_score_rows,
        vs_fixed_rows=vs_fixed_rows, replacement_plan=replacement_plan, selected_queries=selected_queries,
        out_dir=out_dir,
    )
    with open(os.path.join(out_dir, "FULL_RETRIEVAL_PILOT_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[full_retrieval_pilot] wrote outputs to {out_dir}")


# ---------------------------------------------------------------------------
# 5. Report.
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    return base_eval._fmt(v)


def build_report(
    *, results_by_query_rows: Sequence[Dict], defense_score_rows: Sequence[Dict],
    vs_fixed_rows: Sequence[Dict], replacement_plan: Dict[str, Dict[int, ReplacementSlot]],
    selected_queries: Dict[str, Dict], out_dir: str,
) -> str:
    lines: List[str] = []
    lines.append("# Full-Retrieval Pilot -- Normalized `filterrag_targeted` Mutation Bundle 1 (3 queries)")
    lines.append("")
    lines.append(
        "Full-retrieval rerun (real Contriever embedding + dot-product top-k, not fixed-context "
        "reconstruction) of the 3 strongest normalized `filterrag_targeted` mutation cases from "
        "`manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/normalized/filterrag_targeted.normalized.jsonl`, "
        "testing whether the mutated poisoned passages survive retrieval into a fresh top-10 and, if so, "
        "whether they still weaken RAGDefender / FilterRAG (semantic, epsilon=0.2) / ML-FilterRAG-top-k "
        "(t in {0.35, 0.4, 0.5})."
    )
    lines.append("")
    lines.append("## Queries evaluated")
    lines.append("")
    for qid in SELECTED_QUERY_IDS:
        q = selected_queries.get(qid, {})
        lines.append(f"- `{qid}` -- {q.get('question', 'n/a')} (target wrong answer: {q.get('target_wrong_answer', 'n/a')})")
    lines.append("")

    lines.append("## Retrieval survival")
    lines.append("")
    lines.append(
        "| query_id | mutated poison retrieved (of 5) | poison ranks | mean poison rank | "
        "clean retrieved | survival rate | all 5 survive | baseline (recomputed) poison retrieved | "
        "baseline reproduces archived top-10 exactly |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---|---:|---|")
    by_qid = {r["query_id"]: r for r in results_by_query_rows}
    for qid in SELECTED_QUERY_IDS:
        r = by_qid[qid]
        lines.append(
            f"| `{qid}` | {r['canonical_poison_survived_count']} | {r['canonical_poison_survived_ranks'] or '(none)'} | "
            f"{_fmt(r['mean_poison_retrieval_rank'])} | {r['retrieved_clean_count']} | "
            f"{_fmt(r['retrieval_survival_rate'])} | {r['all_5_poison_survive']} | "
            f"{r['baseline_recomputed_retrieved_poison_count']} | "
            f"{r['baseline_recomputed_reproduces_archived_topk_exactly']} |"
        )
    lines.append("")

    lines.append("## Defense outcomes on the freshly-retrieved mutated top-10")
    lines.append("")
    lines.append(
        "| query_id | condition | ragdefender removed_poison | filterrag removed_poison | "
        "ml removed_poison t0.35/0.4/0.5 | ml mean_poison_probability |"
    )
    lines.append("|---|---|---:|---:|---|---:|")
    for r in defense_score_rows:
        lines.append(
            f"| `{r['query_id']}` | {r['condition']} | {r['ragdefender_removed_poison']} | "
            f"{r['filterrag_removed_poison']} | "
            f"{r['ml_removed_poison_t035']}/{r['ml_removed_poison_t04']}/{r['ml_removed_poison_t05']} | "
            f"{_fmt(r['ml_mean_poison_probability'])} |"
        )
    lines.append("")

    lines.append("## Full retrieval vs. fixed-context (normalized bundle 1, `filterrag_targeted`)")
    lines.append("")
    lines.append(
        "| query_id | delta N_retrieved_poison | delta ragdefender removed_poison | "
        "delta filterrag removed_poison | delta ml removed_poison t0.4 | delta ml mean_poison_probability |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    vs_by_qid = {r["query_id"]: r for r in vs_fixed_rows}
    for qid in SELECTED_QUERY_IDS:
        r = vs_by_qid[qid]
        lines.append(
            f"| `{qid}` | {_fmt(r['delta_N_retrieved_poison'])} | {_fmt(r['delta_ragdefender_removed_poison'])} | "
            f"{_fmt(r['delta_filterrag_removed_poison'])} | {_fmt(r['delta_ml_removed_poison_t04'])} | "
            f"{_fmt(r['delta_ml_mean_poison_probability'])} |"
        )
    lines.append("")

    all_survive = all(by_qid[qid]["all_5_poison_survive"] for qid in SELECTED_QUERY_IDS)
    any_survive = any(by_qid[qid]["canonical_poison_survived_count"] > 0 for qid in SELECTED_QUERY_IDS)
    mutated_rows_by_qid = {r["query_id"]: r for r in defense_score_rows if r["condition"] == "mutated"}

    def _defense_robust(defense_key: str) -> bool:
        return all(
            mutated_rows_by_qid[qid][defense_key] == mutated_rows_by_qid[qid]["N_retrieved_poison"]
            for qid in SELECTED_QUERY_IDS
        )

    ragdefender_robust = _defense_robust("ragdefender_removed_poison")
    filterrag_robust = _defense_robust("filterrag_removed_poison")
    ml_t04_robust = _defense_robust("ml_removed_poison_t04")

    best_candidate = min(
        SELECTED_QUERY_IDS,
        key=lambda qid: (
            mutated_rows_by_qid[qid]["ragdefender_removed_poison"]
            + mutated_rows_by_qid[qid]["filterrag_removed_poison"]
            + mutated_rows_by_qid[qid]["ml_removed_poison_t04"],
            mutated_rows_by_qid[qid]["ml_mean_poison_probability"]
            if mutated_rows_by_qid[qid]["ml_mean_poison_probability"] is not None else 1.0,
        ),
    )

    budget_preserved = all(len(replacement_plan[qid]) == 5 for qid in SELECTED_QUERY_IDS)

    lines.append("## Answers")
    lines.append("")
    lines.append(
        f"**1. Did the mutated poisoned passages survive retrieval into top-k?** "
        f"{'Yes, all 3 queries retained all 5 mutated poison passages in the fresh top-10.' if all_survive else ('Partially -- at least one query lost at least one poison passage from the top-10 after mutation.' if any_survive else 'No -- none of the 3 queries retained any mutated poison passage in the fresh top-10.')} "
        + "; ".join(
            f"`{qid}`: {by_qid[qid]['canonical_poison_survived_count']}/5 survived (ranks {by_qid[qid]['canonical_poison_survived_ranks'] or 'none'})"
            for qid in SELECTED_QUERY_IDS
        ) + "."
    )
    lines.append("")
    lines.append(
        "**2. Did the fixed-context failures reproduce after retrieval?** "
        "See the delta table above (full-retrieval minus fixed-context, on the same removed_poison metrics); "
        "a delta of 0 means the fixed-context result reproduced exactly under real retrieval; a non-zero "
        "delta means retrieval changed which/how many poison passages were present relative to the fixed "
        "k=10 context the earlier pilot assumed, so the two are not directly comparable outcome-for-outcome "
        "on that query."
    )
    lines.append("")
    lines.append(
        f"**3. Which defense remained robust after retrieval?** "
        f"RAGDefender: {'robust (removed all retrieved poison on every query)' if ragdefender_robust else 'not fully robust on at least one query'}. "
        f"FilterRAG (semantic, epsilon=0.2): {'robust' if filterrag_robust else 'not fully robust on at least one query'}. "
        f"ML-FilterRAG-top-k (t=0.4): {'robust' if ml_t04_robust else 'not fully robust on at least one query'}."
    )
    lines.append("")
    lines.append(
        f"**4. Which candidate is strongest for paper-level follow-up?** `{best_candidate}` -- "
        "lowest combined removed_poison across the three defenses on the freshly-retrieved mutated "
        f"top-10 (RAGDefender={mutated_rows_by_qid[best_candidate]['ragdefender_removed_poison']}, "
        f"FilterRAG={mutated_rows_by_qid[best_candidate]['filterrag_removed_poison']}, "
        f"ML-FilterRAG t0.4={mutated_rows_by_qid[best_candidate]['ml_removed_poison_t04']}, out of "
        f"{mutated_rows_by_qid[best_candidate]['N_retrieved_poison']} retrieved poison passages; "
        f"mean ML poison probability={_fmt(mutated_rows_by_qid[best_candidate]['ml_mean_poison_probability'])})."
    )
    lines.append("")
    lines.append(
        f"**5. Did replacement preserve the original poison budget?** "
        f"{'Yes' if budget_preserved else 'No'} -- every selected query had exactly 5 poison slots replaced "
        "(never augmented alongside the originals); see "
        "`full_retrieval_candidate_inputs.jsonl` and the automated budget assertions in "
        "`apply_replacements`/`assert_budget_preserved` (also exercised by "
        "`tests/test_run_full_retrieval_pilot_bundle1.py`)."
    )
    lines.append("")
    lines.append(
        "**6. Should the next step be broader replacement reruns, augmentation ablation, or another "
        "mutation round?** "
        + (
            "Broader replacement reruns -- since the mutated poison survived retrieval and at least one "
            "defense was measurably weakened relative to the fixed-context result on at least one query, "
            "this 3-query pilot justifies extending the same real-retrieval replacement methodology to more "
            "of the selected/backup queries and mutation families before considering augmentation ablations "
            "or a further mutation round."
            if (all_survive and not (ragdefender_robust and filterrag_robust and ml_t04_robust))
            else (
                "Another mutation round -- the mutated poison did not reliably survive retrieval and/or every "
                "defense remained robust on the freshly-retrieved context, so a stronger mutation strategy "
                "(rather than broader replacement of the current one) is the better next step."
                if not all_survive
                else "Augmentation ablation -- the mutated poison survived retrieval but every defense remained "
                "fully robust on the freshly-retrieved context, so testing whether augmenting (rather than "
                "replacing) the corpus changes the outcome is a reasonable next diagnostic before investing in "
                "a broader rerun or a new mutation round."
            )
        )
    )
    lines.append("")

    lines.append("## Methodology notes")
    lines.append("")
    lines.append(
        "- Retrieval model: Contriever (`facebook/contriever`), score_function=`dot`, exactly matching "
        "`scripts/build_ml_filterrag_dataset.py`/`scripts/evaluate_ml_filterrag.py`/`main.py`."
    )
    lines.append(
        "- The full 50-query adversarial candidate pool (`results/diagnostics/ml_filterrag_dataset_hotpotqa_50q/"
        "dataset_config.json::target_query_ids`) was rebuilt in its original order via "
        "`Attacker.get_attack()` (offline template substitution, no LLM/GPT call) so that the 3 selected "
        "queries' canonical `adv::LM_targeted::<qid>::<j>` doc_ids resolve to the same global index `j` "
        "they did when `mutation_input_passages.csv` was built; only the 15 (3 queries x 5 slots) global "
        "indices belonging to the 3 selected queries were overwritten with normalized `mutated_text` -- "
        "the other 47 pool queries' own poison text is byte-identical to the original 50-query pool."
    )
    lines.append(
        "- Clean-corpus candidates reused `results/beir_results/hotpotqa-contriever.json`'s existing "
        "precomputed top-10 scores per query (same file the 50-query dataset build reads); their text was "
        "looked up with a single streaming pass over `datasets/hotpotqa/corpus.jsonl` for just those "
        "doc_ids, rather than loading the full ~5.2M-passage corpus via BEIR's `GenericDataLoader`."
    )
    lines.append(
        "- Retrieval was rerun for the 3 selected queries only; `results_by_query`/`defense_scores` never "
        "include or report on any other pool query_id."
    )
    lines.append(
        "- Defense scoring reused `scripts/run_text_mutation_fixed_context_eval.py`'s `score_context()` "
        "(RAGDefender via `defense.dispatch.run_defense`, FilterRAG semantic epsilon=0.2 via "
        "`defense.filterrag.filterrag_defense`, ML-FilterRAG-top-k via `defense.ml_filterrag.extract_features` "
        "+ the existing trained `models/ml_filterrag/hotpotqa_50q_mlfilterrag_topk_rf.joblib` classifier) "
        "completely unmodified; no defense code was edited."
    )
    lines.append(
        "- \"Fixed-context\" comparison values are read from "
        "`mutation_bundle_1/evaluation_normalized/normalized_targeted_family_bundle_scores.csv` (family="
        f"`{FAMILY}`, the same 3 query_ids), not recomputed by this script."
    )
    lines.append("")

    lines.append("## Process confirmation")
    lines.append("")
    lines.append("- No GPT/API calls were made.")
    lines.append("- No `llm.query()` calls were made.")
    lines.append(
        "- Retrieval WAS rerun (real Contriever embedding + dot-product top-k), for the 3 selected "
        "query_ids only."
    )
    lines.append("- No model was trained or retrained (Contriever, SLM, LM, RAGDefender embedder, and the "
                  "ML-FilterRAG classifier were all loaded read-only for inference).")
    lines.append("- No defense code (`defense/*.py`) was modified.")
    lines.append(
        "- The attack budget was preserved: exactly 5 poisoned passages per selected query before and "
        "after mutation; the 50-query adversarial pool was never augmented with both original and "
        "mutated poison for the same query."
    )
    lines.append(f"- Output directory: `{os.path.relpath(out_dir, REPO_ROOT)}`.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"[full_retrieval_pilot] total run time: {time.perf_counter() - t0:.1f}s")
