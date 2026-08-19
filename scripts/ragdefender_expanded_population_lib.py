"""Shared library: prospective HotpotQA k=10 population definition + text
recovery for the RAGDefender population-expansion sequence (STEP 2-6,
Gate-B follow-up). Imported by:

- `scripts/build_ragdefender_expanded_population.py` (STEP 3: freezes the
  query-id list + observed composition BEFORE any `ragdefender_paper`
  run);
- `scripts/run_ragdefender_expanded_baseline.py` (STEP 4: reads the frozen
  list, recovers the SAME texts, runs Stella + Stage 1/2);
- `scripts/run_ragdefender_expanded_gate_c.py` (STEP 5: reuses the
  baseline's saved matrices/composition).

==========================================================================
TEXT-RECOVERY MECHANISM (no new retrieval)
==========================================================================
Uses ONLY existing, already-computed artifacts:

- `results/diagnostics/ml_filterrag_dataset_hotpotqa_50q/dataset_config.json`
  -- the 50-query eligible pool (`target_query_ids`), already frozen for
  an unrelated purpose (ML-FilterRAG train/test split), long before this
  RAGDefender population-expansion task existed.
- `results/diagnostics/ml_filterrag_dataset_hotpotqa_50q/features.csv`
  -- per-(query_id, k, doc_id) retrieved composition (`is_poison` ground
  truth), already computed by the existing retrieval + poisoning
  pipeline. Retrieval is NOT rerun; this file's rows ARE the retrieval
  result.
- `results/query_results/main/hotpotqa-contriever-gpt4-Top5--M10x10-adv-LM_targeted-dot-5-5.json`
  -- gives the exact query-id ORDER used to build the flattened
  adversarial-text pool (`ordered_qids`); see `_pool_index_to_source`
  below for the verified mapping arithmetic.
- `results/adv_targeted_results/hotpotqa.json` -- per-source-query
  `question` + `adv_texts` (5 per query), used to reconstruct poisoned
  passage TEXT for any `pool_index` via
  `question + "." + adv_texts[pool_index % 5]`
  (matches `src/attack.py`'s `LM_targeted` construction; see
  `manual_text_mutation_pilot/hotpotqa_50q_k10/SELECTION_REPORT.md` for
  the prior, independently-verified derivation of this exact formula).
- `datasets/hotpotqa/corpus.jsonl` -- clean passage TEXT lookup by the
  already-retrieved `doc_id` (a plain corpus id for non-poison docs).
  Looked up, never re-retrieved.

This mapping was independently verified against real `features.csv` rows
before being encoded here (see the population-expansion session's
exploration; not re-derivable from this file alone, but the formula below
reproduces it exactly and is covered by
`tests/test_ragdefender_expanded_population.py`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_CONFIG_PATH = REPO_ROOT / "results/diagnostics/ml_filterrag_dataset_hotpotqa_50q/dataset_config.json"
FEATURES_CSV_PATH = REPO_ROOT / "results/diagnostics/ml_filterrag_dataset_hotpotqa_50q/features.csv"
ORDERED_QIDS_SOURCE_PATH = (
    REPO_ROOT / "results/query_results/main/hotpotqa-contriever-gpt4-Top5--M10x10-adv-LM_targeted-dot-5-5.json"
)
ADV_TARGETED_RESULTS_PATH = REPO_ROOT / "results/adv_targeted_results/hotpotqa.json"
CORPUS_JSONL_PATH = REPO_ROOT / "datasets/hotpotqa/corpus.jsonl"

# The 8 queries already used for Gates A/B/C (the pre-existing,
# legacy-cluster-viz-derived diagnostic development sample). Excluded from
# the prospective sample per the explicit instruction -- they remain a
# separate, clearly-labeled exploratory/development population.
GATE_BC_EXCLUDED_QUERY_IDS = frozenset({
    "5adbf0a255429947ff17385a",
    "5a8cb288554299585d9e3726",
    "5ab56e32554299637185c594",
    "5ab29c24554299449642c932",
    "5ae6050f55429929b0807a5e",
    "5ae2070a5542994d89d5b313",
    "5a722b8655429971e9dc9329",
    "5adf37a95542995ec70e8f97",
})

POOL_TEXTS_PER_QUERY = 5  # N=5, fixed by the existing attack configuration.


class PopulationBuildError(RuntimeError):
    """Raised for any inconsistency in the existing-artifact recovery
    pipeline (never for a live-model/network failure -- this module makes
    no such calls)."""


def load_dataset_config() -> dict:
    if not DATASET_CONFIG_PATH.exists():
        raise PopulationBuildError(f"Missing {DATASET_CONFIG_PATH}")
    with open(DATASET_CONFIG_PATH) as f:
        return json.load(f)


def load_eligible_pool() -> List[str]:
    """The 50-query `target_query_ids` pool, minus the 8 Gate-A/B/C
    queries. This is a purely mechanical set-difference over a
    pre-existing, unrelated-purpose split -- no RAGDefender outcome of
    any kind is consulted."""
    cfg = load_dataset_config()
    target_query_ids: List[str] = cfg["target_query_ids"]
    eligible = [qid for qid in target_query_ids if qid not in GATE_BC_EXCLUDED_QUERY_IDS]
    return eligible


def load_ordered_qids() -> List[str]:
    """The exact query-id order used to build the flattened M=100-query
    adversarial-text pool (100 queries x N=5 texts = 500-slot pool;
    `pool_index // 5` indexes into this list). Recovered from the
    `M10x10` retrieval-results JSON's own iteration order (`iter_0`
    .. `iter_9`, each with 10 queries) -- not re-derived or guessed."""
    if not ORDERED_QIDS_SOURCE_PATH.exists():
        raise PopulationBuildError(f"Missing {ORDERED_QIDS_SOURCE_PATH}")
    with open(ORDERED_QIDS_SOURCE_PATH) as f:
        data = json.load(f)
    ordered_qids: List[str] = []
    for item in data:
        for _iter_key, queries in item.items():
            ordered_qids.extend(q["id"] for q in queries)
    return ordered_qids


def load_adv_targeted_results() -> dict:
    if not ADV_TARGETED_RESULTS_PATH.exists():
        raise PopulationBuildError(f"Missing {ADV_TARGETED_RESULTS_PATH}")
    with open(ADV_TARGETED_RESULTS_PATH) as f:
        return json.load(f)


def load_features_df() -> pd.DataFrame:
    if not FEATURES_CSV_PATH.exists():
        raise PopulationBuildError(f"Missing {FEATURES_CSV_PATH}")
    return pd.read_csv(FEATURES_CSV_PATH)


def pool_index_to_source(pool_index: int, ordered_qids: Sequence[str]) -> Tuple[str, int]:
    """`adv::LM_targeted::<eval_qid>::<pool_index>` -> (source_query_id,
    local_index_into_that_query's_5_adv_texts). `eval_qid` (embedded in
    the doc_id) is NOT necessarily the source -- retrieval can surface a
    DIFFERENT query's crafted poison text ("cross-query poison") if it is
    semantically close enough; only `pool_index` determines provenance."""
    source_qid = ordered_qids[pool_index // POOL_TEXTS_PER_QUERY]
    local_idx = pool_index % POOL_TEXTS_PER_QUERY
    return source_qid, local_idx


def reconstruct_poison_text(pool_index: int, ordered_qids: Sequence[str], adv_data: dict) -> str:
    source_qid, local_idx = pool_index_to_source(pool_index, ordered_qids)
    if source_qid not in adv_data:
        raise PopulationBuildError(f"pool_index={pool_index}: source query {source_qid} not in adv_targeted_results")
    entry = adv_data[source_qid]
    question = entry["question"]
    adv_text = entry["adv_texts"][local_idx]
    # Matches src/attack.py's LM_targeted construction: question + "." + adv_text
    # (verified against manual_text_mutation_pilot/hotpotqa_50q_k10/SELECTION_REPORT.md).
    return f"{question}.{adv_text}"


def _parse_poison_doc_id(doc_id: str) -> int:
    """`adv::LM_targeted::<eval_qid>::<pool_index>` -> pool_index (int)."""
    parts = doc_id.split("::")
    if len(parts) != 4 or parts[0] != "adv":
        raise PopulationBuildError(f"Unrecognized poison doc_id format: {doc_id!r}")
    return int(parts[-1])


def collect_needed_clean_doc_ids(features_df: pd.DataFrame, query_ids: Sequence[str], k: int) -> set:
    subset = features_df[(features_df["k"] == k) & (features_df["query_id"].isin(query_ids)) & (~features_df["is_poison"])]
    return set(subset["doc_id"].astype(str).tolist())


def lookup_clean_texts(doc_ids_needed: set) -> Dict[str, str]:
    """Single streaming pass over the (large, ~5.2M-line) local HotpotQA
    corpus file, collecting text ONLY for the requested doc_ids -- a
    lookup by already-retrieved id, not a new retrieval/ranking
    operation."""
    if not doc_ids_needed:
        return {}
    if not CORPUS_JSONL_PATH.exists():
        raise PopulationBuildError(f"Missing {CORPUS_JSONL_PATH}")
    found: Dict[str, str] = {}
    remaining = set(doc_ids_needed)
    with open(CORPUS_JSONL_PATH) as f:
        for line in f:
            if not remaining:
                break
            # Cheap pre-filter before a full json.loads on every one of
            # 5.2M lines: every needed id appears verbatim as the _id
            # field's JSON-quoted value.
            rec = json.loads(line)
            doc_id = str(rec.get("_id"))
            if doc_id in remaining:
                found[doc_id] = rec.get("text", "")
                remaining.discard(doc_id)
    if remaining:
        raise PopulationBuildError(f"Could not find corpus text for doc_ids: {sorted(remaining)}")
    return found


def classify_regime(k: int, m_poison: int, c_clean: int) -> str:
    """STEP 2 regime classification (fidelity audit §"Stage-1 count
    ceiling and poison-fraction stress design"):

    A. BELOW CEILING:               M < floor(k/2)
    B. AT CEILING:                  M == floor(k/2)
    C. ABOVE CEILING / MAJORITY POISON (>=1 clean): M > floor(k/2), C >= 1
    D. ALL POISON:                  C == 0

    D is checked FIRST and is mutually exclusive with A/B/C (never mixed
    -- an all-poison context is classified D regardless of how M compares
    to the ceiling)."""
    if k != m_poison + c_clean:
        raise PopulationBuildError(f"k={k} != M({m_poison}) + C({c_clean})")
    if c_clean == 0:
        return "D_ALL_POISON"
    ceiling = k // 2
    if m_poison < ceiling:
        return "A_BELOW_CEILING"
    if m_poison == ceiling:
        return "B_AT_CEILING"
    return "C_ABOVE_CEILING"


def recover_context_for_query(
    query_id: str,
    k: int,
    features_df: pd.DataFrame,
    ordered_qids: Sequence[str],
    adv_data: dict,
    clean_text_lookup: Dict[str, str],
) -> dict:
    """Recover the full k-passage retrieved context (doc_ids, is_poison,
    texts, in a DETERMINISTIC order sorted by `doc_id` for reproducibility
    -- Stage 1/2 are permutation-invariant over the passage SET, so any
    fixed, deterministic order is valid and does not affect `N_adv` or
    the Stage-2 removal set)."""
    subset = features_df[(features_df["k"] == k) & (features_df["query_id"] == query_id)]
    if len(subset) != k:
        raise PopulationBuildError(f"{query_id}: expected {k} rows at k={k}, found {len(subset)}")
    subset = subset.sort_values("doc_id").reset_index(drop=True)

    doc_ids: List[str] = []
    is_poison: List[bool] = []
    texts: List[str] = []
    for _, row in subset.iterrows():
        doc_id = str(row["doc_id"])
        poison = bool(row["is_poison"])
        if poison:
            pool_index = _parse_poison_doc_id(doc_id)
            text = reconstruct_poison_text(pool_index, ordered_qids, adv_data)
        else:
            if doc_id not in clean_text_lookup:
                raise PopulationBuildError(f"{query_id}: no corpus text recovered for clean doc_id={doc_id}")
            text = clean_text_lookup[doc_id]
        doc_ids.append(doc_id)
        is_poison.append(poison)
        texts.append(text)

    m_poison = sum(is_poison)
    c_clean = k - m_poison
    return {
        "query_id": query_id,
        "k": k,
        "doc_ids": doc_ids,
        "is_poison": is_poison,
        "texts": texts,
        "m_poison": m_poison,
        "c_clean": c_clean,
        "rho": m_poison / k,
        "ceiling": k // 2,
        "regime": classify_regime(k, m_poison, c_clean),
    }


def recover_all_contexts(query_ids: Sequence[str], k: int) -> List[dict]:
    """Batch recovery for many queries at once -- performs exactly ONE
    streaming pass over the corpus file, regardless of population size."""
    features_df = load_features_df()
    ordered_qids = load_ordered_qids()
    adv_data = load_adv_targeted_results()
    needed_clean_doc_ids = collect_needed_clean_doc_ids(features_df, query_ids, k)
    clean_text_lookup = lookup_clean_texts(needed_clean_doc_ids)
    return [
        recover_context_for_query(qid, k, features_df, ordered_qids, adv_data, clean_text_lookup)
        for qid in query_ids
    ]
