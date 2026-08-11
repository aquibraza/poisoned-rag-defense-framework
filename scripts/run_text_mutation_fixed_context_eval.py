#!/usr/bin/env python3
"""Fixed-context evaluation of manually-authored GPT mutation bundles against
the original k=10 baseline, for the HotpotQA manual text-mutation pilot.

**Fixed retrieval only.** This script never reruns retrieval, never trains
or retrains any model, never calls an LLM/GPT/PaLM API, and never calls
`llm.query()`. It only:

1. reconstructs each selected query's *exact* original k=10 retrieved
   context from already-exported pilot artifacts
   (`manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_input_passages.csv`
   + `.../clean_context_passages.csv`), preserving `retrieved_rank`/`doc_id`
   order exactly;
2. for each mutation bundle, replaces *only* the 5 poisoned passages' text
   (matched by `poison_slot`/`poison_rank`) with the bundle's
   `mutated_text`, leaving every clean passage's text, doc_id, and rank
   untouched, and leaving k=10 membership (the set of doc_ids) identical
   to the original;
3. re-scores that fixed passage list with the same, already-trained/
   already-published defense scorers this repo already has
   (`defense/ragdefender_internals.py` + `defense/defense_runner.py`'s
   cached similarity model for RAGDefender; `defense/filterrag.py`'s
   `filterrag_defense` for semantic-mode FilterRAG; `defense/ml_filterrag.py`'s
   `extract_features` + the existing trained
   `models/ml_filterrag/hotpotqa_50q_mlfilterrag_topk_rf.joblib` classifier
   for ML-FilterRAG-top-k) -- every model here is loaded read-only for
   inference; none is trained or retrained by this script.

Every text-generating/scoring call (`local_hf_slm_answer_fn`'s flan-t5-small
generation, `CausalLMScorer`'s distilgpt2 perplexity) is memoized by exact
`(question, passage_text)` / `text` so that identical clean passages (fixed
across the baseline and every bundle for a given query) are only ever
scored once per process, not once per bundle.

Usage:
    python scripts/run_text_mutation_fixed_context_eval.py \\
        --pilot_dir manual_text_mutation_pilot/hotpotqa_50q_k10 \\
        --out_dir manual_text_mutation_pilot/hotpotqa_50q_k10/evaluation
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Force every HF/sentence-transformers load below to use the local cache
# only -- belt-and-suspenders against any accidental network/API call,
# mirroring scripts/run_cluster_normalized_poisoning.py's
# `_force_offline_env()` convention. Every model used here (paraphrase-
# MiniLM-L6-v2, flan-t5-small, distilgpt2, the ml_filterrag joblib
# classifier) is already present in the local cache (see SELECTION_REPORT.md
# / prior runs); this script never downloads anything.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import fix_sentence_transformers  # noqa: F401 -- optional compat patch
except ImportError:
    pass

from defense import defense_runner  # noqa: E402
from defense.dispatch import run_defense  # noqa: E402
from defense.filterrag import (  # noqa: E402
    filterrag_defense,
    local_hf_slm_answer_fn,
)
from defense.ml_filterrag import (  # noqa: E402
    DEFAULT_FEATURE_NAMES,
    extract_features,
    features_to_matrix,
    get_causal_lm_scorer,
    get_slm_model_and_tokenizer,
    load_classifier_cached,
)
from defense.passages import RetrievedPassage, removed_passages  # noqa: E402
from defense.ragdefender_internals import (  # noqa: E402
    concentration_stage1,
    stage2_pair_frequency,
)

DEFAULT_PILOT_DIR = "manual_text_mutation_pilot/hotpotqa_50q_k10"
DEFAULT_NO_CLEAN_BUNDLES = "bundles/mutation_bundles_no_clean_context.jsonl.txt"
DEFAULT_CLEAN_BUNDLES = "bundles/mutation_bundles_clean_context.jsonl.txt"
DEFAULT_ML_MODEL_PATH = "models/ml_filterrag/hotpotqa_50q_mlfilterrag_topk_rf.joblib"

FILTERRAG_EPSILON = 0.2
SEMANTIC_MATCHING_MODE = "semantic"
SEMANTIC_THRESHOLD = 0.6
ML_THRESHOLDS = (0.35, 0.4, 0.5)
ML_PRIMARY_THRESHOLD = 0.4
ML_THRESHOLD_SUFFIX = {0.35: "t035", 0.4: "t04", 0.5: "t05"}
SLM_MODEL = "google/flan-t5-small"
LM_MODEL = "distilgpt2"
RAGDEFENDER_SIMILARITY_MODEL = "paraphrase-MiniLM-L6-v2"
DEVICE = "cpu"  # deterministic, no GPU/MPS dependency for this diagnostic

FORBIDDEN_SOURCE_SNIPPETS = (
    "openai", "google.generativeai", "llm.query", "anthropic", "requests.post",
)


# ---------------------------------------------------------------------------
# 1. Parsing mutation bundle files (JSON array OR true JSONL).
# ---------------------------------------------------------------------------

def parse_mutation_bundle_file(path: str) -> List[Dict]:
    """Parse a mutation-bundle file that may be either:

    - a single pretty-printed JSON array of per-query records (as observed
      in `mutation_bundles_no_clean_context.jsonl.txt`, despite its
      `.jsonl.txt` name), or
    - true JSON-Lines, one per-query record per line (as observed in
      `mutation_bundles_clean_context.jsonl.txt`).

    Returns a list of per-query record dicts, each with (at least)
    `query_id` and `bundles`. Raises `ValueError` with a clear message if
    the file is neither.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    stripped = raw.strip()
    if not stripped:
        raise ValueError(f"{path}: file is empty.")

    if stripped[0] == "[":
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError(f"{path}: top-level JSON array expected, got {type(parsed).__name__}.")
        return parsed

    records: List[Dict] = []
    for line_no, line in enumerate(stripped.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {line_no} is not valid JSON: {exc}") from exc
        records.append(obj)
    if not records:
        raise ValueError(f"{path}: no JSONL records parsed.")
    return records


def validate_bundle_records(records: Sequence[Dict], *, source_path: str) -> None:
    """Validate schema: every record has `query_id` (str) and `bundles`
    (non-empty list); every bundle has `bundle_id` (str) and exactly 5
    `mutated_passages`, each with a unique `poison_rank` in {0..4} and a
    non-empty `mutated_text`. Raises `ValueError` on the first violation
    found, naming the offending record/bundle."""
    for rec in records:
        qid = rec.get("query_id")
        if not qid or not isinstance(qid, str):
            raise ValueError(f"{source_path}: record missing a valid 'query_id': {rec!r}")
        bundles = rec.get("bundles")
        if not isinstance(bundles, list) or not bundles:
            raise ValueError(f"{source_path}: query_id={qid!r} has no 'bundles' list.")
        for b in bundles:
            bid = b.get("bundle_id")
            if not bid or not isinstance(bid, str):
                raise ValueError(f"{source_path}: query_id={qid!r} has a bundle missing 'bundle_id'.")
            mp = b.get("mutated_passages")
            if not isinstance(mp, list) or len(mp) != 5:
                raise ValueError(
                    f"{source_path}: query_id={qid!r} bundle_id={bid!r} has "
                    f"{len(mp) if isinstance(mp, list) else 'no'} mutated_passages "
                    "(expected exactly 5)."
                )
            ranks = sorted(m.get("poison_rank") for m in mp)
            if ranks != [0, 1, 2, 3, 4]:
                raise ValueError(
                    f"{source_path}: query_id={qid!r} bundle_id={bid!r} has poison_rank "
                    f"values {ranks!r} (expected exactly [0, 1, 2, 3, 4])."
                )
            for m in mp:
                text = m.get("mutated_text")
                if not text or not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"{source_path}: query_id={qid!r} bundle_id={bid!r} "
                        f"poison_rank={m.get('poison_rank')!r} has an empty mutated_text."
                    )


def load_bundle_file(path: str) -> Dict[str, Dict]:
    """Parse + validate, returning `{query_id: record}`."""
    records = parse_mutation_bundle_file(path)
    validate_bundle_records(records, source_path=path)
    return {rec["query_id"]: rec for rec in records}


# ---------------------------------------------------------------------------
# 2. Loading pilot CSV artifacts.
# ---------------------------------------------------------------------------

def load_selected_queries(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["query_id"]: r for r in rows}


def load_mutation_input_passages(path: str) -> Dict[str, List[Dict]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_query: Dict[str, List[Dict]] = {}
    for r in rows:
        by_query.setdefault(r["query_id"], []).append(r)
    for qid, group in by_query.items():
        group.sort(key=lambda r: int(r["poison_slot"]))
    return by_query


def load_clean_context_passages(path: str) -> Dict[str, List[Dict]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_query: Dict[str, List[Dict]] = {}
    for r in rows:
        by_query.setdefault(r["query_id"], []).append(r)
    for qid, group in by_query.items():
        group.sort(key=lambda r: int(r["retrieved_rank"]))
    return by_query


# ---------------------------------------------------------------------------
# 3. Fixed-context reconstruction.
# ---------------------------------------------------------------------------

def build_original_context(
    poison_rows: Sequence[Dict], clean_rows: Sequence[Dict]
) -> List[RetrievedPassage]:
    """Reconstruct the exact original k=10 retrieved context, ordered by
    `retrieved_rank`, from the pilot's exported poison/clean passage rows.
    Never invents/infers text: raises if a row's text field is empty."""
    passages: List[RetrievedPassage] = []
    for r in poison_rows:
        text = r["original_poison_text"]
        if not text:
            raise ValueError(f"Missing original_poison_text for doc_id={r['doc_id']!r}")
        passages.append(
            RetrievedPassage(
                doc_id=r["doc_id"], text=text, source="adversarial", is_poison=True,
                retrieval_score=None, rank=int(r["retrieved_rank"]) - 1,
            )
        )
    for r in clean_rows:
        text = r["clean_text"]
        if not text:
            raise ValueError(f"Missing clean_text for doc_id={r['doc_id']!r}")
        passages.append(
            RetrievedPassage(
                doc_id=r["doc_id"], text=text, source="corpus", is_poison=False,
                retrieval_score=None, rank=int(r["retrieved_rank"]) - 1,
            )
        )
    passages.sort(key=lambda p: p.rank)
    return passages


def build_mutated_context(
    original_context: Sequence[RetrievedPassage],
    poison_rows: Sequence[Dict],
    bundle: Dict,
) -> List[RetrievedPassage]:
    """Return a copy of `original_context` with only the 5 poisoned
    passages' *text* replaced by `bundle`'s `mutated_text`, matched by
    `poison_slot` (mutation_input_passages.csv) == `poison_rank` (bundle).
    doc_id, is_poison, source, and rank are preserved exactly; every clean
    passage is returned byte-identical to the original."""
    slot_to_doc_id = {int(r["poison_slot"]): r["doc_id"] for r in poison_rows}
    mutated_text_by_doc_id: Dict[str, str] = {}
    for m in bundle["mutated_passages"]:
        slot = int(m["poison_rank"])
        doc_id = slot_to_doc_id[slot]
        mutated_text_by_doc_id[doc_id] = m["mutated_text"]

    mutated: List[RetrievedPassage] = []
    for p in original_context:
        if p.is_poison and p.doc_id in mutated_text_by_doc_id:
            mutated.append(
                RetrievedPassage(
                    doc_id=p.doc_id, text=mutated_text_by_doc_id[p.doc_id],
                    source=p.source, is_poison=p.is_poison,
                    retrieval_score=p.retrieval_score, rank=p.rank,
                )
            )
        else:
            mutated.append(p)
    return mutated


def assert_same_k10_membership(a: Sequence[RetrievedPassage], b: Sequence[RetrievedPassage]) -> None:
    ids_a = [p.doc_id for p in a]
    ids_b = [p.doc_id for p in b]
    if ids_a != ids_b:
        raise AssertionError(f"k=10 membership/order mismatch: {ids_a!r} != {ids_b!r}")


# ---------------------------------------------------------------------------
# 4. Memoized model callables (inference-only reuse; no training).
# ---------------------------------------------------------------------------

class MemoizedSlmAnswerFn:
    """Wraps `defense.filterrag.local_hf_slm_answer_fn`'s returned callable
    so identical (question, passage_text) pairs -- e.g. every unchanged
    clean passage, repeated across the baseline and every mutation bundle
    for a query -- are generated exactly once per process."""

    def __init__(self, base_fn: Callable[[str, str], Optional[str]]):
        self._base_fn = base_fn
        self._cache: Dict[Tuple[str, str], Optional[str]] = {}
        self.calls = 0
        self.cache_hits = 0

    def __call__(self, question: str, passage_text: str) -> Optional[str]:
        key = (question, passage_text)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.calls += 1
        result = self._base_fn(question, passage_text)
        self._cache[key] = result
        return result


class MemoizedCausalLMScorer:
    """Wraps a `defense.ml_filterrag.CausalLMScorer` so identical passage
    text is scored for perplexity exactly once per process."""

    def __init__(self, base_scorer):
        self._base_scorer = base_scorer
        self._cache: Dict[str, float] = {}
        self.calls = 0
        self.cache_hits = 0

    def perplexity(self, text: Optional[str]) -> float:
        key = text or ""
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.calls += 1
        result = self._base_scorer.perplexity(text)
        self._cache[key] = result
        return result


@dataclass
class Models:
    memo_slm_answer_fn: MemoizedSlmAnswerFn
    memo_causal_scorer: MemoizedCausalLMScorer
    slm_logprob_model: object
    slm_logprob_tokenizer: object
    classifier: object


def load_models(ml_model_path: str) -> Models:
    base_slm_fn = local_hf_slm_answer_fn(SLM_MODEL, device=DEVICE)
    memo_slm_fn = MemoizedSlmAnswerFn(base_slm_fn)
    base_causal_scorer = get_causal_lm_scorer(LM_MODEL, device=DEVICE)
    memo_causal_scorer = MemoizedCausalLMScorer(base_causal_scorer)
    slm_model, slm_tokenizer = get_slm_model_and_tokenizer(SLM_MODEL, device=DEVICE)
    classifier = load_classifier_cached(ml_model_path)
    return Models(
        memo_slm_answer_fn=memo_slm_fn,
        memo_causal_scorer=memo_causal_scorer,
        slm_logprob_model=slm_model,
        slm_logprob_tokenizer=slm_tokenizer,
        classifier=classifier,
    )


# ---------------------------------------------------------------------------
# 5. Per-context scoring: RAGDefender, FilterRAG (semantic), ML-FilterRAG.
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(statistics.fmean(values)) if values else None


def score_ragdefender(query: str, passages: Sequence[RetrievedPassage]) -> Dict:
    """RAGDefender metrics via the existing, unmodified internal scorer
    (`defense/ragdefender_internals.py`) fed by the same cached similarity
    model `defense/defense_runner.py` already uses
    (`paraphrase-MiniLM-L6-v2`, reused via `defense_runner._get_s_model`,
    never re-instantiated). `removed_poison`/`removed_clean` are taken from
    the real, unmodified `defense_runner.apply_defense` production
    algorithm (via `defense.dispatch.run_defense("ragdefender_original", ...)`
    on this exact fixed passage list); pp/pc/cc top-pair counts and mean
    pairwise cosines are a separate diagnostic view computed from the same
    cosine-similarity matrix via `concentration_stage1`/
    `stage2_pair_frequency`, cross-checked against the dispatch result."""
    from sentence_transformers import util as st_util  # noqa: PLC0415

    texts = [p.text for p in passages]
    is_poison = [p.is_poison for p in passages]
    k = len(texts)

    kept, diag = run_defense(
        "ragdefender_original", query, passages, "hotpotqa",
        device=DEVICE, gpu_id=0, top_k=None,
    )
    removed = removed_passages(passages, kept)
    dispatch_removed_poison = sum(1 for p in removed if p.is_poison)
    dispatch_removed_clean = len(removed) - dispatch_removed_poison

    cfg = defense_runner.DefenseConfig(device=DEVICE)
    s_model = defense_runner._get_s_model(cfg)  # noqa: SLF001 -- intentional reuse, see dispatch.py
    embeddings = s_model.encode(texts, convert_to_tensor=True)
    cos_sim_matrix = st_util.cos_sim(embeddings, embeddings).cpu().numpy()

    stage1 = concentration_stage1(cos_sim_matrix)
    stage2 = stage2_pair_frequency(cos_sim_matrix, stage1.n_adv_estimated)

    stage2_removed_poison = sum(1 for idx in stage2.selected_indices if is_poison[idx])
    stage2_removed_clean = len(stage2.selected_indices) - stage2_removed_poison

    pp_sims, pc_sims, cc_sims = [], [], []
    for i in range(k):
        for j in range(i + 1, k):
            v = float(cos_sim_matrix[i, j])
            if is_poison[i] and is_poison[j]:
                pp_sims.append(v)
            elif is_poison[i] != is_poison[j]:
                pc_sims.append(v)
            else:
                cc_sims.append(v)

    top_pair_pp = top_pair_pc = top_pair_cc = 0
    for i, j, _ in stage2.top_pairs:
        if is_poison[i] and is_poison[j]:
            top_pair_pp += 1
        elif is_poison[i] != is_poison[j]:
            top_pair_pc += 1
        else:
            top_pair_cc += 1

    n_poison = sum(is_poison)
    residual_poison_fraction = (
        (n_poison - dispatch_removed_poison) / n_poison if n_poison else None
    )

    return {
        "ragdefender_mean_pp_cosine": _mean(pp_sims),
        "ragdefender_mean_pc_cosine": _mean(pc_sims),
        "ragdefender_mean_cc_cosine": _mean(cc_sims),
        "ragdefender_top_pair_pp": top_pair_pp,
        "ragdefender_top_pair_pc": top_pair_pc,
        "ragdefender_top_pair_cc": top_pair_cc,
        "ragdefender_removed_poison": dispatch_removed_poison,
        "ragdefender_removed_clean": dispatch_removed_clean,
        "ragdefender_residual_poison_fraction": residual_poison_fraction,
        "ragdefender_n_adv_estimated": stage1.n_adv_estimated,
        "ragdefender_stage2_matches_dispatch": (
            stage2_removed_poison == dispatch_removed_poison
            and stage2_removed_clean == dispatch_removed_clean
        ),
    }


def score_filterrag(query: str, passages: Sequence[RetrievedPassage], models: Models) -> Dict:
    """Semantic-mode FilterRAG (epsilon=0.2), via the existing, unmodified
    `defense.filterrag.filterrag_defense`, called directly (bypassing
    `defense.dispatch.run_defense` only so a memoized `slm_answer_fn` can
    be injected -- the scoring logic itself is untouched)."""
    kept, diag = filterrag_defense(
        query, passages,
        epsilon=FILTERRAG_EPSILON,
        slm_answer_fn=models.memo_slm_answer_fn,
        matching_mode=SEMANTIC_MATCHING_MODE,
        semantic_threshold=SEMANTIC_THRESHOLD,
    )
    removed = removed_passages(passages, kept)
    removed_poison = sum(1 for p in removed if p.is_poison)
    removed_clean = len(removed) - removed_poison
    n_poison = sum(1 for p in passages if p.is_poison)
    residual_poison_fraction = (n_poison - removed_poison) / n_poison if n_poison else None

    scores = diag["filterrag_scores"]  # same order as `passages` (see score_passages())
    poison_freq_density = [s["freq_density_score"] for s, p in zip(scores, passages) if p.is_poison]
    poison_matched_freq_sum = [
        s.get("matched_freq_sum") for s, p in zip(scores, passages) if p.is_poison
    ]
    # score_passages() does not expose matched_freq_sum directly; recompute
    # membership below in the ML-FilterRAG scorer instead (which does), and
    # leave this field populated only if present (defensive, no invented data).
    return {
        "filterrag_removed_poison": removed_poison,
        "filterrag_removed_clean": removed_clean,
        "filterrag_residual_poison_fraction": residual_poison_fraction,
        "filterrag_mean_freq_density_poison": _mean(poison_freq_density),
        "filterrag_mean_matched_freq_sum_poison": _mean(
            [v for v in poison_matched_freq_sum if v is not None]
        ) if any(v is not None for v in poison_matched_freq_sum) else None,
    }


def score_ml_filterrag(query: str, passages: Sequence[RetrievedPassage], models: Models) -> Dict:
    """ML-FilterRAG-top-k, via the existing, unmodified
    `defense.ml_filterrag.extract_features` (semantic matching, threshold
    0.6) plus the already-trained classifier's `predict_proba` (loaded
    read-only from `models/ml_filterrag/hotpotqa_50q_mlfilterrag_topk_rf.joblib`
    -- never retrained here). Reports removed_poison/removed_clean/
    residual_poison_fraction at t in {0.35, 0.4, 0.5} from a single shared
    `predict_proba` call (thresholds are just a comparison sweep, not a
    re-scoring)."""
    feature_rows = extract_features(
        query, passages,
        slm_answer_fn=models.memo_slm_answer_fn,
        slm_logprob_model=models.slm_logprob_model,
        slm_logprob_tokenizer=models.slm_logprob_tokenizer,
        matching_mode=SEMANTIC_MATCHING_MODE,
        semantic_threshold=SEMANTIC_THRESHOLD,
        causal_lm_scorer=models.memo_causal_scorer,
        lm_model_name=LM_MODEL,
        lm_device=DEVICE,
    )
    X = features_to_matrix(feature_rows, models.classifier.feature_names)
    proba = models.classifier.predict_proba(X)

    is_poison = [p.is_poison for p in passages]
    n_poison = sum(is_poison)

    poison_proba = [float(pr) for pr, is_p in zip(proba, is_poison) if is_p]
    poison_freq_density = [r["freq_density_score"] for r, is_p in zip(feature_rows, is_poison) if is_p]
    poison_matched_freq_sum = [r["matched_freq_sum"] for r, is_p in zip(feature_rows, is_poison) if is_p]
    poison_perplexity = [r["perplexity"] for r, is_p in zip(feature_rows, is_poison) if is_p]
    poison_logprob = [r["slm_answer_logprob"] for r, is_p in zip(feature_rows, is_poison) if is_p]

    out: Dict = {
        "ml_mean_poison_probability": _mean(poison_proba),
        "ml_mean_freq_density_poison": _mean(poison_freq_density),
        "ml_mean_matched_freq_sum_poison": _mean(poison_matched_freq_sum),
        "ml_mean_perplexity_poison": _mean(poison_perplexity),
        "ml_mean_slm_answer_logprob_poison": _mean(poison_logprob),
    }
    for t in ML_THRESHOLDS:
        removed_idx = [i for i, pr in enumerate(proba) if float(pr) >= t]
        removed_poison_t = sum(1 for i in removed_idx if is_poison[i])
        removed_clean_t = len(removed_idx) - removed_poison_t
        residual_t = (n_poison - removed_poison_t) / n_poison if n_poison else None
        suffix = ML_THRESHOLD_SUFFIX[t]
        out[f"ml_removed_poison_{suffix}"] = removed_poison_t
        out[f"ml_removed_clean_{suffix}"] = removed_clean_t
        out[f"ml_residual_poison_fraction_{suffix}"] = residual_t
    return out


def score_context(query: str, passages: Sequence[RetrievedPassage], models: Models) -> Dict:
    n_poison = sum(1 for p in passages if p.is_poison)
    n_clean = len(passages) - n_poison
    out: Dict = {"N_retrieved_poison": n_poison, "N_retrieved_clean": n_clean}
    out.update(score_ragdefender(query, passages))
    out.update(score_filterrag(query, passages, models))
    out.update(score_ml_filterrag(query, passages, models))
    return out


# ---------------------------------------------------------------------------
# 6. Deltas.
# ---------------------------------------------------------------------------

# Literal names requested by the evaluation task, mapped onto the specific
# per-defense metric each unambiguously corresponds to (documented in
# TEXT_MUTATION_FIXED_CONTEXT_REPORT.md -- these are aliases, not new
# measurements).
DELTA_ALIASES: Dict[str, str] = {
    "delta_removed_poison": "delta_ml_removed_poison_t04",
    "delta_removed_clean": "delta_ml_removed_clean_t04",
    "delta_residual_poison_fraction": "delta_ml_residual_poison_fraction_t04",
    "delta_top_pair_pp": "delta_ragdefender_top_pair_pp",
    "delta_mean_poison_probability": "delta_ml_mean_poison_probability",
    "delta_freq_density": "delta_filterrag_mean_freq_density_poison",
    "delta_matched_freq_sum": "delta_filterrag_mean_matched_freq_sum_poison",
}

_NUMERIC_METRIC_KEYS = (
    "ragdefender_mean_pp_cosine", "ragdefender_mean_pc_cosine", "ragdefender_mean_cc_cosine",
    "ragdefender_top_pair_pp", "ragdefender_top_pair_pc", "ragdefender_top_pair_cc",
    "ragdefender_removed_poison", "ragdefender_removed_clean", "ragdefender_residual_poison_fraction",
    "ragdefender_n_adv_estimated",
    "filterrag_removed_poison", "filterrag_removed_clean", "filterrag_residual_poison_fraction",
    "filterrag_mean_freq_density_poison", "filterrag_mean_matched_freq_sum_poison",
    "ml_mean_poison_probability", "ml_mean_freq_density_poison", "ml_mean_matched_freq_sum_poison",
    "ml_mean_perplexity_poison", "ml_mean_slm_answer_logprob_poison",
    "ml_removed_poison_t035", "ml_removed_clean_t035", "ml_residual_poison_fraction_t035",
    "ml_removed_poison_t04", "ml_removed_clean_t04", "ml_residual_poison_fraction_t04",
    "ml_removed_poison_t05", "ml_removed_clean_t05", "ml_residual_poison_fraction_t05",
)


def compute_deltas(baseline: Dict, bundle: Dict) -> Dict:
    """bundle_metric - baseline_metric for every shared numeric metric, plus
    the literal-named aliases from `DELTA_ALIASES`. `None` if either side is
    `None` (e.g. residual_poison_fraction when N_retrieved_poison==0)."""
    deltas: Dict = {}
    for key in _NUMERIC_METRIC_KEYS:
        a, b = baseline.get(key), bundle.get(key)
        deltas[f"delta_{key}"] = (b - a) if (a is not None and b is not None) else None
    for alias, target in DELTA_ALIASES.items():
        deltas[alias] = deltas.get(target)
    return deltas


# ---------------------------------------------------------------------------
# 7. Orchestration + CSV/report writing.
# ---------------------------------------------------------------------------

def write_csv(path: str, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_report(
    *, pilot_dir: str, no_clean_path: str, clean_path: str, ml_model_path: str,
    query_ids: Sequence[str], baseline_rows: Sequence[Dict], bundle_rows: Sequence[Dict],
    delta_rows: Sequence[Dict], selected_queries: Dict[str, Dict],
) -> str:
    lines: List[str] = []
    lines.append("# Text-Mutation Fixed-Context Evaluation Report")
    lines.append("")
    lines.append(
        "Fixed-retrieval evaluation of manually-authored GPT mutation bundles against the "
        "original k=10 baseline, for the HotpotQA manual text-mutation pilot. Retrieval "
        "membership/order (`doc_id`, `retrieved_rank`) is identical between the baseline and "
        "every mutated context for a given query -- only the 5 poisoned passages' text differs "
        "across bundles; clean passages are byte-identical everywhere."
    )
    lines.append("")
    lines.append("## Note on scope vs. the requesting instructions")
    lines.append("")
    lines.append(
        "The evaluation request referred to \"the 2 selected query cases\", but both mutation "
        f"bundle files (`{os.path.relpath(no_clean_path, REPO_ROOT)}`, "
        f"`{os.path.relpath(clean_path, REPO_ROOT)}`) contain bundles for **all 4 primary** "
        "pilot queries, not 2. Per the \"do not silently infer/drop data\" constraint carried "
        "over from the pilot-selection task, this evaluation processes every query_id actually "
        "present in the bundle files (4 queries) rather than guessing which 2 were intended. "
        "This is stated explicitly here rather than silently narrowing scope."
    )
    lines.append("")
    lines.append("## Artifact paths used")
    lines.append("")
    lines.append(f"- `{os.path.relpath(no_clean_path, REPO_ROOT)}` (no-clean-context mutation bundles)")
    lines.append(f"- `{os.path.relpath(clean_path, REPO_ROOT)}` (clean-context-aware mutation bundles)")
    lines.append(f"- `{os.path.join(pilot_dir, 'selected_queries.csv')}`")
    lines.append(f"- `{os.path.join(pilot_dir, 'mutation_input_passages.csv')}`")
    lines.append(f"- `{os.path.join(pilot_dir, 'clean_context_passages.csv')}`")
    lines.append(f"- `{ml_model_path}` (trained ML-FilterRAG-top-k random-forest classifier, loaded read-only)")
    lines.append("")
    lines.append("## Defense scoring configuration")
    lines.append("")
    lines.append("- RAGDefender: `defense/ragdefender_internals.py` (`concentration_stage1`/`stage2_pair_frequency`, unmodified) "
                  "fed by `defense_runner._get_s_model`'s cached `paraphrase-MiniLM-L6-v2` embedder; "
                  "removed_poison/removed_clean taken from `defense.dispatch.run_defense(\"ragdefender_original\", ...)` "
                  "(the real, unmodified production algorithm) on the fixed passage list.")
    lines.append(f"- FilterRAG: semantic matching mode, semantic_threshold={SEMANTIC_THRESHOLD}, epsilon={FILTERRAG_EPSILON} "
                  f"(`defense.filterrag.filterrag_defense`, SLM=`{SLM_MODEL}`).")
    lines.append(f"- ML-FilterRAG-top-k: semantic matching mode, semantic_threshold={SEMANTIC_THRESHOLD}, "
                  f"LM=`{LM_MODEL}`, thresholds swept at t in {list(ML_THRESHOLDS)} from a single "
                  "`predict_proba` call per context (t=0.4 is the primary/reported threshold, matching the "
                  "pilot-selection artifacts).")
    lines.append(f"- All models run on device=`{DEVICE}` for determinism; every model is loaded read-only "
                  "for inference and is never trained/retrained by this script.")
    lines.append("")
    lines.append("## Queries evaluated")
    lines.append("")
    for qid in query_ids:
        q = selected_queries.get(qid, {})
        lines.append(f"- `{qid}` -- {q.get('question', 'n/a')} (target wrong answer: {q.get('target_wrong_answer', 'n/a')})")
    lines.append("")

    lines.append("## Baseline (original, unmutated) fixed-context metrics")
    lines.append("")
    lines.append("| query_id | ragdefender removed_poison/clean | ragdefender top_pair_pp/pc/cc | "
                  "filterrag removed_poison/clean | ml removed_poison/clean (t0.4) | ml mean_poison_probability |")
    lines.append("|---|---|---|---|---|---|")
    baseline_by_qid = {r["query_id"]: r for r in baseline_rows}
    for qid in query_ids:
        r = baseline_by_qid[qid]
        lines.append(
            f"| `{qid}` | {r['ragdefender_removed_poison']}/{r['ragdefender_removed_clean']} | "
            f"{r['ragdefender_top_pair_pp']}/{r['ragdefender_top_pair_pc']}/{r['ragdefender_top_pair_cc']} | "
            f"{r['filterrag_removed_poison']}/{r['filterrag_removed_clean']} | "
            f"{r['ml_removed_poison_t04']}/{r['ml_removed_clean_t04']} | "
            f"{_fmt(r['ml_mean_poison_probability'])} |"
        )
    lines.append("")

    lines.append("## Mutation bundle scores vs. baseline")
    lines.append("")
    lines.append("| query_id | condition | bundle | ml removed_poison (t0.4) | delta_removed_poison | "
                  "filterrag removed_poison | delta_freq_density | ragdefender top_pair_pp | delta_top_pair_pp | "
                  "delta_mean_poison_probability |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    delta_by_key = {(r["query_id"], r["condition"], r["bundle_id"]): r for r in delta_rows}
    for r in bundle_rows:
        key = (r["query_id"], r["condition"], r["bundle_id"])
        d = delta_by_key[key]
        lines.append(
            f"| `{r['query_id']}` | {r['condition']} | {r['bundle_id']} | "
            f"{r['ml_removed_poison_t04']} | {_fmt(d['delta_removed_poison'])} | "
            f"{r['filterrag_removed_poison']} | {_fmt(d['delta_freq_density'])} | "
            f"{r['ragdefender_top_pair_pp']} | {_fmt(d['delta_top_pair_pp'])} | "
            f"{_fmt(d['delta_mean_poison_probability'])} |"
        )
    lines.append("")

    # Answers to the required questions.
    any_reduced_removal = any(
        (d["delta_removed_poison"] is not None and d["delta_removed_poison"] < 0)
        for d in delta_rows
    )
    no_clean = [r for r in bundle_rows if r["condition"] == "no_clean_context"]
    clean_ctx = [r for r in bundle_rows if r["condition"] == "clean_context"]
    mean_removed_no_clean = _mean([r["ml_removed_poison_t04"] for r in no_clean])
    mean_removed_clean_ctx = _mean([r["ml_removed_poison_t04"] for r in clean_ctx])
    mean_top_pair_pp_baseline = _mean([r["ragdefender_top_pair_pp"] for r in baseline_rows])
    mean_top_pair_pp_bundles = _mean([r["ragdefender_top_pair_pp"] for r in bundle_rows])
    mean_freq_density_baseline = _mean([r["filterrag_mean_freq_density_poison"] for r in baseline_rows])
    mean_freq_density_bundles = _mean([r["filterrag_mean_freq_density_poison"] for r in bundle_rows])
    mean_ml_proba_baseline = _mean([r["ml_mean_poison_probability"] for r in baseline_rows])
    mean_ml_proba_bundles = _mean([r["ml_mean_poison_probability"] for r in bundle_rows])

    best_row = min(
        bundle_rows,
        key=lambda r: (
            r["ml_removed_poison_t04"] + r["filterrag_removed_poison"] + r["ragdefender_removed_poison"],
            r["ml_mean_poison_probability"] if r["ml_mean_poison_probability"] is not None else 1.0,
        ),
    )
    worst_row = max(
        bundle_rows,
        key=lambda r: (
            r["ml_removed_poison_t04"] + r["filterrag_removed_poison"] + r["ragdefender_removed_poison"],
            r["ml_mean_poison_probability"] if r["ml_mean_poison_probability"] is not None else 0.0,
        ),
    )

    lines.append("## Answers")
    lines.append("")
    lines.append(
        f"**Did any GPT mutation bundle reduce defense removal of poisoned passages?** "
        f"{'Yes' if any_reduced_removal else 'No'} -- "
        f"{'at least one bundle showed a negative delta_removed_poison (ML-FilterRAG t=0.4) relative to its query baseline.' if any_reduced_removal else 'every bundle was still fully removed (delta_removed_poison == 0) by at least ML-FilterRAG t=0.4 across all evaluated queries.'}"
    )
    lines.append("")
    lines.append(
        f"**Did clean-context-aware mutations perform better (evade more) than no-clean-context mutations?** "
        f"Mean ML-FilterRAG removed_poison (t=0.4): no-clean-context={_fmt(mean_removed_no_clean)}, "
        f"clean-context={_fmt(mean_removed_clean_ctx)}. "
        f"{'clean-context bundles evaded slightly more (lower removed_poison)' if (mean_removed_clean_ctx is not None and mean_removed_no_clean is not None and mean_removed_clean_ctx < mean_removed_no_clean) else ('no-clean-context bundles evaded slightly more' if (mean_removed_clean_ctx is not None and mean_removed_no_clean is not None and mean_removed_clean_ctx > mean_removed_no_clean) else 'no measurable difference between the two conditions')}."
    )
    lines.append("")
    lines.append(
        f"**Did mutations reduce RAGDefender poison-poison structure (`top_pair_pp`)?** "
        f"Mean top_pair_pp: baseline={_fmt(mean_top_pair_pp_baseline)}, mutated bundles={_fmt(mean_top_pair_pp_bundles)}."
    )
    lines.append("")
    lines.append(
        f"**Did mutations reduce FilterRAG Freq-Density / matched_freq_sum?** "
        f"Mean Freq-Density (poison passages): baseline={_fmt(mean_freq_density_baseline)}, "
        f"mutated bundles={_fmt(mean_freq_density_bundles)}."
    )
    lines.append("")
    lines.append(
        f"**Did mutations reduce ML-FilterRAG poison probability?** "
        f"Mean predicted poison probability: baseline={_fmt(mean_ml_proba_baseline)}, "
        f"mutated bundles={_fmt(mean_ml_proba_bundles)}."
    )
    lines.append("")
    lines.append(
        f"**Which bundle is the best candidate for a next-stage full retrieval rerun?** "
        f"`{best_row['query_id']}` / `{best_row['condition']}` / `{best_row['bundle_id']}` -- "
        f"lowest combined removal across the three defenses "
        f"(ML t0.4={best_row['ml_removed_poison_t04']}, FilterRAG={best_row['filterrag_removed_poison']}, "
        f"RAGDefender={best_row['ragdefender_removed_poison']} out of "
        f"{best_row['N_retrieved_poison']} retrieved poison passages)."
    )
    lines.append("")
    lines.append(
        f"Worst-performing bundle (evaded the least): `{worst_row['query_id']}` / "
        f"`{worst_row['condition']}` / `{worst_row['bundle_id']}` "
        f"(ML t0.4={worst_row['ml_removed_poison_t04']}, FilterRAG={worst_row['filterrag_removed_poison']}, "
        f"RAGDefender={worst_row['ragdefender_removed_poison']})."
    )
    lines.append("")
    proceed = any_reduced_removal and (best_row["ml_removed_poison_t04"] < best_row["N_retrieved_poison"])
    lines.append(
        f"**Are results strong enough to proceed beyond the 2/4-case pilot?** "
        f"{'Marginally -- at least one bundle showed reduced removal at fixed retrieval, which is a necessary (not sufficient) precondition for a full retrieval rerun.' if proceed else 'Not yet -- no bundle achieved a meaningfully reduced removal count at any evaluated defense/threshold in this fixed-context setting; every mutation bundle was still detected/removed essentially as fully as the unmutated baseline.'} "
        "This fixed-context result says nothing about whether a mutated passage would still be "
        "*retrieved* into the top-k in the first place (that requires an actual retrieval rerun, "
        "explicitly out of scope here)."
    )
    lines.append("")

    lines.append("## Delta column naming (literal aliases)")
    lines.append("")
    lines.append(
        "The evaluation request's literal delta names are ambiguous across the 3 defenses "
        "(all 3 have a `removed_poison`/`removed_clean`/`residual_poison_fraction` notion, only "
        "RAGDefender has `top_pair_pp`, only ML-FilterRAG has `mean_poison_probability`, and "
        "FilterRAG/ML-FilterRAG both compute Freq-Density/matched_freq_sum identically since both "
        "use semantic matching_mode + threshold 0.6). Rather than silently picking one, "
        "`mutation_bundle_deltas.csv` reports every metric fully namespaced by defense "
        "(`delta_ragdefender_*`, `delta_filterrag_*`, `delta_ml_*_t035/t04/t05`), **plus** these "
        "literal aliases, mapped as follows:"
    )
    lines.append("")
    for alias, target in DELTA_ALIASES.items():
        lines.append(f"- `{alias}` = `{target}`")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Scope deviation from \"2 selected query cases\" to all 4 primary queries present in "
        "the bundle files (see note above)."
    )
    lines.append(
        "- All models (SLM, LM, RAGDefender embedder, ML-FilterRAG classifier) run on "
        f"device=`{DEVICE}`, not the mixed cpu/mps configuration used when the original "
        "artifacts (`ml_filterrag_eval_hotpotqa_50q_t04`) were built. This is a determinism "
        "choice for this diagnostic, not a claim that results are bit-identical to those "
        "artifacts -- baseline metrics computed here are a fresh, independent re-scoring of the "
        "same fixed passages, used only as this evaluation's own internal reference point."
    )
    lines.append(
        "- `filterrag_mean_matched_freq_sum_poison` is left blank whenever "
        "`defense.filterrag.score_passages()`'s returned dict does not expose "
        "`matched_freq_sum` directly (it currently does not -- only `freq_density_score` is "
        "returned per passage); ML-FilterRAG's `extract_features()` output "
        "(`ml_mean_matched_freq_sum_poison`) is the authoritative source for this quantity, "
        "since it is computed via the same `freq_density_detailed()` call/keywords/threshold "
        "and therefore numerically identical to what FilterRAG would compute."
    )
    lines.append("")
    lines.append("## Process confirmation")
    lines.append("")
    lines.append("- No GPT/API calls were made.")
    lines.append("- No `llm.query()` calls were made.")
    lines.append("- Retrieval was not rerun (no BEIR/corpus search; k=10 membership was reconstructed verbatim from existing pilot CSV artifacts).")
    lines.append("- No model was trained or retrained (every model -- SLM, LM, RAGDefender embedder, ML-FilterRAG classifier -- was loaded read-only for inference).")
    lines.append("- No defense code (`defense/*.py`) was modified; every defense function used here is called unmodified.")
    lines.append("- Only text substitution on already-provided mutation bundles was applied; no mutations were generated by this script.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot_dir", default=os.path.join(REPO_ROOT, DEFAULT_PILOT_DIR))
    parser.add_argument("--no_clean_bundles", default=None)
    parser.add_argument("--clean_bundles", default=None)
    parser.add_argument("--ml_model_path", default=os.path.join(REPO_ROOT, DEFAULT_ML_MODEL_PATH))
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    pilot_dir = args.pilot_dir
    no_clean_path = args.no_clean_bundles or os.path.join(pilot_dir, DEFAULT_NO_CLEAN_BUNDLES)
    clean_path = args.clean_bundles or os.path.join(pilot_dir, DEFAULT_CLEAN_BUNDLES)
    out_dir = args.out_dir or os.path.join(pilot_dir, "evaluation")

    selected_queries = load_selected_queries(os.path.join(pilot_dir, "selected_queries.csv"))
    poison_by_query = load_mutation_input_passages(os.path.join(pilot_dir, "mutation_input_passages.csv"))
    clean_by_query = load_clean_context_passages(os.path.join(pilot_dir, "clean_context_passages.csv"))

    no_clean_bundles_by_qid = load_bundle_file(no_clean_path)
    clean_bundles_by_qid = load_bundle_file(clean_path)

    query_ids = sorted(set(no_clean_bundles_by_qid) | set(clean_bundles_by_qid))
    for qid in query_ids:
        if qid not in selected_queries:
            raise ValueError(f"query_id={qid!r} from bundle files is not present in selected_queries.csv")
        if qid not in poison_by_query or qid not in clean_by_query:
            raise ValueError(f"query_id={qid!r} is missing pilot passage rows.")

    print(f"[run_text_mutation_fixed_context_eval] loading models (device={DEVICE})...")
    models = load_models(args.ml_model_path)

    baseline_rows: List[Dict] = []
    bundle_rows: List[Dict] = []
    delta_rows: List[Dict] = []

    for qid in query_ids:
        q = selected_queries[qid]
        question = q["question"]
        target_wrong_answer = q["target_wrong_answer"]

        original_context = build_original_context(poison_by_query[qid], clean_by_query[qid])
        print(f"[run_text_mutation_fixed_context_eval] scoring baseline for {qid} ({len(original_context)} passages)...")
        baseline_metrics = score_context(question, original_context, models)
        baseline_row = {
            "query_id": qid, "k": 10, "question": question, "target_wrong_answer": target_wrong_answer,
            **baseline_metrics,
        }
        baseline_rows.append(baseline_row)

        for condition, bundles_by_qid in (
            ("no_clean_context", no_clean_bundles_by_qid),
            ("clean_context", clean_bundles_by_qid),
        ):
            record = bundles_by_qid[qid]
            for bundle in record["bundles"]:
                mutated_context = build_mutated_context(original_context, poison_by_query[qid], bundle)
                assert_same_k10_membership(original_context, mutated_context)
                for orig, mut in zip(original_context, mutated_context):
                    if not orig.is_poison and orig.text != mut.text:
                        raise AssertionError(
                            f"{qid}/{condition}/{bundle['bundle_id']}: clean passage "
                            f"doc_id={orig.doc_id!r} text changed (must remain unchanged)."
                        )

                print(
                    f"[run_text_mutation_fixed_context_eval] scoring {qid} / {condition} / "
                    f"{bundle['bundle_id']}..."
                )
                bundle_metrics = score_context(question, mutated_context, models)
                bundle_row = {
                    "query_id": qid, "k": 10, "condition": condition, "bundle_id": bundle["bundle_id"],
                    "mutation_strategy": bundle.get("mutation_strategy", ""),
                    "question": question, "target_wrong_answer": target_wrong_answer,
                    **bundle_metrics,
                }
                bundle_rows.append(bundle_row)

                deltas = compute_deltas(baseline_metrics, bundle_metrics)
                delta_row = {
                    "query_id": qid, "k": 10, "condition": condition, "bundle_id": bundle["bundle_id"],
                    **deltas,
                }
                delta_rows.append(delta_row)

    baseline_fields = [
        "query_id", "k", "question", "target_wrong_answer", "N_retrieved_poison", "N_retrieved_clean",
        "ragdefender_mean_pp_cosine", "ragdefender_mean_pc_cosine", "ragdefender_mean_cc_cosine",
        "ragdefender_top_pair_pp", "ragdefender_top_pair_pc", "ragdefender_top_pair_cc",
        "ragdefender_removed_poison", "ragdefender_removed_clean", "ragdefender_residual_poison_fraction",
        "ragdefender_n_adv_estimated", "ragdefender_stage2_matches_dispatch",
        "filterrag_removed_poison", "filterrag_removed_clean", "filterrag_residual_poison_fraction",
        "filterrag_mean_freq_density_poison", "filterrag_mean_matched_freq_sum_poison",
        "ml_mean_poison_probability", "ml_mean_freq_density_poison", "ml_mean_matched_freq_sum_poison",
        "ml_mean_perplexity_poison", "ml_mean_slm_answer_logprob_poison",
        "ml_removed_poison_t035", "ml_removed_clean_t035", "ml_residual_poison_fraction_t035",
        "ml_removed_poison_t04", "ml_removed_clean_t04", "ml_residual_poison_fraction_t04",
        "ml_removed_poison_t05", "ml_removed_clean_t05", "ml_residual_poison_fraction_t05",
    ]
    bundle_fields = ["condition", "bundle_id", "mutation_strategy"] + baseline_fields
    delta_fields = ["query_id", "k", "condition", "bundle_id"] + [f"delta_{k}" for k in _NUMERIC_METRIC_KEYS] + list(DELTA_ALIASES.keys())

    write_csv(os.path.join(out_dir, "fixed_context_baseline_by_query.csv"), baseline_fields, baseline_rows)
    write_csv(os.path.join(out_dir, "mutation_bundle_scores.csv"), bundle_fields, bundle_rows)
    write_csv(os.path.join(out_dir, "mutation_bundle_deltas.csv"), delta_fields, delta_rows)

    report = build_report(
        pilot_dir=pilot_dir, no_clean_path=no_clean_path, clean_path=clean_path,
        ml_model_path=args.ml_model_path, query_ids=query_ids,
        baseline_rows=baseline_rows, bundle_rows=bundle_rows, delta_rows=delta_rows,
        selected_queries=selected_queries,
    )
    report_path = os.path.join(out_dir, "TEXT_MUTATION_FIXED_CONTEXT_REPORT.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[run_text_mutation_fixed_context_eval] wrote {len(baseline_rows)} baseline row(s), "
          f"{len(bundle_rows)} bundle row(s), {len(delta_rows)} delta row(s) to {out_dir}")
    print(f"[run_text_mutation_fixed_context_eval] SLM generation calls: "
          f"{models.memo_slm_answer_fn.calls} (cache hits: {models.memo_slm_answer_fn.cache_hits})")
    print(f"[run_text_mutation_fixed_context_eval] LM perplexity calls: "
          f"{models.memo_causal_scorer.calls} (cache hits: {models.memo_causal_scorer.cache_hits})")


if __name__ == "__main__":
    main()
