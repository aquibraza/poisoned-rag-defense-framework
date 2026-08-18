from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from defense import ragdefender_internals

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _lazy_st():
    """Lazy import sentence_transformers (requires fix_sentence_transformers to run first)."""
    from sentence_transformers import SentenceTransformer, util as st_util
    return SentenceTransformer, st_util


# RAGDefender internal passage-similarity encoder presets, keyed by
# `ragdefender_version`. Used only when `DefenseConfig.similarity_model` is
# left unset (None) -- an explicit `similarity_model` always wins.
#
# "legacy" == today's historical default (unchanged, byte-identical --
#   `paraphrase-MiniLM-L6-v2`, resolved by sentence-transformers' shorthand
#   lookup, same string as before this preset map existed).
# "paper" == the FINAL ACSAC 2025 paper's Sentence Transformers + Stella
#   embedding model (Kim, Lee, Koo, ACSAC 2025, Sec. 5), mapped to
#   dunzhang/stella_en_1.5B_v5 per the authors' own repository's
#   paper-faithful Stella preset. NOT the victim RAG retriever's encoder --
#   see docs/RAGDEFENDER_FIDELITY_AUDIT_V2.md for that distinction.
RAGDEFENDER_EMBEDDER_PRESETS = {
    "legacy": "paraphrase-MiniLM-L6-v2",
    "paper": "dunzhang/stella_en_1.5B_v5",
}


@dataclass(frozen=True)
class DefenseConfig:
    name: str = "ragdefender"
    device: str = "cuda"
    gpu_id: int = 0
    top_k: Optional[int] = None
    similarity_model: Optional[str] = None
    ragdefender_version: str = "legacy"  # "legacy" | "paper" -- see RAGDEFENDER_EMBEDDER_PRESETS


def _resolve_similarity_model(cfg: DefenseConfig) -> str:
    """Resolve the actual sentence-transformers model id to load: an
    explicit `cfg.similarity_model` always wins (ablation/override
    use-case); otherwise fall back to `RAGDEFENDER_EMBEDDER_PRESETS` keyed by
    `cfg.ragdefender_version`, defaulting to the legacy preset for an
    unrecognized version string. Pure string logic -- does not import
    `sentence_transformers` and makes no network access, so this is
    unit-testable with zero heavy dependencies."""
    if cfg.similarity_model is not None:
        return cfg.similarity_model
    return RAGDEFENDER_EMBEDDER_PRESETS.get(
        cfg.ragdefender_version, RAGDEFENDER_EMBEDDER_PRESETS["legacy"]
    )


_S_MODEL_CACHE: dict = {}

# Recognized explicit device strings that this function will force the
# loaded model onto (via `.to(device)`), rather than allowing
# sentence-transformers' own auto-detection (cuda > mps > cpu) to silently
# override `DefenseConfig.device`. Any other/unrecognized string falls back
# to the pre-existing (legacy) behavior of doing nothing extra -- i.e. this
# set only *adds* explicit handling for device strings that were previously
# being silently ignored (`cpu`, `mps`); it never changes the already
# explicit `cuda` handling below.
_EXPLICIT_DEVICE_STRINGS = ("cuda", "cpu", "mps")


def _apply_stella_dynamic_cache_compat_shim() -> None:
    """Back-fill `transformers.cache_utils.DynamicCache.get_usable_length`,
    removed upstream in the transformers 4.46 KV-cache refactor
    (huggingface/transformers#39106, `get_seq_length()` is the replacement).

    Stella's `trust_remote_code=True` modeling code (`dunzhang/stella_en_1.5B_v5`,
    revision `7817065102fd9e1b031fe874e910c01f40b2f001`, `modeling_qwen.py`)
    is vendored on the HF Hub and was last updated for the pre-4.46 API; it
    still calls `past_key_values.get_usable_length(seq_length)` during a
    forward pass, which raises `AttributeError` against transformers>=4.46
    (measured against transformers==4.57.6 -- see
    docs/RAGDEFENDER_FIDELITY_AUDIT_V2.md / GATE_B readiness notes).

    Two remediation options were considered:
      A. Pin the environment's `transformers` to a pre-4.46 version. Rejected
         as the more invasive option: `transformers` is a shared dependency
         used across this repo's Contriever/BEIR retrieval, GPT-2/DistilGPT2/
         Flan-T5 generation, and legacy-MiniLM embedding code paths, so a
         global downgrade risks altering unrelated model-loading behavior
         (or being incompatible with a newer torch/sentence-transformers
         pin) well beyond RAGDefender's Stella encoder.
      B. (chosen) A narrowly scoped, idempotent compatibility shim applied
         only immediately before loading Stella (i.e. only when
         `cfg.ragdefender_version == "paper"`), which purely *adds back* a
         method that no longer exists -- it never overrides or changes the
         behavior of any method transformers itself still provides, so it
         cannot alter behavior for any other model (MiniLM, Contriever,
         generation models, or any other trust_remote_code model that does
         not call this specific removed method).

    The added method is semantically equivalent to the removed one for this
    encode-only (non-incremental-generation, single-forward-pass) use case:
    `get_usable_length(new_seq_length, layer_idx)` previously returned "how
    much of the cache is already usable," which is just the cache's current
    sequence length, i.e. `get_seq_length(layer_idx)`.
    """
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length, layer_idx=0):  # noqa: ARG001
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = _get_usable_length


def _get_s_model(cfg: DefenseConfig):
    SentenceTransformer, _ = _lazy_st()
    model_name = _resolve_similarity_model(cfg)
    key = (model_name, cfg.device, cfg.gpu_id)
    if key not in _S_MODEL_CACHE:
        # Stella (dunzhang/stella_en_1.5B_v5) ships custom modeling code and
        # requires trust_remote_code=True to load at all; the legacy MiniLM
        # checkpoint does not need it and this flag has no effect either way
        # for a plain sentence-transformers-org model, but is scoped to the
        # paper version only to avoid silently changing legacy load behavior.
        st_kwargs = {"trust_remote_code": True} if cfg.ragdefender_version == "paper" else {}
        if cfg.ragdefender_version == "paper":
            _apply_stella_dynamic_cache_compat_shim()
        model = SentenceTransformer(model_name, **st_kwargs)
        if cfg.device == "cuda":
            torch.cuda.set_device(cfg.gpu_id)
            model = model.to(cfg.device)
        elif cfg.device in _EXPLICIT_DEVICE_STRINGS:
            # Previously silently ignored for "cpu"/"mps" -- sentence-
            # transformers' own auto-detection (cuda > mps > cpu) would run
            # instead, which could place a model requested as "cpu" onto an
            # available MPS/CUDA device without the caller asking for that.
            # An explicit `DefenseConfig.device` must always win.
            model = model.to(cfg.device)
        _S_MODEL_CACHE[key] = model
    return _S_MODEL_CACHE[key]


def _dataset_to_mode(dataset: str) -> str:
    ds = (dataset or "").lower()
    if ds == "hotpotqa":
        return "multihop"
    return "singlehop"


# ---------------------------------------------------------------------------
# Detection algorithms — ported from RAGDefender artifacts/main.py
# ---------------------------------------------------------------------------

def _find_num_adversarial(text_list: List[str], s_model) -> int:
    """Multihop detection (HotpotQA): similarity-based."""
    _, st_util = _lazy_st()
    embeddings = s_model.encode(text_list, convert_to_tensor=True)
    cos_sim_matrix = st_util.cos_sim(embeddings, embeddings)

    avg = torch.mean(cos_sim_matrix, dim=0)
    median = torch.median(cos_sim_matrix, dim=0)
    avg_avg = avg.mean()
    avg_median = median.values.median()

    above_avg = [1 if score > avg_avg else 0 for score in avg]
    above_median = [1 if score > (avg_median + avg_avg) / 2 else 0 for score in median.values]
    final = [1 if above_avg[i] == 1 or above_median[i] == 1 else 0 for i in range(len(above_avg))]

    result = sum(final) if sum(final) > 0 and avg_avg < avg_median else len(text_list) - sum(final)

    del embeddings, cos_sim_matrix, avg, median
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _find_num_adversarial_paper(text_list: List[str], s_model) -> int:
    """Multihop detection (HotpotQA), FINAL ACSAC 2025 paper Eq. (3):
    self-excluded, AND-logic, no-flip concentration estimator -- the
    `ragdefender_paper` counterpart to `_find_num_adversarial` above.

    Delegates the actual math to
    `ragdefender_internals.concentration_stage1_paper` (single source of
    truth for the paper equations -- see that function's docstring for the
    authority rule governing its one paper-silent choice, median
    tie-breaking). `_find_num_adversarial` (the legacy estimator) is left
    completely untouched by this addition."""
    _, st_util = _lazy_st()
    embeddings = s_model.encode(text_list, convert_to_tensor=True)
    cos_sim_matrix = st_util.cos_sim(embeddings, embeddings).cpu().numpy()
    result = ragdefender_internals.concentration_stage1_paper(cos_sim_matrix)

    del embeddings, cos_sim_matrix
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result.n_adv_estimated


def _find_num_adversarial_tfidf(text_list: List[str]) -> int:
    """TF-IDF–based count estimator."""
    import sklearn.feature_extraction.text as sk_text
    import pandas as pd

    stop_words = list(sk_text.ENGLISH_STOP_WORDS)
    tfidf = sk_text.TfidfVectorizer(stop_words=stop_words)
    X = tfidf.fit_transform(text_list)
    all_data = tfidf.get_feature_names_out()
    dense = X.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=all_data)
    dict_tfidf = df.T.sum(axis=1)
    dict_tfidf = dict_tfidf.sort_values(ascending=False)
    top_words = dict_tfidf[:5]
    indices = []
    for word in top_words.index:
        indices.append([1 if word in sentence else 0 for sentence in text_list])
    final = [
        1 if sum(index[i] for index in indices) > math.floor(len(indices) / 2) else 0
        for i in range(len(text_list))
    ]
    return sum(final)


def _find_num_adversarial_agg(text_list: List[str], s_model) -> int:
    """Singlehop detection (NQ, MSMARCO): agglomerative clustering + TF-IDF."""
    from sklearn.cluster import AgglomerativeClustering

    embeddings = s_model.encode(text_list, convert_to_tensor=True)
    model = AgglomerativeClustering(n_clusters=2)
    model.fit(embeddings.cpu().detach().numpy())
    labels = list(model.labels_)
    num_labels = sum(labels)
    num_tfidf = _find_num_adversarial_tfidf(text_list)

    result = (
        min(num_labels, len(text_list) - num_labels)
        if num_labels > 0 and num_tfidf <= int(len(text_list) / 2)
        else max(num_labels, len(text_list) - num_labels)
    )

    del embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _top_similar_pairs(
    texts: List[str], s_model, top_k: int
) -> List[Tuple[int, int, float]]:
    """Return the top_k most similar (i, j, cosine_sim) pairs."""
    _, st_util = _lazy_st()
    embeddings = s_model.encode(texts, convert_to_tensor=True)
    cos_similarities = st_util.cos_sim(embeddings, embeddings)

    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pairs.append((i, j, cos_similarities[i][j].item()))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def _apply_defense_paper(doc_list: List[str], mode: str, s_model, top_k: Optional[int]) -> List[str]:
    """`ragdefender_paper` path: FINAL ACSAC 2025 paper Eq. 1-7.

    - Single-hop (NQ/MS MARCO) grouping: `_find_num_adversarial_agg`,
      reused UNCHANGED -- already paper-faithful (Eq. 1-2) modulo one
      never-firing edge-case guard; see plan §0a item 1 / the
      "clustering-path verification" test. Not reimplemented here.
    - Multi-hop (HotpotQA) grouping: `_find_num_adversarial_paper` (Eq. 3;
      self-excluded, AND logic, no flip).
    - Identification (both modes): the shared, paper-faithful
      `ragdefender_internals.stage2_pair_frequency` (Eq. 4-7), computed once
      from a fresh encode of `doc_list` (mirrors `_top_similar_pairs`'
      encode-again pattern in the legacy path below, kept for structural
      parity rather than optimized away).
    """
    if mode == "singlehop":
        num_adv = _find_num_adversarial_agg(doc_list, s_model)
    else:
        num_adv = _find_num_adversarial_paper(doc_list, s_model)

    if num_adv <= 0:
        return doc_list[:top_k] if top_k else doc_list

    _, st_util = _lazy_st()
    embeddings = s_model.encode(doc_list, convert_to_tensor=True)
    cos_sim_matrix = st_util.cos_sim(embeddings, embeddings).cpu().numpy()
    stage2 = ragdefender_internals.stage2_pair_frequency(cos_sim_matrix, num_adv, p=2.0)
    suspect_indices = set(stage2.selected_indices)

    del embeddings, cos_sim_matrix
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    clean_docs = [doc for i, doc in enumerate(doc_list) if i not in suspect_indices]
    if not clean_docs:
        clean_docs = doc_list
    return clean_docs[:top_k] if top_k else clean_docs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_defense(
    query: str,
    doc_list: List[str],
    dataset: str,
    *,
    device: str = "cuda",
    gpu_id: int = 0,
    top_k: Optional[int] = None,
    ragdefender_version: str = "legacy",
    similarity_model: Optional[str] = None,
) -> List[str]:
    """
    RAGDefender pair-based identification, with two selectable variants.

    `ragdefender_version="legacy"` (default, UNCHANGED from before this
    parameter existed): the authors' released/artifact algorithm (matching
    `RAGDefender/artifacts/main.py` byte-for-byte), documented divergences
    from the final published paper and all. Steps:
      1. Estimate num_poisoned via clustering / similarity (OR logic, hybrid
         threshold, flip branch for the multi-hop case).
      2. Find top similar pairs among documents.
      3. Score each doc by weighted pair frequency (sim^2).
      4. Remove the top-scoring docs (suspected poisoned).
      5. Return the remaining clean documents.
    This branch's code is completely untouched by the addition of the
    `ragdefender_version` parameter -- see `_apply_defense_paper` for the
    new variant, dispatched to below before any of this code runs.

    `ragdefender_version="paper"`: the FINAL ACSAC 2025 paper's Eq. 1-7
    (self-excluded AND-logic Stage 1 for multi-hop, unchanged
    already-paper-faithful clustering Stage 1 for single-hop, paper-faithful
    Stage 2), Stella embedder by default -- see `_apply_defense_paper` and
    docs/RAGDEFENDER_FIDELITY_AUDIT_V2.md.

    `similarity_model`, if given, overrides the version's embedder preset
    (`RAGDEFENDER_EMBEDDER_PRESETS`) for either variant -- e.g. for ablation
    runs.
    """
    if not doc_list:
        return []

    cfg = DefenseConfig(
        device=device,
        gpu_id=gpu_id,
        top_k=top_k,
        ragdefender_version=ragdefender_version,
        similarity_model=similarity_model,
    )
    s_model = _get_s_model(cfg)
    mode = _dataset_to_mode(dataset)

    if ragdefender_version == "paper":
        return _apply_defense_paper(doc_list, mode, s_model, top_k)

    # ---- ragdefender_legacy path: byte-identical to before this parameter existed ----
    if mode == "singlehop":
        num_adv = _find_num_adversarial_agg(doc_list, s_model)
    else:
        num_adv = _find_num_adversarial(doc_list, s_model)

    if num_adv == 0:
        return doc_list[:top_k] if top_k else doc_list

    gen_num = max(1, int(num_adv * (num_adv - 1) / 2))
    adv_pairs = _top_similar_pairs(doc_list, s_model, gen_num)

    pair_cnt: Counter = Counter()
    for x, y, sim in adv_pairs:
        pair_cnt[x] += math.copysign(sim * sim, sim)
        pair_cnt[y] += math.copysign(sim * sim, sim)

    sorted_pairs = sorted(pair_cnt.items(), key=lambda item: item[1], reverse=True)
    suspect_indices = set(idx for idx, _ in sorted_pairs[:num_adv])

    clean_docs = [doc for i, doc in enumerate(doc_list) if i not in suspect_indices]

    if not clean_docs:
        clean_docs = doc_list

    return clean_docs[:top_k] if top_k else clean_docs
