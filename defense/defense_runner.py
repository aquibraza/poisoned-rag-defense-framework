from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _lazy_st():
    """Lazy import sentence_transformers (requires fix_sentence_transformers to run first)."""
    from sentence_transformers import SentenceTransformer, util as st_util
    return SentenceTransformer, st_util


@dataclass(frozen=True)
class DefenseConfig:
    name: str = "ragdefender"
    device: str = "cuda"
    gpu_id: int = 0
    top_k: Optional[int] = None
    similarity_model: str = "paraphrase-MiniLM-L6-v2"


_S_MODEL_CACHE: dict = {}


def _get_s_model(cfg: DefenseConfig):
    SentenceTransformer, _ = _lazy_st()
    key = (cfg.similarity_model, cfg.device, cfg.gpu_id)
    if key not in _S_MODEL_CACHE:
        model = SentenceTransformer(cfg.similarity_model)
        if cfg.device == "cuda":
            torch.cuda.set_device(cfg.gpu_id)
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
) -> List[str]:
    """
    Full RAGDefender paper algorithm (pair-based identification).

    Steps (matching artifacts/main.py):
      1. Estimate num_poisoned via clustering / similarity.
      2. Find top similar pairs among documents.
      3. Score each doc by weighted pair frequency (sim^2).
      4. Remove the top-scoring docs (suspected poisoned).
      5. Return the remaining clean documents.
    """
    if not doc_list:
        return []

    cfg = DefenseConfig(device=device, gpu_id=gpu_id, top_k=top_k)
    s_model = _get_s_model(cfg)
    mode = _dataset_to_mode(dataset)

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
