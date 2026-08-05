"""FilterRAG baseline defense (Edemacu et al. 2025, "Defending Against
Knowledge Poisoning Attacks During Retrieval-Augmented Generation") --
passage-level statistical filtering via a Freq-Density score.

Paper algorithm, per retrieved passage `d` for query `q`:

1. Generate `SLM_answer` -- an answer to `q` produced by a *small* language
   model conditioned on `d` alone (not the full retrieved context).
2. `Freq-Density(d) = sum(freq(w, d) for w in (q ⊕ SLM_answer) ∩ d) /
   UniqueWords(d)`, where `∩` is a *semantic* word-similarity match (cosine
   similarity of sentence-transformer word embeddings >= a threshold, paper
   default 0.6 on `all-MiniLM-L6-v2`; see `matching_mode="semantic"` below),
   not literal string equality.
3. Drop `d` if `Freq-Density(d) >= epsilon` (paper default epsilon = 0.2).

Rationale: PoisonedRAG-style adversarial passages are deliberately stuffed
with query/answer keywords (to win both retrieval and generation), which
produces measurably higher Freq-Density than ordinary corpus text -- this
is a per-passage statistical signal, unlike RAGDefender's cross-passage
clustering, so it does not require a "clean anchor cluster" and should be
more robust when the retrieved context is 100% poisoned (see
docs/ANALYSIS.md and docs/RAGDEFENDER_DIAGNOSTIC_PLAN.md for why that
saturation case breaks RAGDefender).

Known deviations from the published method in this implementation (see
docs/FILTERRAG_FIDELITY_AUDIT.md for the full audit and
docs/FILTERRAG_BASELINE.md for the broader baseline writeup):

- The paper uses LLaMA-2/3 as the SLM. This repo has no CUDA/NVIDIA GPU
  available, so `local_hf_slm_answer_fn()` defaults to a much smaller
  seq2seq model (google/flan-t5-small, ~80M params) as a practical proxy.
  This is a known fidelity tradeoff, not a hidden one -- the SLM is fully
  pluggable (`slm_answer_fn` parameter) so a larger model can be swapped in
  later. Device placement is automatic (`resolve_slm_device`): Apple Silicon
  Metal/MPS is used when available, else CUDA, else CPU -- see
  `--filterrag_slm_device` in main.py to override.
- `matching_mode="exact"` (the default, for backward compatibility with
  existing diagnostics/scripts) matches keywords to passage words by
  verbatim (case-folded) string equality, which is equivalent to the
  paper's own similarity-threshold ablation at threshold=1.0 -- the
  *worst*-performing setting the paper reports (Table II), since it lets
  an attacker evade detection with a synonym or word-form change. Pass
  `matching_mode="semantic"` (`--filterrag_matching_mode semantic` at the
  CLI) for the paper-faithful default: cosine-similarity word matching via
  `sentence-transformers/all-MiniLM-L6-v2`, threshold 0.6
  (`DEFAULT_SEMANTIC_THRESHOLD` / `--filterrag_semantic_threshold`). See
  docs/FILTERRAG_FIDELITY_AUDIT.md §3.2/§4 for the full writeup of this
  deviation and why `exact` is *not* promoted to the default here.
- `filterrag_query_only` mode (`slm_answer_fn=None`) is a diagnostic
  ablation, not in the paper: it scores passages using only the query's own
  keywords, skipping the SLM step entirely. This is useful as a fast,
  fully-offline correctness check and cost-free diagnostic baseline, but it
  is *not* the full published algorithm and is expected to be weaker,
  since it can't catch passages stuffed with the *answer* but not much of
  the question text. This is orthogonal to `matching_mode` -- `query_only`
  can use either `exact` or `semantic` matching, but neither makes it
  paper-faithful, since the SLM step is still skipped entirely.
- ML-FilterRAG (Freq-Density + perplexity + log-probability -> trained
  classifier) is out of scope here; only threshold-based FilterRAG is
  implemented. See docs/FILTERRAG_BASELINE.md and
  docs/FILTERRAG_FIDELITY_AUDIT.md §5.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from defense.passages import RetrievedPassage

DEFAULT_EPSILON = 0.2
DEFAULT_SLM_MODEL = "google/flan-t5-small"

# Paper Section IV-B2: "we employ a huggingface sentence transformer and set
# a default similarity threshold value of 0.6 for cosine similarity", footnoted
# as https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2.
DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEMANTIC_THRESHOLD = 0.6

VALID_MATCHING_MODES = ("exact", "semantic")

_WORD_RE = re.compile(r"[a-z0-9']+")

# (question, passage_text) -> generated answer text, or None on failure.
SlmAnswerFn = Callable[[str, str], Optional[str]]


def _tokenize(text: Optional[str]) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _validate_matching_mode(matching_mode: str) -> None:
    if matching_mode not in VALID_MATCHING_MODES:
        raise ValueError(
            f"Unknown filterrag matching_mode {matching_mode!r}; expected one of {VALID_MATCHING_MODES}"
        )


def freq_density_detailed(
    passage_text: str,
    keywords: Sequence[str],
    *,
    matching_mode: str = "exact",
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    semantic_matcher: Optional["SemanticWordMatcher"] = None,
) -> Dict:
    """Compute Freq-Density plus the full per-passage match breakdown.

    `Freq-Density(d) = sum(freq(w, d) for w in matched_doc_words) /
    UniqueWords(d)`, where `matched_doc_words` is the set of unique words in
    `d` that match at least one keyword in `keywords` (typically the token
    set of `query ⊕ SLM_answer`):

    - `matching_mode="exact"` (default, legacy/backward-compatible): a
      passage word matches a keyword iff they are string-equal
      (case-folded). Equivalent to the paper's own similarity-threshold
      ablation at threshold=1.0 -- see module docstring and
      docs/FILTERRAG_FIDELITY_AUDIT.md.
    - `matching_mode="semantic"` (paper-faithful, Section IV-B2): a passage
      word matches a keyword iff the cosine similarity of their
      `sentence-transformers/all-MiniLM-L6-v2` embeddings (or
      `semantic_matcher`'s model, if a custom one is supplied) is
      `>= semantic_threshold` (paper default 0.6). `semantic_matcher`
      defaults to a lazily-loaded, module-cached `SemanticWordMatcher` (see
      `get_semantic_word_matcher()`) -- `sentence_transformers` is only
      ever imported the first time this path actually runs.

    Returns a dict with keys `freq_density_score`, `matched_freq_sum`,
    `unique_word_count`, `matched_keyword_count`, `matched_keywords` (the
    *keywords* -- not passage words -- that had at least one match;
    deduplicated, not capped here, callers may cap for display),
    `matching_mode`, and `semantic_threshold` (`None` when
    `matching_mode="exact"`, since the threshold is meaningless there).

    `matched_freq_sum` is the raw Freq-Density *numerator* alone (i.e.
    `freq_density_score * unique_word_count`, before dividing by
    `UniqueWords(dj)`) -- added for ML-FilterRAG (Edemacu et al. 2025,
    Section III-B2), which uses this exact quantity ("sum of frequencies
    of semantically similar words between (qi⊕aj) and dj") as a feature
    distinct from the Freq-Density ratio itself. Purely additive: every
    pre-existing key/behavior of this function is unchanged. See
    `defense/ml_filterrag.py` and `docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md`.

    Returns an all-zero/empty breakdown for a passage with no tokens at all,
    or an empty `keywords` sequence (avoids a division-by-zero; there's no
    lexical evidence either way).
    """
    _validate_matching_mode(matching_mode)
    doc_tokens = _tokenize(passage_text)
    doc_counts = Counter(doc_tokens)
    unique_word_count = len(doc_counts)
    # Sorted for deterministic output (matched_keywords order, cache-friendly
    # embedding batches) -- does not affect the score itself.
    keyword_list = sorted({k.lower() for k in keywords})

    if unique_word_count == 0 or not keyword_list:
        return {
            "freq_density_score": 0.0,
            "matched_freq_sum": 0,
            "unique_word_count": unique_word_count,
            "matched_keyword_count": 0,
            "matched_keywords": [],
            "matching_mode": matching_mode,
            "semantic_threshold": semantic_threshold if matching_mode == "semantic" else None,
        }

    if matching_mode == "exact":
        matched_keywords = [k for k in keyword_list if k in doc_counts]
        total_freq = sum(doc_counts[k] for k in matched_keywords)
    else:  # "semantic"
        matcher = semantic_matcher if semantic_matcher is not None else get_semantic_word_matcher()
        doc_words = list(doc_counts.keys())
        sims = matcher.similarity_matrix(doc_words, keyword_list)
        matched_keyword_idx = set()
        total_freq = 0
        for i, w in enumerate(doc_words):
            hit_idx = [j for j, s in enumerate(sims[i]) if s >= semantic_threshold]
            if hit_idx:
                total_freq += doc_counts[w]
                matched_keyword_idx.update(hit_idx)
        matched_keywords = [keyword_list[j] for j in sorted(matched_keyword_idx)]

    return {
        "freq_density_score": total_freq / unique_word_count,
        "matched_freq_sum": total_freq,
        "unique_word_count": unique_word_count,
        "matched_keyword_count": len(matched_keywords),
        "matched_keywords": matched_keywords,
        "matching_mode": matching_mode,
        "semantic_threshold": semantic_threshold if matching_mode == "semantic" else None,
    }


def freq_density(
    passage_text: str,
    keywords: Sequence[str],
    *,
    matching_mode: str = "exact",
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    semantic_matcher: Optional["SemanticWordMatcher"] = None,
) -> float:
    """Freq-Density(d) = sum(freq(w, d) for w in keywords if w "matches" d)
    / UniqueWords(d). Thin wrapper around `freq_density_detailed()` that
    returns just the score, preserved for backward compatibility with
    existing callers (e.g. `scripts/filterrag_score_inspection.py`) that
    only need the float. See `freq_density_detailed()` for the full
    per-passage match breakdown (matched keyword count/list, matching mode,
    etc.) and for `matching_mode` semantics.

    `keywords` is typically the token set of (query ⊕ SLM_answer); duplicate
    keywords are harmless (deduplicated internally, matching the paper's set
    union `∩`/`⊕` notation). Returns 0.0 for a passage with no tokens at all
    (avoids a division-by-zero; there's no lexical evidence either way).
    """
    return freq_density_detailed(
        passage_text,
        keywords,
        matching_mode=matching_mode,
        semantic_threshold=semantic_threshold,
        semantic_matcher=semantic_matcher,
    )["freq_density_score"]


def score_passages(
    query: str,
    passages: Sequence[RetrievedPassage],
    *,
    slm_answer_fn: Optional[SlmAnswerFn] = None,
    matching_mode: str = "exact",
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    semantic_matcher: Optional["SemanticWordMatcher"] = None,
    matched_keywords_sample_limit: int = 20,
) -> List[Dict]:
    """Compute a Freq-Density score (plus supporting detail) per passage.

    If `slm_answer_fn` is None, keywords = tokenize(query) only (the
    `filterrag_query_only` diagnostic ablation -- see module docstring).
    Otherwise keywords = tokenize(query) ∪ tokenize(slm_answer_fn(query,
    passage_text)), matching the paper's (query ⊕ SLM_answer) term.

    `matching_mode`/`semantic_threshold`/`semantic_matcher` are forwarded to
    `freq_density_detailed()` -- see there for `matching_mode` semantics.
    When `matching_mode="semantic"` and no `semantic_matcher` is supplied, a
    module-cached `SemanticWordMatcher` is lazily created (one HF model
    load, reused across all passages/queries in this process -- see
    `get_semantic_word_matcher()`); `sentence_transformers` is never
    imported for `matching_mode="exact"` (the default).

    Each returned dict has: `doc_id`, `freq_density_score`, `slm_answer`
    (all pre-existing, unchanged), plus `matching_mode`, `semantic_threshold`
    (`None` for exact mode), `unique_word_count`, `matched_keyword_count`,
    and `matched_keywords_sample` (the matched keywords, truncated to
    `matched_keywords_sample_limit` entries so a passage with many matches
    doesn't blow up diagnostic output size -- `matched_keyword_count` is
    always the *full*, uncapped count).
    """
    _validate_matching_mode(matching_mode)
    if matching_mode == "semantic" and semantic_matcher is None:
        semantic_matcher = get_semantic_word_matcher()

    query_tokens = _tokenize(query)
    scores: List[Dict] = []
    for p in passages:
        slm_answer = slm_answer_fn(query, p.text) if slm_answer_fn is not None else None
        keywords = list(query_tokens) + (_tokenize(slm_answer) if slm_answer else [])
        detail = freq_density_detailed(
            p.text,
            keywords,
            matching_mode=matching_mode,
            semantic_threshold=semantic_threshold,
            semantic_matcher=semantic_matcher,
        )
        scores.append({
            "doc_id": p.doc_id,
            "freq_density_score": detail["freq_density_score"],
            "slm_answer": slm_answer,
            "matching_mode": detail["matching_mode"],
            "semantic_threshold": detail["semantic_threshold"],
            "unique_word_count": detail["unique_word_count"],
            "matched_keyword_count": detail["matched_keyword_count"],
            "matched_keywords_sample": detail["matched_keywords"][:matched_keywords_sample_limit],
        })
    return scores


def filterrag_defense(
    query: str,
    passages: Sequence[RetrievedPassage],
    *,
    epsilon: float = DEFAULT_EPSILON,
    slm_answer_fn: Optional[SlmAnswerFn] = None,
    matching_mode: str = "exact",
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    semantic_matcher: Optional["SemanticWordMatcher"] = None,
) -> Tuple[List[RetrievedPassage], Dict]:
    """Apply FilterRAG: drop every passage with Freq-Density >= epsilon.

    Returns (kept_passages, diag_extra) in the same shape every other
    defense in defense/dispatch.py returns, so it plugs into the existing
    diagnostics pipeline (defense/diagnostics.py) unchanged.

    `matching_mode="exact"` (default) preserves the original, backward
    compatible behavior; `matching_mode="semantic"` is the paper-faithful
    mode (cosine similarity of `all-MiniLM-L6-v2` word embeddings, default
    threshold 0.6) -- see module docstring and
    docs/FILTERRAG_FIDELITY_AUDIT.md.

    `diag_extra["N_adv_estimated_by_ragdefender"]` is repo-wide diagnostic
    schema field name (shared across all defenses, not RAGDefender-specific
    despite the name -- see defense/diagnostics.py); here it is FilterRAG's
    own count of passages it flagged as adversarial. `diag_extra["notes"]`
    records the epsilon, SLM-vs-query-only mode, and matching mode
    (+ threshold, if semantic) used, and `diag_extra["filterrag_scores"]`
    carries the full per-passage score breakdown (see `score_passages()`)
    for deeper analysis beyond what the per-query diagnostic schema
    captures.
    """
    scores = score_passages(
        query,
        passages,
        slm_answer_fn=slm_answer_fn,
        matching_mode=matching_mode,
        semantic_threshold=semantic_threshold,
        semantic_matcher=semantic_matcher,
    )
    score_by_doc_id = {s["doc_id"]: s["freq_density_score"] for s in scores}
    removed_doc_ids = {doc_id for doc_id, score in score_by_doc_id.items() if score >= epsilon}
    kept = [p for p in passages if p.doc_id not in removed_doc_ids]

    mode = "slm" if slm_answer_fn is not None else "query_only_ablation"
    notes = f"filterrag mode={mode} epsilon={epsilon} matching_mode={matching_mode}"
    if matching_mode == "semantic":
        notes += f" semantic_threshold={semantic_threshold}"
    diag_extra = {
        "N_adv_estimated_by_ragdefender": len(removed_doc_ids),
        "filterrag_scores": scores,
        "notes": notes,
    }
    return kept, diag_extra


# ---------------------------------------------------------------------------
# Semantic word matcher: sentence-transformers/all-MiniLM-L6-v2 (lazy-loaded).
#
# Only used by matching_mode="semantic" (see freq_density_detailed()). None
# of this module's top-level imports pull in `sentence_transformers` --
# `import defense.filterrag` stays free of that dependency (and of `torch`)
# unless semantic matching is actually exercised, matching this file's
# existing lazy-import convention for the SLM backend below.
# ---------------------------------------------------------------------------


def _cosine_similarity_matrix(embeddings_a: Sequence[Sequence[float]], embeddings_b: Sequence[Sequence[float]]):
    """Pairwise cosine similarity matrix between two lists of equal-length
    numeric vectors, shape (len(embeddings_a), len(embeddings_b)).

    Implemented with plain `numpy` (already a hard dependency of this repo,
    e.g. `main.py`) rather than `sentence_transformers.util.cos_sim`/`torch`,
    so the similarity math itself has no `torch` dependency -- only loading
    the embedding *model* does (see `SemanticWordMatcher._ensure_model()`).
    This also makes `SemanticWordMatcher` trivially mockable in tests: a
    fake matcher just needs to return plain nested lists/arrays from
    `similarity_matrix()`, no `torch`/`sentence_transformers` involved.
    """
    import numpy as np  # noqa: PLC0415 -- keep numpy off this module's import path too

    a = np.asarray(embeddings_a, dtype=float)
    b = np.asarray(embeddings_b, dtype=float)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


class SemanticWordMatcher:
    """Word-level cosine-similarity matcher backed by a HuggingFace sentence
    transformer (paper default: `sentence-transformers/all-MiniLM-L6-v2`,
    Section IV-B2).

    The underlying `SentenceTransformer` model is loaded lazily -- only on
    the first call to `similarity_matrix()` -- and per-word embeddings are
    cached (words repeat heavily across queries/passages within a run:
    common English words, repeated query tokens), so repeated calls don't
    re-embed the same word. Construct via `get_semantic_word_matcher()` for
    the module-level cached instance; construct directly only for tests
    (e.g. to swap in a fake `_ensure_model`) or to use a non-default model.
    """

    def __init__(self, model_name: str = DEFAULT_SEMANTIC_MODEL):
        self.model_name = model_name
        self._model = None
        self._embedding_cache: Dict[str, object] = {}

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415 -- intentional lazy import
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, words: Sequence[str]):
        import numpy as np  # noqa: PLC0415

        uncached = [w for w in words if w not in self._embedding_cache]
        if uncached:
            model = self._ensure_model()
            vectors = model.encode(list(uncached), convert_to_numpy=True)
            for w, v in zip(uncached, vectors):
                self._embedding_cache[w] = v
        return np.stack([self._embedding_cache[w] for w in words])

    def similarity_matrix(self, words_a: Sequence[str], words_b: Sequence[str]):
        """Return the (len(words_a), len(words_b)) cosine similarity matrix
        between the embeddings of `words_a` and `words_b`."""
        import numpy as np  # noqa: PLC0415

        if not words_a or not words_b:
            return np.zeros((len(words_a), len(words_b)))
        emb_a = self._embed(words_a)
        emb_b = self._embed(words_b)
        return _cosine_similarity_matrix(emb_a, emb_b)


_SEMANTIC_MATCHER_CACHE: Dict[str, SemanticWordMatcher] = {}


def get_semantic_word_matcher(model_name: str = DEFAULT_SEMANTIC_MODEL) -> SemanticWordMatcher:
    """Return a process-wide cached `SemanticWordMatcher` for `model_name`,
    creating it (but not yet loading the HF model -- that happens lazily on
    first `similarity_matrix()` call) if it doesn't exist yet."""
    if model_name not in _SEMANTIC_MATCHER_CACHE:
        _SEMANTIC_MATCHER_CACHE[model_name] = SemanticWordMatcher(model_name)
    return _SEMANTIC_MATCHER_CACHE[model_name]


# ---------------------------------------------------------------------------
# SLM backend: small local HF seq2seq model (lazy-loaded, device-aware).
# ---------------------------------------------------------------------------

VALID_SLM_DEVICES = ("auto", "cpu", "mps", "cuda")

_SLM_PIPELINE_CACHE: Dict[Tuple[str, str], object] = {}
_SLM_DEVICE_LOGGED = False
_SLM_ANSWER_FAILURE_LOGGED = False

# A tiny fixed smoke-test prompt used to verify a freshly-loaded SLM pipeline
# can actually run generate() on its resolved device before it's handed out
# for real use -- see _get_local_hf_slm_pipeline().
_SLM_SMOKE_TEST_PROMPT = (
    "Answer the question using only the context below.\n"
    "Context: Paris is the capital of France.\n"
    "Question: What is the capital of France?\nAnswer:"
)


def resolve_slm_device(requested: str = "auto") -> str:
    """Resolve 'auto'/'cpu'/'mps'/'cuda' into a concrete torch device string.

    'auto' (the default): use Apple Silicon Metal/MPS if available, else
    CUDA if available, else CPU. This repo has no CUDA/NVIDIA GPU on the
    development machine (Apple Silicon), so in practice 'auto' currently
    resolves to 'mps' there -- but the CUDA branch is kept so this behaves
    correctly on a CUDA-equipped machine too.

    An explicit 'cpu'/'mps'/'cuda' is honored if that backend reports
    itself available; otherwise this logs a warning and falls back the same
    way 'auto' would, rather than raising, so a misconfigured
    --filterrag_slm_device doesn't hard-fail an entire run.

    Imports torch lazily: this (and everything else in the "SLM backend"
    section below) only ever runs when FilterRAG's SLM mode
    (--defense filterrag) is actually used, never for
    filterrag_query_only or any other defense -- see the module docstring.
    """
    import torch  # noqa: PLC0415 -- intentional lazy import

    requested = (requested or "auto").lower()
    if requested not in VALID_SLM_DEVICES:
        raise ValueError(f"Unknown filterrag_slm_device {requested!r}; expected one of {VALID_SLM_DEVICES}")

    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()

    if requested == "auto":
        if mps_available:
            return "mps"
        if cuda_available:
            return "cuda"
        return "cpu"
    if requested == "mps" and not mps_available:
        print("[FilterRAG] --filterrag_slm_device=mps requested but MPS is not available on this machine; falling back to auto-detection.")
        return resolve_slm_device("auto")
    if requested == "cuda" and not cuda_available:
        print("[FilterRAG] --filterrag_slm_device=cuda requested but CUDA is not available on this machine; falling back to auto-detection.")
        return resolve_slm_device("auto")
    return requested


def _get_local_hf_slm_pipeline(model_name: str = DEFAULT_SLM_MODEL, device: str = "auto"):
    """Lazily load (and cache) a small local HF text2text-generation
    pipeline on the resolved device. Imported lazily so `import
    defense.filterrag` -- and anything that transitively imports it, like
    defense/dispatch.py -- never requires `transformers`/`torch` unless
    FilterRAG's SLM mode is actually used.

    A freshly-loaded non-CPU pipeline is smoke-tested with one throwaway
    generate() call before being cached/returned. This matters because
    `local_hf_slm_answer_fn._answer()` swallows per-call generation
    exceptions (treating them as "no SLM answer" for that one passage, so a
    rare failure doesn't abort an entire defense run) -- without this
    upfront probe, an accelerator that can *load* a model but can't actually
    run generate() on it (e.g. torch==1.13's MPS backend does not implement
    int64 abs() for T5-family relative-position-bias attention, so
    google/flan-t5-small fails on every single call on MPS with that torch
    version) would silently produce an empty SLM answer for 100% of
    passages -- making `--defense filterrag` silently degrade to the
    `filterrag_query_only` ablation with no error or warning at all.
    """
    global _SLM_DEVICE_LOGGED
    resolved_device = resolve_slm_device(device)
    cache_key = (model_name, resolved_device)
    if cache_key in _SLM_PIPELINE_CACHE:
        return _SLM_PIPELINE_CACHE[cache_key]

    from transformers import pipeline  # noqa: PLC0415 -- intentional lazy import

    pipe = pipeline("text2text-generation", model=model_name, device=resolved_device)

    if resolved_device != "cpu":
        try:
            probe = pipe(_SLM_SMOKE_TEST_PROMPT, max_new_tokens=8, do_sample=False)
            _ = probe[0]["generated_text"]
        except Exception as exc:  # noqa: BLE001 -- any failure means "this device can't run this model"
            print(
                f"[FilterRAG] WARNING: SLM device={resolved_device!r} failed a smoke-test "
                f"generation ({exc!r}); falling back to device='cpu' for model={model_name!r}. "
                "Without this fallback, every generation call would silently fail and be "
                "treated as an empty SLM answer (see defense/filterrag.py)."
            )
            resolved_device = "cpu"
            cache_key = (model_name, resolved_device)
            if cache_key not in _SLM_PIPELINE_CACHE:
                pipe = pipeline("text2text-generation", model=model_name, device=resolved_device)

    if not _SLM_DEVICE_LOGGED:
        print(f"[FilterRAG] SLM device: {resolved_device} (model={model_name})")
        _SLM_DEVICE_LOGGED = True
    _SLM_PIPELINE_CACHE[cache_key] = pipe
    return pipe


def local_hf_slm_answer_fn(
    model_name: str = DEFAULT_SLM_MODEL, max_new_tokens: int = 32, device: str = "auto"
) -> SlmAnswerFn:
    """Build an `SlmAnswerFn` backed by a small local HF seq2seq model.

    The paper uses LLaMA-2/3 as the SLM; this substitutes a much smaller
    model as a practical proxy (see module docstring for the fidelity
    tradeoff). `device` is resolved via `resolve_slm_device()` (default
    'auto': MPS > CUDA > CPU).     Any exception during a single passage's generation (e.g. an
    unexpectedly empty passage) is treated as "no SLM answer" for that one
    passage (keywords fall back to query-only just for it) rather than
    failing the whole defense call -- but the first such failure is logged
    with a warning (not fully silent), since 100% of calls failing here
    would otherwise silently degrade `--defense filterrag` into the
    `filterrag_query_only` ablation with no visible error (this previously
    happened with google/flan-t5-small on MPS + torch==1.13; see
    `_get_local_hf_slm_pipeline`'s upfront smoke test, which now catches
    that specific failure mode before any real passage is scored).
    """
    global _SLM_ANSWER_FAILURE_LOGGED
    pipe = _get_local_hf_slm_pipeline(model_name, device=device)

    def _answer(question: str, passage_text: str) -> Optional[str]:
        global _SLM_ANSWER_FAILURE_LOGGED
        prompt = (
            f"Answer the question using only the context below.\n"
            f"Context: {passage_text}\nQuestion: {question}\nAnswer:"
        )
        try:
            output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
            return output[0]["generated_text"].strip()
        except Exception as exc:  # noqa: BLE001 -- degrade this one passage, not the whole run
            if not _SLM_ANSWER_FAILURE_LOGGED:
                print(
                    f"[FilterRAG] WARNING: SLM generation failed for a passage ({exc!r}); "
                    "treating as no SLM answer for it (query-only keywords used instead). "
                    "This message only prints once even if it recurs."
                )
                _SLM_ANSWER_FAILURE_LOGGED = True
            return None

    return _answer
