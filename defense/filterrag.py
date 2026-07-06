"""FilterRAG baseline defense (Edemacu et al. 2025, "Defending Against
Knowledge Poisoning Attacks During Retrieval-Augmented Generation") --
passage-level statistical filtering via a Freq-Density score.

Paper algorithm, per retrieved passage `d` for query `q`:

1. Generate `SLM_answer` -- an answer to `q` produced by a *small* language
   model conditioned on `d` alone (not the full retrieved context).
2. `Freq-Density(d) = sum(freq(w, d) for w in (q ⊕ SLM_answer) if w in d)
   / UniqueWords(d)`.
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
docs/FILTERRAG_BASELINE.md for the full writeup):

- The paper uses LLaMA-2/3 as the SLM. This repo has no CUDA/NVIDIA GPU
  available, so `local_hf_slm_answer_fn()` defaults to a much smaller
  seq2seq model (google/flan-t5-small, ~80M params) as a practical proxy.
  This is a known fidelity tradeoff, not a hidden one -- the SLM is fully
  pluggable (`slm_answer_fn` parameter) so a larger model can be swapped in
  later. Device placement is automatic (`resolve_slm_device`): Apple Silicon
  Metal/MPS is used when available, else CUDA, else CPU -- see
  `--filterrag_slm_device` in main.py to override.
- `filterrag_query_only` mode (`slm_answer_fn=None`) is a diagnostic
  ablation, not in the paper: it scores passages using only the query's own
  keywords, skipping the SLM step entirely. This is useful as a fast,
  fully-offline correctness check and cost-free diagnostic baseline, but it
  is *not* the full published algorithm and is expected to be weaker,
  since it can't catch passages stuffed with the *answer* but not much of
  the question text.
- ML-FilterRAG (Freq-Density + perplexity + log-probability -> trained
  classifier) is out of scope here; only threshold-based FilterRAG is
  implemented. See docs/FILTERRAG_BASELINE.md.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from defense.passages import RetrievedPassage

DEFAULT_EPSILON = 0.2
DEFAULT_SLM_MODEL = "google/flan-t5-small"

_WORD_RE = re.compile(r"[a-z0-9']+")

# (question, passage_text) -> generated answer text, or None on failure.
SlmAnswerFn = Callable[[str, str], Optional[str]]


def _tokenize(text: Optional[str]) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def freq_density(passage_text: str, keywords: Sequence[str]) -> float:
    """Freq-Density(d) = sum(freq(w, d) for w in keywords if w in d) / UniqueWords(d).

    `keywords` is typically the token set of (query ⊕ SLM_answer); duplicate
    keywords are harmless (deduplicated internally, matching the paper's set
    union `∩`/`⊕` notation). Returns 0.0 for a passage with no tokens at all
    (avoids a division-by-zero; there's no lexical evidence either way).
    """
    doc_tokens = _tokenize(passage_text)
    unique_word_count = len(set(doc_tokens))
    if unique_word_count == 0:
        return 0.0
    doc_counts = Counter(doc_tokens)
    keyword_set = {k.lower() for k in keywords}
    total_freq = sum(doc_counts[w] for w in keyword_set if w in doc_counts)
    return total_freq / unique_word_count


def score_passages(
    query: str,
    passages: Sequence[RetrievedPassage],
    *,
    slm_answer_fn: Optional[SlmAnswerFn] = None,
) -> List[Dict]:
    """Compute a Freq-Density score (plus supporting detail) per passage.

    If `slm_answer_fn` is None, keywords = tokenize(query) only (the
    `filterrag_query_only` diagnostic ablation -- see module docstring).
    Otherwise keywords = tokenize(query) ∪ tokenize(slm_answer_fn(query,
    passage_text)), matching the paper's (query ⊕ SLM_answer) term.
    """
    query_tokens = _tokenize(query)
    scores: List[Dict] = []
    for p in passages:
        slm_answer = slm_answer_fn(query, p.text) if slm_answer_fn is not None else None
        keywords = list(query_tokens) + (_tokenize(slm_answer) if slm_answer else [])
        scores.append({
            "doc_id": p.doc_id,
            "freq_density_score": freq_density(p.text, keywords),
            "slm_answer": slm_answer,
        })
    return scores


def filterrag_defense(
    query: str,
    passages: Sequence[RetrievedPassage],
    *,
    epsilon: float = DEFAULT_EPSILON,
    slm_answer_fn: Optional[SlmAnswerFn] = None,
) -> Tuple[List[RetrievedPassage], Dict]:
    """Apply FilterRAG: drop every passage with Freq-Density >= epsilon.

    Returns (kept_passages, diag_extra) in the same shape every other
    defense in defense/dispatch.py returns, so it plugs into the existing
    diagnostics pipeline (defense/diagnostics.py) unchanged.

    `diag_extra["N_adv_estimated_by_ragdefender"]` is repo-wide diagnostic
    schema field name (shared across all defenses, not RAGDefender-specific
    despite the name -- see defense/diagnostics.py); here it is FilterRAG's
    own count of passages it flagged as adversarial. `diag_extra["notes"]`
    records the epsilon and SLM-vs-query-only mode used, and
    `diag_extra["filterrag_scores"]` carries the full per-passage score
    breakdown (doc_id, freq_density_score, slm_answer) for deeper analysis
    beyond what the per-query diagnostic schema captures.
    """
    scores = score_passages(query, passages, slm_answer_fn=slm_answer_fn)
    score_by_doc_id = {s["doc_id"]: s["freq_density_score"] for s in scores}
    removed_doc_ids = {doc_id for doc_id, score in score_by_doc_id.items() if score >= epsilon}
    kept = [p for p in passages if p.doc_id not in removed_doc_ids]

    mode = "slm" if slm_answer_fn is not None else "query_only_ablation"
    diag_extra = {
        "N_adv_estimated_by_ragdefender": len(removed_doc_ids),
        "filterrag_scores": scores,
        "notes": f"filterrag mode={mode} epsilon={epsilon}",
    }
    return kept, diag_extra


# ---------------------------------------------------------------------------
# SLM backend: small local HF seq2seq model (lazy-loaded, device-aware).
# ---------------------------------------------------------------------------

VALID_SLM_DEVICES = ("auto", "cpu", "mps", "cuda")

_SLM_PIPELINE_CACHE: Dict[Tuple[str, str], object] = {}
_SLM_DEVICE_LOGGED = False


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
    """
    global _SLM_DEVICE_LOGGED
    resolved_device = resolve_slm_device(device)
    cache_key = (model_name, resolved_device)
    if cache_key not in _SLM_PIPELINE_CACHE:
        from transformers import pipeline  # noqa: PLC0415 -- intentional lazy import

        if not _SLM_DEVICE_LOGGED:
            print(f"[FilterRAG] SLM device: {resolved_device} (model={model_name})")
            _SLM_DEVICE_LOGGED = True
        _SLM_PIPELINE_CACHE[cache_key] = pipeline("text2text-generation", model=model_name, device=resolved_device)
    return _SLM_PIPELINE_CACHE[cache_key]


def local_hf_slm_answer_fn(
    model_name: str = DEFAULT_SLM_MODEL, max_new_tokens: int = 32, device: str = "auto"
) -> SlmAnswerFn:
    """Build an `SlmAnswerFn` backed by a small local HF seq2seq model.

    The paper uses LLaMA-2/3 as the SLM; this substitutes a much smaller
    model as a practical proxy (see module docstring for the fidelity
    tradeoff). `device` is resolved via `resolve_slm_device()` (default
    'auto': MPS > CUDA > CPU). Any exception during generation (e.g. an
    unexpectedly empty passage) is swallowed and treated as "no SLM
    answer" (keywords fall back to query-only for that one passage) rather
    than failing the whole defense call.
    """
    pipe = _get_local_hf_slm_pipeline(model_name, device=device)

    def _answer(question: str, passage_text: str) -> Optional[str]:
        prompt = (
            f"Answer the question using only the context below.\n"
            f"Context: {passage_text}\nQuestion: {question}\nAnswer:"
        )
        try:
            output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
            return output[0]["generated_text"].strip()
        except Exception:
            return None

    return _answer
