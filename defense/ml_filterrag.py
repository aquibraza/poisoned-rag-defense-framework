"""ML-FilterRAG-top-k -- MVP of Edemacu et al. 2025's supervised-classifier
defense (Algorithm 2 / Section III-B2).

STATUS -- "ML-FilterRAG-top-k", not a full Algorithm-2 reproduction: the
paper retrieves an oversized `top-s` candidate pool, classifies every
candidate, then takes `top-k` survivors for the LLM. This repo's harness
(`main.py`) retrieves exactly `top_k` directly (no `top-s` superset), so
this module classifies whatever passages it is handed -- always name
results from this module "ML-FilterRAG-top-k" (or equivalent explicit
wording), never bare "ML-FilterRAG", until a `top-s` harness exists. See
`docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md` sections 1, 9, 10.

Paper-cited features (Section III-B2, exactly these 4 -- `DEFAULT_FEATURE_NAMES`):
  - `freq_density_score`   -- semantic Freq-Density (Eq. 4), reused from
    `defense.filterrag.freq_density_detailed()`, never reimplemented here.
  - `matched_freq_sum`     -- the same call's Freq-Density numerator
    ("sum of frequencies of semantically similar words between (qi⊕aj)
    and dj").
  - `perplexity`           -- perplexity of the passage `dj` alone, under a
    small local causal LM (`CausalLMScorer`, default `distilgpt2`) --
    "inspired by" PoisonedRAG's (Zou et al.) use of perplexity; the paper
    does not name a specific model, so this LM choice is a repo proxy.
  - `slm_answer_logprob`   -- the *joint* log-probability of the SLM's
    generated answer `aj`, `-loss * n` (see `slm_answer_joint_logprob()`
    below) -- never the raw mean training loss.

Auxiliary/repo-only features (always computed + logged for diagnostics and
dataset CSVs, but excluded from the default classifier's feature matrix --
`AUXILIARY_FEATURE_NAMES`): `exact_freq_density_score`,
`matched_keyword_count`, `unique_word_count`, `slm_answer_length`,
`passage_length`, `query_passage_lexical_overlap`, `retrieval_score`.
Feeding any of these into the classifier (e.g. via a custom
`feature_names=` argument) produces a *repo-augmented ML-FilterRAG
variant*, not the paper-aligned baseline -- report it as such.

Never calls an LLM/GPT/PaLM/Vicuna API and never runs live generation
through `llm.query()`: `slm_answer_fn` (if supplied) must be a local
function such as `defense.filterrag.local_hf_slm_answer_fn(...)`, and
perplexity/log-probability scoring use only local HF models. `torch` /
`transformers` / `numpy` / `sklearn` / `joblib` are all imported lazily
inside the functions that need them (mirroring `defense/filterrag.py`'s
existing convention), so importing this module itself has no heavy
dependency.

See `docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md` for the full paper-fidelity
audit, data-construction plan, and paper-faithful-vs-proxy summary table.
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

from defense.filterrag import (
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_SLM_MODEL,
    SemanticWordMatcher,
    SlmAnswerFn,
    _get_local_hf_slm_pipeline,
    _tokenize,
    freq_density_detailed,
    get_semantic_word_matcher,
)
from defense.passages import RetrievedPassage

DEFAULT_LM_MODEL = "distilgpt2"
DEFAULT_THRESHOLD = 0.5

# Section III-B2, exactly 4 features -- the paper-aligned classifier's
# feature vector. Order is this repo's own explicit choice (the paper does
# not specify a feature-vector order); do not widen this tuple without also
# updating docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md and calling the result
# a repo-augmented variant, not "ML-FilterRAG".
DEFAULT_FEATURE_NAMES: Tuple[str, ...] = (
    "freq_density_score",
    "matched_freq_sum",
    "perplexity",
    "slm_answer_logprob",
)

# Always computed + logged; never fed to the default classifier.
AUXILIARY_FEATURE_NAMES: Tuple[str, ...] = (
    "exact_freq_density_score",
    "matched_keyword_count",
    "unique_word_count",
    "slm_answer_length",
    "passage_length",
    "query_passage_lexical_overlap",
    "retrieval_score",
)

ALL_FEATURE_NAMES: Tuple[str, ...] = DEFAULT_FEATURE_NAMES + AUXILIARY_FEATURE_NAMES

VALID_MODEL_TYPES = ("random_forest", "xgboost")

# Appendix C, Table VI: Random Forest for HotpotQA/MS-MARCO, XGBoost for NQ.
PAPER_ALIGNED_MODEL_TYPE_BY_DATASET: Dict[str, str] = {
    "hotpotqa": "random_forest",
    "msmarco": "random_forest",
    "nq": "xgboost",
}

# Must match defense.filterrag.local_hf_slm_answer_fn()'s internal prompt
# byte-for-byte (duplicated here rather than imported, since it's a private
# f-string literal there, not a module constant -- this module only ever
# *reads* defense/filterrag.py, per the additive-only constraint on that
# file). slm_answer_logprob() needs to reconstruct the exact prompt the SLM
# conditioned on when it generated `aj`, to teacher-force-score `aj` under
# that same conditioning input.
_SLM_PROMPT_TEMPLATE = (
    "Answer the question using only the context below.\n"
    "Context: {passage_text}\nQuestion: {question}\nAnswer:"
)

_CAUSAL_LM_CACHE: Dict[Tuple[str, str], "CausalLMScorer"] = {}
_CLASSIFIER_CACHE: Dict[str, "MLFilterRAGClassifier"] = {}
_MISSING_LOGPROB_MODEL_WARNED = False


def _safe_float(value) -> float:
    """Coerce to a finite float; `None`/non-numeric/NaN/inf becomes `0.0`
    (a documented fallback -- extracted features must never be NaN/inf, so
    a downstream classifier never silently trains/predicts on garbage)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return f


def _lexical_overlap(query_tokens: Sequence[str], passage_tokens: Sequence[str]) -> float:
    """Jaccard overlap between two token sets. Repo-only auxiliary feature,
    not paper-cited. `0.0` if either side is empty."""
    a, b = set(query_tokens), set(passage_tokens)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class CausalLMScorer:
    """Lazily-loaded local causal LM used to score `perplexity(dj)`.

    Mirrors `defense.filterrag.SemanticWordMatcher`'s lazy-load pattern:
    construction is free, `transformers`/`torch` are only imported the
    first time `perplexity()` actually runs.
    """

    def __init__(self, model_name: str = DEFAULT_LM_MODEL, device: str = "cpu", max_tokens: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
        return self._model, self._tokenizer

    def perplexity(self, text: Optional[str]) -> float:
        """`exp(mean token negative-log-likelihood)` of `text` under this
        causal LM (paper feature: "perplexity", inspired by [Zou et
        al./PoisonedRAG]'s use of perplexity -- scored on the passage `dj`
        alone, matching the paper's Fig. 1 "Freq-Density vs Perplexity"
        pair-plot framing, one point per passage).

        Returns `1.0` (a documented degenerate fallback, not a
        paper-specified value) for empty/whitespace-only text, or text
        that tokenizes to fewer than 2 tokens -- a causal LM has no
        next-token target to score for a single-token sequence.
        """
        import torch  # noqa: PLC0415

        model, tokenizer = self._ensure_model()
        if not text or not text.strip():
            return 1.0
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_tokens)
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.shape[-1] < 2:
            return 1.0
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
        loss = float(outputs.loss.item())
        if not math.isfinite(loss):
            return 1.0
        return float(math.exp(loss))


def get_causal_lm_scorer(model_name: str = DEFAULT_LM_MODEL, device: str = "cpu") -> "CausalLMScorer":
    """Process-wide cached `CausalLMScorer`, keyed by `(model_name, device)`
    -- mirrors `defense.filterrag.get_semantic_word_matcher()`."""
    key = (model_name, device)
    if key not in _CAUSAL_LM_CACHE:
        _CAUSAL_LM_CACHE[key] = CausalLMScorer(model_name, device=device)
    return _CAUSAL_LM_CACHE[key]


def get_slm_model_and_tokenizer(model_name: str = DEFAULT_SLM_MODEL, device: str = "auto"):
    """Fetch `(model, tokenizer)` from FilterRAG's cached SLM pipeline
    (`defense.filterrag._get_local_hf_slm_pipeline`) -- reuses the exact
    same cached pipeline `local_hf_slm_answer_fn()` already builds/caches
    for `aj` generation, so this never triggers a second model load."""
    pipe = _get_local_hf_slm_pipeline(model_name, device=device)
    return pipe.model, pipe.tokenizer


def slm_answer_joint_logprob(
    model,
    tokenizer,
    prompt_text: str,
    answer_text: Optional[str],
    *,
    max_length: int = 512,
) -> Tuple[float, int]:
    """Joint log-probability of `answer_text` under `model`/`tokenizer`,
    teacher-forced on `prompt_text` -- the paper's "joint log probability
    of the SLM's output aj" (Section III-B2), scored under the *same* SLM
    that generated `aj`.

    For an encoder-decoder model, `model(**encoder_inputs,
    labels=answer_ids)` returns the *mean* cross-entropy loss averaged over
    the answer's non-pad tokens (`loss = -mean_i log P(token_i)`), **not**
    the joint log-probability. This function performs the exact conversion
    `joint_logprob = -loss * n`, where `n` is the count of answer tokens
    actually scored (`answer_ids.numel()`; batch size is always 1 here, so
    there is no padding to exclude). The raw `loss` must never be returned
    or treated as if it were `joint_logprob` -- see
    `docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md` section 3.

    Returns `(0.0, 0)` if `answer_text` is `None`/empty (the SLM produced no
    answer for this passage) -- a documented neutral fallback, not a crash.

    `encoder_inputs`/`label_ids` are moved onto `model`'s own device (via
    `next(model.parameters()).device`) *before* the forward pass. Without
    this, a model placed on a non-CPU accelerator (e.g. `--filterrag_slm_device
    mps`/`auto` resolving to `mps`) would be called with CPU tensors from
    `tokenizer(..., return_tensors="pt")`, which raises a device-mismatch
    `RuntimeError` inside `model(...)` -- this is a real bug fix, not a
    silent-fallback path: a device-mismatch failure must surface as an
    exception here, never degrade to `slm_answer_logprob=0.0`.
    """
    if not answer_text or not answer_text.strip():
        return 0.0, 0

    import torch  # noqa: PLC0415

    encoder_inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)
    label_ids = tokenizer(answer_text, return_tensors="pt", truncation=True, max_length=max_length).input_ids
    n = int(label_ids.numel())
    if n == 0:
        return 0.0, 0

    device = next(model.parameters()).device
    encoder_inputs = {k: v.to(device) for k, v in encoder_inputs.items()}
    label_ids = label_ids.to(device)

    with torch.no_grad():
        outputs = model(**encoder_inputs, labels=label_ids)
    loss = float(outputs.loss.item())
    if not math.isfinite(loss):
        return 0.0, n
    return -loss * n, n


def _warn_missing_logprob_model_once() -> None:
    global _MISSING_LOGPROB_MODEL_WARNED
    if not _MISSING_LOGPROB_MODEL_WARNED:
        print(
            "[ML-FilterRAG] WARNING: slm_answer_logprob requested for a passage with a "
            "non-empty SLM answer, but no slm_logprob_model/slm_logprob_tokenizer was "
            "supplied to extract_features(); defaulting slm_answer_logprob=0.0 for every "
            "such passage. Real ml_filterrag runs (via defense/dispatch.py) always supply "
            "these -- this fallback is only expected in low-level/ablation callers."
        )
        _MISSING_LOGPROB_MODEL_WARNED = True


def extract_features(
    query: str,
    passages: Sequence[RetrievedPassage],
    *,
    slm_answer_fn: Optional[SlmAnswerFn] = None,
    slm_logprob_model=None,
    slm_logprob_tokenizer=None,
    matching_mode: str = "semantic",
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    semantic_matcher: Optional[SemanticWordMatcher] = None,
    causal_lm_scorer: Optional[CausalLMScorer] = None,
    lm_model_name: str = DEFAULT_LM_MODEL,
    lm_device: str = "cpu",
) -> List[Dict]:
    """Compute the full ML-FilterRAG feature dict for every passage:
    `doc_id`, `slm_answer` (bookkeeping, not a feature name), every
    `DEFAULT_FEATURE_NAMES` entry, and every `AUXILIARY_FEATURE_NAMES`
    entry.

    Reuses `defense.filterrag.freq_density_detailed()` for every
    Freq-Density-derived value -- word matching is never reimplemented
    here. Label-free: never reads/returns `RetrievedPassage.is_poison`.
    Never calls an LLM/GPT/PaLM/Vicuna API; `slm_answer_fn`, if given, must
    be a local function (e.g.
    `defense.filterrag.local_hf_slm_answer_fn(...)`).

    `slm_logprob_model`/`slm_logprob_tokenizer` should be the encoder-decoder
    SLM's own `(model, tokenizer)` (e.g. via `get_slm_model_and_tokenizer()`)
    so `slm_answer_logprob` scores `aj` under the same model that generated
    it. If omitted while `slm_answer_fn` produces a non-empty answer,
    `slm_answer_logprob` degrades to `0.0` with a one-time warning (see
    `_warn_missing_logprob_model_once`).
    """
    if matching_mode == "semantic" and semantic_matcher is None:
        semantic_matcher = get_semantic_word_matcher()
    if causal_lm_scorer is None:
        causal_lm_scorer = get_causal_lm_scorer(lm_model_name, device=lm_device)

    query_tokens = _tokenize(query)
    rows: List[Dict] = []
    for p in passages:
        slm_answer = slm_answer_fn(query, p.text) if slm_answer_fn is not None else None
        answer_tokens = _tokenize(slm_answer) if slm_answer else []
        keywords = list(query_tokens) + answer_tokens

        primary_detail = freq_density_detailed(
            p.text, keywords,
            matching_mode=matching_mode,
            semantic_threshold=semantic_threshold,
            semantic_matcher=semantic_matcher,
        )
        exact_detail = primary_detail if matching_mode == "exact" else freq_density_detailed(
            p.text, keywords, matching_mode="exact",
        )

        perplexity = causal_lm_scorer.perplexity(p.text)

        if slm_logprob_model is not None and slm_logprob_tokenizer is not None:
            prompt_text = _SLM_PROMPT_TEMPLATE.format(passage_text=p.text, question=query)
            slm_logprob, _n_scored = slm_answer_joint_logprob(
                slm_logprob_model, slm_logprob_tokenizer, prompt_text, slm_answer,
            )
        else:
            if slm_answer:
                _warn_missing_logprob_model_once()
            slm_logprob = 0.0

        row = {
            "doc_id": p.doc_id,
            "slm_answer": slm_answer,
            "freq_density_score": _safe_float(primary_detail["freq_density_score"]),
            "matched_freq_sum": _safe_float(primary_detail["matched_freq_sum"]),
            "perplexity": _safe_float(perplexity),
            "slm_answer_logprob": _safe_float(slm_logprob),
            "exact_freq_density_score": _safe_float(exact_detail["freq_density_score"]),
            "matched_keyword_count": int(primary_detail["matched_keyword_count"]),
            "unique_word_count": int(primary_detail["unique_word_count"]),
            "slm_answer_length": int(len(answer_tokens)),
            "passage_length": int(len(_tokenize(p.text))),
            "query_passage_lexical_overlap": _safe_float(
                _lexical_overlap(query_tokens, _tokenize(p.text))
            ),
            "retrieval_score": _safe_float(p.retrieval_score) if p.retrieval_score is not None else 0.0,
        }
        rows.append(row)
    return rows


def features_to_matrix(feature_rows: Sequence[Dict], feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES):
    """Fixed-column-order `numpy` array from `extract_features()` output,
    columns in exactly `feature_names` order. Raises `KeyError` immediately
    if any `feature_names` entry is missing from a row (fail fast, not a
    silent `0.0`) -- `extract_features()` always populates every
    `DEFAULT_FEATURE_NAMES`/`AUXILIARY_FEATURE_NAMES` key, so a `KeyError`
    here means a caller-supplied `feature_names` entry is misspelled."""
    import numpy as np  # noqa: PLC0415

    rows = [[float(r[name]) for name in feature_names] for r in feature_rows]
    return np.asarray(rows, dtype=float)


def paper_aligned_model_type(dataset: str) -> str:
    """Appendix C, Table VI: Random Forest for HotpotQA/MS-MARCO, XGBoost
    for NQ. Raises for an unrecognized dataset name -- there is no
    paper-aligned default to silently fall back to."""
    key = (dataset or "").lower()
    if key not in PAPER_ALIGNED_MODEL_TYPE_BY_DATASET:
        raise ValueError(
            f"No paper-aligned ML-FilterRAG classifier choice known for dataset {dataset!r}; "
            f"expected one of {sorted(PAPER_ALIGNED_MODEL_TYPE_BY_DATASET)} (Appendix C, Table VI)."
        )
    return PAPER_ALIGNED_MODEL_TYPE_BY_DATASET[key]


def query_level_train_test_split(
    query_ids: Sequence[str], *, test_fraction: float = 0.2, seed: int = 12
) -> Tuple[set, set]:
    """Deterministic, seeded split of the *set* of `query_id`s into
    `(train_ids, test_ids)` -- see docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md
    section 2. The split key is `query_id` alone (never `attack`/`k`/`N`);
    callers (e.g. `scripts/build_ml_filterrag_dataset.py`) must compute this
    once per dataset, before generating any passage rows, so every row for
    a given `query_id` -- across every `k`, `N`, and `attack_method` used to
    build that dataset -- lands entirely in one split.

    Duplicate `query_id`s in the input are deduplicated before splitting.
    Deterministic for a fixed `seed` (same input -> same split every time).
    """
    import random as _random  # noqa: PLC0415 -- stdlib, but kept lazy/local for symmetry with the rest of this module

    unique_ids = sorted(set(query_ids))
    rng = _random.Random(seed)
    rng.shuffle(unique_ids)
    n_test = int(round(len(unique_ids) * test_fraction)) if unique_ids else 0
    n_test = min(max(n_test, 1 if unique_ids else 0), len(unique_ids))
    test_ids = set(unique_ids[:n_test])
    train_ids = set(unique_ids[n_test:])
    return train_ids, test_ids


def assert_no_query_id_leakage(train_ids: Sequence[str], test_ids: Sequence[str]) -> None:
    """Fail loudly (not silently) if any `query_id` appears in both splits.
    Called by `scripts/build_ml_filterrag_dataset.py` before writing output,
    and re-run by `scripts/train_ml_filterrag.py` against whatever subset of
    rows it's actually about to train/evaluate on."""
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        sample = sorted(overlap)[:5]
        raise AssertionError(
            f"Query-level train/test split leakage detected: {len(overlap)} query_id(s) "
            f"present in both splits (e.g. {sample}) -- this must never happen; see "
            "docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md section 2."
        )


class MLFilterRAGClassifier:
    """Uniform train/predict_proba/predict/save/load wrapper around a
    scikit-learn `RandomForestClassifier` or an optional XGBoost
    `XGBClassifier`, with a self-describing artifact (`feature_names`,
    `model_type`, `threshold_default`, `training_meta`).

    `xgboost` is not a hard dependency of this repo; `model_type="xgboost"`
    raises a clear, actionable error at the point it's actually needed
    (first `train()`/`_build_model()` call) if `xgboost` isn't installed,
    rather than making `import defense.ml_filterrag` itself require it.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
        threshold_default: float = DEFAULT_THRESHOLD,
        training_meta: Optional[Dict] = None,
        **model_kwargs,
    ):
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"Unknown ml_filterrag model_type {model_type!r}; expected one of {VALID_MODEL_TYPES}")
        self.model_type = model_type
        self.feature_names = tuple(feature_names)
        self.threshold_default = threshold_default
        self.training_meta = dict(training_meta or {})
        self.model_kwargs = dict(model_kwargs)
        # Built lazily by train() (or set directly by load()) -- see
        # docstring: constructing this wrapper must never require xgboost
        # to be installed just to *load* an already-fitted RF artifact.
        self._model = None

    def _build_model(self):
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

            kwargs = {"random_state": 12, "n_estimators": 100}
            kwargs.update(self.model_kwargs)
            return RandomForestClassifier(**kwargs)

        try:
            from xgboost import XGBClassifier  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "model_type='xgboost' requires the optional 'xgboost' package, which is "
                "not installed in this environment (`pip install xgboost`). This is "
                "expected/documented: XGBoost is the paper-aligned classifier for NQ "
                "(Appendix C, Table VI) but is not a hard dependency of this repo. Use "
                "model_type='random_forest' (paper-aligned for HotpotQA/MS-MARCO), or "
                "install xgboost to train a paper-faithful NQ classifier."
            ) from exc
        kwargs = {"random_state": 12, "eval_metric": "logloss"}
        kwargs.update(self.model_kwargs)
        return XGBClassifier(**kwargs)

    def train(self, X, y) -> "MLFilterRAGClassifier":
        if self._model is None:
            self._model = self._build_model()
        self._model.fit(X, y)
        return self

    def _require_trained(self):
        if self._model is None:
            raise RuntimeError(
                "MLFilterRAGClassifier has not been trained or loaded yet -- call "
                "train(X, y) or MLFilterRAGClassifier.load(path) first."
            )

    def predict_proba(self, X):
        import numpy as np  # noqa: PLC0415

        self._require_trained()
        proba = self._model.predict_proba(np.asarray(X, dtype=float))
        classes = list(self._model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        # Degenerate single-class training data (e.g. a tiny smoke test with
        # only one label present) -- no poison-probability signal exists.
        return np.zeros(proba.shape[0], dtype=float)

    def predict(self, X, threshold: Optional[float] = None):
        t = self.threshold_default if threshold is None else threshold
        return (self.predict_proba(X) >= t).astype(int)

    def save(self, path: str) -> None:
        import joblib  # noqa: PLC0415

        self._require_trained()
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "model_type": self.model_type,
                "feature_names": list(self.feature_names),
                "threshold_default": self.threshold_default,
                "training_meta": self.training_meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: Optional[str]) -> "MLFilterRAGClassifier":
        """Load a saved artifact, verifying it is self-describing before
        trusting it. Raises `FileNotFoundError` for a missing/empty path
        and `ValueError` if the artifact lacks `feature_names`/
        `training_meta` (i.e. it wasn't produced by `save()`, or is
        corrupted) -- never a silent no-op or a raw joblib stack trace."""
        import joblib  # noqa: PLC0415

        if not path:
            raise ValueError(
                "MLFilterRAGClassifier.load() requires a model path "
                "(--ml_filterrag_model_path is required for --defense ml_filterrag)."
            )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ML-FilterRAG model artifact not found at {path!r}. Train one first with "
                "scripts/train_ml_filterrag.py, or check --ml_filterrag_model_path."
            )
        try:
            artifact = joblib.load(path)
        except ImportError as exc:
            raise ImportError(
                f"Failed to load ML-FilterRAG model artifact at {path!r}: {exc}. This "
                "usually means the artifact was trained with model_type='xgboost' but the "
                "'xgboost' package is not installed in this environment."
            ) from exc

        if not isinstance(artifact, dict):
            raise ValueError(
                f"ML-FilterRAG model artifact at {path!r} is not a dict produced by "
                "MLFilterRAGClassifier.save() -- refusing to load it silently."
            )
        missing = [k for k in ("feature_names", "training_meta") if k not in artifact]
        if missing:
            raise ValueError(
                f"ML-FilterRAG model artifact at {path!r} is missing required key(s) "
                f"{missing} -- it was not saved by MLFilterRAGClassifier.save() (or is from "
                "an incompatible/corrupted version); refusing to load it silently."
            )

        obj = cls(
            model_type=artifact.get("model_type", "random_forest"),
            feature_names=artifact["feature_names"],
            threshold_default=artifact.get("threshold_default", DEFAULT_THRESHOLD),
            training_meta=artifact["training_meta"],
        )
        obj._model = artifact["model"]
        return obj


def load_classifier_cached(path: str) -> MLFilterRAGClassifier:
    """Load-and-cache an `MLFilterRAGClassifier` by resolved absolute path
    -- mirrors `defense.filterrag`'s `_SEMANTIC_MATCHER_CACHE`/
    `_SLM_PIPELINE_CACHE` module-level cache pattern, so repeated calls
    (e.g. once per query in a long eval run) reuse the already-loaded
    classifier instead of re-reading the artifact from disk every time."""
    if not path:
        raise ValueError(
            "load_classifier_cached() requires a model path "
            "(--ml_filterrag_model_path is required for --defense ml_filterrag)."
        )
    resolved = os.path.abspath(path)
    if resolved not in _CLASSIFIER_CACHE:
        _CLASSIFIER_CACHE[resolved] = MLFilterRAGClassifier.load(resolved)
    return _CLASSIFIER_CACHE[resolved]


def _artifact_hash(path: Optional[str]) -> Optional[str]:
    """md5 of the model artifact file, so a diagnostic run is traceable to
    an exact artifact (`None` if `path` is falsy or the file is gone)."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ml_filterrag_defense(
    query: str,
    passages: Sequence[RetrievedPassage],
    *,
    classifier: MLFilterRAGClassifier,
    threshold: Optional[float] = None,
    slm_answer_fn: Optional[SlmAnswerFn] = None,
    slm_logprob_model=None,
    slm_logprob_tokenizer=None,
    slm_model_name: str = DEFAULT_SLM_MODEL,
    matching_mode: str = "semantic",
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    semantic_matcher: Optional[SemanticWordMatcher] = None,
    causal_lm_scorer: Optional[CausalLMScorer] = None,
    lm_model_name: str = DEFAULT_LM_MODEL,
    lm_device: str = "cpu",
    model_path_for_diagnostics: Optional[str] = None,
) -> Tuple[List[RetrievedPassage], Dict]:
    """ML-FilterRAG-top-k: compute features for every passage, predict
    poison probability with `classifier`, and drop every passage whose
    predicted probability is `>= threshold` (`classifier.threshold_default`
    if `threshold` is `None`). Returns `(kept_passages, diag_extra)`, same
    shape as every other defense in `defense/dispatch.py`.

    Never reads `RetrievedPassage.is_poison` -- ground truth is not
    available to this function, exactly like every other defense.
    """
    resolved_threshold = classifier.threshold_default if threshold is None else threshold

    feature_rows = extract_features(
        query, passages,
        slm_answer_fn=slm_answer_fn,
        slm_logprob_model=slm_logprob_model,
        slm_logprob_tokenizer=slm_logprob_tokenizer,
        matching_mode=matching_mode,
        semantic_threshold=semantic_threshold,
        semantic_matcher=semantic_matcher,
        causal_lm_scorer=causal_lm_scorer,
        lm_model_name=lm_model_name,
        lm_device=lm_device,
    )
    X = features_to_matrix(feature_rows, classifier.feature_names)
    proba = classifier.predict_proba(X)

    removed_doc_ids = set()
    predictions = []
    for row, p_poison in zip(feature_rows, proba):
        p_poison = float(p_poison)
        removed = p_poison >= resolved_threshold
        if removed:
            removed_doc_ids.add(row["doc_id"])
        predictions.append(
            {
                "doc_id": row["doc_id"],
                "features": {k: row[k] for k in ALL_FEATURE_NAMES},
                "predicted_proba": p_poison,
                "predicted_label": "adversarial" if removed else "non-adversarial",
                "removed": removed,
            }
        )

    kept = [p for p in passages if p.doc_id not in removed_doc_ids]

    is_paper_aligned = (
        matching_mode == "semantic"
        and abs(semantic_threshold - DEFAULT_SEMANTIC_THRESHOLD) < 1e-9
        and tuple(classifier.feature_names) == DEFAULT_FEATURE_NAMES
        and not classifier.training_meta.get("proxy_classifier", False)
    )

    model_hash = _artifact_hash(model_path_for_diagnostics)

    notes = (
        f"ml_filterrag_top_k model_type={classifier.model_type} threshold={resolved_threshold} "
        f"matching_mode={matching_mode} semantic_threshold={semantic_threshold} "
        f"slm_model={slm_model_name} lm_model={lm_model_name} paper_aligned={is_paper_aligned}"
    )

    diag_extra = {
        "N_adv_estimated_by_ragdefender": len(removed_doc_ids),
        "ml_filterrag_predictions": predictions,
        "model_path": model_path_for_diagnostics,
        "model_artifact_hash": model_hash,
        "feature_names": list(classifier.feature_names),
        "model_type": classifier.model_type,
        "threshold": resolved_threshold,
        "matching_mode": matching_mode,
        "semantic_threshold": semantic_threshold,
        "slm_model": slm_model_name,
        "lm_model": lm_model_name,
        "paper_aligned": is_paper_aligned,
        "notes": notes,
    }
    return kept, diag_extra
