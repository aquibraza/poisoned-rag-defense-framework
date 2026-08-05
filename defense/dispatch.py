"""Single entry point for applying a defense (or diagnostic control) to a
list of retrieved passages.

This module deliberately contains all the "glue" needed to support
diagnostics without touching `defense/defense_runner.py`:

- `ragdefender` / `ragdefender_original` delegate to
  `defense_runner.apply_defense` completely unmodified -- the original
  algorithm's behavior is preserved exactly.
- `estimate_num_adversarial` reuses (via import, not edit)
  `defense_runner`'s private estimator functions so diagnostics and
  `random_remove_same_count` can learn RAGDefender's estimated adversarial
  count independent of whether RAGDefender is the active defense.
- `oracle_remove_all_poison` / `random_remove_same_count` delegate to
  `defense/controls.py` (diagnostic controls, not deployable defenses).
- `filterrag` / `filterrag_query_only` delegate to `defense/filterrag.py`
  (Edemacu et al. 2025 baseline; a second, independent defense family from
  RAGDefender, so its own failure modes and comparisons are meaningful).
- `ml_filterrag` delegates to `defense/ml_filterrag.py` -- the supervised
  ML-FilterRAG-top-k MVP (Algorithm 2, same paper); reuses filterrag's
  `aj`-generation SLM flags for its own `slm_answer_fn`, see module
  docstring there and `docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md`.
- `none` is a pass-through.
"""
from __future__ import annotations

import difflib
from typing import Dict, List, Optional, Sequence, Tuple

from defense import defense_runner
from defense.controls import oracle_remove_all_poison, random_remove_same_count
from defense.filterrag import (
    DEFAULT_EPSILON,
    DEFAULT_SEMANTIC_THRESHOLD,
    filterrag_defense,
    local_hf_slm_answer_fn,
)
from defense.ml_filterrag import (
    DEFAULT_LM_MODEL as ML_FILTERRAG_DEFAULT_LM_MODEL,
    DEFAULT_THRESHOLD as ML_FILTERRAG_DEFAULT_THRESHOLD,
    get_slm_model_and_tokenizer,
    load_classifier_cached,
    ml_filterrag_defense,
)
from defense.passages import RetrievedPassage

# Canonical set of values accepted by --defense in main.py.
DEFENSE_CHOICES = (
    "none",
    "ragdefender",  # legacy alias, identical behavior to ragdefender_original
    "ragdefender_original",
    "oracle_remove_all_poison",
    "random_remove_same_count",
    "filterrag",
    "filterrag_query_only",
    "ml_filterrag",
)

# Defenses that are diagnostic ablations of filterrag, not the published
# algorithm (no SLM step) -- see defense/filterrag.py module docstring.
FILTERRAG_ABLATION_DEFENSES = ("filterrag_query_only",)

# Defenses that are diagnostic-only controls, not deployable defenses.
DIAGNOSTIC_CONTROL_DEFENSES = ("oracle_remove_all_poison", "random_remove_same_count")


def estimate_num_adversarial(
    texts: Sequence[str],
    dataset: str,
    *,
    device: str = "cuda",
    gpu_id: int = 0,
) -> int:
    """Reproduce RAGDefender's adversarial-count estimate (the first stage of
    `defense_runner.apply_defense`) without performing any removal.

    Imports `defense_runner`'s private estimator helpers rather than
    modifying that module. Used so diagnostics can report
    `N_adv_estimated_by_ragdefender` for *every* defense mode (not just when
    RAGDefender itself is the active defense), and so
    `random_remove_same_count` can remove "the same number of passages as
    RAGDefender estimated."
    """
    text_list = list(texts)
    if not text_list:
        return 0
    cfg = defense_runner.DefenseConfig(device=device, gpu_id=gpu_id)
    s_model = defense_runner._get_s_model(cfg)  # noqa: SLF001 -- intentional reuse
    mode = defense_runner._dataset_to_mode(dataset)  # noqa: SLF001
    if mode == "singlehop":
        return int(defense_runner._find_num_adversarial_agg(text_list, s_model))  # noqa: SLF001
    return int(defense_runner._find_num_adversarial(text_list, s_model))  # noqa: SLF001


def _match_kept_by_text_subsequence(
    passages: Sequence[RetrievedPassage], kept_texts: Sequence[str]
) -> List[RetrievedPassage]:
    """Map a text-only defense output back onto the original RetrievedPassage
    objects, preserving doc_id/source/is_poison metadata.

    `defense_runner.apply_defense` only ever deletes and/or truncates its
    input list -- it never reorders or rewrites text -- so `kept_texts` is
    guaranteed to be an order-preserving subsequence of
    `[p.text for p in passages]`. `difflib.SequenceMatcher` is used to align
    the two sequences (handles the rare case of duplicate passage text more
    robustly than a naive greedy scan).
    """
    original_texts = [p.text for p in passages]
    matcher = difflib.SequenceMatcher(None, original_texts, list(kept_texts), autojunk=False)
    kept_indices = set()
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            kept_indices.add(block.a + offset)
    return [p for i, p in enumerate(passages) if i in kept_indices]


def _run_ragdefender_original(
    query: str,
    passages: Sequence[RetrievedPassage],
    dataset: str,
    *,
    device: str,
    gpu_id: int,
    top_k: Optional[int],
) -> Tuple[List[RetrievedPassage], Dict]:
    """Call defense_runner.apply_defense() completely unmodified, then map its
    text-only output back onto RetrievedPassage objects."""
    texts_list = [p.text for p in passages]

    # Capture RAGDefender's raw estimate independently of removal, since
    # apply_defense() has an internal fallback that keeps everything when
    # the estimate is large relative to len(texts_list) -- we want the
    # diagnostic to reflect the *estimate*, not just the survivors of that
    # fallback.
    n_estimate = estimate_num_adversarial(texts_list, dataset, device=device, gpu_id=gpu_id)

    kept_texts = defense_runner.apply_defense(
        query, texts_list, dataset, device=device, gpu_id=gpu_id, top_k=top_k
    )
    kept_passages = _match_kept_by_text_subsequence(passages, kept_texts)

    diag_extra = {
        "N_adv_estimated_by_ragdefender": n_estimate,
        "notes": "",
    }
    return kept_passages, diag_extra


def run_defense(
    defense_name: str,
    query: str,
    passages: Sequence[RetrievedPassage],
    dataset: str,
    *,
    device: str = "cuda",
    gpu_id: int = 0,
    top_k: Optional[int] = None,
    seed: int = 12,
    query_id: Optional[str] = None,
    filterrag_epsilon: float = DEFAULT_EPSILON,
    filterrag_slm_model: str = "google/flan-t5-small",
    filterrag_slm_device: str = "auto",
    filterrag_matching_mode: str = "exact",
    filterrag_semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    ml_filterrag_model_path: Optional[str] = None,
    ml_filterrag_threshold: float = ML_FILTERRAG_DEFAULT_THRESHOLD,
    ml_filterrag_matching_mode: str = "semantic",
    ml_filterrag_semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    ml_filterrag_lm_model: str = ML_FILTERRAG_DEFAULT_LM_MODEL,
) -> Tuple[List[RetrievedPassage], Dict]:
    """Apply `defense_name` to `passages` and return (kept_passages, diag_extra).

    `diag_extra` always contains at least `N_adv_estimated_by_ragdefender`
    (may be None) and `notes` (may be an empty string); individual defenses
    may add more keys.

    `query_id`, when provided, is used by `random_remove_same_count` to
    derive a per-query effective seed (`stable_seed_for_query`) so the
    random-removal baseline draws an independent sample per query instead of
    repeating the same relative removal pattern across every query in a run.

    `filterrag_epsilon`/`filterrag_slm_model`/`filterrag_slm_device`/
    `filterrag_matching_mode`/`filterrag_semantic_threshold` only apply to
    `filterrag`/`filterrag_query_only` -- see defense/filterrag.py.
    `filterrag_slm_device` is unused by `filterrag_query_only` (no SLM is
    ever loaded in that mode). `filterrag_matching_mode="exact"` (default)
    preserves legacy/backward-compatible behavior;
    `filterrag_matching_mode="semantic"` is the paper-faithful mode (see
    docs/FILTERRAG_FIDELITY_AUDIT.md) and applies to both `filterrag` and
    `filterrag_query_only` (matching mode and SLM-vs-query-only are
    orthogonal knobs --     `filterrag_query_only` is never paper-faithful
    either way, since it always skips the SLM step).

    `ml_filterrag_model_path`/`ml_filterrag_threshold`/
    `ml_filterrag_matching_mode`/`ml_filterrag_semantic_threshold`/
    `ml_filterrag_lm_model` only apply to `ml_filterrag` -- see
    defense/ml_filterrag.py ("ML-FilterRAG-top-k" MVP). `ml_filterrag`
    reuses `filterrag_slm_model`/`filterrag_slm_device` for its own `aj`
    generation (no separate `ml_filterrag_slm_*` flags -- see
    docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md sec 9). A missing/invalid
    `ml_filterrag_model_path` raises immediately (ValueError/
    FileNotFoundError), before any feature extraction is attempted.
    """
    name = (defense_name or "none").lower()

    if name == "none":
        return list(passages), {"N_adv_estimated_by_ragdefender": None, "notes": ""}

    if name in ("ragdefender", "ragdefender_original"):
        return _run_ragdefender_original(
            query, passages, dataset, device=device, gpu_id=gpu_id, top_k=top_k
        )

    if name == "oracle_remove_all_poison":
        return oracle_remove_all_poison(passages)

    if name == "random_remove_same_count":
        n_estimate = estimate_num_adversarial(
            [p.text for p in passages], dataset, device=device, gpu_id=gpu_id
        )
        return random_remove_same_count(
            passages, n_to_remove=n_estimate, seed=seed, query_id=query_id
        )

    if name == "filterrag_query_only":
        return filterrag_defense(
            query,
            passages,
            epsilon=filterrag_epsilon,
            slm_answer_fn=None,
            matching_mode=filterrag_matching_mode,
            semantic_threshold=filterrag_semantic_threshold,
        )

    if name == "filterrag":
        slm_answer_fn = local_hf_slm_answer_fn(filterrag_slm_model, device=filterrag_slm_device)
        return filterrag_defense(
            query,
            passages,
            epsilon=filterrag_epsilon,
            slm_answer_fn=slm_answer_fn,
            matching_mode=filterrag_matching_mode,
            semantic_threshold=filterrag_semantic_threshold,
        )

    if name == "ml_filterrag":
        if not ml_filterrag_model_path:
            raise ValueError(
                "--ml_filterrag_model_path is required for --defense ml_filterrag "
                "(train one first with scripts/train_ml_filterrag.py)."
            )
        classifier = load_classifier_cached(ml_filterrag_model_path)
        slm_answer_fn = local_hf_slm_answer_fn(filterrag_slm_model, device=filterrag_slm_device)
        slm_model, slm_tokenizer = get_slm_model_and_tokenizer(
            filterrag_slm_model, device=filterrag_slm_device
        )
        return ml_filterrag_defense(
            query,
            passages,
            classifier=classifier,
            threshold=ml_filterrag_threshold,
            slm_answer_fn=slm_answer_fn,
            slm_logprob_model=slm_model,
            slm_logprob_tokenizer=slm_tokenizer,
            slm_model_name=filterrag_slm_model,
            matching_mode=ml_filterrag_matching_mode,
            semantic_threshold=ml_filterrag_semantic_threshold,
            lm_model_name=ml_filterrag_lm_model,
            model_path_for_diagnostics=ml_filterrag_model_path,
        )

    raise ValueError(
        f"Unknown defense {defense_name!r}; expected one of {DEFENSE_CHOICES}"
    )
