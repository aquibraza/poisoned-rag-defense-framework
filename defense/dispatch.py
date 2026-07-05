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
- `none` is a pass-through.
"""
from __future__ import annotations

import difflib
from typing import Dict, List, Optional, Sequence, Tuple

from defense import defense_runner
from defense.controls import oracle_remove_all_poison, random_remove_same_count
from defense.passages import RetrievedPassage

# Canonical set of values accepted by --defense in main.py.
DEFENSE_CHOICES = (
    "none",
    "ragdefender",  # legacy alias, identical behavior to ragdefender_original
    "ragdefender_original",
    "oracle_remove_all_poison",
    "random_remove_same_count",
)

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
) -> Tuple[List[RetrievedPassage], Dict]:
    """Apply `defense_name` to `passages` and return (kept_passages, diag_extra).

    `diag_extra` always contains at least `N_adv_estimated_by_ragdefender`
    (may be None) and `notes` (may be an empty string); individual defenses
    may add more keys.

    `query_id`, when provided, is used by `random_remove_same_count` to
    derive a per-query effective seed (`stable_seed_for_query`) so the
    random-removal baseline draws an independent sample per query instead of
    repeating the same relative removal pattern across every query in a run.
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

    raise ValueError(
        f"Unknown defense {defense_name!r}; expected one of {DEFENSE_CHOICES}"
    )
