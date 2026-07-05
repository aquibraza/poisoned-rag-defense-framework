"""Passage metadata and poison-label propagation for RAGDefender diagnostics.

This module is deliberately independent of any specific attack or defense
implementation. It defines a plain data structure for a single retrieved
passage, plus small pure helper functions to build/filter/diff lists of
passages while preserving the poison label assigned at retrieval time.

Poison labels come from the *source* of a passage (was it injected by the
attacker, or does it come from the original BEIR corpus?) -- they are never
inferred after the fact from text similarity or content. Callers building the
raw passage dicts (see main.py) are responsible for attaching `is_poison`
based on ground-truth membership in the attacker's adversarial text set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class RetrievedPassage:
    """A single retrieved passage with source-based poison ground truth."""

    doc_id: str
    text: str
    source: str  # "corpus" | "adversarial" | "unknown"
    is_poison: bool
    retrieval_score: Optional[float] = None
    rank: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "source": self.source,
            "is_poison": self.is_poison,
            "retrieval_score": self.retrieval_score,
            "rank": self.rank,
        }


def label_passages(
    raw_passages: Sequence[dict],
    *,
    text_key: str = "context",
    score_key: str = "score",
    doc_id_key: str = "doc_id",
    source_key: str = "source",
    is_poison_key: str = "is_poison",
) -> List[RetrievedPassage]:
    """Convert plain dicts (e.g. main.py's topk_results entries) into
    RetrievedPassage objects.

    `rank` is assigned by list order (0-indexed; rank 0 = first/highest
    scored entry in `raw_passages`). Poison labels are read verbatim from
    `is_poison_key`/`source_key` on each raw dict -- they must already have
    been attached at construction time from attack ground truth.
    """
    passages: List[RetrievedPassage] = []
    for rank, raw in enumerate(raw_passages):
        if doc_id_key not in raw:
            raise KeyError(
                f"Passage at rank {rank} is missing '{doc_id_key}'; poison "
                "labels must be attached at retrieval time, not inferred later."
            )
        passages.append(
            RetrievedPassage(
                doc_id=str(raw[doc_id_key]),
                text=raw[text_key],
                source=raw.get(source_key, "unknown"),
                is_poison=bool(raw.get(is_poison_key, False)),
                retrieval_score=raw.get(score_key),
                rank=rank,
            )
        )
    return passages


def filter_by_doc_ids(
    passages: Sequence[RetrievedPassage], keep_doc_ids: Sequence[str]
) -> List[RetrievedPassage]:
    """Subset of `passages` whose doc_id is in `keep_doc_ids`, order preserved."""
    keep_set = set(keep_doc_ids)
    return [p for p in passages if p.doc_id in keep_set]


def removed_passages(
    before: Sequence[RetrievedPassage], after: Sequence[RetrievedPassage]
) -> List[RetrievedPassage]:
    """Passages present in `before` but no longer present (by doc_id) in `after`."""
    kept_ids = {p.doc_id for p in after}
    return [p for p in before if p.doc_id not in kept_ids]


def texts(passages: Sequence[RetrievedPassage]) -> List[str]:
    return [p.text for p in passages]


def doc_ids(passages: Sequence[RetrievedPassage]) -> List[str]:
    return [p.doc_id for p in passages]


def poison_flags(passages: Sequence[RetrievedPassage]) -> List[bool]:
    return [bool(p.is_poison) for p in passages]


def count_poison_clean(passages: Sequence[RetrievedPassage]) -> Tuple[int, int]:
    """Return (n_poison, n_clean) for a list of passages."""
    n_poison = sum(1 for p in passages if p.is_poison)
    n_clean = len(passages) - n_poison
    return n_poison, n_clean
