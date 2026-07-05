"""LLM-generation-independent diagnostic logging for RAGDefender experiments.

This module builds and persists one JSONL record per query describing what
was retrieved, what the defense estimated/removed, and (only if available)
how generation behaved with/without the defense. Every detection-quality
field (retrieved/removed poison & clean counts, poison recall, clean false
positive rate, residual poison fraction, RAGDefender's estimated adversarial
count) is derived purely from passage lists and never requires an LLM call.
Generation-dependent fields (answers, ASR) are left as `None` when the
caller ran in `--dry_run` mode or otherwise skipped generation.

Do not add fields here that require re-deriving poison status from text --
poison labels must already be attached to the `RetrievedPassage` objects
passed in (see defense/passages.py).
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

from defense.asr_match import legacy_match, strict_match
from defense.passages import RetrievedPassage, count_poison_clean, doc_ids, poison_flags, removed_passages

# Canonical schema -- keep in sync with docs/RAGDEFENDER_DIAGNOSTIC_PLAN.md.
DIAGNOSTIC_FIELDS = (
    "query_id",
    "dataset",
    "model",
    "attack",
    "defense",
    "k",
    "N_injected",
    "retrieved_doc_ids",
    "retrieved_is_poison",
    "N_retrieved_poison",
    "N_retrieved_clean",
    "N_adv_estimated_by_ragdefender",
    "removed_doc_ids",
    "removed_is_poison",
    "removed_poison",
    "removed_clean",
    "poison_recall",
    "clean_false_positive_rate",
    "residual_poison_count",
    "residual_clean_count",
    "residual_poison_fraction",
    "answer_no_defense",
    "answer_with_defense",
    "target_wrong_answer",
    "gold_answer",
    "asr_no_defense",
    "asr_with_defense",
    # asr_no_defense/asr_with_defense above are the *legacy* substring-match
    # ASR flags computed by the caller (main.py), preserved byte-for-byte
    # for backward compatibility. The four fields below are computed
    # internally by build_diagnostic_record() from
    # answer_{no_defense,with_defense}/target_wrong_answer using
    # defense/asr_match.py: *_legacy re-derives the same substring check
    # (should always agree with asr_no_defense/asr_with_defense above) and
    # *_strict uses strict token-boundary ASR instead (a standalone
    # yes/no token, or an exact token-subsequence match -- not a semantic
    # yes/no evaluator; see defense/asr_match.py module docstring for its
    # negation-detection limitation and for why legacy substring matching
    # can false-positive, e.g. target "no" matching inside "does NOt
    # provide...").
    "asr_no_defense_legacy",
    "asr_with_defense_legacy",
    "asr_no_defense_strict",
    "asr_with_defense_strict",
    "latency_retrieval_sec",
    "latency_defense_sec",
    "latency_generation_sec",
    "notes",
)

# Fields that require an LLM call to populate; explicitly None in --dry_run.
GENERATION_DEPENDENT_FIELDS = (
    "answer_no_defense",
    "answer_with_defense",
    "target_wrong_answer",
    "gold_answer",
    "asr_no_defense",
    "asr_with_defense",
    "asr_no_defense_legacy",
    "asr_with_defense_legacy",
    "asr_no_defense_strict",
    "asr_with_defense_strict",
    "latency_generation_sec",
)

# Fields that must always be computable without any LLM call.
DETECTION_ONLY_FIELDS = tuple(f for f in DIAGNOSTIC_FIELDS if f not in GENERATION_DEPENDENT_FIELDS)


@contextmanager
def timer():
    """Context manager yielding a dict whose 'elapsed_sec' is filled in on exit.

    Usage:
        with timer() as t:
            do_work()
        t["elapsed_sec"]  # float seconds
    """
    result: Dict[str, Optional[float]] = {"elapsed_sec": None}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_sec"] = time.perf_counter() - start


def _safe_div(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def build_diagnostic_record(
    *,
    query_id: str,
    dataset: str,
    model: str,
    attack: str,
    defense: str,
    k: int,
    N_injected: int,
    retrieved_passages: Sequence[RetrievedPassage],
    kept_passages: Sequence[RetrievedPassage],
    N_adv_estimated_by_ragdefender: Optional[int] = None,
    answer_no_defense: Optional[str] = None,
    answer_with_defense: Optional[str] = None,
    target_wrong_answer: Optional[str] = None,
    gold_answer: Optional[str] = None,
    asr_no_defense: Optional[bool] = None,
    asr_with_defense: Optional[bool] = None,
    latency_retrieval_sec: Optional[float] = None,
    latency_defense_sec: Optional[float] = None,
    latency_generation_sec: Optional[float] = None,
    notes: str = "",
) -> Dict:
    """Build one diagnostic record.

    Only `retrieved_passages` (pre-defense) and `kept_passages` (post-defense)
    are required to populate every detection-quality field. All
    generation-dependent kwargs default to None so this function -- and
    therefore full detection diagnostics -- works identically whether or not
    generation ran (e.g. under --dry_run).

    `asr_no_defense`/`asr_with_defense` are taken as-is from the caller
    (main.py's existing legacy substring-match flags, preserved for
    backward compatibility). `asr_{no_defense,with_defense}_legacy` and
    `asr_{no_defense,with_defense}_strict` are derived internally, purely
    from `target_wrong_answer` and `answer_{no_defense,with_defense}`, via
    `defense/asr_match.py` -- callers don't need to compute these
    themselves.
    """
    removed = removed_passages(retrieved_passages, kept_passages)

    n_retrieved_poison, n_retrieved_clean = count_poison_clean(retrieved_passages)
    n_removed_poison, n_removed_clean = count_poison_clean(removed)

    residual_poison_count = n_retrieved_poison - n_removed_poison
    residual_clean_count = n_retrieved_clean - n_removed_clean

    record: Dict = {
        "query_id": query_id,
        "dataset": dataset,
        "model": model,
        "attack": attack,
        "defense": defense,
        "k": k,
        "N_injected": N_injected,
        "retrieved_doc_ids": doc_ids(retrieved_passages),
        "retrieved_is_poison": poison_flags(retrieved_passages),
        "N_retrieved_poison": n_retrieved_poison,
        "N_retrieved_clean": n_retrieved_clean,
        "N_adv_estimated_by_ragdefender": N_adv_estimated_by_ragdefender,
        "removed_doc_ids": doc_ids(removed),
        "removed_is_poison": poison_flags(removed),
        "removed_poison": n_removed_poison,
        "removed_clean": n_removed_clean,
        "poison_recall": _safe_div(n_removed_poison, n_retrieved_poison),
        "clean_false_positive_rate": _safe_div(n_removed_clean, n_retrieved_clean),
        "residual_poison_count": residual_poison_count,
        "residual_clean_count": residual_clean_count,
        "residual_poison_fraction": _safe_div(
            residual_poison_count, residual_poison_count + residual_clean_count
        ),
        "answer_no_defense": answer_no_defense,
        "answer_with_defense": answer_with_defense,
        "target_wrong_answer": target_wrong_answer,
        "gold_answer": gold_answer,
        "asr_no_defense": asr_no_defense,
        "asr_with_defense": asr_with_defense,
        "asr_no_defense_legacy": legacy_match(target_wrong_answer, answer_no_defense),
        "asr_with_defense_legacy": legacy_match(target_wrong_answer, answer_with_defense),
        "asr_no_defense_strict": strict_match(target_wrong_answer, answer_no_defense),
        "asr_with_defense_strict": strict_match(target_wrong_answer, answer_with_defense),
        "latency_retrieval_sec": latency_retrieval_sec,
        "latency_defense_sec": latency_defense_sec,
        "latency_generation_sec": latency_generation_sec,
        "notes": notes,
    }
    return record


def validate_record(record: Dict) -> List[str]:
    """Return the list of schema fields missing from `record` (empty if valid)."""
    return [f for f in DIAGNOSTIC_FIELDS if f not in record]


def default_diagnostics_path(run_name: str, base_dir: str = "results/diagnostics/ragdefender") -> str:
    return os.path.join(base_dir, f"{run_name}.jsonl")


def append_jsonl(record: Dict, path: str) -> None:
    """Append a single JSON record as one line to `path`, creating dirs as needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: str) -> List[Dict]:
    """Read all JSON records from a JSONL file."""
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
