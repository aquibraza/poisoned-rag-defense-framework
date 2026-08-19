"""RobustRAG-KW: a generation-time isolate-then-aggregate defense proxy.

**This is not certified RobustRAG.** It is a *proxy* inspired by RobustRAG
(Xiang et al., "Certifiably Robust RAG against Retrieval Corruption") that
captures the isolate-then-aggregate design pattern -- isolate each retrieved
passage into its own generator call, extract a short answer from each, then
aggregate the candidates by vote -- **without** reproducing RobustRAG's
certified decoding procedure. It computes no certificate, provides no
certifiable robustness guarantee, and performs no bounded-corruption proof.
Every artifact produced from this module must say "RobustRAG-KW proxy".

Why this is a separate module rather than a `defense/dispatch.py` case
--------------------------------------------------------------------
Every defense routed through `defense.dispatch.run_defense()` returns
`(List[RetrievedPassage], Dict)` -- a *filtered subset of passages*, which
`main.py` then flattens into a single prompt for one generation call.
RobustRAG-KW removes no passages, returns an *answer*, and must call the
generator N times internally. Expressing it as a `run_defense()` case would
corrupt that contract, so it is deliberately kept standalone:
`defense/dispatch.py`, `DEFENSE_CHOICES`, and `main.py` are untouched, and
`robustrag_kw` is intentionally **not** a member of `DEFENSE_CHOICES`.
`tests/test_robustrag_kw.py` pins this.

Dependency policy
-----------------
This module is **stdlib-only** (plus `defense.passages`, `defense.asr_match`,
and `src.prompts`, all of which are themselves stdlib-only). No torch, no
transformers, no sentence-transformers, no numpy. This keeps the whole test
file dependency-free -- runnable on a bare system `python3`, like
`tests/test_asr_match.py` -- and keeps the vote logic auditable in isolation.

For the same reason `defense/asr_match.py` documents for duplicating
`clean_str` locally, `normalize_answer(..., mode="squad")` reimplements the
SQuAD-style normalization from `src/contriever_src/evaluation.py` with the
stdlib `re` instead of the third-party `regex`. `tests/test_robustrag_kw.py`
cross-checks the two agree whenever that module happens to be importable.

No API calls
------------
This module never imports `src.models`, never calls `create_model()`, and
never calls `llm.query()`. The generator is injected by the caller as
`generate_fn: Callable[[str], Optional[str]]` -- the same seam
`defense/filterrag.py` uses for `slm_answer_fn`. That makes "no API calls in
unit tests" a structural property rather than an observed one, and it is what
lets `GenerationCache` make the aggregation/abstention sweeps free.

Ground-truth labels
-------------------
`RetrievedPassage.is_poison` is attack-injection ground truth and is
**diagnostics-only** (see `defense/passages.py`). It is never read while
deciding an answer. The poison/clean/self-query/cross-query flags on
`IsolatedAnswer` are attached for reporting only, after the vote has already
been decided, exactly like the rest of this repo's diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from defense.asr_match import _legacy_clean_str, strict_match
from defense.passages import RetrievedPassage
from src.prompts import wrap_prompt

__all__ = [
    "ABSTAIN_ANSWER",
    "ABSTENTION_POLICIES",
    "NORMALIZATION_MODES",
    "AGGREGATION_MODES",
    "TIE_BREAKERS",
    "RobustRagKwConfig",
    "IsolatedAnswer",
    "RobustRagKwResult",
    "CacheKey",
    "GenerationCache",
    "prompt_hash",
    "build_groups",
    "extract_short_answer",
    "normalize_answer",
    "is_abstention",
    "aggregate_votes",
    "decide",
    "robustrag_kw_answer",
]

ABSTAIN_ANSWER = "I don't know"

ABSTENTION_POLICIES: Tuple[str, ...] = ("discard_abstentions", "include_abstentions")
NORMALIZATION_MODES: Tuple[str, ...] = ("raw", "legacy_clean", "squad", "token")
AGGREGATION_MODES: Tuple[str, ...] = ("exact", "keyword")
TIE_BREAKERS: Tuple[str, ...] = ("abstain", "first_rank")

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_PUNCT_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]")
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")

# Responses that are *entirely* a refusal.
ABSTENTION_EXACT: Tuple[str, ...] = (
    "unknown",
    "no answer",
    "no context",
    "n/a",
)

# Substrings that mark a generator response as "no answer". This is a superset
# of `scripts/run_answer_generation_smoke_bundle1.py::_UNCERTAIN_PHRASES`; a
# unit test asserts every phrase the smoke script recognizes is also caught
# here, so the canonical detector can never become weaker than the published
# one. Bare "unknown" is deliberately exact-match only (see ABSTENTION_EXACT)
# rather than a substring, so it cannot fire inside an ordinary factual
# sentence such as "the cause is unknown to historians".
ABSTENTION_PHRASES: Tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "i dont know",
    "don't know",
    "do not know",
    "cannot find",
    "can not find",
    "can't find",
    "insufficient context",
    "insufficient information",
    "not enough information",
    "not enough context",
    "no relevant context",
    "no relevant information",
    "no information",
    "not mention",  # covers "not mentioned" and "does not mention"
    "unable to determine",
)

DEFAULT_MAX_ANSWER_TOKENS = 12


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobustRagKwConfig:
    """RobustRAG-KW proxy configuration.

    `abstention_policy` is the single most consequential knob:

    - "discard_abstentions" (default, RobustRAG-faithful): the vote
      denominator counts only non-abstaining isolated answers, so passages
      that say "I don't know" drop out entirely.
    - "include_abstentions": the denominator is the total number of isolated
      groups, so silent passages count *against* the winner.

    A poison bloc that outvotes a silent clean majority wins outright under the
    first policy and can fall below `abstain_threshold` under the second. The
    pilot reports both; neither is "the" result.
    """

    group_size: int = 1
    vote_threshold: float = 0.5
    abstain_threshold: float = 0.0
    normalization_mode: str = "squad"
    abstention_policy: str = "discard_abstentions"
    aggregation_mode: str = "exact"
    tie_breaker: str = "abstain"
    max_isolated_calls: int = 16
    max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if self.max_isolated_calls <= 0:
            raise ValueError(
                f"max_isolated_calls must be >= 1, got {self.max_isolated_calls}"
            )
        if self.normalization_mode not in NORMALIZATION_MODES:
            raise ValueError(
                f"normalization_mode must be one of {NORMALIZATION_MODES}, "
                f"got {self.normalization_mode!r}"
            )
        if self.abstention_policy not in ABSTENTION_POLICIES:
            raise ValueError(
                f"abstention_policy must be one of {ABSTENTION_POLICIES}, "
                f"got {self.abstention_policy!r}"
            )
        if self.aggregation_mode not in AGGREGATION_MODES:
            raise ValueError(
                f"aggregation_mode must be one of {AGGREGATION_MODES}, "
                f"got {self.aggregation_mode!r}"
            )
        if self.tie_breaker not in TIE_BREAKERS:
            raise ValueError(
                f"tie_breaker must be one of {TIE_BREAKERS}, got {self.tie_breaker!r}"
            )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class IsolatedAnswer:
    """One isolated generator call and everything known about it.

    Group-level semantics: when `group_size > 1`, the poison flags are
    *disjunctive over the group* -- `is_poison=True` means "this group
    contained at least one poisoned passage", not "this group was entirely
    poison". `doc_ids`/`ranks` are lists precisely so the composition stays
    recoverable from the record.

    `raw_answer`, `extracted_answer`, and `normalized_answer` are stored
    separately and never overwrite each other, so extraction and normalization
    can both be re-derived offline from `raw_answer` without regenerating.
    """

    group_index: int
    doc_ids: List[str]
    ranks: List[Optional[int]]
    prompt: str
    prompt_hash: str
    model_name: str
    query_id: str
    context_type: str
    cache_hit: bool
    raw_answer: Optional[str]
    extracted_answer: Optional[str]
    normalized_answer: Optional[str]
    is_clean: bool
    is_poison: bool
    is_self_query_poison: bool
    is_cross_query_poison: bool
    matches_target_wrong_answer_strict: Optional[bool]
    matches_correct_answer_strict: Optional[bool]
    is_abstain: bool
    origin_label: Optional[str] = None
    true_owning_query_id: Optional[str] = None
    true_global_index: Optional[int] = None
    mutation_family: Optional[str] = None
    is_mutated: Optional[bool] = None

    def validate(self) -> None:
        """Assert the amendment-4 flag invariants. Raises ValueError."""
        if self.is_clean == self.is_poison:
            raise ValueError(
                f"group_index={self.group_index}: is_clean and is_poison must be "
                f"opposites, got is_clean={self.is_clean}, is_poison={self.is_poison}"
            )
        if self.is_self_query_poison and self.is_cross_query_poison:
            raise ValueError(
                f"group_index={self.group_index}: is_self_query_poison and "
                "is_cross_query_poison are mutually exclusive"
            )
        if self.is_clean and (self.is_self_query_poison or self.is_cross_query_poison):
            raise ValueError(
                f"group_index={self.group_index}: a clean group cannot carry a "
                "poison-origin flag"
            )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RobustRagKwResult:
    final_answer: str
    abstained: bool
    winning_normalized_answer: Optional[str]
    winning_vote_count: int
    winning_vote_share: Optional[float]
    vote_margin: Optional[float]
    vote_counts: "OrderedDict[str, int]"
    denominator: int
    n_isolated_calls: int
    n_cache_hits: int
    n_abstentions: int
    isolated_answers: List[IsolatedAnswer]
    config: Dict
    latency_sec: Optional[float] = None

    def to_dict(self) -> Dict:
        out = asdict(self)
        out["vote_counts"] = dict(self.vote_counts)
        return out


# ---------------------------------------------------------------------------
# Generation cache (keyed on model + prompt only, so sweeps are free)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheKey:
    prompt_hash: str
    model_name: str


def prompt_hash(prompt: str, model_name: str) -> str:
    """Stable sha256 over (model_name, prompt).

    NUL-separated so no model-name/prompt pair can collide with another by
    concatenation. Stable across processes (unlike Python's salted `hash()`).
    """
    payload = f"{model_name}\x00{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class GenerationCache:
    """Append-only JSONL cache of raw generator outputs.

    Keyed on `(model_name, prompt_hash)` **only** -- deliberately not on
    normalization or aggregation settings, since those are applied downstream.
    That is exactly what lets the aggregation/abstention sweeps re-run at zero
    API cost.

    A cache hit whose recorded `query_id` disagrees with the current call is a
    hard error, not a silent reuse. Because a prompt embeds the question, two
    different queries cannot produce the same prompt, so such a collision can
    only mean the prompt builder is wrong -- worth failing on rather than
    papering over.

    `context_type` is deliberately **not** part of that check. In this pilot
    only the self-query poison passages were mutated, so a clean passage
    produces a byte-identical isolated prompt under both the `original` and
    `mutated` contexts. Reusing one generation for both is correct (identical
    prompt, identical model) and avoids paying twice for the same call; the
    entry records every context it was observed under in `context_types`, and
    the consuming record still reports `cache_hit` so the provenance stays
    visible.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._entries: "OrderedDict[Tuple[str, str], Dict]" = OrderedDict()
        self._pending: List[Dict] = []

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _key_tuple(key: CacheKey) -> Tuple[str, str]:
        return (key.model_name, key.prompt_hash)

    def load(self) -> "GenerationCache":
        """Read the JSONL file if it exists. Absent file is not an error."""
        if not self.path or not os.path.exists(self.path):
            return self
        with open(self.path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.path}:{line_no} is not valid JSON: {exc}"
                    ) from exc
                self._entries[(rec["model_name"], rec["prompt_hash"])] = rec
        return self

    def get(self, key: CacheKey, *, query_id: Optional[str] = None,
            context_type: Optional[str] = None) -> Optional[str]:
        rec = self._entries.get(self._key_tuple(key))
        if rec is None:
            return None
        recorded_qid = rec.get("query_id")
        if query_id is not None and recorded_qid is not None and recorded_qid != query_id:
            raise ValueError(
                f"Cache collision on prompt_hash={key.prompt_hash[:12]}...: "
                f"cached query_id={recorded_qid!r} but this call has "
                f"{query_id!r}. A prompt embeds its question, so identical "
                "prompts for different queries means the prompt builder is "
                "wrong; refusing to reuse."
            )
        if context_type is not None:
            seen = rec.setdefault("context_types", [])
            if context_type not in seen:
                seen.append(context_type)
        return rec.get("raw_answer")

    def put(self, key: CacheKey, raw_answer: Optional[str], meta: Dict) -> None:
        rec = dict(meta)
        rec.update({
            "prompt_hash": key.prompt_hash,
            "model_name": key.model_name,
            "raw_answer": raw_answer,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        rec.setdefault("context_types",
                       [meta["context_type"]] if meta.get("context_type") else [])
        self._entries[self._key_tuple(key)] = rec
        self._pending.append(rec)

    def flush(self) -> int:
        """Append pending records to the JSONL file. Returns count written."""
        if not self.path or not self._pending:
            n = len(self._pending)
            self._pending = []
            return n
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            for rec in self._pending:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n = len(self._pending)
        self._pending = []
        return n


def raising_generate_fn(_prompt: str) -> str:
    """A `generate_fn` that always raises.

    Installed by `--dry_run` and `--sweep_only` so that "zero API calls" is
    structurally enforced: a cache miss becomes a loud failure instead of a
    silent charge.
    """
    raise AssertionError(
        "generate_fn was invoked while generation was disabled (dry-run or "
        "sweep-only). This means a prompt was not present in the cache."
    )


# ---------------------------------------------------------------------------
# Pure helpers: grouping, extraction, normalization, abstention
# ---------------------------------------------------------------------------

def build_groups(
    passages: Sequence[RetrievedPassage], group_size: int
) -> List[List[RetrievedPassage]]:
    """Split rank-ordered passages into consecutive chunks of `group_size`.

    `group_size=1` is single-passage isolation (RobustRAG-faithful, maximum
    isolation). `group_size >= len(passages)` yields exactly one group, which
    is mechanically the no-defense condition and is useful as a sanity control.
    Nothing is dropped or duplicated; the final group may be short.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    return [
        list(passages[i:i + group_size])
        for i in range(0, len(passages), group_size)
    ]


def normalize_answer(text: Optional[str], mode: str = "squad") -> Optional[str]:
    """Normalize a candidate answer for vote-key comparison.

    - "raw": passthrough (control; shows how much normalization matters).
    - "legacy_clean": byte-identical to `defense.asr_match._legacy_clean_str`.
    - "squad": lowercase, strip punctuation, drop articles, collapse
      whitespace -- same semantics as
      `src/contriever_src/evaluation.py::normalize_answer`, reimplemented with
      the stdlib `re` to keep this module dependency-free.
    - "token": lowercase `[a-z0-9']+` tokens rejoined by single spaces,
      matching `defense.asr_match._tokenize`, so vote keys are directly
      comparable to strict-ASR tokenization.
    """
    if mode not in NORMALIZATION_MODES:
        raise ValueError(
            f"mode must be one of {NORMALIZATION_MODES}, got {mode!r}"
        )
    if text is None:
        return None
    if mode == "raw":
        return text
    if mode == "legacy_clean":
        return _legacy_clean_str(text)
    if mode == "token":
        return " ".join(_TOKEN_RE.findall(text.lower()))
    lowered = text.lower()
    no_punct = _PUNCT_RE.sub("", lowered)
    no_articles = _ARTICLE_RE.sub(" ", no_punct)
    return " ".join(no_articles.split())


def extract_short_answer(
    raw: Optional[str], max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS
) -> Optional[str]:
    """Pull a short answer candidate out of a free-form generator response.

    Heuristic by construction (see this module's risk notes): takes the first
    non-empty line, strips a leading "Answer:" label, and truncates to
    `max_answer_tokens` whitespace tokens. A leading "Yes,"/"No," is preserved
    rather than stripped, because for yes/no targets that token *is* the
    answer.

    Returns None for None/empty input. The raw response is always stored
    alongside this, so a bad extraction can be re-derived offline without
    regenerating.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    text = re.sub(r"^\s*(answer|final answer)\s*[:\-]\s*", "", text,
                  flags=re.IGNORECASE)
    # First sentence only: split on a period that ends a sentence, but keep
    # short yes/no leads ("No, Ferocactus is...") intact by taking the whole
    # first sentence rather than the first comma-clause.
    match = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    if match:
        text = match[0]
    text = text.strip()
    if not text:
        return None
    tokens = text.split()
    if len(tokens) > max_answer_tokens:
        tokens = tokens[:max_answer_tokens]
    return " ".join(tokens).strip()


def is_abstention(raw: Optional[str]) -> bool:
    """True when the generator declined to answer.

    Empty/None output counts as an abstention here. This deliberately differs
    from `run_answer_generation_smoke_bundle1.is_no_answer_or_uncertain`, which
    returns False for None because it is labelling an already-generated answer.
    For voting, a missing response must not be allowed to become a vote, so it
    is treated as an abstention instead.

    The PoisonedRAG prompt template explicitly instructs the model to say
    "I don't know", which is what makes this detectable without a custom prompt.
    """
    if raw is None:
        return True
    text = _legacy_clean_str(raw)
    if not text:
        return True
    if text in ABSTENTION_EXACT:
        return True
    return any(phrase in text for phrase in ABSTENTION_PHRASES)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _keyword_merge(counts: "OrderedDict[str, int]") -> "OrderedDict[str, int]":
    """Pool votes whose token sets are subsets of a longer candidate's.

    Purely lexical containment -- no embeddings. Embedding similarity is
    deliberately excluded: it would merge "world's best defender" and
    "world's best goalkeeper", near-identical strings with opposite meanings,
    which is exactly one of this pilot's cases.

    The absorbing (longer) candidate keeps the pooled count. Ties in length are
    resolved by first-seen order so the result is deterministic.
    """
    keys = list(counts.keys())
    position = {k: i for i, k in enumerate(keys)}
    token_sets = {k: set(k.split()) for k in keys}

    # Each key is absorbed into the *longest* candidate that is a superset of
    # it; length ties go to the earliest-seen, so the mapping is deterministic
    # and a subset chain (a < b < c) collapses onto c rather than onto a peer.
    canonical: Dict[str, str] = {}
    for key in keys:
        best = key
        if not token_sets[key]:
            canonical[key] = best
            continue
        for other in keys:
            if other == key or not token_sets[other]:
                continue
            if not token_sets[key].issubset(token_sets[other]):
                continue
            if (len(token_sets[other]), -position[other]) > (
                len(token_sets[best]), -position[best]
            ):
                best = other
        canonical[key] = best

    merged: "OrderedDict[str, int]" = OrderedDict()
    for key in keys:
        target = canonical[key]
        merged[target] = merged.get(target, 0) + counts[key]
    return OrderedDict(
        sorted(merged.items(), key=lambda kv: (-kv[1], position[kv[0]]))
    )


def aggregate_votes(
    normalized_answers: Sequence[Optional[str]],
    abstentions: Sequence[bool],
    *,
    aggregation_mode: str = "exact",
) -> "OrderedDict[str, int]":
    """Count votes over non-abstaining, non-empty normalized answers.

    Ordering is deterministic: descending count, then first-seen order. This is
    explicit rather than relying on `Counter.most_common`/dict iteration, so
    results never depend on insertion-hash behavior.
    """
    if aggregation_mode not in AGGREGATION_MODES:
        raise ValueError(
            f"aggregation_mode must be one of {AGGREGATION_MODES}, "
            f"got {aggregation_mode!r}"
        )
    if len(normalized_answers) != len(abstentions):
        raise ValueError(
            "normalized_answers and abstentions must be the same length, got "
            f"{len(normalized_answers)} and {len(abstentions)}"
        )
    first_seen: List[str] = []
    raw_counts: Counter = Counter()
    for answer, abstained in zip(normalized_answers, abstentions):
        if abstained or answer is None or not str(answer).strip():
            continue
        key = str(answer)
        if key not in raw_counts:
            first_seen.append(key)
        raw_counts[key] += 1
    ordered: "OrderedDict[str, int]" = OrderedDict(
        (k, raw_counts[k]) for k in sorted(
            first_seen, key=lambda k: (-raw_counts[k], first_seen.index(k))
        )
    )
    if aggregation_mode == "keyword":
        ordered = _keyword_merge(ordered)
    return ordered


def decide(
    vote_counts: "OrderedDict[str, int]",
    *,
    n_counted: int,
    n_groups: int,
    config: RobustRagKwConfig,
) -> Tuple[str, bool, Optional[str], int, Optional[float], Optional[float], int]:
    """Turn vote counts into a final answer.

    Returns
    `(final_answer, abstained, winner, winner_count, winner_share, margin,
    denominator)`.

    The denominator follows `config.abstention_policy`:
    "discard_abstentions" uses the number of counted (non-abstaining) votes;
    "include_abstentions" uses the total number of isolated groups, so silent
    passages count against the winner.
    """
    denominator = (
        n_counted if config.abstention_policy == "discard_abstentions" else n_groups
    )
    if not vote_counts or denominator <= 0:
        return ABSTAIN_ANSWER, True, None, 0, None, None, max(denominator, 0)

    items = list(vote_counts.items())
    winner, winner_count = items[0]
    runner_up_count = items[1][1] if len(items) > 1 else 0
    winner_share = winner_count / denominator
    margin = (winner_count - runner_up_count) / denominator

    is_tie = len(items) > 1 and runner_up_count == winner_count
    if is_tie and config.tie_breaker == "abstain":
        return ABSTAIN_ANSWER, True, winner, winner_count, winner_share, margin, denominator

    if winner_share >= config.vote_threshold and winner_share >= config.abstain_threshold:
        return winner, False, winner, winner_count, winner_share, margin, denominator
    return ABSTAIN_ANSWER, True, winner, winner_count, winner_share, margin, denominator


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _group_flags(group: Sequence[RetrievedPassage],
                 origin_by_doc_id: Optional[Dict[str, Dict]]) -> Dict:
    """Derive the diagnostic flag set for one group (disjunctive over members).

    Ground-truth only, used for reporting after the fact -- never consulted
    while deciding an answer.
    """
    is_poison = any(bool(p.is_poison) for p in group)
    origin_labels: List[str] = []
    owning_ids: List[Optional[str]] = []
    global_indices: List[Optional[int]] = []
    families: List[Optional[str]] = []
    mutated_flags: List[Optional[bool]] = []
    for p in group:
        meta = (origin_by_doc_id or {}).get(p.doc_id) or {}
        if meta.get("origin_label"):
            origin_labels.append(str(meta["origin_label"]))
        owning_ids.append(meta.get("true_owning_query_id"))
        global_indices.append(meta.get("true_global_index"))
        families.append(meta.get("mutation_family"))
        mutated_flags.append(meta.get("is_mutated"))

    self_query = any(lbl == "mutated_self_query_poison" or lbl == "self_query_poison"
                     for lbl in origin_labels)
    cross_query = any(lbl == "cross_query_poison" for lbl in origin_labels)
    # Self-query dominates a mixed group: the attacker's bloc for *this*
    # question is the effect under test, so a group containing it is counted
    # there rather than as cross-query.
    if self_query and cross_query:
        cross_query = False

    def _first(values: Sequence) -> Optional[object]:
        for v in values:
            if v is not None:
                return v
        return None

    return {
        "is_poison": is_poison,
        "is_clean": not is_poison,
        "is_self_query_poison": bool(self_query),
        "is_cross_query_poison": bool(cross_query),
        "origin_label": origin_labels[0] if origin_labels else (
            None if is_poison else "clean"
        ),
        "true_owning_query_id": _first(owning_ids),
        "true_global_index": _first(global_indices),
        "mutation_family": _first(families),
        "is_mutated": _first(mutated_flags),
    }


def robustrag_kw_answer(
    query: str,
    passages: Sequence[RetrievedPassage],
    *,
    generate_fn: Callable[[str], Optional[str]],
    config: Optional[RobustRagKwConfig] = None,
    cache: Optional[GenerationCache] = None,
    model_name: str = "unknown",
    query_id: str = "",
    context_type: str = "",
    target_wrong_answer: Optional[str] = None,
    correct_answer: Optional[str] = None,
    origin_by_doc_id: Optional[Dict[str, Dict]] = None,
) -> RobustRagKwResult:
    """Isolate each passage (or small group), generate, then aggregate by vote.

    `generate_fn` is called at most once per group, and not at all for groups
    already present in `cache`. Every prompt is built with the repo's canonical
    `wrap_prompt(..., prompt_id=4)` so an isolated prompt is byte-identical in
    *form* to the full-context prompt -- any ASR difference is then
    attributable to isolation rather than prompt drift.

    Raises ValueError if the number of groups exceeds
    `config.max_isolated_calls` (a hard cap, never a silent truncation).
    """
    cfg = config or RobustRagKwConfig()
    groups = build_groups(passages, cfg.group_size)
    if len(groups) > cfg.max_isolated_calls:
        raise ValueError(
            f"{len(groups)} isolated calls required (len(passages)="
            f"{len(passages)}, group_size={cfg.group_size}) but "
            f"max_isolated_calls={cfg.max_isolated_calls}. Raise the cap "
            "explicitly rather than silently truncating the context."
        )

    started = time.time()
    isolated: List[IsolatedAnswer] = []
    n_calls = 0
    n_cache_hits = 0

    for group_index, group in enumerate(groups):
        group_texts = [p.text for p in group]
        prompt = wrap_prompt(query, group_texts, prompt_id=4)
        phash = prompt_hash(prompt, model_name)
        key = CacheKey(prompt_hash=phash, model_name=model_name)

        raw: Optional[str] = None
        cache_hit = False
        if cache is not None:
            raw = cache.get(key, query_id=query_id or None,
                            context_type=context_type or None)
            cache_hit = raw is not None
        if not cache_hit:
            raw = generate_fn(prompt)
            n_calls += 1
        else:
            n_cache_hits += 1

        flags = _group_flags(group, origin_by_doc_id)
        meta = {
            "query_id": query_id,
            "context_type": context_type,
            "group_index": group_index,
            "doc_ids": [p.doc_id for p in group],
            "ranks": [p.rank for p in group],
            "origin_label": flags["origin_label"],
            "true_owning_query_id": flags["true_owning_query_id"],
            "true_global_index": flags["true_global_index"],
            "mutation_family": flags["mutation_family"],
            "is_mutated": flags["is_mutated"],
        }
        if cache is not None and not cache_hit:
            cache.put(key, raw, meta)

        extracted = extract_short_answer(raw, cfg.max_answer_tokens)
        normalized = normalize_answer(extracted, cfg.normalization_mode)
        abstained = is_abstention(raw)

        record = IsolatedAnswer(
            group_index=group_index,
            doc_ids=[p.doc_id for p in group],
            ranks=[p.rank for p in group],
            prompt=prompt,
            prompt_hash=phash,
            model_name=model_name,
            query_id=query_id,
            context_type=context_type,
            cache_hit=cache_hit,
            raw_answer=raw,
            extracted_answer=extracted,
            normalized_answer=normalized,
            is_clean=flags["is_clean"],
            is_poison=flags["is_poison"],
            is_self_query_poison=flags["is_self_query_poison"],
            is_cross_query_poison=flags["is_cross_query_poison"],
            matches_target_wrong_answer_strict=strict_match(target_wrong_answer, raw),
            matches_correct_answer_strict=strict_match(correct_answer, raw),
            is_abstain=abstained,
            origin_label=flags["origin_label"],
            true_owning_query_id=flags["true_owning_query_id"],
            true_global_index=flags["true_global_index"],
            mutation_family=flags["mutation_family"],
            is_mutated=flags["is_mutated"],
        )
        record.validate()
        isolated.append(record)

    return aggregate_isolated(
        isolated,
        config=cfg,
        n_calls=n_calls,
        n_cache_hits=n_cache_hits,
        latency_sec=time.time() - started,
    )


def aggregate_isolated(
    isolated: Sequence[IsolatedAnswer],
    *,
    config: RobustRagKwConfig,
    n_calls: int = 0,
    n_cache_hits: int = 0,
    latency_sec: Optional[float] = None,
) -> RobustRagKwResult:
    """Re-run steps 3-6 over already-generated isolated answers.

    This is the zero-API-cost path used by the aggregation/abstention sweeps:
    the vote keys are recomputed from each record's stored `extracted_answer`
    under the sweep's `normalization_mode`, so a single generation run supports
    every configuration.

    The `IsolatedAnswer` records are returned unmodified, so their
    `normalized_answer` field still reflects the *generation-time* mode. The
    mode actually used for this aggregation is in the returned
    `config["normalization_mode"]`.
    """
    normalized = [
        normalize_answer(rec.extracted_answer, config.normalization_mode)
        for rec in isolated
    ]
    abstentions = [rec.is_abstain for rec in isolated]
    vote_counts = aggregate_votes(
        normalized, abstentions, aggregation_mode=config.aggregation_mode
    )
    n_counted = sum(
        1 for a, ab in zip(normalized, abstentions)
        if not ab and a is not None and str(a).strip()
    )
    (final_answer, abstained, winner, winner_count, winner_share, margin,
     denominator) = decide(
        vote_counts, n_counted=n_counted, n_groups=len(isolated), config=config
    )
    return RobustRagKwResult(
        final_answer=final_answer,
        abstained=abstained,
        winning_normalized_answer=winner,
        winning_vote_count=winner_count,
        winning_vote_share=winner_share,
        vote_margin=margin,
        vote_counts=vote_counts,
        denominator=denominator,
        n_isolated_calls=n_calls,
        n_cache_hits=n_cache_hits,
        n_abstentions=sum(1 for a in abstentions if a),
        isolated_answers=list(isolated),
        config=asdict(config),
        latency_sec=latency_sec,
    )
