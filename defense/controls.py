"""Diagnostic-only defense controls: oracle and random removal.

IMPORTANT: Neither function in this module is a deployable defense.

- `oracle_remove_all_poison` uses ground-truth poison labels that a real
  defense could never observe at inference time. It exists purely as an
  upper-bound diagnostic control: if removing every poisoned passage
  perfectly still fails to drop ASR, the problem is not detection at all
  (see prompt construction / answer matching / clean-evidence quality).
- `random_remove_same_count` removes an arbitrary N passages with no
  signal whatsoever about which are poisoned. It exists purely to test
  whether RAGDefender's targeted removals are any better than chance.

Both are intended only for `results/diagnostics/ragdefender/` runs and must
never be presented as, or substituted for, a real deployable defense.
"""
from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Optional, Sequence, Tuple

from defense.passages import RetrievedPassage

ORACLE_NOTE = "oracle_remove_all_poison: diagnostic control, not a deployable defense."
RANDOM_NOTE_TEMPLATE = (
    "random_remove_same_count(base_seed={base_seed}, query_id={query_id!r}, "
    "effective_seed={effective_seed}): diagnostic control, not a deployable defense."
)


def stable_seed_for_query(base_seed: int, query_id: str) -> int:
    """Derive a per-query random seed from a base seed and a query id.

    `random_remove_same_count` previously reused the same bare `seed` for
    every query in a run. Since `random.Random(seed).sample(range(n), k)` is
    a deterministic function of `(seed, n, k)`, and most queries in a given
    sweep share the same passage-list length and the same estimated removal
    count, this meant the *same relative indices* were removed for every
    query -- e.g. always the first `k` positions -- which is not actually a
    random baseline once passages have a consistent ordering (such as
    poison-before-clean from score sorting).

    This hashes `(base_seed, query_id)` into a large, deterministic-but-
    query-varying integer so each query gets its own independent-looking
    draw, while the whole run stays fully reproducible from `base_seed`
    alone (same base_seed + same query_id always yields the same derived
    seed).
    """
    digest = hashlib.sha256(f"{base_seed}:{query_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def oracle_remove_all_poison(
    passages: Sequence[RetrievedPassage],
) -> Tuple[List[RetrievedPassage], Dict]:
    """DIAGNOSTIC CONTROL, NOT A DEPLOYABLE DEFENSE.

    Removes every passage labeled `is_poison=True` using ground-truth attack
    labels. This is an upper-bound oracle: no real defense has access to
    this information at inference time.

    Returns (kept_passages, diag_extra); diag_extra's
    `N_adv_estimated_by_ragdefender` is the true poison count (the oracle's
    "estimate" is exact by construction).
    """
    kept = [p for p in passages if not p.is_poison]
    n_poison = sum(1 for p in passages if p.is_poison)
    diag_extra = {
        "N_adv_estimated_by_ragdefender": n_poison,
        "notes": ORACLE_NOTE,
    }
    return kept, diag_extra


def random_remove_same_count(
    passages: Sequence[RetrievedPassage],
    n_to_remove: int,
    seed: int = 12,
    query_id: Optional[str] = None,
) -> Tuple[List[RetrievedPassage], Dict]:
    """DIAGNOSTIC CONTROL, NOT A DEPLOYABLE DEFENSE.

    Removes exactly `n_to_remove` passages chosen uniformly at random, with
    no signal about which passages are poisoned. Used to test whether
    RAGDefender's targeted removals beat blind removal of the same number of
    passages. `n_to_remove` is clamped to [0, len(passages)].

    If `query_id` is given, the effective RNG seed is derived from
    `stable_seed_for_query(seed, query_id)` so each query in a run gets an
    independent-looking draw instead of always removing the same relative
    positions (see `stable_seed_for_query` docstring for why this matters).
    If `query_id` is omitted, `seed` is used directly as before (kept for
    backward compatibility / unit tests that don't care about per-query
    variation).
    """
    n = max(0, min(int(n_to_remove), len(passages)))
    effective_seed = seed if query_id is None else stable_seed_for_query(seed, query_id)
    rng = random.Random(effective_seed)
    remove_indices = set(rng.sample(range(len(passages)), n)) if n > 0 else set()
    kept = [p for i, p in enumerate(passages) if i not in remove_indices]
    diag_extra = {
        "N_adv_estimated_by_ragdefender": n_to_remove,
        "notes": RANDOM_NOTE_TEMPLATE.format(
            base_seed=seed, query_id=query_id, effective_seed=effective_seed
        ),
    }
    return kept, diag_extra
