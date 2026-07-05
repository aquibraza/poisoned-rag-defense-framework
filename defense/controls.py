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

import random
from typing import Dict, List, Sequence, Tuple

from defense.passages import RetrievedPassage

ORACLE_NOTE = "oracle_remove_all_poison: diagnostic control, not a deployable defense."
RANDOM_NOTE_TEMPLATE = (
    "random_remove_same_count(seed={seed}): diagnostic control, not a deployable defense."
)


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
) -> Tuple[List[RetrievedPassage], Dict]:
    """DIAGNOSTIC CONTROL, NOT A DEPLOYABLE DEFENSE.

    Removes exactly `n_to_remove` passages chosen uniformly at random (seeded
    for reproducibility), with no signal about which passages are poisoned.
    Used to test whether RAGDefender's targeted removals beat blind removal
    of the same number of passages. `n_to_remove` is clamped to
    [0, len(passages)].
    """
    n = max(0, min(int(n_to_remove), len(passages)))
    rng = random.Random(seed)
    remove_indices = set(rng.sample(range(len(passages)), n)) if n > 0 else set()
    kept = [p for i, p in enumerate(passages) if i not in remove_indices]
    diag_extra = {
        "N_adv_estimated_by_ragdefender": n_to_remove,
        "notes": RANDOM_NOTE_TEMPLATE.format(seed=seed),
    }
    return kept, diag_extra
