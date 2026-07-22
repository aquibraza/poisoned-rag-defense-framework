"""Faithful, exposed-intermediate reimplementation of RAGDefender's Stage-1
concentration estimator and Stage-2 pair-frequency identification, for
diagnostics/visualization purposes only.

This module is purely additive and is never imported by
`defense/defense_runner.py`, `defense/dispatch.py`, or `main.py` --
`ragdefender_original`'s actual behavior is untouched. Everything here is a
side-by-side reimplementation of the same math, meant to be cross-checked
against real diagnostics output (see `scripts/visualize_ragdefender_clusters.py`),
not a replacement for the baseline code path.

Faithfulness notes (mirror implementation behavior, not paper notation):

- `concentration_stage1` reproduces `defense_runner._find_num_adversarial`
  exactly, including computing the per-passage mean/median cosine similarity
  over the *full* similarity matrix -- i.e. **including the diagonal
  self-similarity term** (`sim(i, i) == 1.0`). `torch.mean(matrix, dim=0)` and
  `torch.median(matrix, dim=0)` do not exclude `j == i`, so this module does
  not either. Excluding the diagonal would be a "paper-correct" variant, not
  a reproduction of the actual baseline behavior -- that variant is
  deliberately not implemented here.
- `torch.median` (unlike `numpy.median`) returns the **lower** of the two
  middle values when the input has an even number of elements (rather than
  averaging them). Since `k` is frequently even (e.g. k=10) in the existing
  diagnostics this repo is checked against, `_torch_style_median` replicates
  that exact tie-breaking rule rather than using `numpy.median`'s default
  (which would silently diverge from `defense_runner.py` for even `k`).
- `stage2_pair_frequency` reproduces `defense_runner._top_similar_pairs` +
  `apply_defense`'s `Counter`/`math.copysign(sim**p, sim)` pair-scoring
  exactly, including using only `i < j` non-self pairs (`top_similar_pairs`
  never includes `i == j`), and including the *insertion-order* tie-breaking
  that `collections.Counter`/`dict` + a stable `sorted()` produce (a plain
  `dict` here, populated in the same iteration order as the real code, gives
  bit-for-bit identical tie-breaking).
- `p` defaults to `2` -- the baseline's hardcoded exponent
  (`math.copysign(sim * sim, sim)` == `math.copysign(abs(sim) ** 2, sim)`) --
  exposed as a parameter only for documentation/future generalization, never
  changed by default.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "ConcentrationResult",
    "Stage2Result",
    "concentration_stage1",
    "stage2_pair_frequency",
]


def _torch_style_median_1d(values: np.ndarray) -> float:
    """Median of a 1D array, matching `torch.median`'s tie-breaking: for an
    even-length input, returns the *lower* of the two middle values (not the
    average of them, unlike `numpy.median`'s default)."""
    sorted_vals = np.sort(np.asarray(values, dtype=np.float64))
    n = sorted_vals.shape[0]
    if n == 0:
        raise ValueError("_torch_style_median_1d: empty input")
    idx = (n - 1) // 2
    return float(sorted_vals[idx])


def _torch_style_median_axis0(matrix: np.ndarray) -> np.ndarray:
    """Per-column median of a 2D array, matching `torch.median(x, dim=0)`'s
    tie-breaking (see `_torch_style_median_1d`)."""
    sorted_cols = np.sort(np.asarray(matrix, dtype=np.float64), axis=0)
    n = sorted_cols.shape[0]
    if n == 0:
        raise ValueError("_torch_style_median_axis0: empty input")
    idx = (n - 1) // 2
    return sorted_cols[idx, :]


@dataclass(frozen=True)
class ConcentrationResult:
    """Every intermediate value from `defense_runner._find_num_adversarial`,
    exposed instead of collapsed into a single count."""

    avg: np.ndarray  # per-passage mean similarity, includes self-similarity
    median: np.ndarray  # per-passage median similarity, includes self-similarity (torch tie-break)
    avg_avg: float  # mean of `avg` across all passages
    avg_median: float  # torch-style median of `median` across all passages
    combined_threshold: float  # (avg_median + avg_avg) / 2
    above_avg: np.ndarray  # bool, avg[i] > avg_avg
    above_median: np.ndarray  # bool, median[i] > combined_threshold
    raw_or_flag: np.ndarray  # bool, above_avg OR above_median (pre-flip)
    flipped: bool  # whether the "len(text_list) - sum(final)" branch fired
    adv_side_flag: np.ndarray  # bool, per-passage membership in the *estimated* adversarial set (post-flip)
    n_adv_estimated: int  # == adv_side_flag.sum(), the Stage-1 N_adv estimate


def concentration_stage1(cos_sim_matrix: np.ndarray) -> ConcentrationResult:
    """Recompute RAGDefender's multi-hop (HotpotQA) Stage-1 concentration
    estimator, exposing every intermediate.

    `cos_sim_matrix` must be a k x k symmetric cosine-similarity matrix
    (including the diagonal, i.e. `cos_sim_matrix[i, i] == 1.0`), exactly the
    shape/contents `sentence_transformers.util.cos_sim(embeddings,
    embeddings)` produces for `k` passages.
    """
    matrix = np.asarray(cos_sim_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"cos_sim_matrix must be square 2D, got shape {matrix.shape}")
    k = matrix.shape[0]
    if k == 0:
        raise ValueError("concentration_stage1: empty similarity matrix")

    # torch.mean(matrix, dim=0) / torch.median(matrix, dim=0): column-wise
    # reductions over the FULL matrix, diagonal included -- see module
    # docstring. `matrix` is symmetric, so this equals the row-wise reduction
    # too, but dim=0 is what the baseline actually calls.
    avg = matrix.mean(axis=0)
    median = _torch_style_median_axis0(matrix)

    avg_avg = float(avg.mean())
    avg_median = _torch_style_median_1d(median)
    combined_threshold = (avg_median + avg_avg) / 2.0

    above_avg = avg > avg_avg
    above_median = median > combined_threshold
    raw_or_flag = above_avg | above_median

    or_count = int(raw_or_flag.sum())
    # result = sum(final) if sum(final) > 0 and avg_avg < avg_median else len(text_list) - sum(final)
    non_flip_condition = or_count > 0 and avg_avg < avg_median
    flipped = not non_flip_condition

    if non_flip_condition:
        n_adv_estimated = or_count
        adv_side_flag = raw_or_flag.copy()
    else:
        n_adv_estimated = k - or_count
        adv_side_flag = ~raw_or_flag

    return ConcentrationResult(
        avg=avg,
        median=median,
        avg_avg=avg_avg,
        avg_median=avg_median,
        combined_threshold=combined_threshold,
        above_avg=above_avg,
        above_median=above_median,
        raw_or_flag=raw_or_flag,
        flipped=flipped,
        adv_side_flag=adv_side_flag,
        n_adv_estimated=n_adv_estimated,
    )


@dataclass(frozen=True)
class Stage2Result:
    """Every intermediate value from RAGDefender's Stage-2 pair-frequency
    identification (`_top_similar_pairs` + `apply_defense`'s `Counter`
    scoring), exposed instead of collapsed into a suspect-index set."""

    n_pairs: int  # max(1, C(n_adv, 2)), or 0 if n_adv <= 0
    top_pairs: List[Tuple[int, int, float]]  # (i, j, sim), i < j, sorted desc by sim
    frequency_scores: np.ndarray  # length k, 0.0 for indices absent from every top pair
    selected_indices: List[int]  # top n_adv indices by frequency_scores, Counter-order tie-break


def stage2_pair_frequency(cos_sim_matrix: np.ndarray, n_adv: int, p: float = 2.0) -> Stage2Result:
    """Recompute RAGDefender's Stage-2 pair-frequency identification,
    exposing every intermediate.

    Mirrors `defense_runner._top_similar_pairs` (only `i < j` non-self pairs,
    sorted by similarity descending, top `n_pairs` kept) and `apply_defense`'s
    `Counter` frequency scoring (`math.copysign(sim ** p, sim)` per pair,
    accumulated per index, ranked by score with the same insertion-order
    tie-breaking a real `Counter` + stable `sorted()` would produce).

    If `n_adv <= 0`, returns an empty result -- matches `apply_defense`'s own
    early return (`if num_adv == 0: return doc_list[...]`), which never
    reaches Stage 2 at all in that case.
    """
    matrix = np.asarray(cos_sim_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"cos_sim_matrix must be square 2D, got shape {matrix.shape}")
    k = matrix.shape[0]

    if n_adv <= 0:
        return Stage2Result(n_pairs=0, top_pairs=[], frequency_scores=np.zeros(k), selected_indices=[])

    n_pairs = max(1, int(n_adv * (n_adv - 1) / 2))

    pairs: List[Tuple[int, int, float]] = [
        (i, j, float(matrix[i, j])) for i in range(k) for j in range(i + 1, k)
    ]
    pairs.sort(key=lambda t: t[2], reverse=True)
    top_pairs = pairs[:n_pairs]

    # Plain dict, not collections.Counter, but identical tie-breaking:
    # CPython dicts (3.7+) preserve first-insertion order for a key even
    # when its value is later updated via `+=`-style reassignment, exactly
    # like Counter. The subsequent `sorted(..., reverse=True)` is stable, so
    # ties resolve in the same first-seen order the real code would produce.
    freq: Dict[int, float] = {}
    for x, y, sim in top_pairs:
        contribution = math.copysign(abs(sim) ** p, sim)
        for idx in (x, y):
            freq[idx] = freq.get(idx, 0.0) + contribution

    ranked = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    selected_indices = [idx for idx, _ in ranked[:n_adv]]

    frequency_scores = np.zeros(k, dtype=np.float64)
    for idx, score in freq.items():
        frequency_scores[idx] = score

    return Stage2Result(
        n_pairs=n_pairs,
        top_pairs=top_pairs,
        frequency_scores=frequency_scores,
        selected_indices=selected_indices,
    )
