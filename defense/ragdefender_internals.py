"""RAGDefender Stage-1 concentration estimators and Stage-2 pair-frequency
identification, with every intermediate exposed.

Two Stage-1 multi-hop (HotpotQA) estimators live here:

- `concentration_stage1` -- a faithful reimplementation of
  `defense_runner._find_num_adversarial` (the `ragdefender_legacy` behavior:
  OR logic, diagonal-inclusive mean/median, a hybrid threshold, and a flip
  branch). This half of the module is diagnostics/visualization-only and is
  NOT imported by `defense/defense_runner.py` or `defense/dispatch.py` --
  `ragdefender_legacy`'s actual behavior remains its own independent,
  untouched inline code in `defense_runner.py`. It exists here purely to be
  cross-checked against real diagnostics output (see
  `scripts/visualize_ragdefender_clusters.py`).
- `concentration_stage1_paper` -- the FINAL ACSAC 2025 paper's Eq. (3)
  estimator (self-excluded, AND logic, no flip). Unlike `concentration_stage1`
  above, this IS the actual implementation `defense_runner._find_num_adversarial_paper`
  calls for `ragdefender_paper` -- there is no separate, independent copy of
  this math in `defense_runner.py`. See docs/RAGDEFENDER_FIDELITY_AUDIT_V2.md
  for the full paper-vs-legacy fidelity matrix.

`stage2_pair_frequency` (Eq. 4-7) is shared conceptually by both variants:
`ragdefender_legacy`'s Stage 2 is its own untouched inline code in
`defense_runner.apply_defense` (proven mathematically equivalent to this
function -- see its docstring below), kept separate for zero risk to
historical byte-identical reproducibility; `ragdefender_paper` calls this
function directly, since Stage 2 was already paper-faithful and there is no
reason to fork it for the new variant.

Faithfulness notes for `concentration_stage1` (mirror `ragdefender_legacy`
implementation behavior, not paper notation):

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
    "ConcentrationResultPaper",
    "Stage2Result",
    "concentration_stage1",
    "concentration_stage1_paper",
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
class ConcentrationResultPaper:
    """Every intermediate value from the FINAL ACSAC 2025 paper's Eq. (3)
    multi-hop concentration estimator, exposed instead of collapsed into a
    single count.

    Field names deliberately do NOT mirror `ConcentrationResult` 1:1 (no
    `flipped`, no `combined_threshold`) -- those fields describe
    `ragdefender_legacy`-specific mechanics (a hybrid threshold, a flip
    branch) that simply do not exist in the paper-faithful estimator, and
    forcing a shared shape would misleadingly imply they do.
    """

    s_mean: np.ndarray  # per-passage mean concentration, self EXCLUDED, 1/(k-1)-normalized (paper's s^mean_i)
    s_median: np.ndarray  # per-passage median concentration, self EXCLUDED (paper's s^median_i)
    s_bar: float  # global mean of s_mean (paper's \bar{s})
    s_tilde: float  # global median of s_median (paper's \tilde{s})
    above_mean: np.ndarray  # bool, s_mean[i] > s_bar
    above_median: np.ndarray  # bool, s_median[i] > s_tilde
    adv_flag: np.ndarray  # bool, above_mean AND above_median (Eq. 3 -- AND, no flip)
    n_adv_estimated: int  # == adv_flag.sum(), the paper-faithful N_adv estimate


def concentration_stage1_paper(cos_sim_matrix: np.ndarray) -> ConcentrationResultPaper:
    """FINAL ACSAC 2025 paper's Eq. (3) multi-hop (HotpotQA) Stage-1
    concentration estimator -- the authoritative target for
    `ragdefender_paper`, called directly by
    `defense_runner._find_num_adversarial_paper` (not a diagnostics-only
    reimplementation, unlike `concentration_stage1` above).

    Differs from `concentration_stage1` (`ragdefender_legacy`) in every
    material way documented in docs/RAGDEFENDER_FIDELITY_AUDIT_V2.md:
      - excludes the diagonal self-similarity term (S_ii) from both the
        per-passage mean and median, using a 1/(k-1) denominator for the
        mean (paper's s^mean_i definition: s^mean_i = 1/(|R|-1) * sum_{j!=i}
        sim(r_i, r_j));
      - combines the two above-threshold flags with AND, not OR;
      - uses a single global median-of-medians threshold (paper's
        \\tilde{s} = median({s^median_i})), not the legacy hybrid
        (avg_median + avg_avg) / 2;
      - never flips to the complement set (no `k - count` branch) -- Eq. (3)
        is a direct sum of indicator products, with no case split.

    AUTHORITY RULE applied here (see plan §0a item 2): the final published
    paper governs every explicitly-specified behavior above; the authors'
    officially released code is consulted ONLY to fill in the one place the
    paper is silent -- how to break a tie when taking a median over an even
    number of values (both the per-passage median over k-1 non-self
    similarities when k is odd, and the global median-of-medians when the
    passage count is even). The paper never states "average the two middle
    values" or "take the lower one," so rather than inventing an arbitrary
    new convention, this function reuses `_torch_style_median_1d` (the same
    lower-of-two-middle rule `torch.median` -- and therefore the authors' own
    code -- applies). This is documented here as an implementation choice
    attributed to the authority rule, not to the paper itself. See the "TIE
    BREAK" test in tests/test_ragdefender_paper_fidelity.py.

    `cos_sim_matrix` must be a k x k symmetric cosine-similarity matrix. The
    diagonal value is irrelevant -- it is excluded from every computation
    here (see the "SELF-SIMILARITY EXCLUSION" test).
    """
    matrix = np.asarray(cos_sim_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"cos_sim_matrix must be square 2D, got shape {matrix.shape}")
    k = matrix.shape[0]
    if k < 2:
        raise ValueError("concentration_stage1_paper: need at least 2 passages to exclude self")

    s_mean = np.zeros(k, dtype=np.float64)
    s_median = np.zeros(k, dtype=np.float64)
    for i in range(k):
        off_diag_row = np.delete(matrix[i, :], i)  # excludes S_ii, length k-1
        s_mean[i] = off_diag_row.mean()  # == 1/(k-1) * sum_{j != i} sim(i, j)
        s_median[i] = _torch_style_median_1d(off_diag_row)

    s_bar = float(s_mean.mean())
    s_tilde = _torch_style_median_1d(s_median)

    above_mean = s_mean > s_bar
    above_median = s_median > s_tilde
    adv_flag = above_mean & above_median  # Eq. (3): AND, no flip branch

    return ConcentrationResultPaper(
        s_mean=s_mean,
        s_median=s_median,
        s_bar=s_bar,
        s_tilde=s_tilde,
        above_mean=above_mean,
        above_median=above_median,
        adv_flag=adv_flag,
        n_adv_estimated=int(adv_flag.sum()),
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
