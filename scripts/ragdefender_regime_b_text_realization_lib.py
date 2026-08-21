"""Pure-function library for the REGIME-B STAGE-1 TEXT-MANIFOLD
REALIZATION STUDY (the "realizability bridge" between the Regime-B
matrix-space oracle and actual Stella text geometry).

Every function here is a plain NumPy/stdlib computation -- no Stella, no
`sentence_transformers`, no retrieval, no generation, no network access.
This module is imported by both the driver script
(`scripts/run_ragdefender_regime_b_text_realization.py`, which DOES call
Stella) and the test suite (which mostly does NOT).

Evidence-label discipline (see task spec / report):
    L2 -- matrix/statistic oracle (Regime-B matrix-space oracle, upstream)
    L3 -- fixed-context text realization (this module's primary target)
    L4 -- retrieval-preserving text realization (downstream, separate module)

CLAIM DISCIPLINE: nothing in this module or its callers may label an L3
result an "attack" or a rewritten-clean-passage result "attacker-
realizable" -- see `classify_realization` and the report for the exact
wording rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# PHASE 2 -- oracle-direction alignment metrics for the actual Stella
# text-induced similarity-delta vector.
# ---------------------------------------------------------------------------

MODE_SIGN = {"boost": 1.0, "decrease": -1.0}


@dataclass(frozen=True)
class AlignmentResult:
    delta: np.ndarray  # d_j = cos(e_i_rewrite, e_j) - cos(e_i_original, e_j), j != i, in fixed j-order
    mean_signed_alignment: float  # mode_sign * mean(d_j)
    median_signed_alignment: float  # mode_sign * median(d_j)
    fraction_entries_in_oracle_direction: float  # fraction of d_j with sign(d_j) == mode_sign
    max_delta: float
    min_delta: float
    cosine_alignment: float  # cos(d, u), u = mode_sign * ones_like(d)
    fitted_beta: float  # argmin_{beta>=0} ||d - beta*u||_2 -- closed form projection, clipped at 0
    oracle_profile_residual: float  # ||d - fitted_beta*u||_2


def compute_delta_vector(
    sim_row_original: np.ndarray, sim_row_rewrite: np.ndarray, candidate_index: int
) -> np.ndarray:
    """`sim_row_original`/`sim_row_rewrite` are the candidate passage's full
    similarity row (length k, including the self entry) against the ORIGINAL
    and REWRITTEN 10-passage contexts respectively. Returns d_j for all
    j != candidate_index, in ascending-j order (length k-1)."""
    if sim_row_original.shape != sim_row_rewrite.shape:
        raise ValueError("Original and rewrite similarity rows must have the same shape.")
    k = sim_row_original.shape[0]
    mask = np.ones(k, dtype=bool)
    mask[candidate_index] = False
    return sim_row_rewrite[mask] - sim_row_original[mask]


def compute_alignment(delta: np.ndarray, oracle_mode: str) -> AlignmentResult:
    if oracle_mode not in MODE_SIGN:
        raise ValueError(f"oracle_mode must be 'boost' or 'decrease', got {oracle_mode!r}")
    sign = MODE_SIGN[oracle_mode]
    delta = np.asarray(delta, dtype=np.float64)
    n = delta.shape[0]
    if n == 0:
        raise ValueError("delta vector must be non-empty.")

    u = np.full(n, sign, dtype=np.float64)

    mean_signed_alignment = float(sign * np.mean(delta))
    median_signed_alignment = float(sign * np.median(delta))
    fraction_in_direction = float(np.mean(np.sign(delta) == sign))
    max_delta = float(np.max(delta))
    min_delta = float(np.min(delta))

    delta_norm = float(np.linalg.norm(delta))
    u_norm = float(np.linalg.norm(u))
    if delta_norm == 0.0 or u_norm == 0.0:
        cosine_alignment = 0.0
    else:
        cosine_alignment = float(np.dot(delta, u) / (delta_norm * u_norm))

    # beta* = argmin_{beta>=0} ||delta - beta*u||_2. Unconstrained OLS
    # projection is beta_hat = (delta . u) / (u . u); clip at 0 for the
    # beta>=0 constraint (a 1-D non-negative least squares problem).
    uu = float(np.dot(u, u))
    beta_unconstrained = float(np.dot(delta, u) / uu) if uu > 0 else 0.0
    fitted_beta = max(0.0, beta_unconstrained)
    residual = float(np.linalg.norm(delta - fitted_beta * u))

    return AlignmentResult(
        delta=delta,
        mean_signed_alignment=mean_signed_alignment,
        median_signed_alignment=median_signed_alignment,
        fraction_entries_in_oracle_direction=fraction_in_direction,
        max_delta=max_delta,
        min_delta=min_delta,
        cosine_alignment=cosine_alignment,
        fitted_beta=fitted_beta,
        oracle_profile_residual=residual,
    )


# ---------------------------------------------------------------------------
# PHASE 4 -- semantic-preservation checks (rule-based; MiniLM cosine is
# computed by the driver, which owns the encoder, and merged in by the
# caller via `preservation_pass`).
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
# Capitalized multi-word-safe token proxy for "obvious named entities": any
# capitalized word not at sentence-initial position is a weak but
# deterministic, dependency-free proxy check (full NER is out of scope --
# the task explicitly allows deterministic rule-based checks when a
# secondary semantic encoder is unavailable, and even when it IS available,
# only as a supplementary check).
_CAP_WORD_RE = re.compile(r"\b[A-Z][a-zA-Z]+\b")


@dataclass(frozen=True)
class SemanticPreservationCheck:
    word_count_original: int
    word_count_rewrite: int
    length_ratio: float
    length_ratio_pass: bool  # 0.70 <= ratio <= 1.40
    numbers_original: List[str]
    numbers_rewrite: List[str]
    numbers_preserved: bool  # every number in original appears in rewrite
    years_original: List[str]
    years_rewrite: List[str]
    years_preserved: bool
    exact_duplicate: bool  # rewrite == original (stripped) -- not a real rewrite
    lexical_overlap_jaccard: float  # token-set Jaccard(original, rewrite)
    minilm_cosine: Optional[float]  # None if MiniLM unavailable
    minilm_available: bool
    semantic_preservation_pass: bool


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def rule_based_semantic_check(
    original_text: str,
    rewritten_text: str,
    minilm_cosine: Optional[float] = None,
    minilm_available: bool = False,
    minilm_threshold: float = 0.80,
) -> SemanticPreservationCheck:
    orig_words = original_text.split()
    new_words = rewritten_text.split()
    wc_orig, wc_new = len(orig_words), len(new_words)
    ratio = wc_new / wc_orig if wc_orig > 0 else 0.0
    length_ratio_pass = 0.70 <= ratio <= 1.40

    nums_orig = _NUMBER_RE.findall(original_text)
    nums_new = _NUMBER_RE.findall(rewritten_text)
    numbers_preserved = set(nums_orig).issubset(set(nums_new))

    years_orig = _YEAR_RE.findall(original_text)
    years_new = _YEAR_RE.findall(rewritten_text)
    years_preserved = set(years_orig).issubset(set(years_new))

    exact_duplicate = original_text.strip() == rewritten_text.strip()

    tok_orig, tok_new = set(_tokenize(original_text)), set(_tokenize(rewritten_text))
    union = tok_orig | tok_new
    jaccard = len(tok_orig & tok_new) / len(union) if union else 0.0

    rule_based_pass = length_ratio_pass and numbers_preserved and years_preserved and not exact_duplicate

    if minilm_available and minilm_cosine is not None:
        overall_pass = rule_based_pass and (minilm_cosine >= minilm_threshold)
    else:
        overall_pass = rule_based_pass

    return SemanticPreservationCheck(
        word_count_original=wc_orig,
        word_count_rewrite=wc_new,
        length_ratio=ratio,
        length_ratio_pass=length_ratio_pass,
        numbers_original=nums_orig,
        numbers_rewrite=nums_new,
        numbers_preserved=numbers_preserved,
        years_original=years_orig,
        years_rewrite=years_new,
        years_preserved=years_preserved,
        exact_duplicate=exact_duplicate,
        lexical_overlap_jaccard=jaccard,
        minilm_cosine=minilm_cosine,
        minilm_available=minilm_available,
        semantic_preservation_pass=overall_pass,
    )


# ---------------------------------------------------------------------------
# PHASE 5 -- L3 realization classification (A/B/C/D).
# ---------------------------------------------------------------------------

REALIZATION_FULL = "A. FULL REALIZATION"
REALIZATION_MECHANISM_PARTIAL = "B. MECHANISM-PARTIAL"
REALIZATION_GEOMETRY_ALIGNED_ONLY = "C. GEOMETRY-ALIGNED ONLY"
REALIZATION_NON_ALIGNED = "D. NON-ALIGNED"


@dataclass(frozen=True)
class MedianLimitedProgress:
    exact_tie_broken: bool  # rank5 != rank6 similarity value in the rewrite matrix (within tol)
    median_gap_became_positive: bool
    n_above_median_increased: bool


@dataclass(frozen=True)
class MeanGatedProgress:
    blocking_margin_moved_toward_zero: bool
    blocking_margin_crossed_zero: bool


def classify_realization(
    n_adv_original: int,
    n_adv_rewrite: int,
    mechanism: str,
    alignment: AlignmentResult,
    median_progress: Optional[MedianLimitedProgress] = None,
    mean_progress: Optional[MeanGatedProgress] = None,
    tol: float = 1e-8,
) -> str:
    """A (full) requires n_adv_original == 4 and n_adv_rewrite >= 5
    (operationally 4 -> 5, per the k=10 structural ceiling floor(k/2)=5).

    B/C/D otherwise, based on whether the Stage-1 *blocker itself* moved
    (mechanism-partial) vs. whether only the raw similarity row moved in
    the oracle direction without touching the blocker (geometry-aligned
    only) vs. neither (non-aligned).
    """
    if n_adv_original == 4 and n_adv_rewrite >= 5:
        return REALIZATION_FULL

    blocker_moved = False
    if mechanism == "median-limited" and median_progress is not None:
        blocker_moved = (
            median_progress.exact_tie_broken
            or median_progress.median_gap_became_positive
            or median_progress.n_above_median_increased
        )
    elif mechanism == "mean-gated" and mean_progress is not None:
        blocker_moved = mean_progress.blocking_margin_moved_toward_zero or mean_progress.blocking_margin_crossed_zero

    if blocker_moved:
        return REALIZATION_MECHANISM_PARTIAL

    geometry_aligned = alignment.mean_signed_alignment > 0 or alignment.fraction_entries_in_oracle_direction > 0.5
    if geometry_aligned:
        return REALIZATION_GEOMETRY_ALIGNED_ONLY

    return REALIZATION_NON_ALIGNED


# ---------------------------------------------------------------------------
# Stage-2 outcome labeling (Phase 6).
# ---------------------------------------------------------------------------

STAGE2_SUCCESS = "L3-A"
STAGE2_DEGRADED = "L3-B"


def classify_stage2_outcome(removed_poison: int, removed_clean: int, m_poison: int = 5) -> str:
    if removed_poison == m_poison and removed_clean == 0:
        return STAGE2_SUCCESS
    return STAGE2_DEGRADED


# ---------------------------------------------------------------------------
# Threat-model label audit wording (Phase 9) -- pure string logic, no label
# selection here (selection happens upstream; this only maps a boolean to
# the mandated wording so the driver/report cannot drift from it).
# ---------------------------------------------------------------------------

WORDING_ATTACKER_CONTROLLED = "threat-model-compatible passage control"
WORDING_NON_ATTACKER_CONTROLLED = "text-manifold realization only"


def threat_model_wording(candidate_is_poison: bool) -> str:
    return WORDING_ATTACKER_CONTROLLED if candidate_is_poison else WORDING_NON_ATTACKER_CONTROLLED


# ---------------------------------------------------------------------------
# Deterministic style selection for R5 (Phase 7) -- chosen from a hash of
# query_id so the discourse style is not selected based on any observed
# outcome.
# ---------------------------------------------------------------------------

R5_STYLES = ["reference style", "historical/background style", "explanatory style", "report style"]


def r5_style_for_query(query_id: str) -> str:
    import hashlib

    h = hashlib.sha256(query_id.encode("utf-8")).hexdigest()
    idx = int(h, 16) % len(R5_STYLES)
    return R5_STYLES[idx]
