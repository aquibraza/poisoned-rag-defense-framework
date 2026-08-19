"""Programmatic exhaustive/synthetic validation of the structural Stage-1
count ceiling `N_adv <= floor(k/2)` implied by the final paper's Eq. (3).

==========================================================================
SCOPE
==========================================================================
See `results/diagnostics/ragdefender_count_ceiling/COUNT_CEILING_ANALYSIS.md`
for the mathematical proof this script validates empirically. This script
generates random and hand-computable-edge synthetic symmetric similarity
matrices for k in {2, 3, 4, 5, 6, 10, 11}, and asserts, for BOTH the
production lower-of-two-middle median convention
(`ragdefender_internals.concentration_stage1_paper`, UNCHANGED) and the
diagnostic-only average-of-two-middle convention
(`scripts.run_ragdefender_median_sensitivity._concentration_stage1_average_median`,
also UNCHANGED here -- only imported):

    n_adv_estimated <= floor(k / 2)

No retrieval, generation, E1/CORAL/MMD, or LLM/API experiment was run. No
production code was modified. Writes ONLY to
`results/diagnostics/ragdefender_count_ceiling/`.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from defense import ragdefender_internals  # noqa: E402
from run_ragdefender_median_sensitivity import (  # noqa: E402
    _concentration_stage1_average_median,
)

OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_count_ceiling"
REQUIRED_K_VALUES = [2, 3, 4, 5, 6, 10, 11]
N_RANDOM_TRIALS_PER_REGIME = 25


class CountCeilingViolation(RuntimeError):
    """Raised if the production or diagnostic estimator ever exceeds
    floor(k/2) on any generated matrix -- would mean either the proof or
    its transcription into code is wrong. Must never fire."""


# ---------------------------------------------------------------------------
# Synthetic matrix generators
# ---------------------------------------------------------------------------

def _symmetrize(raw: np.ndarray) -> np.ndarray:
    """Force exact symmetry + unit diagonal, matching a real cosine
    similarity matrix's structural constraints (diagonal value itself is
    irrelevant to `concentration_stage1_paper`, which excludes it, but is
    set to 1.0 for realism)."""
    sym = (raw + raw.T) / 2.0
    np.fill_diagonal(sym, 1.0)
    return np.clip(sym, -1.0, 1.0)


def gen_uniform_random(k: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.uniform(-1.0, 1.0, size=(k, k))
    return _symmetrize(raw)


def gen_low_variance(k: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.uniform(0.4, 0.6)
    raw = base + rng.uniform(-0.01, 0.01, size=(k, k))
    return _symmetrize(raw)


def gen_two_cluster(k: int, rng: np.random.Generator) -> np.ndarray:
    """Poison/clean-like block structure: within-block similarity high,
    cross-block similarity low, with noise."""
    split = max(1, k // 2)
    labels = np.array([0] * split + [1] * (k - split))
    rng.shuffle(labels)
    raw = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            base = 0.85 if labels[i] == labels[j] else 0.15
            raw[i, j] = base + rng.uniform(-0.05, 0.05)
    return _symmetrize(raw)


def gen_near_degenerate(k: int, rng: np.random.Generator) -> np.ndarray:
    """Many exact ties -- stresses the median tie-break / convention
    boundary directly."""
    levels = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], size=(k, k))
    return _symmetrize(levels)


REGIME_GENERATORS = {
    "uniform_random": gen_uniform_random,
    "low_variance": gen_low_variance,
    "two_cluster": gen_two_cluster,
    "near_degenerate": gen_near_degenerate,
}


# ---------------------------------------------------------------------------
# Hand-computable edge fixtures: designed to push N_adv as close to
# floor(k/2) as the structural bound allows.
# ---------------------------------------------------------------------------

def gen_ceiling_edge_fixture(k: int, seed: int = 0) -> np.ndarray:
    """Hand-describable three-level block construction, engineered to get
    `N_adv` as close to `floor(k/2)` as the structural ceiling (§2-3 of
    `COUNT_CEILING_ANALYSIS.md`) allows -- and, for every required `k`
    except `k=2` (see note below), to hit it EXACTLY.

    Construction: split into a HUB group of size `h = floor(k/2)` and a
    BACKGROUND group of size `b = k - h`.
      - HUB-HUB similarity   = A = 0.9  (dense, high)
      - HUB-BACKGROUND       = C = 0.5  (medium, shared "cross" value)
      - BACKGROUND-BACKGROUND = B = 0.1  (sparse, low)
    plus a tiny deterministic asymmetric perturbation (`eps=0.001`,
    `np.random.default_rng(seed)`).

    WHY THE PERTURBATION IS NECESSARY (a genuine, provable structural
    fact about this specific 3-level block shape, discovered while
    building this fixture -- documented here rather than only in the
    analysis doc, since it explains a design choice a reader could
    otherwise mistake for an arbitrary "add noise to make the test
    pass" hack): for EVEN `k`, `h == b == k/2` exactly. A HUB row then
    contains exactly `b = k/2` copies of the shared cross value `C`
    (from its `b` background neighbors) among its `k-1` off-diagonal
    entries, and a BACKGROUND row contains exactly `h = k/2` copies of
    that SAME value `C` (from its `h` hub neighbors). Because the
    torch-style median position for a `(k-1)`-length row (`k` even, so
    `k-1` is odd) is the `(k/2)`-th order statistic, and there are
    EXACTLY `k/2` copies of `C` in BOTH kinds of row, the median of
    *every* row -- hub or background -- lands exactly on `C`,
    regardless of how `A` and `B` are chosen (as long as `A > C > B`).
    All `k` per-passage medians become identically `C`, so `s_tilde = C`
    too, and NO passage is strictly `> s_tilde`: this pure 3-level
    construction gives `N_adv = 0` for every even `k > 2`, not the
    ceiling. A tiny deterministic asymmetric perturbation breaks this
    exact positional tie (empirically verified for every required
    EVEN `k` in {4, 6, 10}) without changing the qualitative
    hub/background structure. For ODD `k`, `h != b` and no such forced
    tie exists -- the un-perturbed 3-level construction already hits the
    ceiling exactly (verified for `k` in {3, 5, 11}); the perturbation
    is applied uniformly for all `k` anyway, for a single, consistent
    fixture definition.

    SPECIAL CASE `k=2`: `N_adv` is **provably always exactly 0** for
    ANY symmetric similarity matrix at `k=2`, including this fixture --
    see `COUNT_CEILING_ANALYSIS.md` §5 / `TestK2AlwaysZero` in the test
    module. With only 2 passages, `s_mean_0 == s_mean_1` is forced by
    matrix symmetry (each passage's only off-diagonal neighbor is the
    other one), so `s_bar` exactly equals both of them and neither can
    be STRICTLY greater than `s_bar`. This is a strictly stronger,
    exact fact than the ceiling bound (`floor(2/2)=1`) alone implies,
    so `k=2` is the one required `k` where this fixture does NOT reach
    the ceiling -- by mathematical necessity, not a fixture-design gap.
    """
    h = k // 2
    rng = np.random.default_rng(seed)
    eps = 0.001
    a_val, c_val, b_val = 0.9, 0.5, 0.1
    raw = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            i_hub, j_hub = i < h, j < h
            if i_hub and j_hub:
                base = a_val
            elif (not i_hub) and (not j_hub):
                base = b_val
            else:
                base = c_val
            noise = rng.uniform(-eps, eps) if i != j else 0.0
            raw[i, j] = base + noise
            raw[j, i] = base + noise
    return _symmetrize(raw)


EDGE_FIXTURE_GENERATORS = {
    "ceiling_edge_three_level_block": gen_ceiling_edge_fixture,
}


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def _validate_matrix(k: int, matrix: np.ndarray, regime: str, trial_id: str) -> List[dict]:
    ceiling = k // 2
    rows: List[dict] = []

    result_a = ragdefender_internals.concentration_stage1_paper(matrix)
    above_median_count_a = int(result_a.above_median.sum())
    n_adv_a = result_a.n_adv_estimated
    if n_adv_a > ceiling:
        raise CountCeilingViolation(
            f"k={k} trial={trial_id} regime={regime}: production N_adv={n_adv_a} > ceiling={ceiling}"
        )
    if above_median_count_a > ceiling:
        raise CountCeilingViolation(
            f"k={k} trial={trial_id} regime={regime}: production above_median_count={above_median_count_a} "
            f"> ceiling={ceiling}"
        )
    if n_adv_a > above_median_count_a:
        raise CountCeilingViolation(
            f"k={k} trial={trial_id} regime={regime}: AND-combined N_adv={n_adv_a} exceeds "
            f"median-only count={above_median_count_a} -- AND must never exceed either operand."
        )
    rows.append({
        "k": k,
        "trial_id": trial_id,
        "regime": regime,
        "convention": "A_lower_of_two_middle_production",
        "n_adv_estimated": n_adv_a,
        "above_median_count": above_median_count_a,
        "ceiling": ceiling,
        "n_adv_leq_ceiling": n_adv_a <= ceiling,
        "above_median_leq_ceiling": above_median_count_a <= ceiling,
        "and_leq_median_only": n_adv_a <= above_median_count_a,
    })

    result_b = _concentration_stage1_average_median(matrix)
    above_median_count_b = int(result_b.above_median.sum())
    n_adv_b = result_b.n_adv_estimated
    if n_adv_b > ceiling:
        raise CountCeilingViolation(
            f"k={k} trial={trial_id} regime={regime}: diagnostic-convention N_adv={n_adv_b} > ceiling={ceiling}"
        )
    if above_median_count_b > ceiling:
        raise CountCeilingViolation(
            f"k={k} trial={trial_id} regime={regime}: diagnostic-convention above_median_count="
            f"{above_median_count_b} > ceiling={ceiling}"
        )
    if n_adv_b > above_median_count_b:
        raise CountCeilingViolation(
            f"k={k} trial={trial_id} regime={regime}: AND-combined N_adv={n_adv_b} (diagnostic convention) "
            f"exceeds median-only count={above_median_count_b}."
        )
    rows.append({
        "k": k,
        "trial_id": trial_id,
        "regime": regime,
        "convention": "B_average_of_two_middle_diagnostic_only",
        "n_adv_estimated": n_adv_b,
        "above_median_count": above_median_count_b,
        "ceiling": ceiling,
        "n_adv_leq_ceiling": n_adv_b <= ceiling,
        "above_median_leq_ceiling": above_median_count_b <= ceiling,
        "and_leq_median_only": n_adv_b <= above_median_count_b,
    })
    return rows


def run_validation(seed: int = 20260818) -> List[dict]:
    rows: List[dict] = []
    rng = np.random.default_rng(seed)

    for k in REQUIRED_K_VALUES:
        for regime, generator in REGIME_GENERATORS.items():
            for trial in range(N_RANDOM_TRIALS_PER_REGIME):
                matrix = generator(k, rng)
                rows.extend(_validate_matrix(k, matrix, regime=regime, trial_id=f"random_{trial}"))
        for fixture_name, fixture_generator in EDGE_FIXTURE_GENERATORS.items():
            matrix = fixture_generator(k)
            rows.extend(_validate_matrix(k, matrix, regime=fixture_name, trial_id="edge_fixture"))

    return rows


def build_summary(rows: List[dict]) -> Dict:
    n_total = len(rows)
    n_violations = sum(
        1 for r in rows
        if not (r["n_adv_leq_ceiling"] and r["above_median_leq_ceiling"] and r["and_leq_median_only"])
    )
    exact_ceiling_hits = sum(1 for r in rows if r["n_adv_estimated"] == r["ceiling"] and r["ceiling"] > 0)
    edge_fixture_rows = [r for r in rows if r["regime"] == "ceiling_edge_three_level_block"]
    edge_fixture_exact = sum(1 for r in edge_fixture_rows if r["n_adv_estimated"] == r["ceiling"])
    return {
        "n_total_checks": n_total,
        "n_violations": n_violations,
        "n_exact_ceiling_hits": exact_ceiling_hits,
        "n_edge_fixture_trials": len(edge_fixture_rows),
        "n_edge_fixture_exact_ceiling_hits": edge_fixture_exact,
        "k_values_tested": _pipe(REQUIRED_K_VALUES),
    }


def _pipe(values) -> str:
    return "|".join(str(v) for v in values)


def main() -> None:
    out_csv = OUTPUT_DIR / "synthetic_validation.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = run_validation()
    summary = build_summary(rows)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Count-ceiling validation complete: {summary['n_total_checks']} checks, "
          f"{summary['n_violations']} violations.")
    print(f"Exact ceiling hits: {summary['n_exact_ceiling_hits']}/{summary['n_total_checks']}")
    print(f"Edge-fixture exact ceiling hits: {summary['n_edge_fixture_exact_ceiling_hits']}/"
          f"{summary['n_edge_fixture_trials']}")
    print(f"Wrote: {out_csv}")

    if summary["n_violations"] > 0:
        raise CountCeilingViolation(f"{summary['n_violations']} ceiling violations found -- see {out_csv}")


if __name__ == "__main__":
    main()
