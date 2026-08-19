"""Nominal HotpotQA k=2 mathematical/code consistency audit -- regression
tests.

Scope note (read before extending): these tests are PURE mathematics/code
tests on literal 2-element (`k=2`) similarity matrices and short synthetic
document lists. NO Stella model is loaded, NO network access occurs, and
NO real HotpotQA retrieval is used -- every embedding is either a raw
NumPy similarity matrix or a small deterministic fake encoder producing
exactly the requested cosine similarity via unit-vector geometry
(`u=(1,0)`, `v=(s, sqrt(1-s^2))` => `cos_sim(u,v)=s` exactly).

Covers, per `results/diagnostics/ragdefender_k2_consistency/
RAGDEFENDER_K2_CONSISTENCY_REPORT.md` STEP 6:
  1. paper Eq. (3): N_adv == 0 for many symmetric 2x2 matrices.
  2. paper result independent of s (boundary values included).
  3. lower-middle vs average-middle median: both still N_adv == 0.
  4. legacy estimator behavior (authors' released find_num_adv,
     UNMODIFIED local port): pinned, independent of s.
  5. full `ragdefender_paper` (`_apply_defense_paper`): returned context
     at k=2 is the full, unfiltered input.
  6. full `ragdefender_legacy` (`apply_defense`): returned context at
     k=2 is ALSO the full, unfiltered input (via the restore-all
     fallback), for a completely different internal reason.
  7. the two variants' internal computations (N_adv, Stage-2 selected
     indices) provably differ even though their FINAL returned context
     is identical.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from defense import defense_runner, ragdefender_internals as ri  # noqa: E402
from run_ragdefender_median_sensitivity import (  # noqa: E402
    _concentration_stage1_average_median,
)

# Representative s values explicitly requested by the task, plus dense
# sweeps and the mathematical boundary values.
REQUIRED_S_VALUES = [-0.5, 0.0, 0.3, 0.8, 0.99]
DENSE_S_VALUES = list(np.round(np.linspace(-0.97, 0.97, 15), 4))
BOUNDARY_S_VALUES = [-1.0, 1.0]
ALL_S_VALUES = sorted(set(REQUIRED_S_VALUES + DENSE_S_VALUES + BOUNDARY_S_VALUES))


def _k2_matrix(s: float) -> np.ndarray:
    return np.array([[1.0, s], [s, 1.0]])


class _FakeEncoderIgnored:
    """Placeholder encoder whose `.encode()` output is never actually
    consulted for similarity -- see `_FakeCosSimUtil` below, which is
    monkeypatched in as the `sentence_transformers.util` stand-in and
    returns an EXACT, precomputed `[[1, s], [s, 1]]` matrix regardless of
    the embeddings passed to it. This sidesteps a real, independently
    interesting numerical fragility (see
    `TestFloatPrecisionFragilityIsReal` below): a literal unit-vector
    encoder (`u=(1,0)`, `v=(s, sqrt(1-s^2))`) does NOT reliably reproduce
    the exact `avg[0]==avg[1]`/`median[0]==median[1]` ties the k=2
    structural proofs depend on, once float32 embedding-normalization
    rounding is involved -- the SAME fragility a real Stella/MiniLM
    pipeline would face. Mocking `cos_sim` directly (as done here) is the
    only way to test the FULL `apply_defense`/`_apply_defense_paper` call
    path deterministically at exact ties."""

    def encode(self, text_list, convert_to_tensor=True):
        import torch

        return torch.zeros((len(text_list), 4), dtype=torch.float64)


class _FakeCosSimUtil:
    """Stand-in for `sentence_transformers.util`, monkeypatched in via
    `defense_runner._lazy_st`. `.cos_sim(a, b)` ignores its arguments and
    always returns the exact precomputed k=2 matrix `[[1, s], [s, 1]]`."""

    def __init__(self, s: float):
        self.s = s

    def cos_sim(self, a, b):
        import torch

        return torch.tensor(_k2_matrix(self.s), dtype=torch.float64)


def _patch_lazy_st(s: float):
    """Returns a `unittest.mock.patch.object(...)` context manager that
    makes every `defense_runner._lazy_st()` call return
    `(None, _FakeCosSimUtil(s))` -- i.e. every downstream `st_util.cos_sim`
    call in `_find_num_adversarial`, `_find_num_adversarial_paper`,
    `_top_similar_pairs`, and `_apply_defense_paper` sees the exact,
    noise-free `[[1, s], [s, 1]]` matrix."""
    from unittest import mock

    return mock.patch.object(defense_runner, "_lazy_st", return_value=(None, _FakeCosSimUtil(s)))


# ---------------------------------------------------------------------------
# 1-2. paper Eq. (3): N_adv == 0 for many symmetric 2x2 matrices, and this
#      is independent of s (including both boundary values s=-1, s=1).
# ---------------------------------------------------------------------------

class TestPaperEqThreeK2StructuralZero(unittest.TestCase):
    def test_n_adv_zero_for_all_representative_and_boundary_s(self):
        for s in ALL_S_VALUES:
            with self.subTest(s=s):
                result = ri.concentration_stage1_paper(_k2_matrix(s))
                self.assertEqual(result.n_adv_estimated, 0)
                self.assertFalse(bool(result.above_mean[0]))
                self.assertFalse(bool(result.above_mean[1]))
                self.assertFalse(bool(result.above_median[0]))
                self.assertFalse(bool(result.above_median[1]))

    def test_s_mean_equals_s_bar_exactly_at_k2(self):
        """The structural reason N_adv=0 always holds: with only one other
        passage to average over, s_mean_i IS sim(r_i, r_j) for the other
        passage j, and s_bar (their mean) is therefore identical to both
        s_mean_0 and s_mean_1 -- so `s_mean_i > s_bar` can never be true
        (strict `>` against an identical value)."""
        for s in REQUIRED_S_VALUES:
            with self.subTest(s=s):
                result = ri.concentration_stage1_paper(_k2_matrix(s))
                self.assertAlmostEqual(float(result.s_mean[0]), s, places=9)
                self.assertAlmostEqual(float(result.s_mean[1]), s, places=9)
                self.assertAlmostEqual(float(result.s_bar), s, places=9)

    def test_n_adv_independent_of_s_across_random_sweep(self):
        rng = np.random.default_rng(12345)
        for s in rng.uniform(-1.0, 1.0, size=50):
            result = ri.concentration_stage1_paper(_k2_matrix(float(s)))
            self.assertEqual(result.n_adv_estimated, 0)


# ---------------------------------------------------------------------------
# 3. Median-convention robustness: lower-of-two-middle (production) vs.
#    average-of-two-middle (diagnostic-only) -- both still N_adv == 0.
# ---------------------------------------------------------------------------

class TestMedianConventionRobustnessAtK2(unittest.TestCase):
    def test_average_median_convention_also_yields_zero(self):
        for s in REQUIRED_S_VALUES:
            with self.subTest(s=s):
                result = _concentration_stage1_average_median(_k2_matrix(s))
                self.assertEqual(result.n_adv_estimated, 0)

    def test_both_conventions_agree_at_k2(self):
        """At k=2 there is only ONE off-diagonal value per row, so the
        median of a single-element list is trivially that value itself --
        lower-of-two-middle and average-of-two-middle cannot differ here,
        unlike at k>=4 (see the median-sensitivity diagnostic)."""
        for s in REQUIRED_S_VALUES:
            with self.subTest(s=s):
                production = ri.concentration_stage1_paper(_k2_matrix(s))
                diagnostic = _concentration_stage1_average_median(_k2_matrix(s))
                self.assertEqual(production.n_adv_estimated, diagnostic.n_adv_estimated)
                self.assertAlmostEqual(
                    float(production.s_median[0]), float(diagnostic.s_median[0]), places=9
                )


# ---------------------------------------------------------------------------
# 4. Legacy estimator (authors' released find_num_adv, observationally
#    reproduced via the UNMODIFIED local ragdefender_internals.
#    concentration_stage1 port): pinned behavior at k=2, independent of s.
# ---------------------------------------------------------------------------

class TestLegacyEstimatorPinnedBehaviorAtK2(unittest.TestCase):
    """`ragdefender_legacy`'s Stage-1 estimator (`_find_num_adversarial` /
    `RAGDefender/artifacts/main.py::find_num_adv`) is NOT modified by this
    test file or by the k=2 audit task -- these tests only PIN its
    already-existing, unmodified behavior as an observational reference.
    """

    def test_legacy_flip_fires_and_n_adv_equals_k_for_all_s(self):
        for s in ALL_S_VALUES:
            with self.subTest(s=s):
                result = ri.concentration_stage1(_k2_matrix(s))
                self.assertTrue(result.flipped)
                self.assertEqual(result.n_adv_estimated, 2)
                self.assertFalse(bool(result.above_avg[0]))
                self.assertFalse(bool(result.above_avg[1]))
                self.assertFalse(bool(result.above_median[0]))
                self.assertFalse(bool(result.above_median[1]))

    def test_legacy_avg_and_median_are_diagonal_inclusive_and_identical_across_indices(self):
        """Structural reason: legacy's avg/median are torch.mean/median(dim=0)
        over the FULL (diagonal-included) column -- for k=2 that column is
        {1.0 (self), s (other)} for BOTH indices (by symmetry), so avg[0]
        == avg[1] == (1+s)/2 and median[0] == median[1] == min(1, s), and
        both indices are therefore always on the same side of every
        threshold -- there is no s for which the legacy OR-condition can
        fire at k=2."""
        for s in REQUIRED_S_VALUES:
            with self.subTest(s=s):
                result = ri.concentration_stage1(_k2_matrix(s))
                self.assertAlmostEqual(float(result.avg[0]), float(result.avg[1]), places=9)
                self.assertAlmostEqual(float(result.avg[0]), (1.0 + s) / 2.0, places=9)
                self.assertAlmostEqual(float(result.median[0]), float(result.median[1]), places=9)
                self.assertEqual(int(result.raw_or_flag.sum()), 0)

    def test_legacy_stage2_selects_both_indices_at_k2(self):
        for s in REQUIRED_S_VALUES:
            with self.subTest(s=s):
                stage1 = ri.concentration_stage1(_k2_matrix(s))
                stage2 = ri.stage2_pair_frequency(_k2_matrix(s), n_adv=stage1.n_adv_estimated, p=2.0)
                self.assertEqual(stage2.n_pairs, 1)
                self.assertEqual(set(stage2.selected_indices), {0, 1})


# ---------------------------------------------------------------------------
# 5-6. Full pipeline: returned context at k=2 for BOTH ragdefender_paper
#      and ragdefender_legacy is the full, unfiltered 2-passage input --
#      for two entirely different internal reasons (see test 7).
# ---------------------------------------------------------------------------

class TestFullPipelineReturnedContextAtK2(unittest.TestCase):
    def test_ragdefender_paper_returns_full_unfiltered_context(self):
        doc_list = ["passage zero text content here", "passage one text content here"]
        for s in REQUIRED_S_VALUES:
            with self.subTest(s=s):
                with _patch_lazy_st(s):
                    result = defense_runner._apply_defense_paper(  # noqa: SLF001
                        list(doc_list), mode="multihop", s_model=_FakeEncoderIgnored(), top_k=None
                    )
                self.assertEqual(result, doc_list)

    def test_ragdefender_legacy_returns_full_unfiltered_context_via_restore_all_fallback(self):
        from unittest import mock

        doc_list = ["passage zero text content here", "passage one text content here"]
        for s in REQUIRED_S_VALUES:
            with self.subTest(s=s):
                with _patch_lazy_st(s), mock.patch.object(
                    defense_runner, "_get_s_model", return_value=_FakeEncoderIgnored()
                ):
                    result = defense_runner.apply_defense(
                        "some query", list(doc_list), dataset="hotpotqa",
                        device="cpu", ragdefender_version="legacy",
                    )
                self.assertEqual(result, doc_list)

    def test_ragdefender_paper_top_k_truncation_still_applies_after_noop_filtering(self):
        """Even though Stage 1/2 remove nothing at k=2, an explicit top_k
        argument (the SEPARATE final-truncation step, analogous to the
        authors' released code's `args.top_k` trim -- see STEP 3/5 of the
        k2 report) still applies on top of the (unfiltered) safe context."""
        doc_list = ["passage zero text content here", "passage one text content here"]
        with _patch_lazy_st(0.3):
            result = defense_runner._apply_defense_paper(  # noqa: SLF001
                list(doc_list), mode="multihop", s_model=_FakeEncoderIgnored(), top_k=1
            )
        self.assertEqual(result, doc_list[:1])


class TestFloatPrecisionFragilityIsReal(unittest.TestCase):
    """Independently interesting, real numerical fragility uncovered while
    building this audit (documented in the k2 consistency report, not
    just asserted here): the "legacy always yields N_adv=k=2" corollary
    depends on EXACT floating equality of `avg[0]==avg[1]` and
    `median[0]==median[1]` -- true in exact arithmetic and reproduced
    deterministically via direct float64 matrix construction (see
    `TestLegacyEstimatorPinnedBehaviorAtK2` above), but NOT guaranteed
    once similarity comes from an actual embedding-normalization
    pipeline (float32 `sentence_transformers.util.cos_sim`), because
    self-similarity is then only APPROXIMATELY 1.0 (e.g.
    `1.0000001192` for one row vs. `1.0` for the other), which can break
    the tie and flip `above_avg`/`above_median` asymmetrically for one
    of the two indices. This does not change the k=2 MATHEMATICAL
    corollary (which concerns exact similarity values, not any specific
    floating encoder), but it does mean a literal unit-vector fake
    encoder is NOT a reliable way to test the full pipeline at exact
    ties -- hence `_patch_lazy_st` above, which bypasses embedding
    normalization entirely for the other tests in this file."""

    def test_float32_unit_vector_encoding_can_break_the_exact_legacy_tie(self):
        import torch

        s = 0.8  # the specific value on which this fragility was first observed.
        u = [1.0, 0.0]
        v = [s, math.sqrt(max(0.0, 1.0 - s * s))]
        embeddings = torch.tensor([u, v], dtype=torch.float32)

        from sentence_transformers import util as st_util

        cos_sim_matrix = st_util.cos_sim(embeddings, embeddings)
        # The float32 self-similarity is not exactly 1.0 for both rows --
        # this is the documented source of the tie-break fragility.
        self.assertNotEqual(float(cos_sim_matrix[0, 0]), float(cos_sim_matrix[1, 1]))

        avg = torch.mean(cos_sim_matrix, dim=0)
        # Because avg[0] != avg[1] here (unlike the exact-matrix case),
        # the legacy estimator's OR-condition is NOT guaranteed to stay
        # all-False at k=2 once real (float32) embedding geometry is used.
        self.assertNotEqual(float(avg[0]), float(avg[1]))


# ---------------------------------------------------------------------------
# 7. The two variants' INTERNAL computations provably differ (N_adv=0 vs.
#    N_adv=2, and different Stage-2 selected-index sets pre-fallback) even
#    though their FINAL returned context is identical.
# ---------------------------------------------------------------------------

class TestInternalPathsDifferDespiteIdenticalFinalContext(unittest.TestCase):
    def test_n_adv_differs_between_paper_and_legacy_at_k2(self):
        for s in REQUIRED_S_VALUES:
            with self.subTest(s=s):
                paper = ri.concentration_stage1_paper(_k2_matrix(s))
                legacy = ri.concentration_stage1(_k2_matrix(s))
                self.assertEqual(paper.n_adv_estimated, 0)
                self.assertEqual(legacy.n_adv_estimated, 2)
                self.assertNotEqual(paper.n_adv_estimated, legacy.n_adv_estimated)

    def test_stage2_never_even_invoked_for_paper_but_selects_everything_for_legacy(self):
        """Paper: n_adv<=0 short-circuits before Stage 2 is ever called
        (see `_apply_defense_paper`'s early return). Legacy: Stage 2 IS
        invoked and selects both indices; only the restore-all fallback
        (applied AFTER Stage 2, in `apply_defense`) makes the final
        output match the paper path's output."""
        s = 0.3
        paper = ri.concentration_stage1_paper(_k2_matrix(s))
        self.assertLessEqual(paper.n_adv_estimated, 0)

        legacy = ri.concentration_stage1(_k2_matrix(s))
        stage2_legacy = ri.stage2_pair_frequency(_k2_matrix(s), n_adv=legacy.n_adv_estimated, p=2.0)
        self.assertEqual(set(stage2_legacy.selected_indices), {0, 1})

    def test_final_contexts_agree_even_though_n_adv_and_stage2_do_not(self):
        from unittest import mock

        doc_list = ["passage zero text content here", "passage one text content here"]
        s = 0.3

        with _patch_lazy_st(s):
            paper_result = defense_runner._apply_defense_paper(  # noqa: SLF001
                list(doc_list), mode="multihop", s_model=_FakeEncoderIgnored(), top_k=None
            )
        with _patch_lazy_st(s), mock.patch.object(
            defense_runner, "_get_s_model", return_value=_FakeEncoderIgnored()
        ):
            legacy_result = defense_runner.apply_defense(
                "some query", list(doc_list), dataset="hotpotqa",
                device="cpu", ragdefender_version="legacy",
            )

        # Final outputs agree (both no-ops on this input) ...
        self.assertEqual(paper_result, legacy_result)
        self.assertEqual(paper_result, doc_list)

        # ... but the INTERNAL Stage-1 estimates that produced them do not.
        paper_stage1 = ri.concentration_stage1_paper(_k2_matrix(s))
        legacy_stage1 = ri.concentration_stage1(_k2_matrix(s))
        self.assertNotEqual(paper_stage1.n_adv_estimated, legacy_stage1.n_adv_estimated)


if __name__ == "__main__":
    unittest.main()
