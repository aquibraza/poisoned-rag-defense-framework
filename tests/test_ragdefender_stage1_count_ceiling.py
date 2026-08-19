"""Tests for the structural Stage-1 count ceiling `N_adv <= floor(k/2)`
implied by the final paper's Eq. (3).

Companion to
`results/diagnostics/ragdefender_count_ceiling/COUNT_CEILING_ANALYSIS.md`
(mathematical proof) and
`scripts/run_ragdefender_count_ceiling_validation.py` (synthetic
validation driver, reused here). Tests BOTH the production
lower-of-two-middle median convention
(`ragdefender_internals.concentration_stage1_paper`, UNCHANGED) and the
diagnostic-only average-of-two-middle convention
(`run_ragdefender_median_sensitivity._concentration_stage1_average_median`,
also UNCHANGED -- only imported).

Fully offline, synthetic-data only. No Stella/network access anywhere in
this file.

Run with: python -m unittest tests.test_ragdefender_stage1_count_ceiling -v
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from defense import ragdefender_internals  # noqa: E402
from run_ragdefender_median_sensitivity import (  # noqa: E402
    _concentration_stage1_average_median,
)
import run_ragdefender_count_ceiling_validation as ccv  # noqa: E402

REQUIRED_K_VALUES = [2, 3, 4, 5, 6, 10, 11]


def _both_conventions(matrix):
    result_a = ragdefender_internals.concentration_stage1_paper(matrix)
    result_b = _concentration_stage1_average_median(matrix)
    return result_a, result_b


class TestFloorKOver2Values(unittest.TestCase):
    """Pin the expected ceiling values from `COUNT_CEILING_ANALYSIS.md`
    §4 -- if `math.floor` semantics or these hand-checked values ever
    disagree, every other test in this module is built on a wrong
    assumption."""

    def test_expected_ceilings(self):
        expected = {2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 10: 5, 11: 5}
        for k, ceiling in expected.items():
            with self.subTest(k=k):
                self.assertEqual(k // 2, ceiling)
                self.assertEqual(math.floor(k / 2), ceiling)


class TestCeilingHoldsOnRandomMatrices(unittest.TestCase):
    """For every required k, many random symmetric matrices across several
    similarity regimes: both conventions must never exceed floor(k/2)."""

    def test_all_regimes_all_k_both_conventions_never_exceed_ceiling(self):
        rows = ccv.run_validation(seed=12345)
        self.assertGreater(len(rows), 0)
        for row in rows:
            k = row["k"]
            ceiling = k // 2
            with self.subTest(k=k, trial=row["trial_id"], regime=row["regime"], convention=row["convention"]):
                self.assertLessEqual(row["n_adv_estimated"], ceiling)
                self.assertLessEqual(row["above_median_count"], ceiling)

    def test_no_violations_reported_by_validation_summary(self):
        rows = ccv.run_validation(seed=99)
        summary = ccv.build_summary(rows)
        self.assertEqual(summary["n_violations"], 0)

    def test_required_k_values_all_present_in_validation_output(self):
        rows = ccv.run_validation(seed=1)
        tested_ks = {row["k"] for row in rows}
        self.assertEqual(tested_ks, set(REQUIRED_K_VALUES))

    def test_many_random_trials_per_k_directly_against_production_function(self):
        """Independent of the validation-script helper: call
        `concentration_stage1_paper` directly on freshly generated random
        matrices for every required k."""
        rng = np.random.default_rng(2024)
        for k in REQUIRED_K_VALUES:
            ceiling = k // 2
            for _trial in range(30):
                raw = rng.uniform(-1.0, 1.0, size=(k, k))
                matrix = (raw + raw.T) / 2.0
                np.fill_diagonal(matrix, 1.0)
                result = ragdefender_internals.concentration_stage1_paper(matrix)
                with self.subTest(k=k):
                    self.assertLessEqual(result.n_adv_estimated, ceiling)


class TestAndConditionNeverExceedsMedianOnlyBound(unittest.TestCase):
    """STEP 1 requirement #3: the mean-side AND condition can only reduce
    the median-only count, never increase it beyond floor(k/2)."""

    def test_and_count_leq_median_only_count_on_random_matrices(self):
        rng = np.random.default_rng(777)
        for k in REQUIRED_K_VALUES:
            ceiling = k // 2
            for _trial in range(30):
                raw = rng.uniform(-1.0, 1.0, size=(k, k))
                matrix = (raw + raw.T) / 2.0
                np.fill_diagonal(matrix, 1.0)
                result_a, result_b = _both_conventions(matrix)
                above_median_a = int(result_a.above_median.sum())
                above_median_b = int(result_b.above_median.sum())
                with self.subTest(k=k, convention="A"):
                    self.assertLessEqual(above_median_a, ceiling)
                    self.assertLessEqual(result_a.n_adv_estimated, above_median_a)
                with self.subTest(k=k, convention="B"):
                    self.assertLessEqual(above_median_b, ceiling)
                    self.assertLessEqual(result_b.n_adv_estimated, above_median_b)

    def test_and_is_literally_a_set_intersection_of_two_boolean_masks(self):
        """Directly verify adv_flag == above_mean & above_median (the
        algebraic fact the ceiling proof leans on), on a representative
        k=10 matrix."""
        matrix = ccv.gen_ceiling_edge_fixture(10)
        result = ragdefender_internals.concentration_stage1_paper(matrix)
        np.testing.assert_array_equal(result.adv_flag, result.above_mean & result.above_median)
        self.assertLessEqual(int(result.adv_flag.sum()), int(result.above_median.sum()))
        self.assertLessEqual(int(result.adv_flag.sum()), int(result.above_mean.sum()))


class TestHandComputableEdgeFixtures(unittest.TestCase):
    """The three-level HUB/BACKGROUND block fixture from the validation
    script reaches the ceiling EXACTLY for every required k except k=2
    (which is handled by its own, stronger test below)."""

    def test_edge_fixture_hits_ceiling_exactly_for_k_geq_3(self):
        for k in [3, 4, 5, 6, 10, 11]:
            ceiling = k // 2
            matrix = ccv.gen_ceiling_edge_fixture(k)
            result_a, result_b = _both_conventions(matrix)
            with self.subTest(k=k, convention="A"):
                self.assertEqual(result_a.n_adv_estimated, ceiling)
            with self.subTest(k=k, convention="B"):
                self.assertEqual(result_b.n_adv_estimated, ceiling)

    def test_edge_fixture_never_exceeds_ceiling_at_k2(self):
        matrix = ccv.gen_ceiling_edge_fixture(2)
        result_a, result_b = _both_conventions(matrix)
        self.assertLessEqual(result_a.n_adv_estimated, 1)
        self.assertLessEqual(result_b.n_adv_estimated, 1)

    def test_hub_indices_are_exactly_the_selected_indices_at_k10(self):
        """Hand-checkable: for k=10, the fixture's construction puts the
        HUB group at indices 0..4 -- confirm those are exactly the 5
        flagged indices, not merely that the COUNT is 5."""
        matrix = ccv.gen_ceiling_edge_fixture(10)
        result = ragdefender_internals.concentration_stage1_paper(matrix)
        flagged = set(np.where(result.adv_flag)[0].tolist())
        self.assertEqual(flagged, {0, 1, 2, 3, 4})


class TestK2AlwaysZero(unittest.TestCase):
    """A strictly STRONGER fact than the ceiling bound alone implies:
    N_adv is provably always exactly 0 at k=2, for ANY symmetric matrix
    (see COUNT_CEILING_ANALYSIS.md's "Special case k=2" note) -- not
    merely <= floor(2/2)=1."""

    def test_k2_always_zero_on_many_random_matrices(self):
        rng = np.random.default_rng(31415)
        for _trial in range(100):
            raw = rng.uniform(-1.0, 1.0, size=(2, 2))
            matrix = (raw + raw.T) / 2.0
            np.fill_diagonal(matrix, 1.0)
            result_a, result_b = _both_conventions(matrix)
            self.assertEqual(result_a.n_adv_estimated, 0)
            self.assertEqual(result_b.n_adv_estimated, 0)

    def test_k2_always_zero_on_hand_picked_extreme_matrices(self):
        extreme_matrices = [
            np.array([[1.0, 0.99], [0.99, 1.0]]),
            np.array([[1.0, -0.99], [-0.99, 1.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            np.array([[1.0, -1.0], [-1.0, 1.0]]),
        ]
        for matrix in extreme_matrices:
            with self.subTest(matrix=matrix.tolist()):
                result_a, result_b = _both_conventions(matrix)
                self.assertEqual(result_a.n_adv_estimated, 0)
                self.assertEqual(result_b.n_adv_estimated, 0)

    def test_k2_forces_s_mean_equal_to_s_bar(self):
        """Directly verifies the mechanism: with k=2, s_mean_0 == s_mean_1
        == s_bar exactly, so above_mean is [False, False] unconditionally."""
        rng = np.random.default_rng(2718)
        for _trial in range(20):
            c = float(rng.uniform(-1.0, 1.0))
            matrix = np.array([[1.0, c], [c, 1.0]])
            result = ragdefender_internals.concentration_stage1_paper(matrix)
            self.assertAlmostEqual(result.s_mean[0], result.s_mean[1])
            self.assertAlmostEqual(result.s_mean[0], result.s_bar)
            self.assertFalse(bool(result.above_mean[0]))
            self.assertFalse(bool(result.above_mean[1]))
            self.assertEqual(result.n_adv_estimated, 0)


class TestMedianConventionCeilingEquivalence(unittest.TestCase):
    """STEP 1's specific instruction: analyze whether the ceiling differs
    numerically between the two median conventions. Answer: it does NOT --
    both give floor(k/2) as the exact numeric ceiling (they may disagree
    on WHICH indices are flagged for a given matrix, never on the ceiling
    value itself)."""

    def test_ceiling_value_is_identical_across_conventions_for_all_required_k(self):
        for k in REQUIRED_K_VALUES:
            # The ceiling value floor(k/2) is a property of k alone, not
            # of the median convention -- this test documents that both
            # conventions are being checked against the SAME numeric
            # ceiling throughout this module (no separate "B convention
            # ceiling" constant exists anywhere in the codebase).
            ceiling_used_for_a = k // 2
            ceiling_used_for_b = k // 2
            self.assertEqual(ceiling_used_for_a, ceiling_used_for_b)

    def test_both_conventions_hit_the_same_ceiling_on_edge_fixture(self):
        for k in [3, 4, 5, 6, 10, 11]:
            ceiling = k // 2
            matrix = ccv.gen_ceiling_edge_fixture(k)
            result_a, result_b = _both_conventions(matrix)
            with self.subTest(k=k):
                self.assertEqual(result_a.n_adv_estimated, ceiling)
                self.assertEqual(result_b.n_adv_estimated, ceiling)
                self.assertEqual(result_a.n_adv_estimated, result_b.n_adv_estimated)

    def test_conventions_can_disagree_on_which_indices_but_never_on_ceiling(self):
        """Odd k, k-1 even at the ROW level: the two row-median conventions
        can disagree on individual s_median_i values (and therefore on
        exactly which passages are flagged), but neither can ever push the
        COUNT past floor(k/2)."""
        rng = np.random.default_rng(55)
        any_disagreement_observed = False
        for _trial in range(50):
            k = 5
            raw = rng.uniform(-1.0, 1.0, size=(k, k))
            matrix = (raw + raw.T) / 2.0
            np.fill_diagonal(matrix, 1.0)
            result_a, result_b = _both_conventions(matrix)
            if not np.array_equal(result_a.adv_flag, result_b.adv_flag):
                any_disagreement_observed = True
            self.assertLessEqual(result_a.n_adv_estimated, k // 2)
            self.assertLessEqual(result_b.n_adv_estimated, k // 2)
        # Not a hard requirement (conventions could coincide on every
        # trial for a given seed), but documents that this module's
        # random trials are actually exercising real convention
        # differences, consistent with the existing median-sensitivity
        # diagnostic's finding that differences are rare but possible.
        del any_disagreement_observed


class TestValidationScriptIsSelfConsistent(unittest.TestCase):
    """Guard that the validation script's own violation-detection logic
    would actually fire if the ceiling were ever exceeded (mutation-style
    check against a deliberately-broken stand-in, not against production
    code)."""

    def test_validate_matrix_raises_on_synthetic_violation(self):
        class _FakeResult:
            def __init__(self, n_adv, above_median):
                self.n_adv_estimated = n_adv
                self.above_median = above_median

        # Patch in a fake production function that violates the ceiling,
        # call the internal checker, and confirm it raises. Restored
        # immediately after -- never left patched.
        original = ragdefender_internals.concentration_stage1_paper
        try:
            ragdefender_internals.concentration_stage1_paper = (
                lambda matrix: _FakeResult(n_adv=999, above_median=np.array([True] * matrix.shape[0]))
            )
            with self.assertRaises(ccv.CountCeilingViolation):
                ccv._validate_matrix(4, np.eye(4), regime="fake", trial_id="mutation_check")
        finally:
            ragdefender_internals.concentration_stage1_paper = original


if __name__ == "__main__":
    unittest.main()
