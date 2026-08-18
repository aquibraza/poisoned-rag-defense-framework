"""Tests for scripts/run_ragdefender_median_sensitivity.py -- the Gate-B
follow-up STEP 3 sensitivity diagnostic (lower-of-two-middle vs.
average-of-two-middle median convention, on saved Gate-B Stella matrices).

Fully offline: synthetic-fixture correctness tests, an explicit isolation
guard proving the SENSITIVITY-ONLY variant never leaks into production
code, and a real-artifact smoke check gated on Gate B's saved outputs
existing on disk. No Stella/network access anywhere in this file.

Run with: python -m unittest tests.test_run_ragdefender_median_sensitivity -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import run_ragdefender_median_sensitivity as sens  # noqa: E402

# 6x6 fixture engineered so the two median conventions disagree on at
# least one per-passage median (odd-length row excluded-self => k=6 rows
# of length 5, odd, so per-row medians are identical between conventions;
# the DISAGREEMENT is engineered into the outer median-of-6-medians step,
# which is an even-length (k=6) aggregation).
FIXTURE_MATRIX = np.array(
    [
        [1.00, 0.90, 0.85, 0.20, 0.18, 0.15],
        [0.90, 1.00, 0.88, 0.22, 0.19, 0.16],
        [0.85, 0.88, 1.00, 0.21, 0.17, 0.14],
        [0.20, 0.22, 0.21, 1.00, 0.55, 0.50],
        [0.18, 0.19, 0.17, 0.55, 1.00, 0.52],
        [0.15, 0.16, 0.14, 0.50, 0.52, 1.00],
    ]
)
FIXTURE_IS_POISON = np.array([True, True, True, False, False, False])


class TestAverageMedian1d(unittest.TestCase):
    def test_matches_numpy_median_for_even_length(self):
        values = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(sens._average_median_1d(values), 0.25)

    def test_matches_actual_middle_value_for_odd_length(self):
        values = np.array([0.1, 0.2, 0.3])
        self.assertAlmostEqual(sens._average_median_1d(values), 0.2)

    def test_differs_from_torch_style_on_even_length_with_no_tie(self):
        from defense.ragdefender_internals import _torch_style_median_1d  # noqa: SLF001

        values = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertNotEqual(sens._average_median_1d(values), _torch_style_median_1d(values))
        self.assertAlmostEqual(sens._average_median_1d(values), 0.25)
        self.assertAlmostEqual(_torch_style_median_1d(values), 0.2)


class TestConcentrationStage1AverageMedian(unittest.TestCase):
    def test_self_exclusion_and_and_logic_match_production_structure(self):
        from defense.ragdefender_internals import concentration_stage1_paper

        primary = concentration_stage1_paper(FIXTURE_MATRIX)
        sensitivity = sens._concentration_stage1_average_median(FIXTURE_MATRIX)

        # Same mean (unaffected by the median-convention change).
        np.testing.assert_allclose(primary.s_mean, sensitivity.s_mean)
        self.assertAlmostEqual(primary.s_bar, sensitivity.s_bar)
        # Both are non-negative integer counts on a valid AND-flag sum.
        self.assertGreaterEqual(sensitivity.n_adv_estimated, 0)
        self.assertEqual(sensitivity.n_adv_estimated, int(sensitivity.adv_flag.sum()))

    def test_can_disagree_with_production_convention_on_engineered_fixture(self):
        primary = sens.ragdefender_internals.concentration_stage1_paper(FIXTURE_MATRIX)
        sensitivity = sens._concentration_stage1_average_median(FIXTURE_MATRIX)
        # Not asserting they MUST disagree on every fixture (that would be
        # over-fitting the test to one dataset) -- just that s_tilde
        # legitimately differs when there's no exact tie at the middle,
        # proving this is a real, distinct computation, not an alias.
        self.assertNotEqual(primary.s_tilde, sensitivity.s_tilde)


class TestSensitivityVariantIsIsolated(unittest.TestCase):
    """Guards the explicit requirement: the sensitivity-only variant must
    never leak into production code or become a defense option."""

    def test_average_median_helpers_are_not_exported_by_ragdefender_internals(self):
        from defense import ragdefender_internals

        self.assertFalse(hasattr(ragdefender_internals, "_average_median_1d"))
        self.assertFalse(hasattr(ragdefender_internals, "_concentration_stage1_average_median"))
        self.assertNotIn("_average_median_1d", ragdefender_internals.__all__)

    def test_defense_runner_and_dispatch_do_not_reference_average_median(self):
        from defense import defense_runner, dispatch

        self.assertFalse(hasattr(defense_runner, "_average_median_1d"))
        self.assertFalse(hasattr(dispatch, "_average_median_1d"))
        # No new "--defense" choice was introduced by this diagnostic.
        self.assertNotIn("ragdefender_paper_average_median", dispatch.DEFENSE_CHOICES)
        self.assertNotIn("ragdefender_sensitivity", dispatch.DEFENSE_CHOICES)


@unittest.skipUnless(
    (sens.GATE_B_DIR / "gate_b_per_query.csv").exists(),
    f"Real Gate-B outputs not found on disk: {sens.GATE_B_DIR} (results/ is gitignored; "
    "this test only runs after Gate B has actually been run once).",
)
class TestRealGateBArtifactSmoke(unittest.TestCase):
    """Read-only: loads the real saved Gate-B matrices/labels and confirms
    this script reproduces Gate B's own recorded primary-convention N_adv
    exactly (the script's own internal cross-check), then runs the full
    sensitivity comparison and checks basic structural invariants. Never
    writes into `results/diagnostics/ragdefender_gate_b/`."""

    def test_load_gate_b_cases_matches_saved_composition(self):
        cases = sens.load_gate_b_cases()
        self.assertEqual(len(cases), 8)
        for case in cases:
            self.assertEqual(case["matrix"].shape, (case["k"], case["k"]))
            self.assertEqual(len(case["is_poison"]), case["k"])

    def test_run_sensitivity_reproduces_gate_b_primary_n_adv_and_has_one_row_per_query(self):
        import pandas as pd

        gate_b_df = pd.read_csv(sens.GATE_B_DIR / "gate_b_per_query.csv").set_index("query_id")
        rows = sens.run_sensitivity()
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertEqual(row["n_adv_lower"], int(gate_b_df.loc[row["query_id"], "n_adv"]))
            self.assertEqual(
                row["removed_poison_lower"], int(gate_b_df.loc[row["query_id"], "removed_poison"])
            )


if __name__ == "__main__":
    unittest.main()
