"""Tests for `scripts/run_ragdefender_expanded_gate_c.py` -- STEP 5
expanded ORACLE-COUNT decomposition over the prospective population.

Fully offline: synthetic-fixture correctness tests for
`_run_stage2_metrics`/`_classify_decomposition`/`run_gate_c_query`/
`build_regime_decomposition`/`build_count_error_analysis`, plus a
real-artifact structural check gated on the expanded baseline (STEP 4)
having actually been run. No Stella/network access anywhere in this file.

Run with: python -m unittest tests.test_run_ragdefender_expanded_gate_c -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import run_ragdefender_expanded_gate_c as gc  # noqa: E402

# Reuse the same synthetic fixtures as the original (n=8) Gate-C tests.
COUNT_LIMITED_MATRIX = np.array([
    [1.00, 0.92, 0.90, 0.10, 0.08],
    [0.92, 1.00, 0.91, 0.09, 0.07],
    [0.90, 0.91, 1.00, 0.11, 0.09],
    [0.10, 0.09, 0.11, 1.00, 0.30],
    [0.08, 0.07, 0.09, 0.30, 1.00],
])
COUNT_LIMITED_IS_POISON = np.array([True, True, True, False, False])
COUNT_LIMITED_TRUE_POISON_COUNT = 3
COUNT_LIMITED_UNDERESTIMATE = 2

IDENTIFICATION_LIMITED_MATRIX = np.array([
    [1.00, 0.50, 0.05, 0.05],
    [0.50, 1.00, 0.05, 0.05],
    [0.05, 0.05, 1.00, 0.95],
    [0.05, 0.05, 0.95, 1.00],
])
IDENTIFICATION_LIMITED_IS_POISON = np.array([True, True, False, False])
IDENTIFICATION_LIMITED_TRUE_POISON_COUNT = 2


class TestRunStage2Metrics(unittest.TestCase):
    def test_underestimated_count_leaves_one_poison_passage_unremoved(self):
        metrics = gc._run_stage2_metrics(
            COUNT_LIMITED_MATRIX, COUNT_LIMITED_IS_POISON,
            n_adv=COUNT_LIMITED_UNDERESTIMATE, true_poison_count=COUNT_LIMITED_TRUE_POISON_COUNT,
        )
        self.assertEqual(metrics["removed_poison"], 2)
        self.assertEqual(metrics["removed_clean"], 0)
        self.assertEqual(metrics["residual_poison"], 1)

    def test_correct_count_removes_all_three_poison_with_no_clean_cost(self):
        metrics = gc._run_stage2_metrics(
            COUNT_LIMITED_MATRIX, COUNT_LIMITED_IS_POISON,
            n_adv=COUNT_LIMITED_TRUE_POISON_COUNT, true_poison_count=COUNT_LIMITED_TRUE_POISON_COUNT,
        )
        self.assertEqual(metrics["removed_poison"], 3)
        self.assertEqual(metrics["removed_clean"], 0)
        self.assertEqual(metrics["residual_poison"], 0)


class TestClassifyDecomposition(unittest.TestCase):
    def test_only_the_four_allowed_labels_are_ever_produced(self):
        import itertools

        for residual_poison, count_error, oracle_residual, oracle_clean in itertools.product(
            [0, 1, 2], [-1, 0, 1], [0, 1, 2], [0, 1, 2]
        ):
            estimated = {"residual_poison": residual_poison, "count_error": count_error}
            oracle = {"residual_poison": oracle_residual, "removed_clean": oracle_clean}
            label = gc._classify_decomposition(estimated, oracle)
            self.assertIn(label, gc.VALID_LABELS)

    def test_count_limited_when_oracle_fixes_with_no_clean_cost(self):
        estimated = {"residual_poison": 1, "count_error": -1}
        oracle = {"residual_poison": 0, "removed_clean": 0}
        self.assertEqual(gc._classify_decomposition(estimated, oracle), "A. COUNT-LIMITED")


class TestRunGateCQuery(unittest.TestCase):
    """End-to-end (synthetic, offline) through `run_gate_c_query`."""

    def _case(self, matrix, is_poison, true_poison_count, c_clean, regime="C_ABOVE_CEILING"):
        from defense import ragdefender_internals

        stage1 = ragdefender_internals.concentration_stage1_paper(matrix)
        return {
            "query_id": "synthetic_test_query",
            "matrix": matrix,
            "is_poison": is_poison,
            "m_poison": true_poison_count,
            "c_clean": c_clean,
            "k": matrix.shape[0],
            "regime": regime,
            "baseline_n_adv": stage1.n_adv_estimated,
            "top_pair_pp": None,
        }

    def test_count_limited_case_end_to_end(self):
        case = self._case(COUNT_LIMITED_MATRIX, COUNT_LIMITED_IS_POISON, COUNT_LIMITED_TRUE_POISON_COUNT, c_clean=2)
        result = gc.run_gate_c_query(case)
        self.assertEqual(result["m_poison"], 3)
        self.assertEqual(result["oracle_residual_poison"], 0)
        self.assertEqual(result["oracle_removed_clean"], 0)
        self.assertIn(result["decomposition_label"], ("A. COUNT-LIMITED", "D. BASELINE SUCCESS"))

    def test_stop_condition_raised_if_baseline_n_adv_mismatches_recomputed_stage1(self):
        case = self._case(COUNT_LIMITED_MATRIX, COUNT_LIMITED_IS_POISON, COUNT_LIMITED_TRUE_POISON_COUNT, c_clean=2)
        case["baseline_n_adv"] = 999
        with self.assertRaises(gc.ExpandedGateCStopCondition):
            gc.run_gate_c_query(case)

    def test_oracle_receives_only_the_count_never_passage_identities(self):
        """Structural guard: the oracle call in `run_gate_c_query` passes
        `n_adv=true_poison_count` (an int, the observed M), and
        `stage2_pair_frequency`'s signature exposes no parameter through
        which specific passage identities could be injected."""
        import inspect

        sig = inspect.signature(gc.ragdefender_internals.stage2_pair_frequency)
        self.assertEqual(list(sig.parameters), ["cos_sim_matrix", "n_adv", "p"])

    def test_delta_n_equals_n_adv_minus_m(self):
        case = self._case(COUNT_LIMITED_MATRIX, COUNT_LIMITED_IS_POISON, COUNT_LIMITED_TRUE_POISON_COUNT, c_clean=2)
        result = gc.run_gate_c_query(case)
        self.assertEqual(result["delta_N"], result["estimated_N_adv"] - result["m_poison"])


class TestBuildRegimeDecomposition(unittest.TestCase):
    def _row(self, regime, label, residual_poison=0, fixes=False, clean_intro=False):
        return {
            "regime": regime,
            "decomposition_label": label,
            "estimated_residual_poison": residual_poison,
            "oracle_count_fixes_failure": fixes,
            "oracle_residual_poison": 0,
            "oracle_count_introduces_clean_removal": clean_intro,
        }

    def test_all_four_regimes_present_even_if_empty(self):
        rows = [self._row("B_AT_CEILING", "D. BASELINE SUCCESS")]
        aggregates = gc.build_regime_decomposition(rows)
        regimes = [a["regime"] for a in aggregates]
        self.assertEqual(regimes, gc.REGIME_ORDER)
        regime_a = next(a for a in aggregates if a["regime"] == "A_BELOW_CEILING")
        self.assertEqual(regime_a["n_queries"], 0)

    def test_label_counts_correct_within_regime(self):
        rows = [
            self._row("C_ABOVE_CEILING", "A. COUNT-LIMITED", residual_poison=1, fixes=True),
            self._row("C_ABOVE_CEILING", "A. COUNT-LIMITED", residual_poison=1, fixes=True),
            self._row("C_ABOVE_CEILING", "D. BASELINE SUCCESS", residual_poison=0),
        ]
        aggregates = gc.build_regime_decomposition(rows)
        regime_c = next(a for a in aggregates if a["regime"] == "C_ABOVE_CEILING")
        self.assertEqual(regime_c["n_queries"], 3)
        self.assertEqual(regime_c["n_label_A"], 2)
        self.assertEqual(regime_c["n_label_D"], 1)
        self.assertEqual(regime_c["n_estimated_failures"], 2)
        self.assertEqual(regime_c["n_failures_fixed_by_oracle_count"], 2)
        self.assertAlmostEqual(regime_c["fraction_failures_fixed"], 1.0)

    def test_fraction_failures_fixed_is_none_when_zero_failures(self):
        rows = [self._row("B_AT_CEILING", "D. BASELINE SUCCESS", residual_poison=0)]
        aggregates = gc.build_regime_decomposition(rows)
        regime_b = next(a for a in aggregates if a["regime"] == "B_AT_CEILING")
        self.assertIsNone(regime_b["fraction_failures_fixed"])


class TestBuildCountErrorAnalysis(unittest.TestCase):
    def _row(self, delta_n, residual_poison):
        return {"delta_N": delta_n, "abs_delta_N": abs(delta_n), "estimated_residual_poison": residual_poison}

    def test_distribution_and_conditional_probabilities(self):
        rows = [
            self._row(-1, 1),  # delta<0, residual>0
            self._row(-1, 1),  # delta<0, residual>0
            self._row(0, 0),  # delta==0, residual==0
            self._row(0, 0),  # delta==0, residual==0
            self._row(1, 0),  # delta>0, residual==0
        ]
        analysis = gc.build_count_error_analysis(rows)
        self.assertEqual(analysis["delta_n_distribution"], {-1: 2, 0: 2, 1: 1})
        self.assertAlmostEqual(analysis["mean_delta_n"], -1 / 5)

        p_neg = analysis["p_residual_given_delta_n_negative"]
        self.assertEqual(p_neg["n"], 2)
        self.assertEqual(p_neg["n_residual_positive"], 2)
        self.assertAlmostEqual(p_neg["p"], 1.0)

        p_zero = analysis["p_residual_given_delta_n_zero"]
        self.assertEqual(p_zero["n"], 2)
        self.assertEqual(p_zero["n_residual_positive"], 0)
        self.assertAlmostEqual(p_zero["p"], 0.0)

    def test_conditional_probability_none_when_bucket_empty(self):
        rows = [self._row(-1, 1)]
        analysis = gc.build_count_error_analysis(rows)
        self.assertIsNone(analysis["p_residual_given_delta_n_zero"])
        self.assertIsNone(analysis["p_residual_given_delta_n_positive"])

    def test_cross_tab_counts_are_raw_never_only_rates(self):
        rows = [self._row(-2, 1), self._row(-1, 0)]
        analysis = gc.build_count_error_analysis(rows)
        self.assertIn("cross_tab", analysis)
        self.assertEqual(analysis["cross_tab"]["delta_N<0"]["residual_poison>0"], 1)
        self.assertEqual(analysis["cross_tab"]["delta_N<0"]["residual_poison==0"], 1)


class TestNoOverwriteSafeguard(unittest.TestCase):
    def test_check_no_overwrite_raises_if_any_path_exists(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            existing = Path(tmpdir) / "already_here.csv"
            existing.write_text("x")
            with self.assertRaises(gc.ExpandedGateCStopCondition):
                gc._check_no_overwrite([existing])

    def test_check_no_overwrite_passes_when_none_exist(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "not_here.csv"
            gc._check_no_overwrite([missing])  # should not raise


@unittest.skipUnless(
    (gc.BASELINE_DIR / "expanded_baseline_per_query.csv").exists(),
    f"Real expanded-baseline output not found on disk: {gc.BASELINE_DIR} "
    "(results/ is gitignored; this test only runs after STEP 4 has actually been run once).",
)
class TestRealExpandedGateCEndToEnd(unittest.TestCase):
    """Read-only against the real saved STEP-4 outputs; never writes into
    `ragdefender_expanded_baseline/` or overwrites Gate A/B/C artifacts."""

    def test_run_expanded_gate_c_reproduces_baseline_n_adv_exactly(self):
        import pandas as pd

        baseline_df = pd.read_csv(gc.BASELINE_DIR / "expanded_baseline_per_query.csv").set_index("query_id")
        rows = gc.run_expanded_gate_c()
        self.assertEqual(len(rows), len(baseline_df))
        for row in rows:
            baseline_row = baseline_df.loc[row["query_id"]]
            self.assertEqual(row["estimated_N_adv"], int(baseline_row["n_adv"]))
            self.assertIn(row["decomposition_label"], gc.VALID_LABELS)

    def test_no_baseline_output_files_are_modified_by_a_dry_run(self):
        import hashlib

        def _hash(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        baseline_csv = gc.BASELINE_DIR / "expanded_baseline_per_query.csv"
        before = _hash(baseline_csv)
        gc.run_expanded_gate_c()
        after = _hash(baseline_csv)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
