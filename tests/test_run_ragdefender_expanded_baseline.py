"""Tests for `scripts/run_ragdefender_expanded_baseline.py` -- STEP 4
expanded paper-faithful baseline.

Fully offline: synthetic-fixture tests for `_classify_pair`,
`build_regime_aggregates`, the Regime-C invariant check, and the
no-overwrite safeguard, plus real-artifact structural checks gated on the
expanded baseline having actually been run once. No Stella/network access
anywhere in this file.

Run with: python -m unittest tests.test_run_ragdefender_expanded_baseline -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import run_ragdefender_expanded_baseline as baseline  # noqa: E402


class TestClassifyPair(unittest.TestCase):
    def test_pp(self):
        is_poison = np.array([True, True, False])
        self.assertEqual(baseline._classify_pair(0, 1, is_poison), "PP")

    def test_cc(self):
        is_poison = np.array([True, False, False])
        self.assertEqual(baseline._classify_pair(1, 2, is_poison), "CC")

    def test_pc(self):
        is_poison = np.array([True, False, False])
        self.assertEqual(baseline._classify_pair(0, 1, is_poison), "PC")
        self.assertEqual(baseline._classify_pair(1, 0, is_poison), "PC")


def _fake_row(query_id, regime, m_poison, n_adv, removal_precision=1.0, poison_removal_recall=1.0, clean_removed=False):
    count_error = n_adv - m_poison
    return {
        "query_id": query_id,
        "regime": regime,
        "m_poison": m_poison,
        "n_adv": n_adv,
        "ceiling": 5,
        "count_error": count_error,
        "count_exact": count_error == 0,
        "count_underestimated": count_error < 0,
        "count_overestimated": count_error > 0,
        "zero_residual_poison_success": count_error >= 0,
        "removal_precision": removal_precision,
        "poison_removal_recall": poison_removal_recall,
        "clean_removed": clean_removed,
    }


class TestBuildRegimeAggregates(unittest.TestCase):
    def test_empty_regime_reports_zero_n_queries(self):
        rows = [_fake_row("q1", "B_AT_CEILING", 5, 5)]
        aggregates = baseline.build_regime_aggregates(rows)
        regime_a = next(a for a in aggregates if a["regime"] == "A_BELOW_CEILING")
        self.assertEqual(regime_a["n_queries"], 0)
        self.assertNotIn("mean_n_adv", regime_a)

    def test_aggregate_metrics_computed_correctly(self):
        rows = [
            _fake_row("q1", "B_AT_CEILING", 5, 5),  # exact
            _fake_row("q2", "B_AT_CEILING", 5, 4),  # undercount
        ]
        aggregates = baseline.build_regime_aggregates(rows)
        regime_b = next(a for a in aggregates if a["regime"] == "B_AT_CEILING")
        self.assertEqual(regime_b["n_queries"], 2)
        self.assertAlmostEqual(regime_b["exact_count_rate"], 0.5)
        self.assertAlmostEqual(regime_b["undercount_rate"], 0.5)
        self.assertAlmostEqual(regime_b["mean_signed_count_error"], -0.5)

    def test_regime_c_invariant_check_passes_when_n_adv_leq_ceiling_lt_m(self):
        # ceiling=5 (hardcoded in _fake_row), M=7 > 5 -- valid Regime C.
        rows = [_fake_row("q1", "C_ABOVE_CEILING", 7, 5)]
        # Should not raise.
        aggregates = baseline.build_regime_aggregates(rows)
        regime_c = next(a for a in aggregates if a["regime"] == "C_ABOVE_CEILING")
        self.assertEqual(regime_c["n_queries"], 1)

    def test_regime_c_invariant_check_raises_on_violation(self):
        # Deliberately construct a Regime-C row where n_adv > ceiling --
        # must raise ExpandedBaselineStopCondition, never silently pass.
        rows = [_fake_row("q1", "C_ABOVE_CEILING", 7, 6)]  # n_adv=6 > ceiling=5
        with self.assertRaises(baseline.ExpandedBaselineStopCondition):
            baseline.build_regime_aggregates(rows)

    def test_regime_c_invariant_check_raises_if_m_not_above_ceiling(self):
        # M=5 == ceiling, not > ceiling -- violates "ceiling < m_poison".
        rows = [_fake_row("q1", "C_ABOVE_CEILING", 5, 5)]
        with self.assertRaises(baseline.ExpandedBaselineStopCondition):
            baseline.build_regime_aggregates(rows)

    def test_all_four_regimes_always_present_in_output_order(self):
        rows = [_fake_row("q1", "D_ALL_POISON", 10, 5)]
        aggregates = baseline.build_regime_aggregates(rows)
        regimes = [a["regime"] for a in aggregates]
        self.assertEqual(regimes, baseline.REGIME_ORDER)


class TestWriteCsvHandlesHeterogeneousRowKeys(unittest.TestCase):
    """Regression test: `build_regime_aggregates` can legitimately put an
    EMPTY regime (fewer keys) before a POPULATED regime (more keys) in
    `REGIME_ORDER` (Regime A has zero representation in the real
    prospective population). `_write_csv` must use the union of keys
    across ALL rows, not merely `rows[0].keys()`, or `csv.DictWriter`
    raises `ValueError: dict contains fields not in fieldnames`."""

    def test_write_csv_with_empty_regime_first_then_populated_regime(self):
        import csv
        import tempfile
        from pathlib import Path

        rows = [
            {"regime": "A_BELOW_CEILING", "n_queries": 0},
            {"regime": "B_AT_CEILING", "n_queries": 2, "mean_n_adv": 4.5, "exact_count_rate": 0.5},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.csv"
            baseline._write_csv(rows, path)  # must not raise
            with open(path, newline="") as f:
                reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 2)
            self.assertEqual(reader[0]["regime"], "A_BELOW_CEILING")
            self.assertEqual(reader[1]["mean_n_adv"], "4.5")

    def test_real_regime_aggregates_output_is_csv_writable(self):
        """End-to-end through the actual aggregation function with the
        real-world shape (Regime A empty, others populated) -- exercises
        the exact scenario that crashed the real 42-query run."""
        rows = [
            _fake_row("q1", "B_AT_CEILING", 5, 5),
            _fake_row("q2", "C_ABOVE_CEILING", 7, 5),
            _fake_row("q3", "D_ALL_POISON", 10, 5),
        ]
        aggregates = baseline.build_regime_aggregates(rows)
        self.assertEqual(aggregates[0]["regime"], "A_BELOW_CEILING")
        self.assertEqual(aggregates[0]["n_queries"], 0)
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "by_regime.csv"
            baseline._write_csv(aggregates, path)  # must not raise


class TestNoOverwriteSafeguard(unittest.TestCase):
    def test_check_no_overwrite_raises_if_any_path_exists(self, tmp_path=None):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            existing = Path(tmpdir) / "already_here.csv"
            existing.write_text("x")
            missing = Path(tmpdir) / "not_here.csv"
            with self.assertRaises(baseline.ExpandedBaselineStopCondition):
                baseline._check_no_overwrite([existing, missing])

    def test_check_no_overwrite_passes_when_none_exist(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            a = Path(tmpdir) / "a.csv"
            b = Path(tmpdir) / "b.csv"
            baseline._check_no_overwrite([a, b])  # should not raise


class TestCountErrorMetricsLogic(unittest.TestCase):
    """Directly pins the count-error boolean/sign conventions used
    throughout STEP 4 (count_error = N_adv - M; exactly one of
    exact/under/over is ever true)."""

    def test_exactly_one_of_exact_under_over_is_true(self):
        for m_poison in range(0, 8):
            for n_adv in range(0, 6):
                row = _fake_row("q", "B_AT_CEILING", m_poison, n_adv)
                flags = [row["count_exact"], row["count_underestimated"], row["count_overestimated"]]
                with self.subTest(m_poison=m_poison, n_adv=n_adv):
                    self.assertEqual(sum(flags), 1)

    def test_count_error_sign_matches_under_over_flags(self):
        row_under = _fake_row("q", "B_AT_CEILING", 5, 3)
        self.assertLess(row_under["count_error"], 0)
        self.assertTrue(row_under["count_underestimated"])

        row_over = _fake_row("q", "B_AT_CEILING", 3, 5)
        self.assertGreater(row_over["count_error"], 0)
        self.assertTrue(row_over["count_overestimated"])

        row_exact = _fake_row("q", "B_AT_CEILING", 5, 5)
        self.assertEqual(row_exact["count_error"], 0)
        self.assertTrue(row_exact["count_exact"])


@unittest.skipUnless(
    (baseline.OUTPUT_DIR / "expanded_baseline_per_query.csv").exists(),
    f"Real expanded-baseline output not found on disk: {baseline.OUTPUT_DIR} "
    "(results/ is gitignored; this test only runs after STEP 4 has actually been run once).",
)
class TestRealExpandedBaselineArtifactStructuralChecks(unittest.TestCase):
    """Read-only structural checks against the real saved STEP-4 outputs.
    Never re-runs Stella, never writes anywhere."""

    def test_all_rows_respect_the_structural_ceiling(self):
        import pandas as pd

        df = pd.read_csv(baseline.OUTPUT_DIR / "expanded_baseline_per_query.csv")
        for _, row in df.iterrows():
            with self.subTest(query_id=row["query_id"]):
                self.assertLessEqual(row["n_adv"], row["k"] // 2)

    def test_all_rows_have_valid_regime_labels(self):
        import pandas as pd

        df = pd.read_csv(baseline.OUTPUT_DIR / "expanded_baseline_per_query.csv")
        allowed = set(baseline.REGIME_ORDER)
        self.assertTrue(set(df["regime"].unique()).issubset(allowed))

    def test_regime_c_rows_satisfy_n_adv_leq_ceiling_lt_m(self):
        import pandas as pd

        df = pd.read_csv(baseline.OUTPUT_DIR / "expanded_baseline_per_query.csv")
        regime_c = df[df["regime"] == "C_ABOVE_CEILING"]
        for _, row in regime_c.iterrows():
            with self.subTest(query_id=row["query_id"]):
                self.assertLessEqual(row["n_adv"], row["ceiling"])
                self.assertLess(row["ceiling"], row["m_poison"])

    def test_all_queries_are_k10_only(self):
        import pandas as pd

        df = pd.read_csv(baseline.OUTPUT_DIR / "expanded_baseline_per_query.csv")
        self.assertTrue((df["k"] == 10).all())


if __name__ == "__main__":
    unittest.main()
