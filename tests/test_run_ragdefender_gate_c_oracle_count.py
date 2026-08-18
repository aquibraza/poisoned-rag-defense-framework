"""Tests for scripts/run_ragdefender_gate_c_oracle_count.py -- the Gate-B
follow-up STEP 4 ORACLE-COUNT decomposition diagnostic.

Fully offline: synthetic-fixture correctness tests for
`_run_stage2_metrics`/`_classify_decomposition`/`run_gate_c_query`, plus a
real-artifact structural check gated on Gate B's saved outputs existing on
disk. No Stella/network access anywhere in this file.

Run with: python -m unittest tests.test_run_ragdefender_gate_c_oracle_count -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import run_ragdefender_gate_c_oracle_count as gate_c  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic fixture A: a tight 3-poison cluster + a looser 2-clean cluster.
# Designed so an UNDER-estimated count (2, when the true poison count is 3)
# still selects only within-poison-cluster passages -- i.e. count
# underestimation alone causes one residual poison, and correcting the
# count to 3 (oracle) removes all three poison passages with zero clean
# cost.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Synthetic fixture B: 2 poison passages with only MODERATE mutual
# similarity, and 2 clean passages with a HIGHER mutual similarity. Even
# with the CORRECT count (2) supplied directly, Stage 2's top-similarity
# pair is the clean-clean pair, so it selects the wrong (clean) passages
# -- a pure Stage-2 identification error, count_error == 0.
# ---------------------------------------------------------------------------
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
        metrics = gate_c._run_stage2_metrics(
            COUNT_LIMITED_MATRIX, COUNT_LIMITED_IS_POISON,
            n_adv=COUNT_LIMITED_UNDERESTIMATE, true_poison_count=COUNT_LIMITED_TRUE_POISON_COUNT,
        )
        self.assertEqual(metrics["removed_poison"], 2)
        self.assertEqual(metrics["removed_clean"], 0)
        self.assertEqual(metrics["residual_poison"], 1)

    def test_correct_count_removes_all_three_poison_with_no_clean_cost(self):
        metrics = gate_c._run_stage2_metrics(
            COUNT_LIMITED_MATRIX, COUNT_LIMITED_IS_POISON,
            n_adv=COUNT_LIMITED_TRUE_POISON_COUNT, true_poison_count=COUNT_LIMITED_TRUE_POISON_COUNT,
        )
        self.assertEqual(metrics["removed_poison"], 3)
        self.assertEqual(metrics["removed_clean"], 0)
        self.assertEqual(metrics["residual_poison"], 0)

    def test_precision_and_recall_calculations(self):
        # 2 poison, 0 clean removed out of 3 true poison: precision = 2/2
        # = 1.0 (defined, since denom > 0); recall = 2/3.
        metrics = gate_c._run_stage2_metrics(
            COUNT_LIMITED_MATRIX, COUNT_LIMITED_IS_POISON,
            n_adv=COUNT_LIMITED_UNDERESTIMATE, true_poison_count=COUNT_LIMITED_TRUE_POISON_COUNT,
        )
        self.assertAlmostEqual(metrics["removal_precision"], 1.0)
        self.assertAlmostEqual(metrics["poison_recall"], 2 / 3)

    def test_precision_is_none_when_nothing_removed(self):
        metrics = gate_c._run_stage2_metrics(
            COUNT_LIMITED_MATRIX, COUNT_LIMITED_IS_POISON,
            n_adv=0, true_poison_count=COUNT_LIMITED_TRUE_POISON_COUNT,
        )
        self.assertEqual(metrics["removed_poison"], 0)
        self.assertEqual(metrics["removed_clean"], 0)
        self.assertIsNone(metrics["removal_precision"])
        self.assertAlmostEqual(metrics["poison_recall"], 0.0)

    def test_identification_error_selects_clean_pair_despite_correct_count(self):
        metrics = gate_c._run_stage2_metrics(
            IDENTIFICATION_LIMITED_MATRIX, IDENTIFICATION_LIMITED_IS_POISON,
            n_adv=IDENTIFICATION_LIMITED_TRUE_POISON_COUNT,
            true_poison_count=IDENTIFICATION_LIMITED_TRUE_POISON_COUNT,
        )
        self.assertEqual(metrics["removed_poison"], 0)
        self.assertEqual(metrics["removed_clean"], 2)
        self.assertEqual(metrics["residual_poison"], 2)


class TestClassifyDecomposition(unittest.TestCase):
    """Exercises the exact priority-ordered decision tree over the four
    allowed labels."""

    def test_baseline_success_when_estimated_pipeline_already_succeeds(self):
        estimated = {"residual_poison": 0, "count_error": -1}
        oracle = {"residual_poison": 0, "removed_clean": 0}
        self.assertEqual(gate_c._classify_decomposition(estimated, oracle), "D. BASELINE SUCCESS")

    def test_count_limited_when_oracle_fixes_with_no_clean_cost(self):
        estimated = {"residual_poison": 1, "count_error": -1}
        oracle = {"residual_poison": 0, "removed_clean": 0}
        self.assertEqual(gate_c._classify_decomposition(estimated, oracle), "A. COUNT-LIMITED")

    def test_identification_limited_when_count_error_is_zero_and_still_fails(self):
        estimated = {"residual_poison": 2, "count_error": 0}
        oracle = {"residual_poison": 2, "removed_clean": 2}  # identical to estimated by construction
        self.assertEqual(gate_c._classify_decomposition(estimated, oracle), "C. IDENTIFICATION LIMITED")

    def test_count_plus_identification_limited_when_oracle_helps_but_not_fully(self):
        estimated = {"residual_poison": 2, "count_error": -2}
        oracle = {"residual_poison": 1, "removed_clean": 1}  # improved, but still leaves poison/clean cost
        self.assertEqual(gate_c._classify_decomposition(estimated, oracle), "B. COUNT + IDENTIFICATION LIMITED")

    def test_count_plus_identification_limited_when_oracle_removes_clean_even_if_poison_fully_removed(self):
        estimated = {"residual_poison": 1, "count_error": -1}
        oracle = {"residual_poison": 0, "removed_clean": 1}  # succeeds on poison but at a clean-removal cost
        self.assertEqual(gate_c._classify_decomposition(estimated, oracle), "B. COUNT + IDENTIFICATION LIMITED")

    def test_only_the_four_allowed_labels_are_ever_produced(self):
        import itertools

        for residual_poison, count_error, oracle_residual, oracle_clean in itertools.product(
            [0, 1, 2], [-1, 0, 1], [0, 1, 2], [0, 1, 2]
        ):
            estimated = {"residual_poison": residual_poison, "count_error": count_error}
            oracle = {"residual_poison": oracle_residual, "removed_clean": oracle_clean}
            label = gate_c._classify_decomposition(estimated, oracle)
            self.assertIn(label, gate_c.VALID_LABELS)


class TestRunGateCQuery(unittest.TestCase):
    """End-to-end (still synthetic, still offline) through the full
    `run_gate_c_query` entry point, including its own Stage-1
    cross-check against a caller-supplied `gate_b_n_adv`."""

    def _case(self, matrix, is_poison, true_poison_count, n_retrieved_clean):
        from defense import ragdefender_internals

        stage1 = ragdefender_internals.concentration_stage1_paper(matrix)
        return {
            "query_id": "synthetic_test_query",
            "matrix": matrix,
            "is_poison": is_poison,
            "n_retrieved_poison": true_poison_count,
            "n_retrieved_clean": n_retrieved_clean,
            "k": matrix.shape[0],
            "gate_b_n_adv": stage1.n_adv_estimated,
        }

    def test_count_limited_case_end_to_end(self):
        case = self._case(
            COUNT_LIMITED_MATRIX, COUNT_LIMITED_IS_POISON, COUNT_LIMITED_TRUE_POISON_COUNT, n_retrieved_clean=2
        )
        result = gate_c.run_gate_c_query(case)
        self.assertEqual(result["true_poison_count"], 3)
        # Whatever Stage 1 estimates on this matrix, the oracle pipeline
        # (fed the TRUE count, 3) must remove all 3 poison passages with
        # zero clean cost -- the matrix was engineered for exactly this.
        self.assertEqual(result["oracle_residual_poison"], 0)
        self.assertEqual(result["oracle_removed_clean"], 0)
        self.assertIn(result["decomposition_label"], ("A. COUNT-LIMITED", "D. BASELINE SUCCESS"))

    def test_stop_condition_raised_if_gate_b_n_adv_mismatches_recomputed_stage1(self):
        case = self._case(
            COUNT_LIMITED_MATRIX, COUNT_LIMITED_IS_POISON, COUNT_LIMITED_TRUE_POISON_COUNT, n_retrieved_clean=2
        )
        case["gate_b_n_adv"] = 999  # deliberately wrong
        with self.assertRaises(gate_c.GateCStopCondition):
            gate_c.run_gate_c_query(case)

    def test_oracle_never_uses_passage_identities_only_the_count(self):
        """Structural guard: `run_gate_c_query`'s oracle call must pass
        `n_adv=true_poison_count` (an integer) to
        `stage2_pair_frequency`, and that function's signature has no
        parameter through which specific passage identities could be
        injected -- the oracle mechanically cannot "know" which passages
        are poison beyond the count."""
        import inspect

        sig = inspect.signature(gate_c.ragdefender_internals.stage2_pair_frequency)
        self.assertEqual(list(sig.parameters), ["cos_sim_matrix", "n_adv", "p"])


@unittest.skipUnless(
    (gate_c.GATE_B_DIR / "gate_b_per_query.csv").exists(),
    f"Real Gate-B outputs not found on disk: {gate_c.GATE_B_DIR} (results/ is gitignored; "
    "this test only runs after Gate B has actually been run once).",
)
class TestRealGateBArtifactStructuralChecks(unittest.TestCase):
    """Read-only structural checks against the real saved Gate-B
    artifacts. Never writes into
    `results/diagnostics/ragdefender_gate_b/`."""

    def test_load_gate_b_cases_matches_saved_composition(self):
        cases = gate_c.load_gate_b_cases()
        self.assertEqual(len(cases), 8)
        for case in cases:
            self.assertEqual(case["matrix"].shape, (case["k"], case["k"]))
            self.assertEqual(len(case["is_poison"]), case["k"])
            self.assertEqual(int(case["is_poison"].sum()), case["n_retrieved_poison"])

    def test_run_gate_c_reproduces_gate_b_estimated_pipeline_exactly(self):
        import pandas as pd

        gate_b_df = pd.read_csv(gate_c.GATE_B_DIR / "gate_b_per_query.csv").set_index("query_id")
        rows = gate_c.run_gate_c()
        self.assertEqual(len(rows), 8)
        for row in rows:
            gate_b_row = gate_b_df.loc[row["query_id"]]
            self.assertEqual(row["estimated_N_adv"], int(gate_b_row["n_adv"]))
            self.assertEqual(row["estimated_removed_poison"], int(gate_b_row["removed_poison"]))
            self.assertEqual(row["estimated_removed_clean"], int(gate_b_row["removed_clean"]))
            self.assertEqual(row["estimated_residual_poison"], int(gate_b_row["residual_poison"]))
            self.assertIn(row["decomposition_label"], gate_c.VALID_LABELS)

    def test_no_gate_a_or_gate_b_output_files_are_modified_by_a_dry_run(self):
        import hashlib

        def _hash(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        gate_b_csv = gate_c.GATE_B_DIR / "gate_b_per_query.csv"
        before = _hash(gate_b_csv)
        gate_c.run_gate_c()
        after = _hash(gate_b_csv)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
