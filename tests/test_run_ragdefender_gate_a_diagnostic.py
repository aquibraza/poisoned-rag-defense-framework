"""Tests for scripts/run_ragdefender_gate_a_diagnostic.py -- the Gate A
Stage-1 logic-isolation diagnostic (legacy vs. paper Stage 1 on the same
saved MiniLM similarity matrix).

Fully offline and model-free: exercises only the pure aggregation/diff logic
against a small synthetic fixture built with `defense.ragdefender_internals`
directly (the same estimators the real diagnostic calls), plus a smoke check
that the real saved-artifacts loader can read one real case from the
on-disk `ragdefender_cluster_viz` diagnostics run, if present. No
sentence-transformers/Stella/network access.

Run with: python -m unittest tests.test_run_ragdefender_gate_a_diagnostic -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import run_ragdefender_gate_a_diagnostic as gate_a  # noqa: E402


# 4x4 matrix, diagonal = 1.0. Passages 0,1 mutually near-identical (poison
# pair); passages 2,3 are the "clean" pair, loosely similar to each other and
# weakly to 0/1. Ground truth: 0,1 poison; 2,3 clean.
FIXTURE_MATRIX = np.array(
    [
        [1.00, 0.95, 0.20, 0.15],
        [0.95, 1.00, 0.18, 0.12],
        [0.20, 0.18, 1.00, 0.60],
        [0.15, 0.12, 0.60, 1.00],
    ]
)
FIXTURE_IS_POISON = np.array([True, True, False, False])


class TestClassifyPair(unittest.TestCase):
    def test_poison_poison(self):
        self.assertEqual(gate_a._classify_pair(0, 1, FIXTURE_IS_POISON), "PP")

    def test_poison_clean(self):
        self.assertEqual(gate_a._classify_pair(0, 2, FIXTURE_IS_POISON), "PC")
        self.assertEqual(gate_a._classify_pair(2, 0, FIXTURE_IS_POISON), "PC")

    def test_clean_clean(self):
        self.assertEqual(gate_a._classify_pair(2, 3, FIXTURE_IS_POISON), "CC")


class TestRunVariantOnSyntheticFixture(unittest.TestCase):
    def setUp(self):
        self.case = {
            "query_id": "synthetic_q1",
            "matrix": FIXTURE_MATRIX,
            "is_poison": FIXTURE_IS_POISON,
            "n_retrieved_poison": 2,
            "n_retrieved_clean": 2,
            "k": 4,
        }

    def test_legacy_and_paper_rows_have_matching_schema(self):
        legacy = gate_a._run_variant(self.case, "legacy")
        paper = gate_a._run_variant(self.case, "paper")
        self.assertEqual(set(legacy.keys()), set(paper.keys()))
        self.assertEqual(legacy["combine_logic"], "OR")
        self.assertEqual(paper["combine_logic"], "AND")
        self.assertIsNotNone(legacy["flipped"])
        self.assertIsNone(paper["flipped"])
        self.assertIsNone(paper["legacy_hybrid_threshold"])
        self.assertIsNotNone(legacy["legacy_hybrid_threshold"])

    def test_residual_and_removed_counts_are_internally_consistent(self):
        for variant in ("legacy", "paper"):
            row = gate_a._run_variant(self.case, variant)
            self.assertEqual(row["removed_poison"] + row["residual_poison"], self.case["n_retrieved_poison"])
            self.assertEqual(row["removed_clean"] + row["residual_clean"], self.case["n_retrieved_clean"])
            n_removed_from_indices = len(row["removed_indices"].split("|")) if row["removed_indices"] else 0
            self.assertEqual(n_removed_from_indices, row["removed_poison"] + row["removed_clean"])

    def test_pair_class_counts_sum_to_n_pairs(self):
        for variant in ("legacy", "paper"):
            row = gate_a._run_variant(self.case, variant)
            self.assertEqual(row["pp_top_pair_count"] + row["pc_top_pair_count"] + row["cc_top_pair_count"], row["n_pairs"])

    def test_top_pair_pp_is_poison_poison_pair_on_this_fixture(self):
        # Passages 0/1 (both poison) have the highest off-diagonal similarity
        # (0.95) in this fixture, for both variants (Stage 2 pair ranking is
        # variant-independent given the same n_adv > 0 and the same matrix).
        legacy = gate_a._run_variant(self.case, "legacy")
        paper = gate_a._run_variant(self.case, "paper")
        self.assertTrue(legacy["top_pair_pp"])
        self.assertTrue(paper["top_pair_pp"])


class TestSummaryRow(unittest.TestCase):
    def test_changed_flags_match_underlying_set_differences(self):
        case = {
            "query_id": "synthetic_q1",
            "matrix": FIXTURE_MATRIX,
            "is_poison": FIXTURE_IS_POISON,
            "n_retrieved_poison": 2,
            "n_retrieved_clean": 2,
            "k": 4,
        }
        legacy = gate_a._run_variant(case, "legacy")
        paper = gate_a._run_variant(case, "paper")
        summary = gate_a._summary_row(legacy, paper)

        self.assertEqual(summary["n_adv_legacy"], legacy["n_adv_estimated"])
        self.assertEqual(summary["n_adv_paper"], paper["n_adv_estimated"])
        self.assertEqual(summary["n_adv_abs_delta"], abs(paper["n_adv_estimated"] - legacy["n_adv_estimated"]))
        self.assertEqual(
            summary["n_adv_changed"], legacy["n_adv_estimated"] != paper["n_adv_estimated"]
        )
        # If N_adv differs, the flagged index sets are extremely likely (and,
        # for this fixture, actually) to differ too -- confirm the boolean
        # is derived from a real set comparison, not hardcoded.
        legacy_set = set(int(x) for x in legacy["final_adv_flag_indices"].split("|") if x != "")
        paper_set = set(int(x) for x in paper["final_adv_flag_indices"].split("|") if x != "")
        self.assertEqual(summary["stage1_flag_set_changed"], legacy_set != paper_set)


@unittest.skipUnless(
    gate_a.CLUSTER_VIZ_RUN_DIR.exists(),
    f"Real diagnostics run directory not found on disk: {gate_a.CLUSTER_VIZ_RUN_DIR} "
    "(results/ is gitignored; this test only runs in an environment that already "
    "has the ragdefender_cluster_viz artifacts).",
)
class TestLoadRealCaseSmoke(unittest.TestCase):
    def test_loads_one_real_case_with_matching_shapes(self):
        case = gate_a._load_case(gate_a.CLEAN_DENSITY_CASE)
        self.assertEqual(case["k"], 10)
        self.assertEqual(case["matrix"].shape, (10, 10))
        self.assertEqual(len(case["is_poison"]), 10)
        self.assertEqual(case["n_retrieved_poison"] + case["n_retrieved_clean"], 10)

    def test_full_gate_a_run_produces_one_summary_row_per_processed_query(self):
        import json

        with open(gate_a.CLUSTER_VIZ_RUN_DIR / "run_config.json") as f:
            expected_query_ids = set(json.load(f)["query_ids_processed"])

        per_query_rows, summary_rows = gate_a.run_gate_a()
        self.assertEqual(len(summary_rows) * 2, len(per_query_rows))
        self.assertEqual({row["query_id"] for row in summary_rows}, expected_query_ids)


if __name__ == "__main__":
    unittest.main()
