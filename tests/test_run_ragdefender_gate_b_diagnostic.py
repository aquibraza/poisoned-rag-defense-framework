"""Tests for scripts/run_ragdefender_gate_b_diagnostic.py -- the Gate B
true paper-fidelity gate (Stella re-encoding of real HotpotQA passages).

Three tiers, cheapest first:

1. `TestClassifyPair` / `TestGeometryStats` / `TestClassifyQuery` /
   `TestGateAPairComposition` / `TestCheckNoOverwrite` -- pure-logic unit
   tests on synthetic fixtures. Fully offline, no model, no real artifacts
   required.
2. `TestRealArtifactConsistency` -- gated on the real Gate-A run directory
   existing on disk (results/ is gitignored). Exercises the *text-recovery
   and consistency-check* path (`_recover_case_texts`,
   `_verify_matches_gate_a`) against the real 8 queries -- this path never
   touches Stella, so it is cheap and runs by default in that environment.
3. `TestSavedGateBOutputsStructural` -- gated on the real, already-produced
   `results/diagnostics/ragdefender_gate_b/` outputs existing on disk.
   Re-reads (never regenerates) those CSVs/report and checks structural
   invariants. Does NOT load Stella.
4. `TestGateBLiveStellaSmoke` -- gated on `RAGDEFENDER_LOAD_STELLA=1`
   (same opt-in as tests/test_ragdefender_paper_fidelity.py). Actually
   loads Stella and runs `run_gate_b_query` end-to-end for one real query,
   writing to a temp directory (never the real output directory).

Run with: python -m unittest tests.test_run_ragdefender_gate_b_diagnostic -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import run_ragdefender_gate_b_diagnostic as gate_b  # noqa: E402

# Same 4x4 fixture used by tests/test_run_ragdefender_gate_a_diagnostic.py,
# for direct comparability of pure-logic helpers.
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
        self.assertEqual(gate_b._classify_pair(0, 1, FIXTURE_IS_POISON), "PP")

    def test_poison_clean(self):
        self.assertEqual(gate_b._classify_pair(0, 2, FIXTURE_IS_POISON), "PC")
        self.assertEqual(gate_b._classify_pair(2, 0, FIXTURE_IS_POISON), "PC")

    def test_clean_clean(self):
        self.assertEqual(gate_b._classify_pair(2, 3, FIXTURE_IS_POISON), "CC")


class TestGeometryStats(unittest.TestCase):
    def test_matches_hand_computed_means_and_maxes(self):
        stats = gate_b._geometry_stats(FIXTURE_MATRIX, FIXTURE_IS_POISON)
        # Only one PP pair (0,1)=0.95; only one CC pair (2,3)=0.60;
        # four PC pairs: (0,2)=0.20,(0,3)=0.15,(1,2)=0.18,(1,3)=0.12.
        self.assertAlmostEqual(stats["mean_pp_similarity"], 0.95)
        self.assertAlmostEqual(stats["max_pp_similarity"], 0.95)
        self.assertAlmostEqual(stats["mean_cc_similarity"], 0.60)
        self.assertAlmostEqual(stats["max_cc_similarity"], 0.60)
        self.assertAlmostEqual(stats["mean_pc_similarity"], (0.20 + 0.15 + 0.18 + 0.12) / 4.0)
        self.assertAlmostEqual(stats["max_pc_similarity"], 0.20)

    def test_returns_none_for_missing_category(self):
        # All-poison: no CC or PC pairs exist.
        all_poison = np.array([True, True, True])
        matrix = np.array([[1.0, 0.5, 0.6], [0.5, 1.0, 0.7], [0.6, 0.7, 1.0]])
        stats = gate_b._geometry_stats(matrix, all_poison)
        self.assertIsNone(stats["mean_cc_similarity"])
        self.assertIsNone(stats["max_cc_similarity"])
        self.assertIsNone(stats["mean_pc_similarity"])
        self.assertIsNone(stats["max_pc_similarity"])
        self.assertIsNotNone(stats["mean_pp_similarity"])


class TestClassifyQuery(unittest.TestCase):
    def test_zero_residual_success_label(self):
        is_poison = np.array([True, True, False, False])
        adv_flag = np.array([True, True, False, False])
        labels = gate_b._classify_query(
            n_adv=2, is_poison=is_poison, adv_flag=adv_flag, top_pair_label="PP",
            removed_poison=2, removed_clean=0, residual_poison=0,
            pp_count=1, pc_count=0, cc_count=0,
        )
        self.assertIn("zero-residual-poison success", labels)
        self.assertNotIn("residual-poison failure", labels)
        self.assertNotIn("clean over-removal", labels)

    def test_residual_poison_failure_label(self):
        is_poison = np.array([True, True, False, False])
        adv_flag = np.array([True, False, False, False])
        labels = gate_b._classify_query(
            n_adv=1, is_poison=is_poison, adv_flag=adv_flag, top_pair_label="PP",
            removed_poison=1, removed_clean=0, residual_poison=1,
            pp_count=0, pc_count=0, cc_count=0,
        )
        self.assertIn("residual-poison failure", labels)

    def test_clean_over_removal_label(self):
        is_poison = np.array([True, True, False, False])
        adv_flag = np.array([True, True, True, False])
        labels = gate_b._classify_query(
            n_adv=3, is_poison=is_poison, adv_flag=adv_flag, top_pair_label="PP",
            removed_poison=2, removed_clean=1, residual_poison=0,
            pp_count=1, pc_count=1, cc_count=0,
        )
        self.assertIn("clean over-removal", labels)

    def test_clean_density_label_from_cc_top_pair(self):
        is_poison = np.array([True, True, False, False])
        adv_flag = np.array([False, False, True, True])
        labels = gate_b._classify_query(
            n_adv=2, is_poison=is_poison, adv_flag=adv_flag, top_pair_label="CC",
            removed_poison=0, removed_clean=2, residual_poison=2,
            pp_count=0, pc_count=0, cc_count=1,
        )
        self.assertIn("clean-density / clean-top-pair failure", labels)

    def test_clean_density_label_from_majority_clean_flagged(self):
        is_poison = np.array([True, False, False, False])
        adv_flag = np.array([False, True, True, False])
        labels = gate_b._classify_query(
            n_adv=2, is_poison=is_poison, adv_flag=adv_flag, top_pair_label="PP",
            removed_poison=0, removed_clean=2, residual_poison=1,
            pp_count=1, pc_count=0, cc_count=0,
        )
        self.assertIn("clean-density / clean-top-pair failure", labels)

    def test_mixed_pair_failure_label(self):
        # residual_poison must be > 0 for "mixed-pair failure" to fire (a
        # mixed PP/PC/CC top-pair composition alone, with a clean outcome,
        # is not itself a "failure").
        is_poison = np.array([True, True, True, False])
        adv_flag = np.array([True, True, True, False])
        labels = gate_b._classify_query(
            n_adv=3, is_poison=is_poison, adv_flag=adv_flag, top_pair_label="PP",
            removed_poison=2, removed_clean=0, residual_poison=1,
            pp_count=1, pc_count=1, cc_count=0,
        )
        self.assertIn("mixed-pair failure", labels)
        self.assertIn("residual-poison failure", labels)

    def test_other_inspect_manually_when_no_pairs(self):
        is_poison = np.array([True, False])
        adv_flag = np.array([False, False])
        labels = gate_b._classify_query(
            n_adv=0, is_poison=is_poison, adv_flag=adv_flag, top_pair_label=None,
            removed_poison=0, removed_clean=0, residual_poison=1,
            pp_count=0, pc_count=0, cc_count=0,
        )
        self.assertTrue(any("other / inspect manually" in label for label in labels))

    def test_labels_can_be_multiple(self):
        # A query can be both a residual-poison failure AND clean-density.
        is_poison = np.array([True, False, False, False])
        adv_flag = np.array([False, True, True, False])
        labels = gate_b._classify_query(
            n_adv=2, is_poison=is_poison, adv_flag=adv_flag, top_pair_label="CC",
            removed_poison=0, removed_clean=2, residual_poison=1,
            pp_count=0, pc_count=0, cc_count=1,
        )
        self.assertIn("residual-poison failure", labels)
        self.assertIn("clean over-removal", labels)
        self.assertIn("clean-density / clean-top-pair failure", labels)


class TestGateAPairComposition(unittest.TestCase):
    def test_recovers_pp_from_gate_a_row_fields(self):
        import pandas as pd

        row = pd.Series({"top_pairs": "0-1:0.9500|0-2:0.2000", "is_poison_i": "1|1|0|0"})
        pp, pc, cc = gate_b._gate_a_pair_composition(row)
        self.assertTrue(pp)
        self.assertFalse(pc)
        self.assertFalse(cc)

    def test_recovers_cc_from_gate_a_row_fields(self):
        import pandas as pd

        row = pd.Series({"top_pairs": "2-3:0.6000", "is_poison_i": "1|1|0|0"})
        pp, pc, cc = gate_b._gate_a_pair_composition(row)
        self.assertFalse(pp)
        self.assertFalse(pc)
        self.assertTrue(cc)

    def test_empty_top_pairs_returns_all_false(self):
        import pandas as pd

        row = pd.Series({"top_pairs": "", "is_poison_i": "1|1|0|0"})
        pp, pc, cc = gate_b._gate_a_pair_composition(row)
        self.assertEqual((pp, pc, cc), (False, False, False))


class TestCheckNoOverwrite(unittest.TestCase):
    def test_passes_when_no_paths_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "a.csv", Path(tmp) / "b.csv"]
            gate_b._check_no_overwrite(paths)  # should not raise

    def test_raises_when_a_path_already_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "already_there.csv"
            existing.write_text("x")
            with self.assertRaises(gate_b.GateBStopCondition):
                gate_b._check_no_overwrite([existing, Path(tmp) / "new.csv"])


@unittest.skipUnless(
    gate_b.GATE_A_RUN_DIR.exists(),
    f"Real Gate-A diagnostics run directory not found on disk: {gate_b.GATE_A_RUN_DIR} "
    "(results/ is gitignored; this test only runs in an environment that already has it).",
)
class TestRealArtifactConsistency(unittest.TestCase):
    """Exercises the real text-recovery + Gate-A-consistency-check path for
    all 8 queries. Never loads Stella -- purely re-derives the same passage
    texts Gate A's saved matrices were built from and confirms they match
    the saved is_poison ground truth."""

    @classmethod
    def setUpClass(cls):
        import visualize_ragdefender_clusters as viz

        cls.run_config, cls.gate_a_is_poison = gate_b._load_gate_a_context()
        cls.diagnostics_records = viz._read_jsonl(cls.run_config["diagnostics_jsonl"])  # noqa: SLF001
        cls.query_results_index = viz.load_query_results_index(cls.run_config["query_results_dir"])

    def test_recovers_and_validates_all_eight_queries(self):
        query_ids = self.run_config["query_ids_processed"]
        self.assertEqual(len(query_ids), 8)
        for qid in query_ids:
            case = gate_b._recover_case_texts(qid, self.diagnostics_records, self.query_results_index)
            gate_b._verify_matches_gate_a(case, self.gate_a_is_poison)  # must not raise
            self.assertEqual(case["k"], 10)
            self.assertEqual(len(case["texts"]), 10)
            self.assertEqual(case["n_retrieved_poison"] + case["n_retrieved_clean"], 10)

    def test_mismatched_is_poison_raises_stop_condition(self):
        query_ids = self.run_config["query_ids_processed"]
        case = gate_b._recover_case_texts(query_ids[0], self.diagnostics_records, self.query_results_index)
        tampered_gate_a_is_poison = dict(self.gate_a_is_poison)
        tampered = np.array(case["is_poison"])
        tampered[0] = not tampered[0]
        tampered_gate_a_is_poison[query_ids[0]] = tampered
        with self.assertRaises(gate_b.GateBStopCondition):
            gate_b._verify_matches_gate_a(case, tampered_gate_a_is_poison)


GATE_B_OUTPUT_DIR = gate_b.OUTPUT_DIR


@unittest.skipUnless(
    (GATE_B_OUTPUT_DIR / "gate_b_per_query.csv").exists(),
    f"Real Gate-B outputs not found on disk: {GATE_B_OUTPUT_DIR} (results/ is gitignored; "
    "this test only runs after Gate B has actually been run once in this environment). "
    "Never regenerates them -- read-only structural check.",
)
class TestSavedGateBOutputsStructural(unittest.TestCase):
    """Read-only structural/consistency checks on the already-produced real
    Gate-B outputs. Does not load Stella, does not call the network, does
    not regenerate anything -- if this fails, inspect (do not silently
    regenerate) the on-disk artifacts."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd

        cls.per_query = pd.read_csv(GATE_B_OUTPUT_DIR / "gate_b_per_query.csv")
        cls.comparison = pd.read_csv(GATE_B_OUTPUT_DIR / "gate_b_comparison.csv")
        with open(gate_b.GATE_A_RUN_DIR / "run_config.json") as f:
            cls.expected_query_ids = set(json.load(f)["query_ids_processed"])

    def test_per_query_has_one_row_per_processed_query(self):
        self.assertEqual(set(self.per_query["query_id"]), self.expected_query_ids)
        self.assertEqual(len(self.per_query), len(self.expected_query_ids))

    def test_removed_plus_residual_equals_retrieved_composition(self):
        for _, row in self.per_query.iterrows():
            self.assertEqual(row["removed_poison"] + row["residual_poison"], row["n_retrieved_poison"])
            self.assertEqual(row["removed_clean"] + row["residual_clean"], row["n_retrieved_clean"])

    def test_pair_class_counts_sum_to_n_pairs(self):
        for _, row in self.per_query.iterrows():
            self.assertEqual(
                row["pp_top_pair_count"] + row["pc_top_pair_count"] + row["cc_top_pair_count"], row["n_pairs"]
            )

    def test_similarity_and_embedding_artifact_paths_exist_on_disk(self):
        repo_root = gate_b.REPO_ROOT
        for _, row in self.per_query.iterrows():
            self.assertTrue((repo_root / row["embeddings_path"]).exists())
            self.assertTrue((repo_root / row["similarity_matrix_path"]).exists())

    def test_saved_similarity_matrices_have_finite_values_and_unit_diagonal(self):
        for _, row in self.per_query.iterrows():
            matrix = np.load(gate_b.REPO_ROOT / row["similarity_matrix_path"])
            self.assertTrue(np.isfinite(matrix).all())
            self.assertEqual(matrix.shape, (row["k"], row["k"]))
            self.assertTrue(np.allclose(np.diag(matrix), 1.0, atol=1e-3))

    def test_comparison_table_has_one_row_per_query_and_matches_gate_b_n_adv(self):
        self.assertEqual(set(self.comparison["query_id"]), self.expected_query_ids)
        per_query_by_qid = self.per_query.set_index("query_id")
        for _, row in self.comparison.iterrows():
            self.assertEqual(row["n_adv_gate_b"], per_query_by_qid.loc[row["query_id"], "n_adv"])
            self.assertEqual(
                row["n_adv_delta_gate_a_to_gate_b"], row["n_adv_gate_b"] - row["n_adv_gate_a"]
            )

    def test_report_states_no_forbidden_experiments_were_run(self):
        report_text = (GATE_B_OUTPUT_DIR / "GATE_B_STELLA_FIDELITY_REPORT.md").read_text()
        self.assertIn(
            "No retrieval, generation, E1, CORAL, or MMD experiment was run.", report_text
        )
        for forbidden in ("E1", "CORAL", "MMD"):
            self.assertIn(forbidden, report_text)


@unittest.skipUnless(
    os.environ.get("RAGDEFENDER_LOAD_STELLA") == "1",
    "Set RAGDEFENDER_LOAD_STELLA=1 to run this heavy integration test "
    "(downloads/loads the real dunzhang/stella_en_1.5B_v5 model). Not run by "
    "default -- see docs/RAGDEFENDER_FIDELITY_AUDIT_V2.md Gate B.",
)
class TestGateBLiveStellaSmoke(unittest.TestCase):
    """Runs the real `_load_stella_model` + `run_gate_b_query` path
    end-to-end for exactly one real query, writing artifacts to a temp
    directory (never the real `results/diagnostics/ragdefender_gate_b/`
    output). Proves the live path this script uses in production still
    works, independent of the already-saved outputs checked above."""

    def test_one_real_query_end_to_end_on_cpu(self):
        import visualize_ragdefender_clusters as viz

        run_config, gate_a_is_poison = gate_b._load_gate_a_context()
        diagnostics_records = viz._read_jsonl(run_config["diagnostics_jsonl"])  # noqa: SLF001
        query_results_index = viz.load_query_results_index(run_config["query_results_dir"])

        qid = run_config["query_ids_processed"][0]
        case = gate_b._recover_case_texts(qid, diagnostics_records, query_results_index)
        gate_b._verify_matches_gate_a(case, gate_a_is_poison)

        s_model, cfg, actual_device = gate_b._load_stella_model()
        self.assertEqual(actual_device, "cpu")
        _, st_util = gate_b.defense_runner._lazy_st()  # noqa: SLF001

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            (tmp_dir / "similarity").mkdir()
            (tmp_dir / "embeddings").mkdir()
            row = gate_b.run_gate_b_query(case, s_model, st_util, tmp_dir)

        self.assertEqual(row["query_id"], qid)
        self.assertGreaterEqual(row["n_adv"], 0)
        self.assertEqual(row["removed_poison"] + row["residual_poison"], case["n_retrieved_poison"])


if __name__ == "__main__":
    unittest.main()
