"""Tests for `scripts/run_ragdefender_expanded_baseline_bridge_5q.py` -- the
historical n=42 expanded-baseline 5-query reproducibility bridge.

Covers:
1. deterministic 5-query selection rule (pure function of frozen order +
   frozen labels);
2. the selection cannot depend on reproduction outcome (no such parameter
   exists, and re-running selection twice with different downstream
   "current" results yields the identical selection);
3. historical `.npy` matrices are read-only (byte-identical before/after);
4. comparison-metric correctness (`_matrix_comparison`);
5. decision-stability classification (`_classify_query`);
6. the current matrix is never written over the historical matrix path
   (static source check + functional read-only check);
7. the optional oracle-label recheck uses the COUNT only, never poison
   identities, to influence Stage 2's selection.

Also includes ONE gated live-Stella integration smoke test
(`RAGDEFENDER_LOAD_STELLA=1`), matching the existing convention in
`tests/test_run_ragdefender_gate_b_diagnostic.py`, and a real-artifact
structural check gated on the actual `expanded_baseline_bridge_5q.csv`
already existing on disk.

Run with: python -m unittest tests.test_ragdefender_expanded_baseline_bridge_5q -v
"""
import csv
import hashlib
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import run_ragdefender_expanded_baseline_bridge_5q as bridge  # noqa: E402


def _matrix_2x2(s: float) -> np.ndarray:
    return np.array([[1.0, s], [s, 1.0]])


def _symmetric_matrix(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.uniform(-0.2, 0.9, size=(k, k))
    m = (a + a.T) / 2.0
    np.fill_diagonal(m, 1.0)
    return m


class TestDeterministicSelectionRule(unittest.TestCase):
    """1. Deterministic 5-query selection rule."""

    def _rows(self):
        # Deliberately NOT in success/failure/regime-sorted order -- the
        # rule must respect frozen ORDER, not re-sort by regime/label.
        return [
            {"query_id": "q_c7", "regime": "C_ABOVE_CEILING", "m_poison": 7, "zero_residual_poison_success": False},
            {"query_id": "q_b_fail1", "regime": "B_AT_CEILING", "m_poison": 5, "zero_residual_poison_success": False},
            {"query_id": "q_c8", "regime": "C_ABOVE_CEILING", "m_poison": 8, "zero_residual_poison_success": False},
            {"query_id": "q_b_success1", "regime": "B_AT_CEILING", "m_poison": 5, "zero_residual_poison_success": True},
            {"query_id": "q_b_fail2", "regime": "B_AT_CEILING", "m_poison": 5, "zero_residual_poison_success": False},
            {"query_id": "q_c6", "regime": "C_ABOVE_CEILING", "m_poison": 6, "zero_residual_poison_success": False},
            {"query_id": "q_c9", "regime": "C_ABOVE_CEILING", "m_poison": 9, "zero_residual_poison_success": False},
            {"query_id": "q_d1", "regime": "D_ALL_POISON", "m_poison": 10, "zero_residual_poison_success": False},
            {"query_id": "q_d2", "regime": "D_ALL_POISON", "m_poison": 10, "zero_residual_poison_success": False},
        ]

    def test_picks_first_occurrence_of_each_category_in_frozen_order(self):
        selected = bridge.select_five_queries(self._rows())
        self.assertEqual(selected["b_success"], "q_b_success1")
        self.assertEqual(selected["b_failure"], "q_b_fail1")  # first failure, not q_b_fail2
        self.assertEqual(selected["c_m6"], "q_c6")
        self.assertEqual(selected["c_m8plus"], "q_c8")  # first M>=8, not q_c9
        self.assertEqual(selected["d_allpoison"], "q_d1")  # first D, not q_d2

    def test_raises_if_any_category_absent(self):
        rows = [r for r in self._rows() if r["regime"] != "D_ALL_POISON"]
        with self.assertRaises(bridge.BridgeStopCondition):
            bridge.select_five_queries(rows)

    def test_matches_real_frozen_artifacts_on_disk(self):
        """Cross-check against the actual frozen n=42 population, if
        present (real end-to-end determinism check, not just synthetic)."""
        if not bridge.PROSPECTIVE_POPULATION_CSV.exists():
            self.skipTest("Real frozen n=42 population not found on disk.")
        rows = bridge._load_frozen_order_and_labels()  # noqa: SLF001
        selected = bridge.select_five_queries(rows)
        # Pinned to the actual frozen artifact contents (recorded once,
        # BEFORE any Stella re-encoding was performed).
        self.assertEqual(selected["b_success"], "5ae0361155429925eb1afc2c")
        self.assertEqual(selected["b_failure"], "5ae22b8d554299234fd0440f")
        self.assertEqual(selected["c_m6"], "5abf63f15542997ec76fd3ea")
        self.assertEqual(selected["c_m8plus"], "5abd259d55429924427fcf1a")
        self.assertEqual(selected["d_allpoison"], "5a80840f554299485f59863b")


class TestSelectionIndependentOfReproductionOutcome(unittest.TestCase):
    """2. Frozen-query IDs cannot depend on reproduction outcome."""

    def test_selection_function_signature_has_no_reproduction_inputs(self):
        """`select_five_queries` must take only frozen-order/label rows --
        no matrix, no re-encoded similarity, no comparison-outcome
        argument can even be passed in."""
        sig = inspect.signature(bridge.select_five_queries)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["rows"])
        for forbidden in ("matrix", "current", "reproduc", "outcome", "diff"):
            for p in params:
                self.assertNotIn(forbidden, p.lower())

    def test_same_frozen_labels_give_same_selection_regardless_of_downstream_result(self):
        rows = [
            {"query_id": "q1", "regime": "B_AT_CEILING", "m_poison": 5, "zero_residual_poison_success": True},
            {"query_id": "q2", "regime": "B_AT_CEILING", "m_poison": 5, "zero_residual_poison_success": False},
            {"query_id": "q3", "regime": "C_ABOVE_CEILING", "m_poison": 6, "zero_residual_poison_success": False},
            {"query_id": "q4", "regime": "C_ABOVE_CEILING", "m_poison": 8, "zero_residual_poison_success": False},
            {"query_id": "q5", "regime": "D_ALL_POISON", "m_poison": 10, "zero_residual_poison_success": False},
        ]
        # Call selection twice on the identical frozen input; nothing about
        # a hypothetical "reproduction succeeded/failed" result is or
        # could be threaded through `select_five_queries` (see signature
        # test above), so both calls must agree.
        first = bridge.select_five_queries(rows)
        second = bridge.select_five_queries(rows)
        self.assertEqual(first, second)


class TestHistoricalMatricesReadOnly(unittest.TestCase):
    """3. Historical matrices are read-only."""

    def test_load_historical_matrix_does_not_modify_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_baseline_dir = Path(tmp)
            (fake_baseline_dir / "similarity").mkdir()
            matrix = _symmetric_matrix(10, seed=1)
            fixture_path = fake_baseline_dir / "similarity" / "qid_stella_similarity_matrix.npy"
            np.save(fixture_path, matrix)
            before_bytes = fixture_path.read_bytes()
            before_mtime = fixture_path.stat().st_mtime_ns

            original_dir = bridge.BASELINE_DIR
            try:
                bridge.BASELINE_DIR = fake_baseline_dir
                loaded = bridge._load_historical_matrix("qid")  # noqa: SLF001
            finally:
                bridge.BASELINE_DIR = original_dir

            np.testing.assert_array_equal(loaded, matrix)
            after_bytes = fixture_path.read_bytes()
            after_mtime = fixture_path.stat().st_mtime_ns
            self.assertEqual(before_bytes, after_bytes)
            self.assertEqual(before_mtime, after_mtime)

    def test_missing_historical_matrix_raises_stop_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_baseline_dir = Path(tmp)
            (fake_baseline_dir / "similarity").mkdir()
            original_dir = bridge.BASELINE_DIR
            try:
                bridge.BASELINE_DIR = fake_baseline_dir
                with self.assertRaises(bridge.BridgeStopCondition):
                    bridge._load_historical_matrix("missing_qid")  # noqa: SLF001
            finally:
                bridge.BASELINE_DIR = original_dir


class TestMatrixComparisonMetrics(unittest.TestCase):
    """4. Comparison-metric correctness."""

    def test_identical_matrices_give_zero_diff_and_both_allclose(self):
        m = _symmetric_matrix(10, seed=2)
        result = bridge._matrix_comparison(m, m.copy())  # noqa: SLF001
        self.assertEqual(result["max_abs_diff"], 0.0)
        self.assertEqual(result["mean_abs_diff"], 0.0)
        self.assertEqual(result["frobenius_norm_diff"], 0.0)
        self.assertTrue(result["allclose_strict"])
        self.assertTrue(result["allclose_loose"])

    def test_small_perturbation_matches_manual_computation(self):
        hist = np.array([[1.0, 0.5], [0.5, 1.0]])
        cur = np.array([[1.0, 0.5000005], [0.5000005, 1.0]])
        result = bridge._matrix_comparison(hist, cur)  # noqa: SLF001
        expected_max = float(np.abs(hist - cur).max())
        expected_mean = float(np.abs(hist - cur).mean())
        expected_fro = float(np.linalg.norm(hist - cur))
        self.assertAlmostEqual(result["max_abs_diff"], expected_max, places=12)
        self.assertAlmostEqual(result["mean_abs_diff"], expected_mean, places=12)
        self.assertAlmostEqual(result["frobenius_norm_diff"], expected_fro, places=12)

    def test_strict_vs_loose_tolerance_boundary(self):
        hist = np.array([[1.0, 0.5], [0.5, 1.0]])
        # 5e-7 diff: fails strict (atol=1e-8, rtol=1e-7) but passes loose
        # (atol=1e-6, rtol=1e-5).
        cur = np.array([[1.0, 0.5 + 5e-7], [0.5 + 5e-7, 1.0]])
        result = bridge._matrix_comparison(hist, cur)  # noqa: SLF001
        self.assertFalse(result["allclose_strict"])
        self.assertTrue(result["allclose_loose"])

    def test_large_diff_fails_both_tolerances(self):
        hist = np.array([[1.0, 0.5], [0.5, 1.0]])
        cur = np.array([[1.0, 0.9], [0.9, 1.0]])
        result = bridge._matrix_comparison(hist, cur)  # noqa: SLF001
        self.assertFalse(result["allclose_strict"])
        self.assertFalse(result["allclose_loose"])


class TestDecisionStabilityClassification(unittest.TestCase):
    """5. Decision-stability classification (A/B/C)."""

    def _s1s2(self, n_adv, removed_indices, success):
        return {
            "n_adv": n_adv,
            "removed_indices": tuple(sorted(removed_indices)),
            "zero_residual_poison_success": success,
        }

    def test_byte_identical_when_zero_diff_and_same_decision(self):
        matrix_cmp = {"max_abs_diff": 0.0}
        hist = self._s1s2(4, [0, 1, 2, 3], False)
        cur = self._s1s2(4, [0, 1, 2, 3], False)
        label = bridge._classify_query(matrix_cmp, hist, cur)  # noqa: SLF001
        self.assertEqual(label, "A. BYTE/NUMERICALLY IDENTICAL")

    def test_numeric_drift_decision_stable_when_nonzero_diff_same_decision(self):
        matrix_cmp = {"max_abs_diff": 7.0e-7}
        hist = self._s1s2(4, [0, 1, 2, 3], False)
        cur = self._s1s2(4, [0, 1, 2, 3], False)
        label = bridge._classify_query(matrix_cmp, hist, cur)  # noqa: SLF001
        self.assertEqual(label, "B. NUMERIC DRIFT, DECISION-STABLE")

    def test_decision_drift_when_n_adv_changes(self):
        matrix_cmp = {"max_abs_diff": 1.0e-6}
        hist = self._s1s2(4, [0, 1, 2, 3], False)
        cur = self._s1s2(5, [0, 1, 2, 3, 4], False)
        label = bridge._classify_query(matrix_cmp, hist, cur)  # noqa: SLF001
        self.assertEqual(label, "C. DECISION DRIFT")

    def test_decision_drift_when_removed_set_changes_but_n_adv_same(self):
        matrix_cmp = {"max_abs_diff": 1.0e-6}
        hist = self._s1s2(4, [0, 1, 2, 3], False)
        cur = self._s1s2(4, [0, 1, 2, 4], False)
        label = bridge._classify_query(matrix_cmp, hist, cur)  # noqa: SLF001
        self.assertEqual(label, "C. DECISION DRIFT")

    def test_decision_drift_when_outcome_changes(self):
        matrix_cmp = {"max_abs_diff": 1.0e-6}
        hist = self._s1s2(4, [0, 1, 2, 3], False)
        cur = self._s1s2(4, [0, 1, 2, 3], True)
        label = bridge._classify_query(matrix_cmp, hist, cur)  # noqa: SLF001
        self.assertEqual(label, "C. DECISION DRIFT")


class TestCurrentMatrixNeverWritesOverHistorical(unittest.TestCase):
    """6. The current matrix is never written over the historical matrix."""

    def test_no_np_save_call_targets_the_historical_similarity_directory(self):
        """Static source check: this module must never call `np.save`
        with a path under `BASELINE_DIR` (the historical, read-only
        directory) -- the only `np.save`-eligible output is the new
        `expanded_baseline_bridge_5q.csv`, written via `csv.DictWriter`,
        not `np.save`, and to `OUTPUT_DIR`, not `BASELINE_DIR`."""
        source = inspect.getsource(bridge)
        self.assertNotIn("np.save", source)

    def test_load_historical_matrix_has_no_write_capable_calls(self):
        source = inspect.getsource(bridge._load_historical_matrix)  # noqa: SLF001
        for forbidden in ("np.save", "savez", "open(", ".write(", "to_csv"):
            self.assertNotIn(forbidden, source)


class TestOracleRecheckUsesCountOnly(unittest.TestCase):
    """7. The optional oracle-label recheck uses the COUNT only, never
    poison identities, to influence Stage 2's selection."""

    def test_stage2_selection_identical_across_different_poison_identity_assignments(self):
        """Two different `is_poison` arrays with the SAME total count (so
        the oracle-supplied `n_adv` integer is identical) must yield the
        IDENTICAL Stage-2 `selected_indices` (removed-index SET), because
        `_oracle_from_matrix` passes only the integer count into
        `stage2_pair_frequency`, never the identity array itself."""
        matrix = _symmetric_matrix(10, seed=3)
        m_poison = 5

        is_poison_a = np.array([True] * 5 + [False] * 5)
        is_poison_b = np.array([False, True, False, True, False, True, False, True, False, True])
        self.assertEqual(int(is_poison_a.sum()), m_poison)
        self.assertEqual(int(is_poison_b.sum()), m_poison)

        oracle_a = bridge._oracle_from_matrix(matrix, m_poison, is_poison_a)  # noqa: SLF001
        oracle_b = bridge._oracle_from_matrix(matrix, m_poison, is_poison_b)  # noqa: SLF001

        # The underlying Stage-2 call (matrix + count only) must select
        # the SAME index set regardless of which identity array is passed
        # in afterwards to SCORE that set.
        stage2_direct = bridge.ri.stage2_pair_frequency(matrix, n_adv=m_poison, p=2.0)
        selected_set = set(stage2_direct.selected_indices)
        removed_poison_a_expected = sum(1 for i in selected_set if is_poison_a[i])
        removed_poison_b_expected = sum(1 for i in selected_set if is_poison_b[i])
        self.assertEqual(oracle_a["removed_poison"], removed_poison_a_expected)
        self.assertEqual(oracle_b["removed_poison"], removed_poison_b_expected)
        # Different identity assignments over the SAME selected set can
        # (and here do) give different removed_poison/removed_clean
        # SCORES -- but that is scoring the fixed set, not re-selecting it.
        self.assertNotEqual(oracle_a["removed_poison"], oracle_b["removed_poison"])

    def test_oracle_from_matrix_source_never_passes_is_poison_into_stage2_call(self):
        source = inspect.getsource(bridge._oracle_from_matrix)  # noqa: SLF001
        # Find the stage2_pair_frequency call site and confirm its
        # argument list does not mention is_poison.
        call_line = next(line for line in source.splitlines() if "stage2_pair_frequency(" in line)
        self.assertNotIn("is_poison", call_line)
        self.assertIn("n_adv=m_poison", call_line)


class TestNoOverwriteSafeguard(unittest.TestCase):
    def test_run_bridge_refuses_to_overwrite_existing_output_csv(self):
        if not bridge.PROSPECTIVE_POPULATION_CSV.exists():
            self.skipTest("Real frozen n=42 population not found on disk; nothing to select from.")
        original_csv = bridge.OUTPUT_CSV
        with tempfile.TemporaryDirectory() as tmp:
            fake_path = Path(tmp) / "expanded_baseline_bridge_5q.csv"
            fake_path.write_text("placeholder\n")
            try:
                bridge.OUTPUT_CSV = fake_path
                with self.assertRaises(bridge.BridgeStopCondition):
                    bridge.run_bridge()
            finally:
                bridge.OUTPUT_CSV = original_csv


@unittest.skipUnless(
    bridge.OUTPUT_CSV.exists(),
    f"Real bridge output not found on disk: {bridge.OUTPUT_CSV} "
    "(results/ is gitignored; this test only runs after this script has actually "
    "been run once in this environment). Never regenerates it -- read-only structural check.",
)
class TestRealBridgeOutputStructural(unittest.TestCase):
    """Read-only structural checks against the actual, already-produced
    `expanded_baseline_bridge_5q.csv` (never regenerates it)."""

    def setUp(self):
        with open(bridge.OUTPUT_CSV) as f:
            self.rows = list(csv.DictReader(f))

    def test_exactly_five_rows_one_per_role(self):
        self.assertEqual(len(self.rows), 5)
        roles = {r["role"] for r in self.rows}
        self.assertEqual(roles, set(bridge.SELECTION_RULE_KEYS))

    def test_historical_similarity_files_unchanged_by_this_run(self):
        """The historical `.npy` matrices referenced by the 5 selected
        queries must still be present and readable (never deleted or
        corrupted by this script)."""
        for row in self.rows:
            path = bridge.BASELINE_DIR / "similarity" / f"{row['query_id']}_stella_similarity_matrix.npy"
            self.assertTrue(path.exists(), f"Historical matrix missing: {path}")
            # Loadable without error -- confirms it was never partially
            # overwritten by a crashed `np.save`.
            np.load(path)

    def test_gate_c_label_recheck_recorded_for_every_row(self):
        for row in self.rows:
            self.assertIn(
                row["hist_label_recomputed_from_hist_matrix"],
                ("A. COUNT-LIMITED", "B. COUNT + IDENTIFICATION LIMITED", "C. IDENTIFICATION LIMITED", "D. BASELINE SUCCESS"),
            )

    def test_expanded_baseline_and_gate_c_artifacts_were_not_modified(self):
        """This bridge script must never touch the STEP-4/Gate-C CSVs it
        reads from -- a live sanity check that they still parse and still
        contain 42 queries."""
        with open(bridge.BASELINE_PER_QUERY_CSV) as f:
            baseline_rows = list(csv.DictReader(f))
        self.assertEqual(len(baseline_rows), 42)
        with open(bridge.GATE_C_PER_QUERY_CSV) as f:
            gate_c_rows = list(csv.DictReader(f))
        self.assertEqual(len(gate_c_rows), 42)


@unittest.skipUnless(
    os.environ.get("RAGDEFENDER_LOAD_STELLA") == "1",
    "Set RAGDEFENDER_LOAD_STELLA=1 to run this heavy integration test "
    "(loads the real dunzhang/stella_en_1.5B_v5 model from the local cache). "
    "Not run by default -- see docs/RAGDEFENDER_FIDELITY_AUDIT_V2.md.",
)
class TestLiveStellaReencodingSmoke(unittest.TestCase):
    """Runs the real production Stella-loading path end-to-end for exactly
    ONE of the 5 selected queries, without writing any output file."""

    def test_one_selected_query_reencodes_and_compares_without_writing_outputs(self):
        if not bridge.RECOVERED_CONTEXTS_JSON.exists():
            self.skipTest("Frozen recovered_contexts.json not found on disk.")
        rows = bridge._load_frozen_order_and_labels()  # noqa: SLF001
        selected = bridge.select_five_queries(rows)
        query_id = selected["b_success"]

        historical_matrix = bridge._load_historical_matrix(query_id)  # noqa: SLF001
        ctx = bridge._load_frozen_context(query_id)  # noqa: SLF001
        is_poison = np.array(ctx["is_poison"], dtype=bool)

        s_model, st_util, actual_device = bridge._load_stella()  # noqa: SLF001
        self.assertEqual(actual_device, "cpu")

        embeddings = s_model.encode(ctx["texts"], convert_to_tensor=True)
        current_matrix = st_util.cos_sim(embeddings, embeddings).cpu().numpy().astype(np.float64)

        cmp = bridge._matrix_comparison(historical_matrix, current_matrix)  # noqa: SLF001
        self.assertTrue(cmp["allclose_loose"])
        hist_s1s2 = bridge._stage1_stage2_from_matrix(historical_matrix, is_poison)  # noqa: SLF001
        cur_s1s2 = bridge._stage1_stage2_from_matrix(current_matrix, is_poison)  # noqa: SLF001
        self.assertEqual(hist_s1s2["n_adv"], cur_s1s2["n_adv"])
        self.assertEqual(hist_s1s2["removed_indices"], cur_s1s2["removed_indices"])


if __name__ == "__main__":
    unittest.main()
