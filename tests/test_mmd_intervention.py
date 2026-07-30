"""Tests for defense/coral_mmd_intervention.py's MMD-minimizing oracle
optimizer (`mmd_minimize_transform` -- Step 3 of the CORAL/MMD oracle
intervention plan) and scripts/run_mmd_oracle_intervention.py.

Following tests/test_coral_ridge_intervention.py's / tests/
test_coral_mmd_intervention.py's convention: the automated suite never
downloads or loads a real sentence-transformers model. The end-to-end
smoke test monkeypatches `load_embedder` with the same deterministic
`FakeSentenceTransformer` pattern used there.

Run with: python -m unittest tests.test_mmd_intervention -v
"""
import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from defense.cluster_normalized_poisoning import l2_normalize_rows, recombine_poison_clean
from defense.coral_mmd_intervention import (
    mmd_minimize_transform,
    mmd_rbf_squared_raw,
)
from defense.distribution_metrics import mmd_rbf_distance_from_gram
from defense.ragdefender_internals import concentration_stage1, stage2_pair_frequency

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# scripts/ has no __init__.py -- import by file path, same pattern as
# tests/test_coral_ridge_intervention.py.
_RUN_SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "run_mmd_oracle_intervention.py")
_spec = importlib.util.spec_from_file_location("run_mmd_oracle_intervention", _RUN_SCRIPT_PATH)
run_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_script)


def _make_embeddings(seed: int, n: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim))


def _cos_sim_np(z: np.ndarray) -> np.ndarray:
    norm = z / np.linalg.norm(z, axis=1, keepdims=True)
    return norm @ norm.T


LAMBDA_PRESERVES = [0.01, 0.1, 1.0]
STEPS_LIST = [0, 50, 100]


# --------------------------------------------------------------------------
# Test: biased MMD loss is near zero for identical distributions
# --------------------------------------------------------------------------

class TestMmdLossIdenticalDistributions(unittest.TestCase):
    def test_mmd_near_zero_for_identical_points(self):
        z = _make_embeddings(1, 6, 16)
        z_unit = l2_normalize_rows(z)
        mmd = mmd_rbf_squared_raw(z_unit, z_unit, gamma=1.0)
        self.assertAlmostEqual(mmd, 0.0, places=10)

    def test_mmd_near_zero_for_disjoint_identical_copies(self):
        """Two disjoint but identically-distributed samples (same
        underlying generator, different draws) should give a small (not
        exactly zero, but small) MMD -- this is a distinct check from the
        exact-zero identical-points case above."""
        rng = np.random.default_rng(5)
        z1 = l2_normalize_rows(rng.normal(size=(20, 8)))
        z2 = l2_normalize_rows(rng.normal(size=(20, 8)))
        mmd_same_dist = mmd_rbf_squared_raw(z1, z2, gamma=1.0)
        z3 = l2_normalize_rows(rng.normal(loc=5.0, size=(20, 8)))
        mmd_diff_dist = mmd_rbf_squared_raw(z1, z3, gamma=1.0)
        self.assertLess(mmd_same_dist, mmd_diff_dist)

    def test_matches_gram_based_mmd_on_unit_norm_vectors(self):
        """Cross-check: the differentiable torch MMD^2 must numerically
        match the existing Gram-based `mmd_rbf_distance_from_gram` when
        both operate on the same unit-norm vectors and gamma."""
        rng = np.random.default_rng(9)
        zp = l2_normalize_rows(rng.normal(size=(5, 32)))
        zc = l2_normalize_rows(rng.normal(loc=1.0, size=(5, 32)))
        mmd_torch = mmd_rbf_squared_raw(zp, zc, gamma=1.0)
        cos_pp = zp @ zp.T
        cos_pc = zp @ zc.T
        cos_cc = zc @ zc.T
        mmd_gram = mmd_rbf_distance_from_gram(cos_pp, cos_pc, cos_cc, gamma=1.0)
        self.assertAlmostEqual(mmd_torch, mmd_gram, places=9)


# --------------------------------------------------------------------------
# Test: MMD loss decreases on a toy example
# --------------------------------------------------------------------------

class TestMmdLossDecreases(unittest.TestCase):
    def test_mmd_loss_decreases_over_steps_well_separated_clusters(self):
        rng = np.random.default_rng(2)
        z_poison = rng.normal(loc=3.0, size=(5, 16))
        z_clean = rng.normal(loc=-3.0, size=(5, 16))
        result = mmd_minimize_transform(z_poison, z_clean, lambda_preserve=0.01, gamma=1.0, steps=100, lr=0.05)
        mmd_trajectory = [t.mmd_loss for t in result.trace]
        self.assertEqual(len(mmd_trajectory), 101)
        self.assertLess(mmd_trajectory[-1], mmd_trajectory[0])
        # Overall trend should be non-increasing on net across chunks (not
        # necessarily strictly monotone every single step, given the
        # preservation term and unit-sphere projection).
        self.assertLess(np.mean(mmd_trajectory[-10:]), np.mean(mmd_trajectory[:10]))

    def test_higher_lambda_preserve_yields_smaller_displacement(self):
        rng = np.random.default_rng(3)
        z_poison = rng.normal(loc=3.0, size=(5, 16))
        z_clean = rng.normal(loc=-3.0, size=(5, 16))
        result_low = mmd_minimize_transform(z_poison, z_clean, lambda_preserve=0.01, gamma=1.0, steps=100, lr=0.05)
        result_high = mmd_minimize_transform(z_poison, z_clean, lambda_preserve=1.0, gamma=1.0, steps=100, lr=0.05)
        disp_low = result_low.trace[-1].preservation_loss
        disp_high = result_high.trace[-1].preservation_loss
        self.assertLess(disp_high, disp_low)


# --------------------------------------------------------------------------
# Test: steps=0 identity reproduces original cosine matrix and Stage 1/2
# --------------------------------------------------------------------------

class TestMmdStepsZeroIdentity(unittest.TestCase):
    def setUp(self):
        self.k = 10
        self.z_poison = _make_embeddings(8, 5, 32)
        self.z_clean = _make_embeddings(9, 5, 32)
        self.poison_idx = list(range(5))
        self.clean_idx = list(range(5, 10))
        z_original = recombine_poison_clean(self.z_poison, self.z_clean, self.poison_idx, self.clean_idx, self.k)
        self.m_original = _cos_sim_np(z_original)
        self.stage1_original = concentration_stage1(self.m_original)
        self.stage2_original = stage2_pair_frequency(self.m_original, self.stage1_original.n_adv_estimated)

    def test_steps_zero_reproduces_original_cosine_matrix_for_every_lambda_preserve(self):
        for lp in LAMBDA_PRESERVES:
            result = mmd_minimize_transform(self.z_poison, self.z_clean, lambda_preserve=lp, steps=0)
            z_prime = recombine_poison_clean(
                result.z_poison_final, self.z_clean, self.poison_idx, self.clean_idx, self.k
            )
            m_prime = _cos_sim_np(z_prime)
            np.testing.assert_allclose(m_prime, self.m_original, atol=1e-8)

    def test_steps_zero_reproduces_original_stage1_stage2_decision(self):
        for lp in LAMBDA_PRESERVES:
            result = mmd_minimize_transform(self.z_poison, self.z_clean, lambda_preserve=lp, steps=0)
            z_prime = recombine_poison_clean(
                result.z_poison_final, self.z_clean, self.poison_idx, self.clean_idx, self.k
            )
            m_prime = _cos_sim_np(z_prime)
            stage1 = concentration_stage1(m_prime)
            stage2 = stage2_pair_frequency(m_prime, stage1.n_adv_estimated)
            self.assertEqual(stage1.n_adv_estimated, self.stage1_original.n_adv_estimated)
            self.assertEqual(set(stage2.selected_indices), set(self.stage2_original.selected_indices))

    def test_steps_zero_trace_has_single_zero_loss_entry(self):
        result = mmd_minimize_transform(self.z_poison, self.z_clean, lambda_preserve=0.1, steps=0)
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(result.trace[0].step, 0)
        self.assertEqual(result.trace[0].preservation_loss, 0.0)


# --------------------------------------------------------------------------
# Test: output embeddings are finite / unit-normalized
# --------------------------------------------------------------------------

class TestMmdOutputFiniteAndNormalized(unittest.TestCase):
    def test_finite_across_lambda_preserve_and_steps_sweep(self):
        z_poison = _make_embeddings(10, 5, 64)
        z_clean = _make_embeddings(11, 5, 64)
        for lp in LAMBDA_PRESERVES:
            for st in STEPS_LIST:
                result = mmd_minimize_transform(z_poison, z_clean, lambda_preserve=lp, steps=st, lr=0.05)
                self.assertTrue(np.all(np.isfinite(result.z_poison_final)))
                for t in result.trace:
                    self.assertTrue(np.all(np.isfinite(t.z_poison_step)))
                    self.assertTrue(np.isfinite(t.mmd_loss))
                    self.assertTrue(np.isfinite(t.preservation_loss))
                    self.assertTrue(np.isfinite(t.total_loss))

    def test_final_rows_unit_norm(self):
        z_poison = _make_embeddings(12, 5, 48)
        z_clean = _make_embeddings(13, 5, 48)
        for lp in LAMBDA_PRESERVES:
            for st in STEPS_LIST:
                result = mmd_minimize_transform(z_poison, z_clean, lambda_preserve=lp, steps=st, lr=0.05)
                norms = np.linalg.norm(result.z_poison_final, axis=1)
                np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_trace_rows_unit_norm_at_every_step(self):
        z_poison = _make_embeddings(14, 5, 24)
        z_clean = _make_embeddings(15, 5, 24)
        result = mmd_minimize_transform(z_poison, z_clean, lambda_preserve=0.1, steps=20, lr=0.05)
        for t in result.trace:
            norms = np.linalg.norm(t.z_poison_step, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# --------------------------------------------------------------------------
# Test: preservation metrics are finite and valid (via compute_preservation_metrics)
# --------------------------------------------------------------------------

class TestMmdPreservationMetrics(unittest.TestCase):
    def test_preservation_metrics_finite_and_valid_range(self):
        from defense.coral_mmd_intervention import compute_preservation_metrics

        z_poison = _make_embeddings(20, 5, 32)
        z_clean = _make_embeddings(21, 5, 32)
        for lp in LAMBDA_PRESERVES:
            for st in STEPS_LIST:
                result = mmd_minimize_transform(z_poison, z_clean, lambda_preserve=lp, steps=st, lr=0.05)
                metrics = compute_preservation_metrics(z_poison, result.z_poison_final)
                self.assertTrue(np.isfinite(metrics.mean_l2_displacement))
                self.assertTrue(np.isfinite(metrics.max_l2_displacement))
                self.assertTrue(np.isfinite(metrics.mean_original_cosine))
                self.assertTrue(np.isfinite(metrics.min_original_cosine))
                self.assertGreaterEqual(metrics.mean_l2_displacement, 0.0)
                self.assertGreaterEqual(metrics.max_l2_displacement, 0.0)
                self.assertGreaterEqual(metrics.mean_original_cosine, -1.0 - 1e-9)
                self.assertLessEqual(metrics.mean_original_cosine, 1.0 + 1e-9)
                if st == 0:
                    self.assertAlmostEqual(metrics.mean_l2_displacement, 0.0, places=9)
                    self.assertAlmostEqual(metrics.mean_original_cosine, 1.0, places=9)


# --------------------------------------------------------------------------
# Validation / edge cases
# --------------------------------------------------------------------------

class TestMmdValidation(unittest.TestCase):
    def test_raises_for_gamma_non_positive(self):
        z_poison = _make_embeddings(1, 3, 8)
        z_clean = _make_embeddings(2, 3, 8)
        with self.assertRaises(ValueError):
            mmd_minimize_transform(z_poison, z_clean, lambda_preserve=0.1, gamma=0.0, steps=10)

    def test_raises_for_negative_steps(self):
        z_poison = _make_embeddings(1, 3, 8)
        z_clean = _make_embeddings(2, 3, 8)
        with self.assertRaises(ValueError):
            mmd_minimize_transform(z_poison, z_clean, lambda_preserve=0.1, steps=-1)

    def test_raises_for_non_positive_lr(self):
        z_poison = _make_embeddings(1, 3, 8)
        z_clean = _make_embeddings(2, 3, 8)
        with self.assertRaises(ValueError):
            mmd_minimize_transform(z_poison, z_clean, lambda_preserve=0.1, steps=10, lr=0.0)

    def test_raises_for_mismatched_dims(self):
        z_poison = _make_embeddings(1, 5, 16)
        z_clean = _make_embeddings(2, 5, 32)
        with self.assertRaises(ValueError):
            mmd_minimize_transform(z_poison, z_clean, lambda_preserve=0.1, steps=10)

    def test_on_step_callback_invoked_exactly_steps_plus_one_times(self):
        calls = []
        z_poison = _make_embeddings(1, 5, 16)
        z_clean = _make_embeddings(2, 5, 16)
        mmd_minimize_transform(
            z_poison, z_clean, lambda_preserve=0.1, steps=7, lr=0.05,
            on_step=lambda step, z, mmd, pres, total: calls.append(step),
        )
        self.assertEqual(calls, list(range(8)))


# --------------------------------------------------------------------------
# Test: no GPT/API calls / no forbidden imports
# --------------------------------------------------------------------------

class TestNoForbiddenImports(unittest.TestCase):
    def _assert_no_forbidden_imports(self, module_path):
        with open(module_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])
        forbidden = {"openai", "requests", "httpx", "urllib"}
        self.assertEqual(imported_modules & forbidden, set())

    def _assert_only_allowed_top_level_imports(self, module_path):
        """Beyond `torch`/`numpy`/`pandas` and local repo modules, no new
        third-party import should appear in the runner script (mirrors
        the plan's Test 8)."""
        with open(module_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        allowed_stdlib_or_local_prefixes = {
            "argparse", "json", "os", "platform", "subprocess", "sys", "datetime", "pathlib", "typing",
            "numpy", "pandas", "torch",
            "visualize_ragdefender_clusters", "run_cluster_normalized_poisoning",
            "build_batch_comparison_success_cases", "defense", "__future__",
        }
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])
        self.assertTrue(imported_modules.issubset(allowed_stdlib_or_local_prefixes),
                        f"unexpected imports: {imported_modules - allowed_stdlib_or_local_prefixes}")

    def test_runner_script_has_no_forbidden_imports(self):
        self._assert_no_forbidden_imports(_RUN_SCRIPT_PATH)

    def test_runner_script_has_only_allowed_imports(self):
        self._assert_only_allowed_top_level_imports(_RUN_SCRIPT_PATH)

    def test_module_has_no_forbidden_imports(self):
        module_path = os.path.join(REPO_ROOT, "defense", "coral_mmd_intervention.py")
        self._assert_no_forbidden_imports(module_path)

    def test_offline_env_vars_set_on_import(self):
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")


# --------------------------------------------------------------------------
# End-to-end smoke test for scripts/run_mmd_oracle_intervention.py
# --------------------------------------------------------------------------

class FakeSentenceTransformer:
    """Deterministic, dependency-free stand-in for SentenceTransformer, same
    pattern as tests/test_coral_ridge_intervention.py."""

    def encode(self, text_list, convert_to_tensor=True):
        vectors = []
        for t in text_list:
            digest = hashlib.md5(t.encode("utf-8")).hexdigest()
            seed = int(digest[:8], 16)
            gen = torch.Generator().manual_seed(seed)
            vectors.append(torch.rand(24, generator=gen))
        return torch.stack(vectors)


def _make_diag_record(query_id, k=10, n_poison=5, n_clean=5):
    doc_ids = [f"adv::{query_id}::{i}" for i in range(n_poison)] + [f"clean-{i}" for i in range(n_clean)]
    is_poison = [True] * n_poison + [False] * n_clean
    return {
        "query_id": query_id, "dataset": "hotpotqa", "model": "gpt4", "attack": "LM_targeted",
        "defense": "ragdefender_original", "k": k, "N_injected": n_poison,
        "retrieved_doc_ids": doc_ids, "retrieved_is_poison": is_poison,
        "N_retrieved_poison": n_poison, "N_retrieved_clean": n_clean,
        "removed_poison": n_poison, "residual_poison_fraction": 0.0,
    }


def _make_qr_entry(query_id, n_poison=5, n_clean=5):
    texts = [f"Poisoned passage number {i} about the target question for {query_id}." for i in range(n_poison)] + \
            [f"Unrelated clean fact number {i} about something else entirely for {query_id}." for i in range(n_clean)]
    prompt_no_defense = (
        "You are a helpful assistant... \n\nContexts: " + "\n".join(texts) +
        " \n\nQuery: Some question? \n\nAnswer:"
    )
    return {"id": query_id, "question": "Some question?", "input_prompt_no_defense": prompt_no_defense}


class TestEndToEndSmoke(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.tmpdir, "output")
        self.e1_output_dir = os.path.join(self.tmpdir, "e1_output")
        os.makedirs(self.e1_output_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.tested_ids = ["qid_mmd_1", "qid_mmd_2"]
        records = [_make_diag_record(q) for q in self.tested_ids]
        self.diagnostics_path = os.path.join(self.tmpdir, "diag.jsonl")
        with open(self.diagnostics_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        self.query_results_dir = os.path.join(self.tmpdir, "query_results")
        os.makedirs(self.query_results_dir, exist_ok=True)
        qr_entries = [_make_qr_entry(q) for q in self.tested_ids]
        with open(os.path.join(self.query_results_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump([{"iter_0": qr_entries}], f)

        self.patcher = mock.patch.object(run_script.viz, "load_embedder", return_value=FakeSentenceTransformer())
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_run_writes_all_required_files_and_valid_metrics(self):
        run_dir = run_script.main([
            "--diagnostics_jsonl", self.diagnostics_path,
            "--query_results_dir", self.query_results_dir,
            "--output_dir", self.output_dir,
            "--e1_output_dir", self.e1_output_dir,
            "--lambda_preserves", "0.1", "1.0",
            "--steps_list", "0", "5",
        ])
        self.assertTrue(run_dir.exists())
        self.assertIn("_mmd_hotpotqa_k10_N5", run_dir.name)

        expected_files = [
            "run_config.json", "manifest.json", "MMD_SWEEP.csv",
            "MMD_REPORT.md", "METHOD_COMPARISON_FORMAL.csv",
        ]
        for rel in expected_files:
            self.assertTrue((run_dir / rel).exists(), f"missing expected output file: {rel}")
        self.assertTrue((run_dir / "traces").is_dir())
        self.assertTrue((run_dir / "similarity_matrices").is_dir())

        sweep = pd.read_csv(run_dir / "MMD_SWEEP.csv")
        self.assertEqual(len(sweep), len(self.tested_ids) * 2 * 2)  # 2 queries x 2 lambda_preserves x 2 steps
        for col in ["lambda_preserve", "steps", "gamma", "lr", "coral_distance_before", "coral_distance_after",
                    "mmd_distance_before", "mmd_distance_after", "decision_label", "N_adv",
                    "removed_poison", "removed_clean", "residual_poison_fraction", "selected_indices",
                    "top_pair_pp", "top_pair_pc", "top_pair_cc", "mean_poison_l2_displacement",
                    "max_poison_l2_displacement", "mean_poison_original_cosine",
                    "min_poison_original_cosine", "first_failure_step_in_trace"]:
            self.assertIn(col, sweep.columns)
        for col in ["coral_distance_before", "coral_distance_after", "mmd_distance_before",
                    "mmd_distance_after", "mean_poison_l2_displacement", "max_poison_l2_displacement",
                    "mean_poison_original_cosine", "min_poison_original_cosine"]:
            self.assertTrue(np.all(np.isfinite(sweep[col])), f"non-finite values in column {col}")
        self.assertTrue(np.all(sweep["mean_poison_l2_displacement"] >= 0.0))

        steps_zero = sweep[sweep["steps"] == 0]
        self.assertTrue(np.allclose(steps_zero["coral_distance_before"], steps_zero["coral_distance_after"]))
        np.testing.assert_allclose(steps_zero["mean_poison_l2_displacement"], 0.0, atol=1e-6)
        np.testing.assert_allclose(steps_zero["mean_poison_original_cosine"], 1.0, atol=1e-6)

        with open(run_dir / "run_config.json") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["intervention"], "MMD_MINIMIZE")
        self.assertFalse(cfg["oracle_constraints"]["gpt_or_api_calls_made"])
        self.assertFalse(cfg["oracle_constraints"]["claims_text_realizable_attack"])
        self.assertTrue(cfg["oracle_constraints"]["mmd_optimizer_implemented"])
        self.assertFalse(cfg["oracle_constraints"]["dan_trained"])
        self.assertFalse(cfg["oracle_constraints"]["e1_rerun"])
        self.assertFalse(cfg["oracle_constraints"]["coral_pca_rerun"])
        self.assertFalse(cfg["oracle_constraints"]["coral_ridge_rerun"])
        self.assertEqual(cfg["oracle_constraints"]["baseline_files_modified"], [])

        method_comparison = pd.read_csv(run_dir / "METHOD_COMPARISON_FORMAL.csv")
        self.assertIn("method", method_comparison.columns)
        self.assertTrue((method_comparison["method"].str.startswith("MMD_")).any())

        report_text = (run_dir / "MMD_REPORT.md").read_text(encoding="utf-8")
        self.assertIn("No GPT/API calls were made", report_text)
        self.assertIn("Perturbation / preservation metrics", report_text)
        self.assertIn("Method comparison", report_text)
        self.assertIn("Does MMD optimization cause residual-poison failures?", report_text)
        self.assertIn("top_pair_pp", report_text)
        self.assertIn("E1 comparison skipped", report_text)
        self.assertIn("CORAL-PCA comparison skipped", report_text)
        self.assertIn("CORAL-ridge comparison skipped", report_text)

    def test_trace_files_written_with_expected_columns_and_row_counts(self):
        run_dir = run_script.main([
            "--diagnostics_jsonl", self.diagnostics_path,
            "--query_results_dir", self.query_results_dir,
            "--output_dir", self.output_dir,
            "--e1_output_dir", self.e1_output_dir,
            "--lambda_preserves", "0.1",
            "--steps_list", "0", "5",
        ])
        trace_dir = run_dir / "traces"
        for qid in self.tested_ids:
            for st, expected_rows in [(0, 1), (5, 6)]:
                trace_path = trace_dir / f"{qid}_lp0.1_steps{st}_trace.csv"
                self.assertTrue(trace_path.exists(), f"missing trace file: {trace_path}")
                trace_df = pd.read_csv(trace_path)
                self.assertEqual(len(trace_df), expected_rows)
                for col in ["step", "total_loss", "mmd_loss", "preservation_loss", "mean_pp_similarity",
                            "top_pair_pp", "removed_poison", "removed_clean", "residual_poison_fraction",
                            "decision_label"]:
                    self.assertIn(col, trace_df.columns)
                self.assertEqual(list(trace_df["step"]), list(range(expected_rows)))
                self.assertEqual(trace_df.iloc[0]["preservation_loss"], 0.0)

    def test_similarity_matrices_saved(self):
        run_dir = run_script.main([
            "--diagnostics_jsonl", self.diagnostics_path,
            "--query_results_dir", self.query_results_dir,
            "--output_dir", self.output_dir,
            "--e1_output_dir", self.e1_output_dir,
            "--lambda_preserves", "0.1",
            "--steps_list", "0",
        ])
        sim_dir = run_dir / "similarity_matrices"
        for qid in self.tested_ids:
            self.assertTrue((sim_dir / f"{qid}_original_M.npy").exists())
            self.assertTrue((sim_dir / f"{qid}_lp0.1_steps0_M.npy").exists())

    def test_rejects_negative_steps(self):
        with self.assertRaises(ValueError):
            run_script.main([
                "--diagnostics_jsonl", self.diagnostics_path,
                "--query_results_dir", self.query_results_dir,
                "--output_dir", self.output_dir,
                "--e1_output_dir", self.e1_output_dir,
                "--lambda_preserves", "0.1",
                "--steps_list", "-1",
            ])


if __name__ == "__main__":
    unittest.main()
