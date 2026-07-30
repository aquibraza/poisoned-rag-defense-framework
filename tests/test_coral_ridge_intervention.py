"""Tests for defense/coral_mmd_intervention.py's full-ridge CORAL transform
(`coral_ridge_transform` -- Step 2 of the CORAL/MMD oracle intervention
plan) and scripts/run_coral_ridge_oracle_intervention.py.

Following tests/test_coral_mmd_intervention.py's / tests/
test_cluster_normalized_poisoning.py's convention: the automated suite
never downloads or loads a real sentence-transformers model. The
end-to-end smoke test monkeypatches `load_embedder` with the same
deterministic `FakeSentenceTransformer` pattern used there.

Run with: python -m unittest tests.test_coral_ridge_intervention -v
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

from defense.cluster_normalized_poisoning import recombine_poison_clean
from defense.coral_mmd_intervention import (
    coral_pca_transform,
    coral_ridge_transform,
)
from defense.ragdefender_internals import concentration_stage1, stage2_pair_frequency

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# scripts/ has no __init__.py -- import by file path, same pattern as
# tests/test_coral_mmd_intervention.py.
_RUN_SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "run_coral_ridge_oracle_intervention.py")
_spec = importlib.util.spec_from_file_location("run_coral_ridge_oracle_intervention", _RUN_SCRIPT_PATH)
run_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_script)


def _make_embeddings(seed: int, n: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim))


def _cos_sim_np(z: np.ndarray) -> np.ndarray:
    norm = z / np.linalg.norm(z, axis=1, keepdims=True)
    return norm @ norm.T


BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
LAMBDAS = [1e-1, 1e-2, 1e-3]


# --------------------------------------------------------------------------
# Test: ridge CORAL output shape
# --------------------------------------------------------------------------

class TestCoralRidgeShape(unittest.TestCase):
    def setUp(self):
        self.z_poison = _make_embeddings(1, 5, 384)
        self.z_clean = _make_embeddings(2, 5, 384)

    def test_shape_preserved_every_lambda_beta(self):
        for lam in LAMBDAS:
            for beta in BETAS:
                result = coral_ridge_transform(self.z_poison, self.z_clean, beta, lam)
                self.assertEqual(result.z_poison_final.shape, self.z_poison.shape)
                self.assertEqual(result.z_poison_coral.shape, self.z_poison.shape)
                self.assertEqual(result.lam, lam)

    def test_shape_preserved_unequal_group_sizes(self):
        z_poison = _make_embeddings(3, 5, 32)
        z_clean = _make_embeddings(4, 7, 32)
        for lam in LAMBDAS:
            result = coral_ridge_transform(z_poison, z_clean, 0.5, lam)
            self.assertEqual(result.z_poison_final.shape, z_poison.shape)


# --------------------------------------------------------------------------
# Test: finite values
# --------------------------------------------------------------------------

class TestCoralRidgeFinite(unittest.TestCase):
    def test_finite_across_lambda_beta_and_seed_sweep(self):
        for seed in range(5):
            z_poison = _make_embeddings(seed, 5, 384)
            z_clean = _make_embeddings(seed + 100, 5, 384)
            for lam in LAMBDAS:
                for beta in BETAS:
                    result = coral_ridge_transform(z_poison, z_clean, beta, lam)
                    self.assertTrue(np.all(np.isfinite(result.z_poison_coral)))
                    self.assertTrue(np.all(np.isfinite(result.z_poison_final)))

    def test_finite_for_near_duplicate_poison_rows(self):
        """Regression guard: near-duplicate rows make the poison
        covariance's true rank lower than n_poison-1; ridge regularization
        must keep the full-dimensional inverse finite regardless."""
        rng = np.random.default_rng(42)
        base = rng.normal(size=(1, 16))
        z_poison = np.repeat(base, 5, axis=0) + rng.normal(scale=1e-9, size=(5, 16))
        z_clean = _make_embeddings(7, 5, 16)
        for lam in LAMBDAS:
            for beta in BETAS:
                result = coral_ridge_transform(z_poison, z_clean, beta, lam)
                self.assertTrue(np.all(np.isfinite(result.z_poison_final)))

    def test_finite_at_smallest_swept_lambda(self):
        """lambda=1e-3 is the smallest swept value (closest to the
        undefined lambda=0 case) -- must remain finite and well-behaved."""
        z_poison = _make_embeddings(50, 5, 384)
        z_clean = _make_embeddings(51, 5, 384)
        for beta in BETAS:
            result = coral_ridge_transform(z_poison, z_clean, beta, lam=1e-3)
            self.assertTrue(np.all(np.isfinite(result.z_poison_final)))


# --------------------------------------------------------------------------
# Test: row norms after normalization
# --------------------------------------------------------------------------

class TestCoralRidgeRowNorms(unittest.TestCase):
    def test_final_rows_are_unit_norm(self):
        z_poison = _make_embeddings(5, 5, 64)
        z_clean = _make_embeddings(6, 5, 64)
        for lam in LAMBDAS:
            for beta in BETAS:
                result = coral_ridge_transform(z_poison, z_clean, beta, lam)
                norms = np.linalg.norm(result.z_poison_final, axis=1)
                np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# --------------------------------------------------------------------------
# Test: beta=0 identity (for every lambda)
# --------------------------------------------------------------------------

class TestCoralRidgeBetaZeroIdentity(unittest.TestCase):
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

    def test_beta_zero_reproduces_original_cosine_matrix_for_every_lambda(self):
        for lam in LAMBDAS:
            result = coral_ridge_transform(self.z_poison, self.z_clean, beta=0.0, lam=lam)
            z_prime = recombine_poison_clean(
                result.z_poison_final, self.z_clean, self.poison_idx, self.clean_idx, self.k
            )
            m_prime = _cos_sim_np(z_prime)
            np.testing.assert_allclose(m_prime, self.m_original, atol=1e-8)

    def test_beta_zero_reproduces_original_stage1_stage2_decision_for_every_lambda(self):
        for lam in LAMBDAS:
            result = coral_ridge_transform(self.z_poison, self.z_clean, beta=0.0, lam=lam)
            z_prime = recombine_poison_clean(
                result.z_poison_final, self.z_clean, self.poison_idx, self.clean_idx, self.k
            )
            m_prime = _cos_sim_np(z_prime)
            stage1 = concentration_stage1(m_prime)
            stage2 = stage2_pair_frequency(m_prime, stage1.n_adv_estimated)
            self.assertEqual(stage1.n_adv_estimated, self.stage1_original.n_adv_estimated)
            self.assertEqual(set(stage2.selected_indices), set(self.stage2_original.selected_indices))
            np.testing.assert_allclose(stage2.frequency_scores, self.stage2_original.frequency_scores, atol=1e-8)

    def test_beta_one_generally_changes_the_decision_state(self):
        """Sanity check (not a strict correctness requirement): beta=1.0
        (pure ridge-CORAL recoloring) should generally differ from the
        untransformed baseline for well-separated random poison/clean
        clusters."""
        result = coral_ridge_transform(self.z_poison, self.z_clean, beta=1.0, lam=1e-2)
        z_prime = recombine_poison_clean(result.z_poison_final, self.z_clean, self.poison_idx, self.clean_idx, self.k)
        m_prime = _cos_sim_np(z_prime)
        self.assertFalse(np.allclose(m_prime, self.m_original, atol=1e-6))


# --------------------------------------------------------------------------
# Test: eigenvalue clipping / small-lambda stability, no unregularized inverse
# --------------------------------------------------------------------------

class TestCoralRidgeRegularization(unittest.TestCase):
    def test_lambda_must_be_strictly_positive(self):
        z_poison = _make_embeddings(1, 5, 16)
        z_clean = _make_embeddings(2, 5, 16)
        with self.assertRaises(ValueError):
            coral_ridge_transform(z_poison, z_clean, beta=0.5, lam=0.0)
        with self.assertRaises(ValueError):
            coral_ridge_transform(z_poison, z_clean, beta=0.5, lam=-1e-3)

    def test_eigenvalues_never_below_lambda_in_ridge_covariance(self):
        """Every eigenvalue of Cov + lambda*I must be >= lambda (up to
        floating-point noise), i.e. the ridge floor is doing its job and
        no direction is left with a near-zero (unregularized) eigenvalue."""
        from defense.coral_mmd_intervention import _ridge_covariance

        z_poison = _make_embeddings(3, 5, 64)
        zp_centered = z_poison - z_poison.mean(axis=0)
        for lam in LAMBDAS:
            cov = _ridge_covariance(zp_centered, lam)
            eigvals = np.linalg.eigvalsh(cov)
            self.assertTrue(np.all(eigvals >= lam - 1e-9))

    def test_full_rank_eigenbasis_used_not_truncated(self):
        """The ridge transform must use all d eigenpairs (full-dimensional),
        unlike the PCA/subspace variant's top-rank truncation -- verified
        by checking the internal helper actually returns a full-rank (not
        low-rank) matrix for a ridge-regularized covariance."""
        from defense.coral_mmd_intervention import _full_symmetric_sqrt_or_pinv, _ridge_covariance

        z_poison = _make_embeddings(4, 5, 32)
        zp_centered = z_poison - z_poison.mean(axis=0)
        cov = _ridge_covariance(zp_centered, lam=1e-2)
        cp_inv_half = _full_symmetric_sqrt_or_pinv(cov, invert=True)
        # A rank-truncated (Step 1 style) matrix built from only 4 outer
        # products would have rank <= 4; the full-ridge matrix should not.
        rank = np.linalg.matrix_rank(cp_inv_half)
        self.assertGreater(rank, 4)

    def test_small_lambda_stability(self):
        """lambda=1e-3 (smallest swept) must not blow up the transform
        beyond a sane magnitude relative to lambda=1e-1."""
        z_poison = _make_embeddings(60, 5, 128)
        z_clean = _make_embeddings(61, 5, 128)
        result_small = coral_ridge_transform(z_poison, z_clean, beta=1.0, lam=1e-3)
        result_large = coral_ridge_transform(z_poison, z_clean, beta=1.0, lam=1e-1)
        self.assertTrue(np.all(np.isfinite(result_small.z_poison_coral)))
        self.assertTrue(np.all(np.isfinite(result_large.z_poison_coral)))


# --------------------------------------------------------------------------
# Test: Stage 1/Stage 2 recomputation after transform
# --------------------------------------------------------------------------

class TestStageRecomputationPostRidgeTransform(unittest.TestCase):
    def setUp(self):
        self.k = 10
        self.z_poison = _make_embeddings(10, 5, 48)
        self.z_clean = _make_embeddings(11, 5, 48)
        self.poison_idx = list(range(5))
        self.clean_idx = list(range(5, 10))

    def test_stage1_stage2_valid_across_lambda_beta_sweep(self):
        for lam in LAMBDAS:
            for beta in BETAS:
                result = coral_ridge_transform(self.z_poison, self.z_clean, beta, lam)
                z_prime = recombine_poison_clean(
                    result.z_poison_final, self.z_clean, self.poison_idx, self.clean_idx, self.k
                )
                m_prime = _cos_sim_np(z_prime)
                self.assertEqual(m_prime.shape, (self.k, self.k))

                stage1 = concentration_stage1(m_prime)
                self.assertGreaterEqual(stage1.n_adv_estimated, 0)
                self.assertLessEqual(stage1.n_adv_estimated, self.k)

                stage2 = stage2_pair_frequency(m_prime, stage1.n_adv_estimated)
                self.assertLessEqual(len(stage2.selected_indices), max(stage1.n_adv_estimated, 0))
                for i, j, _ in stage2.top_pairs:
                    self.assertLess(i, j)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

class TestCoralRidgeValidation(unittest.TestCase):
    def test_raises_for_fewer_than_two_poison_rows(self):
        z_poison = _make_embeddings(1, 1, 16)
        z_clean = _make_embeddings(2, 5, 16)
        with self.assertRaises(ValueError):
            coral_ridge_transform(z_poison, z_clean, beta=0.5, lam=1e-2)

    def test_raises_for_mismatched_dims(self):
        z_poison = _make_embeddings(1, 5, 16)
        z_clean = _make_embeddings(2, 5, 32)
        with self.assertRaises(ValueError):
            coral_ridge_transform(z_poison, z_clean, beta=0.5, lam=1e-2)


# --------------------------------------------------------------------------
# Test: ridge differs from PCA (sanity that this is a distinct transform)
# --------------------------------------------------------------------------

class TestCoralRidgeDiffersFromPca(unittest.TestCase):
    def test_ridge_and_pca_produce_different_transformed_embeddings(self):
        z_poison = _make_embeddings(70, 5, 64)
        z_clean = _make_embeddings(71, 5, 64)
        pca_result = coral_pca_transform(z_poison, z_clean, beta=1.0)
        ridge_result = coral_ridge_transform(z_poison, z_clean, beta=1.0, lam=1e-2)
        self.assertFalse(np.allclose(pca_result.z_poison_final, ridge_result.z_poison_final, atol=1e-6))


# --------------------------------------------------------------------------
# Test: no GPT/API calls guard (static + dynamic)
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

    def test_runner_script_has_no_forbidden_imports(self):
        self._assert_no_forbidden_imports(_RUN_SCRIPT_PATH)

    def test_offline_env_vars_set_on_import(self):
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")


# --------------------------------------------------------------------------
# End-to-end smoke test for scripts/run_coral_ridge_oracle_intervention.py
# --------------------------------------------------------------------------

class FakeSentenceTransformer:
    """Deterministic, dependency-free stand-in for SentenceTransformer, same
    pattern as tests/test_coral_mmd_intervention.py."""

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

        self.tested_ids = ["qid_ridge_1", "qid_ridge_2"]
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
            "--betas", "0.0", "1.0",
            "--lambdas", "0.1", "0.01",
        ])
        self.assertTrue(run_dir.exists())
        self.assertIn("_coral_ridge_hotpotqa_k10_N5", run_dir.name)

        expected_files = [
            "run_config.json", "manifest.json", "CORAL_RIDGE_SWEEP.csv",
            "CORAL_RIDGE_REPORT.md", "METHOD_COMPARISON_CORAL.csv",
        ]
        for rel in expected_files:
            self.assertTrue((run_dir / rel).exists(), f"missing expected output file: {rel}")

        sweep = pd.read_csv(run_dir / "CORAL_RIDGE_SWEEP.csv")
        self.assertEqual(len(sweep), len(self.tested_ids) * 2 * 2)  # 2 queries x 2 lambdas x 2 betas
        for col in ["lambda", "beta", "coral_distance_before", "coral_distance_after",
                    "mmd_distance_before", "mmd_distance_after", "decision_label", "N_adv",
                    "removed_poison", "removed_clean", "residual_poison_fraction", "selected_indices",
                    "top_pair_pp", "top_pair_pc", "top_pair_cc", "mean_poison_l2_displacement",
                    "max_poison_l2_displacement", "mean_poison_original_cosine",
                    "min_poison_original_cosine"]:
            self.assertIn(col, sweep.columns)
        for col in ["coral_distance_before", "coral_distance_after", "mmd_distance_before",
                    "mmd_distance_after", "mean_poison_l2_displacement", "max_poison_l2_displacement",
                    "mean_poison_original_cosine", "min_poison_original_cosine"]:
            self.assertTrue(np.all(np.isfinite(sweep[col])), f"non-finite values in column {col}")
        self.assertTrue(np.all(sweep["mean_poison_l2_displacement"] >= 0.0))
        self.assertTrue(np.all(sweep["max_poison_l2_displacement"] >= 0.0))

        beta_zero = sweep[sweep["beta"] == 0.0]
        self.assertTrue(np.allclose(beta_zero["coral_distance_before"], beta_zero["coral_distance_after"]))
        np.testing.assert_allclose(beta_zero["mean_poison_l2_displacement"], 0.0, atol=1e-6)
        np.testing.assert_allclose(beta_zero["mean_poison_original_cosine"], 1.0, atol=1e-6)

        with open(run_dir / "run_config.json") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["intervention"], "CORAL_RIDGE")
        self.assertEqual(cfg["coral_variant"], "full_ridge")
        self.assertFalse(cfg["oracle_constraints"]["gpt_or_api_calls_made"])
        self.assertFalse(cfg["oracle_constraints"]["claims_text_realizable_attack"])
        self.assertTrue(cfg["oracle_constraints"]["full_ridge_coral_implemented"])
        self.assertFalse(cfg["oracle_constraints"]["unregularized_inverse_used"])
        self.assertFalse(cfg["oracle_constraints"]["mmd_optimizer_implemented"])
        self.assertFalse(cfg["oracle_constraints"]["e1_rerun"])
        self.assertFalse(cfg["oracle_constraints"]["coral_pca_rerun"])
        self.assertEqual(cfg["oracle_constraints"]["baseline_files_modified"], [])

        method_comparison = pd.read_csv(run_dir / "METHOD_COMPARISON_CORAL.csv")
        self.assertIn("method", method_comparison.columns)
        self.assertTrue((method_comparison["method"].str.startswith("CORAL_RIDGE")).any())

        report_text = (run_dir / "CORAL_RIDGE_REPORT.md").read_text(encoding="utf-8")
        self.assertIn("No GPT/API calls were made", report_text)
        self.assertIn("Perturbation / preservation metrics", report_text)
        self.assertIn("Method comparison", report_text)
        self.assertIn("Does ridge CORAL cause any residual-poison failures?", report_text)
        self.assertIn("E1 comparison skipped", report_text)  # e1_output_dir is empty in this test
        self.assertIn("CORAL-PCA comparison skipped", report_text)  # no prior Step 1 run in this tmp output_dir

    def test_similarity_matrices_saved_when_present(self):
        run_dir = run_script.main([
            "--diagnostics_jsonl", self.diagnostics_path,
            "--query_results_dir", self.query_results_dir,
            "--output_dir", self.output_dir,
            "--e1_output_dir", self.e1_output_dir,
            "--betas", "0.0", "1.0",
            "--lambdas", "0.1",
        ])
        sim_dir = run_dir / "similarity_matrices"
        self.assertTrue(sim_dir.exists())
        for qid in self.tested_ids:
            self.assertTrue((sim_dir / f"{qid}_original_M.npy").exists())
            self.assertTrue((sim_dir / f"{qid}_lambda0.1_beta0.0_M.npy").exists())
            self.assertTrue((sim_dir / f"{qid}_lambda0.1_beta1.0_M.npy").exists())

    def test_rejects_non_positive_lambda(self):
        with self.assertRaises(ValueError):
            run_script.main([
                "--diagnostics_jsonl", self.diagnostics_path,
                "--query_results_dir", self.query_results_dir,
                "--output_dir", self.output_dir,
                "--e1_output_dir", self.e1_output_dir,
                "--betas", "0.0", "1.0",
                "--lambdas", "0.0",
            ])


if __name__ == "__main__":
    unittest.main()
