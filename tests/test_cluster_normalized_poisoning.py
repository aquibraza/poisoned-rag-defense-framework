"""Tests for defense/cluster_normalized_poisoning.py and
scripts/run_cluster_normalized_poisoning.py, per
docs/CLUSTER_NORMALIZED_POISONING_EXECUTION_PLAN.md section 9.

Following tests/test_ragdefender_cluster_viz.py's convention: the automated
suite never downloads or loads a real sentence-transformers model. The
end-to-end smoke test monkeypatches `load_embedder` with the same
deterministic `FakeSentenceTransformer` pattern used there.

Run with: python -m unittest tests.test_cluster_normalized_poisoning -v
"""
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from defense.cluster_normalized_poisoning import (
    ANCHOR_STRATEGIES,
    anchor_interpolate,
    centroid_interpolate,
    is_bijection,
    recombine_poison_clean,
    resolve_anchor_permutation,
    split_poison_clean,
)
from defense.ragdefender_internals import concentration_stage1, stage2_pair_frequency

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# scripts/ has no __init__.py -- import by file path, same pattern as
# tests/test_ragdefender_cluster_viz.py.
_RUN_SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "run_cluster_normalized_poisoning.py")
_spec = importlib.util.spec_from_file_location("run_cluster_normalized_poisoning", _RUN_SCRIPT_PATH)
run_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_script)


# --------------------------------------------------------------------------
# Deterministic fixtures: N=5 poison, N=5 clean, dim=8 (small but enough to
# have non-degenerate cosine structure).
# --------------------------------------------------------------------------

def _make_embeddings(seed: int, n: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim))


class TestSplitRecombine(unittest.TestCase):
    def test_split_and_recombine_roundtrip(self):
        z = np.arange(40).reshape(10, 4).astype(np.float64)
        is_poison = [True, False, True, True, False, False, True, False, True, False]
        z_poison, z_clean, poison_idx, clean_idx = split_poison_clean(z, is_poison)
        self.assertEqual(z_poison.shape, (5, 4))
        self.assertEqual(z_clean.shape, (5, 4))
        recombined = recombine_poison_clean(z_poison, z_clean, poison_idx, clean_idx, k=10)
        np.testing.assert_array_equal(recombined, z)


# --------------------------------------------------------------------------
# Test 1: shape preservation
# --------------------------------------------------------------------------

class TestShapePreservation(unittest.TestCase):
    def setUp(self):
        self.z_poison = _make_embeddings(1, 5, 8)
        self.z_clean = _make_embeddings(2, 5, 8)
        self.alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    def test_e0_shape_preserved_every_alpha(self):
        for alpha in self.alphas:
            out = centroid_interpolate(self.z_poison, self.z_clean, alpha)
            self.assertEqual(out.shape, self.z_poison.shape)

    def test_e1_shape_preserved_every_strategy_every_alpha(self):
        for strategy in ANCHOR_STRATEGIES:
            assignment = resolve_anchor_permutation(self.z_poison, self.z_clean, strategy, random_seed=12)
            for alpha in self.alphas:
                out = anchor_interpolate(self.z_poison, self.z_clean, assignment.pi, alpha)
                self.assertEqual(out.shape, self.z_poison.shape)


# --------------------------------------------------------------------------
# Test 2/3/4: cosine matrix shape + Stage-1/Stage-2 recomputation run
# post-transform (pure math, no embedder needed -- operates directly on
# already-encoded vectors via a plain numpy/torch cosine, matching the
# semantics of sentence_transformers.util.cos_sim).
# --------------------------------------------------------------------------

def _cos_sim_np(z: np.ndarray) -> np.ndarray:
    norm = z / np.linalg.norm(z, axis=1, keepdims=True)
    return norm @ norm.T


class TestStageRecomputationPostTransform(unittest.TestCase):
    def setUp(self):
        self.k = 10
        self.z_poison = _make_embeddings(3, 5, 8)
        self.z_clean = _make_embeddings(4, 5, 8)
        self.is_poison = [True] * 5 + [False] * 5
        self.poison_idx = list(range(5))
        self.clean_idx = list(range(5, 10))
        self.alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    def test_cosine_matrix_shape_and_stage_recompute_e0(self):
        for alpha in self.alphas:
            zp = centroid_interpolate(self.z_poison, self.z_clean, alpha)
            z_prime = recombine_poison_clean(zp, self.z_clean, self.poison_idx, self.clean_idx, self.k)
            m_prime = _cos_sim_np(z_prime)
            self.assertEqual(m_prime.shape, (self.k, self.k))

            stage1 = concentration_stage1(m_prime)
            self.assertGreaterEqual(stage1.n_adv_estimated, 0)
            self.assertLessEqual(stage1.n_adv_estimated, self.k)

            stage2 = stage2_pair_frequency(m_prime, stage1.n_adv_estimated)
            self.assertLessEqual(len(stage2.selected_indices), max(stage1.n_adv_estimated, 0))
            for i, j, _ in stage2.top_pairs:
                self.assertLess(i, j)

    def test_cosine_matrix_shape_and_stage_recompute_e1_every_strategy(self):
        for strategy in ANCHOR_STRATEGIES:
            assignment = resolve_anchor_permutation(self.z_poison, self.z_clean, strategy, random_seed=12)
            for alpha in self.alphas:
                zp = anchor_interpolate(self.z_poison, self.z_clean, assignment.pi, alpha)
                z_prime = recombine_poison_clean(zp, self.z_clean, self.poison_idx, self.clean_idx, self.k)
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
# Test 5: identity regression at alpha=1.0
# --------------------------------------------------------------------------

class TestIdentityRegressionAtAlphaOne(unittest.TestCase):
    def setUp(self):
        self.k = 10
        self.z_poison = _make_embeddings(5, 5, 8)
        self.z_clean = _make_embeddings(6, 5, 8)
        self.is_poison = [True] * 5 + [False] * 5
        self.poison_idx = list(range(5))
        self.clean_idx = list(range(5, 10))
        self.z_original = recombine_poison_clean(self.z_poison, self.z_clean, self.poison_idx, self.clean_idx, self.k)
        self.m_original = _cos_sim_np(self.z_original)
        self.stage1_original = concentration_stage1(self.m_original)
        self.stage2_original = stage2_pair_frequency(self.m_original, self.stage1_original.n_adv_estimated)

    def _assert_matches_original(self, m_prime):
        np.testing.assert_allclose(m_prime, self.m_original, atol=1e-10)
        stage1 = concentration_stage1(m_prime)
        stage2 = stage2_pair_frequency(m_prime, stage1.n_adv_estimated)
        self.assertEqual(stage1.n_adv_estimated, self.stage1_original.n_adv_estimated)
        self.assertEqual(set(stage2.selected_indices), set(self.stage2_original.selected_indices))
        np.testing.assert_allclose(stage2.frequency_scores, self.stage2_original.frequency_scores, atol=1e-10)

    def test_e0_alpha_one_reproduces_original(self):
        zp = centroid_interpolate(self.z_poison, self.z_clean, 1.0)
        z_prime = recombine_poison_clean(zp, self.z_clean, self.poison_idx, self.clean_idx, self.k)
        self._assert_matches_original(_cos_sim_np(z_prime))

    def test_e1_alpha_one_reproduces_original_every_strategy(self):
        for strategy in ANCHOR_STRATEGIES:
            assignment = resolve_anchor_permutation(self.z_poison, self.z_clean, strategy, random_seed=12)
            zp = anchor_interpolate(self.z_poison, self.z_clean, assignment.pi, 1.0)
            z_prime = recombine_poison_clean(zp, self.z_clean, self.poison_idx, self.clean_idx, self.k)
            self._assert_matches_original(_cos_sim_np(z_prime))


# --------------------------------------------------------------------------
# Test 6: anchor-assignment correctness and bijectivity
# --------------------------------------------------------------------------

class TestAnchorAssignmentCorrectness(unittest.TestCase):
    def setUp(self):
        self.z_poison = _make_embeddings(7, 5, 8)
        self.z_clean = _make_embeddings(8, 5, 8)

    def test_rank_aligned_is_identity_permutation(self):
        assignment = resolve_anchor_permutation(self.z_poison, self.z_clean, "rank_aligned")
        self.assertEqual(assignment.pi, [0, 1, 2, 3, 4])
        self.assertTrue(assignment.is_bijection)

    def _brute_force_sum(self, pi):
        zp = self.z_poison / np.linalg.norm(self.z_poison, axis=1, keepdims=True)
        zc = self.z_clean / np.linalg.norm(self.z_clean, axis=1, keepdims=True)
        cos = zp @ zc.T
        return sum(cos[i, pi[i]] for i in range(len(pi)))

    def test_nearest_bijection_maximizes_joint_sum_over_all_permutations(self):
        assignment = resolve_anchor_permutation(self.z_poison, self.z_clean, "nearest_bijection")
        chosen_sum = self._brute_force_sum(assignment.pi)
        best_over_all = max(self._brute_force_sum(list(p)) for p in itertools.permutations(range(5)))
        self.assertAlmostEqual(chosen_sum, best_over_all, places=8)
        self.assertAlmostEqual(assignment.objective_value, best_over_all, places=8)

    def test_farthest_bijection_minimizes_joint_sum_over_all_permutations(self):
        assignment = resolve_anchor_permutation(self.z_poison, self.z_clean, "farthest_bijection")
        chosen_sum = self._brute_force_sum(assignment.pi)
        worst_over_all = min(self._brute_force_sum(list(p)) for p in itertools.permutations(range(5)))
        self.assertAlmostEqual(chosen_sum, worst_over_all, places=8)
        self.assertAlmostEqual(assignment.objective_value, worst_over_all, places=8)

    def test_every_strategy_returns_a_genuine_bijection(self):
        """Direct regression guard for the bug this plan's Rev 2 fixed: an
        earlier independent-argmax/argmin `nearest`/`farthest` could map
        multiple poison points to the same clean anchor."""
        for strategy in ANCHOR_STRATEGIES:
            assignment = resolve_anchor_permutation(self.z_poison, self.z_clean, strategy, random_seed=12)
            self.assertTrue(assignment.is_bijection, f"strategy={strategy} produced a non-bijective pi")
            self.assertTrue(is_bijection(assignment.pi, 5))
            self.assertEqual(sorted(assignment.pi), [0, 1, 2, 3, 4])

    def test_random_with_fixed_seed_is_reproducible(self):
        a1 = resolve_anchor_permutation(self.z_poison, self.z_clean, "random", random_seed=12)
        a2 = resolve_anchor_permutation(self.z_poison, self.z_clean, "random", random_seed=12)
        self.assertEqual(a1.pi, a2.pi)

    def test_random_with_different_seed_differs(self):
        a1 = resolve_anchor_permutation(self.z_poison, self.z_clean, "random", random_seed=12)
        a2 = resolve_anchor_permutation(self.z_poison, self.z_clean, "random", random_seed=99)
        # Not a strict guarantee for arbitrary N, but for N=5 (120 permutations)
        # and these two fixed seeds this is deterministic; assert inequality
        # rather than looping over many seeds to keep the test fast and exact.
        self.assertNotEqual(a1.pi, a2.pi)

    def test_non_bijective_strategy_shape_mismatch_raises(self):
        z_poison_6 = _make_embeddings(9, 6, 8)
        with self.assertRaises(ValueError):
            resolve_anchor_permutation(z_poison_6, self.z_clean, "nearest_bijection")

    def test_no_scipy_dependency(self):
        """The execution plan explicitly forbids a SciPy dependency for the
        brute-force permutation search. Checks for actual `import scipy`
        statements only via the AST (the module's prose docstring
        legitimately contains the substring "SciPy" as a design note, so a
        plain text search would false-positive on that; and `sys.modules`
        can legitimately contain scipy transitively via sklearn/matplotlib
        when the full test suite runs, so that is not checked either --
        only this module's own import statements are)."""
        import ast

        module_path = os.path.join(REPO_ROOT, "defense", "cluster_normalized_poisoning.py")
        with open(module_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])
        self.assertNotIn("scipy", imported_modules)


# --------------------------------------------------------------------------
# Test 7: no GPT/API calls guard + end-to-end smoke test
# --------------------------------------------------------------------------

class FakeSentenceTransformer:
    """Deterministic, dependency-free stand-in for SentenceTransformer, same
    pattern as tests/test_ragdefender_cluster_viz.py."""

    def encode(self, text_list, convert_to_tensor=True):
        vectors = []
        for t in text_list:
            digest = hashlib.md5(t.encode("utf-8")).hexdigest()
            seed = int(digest[:8], 16)
            gen = torch.Generator().manual_seed(seed)
            vectors.append(torch.rand(16, generator=gen))
        return torch.stack(vectors)


def _write_fixtures(tmpdir: str):
    query_id = "test-query-cnp"
    n_poison, n_clean, k = 5, 5, 10
    doc_ids = [f"adv::{query_id}::{i}" for i in range(n_poison)] + [f"clean-{i}" for i in range(n_clean)]
    is_poison = [True] * n_poison + [False] * n_clean
    texts = [f"Poisoned passage number {i} about the target question." for i in range(n_poison)] + \
            [f"Unrelated clean fact number {i} about something else entirely." for i in range(n_clean)]

    diag_record = {
        "query_id": query_id,
        "dataset": "hotpotqa",
        "model": "gpt4",
        "attack": "LM_targeted",
        "defense": "ragdefender_original",
        "k": k,
        "N_injected": n_poison,
        "retrieved_doc_ids": doc_ids,
        "retrieved_is_poison": is_poison,
        "N_retrieved_poison": n_poison,
        "N_retrieved_clean": n_clean,
    }
    diagnostics_path = os.path.join(tmpdir, "diag.jsonl")
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(diag_record) + "\n")

    prompt_no_defense = (
        "You are a helpful assistant... \n\nContexts: " + "\n".join(texts) +
        " \n\nQuery: Some question? \n\nAnswer:"
    )
    query_results_dir = os.path.join(tmpdir, "query_results")
    os.makedirs(query_results_dir, exist_ok=True)
    query_results_payload = [{
        "iter_0": [{
            "id": query_id,
            "question": "Some question?",
            "input_prompt_no_defense": prompt_no_defense,
        }]
    }]
    with open(os.path.join(query_results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(query_results_payload, f)

    return diagnostics_path, query_results_dir, query_id


class TestNoNetworkGuard(unittest.TestCase):
    def test_offline_env_vars_set_on_import(self):
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")

    def test_run_config_records_no_api_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics_path, query_results_dir, query_id = _write_fixtures(tmpdir)
            output_dir = os.path.join(tmpdir, "output")
            with mock.patch.object(run_script.viz, "load_embedder", return_value=FakeSentenceTransformer()):
                run_dir = run_script.main([
                    "--diagnostics_jsonl", diagnostics_path,
                    "--query_results_dir", query_results_dir,
                    "--query_id", query_id,
                    "--intervention", "E0",
                    "--output_dir", output_dir,
                    "--alphas", "1.0", "0.5",
                    "--no_plots",
                ])
            with open(Path(run_dir) / "run_config.json") as f:
                cfg = json.load(f)
            self.assertFalse(cfg["oracle_constraints"]["gpt_or_api_calls_made"])
            self.assertFalse(cfg["oracle_constraints"]["claims_text_realizable_attack"])
            self.assertEqual(cfg["oracle_constraints"]["baseline_files_modified"], [])
            self.assertTrue(cfg["oracle_constraints"]["retrieval_membership_fixed"])
            self.assertTrue(cfg["oracle_constraints"]["generator_text_fixed"])
            self.assertFalse(cfg["ragdefender_package_imported"])


class TestEndToEndSmoke(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.diagnostics_path, self.query_results_dir, self.query_id = _write_fixtures(self.tmpdir)
        self.output_dir = os.path.join(self.tmpdir, "output")
        self.patcher = mock.patch.object(run_script.viz, "load_embedder", return_value=FakeSentenceTransformer())
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_e0_run_writes_all_required_files(self):
        run_dir = Path(run_script.main([
            "--diagnostics_jsonl", self.diagnostics_path,
            "--query_results_dir", self.query_results_dir,
            "--query_id", self.query_id,
            "--intervention", "E0",
            "--output_dir", self.output_dir,
            "--alphas", "1.0", "0.9", "0.5",
            "--no_plots",
        ]))
        self.assertTrue(run_dir.exists())
        self.assertIn(f"_oracle_hotpotqa_k10_N5_E0_{self.query_id}", run_dir.name)

        expected_files = [
            "run_config.json", "manifest.json",
            "original_metrics.csv", "normalized_metrics.csv", "intervention_sweep.csv",
            "stage1_before_after.csv", "stage2_before_after.csv",
            "similarity_matrices/original_M.npy",
            "CLUSTER_NORMALIZED_POISONING_REPORT.md",
        ]
        for rel in expected_files:
            self.assertTrue((run_dir / rel).exists(), f"missing expected output file: {rel}")

        sweep = __import__("pandas").read_csv(run_dir / "intervention_sweep.csv")
        self.assertEqual(len(sweep), 3)  # one row per alpha
        self.assertIn("decision_label", sweep.columns)

        with open(run_dir / "run_config.json") as f:
            cfg = json.load(f)
        self.assertIsNone(cfg["anchor_strategy"])
        self.assertIsNone(cfg["pi"])

    def test_e1_run_records_pi_and_bijection(self):
        run_dir = Path(run_script.main([
            "--diagnostics_jsonl", self.diagnostics_path,
            "--query_results_dir", self.query_results_dir,
            "--query_id", self.query_id,
            "--intervention", "E1",
            "--anchor_strategy", "nearest_bijection",
            "--output_dir", self.output_dir,
            "--alphas", "1.0", "0.5",
            "--no_plots",
        ]))
        self.assertIn("E1-nearest_bijection", run_dir.name)
        with open(run_dir / "run_config.json") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["anchor_strategy"], "nearest_bijection")
        self.assertTrue(cfg["pi_is_bijection"])
        self.assertEqual(sorted(c for _, c in cfg["pi"]), [0, 1, 2, 3, 4])

    def test_e1_requires_anchor_strategy(self):
        with self.assertRaises(SystemExit):
            run_script.main([
                "--diagnostics_jsonl", self.diagnostics_path,
                "--query_results_dir", self.query_results_dir,
                "--query_id", self.query_id,
                "--intervention", "E1",
                "--output_dir", self.output_dir,
                "--no_plots",
            ])


if __name__ == "__main__":
    unittest.main()
