"""Tests for scripts/build_distribution_metrics_batch.py.

Covers:
  - `load_similarity_matrices` / `attach_distribution_metrics` (pure, given
    already-loaded matrices).
  - `compute_extended_config_summary` (baseline/final/reduction/decreased_alpha,
    Pearson correlation).
  - `leading_indicator_categories` (generic leading-indicator-vs-outcome
    classification) and its use in `answer_best_predictor_comparison`'s
    support-fraction-then-lead-gap tiebreak.
  - No forbidden (GPT/API/network) imports, via AST (same convention as
    `tests/test_cluster_normalized_poisoning.py::test_no_scipy_dependency`).
  - An end-to-end smoke test: runs the real oracle script (embedder faked,
    same pattern as `tests/test_build_batch_comparison_success_cases.py`)
    for two synthetic success-case queries across all four E1 strategies,
    then runs this script against that output directory, and checks (a) the
    output files/row counts, and (b) that the alpha=1.0 CORAL/MMD values it
    reports exactly match values computed directly from that same run's
    saved `original_M.npy` -- i.e. alpha=1.0 reproduces the original
    similarity-derived metrics.

Run with: python -m unittest tests.test_build_distribution_metrics_batch -v
"""
import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name: str, rel_path: str):
    path = os.path.join(REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_script = _load_module("run_cluster_normalized_poisoning", os.path.join("scripts", "run_cluster_normalized_poisoning.py"))
batch_script = _load_module("build_batch_comparison_success_cases", os.path.join("scripts", "build_batch_comparison_success_cases.py"))
dist_script = _load_module("build_distribution_metrics_batch", os.path.join("scripts", "build_distribution_metrics_batch.py"))

from defense.distribution_metrics import coral_distance_from_gram, mmd_rbf_distance_from_gram, slice_gram_blocks


# --------------------------------------------------------------------------
# No forbidden imports (GPT/API/network client libraries)
# --------------------------------------------------------------------------

class TestNoForbiddenImports(unittest.TestCase):
    def test_no_gpt_or_network_client_imports(self):
        """This script only reads already-saved local files (diagnostics
        jsonl, intervention_sweep.csv, similarity_matrices/*.npy) -- it
        must never import an LLM/API/network client library, and in
        particular never `sentence_transformers` (no embedder is loaded;
        all cosine matrices are read from disk)."""
        module_path = os.path.join(REPO_ROOT, "scripts", "build_distribution_metrics_batch.py")
        with open(module_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])
        forbidden = {"openai", "requests", "httpx", "urllib", "sentence_transformers"}
        self.assertEqual(imported_modules & forbidden, set())


# --------------------------------------------------------------------------
# load_similarity_matrices / attach_distribution_metrics
# --------------------------------------------------------------------------

class TestLoadSimilarityMatrices(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = Path(self.tmpdir) / "run"
        (self.run_dir / "similarity_matrices").mkdir(parents=True)

    def _save(self, name, m):
        np.save(self.run_dir / "similarity_matrices" / name, m)

    def test_prefers_transformed_alpha_1_0_over_original(self):
        original = np.eye(4)
        transformed_1_0 = np.eye(4) * 2  # deliberately different, to prove which one wins
        self._save("original_M.npy", original)
        self._save("transformed_M_alpha1.0.npy", transformed_1_0)
        self._save("transformed_M_alpha0.5.npy", np.eye(4) * 3)
        matrices = dist_script.load_similarity_matrices(self.run_dir)
        np.testing.assert_array_equal(matrices[1.0], transformed_1_0)
        np.testing.assert_array_equal(matrices[0.5], np.eye(4) * 3)

    def test_falls_back_to_original_when_no_transformed_alpha_1_0(self):
        original = np.eye(3)
        self._save("original_M.npy", original)
        self._save("transformed_M_alpha0.3.npy", np.eye(3) * 5)
        matrices = dist_script.load_similarity_matrices(self.run_dir)
        np.testing.assert_array_equal(matrices[1.0], original)

    def test_lookup_matrix_tolerant_lookup(self):
        matrices = {1.0: np.eye(2), 0.3: np.eye(2) * 9}
        m = dist_script._lookup_matrix(matrices, 0.30000000001)
        np.testing.assert_array_equal(m, np.eye(2) * 9)

    def test_lookup_matrix_raises_for_missing_alpha(self):
        matrices = {1.0: np.eye(2)}
        with self.assertRaises(FileNotFoundError):
            dist_script._lookup_matrix(matrices, 0.5)


class TestAttachDistributionMetrics(unittest.TestCase):
    def test_attached_columns_match_direct_computation(self):
        rng = np.random.default_rng(0)

        def unit(n, d):
            z = rng.normal(size=(n, d))
            return z / np.linalg.norm(z, axis=1, keepdims=True)

        poison_idx, clean_idx = [0, 1, 2], [3, 4, 5]
        alphas = [1.0, 0.5]
        matrices = {}
        for a in alphas:
            z = unit(6, 5)
            matrices[a] = z @ z.T
        df = pd.DataFrame({"alpha": alphas})
        out = dist_script.attach_distribution_metrics(df, matrices, poison_idx, clean_idx)
        for a in alphas:
            g_pp, g_pc, g_cc = slice_gram_blocks(matrices[a], poison_idx, clean_idx)
            expected_coral = coral_distance_from_gram(g_pp, g_pc, g_cc)
            expected_mmd = mmd_rbf_distance_from_gram(g_pp, g_pc, g_cc)
            row = out[out["alpha"] == a].iloc[0]
            self.assertAlmostEqual(row["coral_distance"], expected_coral, places=10)
            self.assertAlmostEqual(row["mmd_distance"], expected_mmd, places=10)

    def test_does_not_mutate_input_df(self):
        z = np.eye(4)
        matrices = {1.0: z}
        df = pd.DataFrame({"alpha": [1.0]})
        dist_script.attach_distribution_metrics(df, matrices, [0, 1], [2, 3])
        self.assertNotIn("coral_distance", df.columns)


# --------------------------------------------------------------------------
# compute_extended_config_summary
# --------------------------------------------------------------------------

def _sweep_df(alphas, coral_values, mmd_values, mean_pp_values, removed_poison, n_retrieved_poison=5):
    rows = []
    for a, c, m, pp in zip(alphas, coral_values, mmd_values, mean_pp_values):
        rows.append({
            "alpha": a, "coral_distance": c, "mmd_distance": m,
            "mean_poison_poison_similarity": pp,
            "removed_poison": removed_poison.get(a, n_retrieved_poison),
            "removed_clean": 0, "N_retrieved_poison": n_retrieved_poison,
            "top_pair_pp": 10, "top_pair_pc": 0, "decision_label": "poison_removal_success",
        })
    return pd.DataFrame(rows)


class TestComputeExtendedConfigSummary(unittest.TestCase):
    def test_baseline_final_reduction_and_decreased_alpha(self):
        alphas = [1.0, 0.9, 0.8, 0.7]
        coral = [1.0, 0.9, 0.5, 0.2]
        mmd = [2.0, 2.0, 1.0, 0.4]  # ties at alpha=0.9, decreases at 0.8
        mean_pp = [0.9, 0.9, 0.9, 0.9]  # never decreases
        df = _sweep_df(alphas, coral, mmd, mean_pp, removed_poison={0.7: 3}, n_retrieved_poison=5)
        summary = dist_script.compute_extended_config_summary("q1", "rank_aligned", df)

        self.assertAlmostEqual(summary["baseline_coral_distance"], 1.0)
        self.assertAlmostEqual(summary["final_coral_distance"], 0.2)
        self.assertAlmostEqual(summary["coral_reduction"], 0.8)
        self.assertAlmostEqual(summary["coral_decreased_alpha"], 0.9)  # first strictly below 1.0

        self.assertAlmostEqual(summary["baseline_mmd_distance"], 2.0)
        self.assertAlmostEqual(summary["mmd_decreased_alpha"], 0.8)  # ties at 0.9, first drop at 0.8

        self.assertIsNone(summary["mean_pp_decreased_alpha"])  # never decreases
        self.assertAlmostEqual(summary["first_residual_poison_alpha"], 0.7)

    def test_pearson_correlation_positive_when_distance_falls_with_alpha(self):
        alphas = [1.0, 0.9, 0.8, 0.7, 0.6]
        coral = [1.0, 0.8, 0.6, 0.4, 0.2]  # perfectly monotonic with alpha
        mmd = [1.0, 0.8, 0.6, 0.4, 0.2]
        mean_pp = [0.9] * 5
        df = _sweep_df(alphas, coral, mmd, mean_pp, removed_poison={})
        summary = dist_script.compute_extended_config_summary("q1", "random", df)
        self.assertAlmostEqual(summary["alpha_coral_pearson_r"], 1.0, places=6)
        self.assertAlmostEqual(summary["alpha_mmd_pearson_r"], 1.0, places=6)


class TestPearsonCorr(unittest.TestCase):
    def test_none_for_constant_series(self):
        self.assertIsNone(dist_script.pearson_corr([1, 1, 1], [1, 2, 3]))

    def test_none_for_too_few_points(self):
        self.assertIsNone(dist_script.pearson_corr([1.0], [2.0]))

    def test_negative_one_for_perfectly_inverse_series(self):
        r = dist_script.pearson_corr([1, 2, 3, 4], [4, 3, 2, 1])
        self.assertAlmostEqual(r, -1.0, places=6)


# --------------------------------------------------------------------------
# leading_indicator_categories / answer_best_predictor_comparison
# --------------------------------------------------------------------------

def _summary_row(query_id, strategy, lead_alpha, outcome_alpha, mean_lead_col="lead_col"):
    return {"query_id": query_id, "strategy": strategy, mean_lead_col: lead_alpha, "outcome_col": outcome_alpha}


class TestLeadingIndicatorCategories(unittest.TestCase):
    def test_categorizes_and_computes_mean_lead_gap(self):
        rows = [
            _summary_row("q1", "s1", 0.7, 0.6),   # precedes, gap=0.1
            _summary_row("q2", "s1", 0.6, 0.6),   # coincides, gap=0.0
            _summary_row("q3", "s1", 0.5, 0.7),   # outcome first -> against
            _summary_row("q4", "s1", 0.6, None),  # leading triggered, no outcome
            _summary_row("q5", "s1", None, 0.6),  # outcome occurs, leading never triggers
            _summary_row("q6", "s1", None, None),  # neither
        ]
        df = pd.DataFrame(rows)
        result = dist_script.leading_indicator_categories(df, "lead_col", "outcome_col")
        self.assertEqual(result["categories"]["leading_precedes_or_coincides_with_outcome"], 2)
        self.assertEqual(result["categories"]["leading_triggered_without_outcome"], 1)
        self.assertEqual(result["categories"]["outcome_without_leading_triggered_first"], 2)
        self.assertEqual(result["categories"]["neither_triggered"], 1)
        self.assertAlmostEqual(result["support_fraction"], 2 / 4)
        self.assertAlmostEqual(result["mean_lead_gap"], (0.1 + 0.0) / 2)

    def test_none_support_fraction_when_no_informative_configs(self):
        df = pd.DataFrame([_summary_row("q1", "s1", None, None)])
        result = dist_script.leading_indicator_categories(df, "lead_col", "outcome_col")
        self.assertIsNone(result["support_fraction"])
        self.assertIsNone(result["mean_lead_gap"])


class TestAnswerBestPredictorComparison(unittest.TestCase):
    def test_ties_on_support_broken_by_smaller_mean_lead_gap(self):
        """Construct a case where all four candidates tie at 100% support,
        but `top_pair_pp` fires much closer to the failure alpha (small
        lead gap) than the others (large lead gap) -- the tiebreaker must
        prefer the tighter/more specific indicator."""
        rows = []
        for qid, failure_alpha in [("q1", 0.6), ("q2", 0.5)]:
            rows.append({
                "query_id": qid, "strategy": "rank_aligned",
                "pp_decreased_alpha": failure_alpha,  # coincides exactly -> gap 0
                "coral_decreased_alpha": 0.9,          # always fires immediately -> large gap
                "mmd_decreased_alpha": 0.9,
                "mean_pp_decreased_alpha": 0.9,
                "first_residual_poison_alpha": failure_alpha,
            })
        df = pd.DataFrame(rows)
        result = dist_script.answer_best_predictor_comparison(df)
        self.assertEqual(result["best"], "top_pair_pp")
        for label in ["coral_distance", "mmd_distance", "mean_poison_poison_similarity"]:
            self.assertAlmostEqual(result["results"][label]["support_fraction"], 1.0)
        self.assertAlmostEqual(result["results"]["top_pair_pp"]["mean_lead_gap"], 0.0)
        self.assertGreater(result["results"]["coral_distance"]["mean_lead_gap"], 0.0)


# --------------------------------------------------------------------------
# End-to-end smoke test: real oracle runs (faked embedder) + this script
# --------------------------------------------------------------------------

class FakeSentenceTransformer:
    """Deterministic, dependency-free stand-in for SentenceTransformer, same
    pattern as tests/test_cluster_normalized_poisoning.py."""

    def encode(self, text_list, convert_to_tensor=True):
        vectors = []
        for t in text_list:
            digest = hashlib.md5(t.encode("utf-8")).hexdigest()
            seed = int(digest[:8], 16)
            gen = torch.Generator().manual_seed(seed)
            vectors.append(torch.rand(16, generator=gen))
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


class TestEndToEndDistributionMetricsBatch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.tmpdir, "output")
        self.patcher = mock.patch.object(run_script.viz, "load_embedder", return_value=FakeSentenceTransformer())
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

        self.tested_ids = ["qid_tested_1", "qid_tested_2"]
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

        for qid in self.tested_ids:
            for strategy in batch_script.E1_STRATEGIES:
                run_script.main([
                    "--diagnostics_jsonl", self.diagnostics_path,
                    "--query_results_dir", self.query_results_dir,
                    "--query_id", qid,
                    "--intervention", "E1", "--anchor_strategy", strategy, "--random_seed", "12",
                    "--output_dir", self.output_dir,
                    "--alphas", "1.0", "0.9", "0.5", "0.3",
                    "--no_plots",
                ])

    def test_output_files_and_row_counts(self):
        md_path = dist_script.main([
            "--diagnostics_jsonl", self.diagnostics_path,
            "--query_results_dir", self.query_results_dir,
            "--output_dir", self.output_dir,
        ])
        self.assertTrue(md_path.exists())
        csv_path = md_path.parent / "DISTRIBUTION_METRICS_BATCH.csv"
        self.assertTrue(csv_path.exists())

        combined = pd.read_csv(csv_path)
        self.assertEqual(set(combined["query_id"].unique()), set(self.tested_ids))
        self.assertEqual(len(combined), 2 * 4 * 4)  # 2 queries x 4 strategies x 4 alphas
        self.assertIn("coral_distance", combined.columns)
        self.assertIn("mmd_distance", combined.columns)
        self.assertIn("first_residual_poison_alpha", combined.columns)
        self.assertTrue(np.all(np.isfinite(combined["coral_distance"])))
        self.assertTrue(np.all(np.isfinite(combined["mmd_distance"])))
        self.assertTrue(np.all(combined["coral_distance"] >= 0))
        self.assertTrue(np.all(combined["mmd_distance"] >= 0))

        report_text = md_path.read_text(encoding="utf-8")
        for heading in ["Q_A.", "Q_B.", "Q_C.", "Q_D.", "Q_E.", "Q_F.", "## Limitations"]:
            self.assertIn(heading, report_text)

    def test_alpha_1_0_reproduces_metrics_from_original_matrix(self):
        """alpha=1.0's coral_distance/mmd_distance in the output CSV must
        exactly match values computed directly from that same run's saved
        alpha=1.0 similarity matrix (via the same `load_similarity_matrices`
        helper the script itself uses) -- i.e. the join pipeline introduces
        no discrepancy at the identity alpha."""
        qid = self.tested_ids[0]
        run_dirs = dist_script.summ.discover_run_dirs(self.output_dir, qid)
        latest = dist_script.summ.latest_run_per_intervention(run_dirs)
        run_dir = latest["E1-rank_aligned"]

        matrices = dist_script.load_similarity_matrices(run_dir)
        original_m = matrices[1.0]
        records_by_id = {r["query_id"]: r for r in
                          dist_script.viz._read_jsonl(self.diagnostics_path)}
        is_poison = [bool(x) for x in records_by_id[qid]["retrieved_is_poison"]]
        poison_idx = [i for i, p in enumerate(is_poison) if p]
        clean_idx = [i for i, p in enumerate(is_poison) if not p]
        g_pp, g_pc, g_cc = slice_gram_blocks(original_m, poison_idx, clean_idx)
        expected_coral = coral_distance_from_gram(g_pp, g_pc, g_cc)
        expected_mmd = mmd_rbf_distance_from_gram(g_pp, g_pc, g_cc)

        md_path = dist_script.main([
            "--diagnostics_jsonl", self.diagnostics_path,
            "--query_results_dir", self.query_results_dir,
            "--output_dir", self.output_dir,
        ])
        combined = pd.read_csv(md_path.parent / "DISTRIBUTION_METRICS_BATCH.csv")
        row = combined[(combined["query_id"] == qid) & (combined["anchor_strategy"] == "rank_aligned")
                        & (combined["alpha"] == 1.0)].iloc[0]
        self.assertAlmostEqual(row["coral_distance"], expected_coral, places=8)
        self.assertAlmostEqual(row["mmd_distance"], expected_mmd, places=8)


if __name__ == "__main__":
    unittest.main()
