"""Tests for defense/ragdefender_internals.py and
scripts/visualize_ragdefender_clusters.py.

Following tests/test_dispatch_smoke.py's convention: the automated suite
never downloads or loads a real sentence-transformers model. The end-to-end
smoke test monkeypatches `load_embedder` with the same deterministic
`FakeSentenceTransformer` pattern used there, so `util.cos_sim` (a real,
already-installed sentence_transformers function -- no network needed) still
runs on realistic torch tensors.

Run with: python -m unittest tests.test_ragdefender_cluster_viz -v
"""
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
import torch

from defense.ragdefender_internals import concentration_stage1, stage2_pair_frequency

# scripts/ has no __init__.py (matches other scripts/*.py in this repo), so
# import the module by file path rather than as a package.
_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "visualize_ragdefender_clusters.py",
)
_spec = importlib.util.spec_from_file_location("visualize_ragdefender_clusters", _SCRIPT_PATH)
viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(viz)


# --------------------------------------------------------------------------
# defense/ragdefender_internals.py -- pure math
# --------------------------------------------------------------------------

class TestConcentrationStage1(unittest.TestCase):
    def test_non_flip_branch_clusters_flagged(self):
        """3 tightly-clustered passages + 1 isolated one: OR-flag fires for
        the cluster, avg_avg < avg_median holds, so the non-flip branch runs
        and the cluster (not the isolated passage) is the estimated
        adversarial set."""
        matrix = np.array([
            [1.00, 0.90, 0.85, 0.10],
            [0.90, 1.00, 0.88, 0.15],
            [0.85, 0.88, 1.00, 0.05],
            [0.10, 0.15, 0.05, 1.00],
        ])
        result = concentration_stage1(matrix)
        self.assertAlmostEqual(result.avg_avg, 0.61625, places=5)
        self.assertAlmostEqual(result.avg_median, 0.85, places=5)
        self.assertFalse(result.flipped)
        self.assertEqual(result.n_adv_estimated, 3)
        np.testing.assert_array_equal(result.adv_side_flag, [True, True, True, False])

    def test_flip_branch_fires_when_or_count_zero(self):
        """A uniform (equi-similar) 3x3 matrix: no column's mean/median
        exceeds the global thresholds (all ties), so raw_or_flag is all-False
        and the flip branch (len - sum(final)) fires, flagging everything."""
        matrix = np.array([
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ])
        result = concentration_stage1(matrix)
        self.assertEqual(int(result.raw_or_flag.sum()), 0)
        self.assertTrue(result.flipped)
        self.assertEqual(result.n_adv_estimated, 3)
        np.testing.assert_array_equal(result.adv_side_flag, [True, True, True])

    def test_torch_style_median_uses_lower_middle_for_even_n(self):
        """Regression guard: numpy.median would average the two middle
        values for even n; torch.median (what defense_runner.py actually
        calls) returns the lower one. This must not silently drift to the
        numpy convention."""
        # Column 0 sorted: [0.1, 0.85, 0.9, 1.0] -- torch-style median (lower
        # of the two middle values, index (4-1)//2=1) is 0.85, not
        # numpy.median's 0.875.
        matrix = np.array([
            [1.00, 0.90, 0.85, 0.10],
            [0.90, 1.00, 0.88, 0.15],
            [0.85, 0.88, 1.00, 0.05],
            [0.10, 0.15, 0.05, 1.00],
        ])
        result = concentration_stage1(matrix)
        self.assertAlmostEqual(result.median[0], 0.85, places=5)
        self.assertNotAlmostEqual(result.median[0], float(np.median(matrix[:, 0])), places=5)

    def test_diagonal_self_similarity_is_included(self):
        """Per the plan's guardrail: mean/median must include j == i (the
        self-similarity 1.0 term), not exclude it -- this is what
        defense_runner._find_num_adversarial actually computes."""
        matrix = np.array([
            [1.00, 0.90, 0.85, 0.10],
            [0.90, 1.00, 0.88, 0.15],
            [0.85, 0.88, 1.00, 0.05],
            [0.10, 0.15, 0.05, 1.00],
        ])
        result = concentration_stage1(matrix)
        # avg[0] computed by hand over ALL 4 rows (including the diagonal
        # 1.0): (1.00+0.90+0.85+0.10)/4 = 0.7125. Excluding the diagonal
        # would give (0.90+0.85+0.10)/3 = 0.6167 instead -- a different value.
        self.assertAlmostEqual(result.avg[0], 0.7125, places=5)


class TestStage2PairFrequency(unittest.TestCase):
    MATRIX = np.array([
        [1.00, 0.90, 0.85, 0.10],
        [0.90, 1.00, 0.88, 0.15],
        [0.85, 0.88, 1.00, 0.05],
        [0.10, 0.15, 0.05, 1.00],
    ])

    def test_n_pairs_and_top_pair_for_n_adv_2(self):
        result = stage2_pair_frequency(self.MATRIX, n_adv=2)
        self.assertEqual(result.n_pairs, 1)  # max(1, C(2,2)=1)
        self.assertEqual(result.top_pairs, [(0, 1, 0.9)])
        np.testing.assert_allclose(result.frequency_scores, [0.81, 0.81, 0.0, 0.0])
        self.assertEqual(sorted(result.selected_indices), [0, 1])

    def test_tie_breaking_matches_counter_insertion_order(self):
        """n_adv=1 with only one top pair (0,1): both indices tie at the
        same frequency score, so exactly one is selected -- it must be index
        0, the one Counter/dict would have inserted first."""
        result = stage2_pair_frequency(self.MATRIX, n_adv=1)
        self.assertEqual(result.n_pairs, 1)
        self.assertEqual(result.selected_indices, [0])

    def test_n_adv_zero_returns_empty_result(self):
        result = stage2_pair_frequency(self.MATRIX, n_adv=0)
        self.assertEqual(result.n_pairs, 0)
        self.assertEqual(result.top_pairs, [])
        self.assertEqual(result.selected_indices, [])
        np.testing.assert_array_equal(result.frequency_scores, np.zeros(4))

    def test_only_non_self_i_less_than_j_pairs_considered(self):
        """Stage 2 must never include i == j (self-pairs) -- matches
        top_similar_pairs' `for j in range(i + 1, len(texts))`."""
        result = stage2_pair_frequency(self.MATRIX, n_adv=4)
        for i, j, _ in result.top_pairs:
            self.assertLess(i, j)


# --------------------------------------------------------------------------
# recover_pre_defense_texts
# --------------------------------------------------------------------------

class TestRecoverPreDefenseTexts(unittest.TestCase):
    def test_recovers_ordered_texts_and_strips_trailing_space(self):
        prompt = (
            'You are a helpful assistant... \n\nContexts: Text A\nText B\nText C '
            '\n\nQuery: What is X? \n\nAnswer:'
        )
        texts = viz.recover_pre_defense_texts({"input_prompt_no_defense": prompt})
        self.assertEqual(texts, ["Text A", "Text B", "Text C"])

    def test_none_record_returns_none(self):
        self.assertIsNone(viz.recover_pre_defense_texts(None))

    def test_missing_field_returns_none(self):
        self.assertIsNone(viz.recover_pre_defense_texts({}))

    def test_null_field_returns_none(self):
        self.assertIsNone(viz.recover_pre_defense_texts({"input_prompt_no_defense": None}))

    def test_malformed_prompt_returns_none(self):
        self.assertIsNone(viz.recover_pre_defense_texts({"input_prompt_no_defense": "no markers here"}))


# --------------------------------------------------------------------------
# End-to-end smoke test of main() -- fully offline
# --------------------------------------------------------------------------

class FakeSentenceTransformer:
    """Deterministic, dependency-free stand-in for SentenceTransformer, same
    pattern as tests/test_dispatch_smoke.py."""

    def encode(self, text_list, convert_to_tensor=True):
        vectors = []
        for t in text_list:
            digest = hashlib.md5(t.encode("utf-8")).hexdigest()
            seed = int(digest[:8], 16)
            gen = torch.Generator().manual_seed(seed)
            vectors.append(torch.rand(16, generator=gen))
        return torch.stack(vectors)


def _write_fixtures(tmpdir: str):
    query_id = "test-query-1"
    doc_ids = ["adv::test-query-1::0", "adv::test-query-1::1", "clean-1", "clean-2"]
    is_poison = [True, True, False, False]
    texts = [
        "Poisoned passage number zero about the target question.",
        "Poisoned passage number one, a paraphrase of passage zero.",
        "An unrelated clean fact about geography.",
        "Another unrelated clean fact about history.",
    ]

    diag_record = {
        "query_id": query_id,
        "dataset": "hotpotqa",
        "model": "gpt4",
        "attack": "LM_targeted",
        "defense": "ragdefender_original",
        "k": 4,
        "N_injected": 2,
        "retrieved_doc_ids": doc_ids,
        "retrieved_is_poison": is_poison,
        "N_retrieved_poison": 2,
        "N_retrieved_clean": 2,
        "N_adv_estimated_by_ragdefender": 2,
        "removed_doc_ids": [doc_ids[0], doc_ids[1]],
        "removed_is_poison": [True, True],
        "removed_poison": 2,
        "removed_clean": 0,
        "poison_recall": 1.0,
        "clean_false_positive_rate": 0.0,
        "residual_poison_count": 0,
        "residual_clean_count": 2,
        "residual_poison_fraction": 0.0,
        "answer_no_defense": "some answer",
        "answer_with_defense": "some other answer",
        "target_wrong_answer": "wrong",
        "gold_answer": "right",
        "asr_no_defense": False,
        "asr_with_defense": False,
        "asr_no_defense_strict": False,
        "asr_with_defense_strict": False,
        "latency_retrieval_sec": 0.01,
        "latency_defense_sec": 0.02,
        "latency_generation_sec": 0.5,
        "notes": "",
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
            "output_poison_no_defense": "some answer",
        }]
    }]
    with open(os.path.join(query_results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(query_results_payload, f)

    return diagnostics_path, query_results_dir, query_id, doc_ids


class TestVisualizeScriptSmoke(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.diagnostics_path, self.query_results_dir, self.query_id, self.doc_ids = _write_fixtures(self.tmpdir)
        self.output_dir = os.path.join(self.tmpdir, "output")
        self.patcher = mock.patch.object(viz, "load_embedder", return_value=FakeSentenceTransformer())
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_main_runs_offline_and_writes_all_required_files(self):
        run_dir = viz.main([
            "--diagnostics_jsonl", self.diagnostics_path,
            "--query_results_dir", self.query_results_dir,
            "--output_dir", self.output_dir,
            "--no_plots",
        ])
        run_dir = Path(run_dir)
        self.assertTrue(run_dir.exists())
        self.assertTrue(run_dir.name.endswith("_p2"))
        self.assertIn("clusterdiag_hotpotqa_k4_N2_ragdefender-original", run_dir.name)

        expected_files = [
            "run_config.json", "manifest.json",
            "stage1_summary.csv", "stage2_summary.csv", "graph_metrics.csv",
            f"passages/{self.query_id}_passages.csv",
            f"similarity/{self.query_id}_similarity_matrix.csv",
            f"similarity/{self.query_id}_similarity_matrix.npy",
            f"similarity/{self.query_id}_similarity_matrix_reordered.csv",
            "RAGDEFENDER_CLUSTER_VISUALIZATION_REPORT.md",
            "logs/run.log",
        ]
        for rel in expected_files:
            self.assertTrue((run_dir / rel).exists(), f"missing expected output file: {rel}")

        # No plots requested -> no plots/*.png should exist.
        self.assertEqual(list((run_dir / "plots").glob("*.png")), [])

        with open(run_dir / "run_config.json") as f:
            run_config = json.load(f)
        self.assertEqual(run_config["run_type"], "ragdefender_cluster_diagnostics")
        self.assertEqual(run_config["dataset"], "hotpotqa")
        self.assertEqual(run_config["k"], 4)
        self.assertEqual(run_config["N"], 2)
        self.assertFalse(run_config["ragdefender_package_imported"])
        self.assertFalse(run_config["gpt_or_api_calls_made"])
        self.assertFalse(run_config["raw_embeddings_saved"])
        self.assertEqual(run_config["query_ids_processed"], [self.query_id])
        self.assertEqual(run_config["query_ids_skipped"], [])

        with open(run_dir / "manifest.json") as f:
            manifest = json.load(f)
        for group in ("config", "csv", "matrices", "report", "logs"):
            self.assertIn(group, manifest)
            self.assertGreater(len(manifest[group]), 0, f"manifest group {group!r} is empty")
        self.assertEqual(manifest["plots"], [])  # --no_plots -> empty, not fabricated

        sim = np.load(run_dir / "similarity" / f"{self.query_id}_similarity_matrix.npy")
        self.assertEqual(sim.shape, (4, 4))

        import pandas as pd
        passages = pd.read_csv(run_dir / "passages" / f"{self.query_id}_passages.csv")
        self.assertEqual(len(passages), 4)
        self.assertFalse(passages["x_pca"].isna().any())
        self.assertFalse(passages["y_pca"].isna().any())
        self.assertTrue(passages["retrieval_score"].isna().all())  # documented limitation

        stage1 = pd.read_csv(run_dir / "stage1_summary.csv")
        self.assertEqual(len(stage1), 1)
        self.assertIn("agreement_with_diagnostics", stage1.columns)

        stage2 = pd.read_csv(run_dir / "stage2_summary.csv")
        self.assertEqual(len(stage2), 1)
        self.assertIn("agreement_with_diagnostics", stage2.columns)

    def test_skipped_query_is_reported_not_silently_dropped(self):
        """A diagnostics record whose query_results has no
        input_prompt_no_defense (dry-run-style) must be skipped and reported,
        not guessed at or silently ignored."""
        dryrun_diag_path = os.path.join(self.tmpdir, "diag_dryrun.jsonl")
        with open(self.diagnostics_path) as f:
            rec = json.loads(f.readline())
        rec["query_id"] = "dry-run-query"
        with open(dryrun_diag_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        empty_query_results_dir = os.path.join(self.tmpdir, "empty_query_results")
        os.makedirs(empty_query_results_dir, exist_ok=True)
        with open(os.path.join(empty_query_results_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump([{"iter_0": [{"id": "dry-run-query", "question": "q?"}]}], f)

        run_dir = Path(viz.main([
            "--diagnostics_jsonl", dryrun_diag_path,
            "--query_results_dir", empty_query_results_dir,
            "--output_dir", self.output_dir,
            "--no_plots",
        ]))
        with open(run_dir / "run_config.json") as f:
            run_config = json.load(f)
        self.assertEqual(run_config["query_ids_processed"], [])
        self.assertEqual(run_config["query_ids_skipped"], ["dry-run-query"])
        self.assertIn("dry-run-query", run_config["skip_reasons"])


if __name__ == "__main__":
    unittest.main()
