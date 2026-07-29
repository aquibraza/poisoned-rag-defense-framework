"""Tests for scripts/build_batch_comparison_success_cases.py.

Covers:
  - `discover_success_case_ids` (pure filter on diagnostics records).
  - `check_text_recoverable` (reuses `visualize_ragdefender_clusters`'s
    no-guessing recovery gate).
  - The per-(query,strategy) derived-summary function and each of the
    seven questions' pure aggregation functions, using small synthetic
    DataFrames so the expected answer is known by construction.
  - An end-to-end smoke test that runs the real oracle script
    (`run_cluster_normalized_poisoning.py`, with the embedder faked exactly
    as in `tests/test_cluster_normalized_poisoning.py`) for two synthetic
    success-case queries across all four E1 strategies, then runs this
    batch script against that output directory and a synthetic
    diagnostics `.jsonl` that also includes one excluded-by-recovery and
    one excluded-by-missing-run-dirs candidate, and checks the resulting
    counts and files.

Run with: python -m unittest tests.test_build_batch_comparison_success_cases -v
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

import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# scripts/ has no __init__.py -- import by file path, same pattern as
# tests/test_cluster_normalized_poisoning.py.
def _load_module(name: str, rel_path: str):
    path = os.path.join(REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_script = _load_module("run_cluster_normalized_poisoning", os.path.join("scripts", "run_cluster_normalized_poisoning.py"))
batch_script = _load_module("build_batch_comparison_success_cases", os.path.join("scripts", "build_batch_comparison_success_cases.py"))


# --------------------------------------------------------------------------
# discover_success_case_ids
# --------------------------------------------------------------------------

class TestDiscoverSuccessCaseIds(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"query_id": "q_success_1", "dataset": "hotpotqa", "k": 10, "N_injected": 5,
             "N_retrieved_poison": 5, "removed_poison": 5, "residual_poison_fraction": 0.0},
            {"query_id": "q_success_2", "dataset": "hotpotqa", "k": 10, "N_injected": 5,
             "N_retrieved_poison": 5, "removed_poison": 5, "residual_poison_fraction": 0.0},
            {"query_id": "q_failed_control", "dataset": "hotpotqa", "k": 10, "N_injected": 5,
             "N_retrieved_poison": 5, "removed_poison": 0, "residual_poison_fraction": 5 / 7},
            {"query_id": "q_partial_removal", "dataset": "hotpotqa", "k": 10, "N_injected": 5,
             "N_retrieved_poison": 5, "removed_poison": 4, "residual_poison_fraction": 1 / 6},
            {"query_id": "q_wrong_k", "dataset": "hotpotqa", "k": 5, "N_injected": 5,
             "N_retrieved_poison": 5, "removed_poison": 5, "residual_poison_fraction": 0.0},
            {"query_id": "q_wrong_dataset", "dataset": "nq", "k": 10, "N_injected": 5,
             "N_retrieved_poison": 5, "removed_poison": 5, "residual_poison_fraction": 0.0},
        ]

    def test_only_full_success_within_dataset_k_n_injected_survives(self):
        result = batch_script.discover_success_case_ids(self.records, "hotpotqa", 10, 5, [])
        self.assertEqual(result, ["q_success_1", "q_success_2"])

    def test_explicit_exclusion_list_removes_matching_ids(self):
        records = self.records + [
            {"query_id": "q_success_3", "dataset": "hotpotqa", "k": 10, "N_injected": 5,
             "N_retrieved_poison": 5, "removed_poison": 5, "residual_poison_fraction": 0.0},
        ]
        result = batch_script.discover_success_case_ids(records, "hotpotqa", 10, 5, ["q_success_1"])
        self.assertEqual(result, ["q_success_2", "q_success_3"])

    def test_zero_retrieved_poison_never_counts_as_success(self):
        records = [{"query_id": "q_zero_poison", "dataset": "hotpotqa", "k": 10, "N_injected": 0,
                    "N_retrieved_poison": 0, "removed_poison": 0, "residual_poison_fraction": 0.0}]
        result = batch_script.discover_success_case_ids(records, "hotpotqa", 10, 0, [])
        self.assertEqual(result, [])


# --------------------------------------------------------------------------
# check_text_recoverable
# --------------------------------------------------------------------------

class TestCheckTextRecoverable(unittest.TestCase):
    def test_matching_line_count_is_recoverable(self):
        texts = [f"passage {i}" for i in range(10)]
        prompt = "prefix \n\nContexts: " + "\n".join(texts) + " \n\nQuery: q \n\nAnswer:"
        qr_index = {"qid": {"input_prompt_no_defense": prompt}}
        rec = {"query_id": "qid", "retrieved_doc_ids": list(range(10))}
        ok, recovered_len, expected_k = batch_script.check_text_recoverable(qr_index, rec)
        self.assertTrue(ok)
        self.assertEqual(recovered_len, 10)
        self.assertEqual(expected_k, 10)

    def test_embedded_newline_causes_mismatch(self):
        texts = [f"passage {i}" for i in range(9)] + ["passage 9a\npassage 9b (embedded newline)"]
        prompt = "prefix \n\nContexts: " + "\n".join(texts) + " \n\nQuery: q \n\nAnswer:"
        qr_index = {"qid": {"input_prompt_no_defense": prompt}}
        rec = {"query_id": "qid", "retrieved_doc_ids": list(range(10))}
        ok, recovered_len, expected_k = batch_script.check_text_recoverable(qr_index, rec)
        self.assertFalse(ok)
        self.assertEqual(recovered_len, 11)
        self.assertEqual(expected_k, 10)

    def test_missing_query_result_is_not_recoverable(self):
        ok, recovered_len, expected_k = batch_script.check_text_recoverable(
            {}, {"query_id": "missing", "retrieved_doc_ids": list(range(10))}
        )
        self.assertFalse(ok)
        self.assertIsNone(recovered_len)


# --------------------------------------------------------------------------
# compute_config_summary + the seven questions' pure aggregation functions
# --------------------------------------------------------------------------

def _sweep_row(alpha, top_pair_pp, top_pair_pc, removed_poison, removed_clean, decision_label,
               n_retrieved_poison=5):
    return {
        "alpha": alpha, "top_pair_pp": top_pair_pp, "top_pair_pc": top_pair_pc,
        "removed_poison": removed_poison, "removed_clean": removed_clean,
        "decision_label": decision_label, "N_retrieved_poison": n_retrieved_poison,
    }


class TestComputeConfigSummary(unittest.TestCase):
    def test_first_trigger_alphas_detected_in_descending_order(self):
        rows = [
            _sweep_row(1.0, 10, 4, 5, 0, "poison_removal_success"),
            _sweep_row(0.9, 10, 4, 5, 0, "poison_removal_success"),
            _sweep_row(0.8, 9, 4, 5, 0, "poison_removal_success"),   # pp first decreases here
            _sweep_row(0.7, 9, 5, 5, 0, "poison_removal_success"),  # pc first increases here
            _sweep_row(0.6, 9, 5, 4, 0, "residual_poison_failure"),  # removal drop + residual-poison failure here
            _sweep_row(0.5, 9, 5, 4, 1, "residual_poison_with_clean_false_positive"),  # clean removal increase here
        ]
        df = pd.DataFrame(rows)
        summary = batch_script.compute_config_summary("qid", "rank_aligned", df)
        self.assertEqual(summary["baseline_decision_label"], "poison_removal_success")
        self.assertEqual(summary["pp_decreased_alpha"], 0.8)
        self.assertEqual(summary["pc_increased_alpha"], 0.7)
        self.assertEqual(summary["fewer_poison_removed_alpha"], 0.6)
        self.assertEqual(summary["first_residual_poison_alpha"], 0.6)
        self.assertEqual(summary["first_label_change_alpha"], 0.6)
        self.assertEqual(summary["clean_removed_increased_alpha"], 0.5)
        self.assertEqual(summary["final_decision_label"], "residual_poison_with_clean_false_positive")

    def test_label_change_and_residual_poison_alphas_diverge_for_over_removal_improvement(self):
        """Regression guard for the exact bug this revision fixes: a baseline
        that over-removes (removes a clean passage as a false positive at
        alpha=1.0) and later stops doing so while still removing all poison
        is a clean-FP *improvement*, not a defense failure.
        `first_residual_poison_alpha` must stay None in that case even
        though `first_label_change_alpha` fires."""
        rows = [
            _sweep_row(1.0, 10, 4, 5, 1, "over_removal_success"),
            _sweep_row(0.9, 10, 4, 5, 1, "over_removal_success"),
            _sweep_row(0.8, 9, 4, 5, 0, "poison_removal_success"),  # label changes, but improves
            _sweep_row(0.7, 9, 4, 5, 0, "poison_removal_success"),
        ]
        df = pd.DataFrame(rows)
        summary = batch_script.compute_config_summary("qid", "rank_aligned", df)
        self.assertEqual(summary["baseline_decision_label"], "over_removal_success")
        self.assertEqual(summary["first_label_change_alpha"], 0.8)
        self.assertIsNone(summary["first_residual_poison_alpha"])  # poison never survives -> no real failure

    def test_never_triggered_conditions_are_none(self):
        rows = [_sweep_row(a, 10, 4, 5, 0, "poison_removal_success") for a in
                [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]
        df = pd.DataFrame(rows)
        summary = batch_script.compute_config_summary("qid", "random", df)
        for key in ("pp_decreased_alpha", "pc_increased_alpha", "fewer_poison_removed_alpha",
                    "first_residual_poison_alpha", "first_label_change_alpha", "clean_removed_increased_alpha"):
            self.assertIsNone(summary[key])


def _summary_row(query_id, strategy, pp_alpha, first_residual_poison_alpha):
    return {
        "query_id": query_id, "strategy": strategy,
        "pp_decreased_alpha": pp_alpha, "first_residual_poison_alpha": first_residual_poison_alpha,
    }


class TestAnyAllCounts(unittest.TestCase):
    def setUp(self):
        # q1: all four strategies trigger by alpha<=0.6; q2: none do; q3: exactly two do.
        rows = []
        for s in batch_script.E1_STRATEGIES:
            rows.append(_summary_row("q1", s, 0.6, None))
        for s in batch_script.E1_STRATEGIES:
            rows.append(_summary_row("q2", s, 0.9, None))
        for i, s in enumerate(batch_script.E1_STRATEGIES):
            rows.append(_summary_row("q3", s, 0.6 if i < 2 else 0.9, None))
        self.df = pd.DataFrame(rows)

    def test_any_and_all_counts(self):
        any_count, all_count, per_query = batch_script._any_all_counts(
            self.df, ["q1", "q2", "q3"], "pp_decreased_alpha", 0.6
        )
        self.assertEqual(any_count, 2)  # q1 (4/4) and q3 (2/4)
        self.assertEqual(all_count, 1)  # only q1
        self.assertEqual(per_query, {"q1": 4, "q2": 0, "q3": 2})


class TestAnswerQ5StrategyWins(unittest.TestCase):
    def test_earliest_break_wins_ties_shared_never_broke_tracked(self):
        rows = [
            _summary_row("q1", "rank_aligned", None, 0.7),
            _summary_row("q1", "nearest_bijection", None, 0.5),
            _summary_row("q1", "farthest_bijection", None, 0.7),  # tied with rank_aligned
            _summary_row("q1", "random", None, 0.3),
            _summary_row("q2", "rank_aligned", None, None),
            _summary_row("q2", "nearest_bijection", None, None),
            _summary_row("q2", "farthest_bijection", None, None),
            _summary_row("q2", "random", None, None),
        ]
        df = pd.DataFrame(rows)
        result = batch_script.answer_q5(df, ["q1", "q2"])
        self.assertEqual(result["wins_by_strategy"]["rank_aligned"], 1)
        self.assertEqual(result["wins_by_strategy"]["farthest_bijection"], 1)
        self.assertEqual(result["wins_by_strategy"]["nearest_bijection"], 0)
        self.assertEqual(result["wins_by_strategy"]["random"], 0)
        self.assertEqual(result["never_broke_count"], 1)
        self.assertEqual(sorted(result["per_query_winner"]["q1"]), ["farthest_bijection", "rank_aligned"])
        self.assertEqual(result["per_query_winner"]["q2"], [])


class TestAnswerQ6PredictiveCategories(unittest.TestCase):
    def test_categorizes_each_config_correctly(self):
        rows = [
            # pp decreases at higher alpha than residual-poison failure -> precedes
            _summary_row("q1", "s1", 0.7, 0.6),
            _summary_row("q2", "s1", 0.6, 0.6),   # simultaneous -> precedes/coincides
            _summary_row("q3", "s1", 0.5, 0.7),   # residual-poison failure before pp decreases -> against
            _summary_row("q4", "s1", 0.6, None),  # pp decreases, residual-poison failure never occurs
            _summary_row("q5", "s1", None, 0.6),  # residual-poison failure occurs, pp never decreases
            _summary_row("q6", "s1", None, None),  # neither
        ]
        df = pd.DataFrame(rows)
        result = batch_script.answer_q6(df)
        self.assertEqual(result["categories"]["pp_precedes_or_coincides_with_residual_poison_failure"], 2)
        self.assertEqual(result["categories"]["pp_decreased_without_residual_poison_failure"], 1)
        self.assertEqual(result["categories"]["residual_poison_failure_without_pp_decrease_first"], 2)
        self.assertEqual(result["categories"]["neither_triggered"], 1)
        # 4 informative configs (q1,q2,q3,q5); 2 support "precedes".
        self.assertAlmostEqual(result["support_fraction_of_informative_configs"], 2 / 4)


class TestAnswerQ7Distribution(unittest.TestCase):
    def test_distribution_and_extremes(self):
        rows = []
        for s in batch_script.E1_STRATEGIES:
            rows.append(_summary_row("q_all_fail", s, None, 0.4))
        for s in batch_script.E1_STRATEGIES:
            rows.append(_summary_row("q_none_fail", s, None, None))
        df = pd.DataFrame(rows)
        result = batch_script.answer_q7(df, ["q_all_fail", "q_none_fail"])
        self.assertEqual(
            result["per_query_n_strategies_with_residual_poison_failure_by_0_5"],
            {"q_all_fail": 4, "q_none_fail": 0},
        )
        self.assertEqual(result["queries_with_zero_strategies_with_residual_poison_failure"], 1)
        self.assertEqual(result["queries_with_all_four_strategies_with_residual_poison_failure"], 1)
        self.assertAlmostEqual(result["mean_strategies_with_residual_poison_failure"], 2.0)


# --------------------------------------------------------------------------
# End-to-end smoke test: real oracle runs (faked embedder) + batch script
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


def _make_qr_entry(query_id, n_poison=5, n_clean=5, extra_newline_in_last_passage=False):
    texts = [f"Poisoned passage number {i} about the target question for {query_id}." for i in range(n_poison)] + \
            [f"Unrelated clean fact number {i} about something else entirely for {query_id}." for i in range(n_clean)]
    if extra_newline_in_last_passage:
        texts[-1] = texts[-1] + "\nSPURIOUS EMBEDDED SECOND LINE"
    prompt_no_defense = (
        "You are a helpful assistant... \n\nContexts: " + "\n".join(texts) +
        " \n\nQuery: Some question? \n\nAnswer:"
    )
    return {"id": query_id, "question": "Some question?", "input_prompt_no_defense": prompt_no_defense}


class TestEndToEndBatch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.tmpdir, "output")
        self.patcher = mock.patch.object(run_script.viz, "load_embedder", return_value=FakeSentenceTransformer())
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

        self.tested_ids = ["qid_tested_1", "qid_tested_2"]
        self.unrecoverable_id = "qid_unrecoverable"
        self.incomplete_id = "qid_incomplete_runs"

        records = (
            [_make_diag_record(q) for q in self.tested_ids]
            + [_make_diag_record(self.unrecoverable_id)]
            + [_make_diag_record(self.incomplete_id)]
        )
        self.diagnostics_path = os.path.join(self.tmpdir, "diag.jsonl")
        with open(self.diagnostics_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        self.query_results_dir = os.path.join(self.tmpdir, "query_results")
        os.makedirs(self.query_results_dir, exist_ok=True)
        qr_entries = (
            [_make_qr_entry(q) for q in self.tested_ids]
            + [_make_qr_entry(self.unrecoverable_id, extra_newline_in_last_passage=True)]
            + [_make_qr_entry(self.incomplete_id)]
        )
        with open(os.path.join(self.query_results_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump([{"iter_0": qr_entries}], f)

        # Fully run all four E1 strategies for the two "tested" queries.
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
        # Deliberately run only 3/4 strategies for the "incomplete" query.
        for strategy in ["rank_aligned", "nearest_bijection", "farthest_bijection"]:
            run_script.main([
                "--diagnostics_jsonl", self.diagnostics_path,
                "--query_results_dir", self.query_results_dir,
                "--query_id", self.incomplete_id,
                "--intervention", "E1", "--anchor_strategy", strategy, "--random_seed", "12",
                "--output_dir", self.output_dir,
                "--alphas", "1.0", "0.9", "0.5", "0.3",
                "--no_plots",
            ])
        # No runs at all for the unrecoverable query (it would raise anyway).

    def test_batch_report_identifies_tests_and_excludes_correctly(self):
        md_path = batch_script.main([
            "--diagnostics_jsonl", self.diagnostics_path,
            "--query_results_dir", self.query_results_dir,
            "--output_dir", self.output_dir,
        ])
        self.assertTrue(md_path.exists())
        csv_path = md_path.parent / "BATCH_COMPARISON_SUCCESS_CASES.csv"
        self.assertTrue(csv_path.exists())

        report_text = md_path.read_text(encoding="utf-8")
        self.assertIn("**4** cases matched the success criterion", report_text)
        self.assertIn("**2** cases were actually tested", report_text)
        self.assertIn(self.unrecoverable_id, report_text)
        self.assertIn("text recovery mismatch", report_text)
        self.assertIn(self.incomplete_id, report_text)
        self.assertIn("missing run directories for: E1-random", report_text)
        for qid in self.tested_ids:
            self.assertIn(qid, report_text)

        combined = pd.read_csv(csv_path)
        self.assertEqual(set(combined["query_id"].unique()), set(self.tested_ids))
        self.assertEqual(len(combined), 2 * 4 * 4)  # 2 queries x 4 strategies x 4 alphas


if __name__ == "__main__":
    unittest.main()
