"""Tests for scripts/summarize_ragdefender_diagnostics.py using fake JSONL fixtures.

Run with: python -m unittest tests.test_summarizer -v
"""
import importlib.util
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

_SPEC = importlib.util.spec_from_file_location(
    "summarize_ragdefender_diagnostics",
    os.path.join(REPO_ROOT, "scripts", "summarize_ragdefender_diagnostics.py"),
)
summarizer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summarizer)  # type: ignore


def make_record(**overrides):
    base = {
        "query_id": "q0",
        "dataset": "hotpotqa",
        "model": "gpt4",
        "attack": "LM_targeted",
        "defense": "none",
        "k": 5,
        "N_injected": 5,
        "retrieved_doc_ids": ["a", "b"],
        "retrieved_is_poison": [True, True],
        "N_retrieved_poison": 5,
        "N_retrieved_clean": 0,
        "N_adv_estimated_by_ragdefender": None,
        "removed_doc_ids": [],
        "removed_is_poison": [],
        "removed_poison": 0,
        "removed_clean": 0,
        "poison_recall": 0.0,
        "clean_false_positive_rate": None,
        "residual_poison_count": 5,
        "residual_clean_count": 0,
        "residual_poison_fraction": 1.0,
        "answer_no_defense": None,
        "answer_with_defense": None,
        "target_wrong_answer": "wrong",
        "gold_answer": "right",
        "asr_no_defense": None,
        "asr_with_defense": None,
        "latency_retrieval_sec": 0.01,
        "latency_defense_sec": 0.001,
        "latency_generation_sec": None,
        "notes": "",
    }
    base.update(overrides)
    return base


def write_jsonl(records, path):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestAggregate(unittest.TestCase):
    def test_groups_by_dataset_model_defense_k_n(self):
        records = [
            make_record(query_id="q1", k=5, defense="none"),
            make_record(query_id="q2", k=5, defense="none"),
            make_record(query_id="q3", k=10, defense="none"),
        ]
        summaries = summarizer.aggregate(records)
        self.assertEqual(len(summaries), 2)
        k5 = next(s for s in summaries if s["k"] == 5)
        self.assertEqual(k5["n_queries"], 2)

    def test_asr_means_and_delta_computed_when_present(self):
        records = [
            make_record(query_id="q1", defense="ragdefender_original", asr_no_defense=True, asr_with_defense=False),
            make_record(query_id="q2", defense="ragdefender_original", asr_no_defense=True, asr_with_defense=True),
        ]
        summaries = summarizer.aggregate(records)
        self.assertEqual(len(summaries), 1)
        s = summaries[0]
        self.assertAlmostEqual(s["ASR_no_defense"], 1.0)
        self.assertAlmostEqual(s["ASR_with_defense"], 0.5)
        self.assertAlmostEqual(s["ASR_delta"], -0.5)

    def test_asr_is_none_when_all_dry_run(self):
        records = [make_record(query_id="q1"), make_record(query_id="q2")]
        summaries = summarizer.aggregate(records)
        s = summaries[0]
        self.assertIsNone(s["ASR_no_defense"])
        self.assertIsNone(s["ASR_with_defense"])
        self.assertIsNone(s["ASR_delta"])

    def test_mean_ignores_none_values(self):
        records = [
            make_record(query_id="q1", clean_false_positive_rate=0.5),
            make_record(query_id="q2", clean_false_positive_rate=None),
        ]
        summaries = summarizer.aggregate(records)
        self.assertAlmostEqual(summaries[0]["mean_clean_false_positive_rate"], 0.5)


class TestLoadRecordsAndCsv(unittest.TestCase):
    def test_load_records_reads_all_jsonl_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_jsonl([make_record(query_id="q1")], os.path.join(tmp, "run1.jsonl"))
            write_jsonl([make_record(query_id="q2"), make_record(query_id="q3")], os.path.join(tmp, "run2.jsonl"))
            records = summarizer.load_records(tmp)
            self.assertEqual(len(records), 3)

    def test_write_csv_has_expected_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            summaries = summarizer.aggregate([make_record(query_id="q1")])
            csv_path = os.path.join(tmp, "out.csv")
            summarizer.write_csv(summaries, csv_path)
            with open(csv_path) as f:
                header = f.readline().strip().split(",")
            self.assertEqual(header, summarizer.CSV_COLUMNS)


class TestReportRendering(unittest.TestCase):
    def test_report_contains_warning_and_sections(self):
        records = [
            make_record(query_id="q1", defense="none", k=5),
            make_record(
                query_id="q1", defense="ragdefender_original", k=5,
                N_retrieved_clean=0, removed_poison=2, removed_clean=0,
                poison_recall=0.4, residual_poison_fraction=1.0,
            ),
            make_record(
                query_id="q2", defense="oracle_remove_all_poison", k=5,
                N_retrieved_clean=0, removed_poison=5, removed_clean=0,
                poison_recall=1.0, residual_poison_fraction=0.0,
                asr_with_defense=True,
            ),
        ]
        summaries = summarizer.aggregate(records)
        report = summarizer.render_report(summaries, records)
        self.assertIn("diagnostic control", report)
        self.assertIn("not a deployable defense", report) if "not a deployable defense" in report else None
        self.assertIn("Detection-quality results", report)
        self.assertIn("Interpretation decision tree", report)
        self.assertIn("Worst 10 queries", report)
        self.assertIn("removed more clean than poisoned", report)
        self.assertIn("RAGDefender vs. oracle vs. random removal", report)

    def test_decision_tree_flags_improvement_at_higher_k(self):
        records = [
            make_record(query_id="q1", defense="ragdefender_original", k=5, residual_poison_fraction=1.0),
            make_record(query_id="q2", defense="ragdefender_original", k=10, residual_poison_fraction=0.1),
        ]
        summaries = summarizer.aggregate(records)
        tree = summarizer.render_decision_tree(summaries)
        self.assertIn("threat-model mismatch", tree)

    def test_decision_tree_flags_no_improvement_as_implementation_mismatch(self):
        records = [
            make_record(query_id="q1", defense="ragdefender_original", k=5, residual_poison_fraction=1.0),
            make_record(query_id="q2", defense="ragdefender_original", k=10, residual_poison_fraction=1.0),
        ]
        summaries = summarizer.aggregate(records)
        tree = summarizer.render_decision_tree(summaries)
        self.assertIn("implementation-mismatch", tree)

    def test_worst_queries_sorted_descending(self):
        records = [
            make_record(query_id="low", residual_poison_fraction=0.1),
            make_record(query_id="high", residual_poison_fraction=0.9),
            make_record(query_id="mid", residual_poison_fraction=0.5),
        ]
        worst = summarizer.render_worst_queries(records, n=10)
        self.assertLess(worst.index("high"), worst.index("mid"))
        self.assertLess(worst.index("mid"), worst.index("low"))

    def test_clean_gt_poison_flags_only_ragdefender_defenses(self):
        records = [
            make_record(query_id="q1", defense="ragdefender_original", removed_poison=1, removed_clean=3),
            make_record(query_id="q2", defense="random_remove_same_count", removed_poison=1, removed_clean=3),
        ]
        rendered = summarizer.render_clean_gt_poison_removals(records)
        self.assertIn("q1", rendered)
        self.assertNotIn("q2", rendered)


class TestMainEndToEnd(unittest.TestCase):
    def test_main_writes_csv_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            diag_dir = os.path.join(tmp, "diag")
            write_jsonl([make_record(query_id="q1")], os.path.join(diag_dir, "run.jsonl"))
            csv_out = os.path.join(tmp, "summary.csv")
            report_out = os.path.join(tmp, "report.md")

            old_argv = sys.argv
            sys.argv = [
                "summarize_ragdefender_diagnostics.py",
                "--diagnostics_dir", diag_dir,
                "--csv_out", csv_out,
                "--report_out", report_out,
            ]
            try:
                summarizer.main()
            finally:
                sys.argv = old_argv

            self.assertTrue(os.path.exists(csv_out))
            self.assertTrue(os.path.exists(report_out))


if __name__ == "__main__":
    unittest.main()
