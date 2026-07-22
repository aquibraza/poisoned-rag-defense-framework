"""Backward-compatibility test: proves ASR-style post-hoc analysis still
works on pre-existing result files under results/query_results/main/,
using only eval_asr.py's own logic -- no live LLM/API call, no retrieval,
no defense re-run.

This guards against the main.py changes in this branch (passage metadata,
defense dispatch, diagnostics) silently changing the on-disk schema of
results/query_results/*.json in a way that would break existing
post-hoc tooling (eval_asr.py, scripts/compute_asr_from_results.py).

Run with: python -m unittest tests.test_existing_results_compat -v
"""
import glob
import importlib.util
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

_SPEC = importlib.util.spec_from_file_location("eval_asr", os.path.join(REPO_ROOT, "eval_asr.py"))
eval_asr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_asr)  # type: ignore

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "query_results", "main")


def _existing_result_files():
    return sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))


@unittest.skipUnless(
    _existing_result_files(), f"No existing result files found under {RESULTS_DIR}; skipping compat check."
)
class TestExistingResultsCompat(unittest.TestCase):
    def test_all_existing_result_files_parse_without_error(self):
        for path in _existing_result_files():
            if path.endswith("_formatted.json"):
                continue  # human-readable derivative, not a primary result file
            with self.subTest(path=path):
                records = eval_asr.load_records(path)
                self.assertIsInstance(records, list)

    def test_clean_str_matches_documented_behavior(self):
        self.assertEqual(eval_asr.clean_str("Yes."), "yes")
        self.assertEqual(eval_asr.clean_str("  No  "), "no")
        self.assertEqual(eval_asr.clean_str("A."), "a")

    def test_asr_computable_on_a_ragdefender_result_file(self):
        candidates = [
            p for p in _existing_result_files()
            if "defense-ragdefender" in os.path.basename(p) and not p.endswith("_formatted.json")
        ]
        if not candidates:
            self.skipTest("No existing 'defense-ragdefender' result file found.")
        path = candidates[0]
        row = eval_asr.compute_asr_for_file(path)
        self.assertGreater(row["n_queries"], 0)
        self.assertIsInstance(row["asr_with_defense"], float)
        self.assertGreaterEqual(row["asr_with_defense"], 0.0)
        self.assertLessEqual(row["asr_with_defense"], 1.0)

    def test_no_defense_result_file_has_no_no_defense_asr(self):
        """A file produced with --defense none never has output_poison_no_defense,
        so ASR_no_defense should come back as None (not silently wrong)."""
        candidates = [
            p for p in _existing_result_files()
            if "defense-ragdefender" not in os.path.basename(p)
            and "adv-LM_targeted" in os.path.basename(p)
            and not p.endswith("_formatted.json")
        ]
        if not candidates:
            self.skipTest("No existing no-defense result file found.")
        row = eval_asr.compute_asr_for_file(candidates[0])
        self.assertIsNone(row["asr_no_defense"])


if __name__ == "__main__":
    unittest.main()
