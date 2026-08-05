"""CLI wiring tests for --ml_filterrag_model_path / --ml_filterrag_threshold /
--ml_filterrag_lm_model / --ml_filterrag_matching_mode /
--ml_filterrag_semantic_threshold in main.py (added alongside
defense/ml_filterrag.py -- see docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md).

Requires main.py's full dependency set to *import* (torch, transformers,
sentence-transformers, beir, etc. -- same requirement as
test_dispatch_smoke.py/test_main_cli_filterrag.py; see tests/README.md).
Does **not** run main()'s actual retrieval/generation pipeline: no
dataset/BEIR access, no GPU required, and no LLM/GPT/API call of any kind.
Only exercises `argparse` via `main.parse_args()`, plus a source-level check
that main.py's `run_defense()` call site forwards the parsed values through.

Run with: python -m unittest tests.test_main_cli_ml_filterrag -v
"""
import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as main_module  # noqa: E402 -- see module docstring re: heavy deps
from defense.filterrag import DEFAULT_SEMANTIC_THRESHOLD, VALID_MATCHING_MODES  # noqa: E402
from defense.ml_filterrag import DEFAULT_LM_MODEL, DEFAULT_THRESHOLD  # noqa: E402


class TestMlFilterragCliArgs(unittest.TestCase):
    def _parse(self, extra_argv):
        old_argv = sys.argv
        sys.argv = ["main.py"] + extra_argv
        try:
            return main_module.parse_args()
        finally:
            sys.argv = old_argv

    def test_default_model_path_is_none(self):
        args = self._parse([])
        self.assertIsNone(args.ml_filterrag_model_path)

    def test_model_path_parses_as_string(self):
        args = self._parse(["--ml_filterrag_model_path", "models/ml_filterrag/hotpotqa_random_forest.joblib"])
        self.assertEqual(args.ml_filterrag_model_path, "models/ml_filterrag/hotpotqa_random_forest.joblib")

    def test_default_threshold_matches_module_default(self):
        args = self._parse([])
        self.assertEqual(args.ml_filterrag_threshold, DEFAULT_THRESHOLD)
        self.assertEqual(DEFAULT_THRESHOLD, 0.5)

    def test_custom_threshold_parses_as_float(self):
        args = self._parse(["--ml_filterrag_threshold", "0.7"])
        self.assertAlmostEqual(args.ml_filterrag_threshold, 0.7)

    def test_default_lm_model_matches_module_default(self):
        args = self._parse([])
        self.assertEqual(args.ml_filterrag_lm_model, DEFAULT_LM_MODEL)
        self.assertEqual(DEFAULT_LM_MODEL, "distilgpt2")

    def test_custom_lm_model_parses_as_string(self):
        args = self._parse(["--ml_filterrag_lm_model", "gpt2"])
        self.assertEqual(args.ml_filterrag_lm_model, "gpt2")

    def test_default_matching_mode_is_semantic(self):
        # Unlike --filterrag_matching_mode (which defaults to "exact" for
        # backward compatibility), ml_filterrag has no pre-existing
        # behavior to preserve, so it defaults straight to the
        # paper-faithful "semantic" mode.
        args = self._parse([])
        self.assertEqual(args.ml_filterrag_matching_mode, "semantic")

    def test_matching_mode_choices_are_exact_and_semantic(self):
        self.assertEqual(set(VALID_MATCHING_MODES), {"exact", "semantic"})
        args = self._parse(["--ml_filterrag_matching_mode", "exact"])
        self.assertEqual(args.ml_filterrag_matching_mode, "exact")

    def test_invalid_matching_mode_rejected_by_argparse(self):
        with self.assertRaises(SystemExit):
            self._parse(["--ml_filterrag_matching_mode", "not_a_mode"])

    def test_default_semantic_threshold_matches_paper(self):
        args = self._parse([])
        self.assertEqual(args.ml_filterrag_semantic_threshold, DEFAULT_SEMANTIC_THRESHOLD)
        self.assertEqual(DEFAULT_SEMANTIC_THRESHOLD, 0.6)

    def test_custom_semantic_threshold_parses_as_float(self):
        args = self._parse(["--ml_filterrag_semantic_threshold", "0.42"])
        self.assertAlmostEqual(args.ml_filterrag_semantic_threshold, 0.42)

    def test_run_defense_call_site_forwards_new_args(self):
        """Static source check that main.py's run_defense(...) call passes
        every ml_filterrag_* parsed arg through -- avoids actually invoking
        main()'s retrieval/generation pipeline."""
        source = inspect.getsource(main_module)
        self.assertIn("ml_filterrag_model_path=args.ml_filterrag_model_path", source)
        self.assertIn("ml_filterrag_threshold=args.ml_filterrag_threshold", source)
        self.assertIn("ml_filterrag_matching_mode=args.ml_filterrag_matching_mode", source)
        self.assertIn("ml_filterrag_semantic_threshold=args.ml_filterrag_semantic_threshold", source)
        self.assertIn("ml_filterrag_lm_model=args.ml_filterrag_lm_model", source)

    def test_ml_filterrag_is_a_valid_defense_choice(self):
        args = self._parse(["--defense", "ml_filterrag"])
        self.assertEqual(args.defense, "ml_filterrag")


if __name__ == "__main__":
    unittest.main()
