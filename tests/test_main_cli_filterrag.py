"""CLI wiring tests for --filterrag_matching_mode / --filterrag_semantic_threshold
in main.py (added alongside defense/filterrag.py's semantic matching option --
see docs/FILTERRAG_FIDELITY_AUDIT.md).

Requires main.py's full dependency set to *import* (torch, transformers,
sentence-transformers, beir, etc. -- same requirement as
test_dispatch_smoke.py; see tests/README.md). Does **not** run main()'s
actual retrieval/generation pipeline: no dataset/BEIR access, no GPU
required, and no LLM/GPT/API call of any kind. Only exercises `argparse` via
`main.parse_args()`, plus a source-level check that main.py's `run_defense()`
call site forwards the parsed values through (rather than executing that
call, which would require real retrieval results and adversarial-text
fixtures on disk).

Run with: python -m unittest tests.test_main_cli_filterrag -v
"""
import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as main_module  # noqa: E402 -- see module docstring re: heavy deps
from defense.filterrag import DEFAULT_SEMANTIC_THRESHOLD, VALID_MATCHING_MODES  # noqa: E402


class TestFilterragCliArgs(unittest.TestCase):
    def _parse(self, extra_argv):
        old_argv = sys.argv
        sys.argv = ["main.py"] + extra_argv
        try:
            return main_module.parse_args()
        finally:
            sys.argv = old_argv

    def test_default_matching_mode_is_exact(self):
        # Default stays "exact" (legacy/backward-compatible), not the
        # paper-faithful "semantic" -- see docs/FILTERRAG_FIDELITY_AUDIT.md §4.
        args = self._parse([])
        self.assertEqual(args.filterrag_matching_mode, "exact")

    def test_default_semantic_threshold_matches_paper(self):
        args = self._parse([])
        self.assertEqual(args.filterrag_semantic_threshold, DEFAULT_SEMANTIC_THRESHOLD)
        self.assertEqual(DEFAULT_SEMANTIC_THRESHOLD, 0.6)  # paper Section IV-B2

    def test_matching_mode_choices_are_exact_and_semantic(self):
        self.assertEqual(set(VALID_MATCHING_MODES), {"exact", "semantic"})
        args = self._parse(["--filterrag_matching_mode", "semantic"])
        self.assertEqual(args.filterrag_matching_mode, "semantic")

    def test_invalid_matching_mode_rejected_by_argparse(self):
        with self.assertRaises(SystemExit):
            self._parse(["--filterrag_matching_mode", "not_a_mode"])

    def test_custom_semantic_threshold_parses_as_float(self):
        args = self._parse(["--filterrag_semantic_threshold", "0.42"])
        self.assertAlmostEqual(args.filterrag_semantic_threshold, 0.42)

    def test_run_defense_call_site_forwards_new_args(self):
        """Static source check that main.py's run_defense(...) call passes
        filterrag_matching_mode=args.filterrag_matching_mode and
        filterrag_semantic_threshold=args.filterrag_semantic_threshold --
        avoids actually invoking main()'s retrieval/generation pipeline."""
        source = inspect.getsource(main_module)
        self.assertIn("filterrag_matching_mode=args.filterrag_matching_mode", source)
        self.assertIn("filterrag_semantic_threshold=args.filterrag_semantic_threshold", source)


if __name__ == "__main__":
    unittest.main()
