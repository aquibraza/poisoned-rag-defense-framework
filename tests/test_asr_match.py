"""Tests for defense/asr_match.py -- legacy substring vs. strict
word-boundary ASR matching.

Run with: python -m unittest tests.test_asr_match -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defense.asr_match import _legacy_clean_str, legacy_match, strict_match


class TestLegacyCleanStrMatchesSrcUtils(unittest.TestCase):
    """_legacy_clean_str is a deliberately-duplicated copy of
    src/utils.py:clean_str (see defense/asr_match.py module docstring for
    why it isn't imported directly). Cross-check the two stay identical
    whenever src.utils happens to be importable (it requires
    torch/transformers/sentence-transformers/beir, so this test is skipped
    -- not failed -- in minimal environments that lack those)."""

    def test_matches_src_utils_clean_str(self):
        try:
            from src.utils import clean_str as real_clean_str
        except ImportError as e:
            self.skipTest(f"src.utils not importable in this environment: {e}")
            return
        for s in ["Texas", "  No.  ", "does NOT provide information.", "", "A.", "a..b."]:
            self.assertEqual(_legacy_clean_str(s), real_clean_str(s))


class TestLegacyMatchUnchanged(unittest.TestCase):
    """legacy_match must keep reproducing the repo's original (buggy)
    substring behavior byte-for-byte -- that's the whole point of keeping
    it around unmodified for backward compatibility."""

    def test_no_substring_matches_inside_not(self):
        self.assertTrue(legacy_match("no", "does not provide information"))

    def test_none_input_returns_none(self):
        self.assertIsNone(legacy_match("no", None))
        self.assertIsNone(legacy_match(None, "some response"))

    def test_basic_substring_match(self):
        self.assertTrue(legacy_match("texas", "The state is Texas."))


class TestStrictMatchRequiredCases(unittest.TestCase):
    """Exact cases requested for the strict ASR evaluator."""

    def test_no_vs_does_not_provide_information_is_false(self):
        self.assertFalse(strict_match("no", "does not provide information"))

    def test_no_vs_standalone_no_token_is_true(self):
        self.assertTrue(strict_match("no", "No, they are not in the same place"))

    def test_yes_vs_standalone_yes_token_is_true(self):
        self.assertTrue(strict_match("yes", "Yes, they are the same person."))

    def test_texas_vs_in_texas_is_true(self):
        self.assertTrue(strict_match("texas", "in Texas"))

    def test_texas_vs_text_as_is_false(self):
        self.assertFalse(strict_match("texas", "text as"))


class TestStrictMatchNoDenylistWords(unittest.TestCase):
    """'no' must never match inside these confusable words."""

    def test_no_does_not_match_not(self):
        self.assertFalse(strict_match("no", "I am not sure."))

    def test_no_does_not_match_none(self):
        self.assertFalse(strict_match("no", "There is none available."))

    def test_no_does_not_match_another(self):
        self.assertFalse(strict_match("no", "This is another option."))

    def test_no_does_not_match_known(self):
        self.assertFalse(strict_match("no", "This is a well known fact."))


class TestStrictMatchIsTokenBoundaryNotSemantic(unittest.TestCase):
    """strict_match is 'strict token-boundary ASR', not a semantic yes/no
    evaluator: it requires a standalone yes/no token (or exact
    token-subsequence match), and does not perform negation detection."""

    def test_negative_answer_without_standalone_no_token_does_not_match(self):
        # A human reads this as "no", but there is no standalone "no"
        # token -- only "not" -- so strict_match correctly does not claim
        # a semantic match here (documented limitation, not a bug).
        self.assertFalse(strict_match("no", "They are not in the same place."))

    def test_negative_answer_with_standalone_no_token_does_match(self):
        # Same underlying claim, but phrased with a leading standalone
        # "No," -- this is the case strict_match is designed to catch.
        self.assertTrue(strict_match("no", "No, they are not in the same place."))


class TestStrictMatchGeneral(unittest.TestCase):
    def test_multi_word_target_matches_as_contiguous_subsequence(self):
        self.assertTrue(
            strict_match("Robert Erskine Childers", "It was Robert Erskine Childers who wrote it.")
        )

    def test_multi_word_target_out_of_order_does_not_match(self):
        self.assertFalse(
            strict_match("Robert Erskine Childers", "Childers, Robert Erskine, wrote it.")
        )

    def test_none_input_returns_none(self):
        self.assertIsNone(strict_match("no", None))
        self.assertIsNone(strict_match(None, "some response"))

    def test_case_insensitive(self):
        self.assertTrue(strict_match("TEXAS", "the state is texas"))

    def test_punctuation_does_not_block_match(self):
        self.assertTrue(strict_match("texas", "Texas, apparently."))


if __name__ == "__main__":
    unittest.main()
