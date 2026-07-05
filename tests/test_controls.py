"""Tests for defense/controls.py -- oracle and random removal diagnostic controls.

Run with: python -m unittest tests.test_controls -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defense.controls import (
    oracle_remove_all_poison,
    random_remove_same_count,
    stable_seed_for_query,
)
from defense.passages import count_poison_clean, doc_ids, label_passages


def make_mixed_passages(n_clean=6, n_poison=3):
    raw = []
    for i in range(n_clean):
        raw.append({"doc_id": f"c{i}", "context": f"clean {i}", "score": 1.0 - i * 0.01, "source": "corpus", "is_poison": False})
    for i in range(n_poison):
        raw.append({"doc_id": f"adv{i}", "context": f"adv {i}", "score": 0.5 - i * 0.01, "source": "adversarial", "is_poison": True})
    return label_passages(raw)


class TestOracleRemoveAllPoison(unittest.TestCase):
    def test_removes_all_poison_and_no_clean(self):
        passages = make_mixed_passages(n_clean=6, n_poison=3)
        kept, diag = oracle_remove_all_poison(passages)
        self.assertEqual(len(kept), 6)
        n_poison_kept, n_clean_kept = count_poison_clean(kept)
        self.assertEqual(n_poison_kept, 0)
        self.assertEqual(n_clean_kept, 6)

    def test_diag_extra_reports_true_poison_count(self):
        passages = make_mixed_passages(n_clean=4, n_poison=5)
        _, diag = oracle_remove_all_poison(passages)
        self.assertEqual(diag["N_adv_estimated_by_ragdefender"], 5)
        self.assertIn("diagnostic control", diag["notes"])
        self.assertIn("not a deployable defense", diag["notes"])

    def test_no_poison_present_removes_nothing(self):
        passages = make_mixed_passages(n_clean=5, n_poison=0)
        kept, diag = oracle_remove_all_poison(passages)
        self.assertEqual(len(kept), 5)
        self.assertEqual(diag["N_adv_estimated_by_ragdefender"], 0)

    def test_all_poison_removes_everything(self):
        passages = make_mixed_passages(n_clean=0, n_poison=4)
        kept, _ = oracle_remove_all_poison(passages)
        self.assertEqual(kept, [])


class TestRandomRemoveSameCount(unittest.TestCase):
    def test_removes_exact_requested_count(self):
        passages = make_mixed_passages(n_clean=6, n_poison=3)
        for n in [0, 1, 3, 5, 9]:
            kept, diag = random_remove_same_count(passages, n_to_remove=n, seed=42)
            self.assertEqual(len(kept), len(passages) - n)
            self.assertEqual(diag["N_adv_estimated_by_ragdefender"], n)

    def test_clamps_count_above_len(self):
        passages = make_mixed_passages(n_clean=2, n_poison=1)
        kept, _ = random_remove_same_count(passages, n_to_remove=100, seed=1)
        self.assertEqual(kept, [])

    def test_clamps_negative_count_to_zero(self):
        passages = make_mixed_passages(n_clean=2, n_poison=1)
        kept, _ = random_remove_same_count(passages, n_to_remove=-5, seed=1)
        self.assertEqual(len(kept), len(passages))

    def test_deterministic_given_seed(self):
        passages = make_mixed_passages(n_clean=6, n_poison=3)
        kept1, _ = random_remove_same_count(passages, n_to_remove=4, seed=7)
        kept2, _ = random_remove_same_count(passages, n_to_remove=4, seed=7)
        self.assertEqual(doc_ids(kept1), doc_ids(kept2))

    def test_different_seeds_can_differ(self):
        passages = make_mixed_passages(n_clean=10, n_poison=10)
        kept_a, _ = random_remove_same_count(passages, n_to_remove=5, seed=1)
        kept_b, _ = random_remove_same_count(passages, n_to_remove=5, seed=2)
        # Not a strict guarantee for all seeds, but overwhelmingly likely for n=20/5.
        self.assertNotEqual(doc_ids(kept_a), doc_ids(kept_b))

    def test_diag_notes_marks_diagnostic_control(self):
        passages = make_mixed_passages(n_clean=3, n_poison=2)
        _, diag = random_remove_same_count(passages, n_to_remove=2, seed=3)
        self.assertIn("diagnostic control", diag["notes"])
        self.assertIn("not a deployable defense", diag["notes"])

    def test_without_query_id_seed_is_used_directly_same_indices_every_call(self):
        """Backward-compat / base-case behavior: with no query_id, the same
        (seed, n, len(passages)) always removes the same relative indices --
        this is exactly the behavior that made the baseline non-random
        *across queries* when every call used the same bare seed."""
        passages = make_mixed_passages(n_clean=6, n_poison=3)
        kept1, _ = random_remove_same_count(passages, n_to_remove=4, seed=12)
        kept2, _ = random_remove_same_count(passages, n_to_remove=4, seed=12)
        self.assertEqual(doc_ids(kept1), doc_ids(kept2))

    def test_query_id_varies_removal_across_queries_with_same_base_seed(self):
        """The actual fix: same base seed, same passage layout, different
        query_id -> different removed set (overwhelmingly likely), so the
        random control no longer removes the same relative positions for
        every query in a run."""
        passages = make_mixed_passages(n_clean=6, n_poison=3)
        kept_q1, _ = random_remove_same_count(
            passages, n_to_remove=4, seed=12, query_id="query-1"
        )
        kept_q2, _ = random_remove_same_count(
            passages, n_to_remove=4, seed=12, query_id="query-2"
        )
        self.assertNotEqual(doc_ids(kept_q1), doc_ids(kept_q2))

    def test_query_id_removal_is_reproducible(self):
        """Same (base_seed, query_id) always yields the same removal --
        the run stays fully reproducible from base_seed alone."""
        passages = make_mixed_passages(n_clean=6, n_poison=3)
        kept_a, _ = random_remove_same_count(
            passages, n_to_remove=4, seed=12, query_id="query-1"
        )
        kept_b, _ = random_remove_same_count(
            passages, n_to_remove=4, seed=12, query_id="query-1"
        )
        self.assertEqual(doc_ids(kept_a), doc_ids(kept_b))

    def test_notes_include_query_id_and_effective_seed(self):
        passages = make_mixed_passages(n_clean=3, n_poison=2)
        _, diag = random_remove_same_count(
            passages, n_to_remove=2, seed=3, query_id="abc123"
        )
        self.assertIn("abc123", diag["notes"])
        self.assertIn("base_seed=3", diag["notes"])


class TestStableSeedForQuery(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(
            stable_seed_for_query(12, "q1"), stable_seed_for_query(12, "q1")
        )

    def test_varies_by_query_id(self):
        self.assertNotEqual(
            stable_seed_for_query(12, "q1"), stable_seed_for_query(12, "q2")
        )

    def test_varies_by_base_seed(self):
        self.assertNotEqual(
            stable_seed_for_query(12, "q1"), stable_seed_for_query(99, "q1")
        )

    def test_returns_int(self):
        self.assertIsInstance(stable_seed_for_query(12, "q1"), int)


if __name__ == "__main__":
    unittest.main()
