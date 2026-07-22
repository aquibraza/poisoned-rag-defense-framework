"""Tests for defense/passages.py -- poison label propagation.

Run with: python -m unittest tests.test_passages -v
(or: python -m unittest discover -s tests)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defense.passages import (
    RetrievedPassage,
    count_poison_clean,
    doc_ids,
    filter_by_doc_ids,
    label_passages,
    poison_flags,
    removed_passages,
    texts,
)


def make_raw_mixed():
    """3 clean corpus passages + 2 adversarial passages, in a fixed order."""
    return [
        {"doc_id": "c1", "context": "clean text 1", "score": 0.9, "source": "corpus", "is_poison": False},
        {"doc_id": "adv::q1::0", "context": "adv text 0", "score": 0.85, "source": "adversarial", "is_poison": True},
        {"doc_id": "c2", "context": "clean text 2", "score": 0.8, "source": "corpus", "is_poison": False},
        {"doc_id": "adv::q1::1", "context": "adv text 1", "score": 0.75, "source": "adversarial", "is_poison": True},
        {"doc_id": "c3", "context": "clean text 3", "score": 0.7, "source": "corpus", "is_poison": False},
    ]


class TestLabelPassages(unittest.TestCase):
    def test_poison_label_propagation(self):
        passages = label_passages(make_raw_mixed())
        self.assertEqual(len(passages), 5)
        # Labels must come from source, not be re-derived.
        expected_poison = [False, True, False, True, False]
        self.assertEqual(poison_flags(passages), expected_poison)
        expected_sources = ["corpus", "adversarial", "corpus", "adversarial", "corpus"]
        self.assertEqual([p.source for p in passages], expected_sources)

    def test_rank_assigned_by_list_order(self):
        passages = label_passages(make_raw_mixed())
        self.assertEqual([p.rank for p in passages], [0, 1, 2, 3, 4])

    def test_doc_id_and_score_preserved(self):
        passages = label_passages(make_raw_mixed())
        self.assertEqual(doc_ids(passages), ["c1", "adv::q1::0", "c2", "adv::q1::1", "c3"])
        self.assertAlmostEqual(passages[0].retrieval_score, 0.9)

    def test_missing_doc_id_raises(self):
        raw = [{"context": "no id here", "score": 0.5, "source": "corpus", "is_poison": False}]
        with self.assertRaises(KeyError):
            label_passages(raw)

    def test_count_poison_clean(self):
        passages = label_passages(make_raw_mixed())
        n_poison, n_clean = count_poison_clean(passages)
        self.assertEqual(n_poison, 2)
        self.assertEqual(n_clean, 3)

    def test_texts_helper(self):
        passages = label_passages(make_raw_mixed())
        self.assertEqual(
            texts(passages),
            ["clean text 1", "adv text 0", "clean text 2", "adv text 1", "clean text 3"],
        )


class TestFilterAndDiff(unittest.TestCase):
    def setUp(self):
        self.passages = label_passages(make_raw_mixed())

    def test_filter_by_doc_ids_preserves_order(self):
        kept = filter_by_doc_ids(self.passages, ["c3", "c1"])
        # Order follows the original passage list, not the keep_doc_ids arg.
        self.assertEqual(doc_ids(kept), ["c1", "c3"])

    def test_removed_passages_is_set_difference_by_doc_id(self):
        after = filter_by_doc_ids(self.passages, ["c1", "c2", "c3"])
        removed = removed_passages(self.passages, after)
        self.assertEqual(doc_ids(removed), ["adv::q1::0", "adv::q1::1"])
        self.assertTrue(all(p.is_poison for p in removed))

    def test_removed_passages_empty_when_nothing_removed(self):
        removed = removed_passages(self.passages, self.passages)
        self.assertEqual(removed, [])


if __name__ == "__main__":
    unittest.main()
