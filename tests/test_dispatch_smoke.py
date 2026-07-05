"""Smoke tests for defense/dispatch.py -- exercised fully offline.

The real `paraphrase-MiniLM-L6-v2` sentence-transformer model requires a
network download, so these tests monkeypatch `defense_runner._get_s_model`
with a deterministic fake encoder. This proves that `none`,
`ragdefender_original`, `oracle_remove_all_poison`, and
`random_remove_same_count` all run end-to-end through the dispatcher without
any network access or LLM/API call.

Run with: python -m unittest tests.test_dispatch_smoke -v
"""
import hashlib
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from defense import defense_runner, dispatch
from defense.passages import count_poison_clean, label_passages


class FakeSentenceTransformer:
    """Deterministic, dependency-free stand-in for SentenceTransformer.

    Embeds each string via an md5-seeded random vector, so identical text
    always maps to the identical embedding (needed for cos_sim /
    AgglomerativeClustering to behave deterministically in tests), with no
    model download or network access required.
    """

    def encode(self, text_list, convert_to_tensor=True):
        vectors = []
        for t in text_list:
            digest = hashlib.md5(t.encode("utf-8")).hexdigest()
            seed = int(digest[:8], 16)
            gen = torch.Generator().manual_seed(seed)
            vectors.append(torch.rand(16, generator=gen))
        return torch.stack(vectors)


def make_raw_hotpotqa_like(n_clean=3, n_poison=2, question="Who was born first?"):
    """Adversarial texts share the question prefix (as LM_targeted attacks
    do), clean texts don't -- mirrors real passage construction enough for
    a smoke test of the plumbing, not the ML estimate quality itself."""
    raw = []
    for i in range(n_clean):
        raw.append(
            {
                "doc_id": f"c{i}",
                "context": f"Some unrelated clean fact number {i} about the world.",
                "score": 0.9 - i * 0.01,
                "source": "corpus",
                "is_poison": False,
            }
        )
    for i in range(n_poison):
        raw.append(
            {
                "doc_id": f"adv{i}",
                "context": f"{question} Variant {i} says the incorrect answer is X.",
                "score": 0.5 - i * 0.01,
                "source": "adversarial",
                "is_poison": True,
            }
        )
    return raw


class DispatchSmokeTestBase(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch.object(
            defense_runner, "_get_s_model", return_value=FakeSentenceTransformer()
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)


class TestNoneDefense(DispatchSmokeTestBase):
    def test_none_is_pass_through(self):
        passages = label_passages(make_raw_hotpotqa_like())
        kept, diag = dispatch.run_defense("none", "q?", passages, "hotpotqa")
        self.assertEqual(len(kept), len(passages))
        self.assertIsNone(diag["N_adv_estimated_by_ragdefender"])


class TestRagdefenderOriginalSmoke(DispatchSmokeTestBase):
    def test_ragdefender_original_runs_offline_hotpotqa(self):
        passages = label_passages(make_raw_hotpotqa_like())
        kept, diag = dispatch.run_defense(
            "ragdefender_original", "Who was born first?", passages, "hotpotqa", device="cpu"
        )
        self.assertIsInstance(kept, list)
        self.assertLessEqual(len(kept), len(passages))
        self.assertIn("N_adv_estimated_by_ragdefender", diag)
        self.assertIsInstance(diag["N_adv_estimated_by_ragdefender"], int)

    def test_ragdefender_legacy_alias_runs_offline_singlehop(self):
        passages = label_passages(make_raw_hotpotqa_like())
        kept, diag = dispatch.run_defense(
            "ragdefender", "some nq-style question?", passages, "nq", device="cpu"
        )
        self.assertIsInstance(kept, list)
        self.assertIsInstance(diag["N_adv_estimated_by_ragdefender"], int)

    def test_kept_passages_preserve_metadata(self):
        """Whatever subset RAGDefender keeps, each kept passage's doc_id/
        is_poison must still match its original metadata (no corruption
        through the text round-trip)."""
        passages = label_passages(make_raw_hotpotqa_like(n_clean=4, n_poison=4))
        by_text = {p.text: p for p in passages}
        kept, _ = dispatch.run_defense(
            "ragdefender_original", "Who was born first?", passages, "hotpotqa", device="cpu"
        )
        for p in kept:
            self.assertEqual(p, by_text[p.text])


class TestOracleAndRandomThroughDispatch(DispatchSmokeTestBase):
    def test_oracle_removes_all_poison_via_dispatch(self):
        passages = label_passages(make_raw_hotpotqa_like(n_clean=3, n_poison=2))
        kept, diag = dispatch.run_defense("oracle_remove_all_poison", "q?", passages, "hotpotqa")
        n_poison, n_clean = count_poison_clean(kept)
        self.assertEqual(n_poison, 0)
        self.assertEqual(n_clean, 3)
        self.assertEqual(diag["N_adv_estimated_by_ragdefender"], 2)

    def test_random_removes_same_count_as_ragdefender_estimate(self):
        passages = label_passages(make_raw_hotpotqa_like(n_clean=3, n_poison=2))
        kept, diag = dispatch.run_defense(
            "random_remove_same_count", "Who was born first?", passages, "hotpotqa", device="cpu", seed=5
        )
        n_estimate = diag["N_adv_estimated_by_ragdefender"]
        self.assertEqual(len(kept), len(passages) - n_estimate)


class TestUnknownDefenseRaises(unittest.TestCase):
    def test_unknown_defense_name_raises(self):
        passages = label_passages(make_raw_hotpotqa_like())
        with self.assertRaises(ValueError):
            dispatch.run_defense("not_a_real_defense", "q?", passages, "hotpotqa")


if __name__ == "__main__":
    unittest.main()
