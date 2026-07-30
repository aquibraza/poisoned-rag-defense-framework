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
from defense import filterrag as filterrag_module
from defense.passages import count_poison_clean, label_passages


class FakeSemanticMatcher:
    """Dependency-free test double for `SemanticWordMatcher`, mirroring
    `tests/test_filterrag.py`'s helper of the same name -- duplicated here
    (rather than imported) so this file stays self-contained, matching the
    existing `FakeSentenceTransformer` convention in this file."""

    def __init__(self, similarities):
        self.similarities = similarities
        self.call_count = 0

    def similarity_matrix(self, words_a, words_b):
        self.call_count += 1
        return [[self.similarities.get((wa, wb), 0.0) for wb in words_b] for wa in words_a]


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

    def test_random_removal_varies_by_query_id_with_same_base_seed(self):
        """Regression test: previously every query in a run passed the same
        bare seed to random_remove_same_count, so with a stable passage
        ordering (poison-before-clean from score sorting) the SAME relative
        positions were removed for every query. run_defense must now thread
        query_id through so different queries get different draws."""
        passages = label_passages(make_raw_hotpotqa_like(n_clean=4, n_poison=4))
        kept_q1, _ = dispatch.run_defense(
            "random_remove_same_count", "Who was born first?", passages, "hotpotqa",
            device="cpu", seed=12, query_id="query-1",
        )
        kept_q2, _ = dispatch.run_defense(
            "random_remove_same_count", "Who was born first?", passages, "hotpotqa",
            device="cpu", seed=12, query_id="query-2",
        )
        doc_ids_q1 = sorted(p.doc_id for p in kept_q1)
        doc_ids_q2 = sorted(p.doc_id for p in kept_q2)
        self.assertNotEqual(doc_ids_q1, doc_ids_q2)


class TestFilterragQueryOnlyThroughDispatch(unittest.TestCase):
    """filterrag_query_only needs no sentence-transformers/torch model at
    all (unlike ragdefender_original), so this doesn't need the
    DispatchSmokeTestBase._get_s_model patch -- it's the cheapest defense to
    smoke test."""

    def test_filterrag_query_only_removes_keyword_stuffed_passage(self):
        raw = [
            {"doc_id": "clean1", "context": "Some unrelated clean fact about the world.", "score": 0.9, "source": "corpus", "is_poison": False},
            {"doc_id": "adv1", "context": "texas texas texas texas is the texas state texas.", "score": 0.85, "source": "adversarial", "is_poison": True},
        ]
        passages = label_passages(raw)
        kept, diag = dispatch.run_defense("filterrag_query_only", "Where is texas?", passages, "hotpotqa")
        kept_ids = {p.doc_id for p in kept}
        self.assertNotIn("adv1", kept_ids)
        self.assertIn("clean1", kept_ids)
        self.assertEqual(diag["N_adv_estimated_by_ragdefender"], 1)
        self.assertIn("filterrag_scores", diag)

    def test_filterrag_epsilon_is_configurable(self):
        raw = [
            {"doc_id": "clean1", "context": "Some unrelated clean fact.", "score": 0.9, "source": "corpus", "is_poison": False},
            {"doc_id": "adv1", "context": "texas texas texas is the state.", "score": 0.85, "source": "adversarial", "is_poison": True},
        ]
        passages = label_passages(raw)
        kept, _ = dispatch.run_defense(
            "filterrag_query_only", "Where is texas?", passages, "hotpotqa", filterrag_epsilon=1000.0
        )
        self.assertEqual(len(kept), len(passages))  # nothing crosses an impossibly high threshold


class TestFilterragMatchingModeThroughDispatch(unittest.TestCase):
    """--filterrag_matching_mode/--filterrag_semantic_threshold (main.py) ->
    filterrag_matching_mode/filterrag_semantic_threshold (dispatch.run_defense)
    are correctly forwarded into defense/filterrag.py, for both
    filterrag_query_only and (mocked-SLM) filterrag. No real
    sentence_transformers model is loaded -- `get_semantic_word_matcher` is
    monkeypatched with `FakeSemanticMatcher`, so this needs no network
    access and makes no LLM/GPT/API call."""

    def _passages(self):
        raw = [
            {"doc_id": "clean1", "context": "Some unrelated clean fact about the world.", "score": 0.9, "source": "corpus", "is_poison": False},
            {"doc_id": "adv1", "context": "automobile automobile automobile automobile is the thing automobile automobile.", "score": 0.85, "source": "adversarial", "is_poison": True},
        ]
        return label_passages(raw)

    def test_matching_mode_and_threshold_forwarded_to_filterrag_query_only(self):
        fake_matcher = FakeSemanticMatcher({("automobile", "sedan"): 0.85})
        with mock.patch.object(filterrag_module, "get_semantic_word_matcher", return_value=fake_matcher):
            kept, diag = dispatch.run_defense(
                "filterrag_query_only", "What powers my sedan?", self._passages(), "hotpotqa",
                filterrag_matching_mode="semantic", filterrag_semantic_threshold=0.6,
            )
        self.assertNotIn("adv1", {p.doc_id for p in kept})  # only catchable via the synonym match
        self.assertIn("matching_mode=semantic", diag["notes"])
        self.assertIn("semantic_threshold=0.6", diag["notes"])
        for s in diag["filterrag_scores"]:
            self.assertEqual(s["matching_mode"], "semantic")
            self.assertIsNone(s["slm_answer"])  # query_only never calls an SLM

    def test_exact_matching_mode_is_still_the_dispatch_default(self):
        kept, diag = dispatch.run_defense(
            "filterrag_query_only", "What powers my sedan?", self._passages(), "hotpotqa",
        )
        self.assertIn("adv1", {p.doc_id for p in kept})  # no verbatim overlap -> exact mode can't catch it
        self.assertIn("matching_mode=exact", diag["notes"])

    def test_matching_mode_forwarded_to_full_filterrag_mode(self):
        """--defense filterrag (SLM mode) also respects filterrag_matching_mode;
        local_hf_slm_answer_fn is mocked so no HF model is downloaded/loaded."""
        fake_matcher = FakeSemanticMatcher({("automobile", "sedan"): 0.85})
        with mock.patch.object(dispatch, "local_hf_slm_answer_fn", return_value=lambda q, p: "an answer"), \
             mock.patch.object(filterrag_module, "get_semantic_word_matcher", return_value=fake_matcher):
            kept, diag = dispatch.run_defense(
                "filterrag", "What powers my sedan?", self._passages(), "hotpotqa",
                filterrag_matching_mode="semantic", filterrag_semantic_threshold=0.42,
            )
        self.assertIn("mode=slm", diag["notes"])
        self.assertIn("matching_mode=semantic", diag["notes"])
        self.assertIn("semantic_threshold=0.42", diag["notes"])
        for s in diag["filterrag_scores"]:
            self.assertEqual(s["slm_answer"], "an answer")  # SLM step actually ran


class TestUnknownDefenseRaises(unittest.TestCase):
    def test_unknown_defense_name_raises(self):
        passages = label_passages(make_raw_hotpotqa_like())
        with self.assertRaises(ValueError):
            dispatch.run_defense("not_a_real_defense", "q?", passages, "hotpotqa")


if __name__ == "__main__":
    unittest.main()
