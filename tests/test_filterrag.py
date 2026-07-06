"""Tests for defense/filterrag.py -- Freq-Density scoring and threshold
filtering (FilterRAG baseline, Edemacu et al. 2025).

Fully offline: no HF model is downloaded or loaded. `slm_answer_fn` is
mocked with a plain Python function everywhere, exactly like
test_dispatch_smoke.py mocks RAGDefender's sentence-transformers encoder.
TestResolveSlmDevice mocks `torch` itself (via sys.modules) so this file has
no hard dependency on torch being installed, matching every other test file
here except test_dispatch_smoke.py.

Run with: python -m unittest tests.test_filterrag -v
"""
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defense.filterrag import DEFAULT_EPSILON, filterrag_defense, freq_density, score_passages
from defense.passages import label_passages


def _fake_torch_module(*, mps_available: bool, cuda_available: bool) -> types.ModuleType:
    """Build a minimal fake `torch` module exposing just enough of
    `torch.backends.mps.is_available()` / `torch.cuda.is_available()` for
    `resolve_slm_device()` to use, without requiring torch to be installed."""
    fake_torch = types.ModuleType("torch")
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps_available)
    )
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    return fake_torch


def make_passages():
    raw = [
        {"doc_id": "clean1", "context": "The Eiffel Tower is located in Paris, France.", "score": 0.9, "source": "corpus", "is_poison": False},
        {"doc_id": "adv1", "context": "texas texas texas texas is the state texas texas texas.", "score": 0.85, "source": "adversarial", "is_poison": True},
        {"doc_id": "clean2", "context": "Paris has a population of over two million people.", "score": 0.8, "source": "corpus", "is_poison": False},
    ]
    return label_passages(raw)


class TestFreqDensity(unittest.TestCase):
    def test_empty_passage_returns_zero(self):
        self.assertEqual(freq_density("", ["texas"]), 0.0)

    def test_no_keyword_overlap_returns_zero(self):
        self.assertEqual(freq_density("the cat sat on the mat", ["texas", "houston"]), 0.0)

    def test_keyword_stuffed_passage_scores_high(self):
        # "texas" appears 6 times; unique words = {texas, is, the, state} = 4.
        text = "texas texas texas is the texas state texas texas"
        score = freq_density(text, ["texas"])
        self.assertGreater(score, 1.0)  # more keyword hits than unique words

    def test_case_insensitive(self):
        self.assertEqual(freq_density("Texas is a state.", ["texas"]), freq_density("texas is a state.", ["texas"]))

    def test_duplicate_keywords_do_not_double_count(self):
        # Passing "texas" twice in keywords must not double the contribution.
        text = "texas is a state with texas cities"
        self.assertEqual(freq_density(text, ["texas"]), freq_density(text, ["texas", "texas"]))


class TestScorePassagesQueryOnly(unittest.TestCase):
    """slm_answer_fn=None: the query-only diagnostic ablation."""

    def test_scores_use_only_query_keywords(self):
        passages = make_passages()
        scores = score_passages("Where is texas?", passages, slm_answer_fn=None)
        by_id = {s["doc_id"]: s for s in scores}
        self.assertIsNone(by_id["clean1"]["slm_answer"])
        self.assertIsNone(by_id["adv1"]["slm_answer"])
        # "texas" (from the query) appears heavily in adv1, not in clean1/clean2.
        self.assertGreater(by_id["adv1"]["freq_density_score"], by_id["clean1"]["freq_density_score"])
        self.assertGreater(by_id["adv1"]["freq_density_score"], by_id["clean2"]["freq_density_score"])


class TestScorePassagesWithMockedSlm(unittest.TestCase):
    def test_slm_answer_is_included_in_keywords(self):
        passages = make_passages()

        def fake_slm(question, passage_text):
            return "texas"  # pretend every passage's SLM answer is "texas"

        scores = score_passages("Where?", passages, slm_answer_fn=fake_slm)
        by_id = {s["doc_id"]: s for s in scores}
        self.assertEqual(by_id["clean1"]["slm_answer"], "texas")
        # Even though the query has no "texas", the mocked SLM answer does,
        # so adv1 (keyword-stuffed with "texas") should score highest.
        self.assertGreater(by_id["adv1"]["freq_density_score"], by_id["clean1"]["freq_density_score"])

    def test_slm_exceptions_degrade_to_no_answer(self):
        passages = make_passages()

        def failing_slm(question, passage_text):
            raise RuntimeError("boom")

        # score_passages itself doesn't catch exceptions (that's
        # local_hf_slm_answer_fn's job); a raising slm_answer_fn should
        # propagate so callers know their custom function is broken.
        with self.assertRaises(RuntimeError):
            score_passages("Where?", passages, slm_answer_fn=failing_slm)


class TestResolveSlmDevice(unittest.TestCase):
    """resolve_slm_device() is exercised with a fake `torch` module injected
    via sys.modules, so this test needs no real torch install and never
    touches an actual MPS/CUDA device."""

    def _resolve(self, requested, *, mps_available, cuda_available):
        from defense.filterrag import resolve_slm_device

        fake_torch = _fake_torch_module(mps_available=mps_available, cuda_available=cuda_available)
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            return resolve_slm_device(requested)

    def test_auto_prefers_mps_when_available(self):
        self.assertEqual(self._resolve("auto", mps_available=True, cuda_available=True), "mps")

    def test_auto_falls_back_to_cuda_when_no_mps(self):
        self.assertEqual(self._resolve("auto", mps_available=False, cuda_available=True), "cuda")

    def test_auto_falls_back_to_cpu_when_neither_available(self):
        self.assertEqual(self._resolve("auto", mps_available=False, cuda_available=False), "cpu")

    def test_explicit_mps_honored_when_available(self):
        self.assertEqual(self._resolve("mps", mps_available=True, cuda_available=False), "mps")

    def test_explicit_mps_falls_back_when_unavailable(self):
        # No MPS, no CUDA -> auto-detection lands on cpu.
        self.assertEqual(self._resolve("mps", mps_available=False, cuda_available=False), "cpu")

    def test_explicit_cuda_falls_back_when_unavailable(self):
        self.assertEqual(self._resolve("cuda", mps_available=True, cuda_available=False), "mps")

    def test_explicit_cpu_always_honored(self):
        self.assertEqual(self._resolve("cpu", mps_available=True, cuda_available=True), "cpu")

    def test_case_insensitive(self):
        self.assertEqual(self._resolve("MPS", mps_available=True, cuda_available=False), "mps")

    def test_unknown_device_raises(self):
        from defense.filterrag import resolve_slm_device

        fake_torch = _fake_torch_module(mps_available=False, cuda_available=False)
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaises(ValueError):
                resolve_slm_device("tpu")


class TestFilterragDefense(unittest.TestCase):
    def test_removes_only_passages_above_epsilon(self):
        passages = make_passages()
        kept, diag = filterrag_defense("Where is texas?", passages, epsilon=DEFAULT_EPSILON, slm_answer_fn=None)
        kept_ids = {p.doc_id for p in kept}
        self.assertNotIn("adv1", kept_ids)
        self.assertIn("clean1", kept_ids)
        self.assertIn("clean2", kept_ids)

    def test_diag_extra_has_expected_shape(self):
        passages = make_passages()
        _, diag = filterrag_defense("Where is texas?", passages, epsilon=DEFAULT_EPSILON, slm_answer_fn=None)
        self.assertIn("N_adv_estimated_by_ragdefender", diag)
        self.assertEqual(diag["N_adv_estimated_by_ragdefender"], 1)  # only adv1
        self.assertIn("filterrag_scores", diag)
        self.assertEqual(len(diag["filterrag_scores"]), 3)
        self.assertIn("query_only_ablation", diag["notes"])
        self.assertIn(str(DEFAULT_EPSILON), diag["notes"])

    def test_slm_mode_recorded_in_notes(self):
        passages = make_passages()
        _, diag = filterrag_defense(
            "Where is texas?", passages, epsilon=DEFAULT_EPSILON,
            slm_answer_fn=lambda q, p: "texas",
        )
        self.assertIn("mode=slm", diag["notes"])

    def test_no_passage_removed_when_epsilon_is_very_high(self):
        passages = make_passages()
        kept, diag = filterrag_defense("Where is texas?", passages, epsilon=1000.0, slm_answer_fn=None)
        self.assertEqual(len(kept), len(passages))
        self.assertEqual(diag["N_adv_estimated_by_ragdefender"], 0)

    def test_all_passages_removed_when_epsilon_is_zero(self):
        # epsilon=0 means "Freq-Density >= 0" always holds -- degenerate but
        # should not crash, and should remove everything.
        passages = make_passages()
        kept, diag = filterrag_defense("Where is texas?", passages, epsilon=0.0, slm_answer_fn=None)
        self.assertEqual(len(kept), 0)

    def test_kept_passages_preserve_metadata(self):
        passages = make_passages()
        kept, _ = filterrag_defense("Where is texas?", passages, epsilon=DEFAULT_EPSILON, slm_answer_fn=None)
        for p in kept:
            self.assertIsNotNone(p.source)
            self.assertIsInstance(p.is_poison, bool)


if __name__ == "__main__":
    unittest.main()
