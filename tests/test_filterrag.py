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


class TestSlmPipelineDeviceFallback(unittest.TestCase):
    """_get_local_hf_slm_pipeline() smoke-tests a freshly-loaded non-cpu
    pipeline with one throwaway generate() call and falls back to cpu if it
    fails -- reproducing (without any real torch/transformers install or
    GPU) the google/flan-t5-small + torch==1.13 + MPS failure mode
    discovered during FilterRAG epsilon calibration: MPS in that torch
    version doesn't implement int64 abs() for T5's relative-position-bias
    attention, so every single SLM call failed and was silently swallowed
    by local_hf_slm_answer_fn(), making `filterrag` and
    `filterrag_query_only` produce byte-identical scores."""

    def setUp(self):
        import defense.filterrag as filterrag_module

        self.filterrag_module = filterrag_module
        self._orig_cache = dict(filterrag_module._SLM_PIPELINE_CACHE)
        self._orig_device_logged = filterrag_module._SLM_DEVICE_LOGGED
        self._orig_answer_logged = filterrag_module._SLM_ANSWER_FAILURE_LOGGED
        filterrag_module._SLM_PIPELINE_CACHE.clear()
        filterrag_module._SLM_DEVICE_LOGGED = False
        filterrag_module._SLM_ANSWER_FAILURE_LOGGED = False

    def tearDown(self):
        self.filterrag_module._SLM_PIPELINE_CACHE.clear()
        self.filterrag_module._SLM_PIPELINE_CACHE.update(self._orig_cache)
        self.filterrag_module._SLM_DEVICE_LOGGED = self._orig_device_logged
        self.filterrag_module._SLM_ANSWER_FAILURE_LOGGED = self._orig_answer_logged

    @staticmethod
    def _fake_transformers_module(*, failing_devices):
        """Fake `transformers` module: pipeline() returns an object that
        raises on __call__ if built for one of `failing_devices`, else
        returns a canned generated_text."""
        fake_transformers = types.ModuleType("transformers")

        class _FakePipe:
            def __init__(self, device):
                self.device = device

            def __call__(self, prompt, **kwargs):
                if self.device in failing_devices:
                    raise TypeError(
                        "Operation 'abs_out_mps()' does not support input type 'int64' in MPS backend."
                    )
                return [{"generated_text": "Paris"}]

        fake_transformers.pipeline = lambda task, model=None, device=None: _FakePipe(device)
        return fake_transformers

    def test_falls_back_to_cpu_when_mps_generation_fails(self):
        fake_torch = _fake_torch_module(mps_available=True, cuda_available=False)
        fake_transformers = self._fake_transformers_module(failing_devices={"mps"})
        with mock.patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
            pipe = self.filterrag_module._get_local_hf_slm_pipeline("fake-model", device="mps")
        self.assertEqual(pipe.device, "cpu")
        self.assertIn(("fake-model", "cpu"), self.filterrag_module._SLM_PIPELINE_CACHE)

    def test_no_fallback_when_device_actually_works(self):
        fake_torch = _fake_torch_module(mps_available=True, cuda_available=False)
        fake_transformers = self._fake_transformers_module(failing_devices=set())
        with mock.patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
            pipe = self.filterrag_module._get_local_hf_slm_pipeline("fake-model", device="mps")
        self.assertEqual(pipe.device, "mps")

    def test_cpu_device_never_smoke_tested(self):
        # cpu is the trusted baseline -- no probe call should run against it,
        # so even a pipe that would fail on cpu is returned unchanged
        # (this only matters for this synthetic test; real cpu pipelines
        # don't hit the MPS-specific op-support failure).
        fake_torch = _fake_torch_module(mps_available=False, cuda_available=False)
        fake_transformers = self._fake_transformers_module(failing_devices={"cpu"})
        with mock.patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
            pipe = self.filterrag_module._get_local_hf_slm_pipeline("fake-model", device="cpu")
        self.assertEqual(pipe.device, "cpu")


class TestLocalHfSlmAnswerFnFailureLogging(unittest.TestCase):
    """Per-passage SLM failures must degrade to `None` (query-only keywords
    for that one passage) but log a warning at least once -- not fail
    completely silently, which is what let the MPS bug above go unnoticed."""

    def setUp(self):
        import defense.filterrag as filterrag_module

        self.filterrag_module = filterrag_module
        self._orig_cache = dict(filterrag_module._SLM_PIPELINE_CACHE)
        self._orig_device_logged = filterrag_module._SLM_DEVICE_LOGGED
        self._orig_answer_logged = filterrag_module._SLM_ANSWER_FAILURE_LOGGED
        filterrag_module._SLM_PIPELINE_CACHE.clear()
        filterrag_module._SLM_DEVICE_LOGGED = False
        filterrag_module._SLM_ANSWER_FAILURE_LOGGED = False

    def tearDown(self):
        self.filterrag_module._SLM_PIPELINE_CACHE.clear()
        self.filterrag_module._SLM_PIPELINE_CACHE.update(self._orig_cache)
        self.filterrag_module._SLM_DEVICE_LOGGED = self._orig_device_logged
        self.filterrag_module._SLM_ANSWER_FAILURE_LOGGED = self._orig_answer_logged

    def test_per_passage_failure_degrades_to_none_and_logs_once(self):
        fake_torch = _fake_torch_module(mps_available=False, cuda_available=False)
        fake_transformers = types.ModuleType("transformers")

        class _AlwaysFailingPipe:
            def __call__(self, prompt, **kwargs):
                raise RuntimeError("boom")

        fake_transformers.pipeline = lambda task, model=None, device=None: _AlwaysFailingPipe()

        with mock.patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
            answer_fn = self.filterrag_module.local_hf_slm_answer_fn("fake-model", device="cpu")
            with mock.patch("builtins.print") as mock_print:
                result1 = answer_fn("Q1?", "passage1")
                result2 = answer_fn("Q2?", "passage2")

        self.assertIsNone(result1)
        self.assertIsNone(result2)
        warning_calls = [c for c in mock_print.call_args_list if "WARNING" in str(c)]
        self.assertEqual(len(warning_calls), 1)  # logged once, not once per failure


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
