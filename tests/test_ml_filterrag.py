"""Tests for defense/ml_filterrag.py -- the "ML-FilterRAG-top-k" MVP
(Edemacu et al. 2025, Algorithm 2 / Section III-B2): feature extraction,
`CausalLMScorer` (perplexity), the joint-log-probability conversion,
`MLFilterRAGClassifier`, and the query-level split helpers.

Fully offline: no HF model is downloaded/loaded, and no LLM/GPT/API call is
ever made anywhere in this file. Perplexity/log-probability scoring is
exercised via dependency-free fakes (`FakeCausalLM`/`FakeCausalTokenizer`
for `CausalLMScorer`, `FakeSeq2SeqModel`/`FakeSeq2SeqTokenizer` for
`slm_answer_joint_logprob`), mirroring `tests/test_filterrag.py`'s existing
fake-`transformers` pattern. `sklearn.ensemble.RandomForestClassifier`
(already a project dependency, see requirements.txt) is used directly for
the classifier round-trip tests -- no fake needed there.

Run with: python -m unittest tests.test_ml_filterrag -v
"""
import math
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defense import ml_filterrag as m
from defense.filterrag import freq_density_detailed
from defense.passages import label_passages


class FakeSemanticMatcher:
    """Dependency-free test double for SemanticWordMatcher (duplicated from
    tests/test_filterrag.py's helper of the same name, per the existing
    self-contained-test-file convention)."""

    def __init__(self, similarities):
        self.similarities = similarities
        self.call_count = 0

    def similarity_matrix(self, words_a, words_b):
        self.call_count += 1
        return [[self.similarities.get((wa, wb), 0.0) for wb in words_b] for wa in words_a]


class FakeCausalLM:
    """Dependency-free stand-in for a HF causal LM: always reports a fixed
    `loss` regardless of input, so perplexity is deterministic and
    computable without downloading any real model."""

    def __init__(self, loss=2.0):
        self.loss = loss

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids=None, labels=None):
        return types.SimpleNamespace(loss=types.SimpleNamespace(item=lambda: self.loss))


class FakeCausalTokenizer:
    """Whitespace-tokenizes and reports a token count via `.shape` on a
    fake tensor -- enough for CausalLMScorer.perplexity() to run without
    torch/transformers actually being invoked."""

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        n = len((text or "").split())

        class _FakeTensor:
            def __init__(self, n):
                self.shape = (1, n)

            def to(self, device):
                return self

        return {"input_ids": _FakeTensor(n)}


class _BatchEncodingLike(dict):
    """Minimal stand-in for HF's BatchEncoding: supports both `**`-unpacking
    (dict) and `.input_ids` attribute access, since
    slm_answer_joint_logprob() uses the latter and extract_features()
    passes encoder_inputs through as `**encoder_inputs`."""

    @property
    def input_ids(self):
        return self["input_ids"]


class _FakeTokenCount:
    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n


class FakeSeq2SeqTokenizer:
    """Whitespace-tokenizes; every call returns a BatchEncoding-like object
    with the right `.numel()`/`**`-unpacking shape for
    slm_answer_joint_logprob(), with no real tokenizer involved."""

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        n = len((text or "").split())
        return _BatchEncodingLike(input_ids=_FakeTokenCount(n))


class FakeSeq2SeqModel:
    """Always reports a fixed mean `loss`, regardless of input -- lets
    slm_answer_joint_logprob()'s `-loss * n` conversion be checked exactly
    against a known value."""

    def __init__(self, loss=0.5):
        self.loss = loss

    def __call__(self, **kwargs):
        return types.SimpleNamespace(loss=types.SimpleNamespace(item=lambda: self.loss))


def make_passages():
    raw = [
        {"doc_id": "clean1", "context": "The Eiffel Tower is located in Paris, France.", "score": 0.9, "source": "corpus", "is_poison": False},
        {"doc_id": "adv1", "context": "texas texas texas texas is the state texas texas texas.", "score": 0.85, "source": "adversarial", "is_poison": True},
    ]
    return label_passages(raw)


class TestFeatureNameConstants(unittest.TestCase):
    def test_default_feature_names_are_exactly_the_four_paper_features(self):
        self.assertEqual(
            m.DEFAULT_FEATURE_NAMES,
            ("freq_density_score", "matched_freq_sum", "perplexity", "slm_answer_logprob"),
        )

    def test_auxiliary_feature_names_do_not_overlap_default(self):
        self.assertEqual(set(m.DEFAULT_FEATURE_NAMES) & set(m.AUXILIARY_FEATURE_NAMES), set())

    def test_all_feature_names_is_default_plus_auxiliary(self):
        self.assertEqual(m.ALL_FEATURE_NAMES, m.DEFAULT_FEATURE_NAMES + m.AUXILIARY_FEATURE_NAMES)


class TestMatchedFreqSumIsAdditive(unittest.TestCase):
    """The one additive change to defense/filterrag.py: matched_freq_sum is
    a new key, and every pre-existing key is untouched."""

    def test_matched_freq_sum_present_alongside_all_pre_existing_keys(self):
        detail = freq_density_detailed("texas is a state with texas cities", ["texas"], matching_mode="exact")
        self.assertIn("matched_freq_sum", detail)
        for key in ("freq_density_score", "unique_word_count", "matched_keyword_count", "matched_keywords", "matching_mode", "semantic_threshold"):
            self.assertIn(key, detail)

    def test_matched_freq_sum_is_the_raw_numerator(self):
        # "texas" appears 2x; unique words = {texas, is, a, state, with, cities} = 6.
        detail = freq_density_detailed("texas is a state with texas cities", ["texas"], matching_mode="exact")
        self.assertEqual(detail["matched_freq_sum"], 2)
        self.assertAlmostEqual(detail["freq_density_score"], 2 / 6)
        self.assertEqual(detail["freq_density_score"] * detail["unique_word_count"], detail["matched_freq_sum"])

    def test_matched_freq_sum_is_zero_for_empty_inputs(self):
        self.assertEqual(freq_density_detailed("", ["texas"])["matched_freq_sum"], 0)
        self.assertEqual(freq_density_detailed("some text", [])["matched_freq_sum"], 0)

    def test_matched_freq_sum_works_in_semantic_mode_too(self):
        matcher = FakeSemanticMatcher({("car", "vehicle"): 0.8})
        detail = freq_density_detailed(
            "car car bus", ["vehicle"], matching_mode="semantic", semantic_threshold=0.6, semantic_matcher=matcher
        )
        self.assertEqual(detail["matched_freq_sum"], 2)  # "car" x2 matched, "bus" not


class TestSemanticFreqDensityIsReusedNotReimplemented(unittest.TestCase):
    def test_extract_features_freq_density_matches_direct_call(self):
        passages = make_passages()
        matcher = FakeSemanticMatcher({("texas", "texas"): 1.0})
        rows = m.extract_features(
            "Where is texas?", passages, slm_answer_fn=None,
            matching_mode="semantic", semantic_threshold=0.6, semantic_matcher=matcher,
            causal_lm_scorer=_fake_causal_scorer(),
        )
        by_id = {r["doc_id"]: r for r in rows}
        for p in passages:
            direct = freq_density_detailed(
                p.text, ["where", "is", "texas"], matching_mode="semantic",
                semantic_threshold=0.6, semantic_matcher=matcher,
            )
            self.assertEqual(by_id[p.doc_id]["freq_density_score"], direct["freq_density_score"])
            self.assertEqual(by_id[p.doc_id]["matched_freq_sum"], direct["matched_freq_sum"])


def _fake_causal_scorer(loss=2.0):
    scorer = m.CausalLMScorer("fake-causal-model")
    scorer._model = FakeCausalLM(loss=loss)
    scorer._tokenizer = FakeCausalTokenizer()
    return scorer


class TestCausalLMScorerPerplexity(unittest.TestCase):
    def test_deterministic_given_fixed_loss(self):
        scorer = _fake_causal_scorer(loss=2.0)
        self.assertAlmostEqual(scorer.perplexity("some passage text here"), math.exp(2.0))

    def test_empty_text_returns_fallback_one(self):
        scorer = _fake_causal_scorer()
        self.assertEqual(scorer.perplexity(""), 1.0)
        self.assertEqual(scorer.perplexity(None), 1.0)

    def test_single_token_returns_fallback_one(self):
        scorer = _fake_causal_scorer()
        self.assertEqual(scorer.perplexity("word"), 1.0)

    def test_perplexity_is_finite_and_positive(self):
        scorer = _fake_causal_scorer(loss=3.5)
        p = scorer.perplexity("a somewhat longer passage of text")
        self.assertTrue(math.isfinite(p))
        self.assertGreater(p, 0.0)

    def test_lazily_loads_model_only_on_first_use(self):
        scorer = m.CausalLMScorer("fake-causal-model")
        self.assertIsNone(scorer._model)


class TestSlmAnswerJointLogprobConversion(unittest.TestCase):
    """Dedicated pinning test for the §3 correction: joint_logprob = -loss * n,
    never the raw mean loss, never a per-token average."""

    def test_exact_conversion_formula(self):
        model = FakeSeq2SeqModel(loss=0.5)
        tokenizer = FakeSeq2SeqTokenizer()
        logprob, n = m.slm_answer_joint_logprob(model, tokenizer, "prompt text", "two words")
        self.assertEqual(n, 2)
        self.assertEqual(logprob, -0.5 * 2)
        self.assertNotEqual(logprob, 0.5)   # never the raw loss
        self.assertNotEqual(logprob, -0.5)  # never just negated, ignoring n

    def test_different_known_loss_and_token_count(self):
        model = FakeSeq2SeqModel(loss=1.25)
        tokenizer = FakeSeq2SeqTokenizer()
        logprob, n = m.slm_answer_joint_logprob(model, tokenizer, "prompt", "one two three four")
        self.assertEqual(n, 4)
        self.assertAlmostEqual(logprob, -1.25 * 4)

    def test_none_answer_returns_zero_and_zero_tokens(self):
        model = FakeSeq2SeqModel(loss=0.5)
        tokenizer = FakeSeq2SeqTokenizer()
        logprob, n = m.slm_answer_joint_logprob(model, tokenizer, "prompt", None)
        self.assertEqual((logprob, n), (0.0, 0))

    def test_empty_answer_returns_zero_and_zero_tokens(self):
        model = FakeSeq2SeqModel(loss=0.5)
        tokenizer = FakeSeq2SeqTokenizer()
        logprob, n = m.slm_answer_joint_logprob(model, tokenizer, "prompt", "   ")
        self.assertEqual((logprob, n), (0.0, 0))


class TestExtractFeaturesFiniteness(unittest.TestCase):
    def setUp(self):
        self._orig_warned = m._MISSING_LOGPROB_MODEL_WARNED
        m._MISSING_LOGPROB_MODEL_WARNED = False

    def tearDown(self):
        m._MISSING_LOGPROB_MODEL_WARNED = self._orig_warned

    def _extract(self, passages, slm_answer_fn=None, **kwargs):
        return m.extract_features(
            "Where is texas?", passages, slm_answer_fn=slm_answer_fn,
            matching_mode="exact", causal_lm_scorer=_fake_causal_scorer(),
            **kwargs,
        )

    def test_all_feature_values_finite_for_normal_passages(self):
        rows = self._extract(make_passages(), slm_answer_fn=lambda q, p: "texas")
        for row in rows:
            for name in m.ALL_FEATURE_NAMES:
                self.assertTrue(math.isfinite(row[name]), f"{name}={row[name]!r} not finite")

    def test_does_not_crash_on_empty_passage(self):
        raw = [{"doc_id": "empty1", "context": "", "score": 0.5, "source": "corpus", "is_poison": False}]
        passages = label_passages(raw)
        rows = self._extract(passages)
        self.assertEqual(len(rows), 1)
        for name in m.ALL_FEATURE_NAMES:
            self.assertTrue(math.isfinite(rows[0][name]))

    def test_does_not_crash_with_no_slm_answer_fn(self):
        rows = self._extract(make_passages(), slm_answer_fn=None)
        for row in rows:
            self.assertIsNone(row["slm_answer"])
            self.assertEqual(row["slm_answer_logprob"], 0.0)
            self.assertEqual(row["slm_answer_length"], 0)

    def test_missing_logprob_model_degrades_to_zero_with_warning(self):
        with mock.patch("builtins.print") as mock_print:
            rows = self._extract(make_passages(), slm_answer_fn=lambda q, p: "texas")
        self.assertTrue(any(r["slm_answer_logprob"] == 0.0 for r in rows))
        self.assertTrue(any("WARNING" in str(c) for c in mock_print.call_args_list))

    def test_retrieval_score_none_defaults_to_zero(self):
        raw = [{"doc_id": "noscore", "context": "some text here", "source": "corpus", "is_poison": False}]
        passages = label_passages(raw)
        rows = self._extract(passages)
        self.assertEqual(rows[0]["retrieval_score"], 0.0)


class TestFeaturesToMatrix(unittest.TestCase):
    def test_default_feature_names_produce_four_columns_in_order(self):
        rows = [
            {"freq_density_score": 1.0, "matched_freq_sum": 2.0, "perplexity": 3.0, "slm_answer_logprob": -4.0},
            {"freq_density_score": 5.0, "matched_freq_sum": 6.0, "perplexity": 7.0, "slm_answer_logprob": -8.0},
        ]
        X = m.features_to_matrix(rows, m.DEFAULT_FEATURE_NAMES)
        self.assertEqual(X.shape, (2, 4))
        self.assertEqual(list(X[0]), [1.0, 2.0, 3.0, -4.0])
        self.assertEqual(list(X[1]), [5.0, 6.0, 7.0, -8.0])

    def test_missing_feature_name_raises_key_error(self):
        rows = [{"freq_density_score": 1.0}]
        with self.assertRaises(KeyError):
            m.features_to_matrix(rows, ["not_a_real_feature"])

    def test_default_matches_default_feature_names_exactly(self):
        row = {name: float(i) for i, name in enumerate(m.ALL_FEATURE_NAMES)}
        X = m.features_to_matrix([row])
        self.assertEqual(X.shape, (1, len(m.DEFAULT_FEATURE_NAMES)))
        expected = [row[name] for name in m.DEFAULT_FEATURE_NAMES]
        self.assertEqual(list(X[0]), expected)


class TestQueryLevelSplitLeakage(unittest.TestCase):
    def test_no_overlap_between_train_and_test(self):
        query_ids = [f"q{i}" for i in range(50)]
        train_ids, test_ids = m.query_level_train_test_split(query_ids, seed=1)
        self.assertEqual(train_ids & test_ids, set())
        m.assert_no_query_id_leakage(train_ids, test_ids)  # must not raise

    def test_split_is_deterministic_for_fixed_seed(self):
        query_ids = [f"q{i}" for i in range(30)]
        train1, test1 = m.query_level_train_test_split(query_ids, seed=7)
        train2, test2 = m.query_level_train_test_split(query_ids, seed=7)
        self.assertEqual(train1, train2)
        self.assertEqual(test1, test2)

    def test_different_seeds_can_produce_different_splits(self):
        query_ids = [f"q{i}" for i in range(30)]
        _, test_a = m.query_level_train_test_split(query_ids, seed=1)
        _, test_b = m.query_level_train_test_split(query_ids, seed=2)
        self.assertNotEqual(test_a, test_b)

    def test_duplicate_query_ids_deduplicated(self):
        query_ids = ["q1", "q1", "q2", "q2", "q3"]
        train_ids, test_ids = m.query_level_train_test_split(query_ids, seed=1)
        self.assertEqual(train_ids | test_ids, {"q1", "q2", "q3"})

    def test_assert_no_leakage_raises_on_overlap(self):
        with self.assertRaises(AssertionError):
            m.assert_no_query_id_leakage({"q1", "q2"}, {"q2", "q3"})

    def test_every_query_id_covered_by_exactly_one_split(self):
        query_ids = [f"q{i}" for i in range(17)]
        train_ids, test_ids = m.query_level_train_test_split(query_ids, seed=3)
        self.assertEqual(train_ids | test_ids, set(query_ids))
        self.assertEqual(train_ids & test_ids, set())


class TestMLFilterRAGClassifier(unittest.TestCase):
    def _toy_data(self):
        import numpy as np

        X = np.array(
            [[0.1, 1, 10, -5], [0.9, 5, 50, -1], [0.15, 1, 11, -6], [0.85, 4, 48, -2], [0.2, 2, 12, -4], [0.8, 4, 45, -3]]
        )
        y = [0, 1, 0, 1, 0, 1]
        return X, y

    def test_invalid_model_type_raises(self):
        with self.assertRaises(ValueError):
            m.MLFilterRAGClassifier(model_type="not_a_real_model_type")

    def test_predict_before_train_raises_clearly(self):
        clf = m.MLFilterRAGClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict_proba([[0.1, 1, 1, 1]])

    def test_train_predict_proba_and_predict_round_trip(self):
        X, y = self._toy_data()
        clf = m.MLFilterRAGClassifier(model_type="random_forest").train(X, y)
        proba = clf.predict_proba(X)
        self.assertEqual(len(proba), len(y))
        for p in proba:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
        preds = clf.predict(X, threshold=0.5)
        self.assertEqual(list(preds), [int(p >= 0.5) for p in proba])

    def test_save_and_load_round_trip_predictions_match(self):
        X, y = self._toy_data()
        clf = m.MLFilterRAGClassifier(model_type="random_forest", training_meta={"dataset": "hotpotqa"}).train(X, y)
        proba_before = list(clf.predict_proba(X))

        path = "/tmp/test_ml_filterrag_artifact.joblib"
        clf.save(path)
        try:
            loaded = m.MLFilterRAGClassifier.load(path)
            proba_after = list(loaded.predict_proba(X))
            self.assertEqual(proba_before, proba_after)
            self.assertEqual(loaded.feature_names, m.DEFAULT_FEATURE_NAMES)
            self.assertEqual(loaded.model_type, "random_forest")
            self.assertEqual(loaded.training_meta, {"dataset": "hotpotqa"})
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_load_missing_path_raises_value_error(self):
        with self.assertRaises(ValueError):
            m.MLFilterRAGClassifier.load(None)
        with self.assertRaises(ValueError):
            m.MLFilterRAGClassifier.load("")

    def test_load_nonexistent_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            m.MLFilterRAGClassifier.load("/tmp/definitely_does_not_exist_ml_filterrag.joblib")

    def test_load_rejects_artifact_missing_required_keys(self):
        import joblib

        path = "/tmp/test_ml_filterrag_bad_artifact.joblib"
        joblib.dump({"model": object(), "model_type": "random_forest"}, path)  # missing feature_names/training_meta
        try:
            with self.assertRaises(ValueError):
                m.MLFilterRAGClassifier.load(path)
        finally:
            os.remove(path)

    def test_xgboost_unavailable_raises_clear_error(self):
        with mock.patch.dict(sys.modules, {"xgboost": None}):
            clf = m.MLFilterRAGClassifier(model_type="xgboost")
            X, y = self._toy_data()
            with self.assertRaises(ImportError):
                clf.train(X, y)

    def test_random_forest_works_without_xgboost_installed(self):
        with mock.patch.dict(sys.modules, {"xgboost": None}):
            X, y = self._toy_data()
            clf = m.MLFilterRAGClassifier(model_type="random_forest").train(X, y)
            self.assertEqual(len(clf.predict_proba(X)), len(y))

    def test_single_class_training_data_predicts_all_zero_proba(self):
        import numpy as np

        X = np.array([[0.1, 1, 10, -5], [0.15, 1, 11, -6], [0.2, 2, 12, -4]])
        y = [0, 0, 0]
        clf = m.MLFilterRAGClassifier(model_type="random_forest").train(X, y)
        proba = clf.predict_proba(X)
        self.assertTrue(all(p == 0.0 for p in proba))


class TestPaperAlignedModelType(unittest.TestCase):
    def test_hotpotqa_and_msmarco_are_random_forest(self):
        self.assertEqual(m.paper_aligned_model_type("hotpotqa"), "random_forest")
        self.assertEqual(m.paper_aligned_model_type("msmarco"), "random_forest")

    def test_nq_is_xgboost(self):
        self.assertEqual(m.paper_aligned_model_type("nq"), "xgboost")

    def test_case_insensitive(self):
        self.assertEqual(m.paper_aligned_model_type("NQ"), "xgboost")

    def test_unknown_dataset_raises(self):
        with self.assertRaises(ValueError):
            m.paper_aligned_model_type("not_a_real_dataset")


class TestMlFilterragDefenseRemoval(unittest.TestCase):
    def _run(self, passages, poison_doc_ids, threshold=0.5):
        class _Clf:
            feature_names = m.DEFAULT_FEATURE_NAMES
            model_type = "random_forest"
            training_meta = {}
            threshold_default = 0.5

            def predict_proba(self, X):
                import numpy as np

                return np.array([1.0 if did in poison_doc_ids else 0.0 for did in _Clf._doc_order])

        # ml_filterrag_defense builds X in the same order as feature_rows,
        # which is the same order as `passages` -- stash that order on the
        # fake classifier so predict_proba can look it up by position.
        _Clf._doc_order = [p.doc_id for p in passages]
        return m.ml_filterrag_defense(
            "Where is texas?", passages, classifier=_Clf(), threshold=threshold,
            slm_answer_fn=None, matching_mode="exact", causal_lm_scorer=_fake_causal_scorer(),
        )

    def test_removes_exactly_the_predicted_poison_passages(self):
        passages = make_passages()
        kept, diag = self._run(passages, poison_doc_ids={"adv1"})
        kept_ids = {p.doc_id for p in kept}
        self.assertEqual(kept_ids, {"clean1"})
        self.assertEqual(diag["N_adv_estimated_by_ragdefender"], 1)

    def test_nothing_removed_when_no_passage_predicted_poison(self):
        passages = make_passages()
        kept, diag = self._run(passages, poison_doc_ids=set())
        self.assertEqual(len(kept), len(passages))
        self.assertEqual(diag["N_adv_estimated_by_ragdefender"], 0)

    def test_diag_extra_has_expected_shape(self):
        passages = make_passages()
        _, diag = self._run(passages, poison_doc_ids={"adv1"})
        for key in ("model_path", "model_artifact_hash", "feature_names", "model_type", "threshold", "matching_mode", "semantic_threshold", "slm_model", "lm_model", "paper_aligned", "notes", "ml_filterrag_predictions"):
            self.assertIn(key, diag)
        self.assertEqual(len(diag["ml_filterrag_predictions"]), len(passages))

    def test_threshold_is_configurable(self):
        passages = make_passages()
        # With a threshold higher than 1.0, nothing crosses it even if
        # flagged poison.
        kept, _ = self._run(passages, poison_doc_ids={"adv1"}, threshold=1.5)
        self.assertEqual(len(kept), len(passages))


class TestNoGptApiImports(unittest.TestCase):
    def test_no_gpt_api_imports_in_ml_filterrag_module(self):
        import inspect

        source = inspect.getsource(m)
        banned = ["openai", "google.generativeai", "anthropic", "cohere"]
        for name in banned:
            self.assertNotIn(name, source, f"found banned API import/reference {name!r} in defense/ml_filterrag.py")

    def test_module_never_calls_llm_query(self):
        """The only occurrences of the literal substring "llm.query(" in
        this module must be the documented backtick-quoted mentions in
        docstrings (explaining the constraint), never an actual call."""
        import inspect

        source = inspect.getsource(m)
        self.assertEqual(source.count("llm.query("), source.count("`llm.query()`"))


if __name__ == "__main__":
    unittest.main()
