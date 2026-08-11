"""Tests for scripts/run_text_mutation_fixed_context_eval.py -- the fixed-
retrieval manual-mutation-bundle evaluator for the HotpotQA text-mutation
pilot.

Fully offline: no real sentence-transformers/flan-t5/distilgpt2 model is
downloaded, and no LLM/GPT/API call is ever made anywhere in this file.
Heavy model calls (`local_hf_slm_answer_fn`, `get_causal_lm_scorer`,
`get_slm_model_and_tokenizer`, `load_classifier_cached`,
`get_semantic_word_matcher`, `defense_runner._get_s_model`) are all
monkeypatched with dependency-free fakes, mirroring
`tests/test_dispatch_smoke.py`'s existing convention.

Run with: python -m unittest tests.test_run_text_mutation_fixed_context_eval -v
"""
import hashlib
import inspect
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import torch  # noqa: E402

from defense import defense_runner, filterrag as filterrag_module, ml_filterrag as ml_filterrag_module  # noqa: E402
from defense.passages import RetrievedPassage  # noqa: E402

import run_text_mutation_fixed_context_eval as m  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _no_clean_context_style_bundle_json(query_id="q1"):
    """Mirrors mutation_bundles_no_clean_context.jsonl.txt's actual shape:
    a single pretty-printed JSON array of per-query records."""
    return json.dumps([
        {
            "query_id": query_id,
            "bundles": [
                {
                    "bundle_id": "gpt_b01",
                    "mutation_strategy": "test strategy",
                    "mutated_passages": [
                        {"poison_rank": i, "mutated_text": f"mutated poison text {i}"}
                        for i in range(5)
                    ],
                }
            ],
        }
    ], indent=2)


def _clean_context_style_bundle_jsonl(query_id="q1"):
    """Mirrors mutation_bundles_clean_context.jsonl.txt's actual shape: true
    JSON-Lines, one compact JSON object per line."""
    record = {
        "query_id": query_id,
        "bundles": [
            {
                "bundle_id": "gpt_b01",
                "mutation_strategy": "test strategy (clean-context-aware)",
                "mutated_passages": [
                    {"poison_rank": i, "mutated_text": f"clean-aware mutated poison text {i}"}
                    for i in range(5)
                ],
            }
        ],
    }
    return json.dumps(record) + "\n"


def _write_temp(contents: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".jsonl.txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(contents)
    return path


def _poison_rows(query_id="q1"):
    return [
        {
            "query_id": query_id, "poison_slot": str(i), "retrieved_rank": str(i + 1),
            "doc_id": f"adv::LM_targeted::{query_id}::{i}",
            "original_poison_text": f"original poison text {i}",
        }
        for i in range(5)
    ]


def _clean_rows(query_id="q1", n=5):
    return [
        {
            "query_id": query_id, "retrieved_rank": str(i + 6), "doc_id": f"clean{i}",
            "clean_text": f"original clean text {i}",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. Bundle file parsing (both formats).
# ---------------------------------------------------------------------------

class TestParseMutationBundleFile(unittest.TestCase):
    def test_parses_json_array_format(self):
        path = _write_temp(_no_clean_context_style_bundle_json("qA"))
        try:
            records = m.parse_mutation_bundle_file(path)
        finally:
            os.remove(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["query_id"], "qA")
        self.assertEqual(len(records[0]["bundles"][0]["mutated_passages"]), 5)

    def test_parses_true_jsonl_format(self):
        contents = _clean_context_style_bundle_jsonl("qB") + _clean_context_style_bundle_jsonl("qC")
        path = _write_temp(contents)
        try:
            records = m.parse_mutation_bundle_file(path)
        finally:
            os.remove(path)
        self.assertEqual([r["query_id"] for r in records], ["qB", "qC"])

    def test_empty_file_raises(self):
        path = _write_temp("   \n")
        try:
            with self.assertRaises(ValueError):
                m.parse_mutation_bundle_file(path)
        finally:
            os.remove(path)

    def test_malformed_jsonl_line_raises(self):
        path = _write_temp('{"query_id": "qA", "bundles": []}\nnot json\n')
        try:
            with self.assertRaises(ValueError):
                m.parse_mutation_bundle_file(path)
        finally:
            os.remove(path)

    def test_load_bundle_file_returns_dict_keyed_by_query_id(self):
        path = _write_temp(_no_clean_context_style_bundle_json("qA"))
        try:
            by_qid = m.load_bundle_file(path)
        finally:
            os.remove(path)
        self.assertEqual(set(by_qid.keys()), {"qA"})
        self.assertEqual(by_qid["qA"]["bundles"][0]["bundle_id"], "gpt_b01")


# ---------------------------------------------------------------------------
# 2. Schema validation.
# ---------------------------------------------------------------------------

class TestValidateBundleRecords(unittest.TestCase):
    def _valid_record(self):
        return json.loads(_no_clean_context_style_bundle_json("qA"))

    def test_valid_record_passes(self):
        m.validate_bundle_records(self._valid_record(), source_path="test.jsonl")  # no raise

    def test_missing_query_id_raises(self):
        rec = self._valid_record()
        del rec[0]["query_id"]
        with self.assertRaises(ValueError):
            m.validate_bundle_records(rec, source_path="test.jsonl")

    def test_wrong_mutated_passage_count_raises(self):
        rec = self._valid_record()
        rec[0]["bundles"][0]["mutated_passages"].pop()  # now only 4
        with self.assertRaises(ValueError):
            m.validate_bundle_records(rec, source_path="test.jsonl")

    def test_duplicate_poison_rank_raises(self):
        rec = self._valid_record()
        rec[0]["bundles"][0]["mutated_passages"][1]["poison_rank"] = 0  # duplicate of index 0
        with self.assertRaises(ValueError):
            m.validate_bundle_records(rec, source_path="test.jsonl")

    def test_empty_mutated_text_raises(self):
        rec = self._valid_record()
        rec[0]["bundles"][0]["mutated_passages"][0]["mutated_text"] = "   "
        with self.assertRaises(ValueError):
            m.validate_bundle_records(rec, source_path="test.jsonl")

    def test_missing_bundle_id_raises(self):
        rec = self._valid_record()
        del rec[0]["bundles"][0]["bundle_id"]
        with self.assertRaises(ValueError):
            m.validate_bundle_records(rec, source_path="test.jsonl")


# ---------------------------------------------------------------------------
# 3. Fixed-context reconstruction: clean preserved, only poison text changes,
#    k=10 membership identical between original and mutated context.
# ---------------------------------------------------------------------------

class TestBuildContexts(unittest.TestCase):
    def test_original_context_is_ordered_by_retrieved_rank_and_has_k10(self):
        ctx = m.build_original_context(_poison_rows(), _clean_rows())
        self.assertEqual(len(ctx), 10)
        self.assertEqual([p.rank for p in ctx], list(range(10)))
        self.assertEqual([p.is_poison for p in ctx], [True] * 5 + [False] * 5)
        self.assertEqual(ctx[0].text, "original poison text 0")
        self.assertEqual(ctx[9].text, "original clean text 4")

    def test_missing_poison_text_raises(self):
        rows = _poison_rows()
        rows[0]["original_poison_text"] = ""
        with self.assertRaises(ValueError):
            m.build_original_context(rows, _clean_rows())

    def test_mutated_context_changes_only_poison_text(self):
        poison_rows = _poison_rows()
        clean_rows = _clean_rows()
        original = m.build_original_context(poison_rows, clean_rows)
        bundle = json.loads(_no_clean_context_style_bundle_json("q1"))[0]["bundles"][0]
        mutated = m.build_mutated_context(original, poison_rows, bundle)

        # doc_id/order/membership identical.
        m.assert_same_k10_membership(original, mutated)

        # Every clean passage byte-identical (text, doc_id, rank, is_poison).
        for orig, mut in zip(original, mutated):
            if not orig.is_poison:
                self.assertEqual(orig.text, mut.text)
                self.assertEqual(orig.doc_id, mut.doc_id)
                self.assertEqual(orig.rank, mut.rank)

        # Every poison passage's text changed to the bundle's mutated_text;
        # doc_id/rank/is_poison unchanged.
        for orig, mut in zip(original, mutated):
            if orig.is_poison:
                self.assertNotEqual(orig.text, mut.text)
                self.assertTrue(mut.text.startswith("mutated poison text"))
                self.assertEqual(orig.doc_id, mut.doc_id)
                self.assertEqual(orig.rank, mut.rank)
                self.assertTrue(mut.is_poison)

    def test_membership_mismatch_is_detected(self):
        a = [RetrievedPassage(doc_id="x", text="t", source="corpus", is_poison=False, rank=0)]
        b = [RetrievedPassage(doc_id="y", text="t", source="corpus", is_poison=False, rank=0)]
        with self.assertRaises(AssertionError):
            m.assert_same_k10_membership(a, b)


# ---------------------------------------------------------------------------
# 4. Delta computation + literal-name aliases.
# ---------------------------------------------------------------------------

class TestComputeDeltas(unittest.TestCase):
    def test_basic_arithmetic_and_none_handling(self):
        baseline = {
            "ml_removed_poison_t04": 5, "ragdefender_top_pair_pp": 10,
            "ml_mean_poison_probability": 0.9, "filterrag_mean_freq_density_poison": 1.2,
            "filterrag_mean_matched_freq_sum_poison": None,
        }
        bundle = {
            "ml_removed_poison_t04": 3, "ragdefender_top_pair_pp": 6,
            "ml_mean_poison_probability": 0.4, "filterrag_mean_freq_density_poison": 0.8,
            "filterrag_mean_matched_freq_sum_poison": None,
        }
        deltas = m.compute_deltas(baseline, bundle)
        self.assertAlmostEqual(deltas["delta_ml_removed_poison_t04"], -2)
        self.assertAlmostEqual(deltas["delta_ragdefender_top_pair_pp"], -4)
        self.assertIsNone(deltas["delta_filterrag_mean_matched_freq_sum_poison"])

    def test_literal_aliases_map_to_documented_targets(self):
        baseline = {k: 1.0 for k in m._NUMERIC_METRIC_KEYS}
        bundle = {k: 0.0 for k in m._NUMERIC_METRIC_KEYS}
        deltas = m.compute_deltas(baseline, bundle)
        for alias, target in m.DELTA_ALIASES.items():
            self.assertEqual(deltas[alias], deltas[target], msg=f"{alias} != {target}")


# ---------------------------------------------------------------------------
# 5. Static checks: no GPT/API calls, no retrieval calls.
# ---------------------------------------------------------------------------

class TestNoForbiddenCalls(unittest.TestCase):
    def test_no_forbidden_api_modules_are_imported(self):
        """Parses the module's AST and checks that no `import`/`from ... import`
        statement ever names a forbidden LLM/API package (openai, anthropic,
        google.generativeai) -- a structural check that can't false-positive
        on the module's own prose disclaiming those same calls (unlike a
        naive whole-source substring search)."""
        import ast

        tree = ast.parse(inspect.getsource(m))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)

        forbidden_modules = ("openai", "anthropic", "google.generativeai")
        for forbidden in forbidden_modules:
            self.assertFalse(
                any(name == forbidden or name.startswith(forbidden + ".") for name in imported_names),
                msg=f"forbidden module {forbidden!r} is imported: {imported_names!r}",
            )

    def test_no_requests_post_or_llm_query_calls(self):
        """Parses the module's AST and checks that no `ast.Call` node's
        callee is `requests.post(...)` or a `*.query(...)` call whose
        receiver is literally named/aliased `llm` -- catches real invocation
        syntax while ignoring prose that merely *mentions* `llm.query()` to
        disclaim it (e.g. this module's own docstring and generated
        report text)."""
        import ast

        tree = ast.parse(inspect.getsource(m))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            attr = node.func
            method_name = attr.attr
            receiver = attr.value
            receiver_name = receiver.id if isinstance(receiver, ast.Name) else None
            self.assertFalse(
                method_name == "post" and receiver_name == "requests",
                msg="found a requests.post(...) call",
            )
            self.assertFalse(
                method_name == "query" and receiver_name == "llm",
                msg="found an llm.query(...) call",
            )

    def test_source_does_not_import_retrieval_pipeline_symbols(self):
        source = inspect.getsource(m)
        for forbidden in ("load_beir_datasets", "from src.attack import", "import src.attack"):
            self.assertNotIn(forbidden, source)

    def test_offline_env_vars_are_forced(self):
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")


# ---------------------------------------------------------------------------
# 6. Fully-mocked end-to-end scoring smoke test.
# ---------------------------------------------------------------------------

class FakeSentenceTransformer:
    """Deterministic, dependency-free stand-in for SentenceTransformer,
    duplicated from tests/test_dispatch_smoke.py's helper of the same name
    (kept self-contained rather than imported, matching that file's own
    convention)."""

    def encode(self, text_list, convert_to_tensor=True):
        vectors = []
        for t in text_list:
            digest = hashlib.md5(t.encode("utf-8")).hexdigest()
            seed = int(digest[:8], 16)
            gen = torch.Generator().manual_seed(seed)
            vectors.append(torch.rand(16, generator=gen))
        return torch.stack(vectors)


class FakeSemanticMatcher:
    def __init__(self, similarities=None):
        self.similarities = similarities or {}

    def similarity_matrix(self, words_a, words_b):
        return [[self.similarities.get((wa, wb), 0.0) for wb in words_b] for wa in words_a]


class FakeCausalScorer:
    def perplexity(self, text):
        return 10.0


class FakeClassifier:
    feature_names = ml_filterrag_module.DEFAULT_FEATURE_NAMES
    model_type = "random_forest"
    training_meta = {}
    threshold_default = 0.5

    def predict_proba(self, X):
        import numpy as np

        # Deterministic: flag every row whose freq_density_score feature > 0.5.
        idx = list(ml_filterrag_module.DEFAULT_FEATURE_NAMES).index("freq_density_score")
        return np.array([1.0 if row[idx] > 0.5 else 0.0 for row in X])


class TestScoreContextOfflineSmoke(unittest.TestCase):
    """Exercises score_context() end-to-end with every heavy model call
    faked, proving the wiring (module/attribute names, dict keys) is
    correct before any real (slow) model is ever loaded."""

    def setUp(self):
        self.s_model_patcher = mock.patch.object(
            defense_runner, "_get_s_model", return_value=FakeSentenceTransformer()
        )
        self.s_model_patcher.start()
        self.addCleanup(self.s_model_patcher.stop)

        self.filterrag_matcher_patcher = mock.patch.object(
            filterrag_module, "get_semantic_word_matcher", return_value=FakeSemanticMatcher()
        )
        self.filterrag_matcher_patcher.start()
        self.addCleanup(self.filterrag_matcher_patcher.stop)

        self.ml_matcher_patcher = mock.patch.object(
            ml_filterrag_module, "get_semantic_word_matcher", return_value=FakeSemanticMatcher()
        )
        self.ml_matcher_patcher.start()
        self.addCleanup(self.ml_matcher_patcher.stop)

    def _fake_models(self):
        return m.Models(
            memo_slm_answer_fn=m.MemoizedSlmAnswerFn(lambda q, p: "a fake slm answer"),
            memo_causal_scorer=m.MemoizedCausalLMScorer(FakeCausalScorer()),
            slm_logprob_model=None,
            slm_logprob_tokenizer=None,
            classifier=FakeClassifier(),
        )

    def _passages(self):
        return [
            RetrievedPassage(doc_id="adv0", text="poisoned passage stuffed with keywords", source="adversarial", is_poison=True, rank=0),
            RetrievedPassage(doc_id="adv1", text="another poisoned passage with keywords", source="adversarial", is_poison=True, rank=1),
            RetrievedPassage(doc_id="clean0", text="an unrelated clean fact about the world", source="corpus", is_poison=False, rank=2),
            RetrievedPassage(doc_id="clean1", text="another unrelated clean fact", source="corpus", is_poison=False, rank=3),
        ]

    def test_score_context_runs_offline_and_has_expected_keys(self):
        models = self._fake_models()
        metrics = m.score_context("a test question?", self._passages(), models)

        expected_keys = {
            "N_retrieved_poison", "N_retrieved_clean",
            "ragdefender_removed_poison", "ragdefender_removed_clean",
            "ragdefender_residual_poison_fraction", "ragdefender_top_pair_pp",
            "ragdefender_top_pair_pc", "ragdefender_top_pair_cc",
            "filterrag_removed_poison", "filterrag_removed_clean",
            "filterrag_residual_poison_fraction",
            "ml_mean_poison_probability", "ml_removed_poison_t04",
            "ml_removed_clean_t04", "ml_residual_poison_fraction_t04",
            "ml_removed_poison_t035", "ml_removed_poison_t05",
        }
        self.assertTrue(expected_keys.issubset(metrics.keys()), msg=sorted(expected_keys - metrics.keys()))
        self.assertEqual(metrics["N_retrieved_poison"], 2)
        self.assertEqual(metrics["N_retrieved_clean"], 2)

    def test_score_context_is_deterministic_across_repeated_calls(self):
        models = self._fake_models()
        passages = self._passages()
        m1 = m.score_context("a test question?", passages, models)
        m2 = m.score_context("a test question?", passages, models)
        self.assertEqual(m1["ml_removed_poison_t04"], m2["ml_removed_poison_t04"])
        self.assertEqual(m1["ragdefender_top_pair_pp"], m2["ragdefender_top_pair_pp"])

    def test_memoized_slm_fn_caches_repeated_identical_calls(self):
        base_calls = {"n": 0}

        def base_fn(q, p):
            base_calls["n"] += 1
            return "answer"

        memo = m.MemoizedSlmAnswerFn(base_fn)
        memo("q", "same text")
        memo("q", "same text")
        memo("q", "different text")
        self.assertEqual(base_calls["n"], 2)
        self.assertEqual(memo.cache_hits, 1)


if __name__ == "__main__":
    unittest.main()
