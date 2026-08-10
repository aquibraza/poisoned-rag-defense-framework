"""Regression tests for a real bug: scripts/evaluate_ml_filterrag.py's
`--held_out_config` handling used to reconstruct the adversarial candidate
pool as *only* the held-out `test_query_ids`, instead of the *full* pool
(every `query_id` used by the build) that
`scripts/build_ml_filterrag_dataset.py::build_feature_rows()` actually used.

Why this matters (see both scripts' module/function docstrings for the full
writeup): for a given `attack_method`, `Attacker.get_attack(target_queries)`
is called *once* with every pool query, and every resulting adversarial text
is scored against *every* pool query's own embedding when building that
query's `merged_results` -- so a query's retrieved top-k can legitimately
include another pool query's adversarial text. Evaluating with a narrower
pool than the one used to build the dataset therefore silently changes
which passages are retrieved/labeled poison for the held-out queries,
relative to that dataset's own `features.csv` -- exactly what was observed
(150 test rows: 118 poison/32 clean in the dataset vs. 113 poison/37 clean
in the held-out evaluation of the *same* query_ids/k values).

Fully offline: `load_beir_datasets`, `load_json`, `load_models`, `Attacker`
are all faked/mocked (no real BEIR corpus, HF model, or `Attacker` is ever
loaded), and `builtins.open` is patched only for the specific in-memory
fake file paths this test cares about (real `open()` is used for
everything else via the fallback in `_patched_open_for_path`). No GPT/API
call, no `llm.query()` call, no live generation anywhere in this file.

Run with: python -m unittest tests.test_ml_filterrag_dataset_eval_consistency -v
"""
import builtins
import contextlib
import json
import os
import sys
import types
import unittest
from collections import defaultdict
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.build_ml_filterrag_dataset as build_script  # noqa: E402
import scripts.evaluate_ml_filterrag as eval_script  # noqa: E402
from defense.ml_filterrag import query_level_train_test_split  # noqa: E402


class _NoOpModel:
    """Stand-in for the contriever query/doc encoder `nn.Module`s -- only
    `.eval()`/`.to(device)` are ever called on them directly by these
    scripts; all real embedding math is bypassed by `_make_fake_get_emb`."""

    def eval(self):
        return self

    def to(self, device):
        return self


class _TextBatch(list):
    """Fake tokenizer output: a plain list of raw strings that also
    supports `.to(device)` (a no-op) so `{k: v.to(device) for k, v in
    encoder_inputs.items()}` (used verbatim by both scripts) works
    unchanged."""

    def to(self, device):
        return self


def _fake_tokenizer(text, padding=True, truncation=True, return_tensors="pt"):
    texts = text if isinstance(text, list) else [text]
    return {"texts": _TextBatch(texts)}


def _make_fake_get_emb(emb_table):
    """Deterministic fake `get_emb(model, inputs)`: looks up each input
    text's scalar embedding value in `emb_table` (default 1.0 for any text
    not explicitly registered, e.g. every query's own question text -- see
    module docstring's "uniform query embedding" scheme), returning a
    (batch, 1) float tensor. With `score_function="dot"`, this makes
    `adv_sim == emb_table[adv_text]` directly (query embedding is always
    1.0), so every test can fully control retrieval ranking just by
    choosing `emb_table` values."""

    def _get_emb(model, inputs):
        texts = inputs["texts"]
        values = [[emb_table.get(t, 1.0)] for t in texts]
        return torch.tensor(values, dtype=torch.float32)

    return _get_emb


def _make_fake_load_models(emb_table):
    def _load_models(model_code):
        return _NoOpModel(), _NoOpModel(), _fake_tokenizer, _make_fake_get_emb(emb_table)

    return _load_models


class _FakeAttacker:
    """Deterministic, fully-offline stand-in for `src.attack.Attacker`:
    generates `adv_per_query` adversarial "texts" per target query, named
    `ADVTEXT::<query_id>::<j>` so a test can pin down each one's embedding
    value via `emb_table` (see `_make_fake_get_emb`). Mirrors the real
    `Attacker.get_attack()`'s (query_id-batched, N-per-query) return shape."""

    def __init__(self, attack_args, **kwargs):
        self._n = attack_args.adv_per_query

    def get_attack(self, target_queries):
        return [[f"ADVTEXT::{q['id']}::{j}" for j in range(self._n)] for q in target_queries]


class _FakeCausalScorer:
    def perplexity(self, text):
        return 1.0


@contextlib.contextmanager
def _patched_open_for_path(path_to_content):
    """Patch `builtins.open` so reading any of `path_to_content`'s (exact,
    normalized) paths returns that in-memory content, while every other
    path falls through to the real `open()` unchanged -- avoids writing
    any fixture file to the real repo tree for these two scripts' bare
    `with open(beir_results_path) as f: json.load(f)` / `--held_out_config`
    reads (neither goes through a mockable helper function)."""
    real_open = builtins.open
    normalized = {os.path.normpath(str(p)): c for p, c in path_to_content.items()}

    def _fake_open(path, *args, **kwargs):
        key = os.path.normpath(str(path))
        if key in normalized:
            return mock.mock_open(read_data=normalized[key])()
        return real_open(path, *args, **kwargs)

    with mock.patch("builtins.open", side_effect=_fake_open):
        yield


def _fake_run_defense(defense_name, query, passages, dataset, **kwargs):
    """Passthrough fake for defense.dispatch.run_defense: this test suite
    only cares about *retrieved* passage composition (is_poison counts),
    which build_diagnostic_record derives from `retrieved_passages`
    regardless of what a defense kept/removed -- so no real ml_filterrag
    classifier or FilterRAG SLM needs to be loaded here."""
    return list(passages), {}


class _FixtureMixin:
    """Shared 6-query-id fixture: 4 "train" + 2 "test" query_ids (per a
    real `query_level_train_test_split()` call, not hand-simulated), with
    adversarial-text embedding values engineered so that one specific
    *train* query_id's adversarial text scores higher than a "clean" corpus
    document for both held-out test queries -- i.e. a real, non-degenerate
    case of cross-query pool contamination reaching from train into test,
    which only a *correct* full-pool reconstruction will reproduce."""

    def _make_fixture(self):
        qids = [f"Q{i}" for i in range(6)]
        train_ids, test_ids = query_level_train_test_split(qids, test_fraction=1 / 3, seed=12)
        test_sorted = sorted(test_ids)
        train_sorted = sorted(train_ids)
        assert len(test_sorted) == 2 and len(train_sorted) == 4, (test_sorted, train_sorted)
        q_test_0, q_test_1 = test_sorted
        q_train_leak = train_sorted[0]
        other_train_ids = train_sorted[1:]

        # Own adversarial texts score highest for their own query; one
        # train query_id's ("q_train_leak") first adversarial text (0.6)
        # is engineered to beat corpus doc "docA" (0.5, clean) but lose to
        # every query's own top adversarial texts -- so it changes the
        # *composition* of the k=3 top-k (displacing a clean passage) only
        # when it's actually present in the pool.
        adv_values_by_qid = {
            q_test_0: [0.9, 0.8],
            q_test_1: [0.06, 0.04],
            q_train_leak: [0.6, 0.02],
        }
        for tid in other_train_ids:
            adv_values_by_qid[tid] = [0.02, 0.01]

        emb_table = {}
        for qid, values in adv_values_by_qid.items():
            for j, v in enumerate(values):
                emb_table[f"ADVTEXT::{qid}::{j}"] = v

        incorrect_answers = [{"id": qid, "question": f"Question for {qid}?"} for qid in qids]
        incorrect_answers_dict = {ia["id"]: ia for ia in incorrect_answers}
        corpus = {"docA": {"text": "CORPUS_TEXT_docA"}, "docB": {"text": "CORPUS_TEXT_docB"}}
        beir_results = {qid: {"docA": 0.5, "docB": 0.05} for qid in qids}

        return {
            "qids": qids,
            "q_test_0": q_test_0,
            "q_test_1": q_test_1,
            "q_train_leak": q_train_leak,
            "incorrect_answers": incorrect_answers,
            "incorrect_answers_dict": incorrect_answers_dict,
            "corpus": corpus,
            "beir_results": beir_results,
            "emb_table": emb_table,
        }

    def _common_args(self, fixture, **overrides):
        args = types.SimpleNamespace(
            eval_dataset="fake_ml_filterrag_ds",
            eval_model_code="fake_model",
            split="test",
            score_function="dot",
            k_values=[2, 3],
            N=2,
            attack_methods=["FAKE_ATTACK"],
            max_queries=len(fixture["qids"]),
            filterrag_slm_model="fake-slm",
            filterrag_slm_device="cpu",
            ml_filterrag_matching_mode="exact",
            ml_filterrag_semantic_threshold=0.6,
            ml_filterrag_lm_model="fake-lm",
            test_fraction=1 / 3,
            split_seed=12,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    @property
    def _beir_results_path(self):
        return "results/beir_results/fake_ml_filterrag_ds-fake_model.json"

    def _build_dataset(self, fixture):
        build_args = self._common_args(fixture)
        with _patched_open_for_path({self._beir_results_path: json.dumps(fixture["beir_results"])}), \
             mock.patch.object(build_script, "load_beir_datasets", return_value=(fixture["corpus"], {}, {})), \
             mock.patch.object(build_script, "load_json", return_value=fixture["incorrect_answers_dict"]), \
             mock.patch.object(build_script, "load_models", side_effect=_make_fake_load_models(fixture["emb_table"])), \
             mock.patch.object(build_script, "Attacker", _FakeAttacker), \
             mock.patch.object(build_script, "local_hf_slm_answer_fn", return_value=(lambda q, p: None)), \
             mock.patch.object(build_script, "get_slm_model_and_tokenizer", return_value=(None, None)), \
             mock.patch.object(build_script, "get_causal_lm_scorer", return_value=_FakeCausalScorer()):
            rows, target_query_ids = build_script.build_feature_rows(build_args)
        return rows, target_query_ids

    def _run_held_out_eval(self, fixture, held_out_config: dict):
        held_out_config_path = "FAKE_HELD_OUT_CONFIG.json"
        eval_args = self._common_args(
            fixture,
            held_out_config=held_out_config_path,
            ml_filterrag_model_path="unused-because-run_defense-is-mocked",
            ml_filterrag_threshold=0.5,
            filterrag_matching_mode="exact",
            filterrag_semantic_threshold=0.6,
            filterrag_epsilon=0.2,
            dry_run=True,
        )
        with _patched_open_for_path({
            self._beir_results_path: json.dumps(fixture["beir_results"]),
            held_out_config_path: json.dumps(held_out_config),
        }), \
             mock.patch.object(eval_script, "load_beir_datasets", return_value=(fixture["corpus"], {}, {})), \
             mock.patch.object(eval_script, "load_json", return_value=fixture["incorrect_answers_dict"]), \
             mock.patch.object(eval_script, "load_models", side_effect=_make_fake_load_models(fixture["emb_table"])), \
             mock.patch.object(eval_script, "Attacker", _FakeAttacker), \
             mock.patch.object(eval_script, "run_defense", side_effect=_fake_run_defense):
            return eval_script.run_evaluation(eval_args)

    @staticmethod
    def _poison_clean_counts_by_query_and_k(rows, query_ids):
        counts = defaultdict(lambda: {"poison": 0, "clean": 0})
        query_ids = set(query_ids)
        for row in rows:
            if row["query_id"] not in query_ids:
                continue
            key = (row["query_id"], row["k"])
            counts[key]["poison" if row["is_poison"] else "clean"] += 1
        return dict(counts)

    @staticmethod
    def _poison_clean_counts_from_records(records):
        return {
            (r["query_id"], r["k"]): {"poison": r["N_retrieved_poison"], "clean": r["N_retrieved_clean"]}
            for r in records
        }


class TestHeldOutEvalMatchesDatasetBuilderCandidatePool(_FixtureMixin, unittest.TestCase):
    def test_held_out_eval_reconstructs_full_pool_and_matches_dataset_csv(self):
        fixture = self._make_fixture()
        rows, target_query_ids = self._build_dataset(fixture)
        self.assertEqual(sorted(target_query_ids), sorted(fixture["qids"]))

        test_qids = [fixture["q_test_0"], fixture["q_test_1"]]
        held_out_config = {
            "target_query_ids": target_query_ids,
            "test_query_ids": test_qids,
            "max_queries": len(fixture["qids"]),
        }
        all_records = self._run_held_out_eval(fixture, held_out_config)

        build_counts = self._poison_clean_counts_by_query_and_k(rows, test_qids)
        eval_counts = self._poison_clean_counts_from_records(all_records["ml_filterrag"])

        self.assertEqual(set(build_counts), set(eval_counts))
        for key in build_counts:
            self.assertEqual(
                eval_counts[key], build_counts[key],
                f"retrieved is_poison composition mismatch for (query_id, k)={key}: "
                f"dataset builder={build_counts[key]!r} vs. held-out evaluator={eval_counts[key]!r}",
            )

        # Sanity: this fixture is deliberately engineered so a *train*
        # query's adversarial text lands in a held-out *test* query's own
        # top-3 -- i.e. the full-pool reconstruction actually mattered here
        # (not a degenerate case the old, narrower-pool code would have
        # gotten right by accident too -- see the companion test below).
        self.assertEqual(build_counts[(fixture["q_test_0"], 3)], {"poison": 3, "clean": 0})
        self.assertEqual(build_counts[(fixture["q_test_1"], 3)], {"poison": 3, "clean": 0})

        # filterrag baseline records are built from the identical
        # retrieved_passages, so they must match too.
        filterrag_counts = self._poison_clean_counts_from_records(all_records["filterrag"])
        self.assertEqual(filterrag_counts, build_counts)

    def test_narrow_held_out_only_pool_reproduces_the_original_bug(self):
        """Companion test proving the fixture above is actually
        discriminating: reconstructing the pool as *just* the held-out
        test_query_ids (the pre-fix behavior) must NOT match the dataset
        builder's counts for this fixture -- if this test ever starts
        failing (i.e. the narrow pool starts matching), the fixture has
        stopped exercising the bug and must be redesigned."""
        fixture = self._make_fixture()
        rows, target_query_ids = self._build_dataset(fixture)
        test_qids = [fixture["q_test_0"], fixture["q_test_1"]]

        # The pre-fix bug: 'target_query_ids' equal to the held-out subset
        # itself, instead of the full build-time pool.
        buggy_held_out_config = {
            "target_query_ids": test_qids,
            "test_query_ids": test_qids,
            "max_queries": len(fixture["qids"]),
        }
        all_records = self._run_held_out_eval(fixture, buggy_held_out_config)

        build_counts = self._poison_clean_counts_by_query_and_k(rows, test_qids)
        eval_counts = self._poison_clean_counts_from_records(all_records["ml_filterrag"])

        self.assertNotEqual(
            eval_counts, build_counts,
            "Expected the narrow (held-out-only) pool to mismatch the dataset builder's "
            "counts for this fixture -- if it no longer does, the fixture must be redesigned "
            "so this regression test still exercises the original bug.",
        )
        # And specifically at k=3, the train-leak passage is simply absent
        # under the narrow pool, so the clean corpus doc survives instead.
        self.assertEqual(eval_counts[(fixture["q_test_0"], 3)], {"poison": 2, "clean": 1})


class TestResolveQueryIdPools(_FixtureMixin, unittest.TestCase):
    """Pure/direct tests of resolve_query_id_pools() -- no retrieval/model
    fakes needed, since this function only reads --held_out_config +
    `incorrect_answers`."""

    def _incorrect_answers(self, qids):
        return [{"id": qid, "question": f"Q for {qid}"} for qid in qids]

    def test_target_query_ids_present_is_used_verbatim_in_order(self):
        incorrect_answers = self._incorrect_answers(["Q2", "Q0", "Q1", "Q3"])
        config_path = "FAKE_CONFIG.json"
        config = {"target_query_ids": ["Q2", "Q0", "Q1", "Q3"], "test_query_ids": ["Q1", "Q3"]}
        args = types.SimpleNamespace(held_out_config=config_path, max_queries=10)
        with _patched_open_for_path({config_path: json.dumps(config)}):
            pool, eval_ids = eval_script.resolve_query_id_pools(args, incorrect_answers)
        self.assertEqual(pool, ["Q2", "Q0", "Q1", "Q3"])
        self.assertEqual(eval_ids, ["Q1", "Q3"])

    def test_old_config_without_target_query_ids_reconstructs_from_max_queries_with_warning(self):
        incorrect_answers = self._incorrect_answers(["Q0", "Q1", "Q2", "Q3", "Q4"])
        config_path = "FAKE_OLD_CONFIG.json"
        config = {"test_query_ids": ["Q3"], "max_queries": 4}  # no 'target_query_ids' -- old format
        args = types.SimpleNamespace(held_out_config=config_path, max_queries=10, eval_dataset="fake_ds")
        with _patched_open_for_path({config_path: json.dumps(config)}), \
             mock.patch("builtins.print") as mock_print:
            pool, eval_ids = eval_script.resolve_query_id_pools(args, incorrect_answers)
        self.assertEqual(pool, ["Q0", "Q1", "Q2", "Q3"])  # first max_queries=4, NOT Q4
        self.assertEqual(eval_ids, ["Q3"])
        warned = any("predates 'target_query_ids'" in str(call) for call in mock_print.call_args_list)
        self.assertTrue(warned, "expected a warning about reconstructing from max_queries")

    def test_old_config_without_target_query_ids_or_max_queries_raises_clear_error(self):
        incorrect_answers = self._incorrect_answers(["Q0", "Q1"])
        config_path = "FAKE_BROKEN_CONFIG.json"
        config = {"test_query_ids": ["Q1"]}  # neither key present
        args = types.SimpleNamespace(held_out_config=config_path, max_queries=10)
        with _patched_open_for_path({config_path: json.dumps(config)}):
            with self.assertRaises(ValueError) as ctx:
                eval_script.resolve_query_id_pools(args, incorrect_answers)
        self.assertIn("target_query_ids", str(ctx.exception))
        self.assertIn("max_queries", str(ctx.exception))

    def test_test_query_ids_not_subset_of_pool_raises(self):
        incorrect_answers = self._incorrect_answers(["Q0", "Q1", "Q2"])
        config_path = "FAKE_INCONSISTENT_CONFIG.json"
        config = {"target_query_ids": ["Q0", "Q1"], "test_query_ids": ["Q2"]}  # Q2 not in pool
        args = types.SimpleNamespace(held_out_config=config_path, max_queries=10)
        with _patched_open_for_path({config_path: json.dumps(config)}):
            with self.assertRaises(ValueError):
                eval_script.resolve_query_id_pools(args, incorrect_answers)

    def test_empty_test_query_ids_raises(self):
        incorrect_answers = self._incorrect_answers(["Q0", "Q1"])
        config_path = "FAKE_EMPTY_TEST_IDS.json"
        config = {"target_query_ids": ["Q0", "Q1"], "test_query_ids": []}
        args = types.SimpleNamespace(held_out_config=config_path, max_queries=10)
        with _patched_open_for_path({config_path: json.dumps(config)}):
            with self.assertRaises(ValueError):
                eval_script.resolve_query_id_pools(args, incorrect_answers)

    def test_no_held_out_config_pool_equals_eval_equals_first_max_queries(self):
        incorrect_answers = self._incorrect_answers(["Q0", "Q1", "Q2", "Q3"])
        args = types.SimpleNamespace(held_out_config=None, max_queries=3)
        pool, eval_ids = eval_script.resolve_query_id_pools(args, incorrect_answers)
        self.assertEqual(pool, ["Q0", "Q1", "Q2"])
        self.assertEqual(eval_ids, ["Q0", "Q1", "Q2"])

    def test_eval_query_ids_truncated_to_max_queries(self):
        incorrect_answers = self._incorrect_answers(["Q0", "Q1", "Q2", "Q3"])
        config_path = "FAKE_CONFIG_TRUNC.json"
        config = {
            "target_query_ids": ["Q0", "Q1", "Q2", "Q3"],
            "test_query_ids": ["Q0", "Q1", "Q2"],
        }
        args = types.SimpleNamespace(held_out_config=config_path, max_queries=2)
        with _patched_open_for_path({config_path: json.dumps(config)}):
            pool, eval_ids = eval_script.resolve_query_id_pools(args, incorrect_answers)
        self.assertEqual(pool, ["Q0", "Q1", "Q2", "Q3"])
        self.assertEqual(len(eval_ids), 2)


class TestWriteConfigJsonTargetQueryIds(unittest.TestCase):
    """write_config_json() must add 'target_query_ids' (original build
    order) while preserving every pre-existing key -- backward
    compatibility for any other reader of dataset_config.json."""

    def test_target_query_ids_written_in_original_order_alongside_existing_keys(self):
        import tempfile

        args = types.SimpleNamespace(
            eval_dataset="hotpotqa", eval_model_code="contriever", split="test",
            score_function="dot", k_values=[5, 10], N=5, attack_methods=["LM_targeted"],
            max_queries=4, filterrag_slm_model="google/flan-t5-small", filterrag_slm_device="auto",
            ml_filterrag_matching_mode="semantic", ml_filterrag_semantic_threshold=0.6,
            ml_filterrag_lm_model="distilgpt2", test_fraction=0.25, split_seed=12,
        )
        rows = [
            {"query_id": "Q2", "split": "train"},
            {"query_id": "Q0", "split": "train"},
            {"query_id": "Q1", "split": "test"},
        ]
        target_query_ids = ["Q2", "Q0", "Q1", "Q3"]  # original build order, NOT sorted

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dataset_config.json")
            build_script.write_config_json(args, rows, target_query_ids, path)
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)

        self.assertEqual(config["target_query_ids"], ["Q2", "Q0", "Q1", "Q3"])
        # Pre-existing keys/behavior unchanged: sorted, derived from rows.
        self.assertEqual(config["train_query_ids"], ["Q0", "Q2"])
        self.assertEqual(config["test_query_ids"], ["Q1"])
        self.assertEqual(config["max_queries"], 4)
        self.assertEqual(config["eval_dataset"], "hotpotqa")
        self.assertTrue(config["no_gpt_api_calls_made"])
        self.assertTrue(config["no_live_generation_through_llm_query"])


if __name__ == "__main__":
    unittest.main()
