"""Tests for scripts/run_normalized_targeted_mutation_bundle_1_eval.py -- the
fixed-retrieval, cross-defense re-evaluator that consumes the *normalized*
`mutation_bundle_1/normalized/*.normalized.jsonl` files (produced by
`scripts/audit_normalize_mutation_bundle_1.py`) instead of the raw
GPT-authored family files.

Fully offline: no real sentence-transformers/flan-t5/distilgpt2 model is
downloaded, and no LLM/GPT/API call is ever made anywhere in this file.

Run with: python -m unittest tests.test_run_normalized_targeted_mutation_bundle_1_eval -v
"""
import inspect
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from defense.passages import RetrievedPassage  # noqa: E402

import run_normalized_targeted_mutation_bundle_1_eval as m  # noqa: E402


def _normalized_record(query_id="q1", doc_id_prefix="canonical", text_fn=None):
    text_fn = text_fn or (lambda slot: f"normalized mutated text {slot}")
    return {
        "query_id": query_id,
        "k": 10,
        "family": "ragdefender_targeted",
        "intended_defense": "ragdefender",
        "selection_role": "primary",
        "question": "Some question?",
        "target_wrong_answer": "wrong answer",
        "mutated_passages": [
            {
                "poison_slot": i,
                "doc_id": f"{doc_id_prefix}::{query_id}::{i}",
                "source_file_doc_id": f"raw_file::{query_id}::{900 + i}",
                "mutated_text": text_fn(i),
                "doc_id_mismatch": False,
                "quality_flags": [],
            }
            for i in range(5)
        ],
    }


def _write_normalized_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _poison_rows(query_id="q1", doc_id_prefix="canonical"):
    return [
        {"query_id": query_id, "poison_slot": str(i), "retrieved_rank": str(i + 1),
         "doc_id": f"{doc_id_prefix}::{query_id}::{i}", "original_poison_text": f"original poison text {i}"}
        for i in range(5)
    ]


def _clean_rows(query_id="q1"):
    return [
        {"query_id": query_id, "retrieved_rank": str(i + 6),
         "doc_id": f"clean_doc_{i}", "clean_text": f"original clean text {i}"}
        for i in range(5)
    ]


# ---------------------------------------------------------------------------
# 1. Normalized files are accepted.
# ---------------------------------------------------------------------------

class TestLoadNormalizedFamily(unittest.TestCase):
    def test_parses_a_valid_normalized_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ragdefender_targeted.normalized.jsonl")
            _write_normalized_jsonl(path, [_normalized_record("q1"), _normalized_record("q2")])
            parsed = m.load_normalized_family(path, "ragdefender_targeted", "ragdefender")
        self.assertEqual(set(parsed), {"q1", "q2"})
        passages = parsed["q1"]["passages"]
        self.assertEqual([p["poison_slot"] for p in passages], [0, 1, 2, 3, 4])
        self.assertEqual(passages[0]["text"], "normalized mutated text 0")
        self.assertEqual(passages[0]["doc_id"], "canonical::q1::0")
        self.assertEqual(passages[0]["source_file_doc_id"], "raw_file::q1::900")

    def test_the_3_real_normalized_files_parse_if_present(self):
        """Regression guard against the actual pilot artifacts, skipped if
        they are not present in this checkout."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        normalized_dir = os.path.join(
            repo_root, "manual_text_mutation_pilot", "hotpotqa_50q_k10", "mutation_bundle_1", "normalized",
        )
        if not os.path.isdir(normalized_dir):
            self.skipTest("real normalized/ directory not present in this checkout")
        for family_key, spec in m.NORMALIZED_FAMILY_SPECS.items():
            path = os.path.join(normalized_dir, spec["normalized_filename"])
            if not os.path.exists(path):
                self.skipTest(f"{path} not present")
            parsed = m.load_normalized_family(path, family_key, spec["intended_defense"])
            self.assertEqual(len(parsed), 6)
            for rec in parsed.values():
                self.assertEqual(len(rec["passages"]), 5)

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "empty.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            with self.assertRaises(ValueError):
                m.load_normalized_family(path, "ragdefender_targeted", "ragdefender")

    def test_missing_query_id_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.jsonl")
            rec = _normalized_record("q1")
            del rec["query_id"]
            _write_normalized_jsonl(path, [rec])
            with self.assertRaises(ValueError):
                m.load_normalized_family(path, "ragdefender_targeted", "ragdefender")

    def test_wrong_passage_count_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.jsonl")
            rec = _normalized_record("q1")
            rec["mutated_passages"] = rec["mutated_passages"][:4]
            _write_normalized_jsonl(path, [rec])
            with self.assertRaises(ValueError):
                m.load_normalized_family(path, "ragdefender_targeted", "ragdefender")

    def test_duplicate_poison_slot_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.jsonl")
            rec = _normalized_record("q1")
            rec["mutated_passages"][1]["poison_slot"] = 0
            _write_normalized_jsonl(path, [rec])
            with self.assertRaises(ValueError):
                m.load_normalized_family(path, "ragdefender_targeted", "ragdefender")

    def test_missing_canonical_doc_id_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.jsonl")
            rec = _normalized_record("q1")
            del rec["mutated_passages"][0]["doc_id"]
            _write_normalized_jsonl(path, [rec])
            with self.assertRaises(ValueError):
                m.load_normalized_family(path, "ragdefender_targeted", "ragdefender")

    def test_empty_mutated_text_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.jsonl")
            rec = _normalized_record("q1")
            rec["mutated_passages"][0]["mutated_text"] = "   "
            _write_normalized_jsonl(path, [rec])
            with self.assertRaises(ValueError):
                m.load_normalized_family(path, "ragdefender_targeted", "ragdefender")


# ---------------------------------------------------------------------------
# 2. Canonical doc_id is preserved / verified against the authoritative CSV.
# ---------------------------------------------------------------------------

class TestValidateCanonicalDocId(unittest.TestCase):
    def test_passes_when_normalized_doc_id_matches_authoritative_csv(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "f.jsonl")
            _write_normalized_jsonl(path, [_normalized_record("q1", doc_id_prefix="canonical")])
            parsed = m.load_normalized_family(path, "ragdefender_targeted", "ragdefender")
        poison_by_query = {"q1": _poison_rows("q1", doc_id_prefix="canonical")}
        # Must not raise.
        m.validate_canonical_doc_id("ragdefender_targeted", parsed, poison_by_query)

    def test_raises_when_normalized_doc_id_does_not_match_authoritative_csv(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "f.jsonl")
            # Normalized file claims a canonical doc_id that disagrees with the CSV --
            # a genuine normalization bug, must be fatal here (unlike the permissive
            # raw-file audit in audit_normalize_mutation_bundle_1.py).
            _write_normalized_jsonl(path, [_normalized_record("q1", doc_id_prefix="wrong_prefix")])
            parsed = m.load_normalized_family(path, "ragdefender_targeted", "ragdefender")
        poison_by_query = {"q1": _poison_rows("q1", doc_id_prefix="canonical")}
        with self.assertRaises(ValueError):
            m.validate_canonical_doc_id("ragdefender_targeted", parsed, poison_by_query)

    def test_raises_when_query_slot_missing_from_authoritative_csv(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "f.jsonl")
            _write_normalized_jsonl(path, [_normalized_record("q_unknown")])
            parsed = m.load_normalized_family(path, "ragdefender_targeted", "ragdefender")
        with self.assertRaises(ValueError):
            m.validate_canonical_doc_id("ragdefender_targeted", parsed, {})


# ---------------------------------------------------------------------------
# 3. Fixed k=10 membership unchanged + clean passages unchanged (reusing
#    base_eval's context-reconstruction primitives, exactly as the previous
#    targeted evaluator does).
# ---------------------------------------------------------------------------

class TestFixedContextReuse(unittest.TestCase):
    def test_k10_membership_unchanged_and_only_poison_text_replaced(self):
        base_eval = m.base_eval
        poison_rows = _poison_rows("q1", doc_id_prefix="canonical")
        clean_rows = _clean_rows("q1")
        original_context = base_eval.build_original_context(poison_rows, clean_rows)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "f.jsonl")
            _write_normalized_jsonl(path, [_normalized_record("q1", doc_id_prefix="canonical")])
            parsed = m.load_normalized_family(path, "ragdefender_targeted", "ragdefender")
        rec = parsed["q1"]
        bundle = m.family_record_to_bundle(rec)

        mutated_context = base_eval.build_mutated_context(original_context, poison_rows, bundle)

        # Fixed k=10 membership/order unchanged.
        base_eval.assert_same_k10_membership(original_context, mutated_context)
        self.assertEqual(len(mutated_context), len(original_context))

        for orig, mut in zip(original_context, mutated_context):
            self.assertEqual(orig.doc_id, mut.doc_id)
            self.assertEqual(orig.rank, mut.rank)
            self.assertEqual(orig.is_poison, mut.is_poison)
            if orig.is_poison:
                self.assertNotEqual(orig.text, mut.text)
                self.assertTrue(mut.text.startswith("normalized mutated text"))
            else:
                # Clean passages remain byte-identical.
                self.assertEqual(orig.text, mut.text)

    def test_family_record_to_bundle_maps_slot_to_rank(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "f.jsonl")
            _write_normalized_jsonl(path, [_normalized_record("q1")])
            parsed = m.load_normalized_family(path, "ragdefender_targeted", "ragdefender")
        bundle = m.family_record_to_bundle(parsed["q1"])
        ranks = sorted(mp["poison_rank"] for mp in bundle["mutated_passages"])
        self.assertEqual(ranks, [0, 1, 2, 3, 4])


# ---------------------------------------------------------------------------
# 4. Comparison-vs-previous-run helpers.
# ---------------------------------------------------------------------------

class TestComparisonHelpers(unittest.TestCase):
    def _minimal_bundle_row(self, **overrides):
        row = {"family": "ragdefender_targeted", "query_id": "q1"}
        for key in m.base_eval._NUMERIC_METRIC_KEYS:  # noqa: SLF001
            row[key] = 1.0
        row.update(overrides)
        return row

    def test_identical_previous_and_new_yields_zero_changes(self):
        new_row = self._minimal_bundle_row()
        previous_by_key = {("ragdefender_targeted", "q1"): dict(new_row)}
        rows = m.build_comparison_rows([new_row], previous_by_key)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_metrics_changed"], 0)
        self.assertFalse(rows[0]["any_metric_changed"])

    def test_a_single_differing_metric_is_detected(self):
        new_row = self._minimal_bundle_row(ragdefender_removed_poison=3.0)
        previous_row = self._minimal_bundle_row(ragdefender_removed_poison=5.0)
        previous_by_key = {("ragdefender_targeted", "q1"): previous_row}
        rows = m.build_comparison_rows([new_row], previous_by_key)
        self.assertEqual(rows[0]["n_metrics_changed"], 1)
        self.assertTrue(rows[0]["any_metric_changed"])
        self.assertIn("ragdefender_removed_poison", rows[0]["changed_metric_names"])
        self.assertAlmostEqual(rows[0]["max_abs_diff"], 2.0)

    def test_missing_previous_run_marks_row_as_unavailable_without_raising(self):
        new_row = self._minimal_bundle_row()
        rows = m.build_comparison_rows([new_row], None)
        self.assertFalse(rows[0]["previous_run_found"])
        self.assertIsNone(rows[0]["any_metric_changed"])
        self.assertEqual(rows[0]["n_metrics_compared"], 0)

    def test_load_previous_bundle_scores_missing_file_returns_none(self):
        self.assertIsNone(m.load_previous_bundle_scores(os.path.join(tempfile.gettempdir(), "definitely_missing_12345.csv")))


# ---------------------------------------------------------------------------
# 5. Static checks: no GPT/API/retrieval/training calls anywhere in this
#    script, and it reuses (does not reimplement) the scoring primitives.
# ---------------------------------------------------------------------------

class TestNoForbiddenCalls(unittest.TestCase):
    def test_no_forbidden_api_modules_are_imported(self):
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

    def test_source_never_calls_train_or_fit(self):
        import ast

        tree = ast.parse(inspect.getsource(m))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(
                    node.func.attr, ("fit", "train"),
                    msg=f"found a .{node.func.attr}(...) call -- this script must never train/retrain a model",
                )

    def test_offline_env_vars_are_forced(self):
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")

    def test_reuses_base_eval_and_prev_eval_rather_than_duplicating_logic(self):
        source = inspect.getsource(m)
        self.assertIn("import run_text_mutation_fixed_context_eval as base_eval", source)
        self.assertIn("import run_targeted_mutation_bundle_1_eval as prev_eval", source)
        for attr in ("build_original_context", "build_mutated_context", "assert_same_k10_membership", "score_context", "load_models", "compute_deltas"):
            self.assertTrue(hasattr(m.base_eval, attr), msg=f"base_eval module missing {attr!r}")
        for attr in ("summarize_family_defense", "build_cross_defense_failure_matrix", "load_first_pilot_deltas"):
            self.assertTrue(hasattr(m.prev_eval, attr), msg=f"prev_eval module missing {attr!r}")


# ---------------------------------------------------------------------------
# 6. RetrievedPassage sanity (import wiring only).
# ---------------------------------------------------------------------------

class TestRetrievedPassageWiring(unittest.TestCase):
    def test_retrieved_passage_importable(self):
        p = RetrievedPassage(doc_id="d1", text="t", source="corpus", is_poison=False, retrieval_score=None, rank=0)
        self.assertEqual(p.doc_id, "d1")


if __name__ == "__main__":
    unittest.main()
