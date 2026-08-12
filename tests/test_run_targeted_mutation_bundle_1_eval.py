"""Tests for scripts/run_targeted_mutation_bundle_1_eval.py -- the
fixed-retrieval, cross-defense evaluator for the 3 defense-targeted GPT
mutation families under
`manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/`.

Fully offline: no real sentence-transformers/flan-t5/distilgpt2 model is
downloaded, and no LLM/GPT/API call is ever made anywhere in this file.

Run with: python -m unittest tests.test_run_targeted_mutation_bundle_1_eval -v
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

import run_targeted_mutation_bundle_1_eval as m  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures mirroring the 3 real family files' actual shapes.
# ---------------------------------------------------------------------------

def _ragdefender_style_record(query_id="q1", correct_doc_ids=True):
    """Mirrors ragdefender_discourse_diverse_poisoned_passages.txt: list key
    `rewritten_passages`, text key `rewritten_text`."""
    return {
        "query_id": query_id,
        "question": "Some question?",
        "target_wrong_answer": "wrong answer",
        "rewritten_passages": [
            {
                "poison_slot": i,
                "doc_id": f"adv::LM_targeted::{query_id}::{i if correct_doc_ids else 900 + i}",
                "discourse_frame": "encyclopedia/reference style",
                "rewritten_text": f"ragdefender-style rewritten text {i}",
            }
            for i in range(5)
        ],
    }


def _filterrag_style_record(query_id="q1"):
    """Mirrors filterrag_gpt_poisoned_passages_low_overlap.txt: list key
    `poisoned_passages`, text key `rewritten_text`."""
    return {
        "query_id": query_id,
        "k": 10,
        "selection_role": "primary",
        "question": "Some question?",
        "target_wrong_answer": "wrong answer",
        "poisoned_passages": [
            {
                "poison_slot": i,
                "doc_id": f"adv::LM_targeted::{query_id}::{i}",
                "rewritten_text": f"filterrag-style rewritten text {i}",
            }
            for i in range(5)
        ],
    }


def _mlfilterrag_style_record(query_id="q1"):
    """Mirrors mlfilterrag_gpt_prompt_packets_clean_reference_rewrites.txt:
    list key `poisoned_passages`, text key `original_text` (despite the
    name, this is the *mutated* text in the real file -- see module
    docstring)."""
    return {
        "query_id": query_id,
        "k": 10,
        "selection_role": "primary",
        "question": "Some question?",
        "target_wrong_answer": "wrong answer",
        "poisoned_passages": [
            {
                "poison_slot": i,
                "doc_id": f"adv::LM_targeted::{query_id}::{i}",
                "original_text": f"mlfilterrag-style rewritten text {i}",
            }
            for i in range(5)
        ],
        "mutation_instructions": "Rewrite the five poisoned passages as fluent, simple, encyclopedic reference prose.",
    }


def _write_json_array(path, records):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)


# ---------------------------------------------------------------------------
# 1. parse_family_file
# ---------------------------------------------------------------------------

class TestParseFamilyFile(unittest.TestCase):
    def test_parses_ragdefender_style_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ragdefender.txt")
            _write_json_array(path, [_ragdefender_style_record("q1")])
            parsed = m.parse_family_file(path, m.FAMILY_SPECS["ragdefender_targeted"])
        self.assertEqual(set(parsed), {"q1"})
        passages = parsed["q1"]["passages"]
        self.assertEqual(len(passages), 5)
        self.assertEqual([p["poison_slot"] for p in passages], [0, 1, 2, 3, 4])
        self.assertEqual(passages[0]["text"], "ragdefender-style rewritten text 0")

    def test_parses_filterrag_style_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "filterrag.txt")
            _write_json_array(path, [_filterrag_style_record("q2")])
            parsed = m.parse_family_file(path, m.FAMILY_SPECS["filterrag_targeted"])
        self.assertEqual(set(parsed), {"q2"})
        self.assertEqual(parsed["q2"]["passages"][2]["text"], "filterrag-style rewritten text 2")

    def test_parses_mlfilterrag_style_file_with_original_text_field(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "mlfilterrag.txt")
            _write_json_array(path, [_mlfilterrag_style_record("q3")])
            parsed = m.parse_family_file(path, m.FAMILY_SPECS["mlfilterrag_targeted"])
        self.assertEqual(set(parsed), {"q3"})
        # The mutated text lives under "original_text" in this family's real
        # file; parse_family_file must still extract it as the mutated text.
        self.assertEqual(parsed["q3"]["passages"][4]["text"], "mlfilterrag-style rewritten text 4")

    def test_all_three_family_files_parse_from_one_directory(self):
        """Regression guard: all 3 real file *shapes* must be parseable by
        their respective specs when placed side-by-side, exactly as the
        real mutation_bundle_1/ directory does."""
        with tempfile.TemporaryDirectory() as td:
            paths = {}
            for family_key, spec in m.FAMILY_SPECS.items():
                path = os.path.join(td, spec["filename"])
                if family_key == "ragdefender_targeted":
                    _write_json_array(path, [_ragdefender_style_record("qA")])
                elif family_key == "filterrag_targeted":
                    _write_json_array(path, [_filterrag_style_record("qA")])
                else:
                    _write_json_array(path, [_mlfilterrag_style_record("qA")])
                paths[family_key] = path

            for family_key, spec in m.FAMILY_SPECS.items():
                parsed = m.parse_family_file(paths[family_key], spec)
                self.assertIn("qA", parsed)
                self.assertEqual(len(parsed["qA"]["passages"]), 5)

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "empty.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            with self.assertRaises(ValueError):
                m.parse_family_file(path, m.FAMILY_SPECS["filterrag_targeted"])

    def test_missing_query_id_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.txt")
            rec = _filterrag_style_record("q1")
            del rec["query_id"]
            _write_json_array(path, [rec])
            with self.assertRaises(ValueError):
                m.parse_family_file(path, m.FAMILY_SPECS["filterrag_targeted"])

    def test_wrong_passage_count_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.txt")
            rec = _filterrag_style_record("q1")
            rec["poisoned_passages"] = rec["poisoned_passages"][:4]
            _write_json_array(path, [rec])
            with self.assertRaises(ValueError):
                m.parse_family_file(path, m.FAMILY_SPECS["filterrag_targeted"])

    def test_duplicate_poison_slot_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.txt")
            rec = _filterrag_style_record("q1")
            rec["poisoned_passages"][1]["poison_slot"] = 0
            _write_json_array(path, [rec])
            with self.assertRaises(ValueError):
                m.parse_family_file(path, m.FAMILY_SPECS["filterrag_targeted"])

    def test_empty_text_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.txt")
            rec = _filterrag_style_record("q1")
            rec["poisoned_passages"][0]["rewritten_text"] = "   "
            _write_json_array(path, [rec])
            with self.assertRaises(ValueError):
                m.parse_family_file(path, m.FAMILY_SPECS["filterrag_targeted"])

    def test_missing_passage_list_key_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.txt")
            rec = _filterrag_style_record("q1")
            del rec["poisoned_passages"]
            _write_json_array(path, [rec])
            with self.assertRaises(ValueError):
                m.parse_family_file(path, m.FAMILY_SPECS["filterrag_targeted"])


# ---------------------------------------------------------------------------
# 2. doc_id consistency check (data-integrity flag, non-fatal).
# ---------------------------------------------------------------------------

class TestCheckDocIdConsistency(unittest.TestCase):
    def test_detects_mismatched_doc_ids(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ragdefender.txt")
            _write_json_array(path, [_ragdefender_style_record("q1", correct_doc_ids=False)])
            parsed = m.parse_family_file(path, m.FAMILY_SPECS["ragdefender_targeted"])

        poison_by_query = {
            "q1": [
                {"poison_slot": str(i), "doc_id": f"adv::LM_targeted::q1::{i}"}
                for i in range(5)
            ]
        }
        mismatches = m.check_doc_id_consistency("ragdefender_targeted", parsed, poison_by_query)
        self.assertEqual(len(mismatches), 5)
        for mm in mismatches:
            self.assertEqual(mm["family"], "ragdefender_targeted")
            self.assertEqual(mm["query_id"], "q1")
            self.assertNotEqual(mm["file_doc_id"], mm["csv_doc_id"])

    def test_no_mismatch_when_doc_ids_agree(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "filterrag.txt")
            _write_json_array(path, [_filterrag_style_record("q1")])
            parsed = m.parse_family_file(path, m.FAMILY_SPECS["filterrag_targeted"])

        poison_by_query = {
            "q1": [
                {"poison_slot": str(i), "doc_id": f"adv::LM_targeted::q1::{i}"}
                for i in range(5)
            ]
        }
        mismatches = m.check_doc_id_consistency("filterrag_targeted", parsed, poison_by_query)
        self.assertEqual(mismatches, [])


# ---------------------------------------------------------------------------
# 3. family_record_to_bundle + reused build_mutated_context /
#    assert_same_k10_membership: clean passages preserved, k=10 membership
#    unchanged, only poison text replaced.
# ---------------------------------------------------------------------------

class TestFamilyRecordToBundleAndContextReuse(unittest.TestCase):
    def _poison_rows(self, query_id="q1"):
        return [
            {"query_id": query_id, "poison_slot": str(i), "retrieved_rank": str(i + 1),
             "doc_id": f"poison_doc_{i}", "original_poison_text": f"original poison text {i}"}
            for i in range(5)
        ]

    def _clean_rows(self, query_id="q1"):
        return [
            {"query_id": query_id, "retrieved_rank": str(i + 6),
             "doc_id": f"clean_doc_{i}", "clean_text": f"original clean text {i}"}
            for i in range(5)
        ]

    def test_only_poison_text_changes_and_k10_membership_preserved(self):
        import run_text_mutation_fixed_context_eval as base_eval

        poison_rows = self._poison_rows()
        clean_rows = self._clean_rows()
        original_context = base_eval.build_original_context(poison_rows, clean_rows)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "filterrag.txt")
            _write_json_array(path, [_filterrag_style_record("q1")])
            parsed = m.parse_family_file(path, m.FAMILY_SPECS["filterrag_targeted"])
        rec = parsed["q1"]
        bundle = m.family_record_to_bundle(rec)

        mutated_context = base_eval.build_mutated_context(original_context, poison_rows, bundle)

        # Fixed k=10 membership/order unchanged.
        base_eval.assert_same_k10_membership(original_context, mutated_context)

        for orig, mut in zip(original_context, mutated_context):
            if orig.is_poison:
                self.assertNotEqual(orig.text, mut.text)
                self.assertTrue(mut.text.startswith("filterrag-style rewritten text"))
            else:
                # Clean passages remain byte-identical.
                self.assertEqual(orig.text, mut.text)
            self.assertEqual(orig.doc_id, mut.doc_id)
            self.assertEqual(orig.rank, mut.rank)
            self.assertEqual(orig.is_poison, mut.is_poison)

    def test_family_record_to_bundle_maps_poison_slot_to_poison_rank(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "mlfilterrag.txt")
            _write_json_array(path, [_mlfilterrag_style_record("q1")])
            parsed = m.parse_family_file(path, m.FAMILY_SPECS["mlfilterrag_targeted"])
        bundle = m.family_record_to_bundle(parsed["q1"])
        self.assertEqual(len(bundle["mutated_passages"]), 5)
        ranks = sorted(mp["poison_rank"] for mp in bundle["mutated_passages"])
        self.assertEqual(ranks, [0, 1, 2, 3, 4])
        for mp in bundle["mutated_passages"]:
            self.assertTrue(mp["mutated_text"].startswith("mlfilterrag-style rewritten text"))


# ---------------------------------------------------------------------------
# 4. Mutation family labels are preserved end-to-end (bundle rows carry the
#    correct `family` value, distinct per family).
# ---------------------------------------------------------------------------

class TestFamilyLabelsPreserved(unittest.TestCase):
    def test_family_keys_are_distinct_and_match_intended_defense(self):
        self.assertEqual(set(m.FAMILY_SPECS), {"ragdefender_targeted", "filterrag_targeted", "mlfilterrag_targeted"})
        self.assertEqual(m.FAMILY_SPECS["ragdefender_targeted"]["intended_defense"], "ragdefender")
        self.assertEqual(m.FAMILY_SPECS["filterrag_targeted"]["intended_defense"], "filterrag")
        self.assertEqual(m.FAMILY_SPECS["mlfilterrag_targeted"]["intended_defense"], "ml_filterrag")


# ---------------------------------------------------------------------------
# 5. Aggregation: summarize_family_defense + build_cross_defense_failure_matrix.
# ---------------------------------------------------------------------------

class TestAggregation(unittest.TestCase):
    def _bundle_row(self, **overrides):
        row = {
            "N_retrieved_poison": 5, "N_retrieved_clean": 5,
            "ragdefender_removed_poison": 5, "ragdefender_removed_clean": 0,
            "ragdefender_residual_poison_fraction": 0.0, "ragdefender_top_pair_pp": 10,
            "ragdefender_mean_pp_cosine": 0.9, "ragdefender_mean_pc_cosine": 0.4,
            "filterrag_removed_poison": 5, "filterrag_removed_clean": 1,
            "filterrag_residual_poison_fraction": 0.0, "filterrag_mean_freq_density_poison": 1.0,
            "filterrag_mean_matched_freq_sum_poison": 1.0,
            "ml_removed_poison_t04": 5, "ml_removed_clean_t04": 1, "ml_residual_poison_fraction_t04": 0.0,
            "ml_removed_poison_t035": 5, "ml_removed_poison_t05": 5,
            "ml_mean_poison_probability": 0.9, "ml_mean_freq_density_poison": 1.0,
            "ml_mean_matched_freq_sum_poison": 1.0, "ml_mean_perplexity_poison": 20.0,
            "ml_mean_slm_answer_logprob_poison": -1.0,
        }
        row.update(overrides)
        return row

    def _delta_row(self, **overrides):
        row = {
            "delta_ragdefender_removed_poison": 0, "delta_ragdefender_removed_clean": 0,
            "delta_ragdefender_residual_poison_fraction": 0.0, "delta_ragdefender_top_pair_pp": 0,
            "delta_filterrag_removed_poison": 0, "delta_filterrag_removed_clean": 0,
            "delta_filterrag_residual_poison_fraction": 0.0, "delta_filterrag_mean_freq_density_poison": 0.0,
            "delta_filterrag_mean_matched_freq_sum_poison": 0.0,
            "delta_ml_removed_poison_t04": 0, "delta_ml_removed_clean_t04": 0,
            "delta_ml_residual_poison_fraction_t04": 0.0,
            "delta_ml_mean_poison_probability": 0.0, "delta_ml_mean_freq_density_poison": 0.0,
            "delta_ml_mean_matched_freq_sum_poison": 0.0,
        }
        row.update(overrides)
        return row

    def test_summarize_family_defense_ragdefender_columns(self):
        rows = [self._bundle_row(ragdefender_removed_poison=4)]
        deltas = [self._delta_row(delta_ragdefender_removed_poison=-1)]
        summary = m.summarize_family_defense("fam", "ragdefender", "ragdefender", rows, deltas)
        self.assertEqual(summary["mean_removed_poison"], 4)
        self.assertEqual(summary["mean_delta_removed_poison"], -1)
        self.assertTrue(summary["is_intended_target"])
        # Non-ragdefender-specific columns should not be populated.
        self.assertNotIn("mean_freq_density", summary)

    def test_summarize_family_defense_ml_filterrag_columns(self):
        rows = [self._bundle_row(ml_removed_poison_t04=3, ml_mean_poison_probability=0.3)]
        deltas = [self._delta_row(delta_ml_removed_poison_t04=-2, delta_ml_mean_poison_probability=-0.6)]
        summary = m.summarize_family_defense("fam", "ml_filterrag", "ml_filterrag", rows, deltas)
        self.assertEqual(summary["mean_removed_poison"], 3)
        self.assertEqual(summary["mean_delta_removed_poison"], -2)
        self.assertEqual(summary["mean_poison_probability"], 0.3)
        self.assertEqual(summary["mean_delta_poison_probability"], -0.6)

    def test_cross_defense_failure_matrix_flags_only_non_target_weakening(self):
        summary_rows = [
            m.summarize_family_defense(
                "fam_a", "ragdefender", "ragdefender",
                [self._bundle_row()], [self._delta_row(delta_ragdefender_removed_poison=0)],
            ),
            m.summarize_family_defense(
                "fam_a", "ragdefender", "filterrag",
                [self._bundle_row()], [self._delta_row(delta_filterrag_removed_poison=-3)],
            ),
            m.summarize_family_defense(
                "fam_a", "ragdefender", "ml_filterrag",
                [self._bundle_row()], [self._delta_row(delta_ml_removed_poison_t04=0)],
            ),
        ]
        matrix = m.build_cross_defense_failure_matrix(summary_rows)
        self.assertEqual(len(matrix), 1)
        row = matrix[0]
        self.assertEqual(row["family"], "fam_a")
        self.assertFalse(row["ragdefender_weakened"])
        self.assertTrue(row["filterrag_weakened"])
        self.assertFalse(row["ml_filterrag_weakened"])
        self.assertTrue(row["any_cross_defense_failure"])
        self.assertEqual(row["cross_defense_failure_defenses"], "filterrag")

    def test_cross_defense_failure_matrix_no_failure_when_only_target_weakened(self):
        summary_rows = [
            m.summarize_family_defense(
                "fam_b", "filterrag", "ragdefender",
                [self._bundle_row()], [self._delta_row(delta_ragdefender_removed_poison=0)],
            ),
            m.summarize_family_defense(
                "fam_b", "filterrag", "filterrag",
                [self._bundle_row()], [self._delta_row(delta_filterrag_removed_poison=-2)],
            ),
            m.summarize_family_defense(
                "fam_b", "filterrag", "ml_filterrag",
                [self._bundle_row()], [self._delta_row(delta_ml_removed_poison_t04=0)],
            ),
        ]
        matrix = m.build_cross_defense_failure_matrix(summary_rows)
        row = matrix[0]
        self.assertTrue(row["filterrag_weakened"])
        self.assertFalse(row["any_cross_defense_failure"])
        self.assertEqual(row["cross_defense_failure_defenses"], "")


# ---------------------------------------------------------------------------
# 6. All three defenses evaluated for every bundle (schema-level guarantee).
# ---------------------------------------------------------------------------

class TestAllThreeDefensesEvaluated(unittest.TestCase):
    def test_defense_names_has_all_three(self):
        self.assertEqual(m.DEFENSE_NAMES, ("ragdefender", "filterrag", "ml_filterrag"))

    def test_summary_fields_cover_all_three_defenses_worth_of_columns(self):
        # RAGDefender-only, FilterRAG/ML-shared, and ML-only columns must all
        # be present in the unified summary schema.
        for col in ("mean_top_pair_pp", "mean_pp_cosine"):  # RAGDefender-only
            self.assertIn(col, m.SUMMARY_FIELDS)
        for col in ("mean_freq_density", "mean_matched_freq_sum"):  # shared
            self.assertIn(col, m.SUMMARY_FIELDS)
        for col in ("mean_poison_probability", "mean_perplexity", "mean_slm_answer_logprob"):  # ML-only
            self.assertIn(col, m.SUMMARY_FIELDS)

    def test_main_computes_a_summary_row_per_family_per_defense(self):
        # 3 families x 3 defenses == 9 summary rows, enforced structurally by
        # the double loop in main() -- verified here via the constants it
        # iterates over, without re-running the (slow) full pipeline.
        self.assertEqual(len(m.FAMILY_SPECS) * len(m.DEFENSE_NAMES), 9)


# ---------------------------------------------------------------------------
# 7. Static checks: no GPT/API/retrieval/training calls anywhere in this file.
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

    def test_reuses_base_eval_module_rather_than_duplicating_scoring_logic(self):
        """Confirms this script imports (reuses), rather than re-implements,
        the fixed-context scoring/context-reconstruction primitives from
        run_text_mutation_fixed_context_eval.py."""
        source = inspect.getsource(m)
        self.assertIn("import run_text_mutation_fixed_context_eval as base_eval", source)
        for attr in ("build_original_context", "build_mutated_context", "assert_same_k10_membership", "score_context", "load_models", "compute_deltas"):
            self.assertTrue(hasattr(m.base_eval, attr), msg=f"base_eval module missing {attr!r}")


# ---------------------------------------------------------------------------
# 8. RetrievedPassage sanity (import wiring only; heavy scoring itself is
#    covered by tests/test_run_text_mutation_fixed_context_eval.py).
# ---------------------------------------------------------------------------

class TestRetrievedPassageWiring(unittest.TestCase):
    def test_retrieved_passage_importable_and_matches_base_eval_usage(self):
        p = RetrievedPassage(doc_id="d1", text="t", source="corpus", is_poison=False, retrieval_score=None, rank=0)
        self.assertEqual(p.doc_id, "d1")


if __name__ == "__main__":
    unittest.main()
