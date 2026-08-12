"""Tests for scripts/audit_normalize_mutation_bundle_1.py -- the metadata/
schema audit and doc_id-normalization pass over the 3 defense-targeted GPT
mutation family files under
`manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/`.

No network, model, retrieval, or defense-scoring code is imported or
exercised anywhere in this file (the script under test is itself a pure
stdlib json/csv audit with no such dependencies).

Run with: python -m unittest tests.test_audit_normalize_mutation_bundle_1 -v
"""
import ast
import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import audit_normalize_mutation_bundle_1 as m  # noqa: E402


AUTHORITATIVE_CSV_HEADER = [
    "query_id", "k", "poison_slot", "retrieved_rank", "doc_id",
    "poison_source_query_id", "is_self_query_poison", "question",
    "target_wrong_answer", "original_poison_text",
    "original_freq_density_score", "original_matched_freq_sum",
    "original_ml_poison_probability_t04", "original_filterrag_removed",
    "original_ml_removed_t04",
]


def _write_authoritative_csv(path, query_id="q1", n_slots=5, original_text_fn=None):
    original_text_fn = original_text_fn or (lambda slot: f"original poison text {slot}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(AUTHORITATIVE_CSV_HEADER)
        for slot in range(n_slots):
            writer.writerow([
                query_id, 10, slot, slot + 1, f"canonical::{query_id}::{slot}",
                query_id, True, "Some question?", "wrong answer",
                original_text_fn(slot), 1.0, 10.0, "", True, True,
            ])


def _write_clean_csv(path, query_id="q1", n_rows=5):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "k", "retrieved_rank", "doc_id", "clean_text", "original_filterrag_removed", "original_ml_removed_t04"])
        for i in range(n_rows):
            writer.writerow([query_id, 10, i + 6, f"clean::{query_id}::{i}", f"clean text {i}", True, False])


def _write_selected_queries_csv(path, query_id="q1", selection_role="primary"):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id", "k", "question", "target_wrong_answer", "N_retrieved_poison",
            "N_retrieved_clean", "ml_removed_poison_t04", "ml_removed_clean_t04",
            "ml_residual_poison_fraction_t04", "filterrag_removed_poison",
            "filterrag_removed_clean", "filterrag_residual_poison_fraction",
            "selection_role", "selection_reason", "provenance_note",
        ])
        writer.writerow([
            query_id, 10, "Some question?", "wrong answer", 5, 5, 5, 0, 0.0, 5, 5, 0.0,
            selection_role, "test fixture", "test fixture",
        ])


def _family_record(query_id="q1", doc_id_offset=0, slots=range(5), text_fn=None, list_key="rewritten_passages", text_key="rewritten_text"):
    text_fn = text_fn or (lambda slot: f"mutated text {slot}")
    return {
        "query_id": query_id,
        "question": "Some question?",
        "target_wrong_answer": "wrong answer",
        list_key: [
            {"poison_slot": slot, "doc_id": f"file::{query_id}::{slot + doc_id_offset}", text_key: text_fn(slot)}
            for slot in slots
        ],
    }


class _TempAuditFixture:
    """Builds a self-contained temp directory with a mutation_input_passages.csv,
    clean_context_passages.csv, selected_queries.csv, and one family file, then
    runs audit_and_normalize_family against it."""

    def __init__(self, tmpdir, family_records, query_id="q1", n_authoritative_slots=5, original_text_fn=None):
        self.tmpdir = tmpdir
        self.pilot_dir = tmpdir
        _write_authoritative_csv(os.path.join(tmpdir, "mutation_input_passages.csv"), query_id, n_authoritative_slots, original_text_fn)
        _write_clean_csv(os.path.join(tmpdir, "clean_context_passages.csv"), query_id)
        _write_selected_queries_csv(os.path.join(tmpdir, "selected_queries.csv"), query_id)
        family_path = os.path.join(tmpdir, "family.txt")
        with open(family_path, "w", encoding="utf-8") as f:
            json.dump(family_records, f)
        self.family_path = family_path

        self.authoritative = m.load_authoritative_passages(os.path.join(tmpdir, "mutation_input_passages.csv"))
        self.known_query_ids = set(qid for qid, _slot in self.authoritative)
        self.selection_roles = m.load_selection_roles(os.path.join(tmpdir, "selected_queries.csv"))

    def run(self, spec_overrides=None, correct_answers=None):
        spec = {
            "filename": "family.txt", "intended_defense": "ragdefender",
            "passage_list_keys": ("rewritten_passages",), "text_field_keys": ("rewritten_text",),
            "normalized_filename": "family.normalized.jsonl",
        }
        if spec_overrides:
            spec.update(spec_overrides)
        return m.audit_and_normalize_family(
            "test_family", spec, self.family_path, self.authoritative, self.known_query_ids,
            correct_answers or {}, self.selection_roles,
        )


class TestCanonicalDocIdUsedWhenConflicting(unittest.TestCase):
    def test_canonical_doc_id_substituted_and_source_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = [_family_record("q1", doc_id_offset=900)]  # every file doc_id is wrong
            fixture = _TempAuditFixture(tmp, record)
            audit_rows, normalized = fixture.run()

            self.assertEqual(len(normalized), 1)
            passages = normalized[0]["mutated_passages"]
            self.assertEqual(len(passages), 5)
            for p in passages:
                slot = p["poison_slot"]
                self.assertEqual(p["doc_id"], f"canonical::q1::{slot}")
                self.assertEqual(p["source_file_doc_id"], f"file::q1::{slot + 900}")
                self.assertTrue(p["doc_id_mismatch"])
                self.assertNotEqual(p["doc_id"], p["source_file_doc_id"])

            mismatches = [r for r in audit_rows if r["doc_id_mismatch"]]
            self.assertEqual(len(mismatches), 5)
            for r in mismatches:
                self.assertEqual(r["canonical_doc_id"], f"canonical::q1::{r['poison_slot']}")

    def test_no_mismatch_when_file_doc_id_already_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            # doc_id_offset=0 matches _write_authoritative_csv's canonical doc_id pattern
            # only if the family file uses the same "file::" prefix as canonical -- here we
            # instead directly reuse the canonical ids to assert doc_id_mismatch=False.
            record = [{
                "query_id": "q1", "question": "Some question?", "target_wrong_answer": "wrong answer",
                "rewritten_passages": [
                    {"poison_slot": s, "doc_id": f"canonical::q1::{s}", "rewritten_text": f"mutated text {s}"}
                    for s in range(5)
                ],
            }]
            fixture = _TempAuditFixture(tmp, record)
            audit_rows, normalized = fixture.run()
            self.assertEqual(len(normalized), 1)
            for r in audit_rows:
                self.assertFalse(r["doc_id_mismatch"])


class TestPoisonSlotAlignmentPreserved(unittest.TestCase):
    def test_slots_map_to_correct_text_even_when_input_order_scrambled(self):
        with tempfile.TemporaryDirectory() as tmp:
            scrambled = [{
                "query_id": "q1", "question": "Some question?", "target_wrong_answer": "wrong answer",
                "rewritten_passages": [
                    {"poison_slot": 3, "doc_id": "file::q1::3", "rewritten_text": "text for slot 3"},
                    {"poison_slot": 0, "doc_id": "file::q1::0", "rewritten_text": "text for slot 0"},
                    {"poison_slot": 4, "doc_id": "file::q1::4", "rewritten_text": "text for slot 4"},
                    {"poison_slot": 1, "doc_id": "file::q1::1", "rewritten_text": "text for slot 1"},
                    {"poison_slot": 2, "doc_id": "file::q1::2", "rewritten_text": "text for slot 2"},
                ],
            }]
            fixture = _TempAuditFixture(tmp, scrambled)
            _audit_rows, normalized = fixture.run()

            self.assertEqual(len(normalized), 1)
            passages = normalized[0]["mutated_passages"]
            # Output must be sorted by poison_slot and each slot's text must be its own,
            # never another slot's text.
            self.assertEqual([p["poison_slot"] for p in passages], [0, 1, 2, 3, 4])
            for p in passages:
                self.assertEqual(p["mutated_text"], f"text for slot {p['poison_slot']}")
                self.assertEqual(p["doc_id"], f"canonical::q1::{p['poison_slot']}")


class TestMutatedTextNotAlteredDuringNormalization(unittest.TestCase):
    def test_text_copied_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            weird_text = "Text with  double spaces, \"quotes\", and\nnewlines\tand tabs."
            record = [_family_record("q1", text_fn=lambda slot: weird_text if slot == 0 else f"mutated text {slot}")]
            fixture = _TempAuditFixture(tmp, record)
            _audit_rows, normalized = fixture.run()
            slot0 = next(p for p in normalized[0]["mutated_passages"] if p["poison_slot"] == 0)
            self.assertEqual(slot0["mutated_text"], weird_text)

    def test_identical_text_to_original_is_flagged_not_altered(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Family passage's text is byte-identical to the authoritative original --
            # this must be flagged, but the (unchanged) text must still pass through verbatim.
            record = [_family_record("q1", text_fn=lambda slot: f"original poison text {slot}")]
            fixture = _TempAuditFixture(tmp, record)
            audit_rows, normalized = fixture.run()
            self.assertTrue(all(r["mutated_text_unchanged_from_original"] for r in audit_rows if r["poison_slot"] is not None))
            passages = normalized[0]["mutated_passages"]
            for p in passages:
                self.assertEqual(p["mutated_text"], f"original poison text {p['poison_slot']}")
                self.assertIn("mutated_text_unchanged_from_original", p["quality_flags"])


class TestMalformedOrMissingSlotsRejected(unittest.TestCase):
    def test_missing_slot_excludes_whole_query_from_normalized_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = [_family_record("q1", slots=[0, 1, 2, 3])]  # slot 4 missing
            fixture = _TempAuditFixture(tmp, record)
            audit_rows, normalized = fixture.run()

            self.assertEqual(len(normalized), 0)
            self.assertTrue(all(r["included_in_normalized_output"] is False for r in audit_rows))
            self.assertTrue(any("missing_poison_slot" in r["record_level_flags"] for r in audit_rows))

    def test_non_integer_slot_is_flagged_and_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = [{
                "query_id": "q1", "question": "Some question?", "target_wrong_answer": "wrong answer",
                "rewritten_passages": [
                    {"poison_slot": "not-an-int", "doc_id": "file::q1::0", "rewritten_text": "text 0"},
                    {"poison_slot": 1, "doc_id": "file::q1::1", "rewritten_text": "text 1"},
                    {"poison_slot": 2, "doc_id": "file::q1::2", "rewritten_text": "text 2"},
                    {"poison_slot": 3, "doc_id": "file::q1::3", "rewritten_text": "text 3"},
                    {"poison_slot": 4, "doc_id": "file::q1::4", "rewritten_text": "text 4"},
                ],
            }]
            fixture = _TempAuditFixture(tmp, record)
            audit_rows, normalized = fixture.run()
            self.assertEqual(len(normalized), 0)
            invalid_rows = [r for r in audit_rows if r["passage_level_flags"] == "invalid_or_missing_poison_slot"]
            self.assertEqual(len(invalid_rows), 1)
            self.assertFalse(invalid_rows[0]["included_in_normalized_output"])

    def test_unknown_query_id_flagged_and_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = [_family_record("query_not_in_authoritative_csv")]
            fixture = _TempAuditFixture(tmp, record, query_id="q1")  # authoritative CSV only has q1
            audit_rows, normalized = fixture.run()
            self.assertEqual(len(normalized), 0)
            self.assertEqual(len(audit_rows), 1)
            self.assertEqual(audit_rows[0]["record_level_flags"], "unknown_query_id_not_in_mutation_input_passages")
            self.assertFalse(audit_rows[0]["included_in_normalized_output"])

    def test_wrong_passage_count_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = [_family_record("q1", slots=[0, 1, 2])]  # only 3 passages, not 5
            fixture = _TempAuditFixture(tmp, record)
            audit_rows, normalized = fixture.run()
            self.assertEqual(len(normalized), 0)
            self.assertTrue(any("wrong_passage_count:3" in r["record_level_flags"] for r in audit_rows))


class TestDuplicateSlotsRejected(unittest.TestCase):
    def test_duplicate_poison_slot_excludes_whole_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = [{
                "query_id": "q1", "question": "Some question?", "target_wrong_answer": "wrong answer",
                "rewritten_passages": [
                    {"poison_slot": 0, "doc_id": "file::q1::0", "rewritten_text": "text 0a"},
                    {"poison_slot": 0, "doc_id": "file::q1::0b", "rewritten_text": "text 0b"},
                    {"poison_slot": 1, "doc_id": "file::q1::1", "rewritten_text": "text 1"},
                    {"poison_slot": 2, "doc_id": "file::q1::2", "rewritten_text": "text 2"},
                    {"poison_slot": 3, "doc_id": "file::q1::3", "rewritten_text": "text 3"},
                ],
            }]
            fixture = _TempAuditFixture(tmp, record)
            audit_rows, normalized = fixture.run()

            self.assertEqual(len(normalized), 0)
            dup_rows = [r for r in audit_rows if r["poison_slot"] == 0]
            self.assertEqual(len(dup_rows), 2)
            for r in dup_rows:
                self.assertIn("duplicate_poison_slot", r["passage_level_flags"])
                self.assertFalse(r["included_in_normalized_output"])
            self.assertTrue(any("duplicate_poison_slot" in r["record_level_flags"] for r in audit_rows))

    def test_duplicate_mutated_text_within_query_flagged_but_structurally_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 5 unique valid slots (structurally fine) but two slots share identical text.
            record = [_family_record("q1", text_fn=lambda slot: "same text" if slot in (0, 1) else f"mutated text {slot}")]
            fixture = _TempAuditFixture(tmp, record)
            audit_rows, normalized = fixture.run()

            # Structurally valid -> still normalized (duplicate text is a quality flag, not
            # a structural rejection), but both slots must be flagged.
            self.assertEqual(len(normalized), 1)
            dup_flagged = [r for r in audit_rows if r["duplicate_mutated_text_within_family_query"]]
            self.assertEqual(sorted(r["poison_slot"] for r in dup_flagged), [0, 1])


class TestTrueAnswerLeakHeuristics(unittest.TestCase):
    def test_leak_detected_for_multi_word_correct_answer_not_in_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = [_family_record("q1", text_fn=lambda slot: "The real correct entity is mentioned here." if slot == 0 else f"mutated text {slot}")]
            fixture = _TempAuditFixture(tmp, record)
            audit_rows, _normalized = fixture.run(correct_answers={"q1": "real correct entity"})
            row0 = next(r for r in audit_rows if r["poison_slot"] == 0)
            self.assertTrue(row0["possible_true_answer_leak"])
            self.assertFalse(row0["true_answer_leak_low_confidence"])

    def test_short_common_word_leak_marked_low_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = [_family_record("q1", text_fn=lambda slot: "It is not a novel idea." if slot == 0 else f"mutated text {slot}")]
            fixture = _TempAuditFixture(tmp, record)
            audit_rows, _normalized = fixture.run(correct_answers={"q1": "not"})
            row0 = next(r for r in audit_rows if r["poison_slot"] == 0)
            self.assertTrue(row0["possible_true_answer_leak"])
            self.assertTrue(row0["true_answer_leak_low_confidence"])

    def test_no_leak_flag_when_correct_answer_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = [_family_record("q1")]
            fixture = _TempAuditFixture(tmp, record)
            audit_rows, _normalized = fixture.run(correct_answers={})
            self.assertTrue(all(r["possible_true_answer_leak"] is None for r in audit_rows))


class TestLoadCorrectAnswers(unittest.TestCase):
    def test_missing_file_returns_empty_dict_without_raising(self):
        result = m.load_correct_answers(os.path.join(tempfile.gettempdir(), "definitely_does_not_exist_12345.json"))
        self.assertEqual(result, {})

    def test_none_path_returns_empty_dict(self):
        self.assertEqual(m.load_correct_answers(None), {})

    def test_loads_correct_answer_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "adv.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"q1": {"correct answer": "the answer", "incorrect answer": "wrong"}}, f)
            self.assertEqual(m.load_correct_answers(path), {"q1": "the answer"})


class TestNoForbiddenCalls(unittest.TestCase):
    """Static (no execution) verification that the audit script cannot make a
    GPT/API/llm.query() call, rerun retrieval, rerun any defense, or train
    any model."""

    SCRIPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "audit_normalize_mutation_bundle_1.py")

    def _load_ast(self):
        with open(self.SCRIPT_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        return ast.parse(source, filename=self.SCRIPT_PATH)

    def test_no_forbidden_modules_are_imported(self):
        forbidden_modules = {
            "openai", "anthropic", "google.generativeai", "requests",
            "torch", "transformers", "sentence_transformers", "sklearn",
        }
        tree = self._load_ast()
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        overlap = imported & forbidden_modules
        self.assertEqual(overlap, set(), f"forbidden module(s) imported: {overlap!r}")

    def test_no_llm_query_or_requests_post_or_defense_imports(self):
        tree = self._load_ast()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "query":
                self.fail("found a '.query(...)' attribute access (possible llm.query() call).")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(alias.name.startswith("defense"), "must not import defense package")
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertFalse(node.module.startswith("defense"), "must not import defense package")


if __name__ == "__main__":
    unittest.main()
