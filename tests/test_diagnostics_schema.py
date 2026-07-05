"""Tests for defense/diagnostics.py -- schema validity and JSONL round-trip.

Run with: python -m unittest tests.test_diagnostics_schema -v
"""
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defense.diagnostics import (
    DETECTION_ONLY_FIELDS,
    DIAGNOSTIC_FIELDS,
    GENERATION_DEPENDENT_FIELDS,
    append_jsonl,
    build_diagnostic_record,
    read_jsonl,
    timer,
    validate_record,
)
from defense.passages import label_passages


def make_retrieved_and_kept():
    raw = [
        {"doc_id": "c1", "context": "clean 1", "score": 0.9, "source": "corpus", "is_poison": False},
        {"doc_id": "adv0", "context": "adv 0", "score": 0.85, "source": "adversarial", "is_poison": True},
        {"doc_id": "c2", "context": "clean 2", "score": 0.8, "source": "corpus", "is_poison": False},
        {"doc_id": "adv1", "context": "adv 1", "score": 0.75, "source": "adversarial", "is_poison": True},
        {"doc_id": "c3", "context": "clean 3", "score": 0.7, "source": "corpus", "is_poison": False},
    ]
    retrieved = label_passages(raw)
    # Simulate a defense that removed one poison (adv0) and one clean (c2).
    kept = [p for p in retrieved if p.doc_id not in {"adv0", "c2"}]
    return retrieved, kept


class TestBuildDiagnosticRecord(unittest.TestCase):
    def test_schema_has_all_required_fields(self):
        retrieved, kept = make_retrieved_and_kept()
        record = build_diagnostic_record(
            query_id="q1",
            dataset="hotpotqa",
            model="gpt4",
            attack="LM_targeted",
            defense="ragdefender_original",
            k=5,
            N_injected=2,
            retrieved_passages=retrieved,
            kept_passages=kept,
            N_adv_estimated_by_ragdefender=2,
        )
        missing = validate_record(record)
        self.assertEqual(missing, [], f"missing fields: {missing}")
        self.assertEqual(set(record.keys()), set(DIAGNOSTIC_FIELDS))

    def test_detection_fields_populated_without_generation(self):
        """Detection-quality fields must be fully populated even when no
        generation-dependent kwargs are passed (i.e. dry-run parity)."""
        retrieved, kept = make_retrieved_and_kept()
        record = build_diagnostic_record(
            query_id="q1",
            dataset="hotpotqa",
            model="gpt4",
            attack="LM_targeted",
            defense="ragdefender_original",
            k=5,
            N_injected=2,
            retrieved_passages=retrieved,
            kept_passages=kept,
            N_adv_estimated_by_ragdefender=2,
        )
        self.assertEqual(record["N_retrieved_poison"], 2)
        self.assertEqual(record["N_retrieved_clean"], 3)
        self.assertEqual(record["removed_poison"], 1)
        self.assertEqual(record["removed_clean"], 1)
        self.assertAlmostEqual(record["poison_recall"], 0.5)
        self.assertAlmostEqual(record["clean_false_positive_rate"], 1 / 3)
        self.assertEqual(record["residual_poison_count"], 1)
        self.assertEqual(record["residual_clean_count"], 2)
        self.assertAlmostEqual(record["residual_poison_fraction"], 1 / 3)
        # Generation-dependent fields must be explicitly None (dry-run).
        for field in GENERATION_DEPENDENT_FIELDS:
            self.assertIsNone(record[field], f"{field} should be None without generation")

    def test_generation_fields_populated_when_provided(self):
        retrieved, kept = make_retrieved_and_kept()
        record = build_diagnostic_record(
            query_id="q1",
            dataset="hotpotqa",
            model="gpt4",
            attack="LM_targeted",
            defense="ragdefender_original",
            k=5,
            N_injected=2,
            retrieved_passages=retrieved,
            kept_passages=kept,
            answer_no_defense="wrong",
            answer_with_defense="right",
            target_wrong_answer="wrong",
            gold_answer="right",
            asr_no_defense=True,
            asr_with_defense=False,
            latency_generation_sec=0.5,
        )
        for field in GENERATION_DEPENDENT_FIELDS:
            self.assertIsNotNone(record[field], f"{field} should be populated")

    def test_no_removal_gives_zero_recall_and_full_residual(self):
        retrieved, _ = make_retrieved_and_kept()
        record = build_diagnostic_record(
            query_id="q2",
            dataset="hotpotqa",
            model="gpt4",
            attack="LM_targeted",
            defense="none",
            k=5,
            N_injected=2,
            retrieved_passages=retrieved,
            kept_passages=retrieved,
        )
        self.assertEqual(record["removed_poison"], 0)
        self.assertEqual(record["removed_clean"], 0)
        self.assertEqual(record["poison_recall"], 0.0)
        self.assertEqual(record["residual_poison_count"], 2)
        self.assertAlmostEqual(record["residual_poison_fraction"], 2 / 5)

    def test_zero_retrieved_poison_gives_none_recall(self):
        raw = [{"doc_id": "c1", "context": "clean", "score": 0.9, "source": "corpus", "is_poison": False}]
        retrieved = label_passages(raw)
        record = build_diagnostic_record(
            query_id="q3",
            dataset="nq",
            model="gpt4",
            attack="LM_targeted",
            defense="none",
            k=1,
            N_injected=0,
            retrieved_passages=retrieved,
            kept_passages=retrieved,
        )
        self.assertIsNone(record["poison_recall"])


class TestJsonlRoundTrip(unittest.TestCase):
    def test_append_and_read_jsonl(self):
        retrieved, kept = make_retrieved_and_kept()
        record = build_diagnostic_record(
            query_id="q1",
            dataset="hotpotqa",
            model="gpt4",
            attack="LM_targeted",
            defense="ragdefender_original",
            k=5,
            N_injected=2,
            retrieved_passages=retrieved,
            kept_passages=kept,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "run.jsonl")
            append_jsonl(record, path)
            append_jsonl(record, path)
            loaded = read_jsonl(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["query_id"], "q1")
            self.assertEqual(validate_record(loaded[0]), [])


class TestTimer(unittest.TestCase):
    def test_timer_measures_elapsed(self):
        with timer() as t:
            time.sleep(0.01)
        self.assertIsNotNone(t["elapsed_sec"])
        self.assertGreaterEqual(t["elapsed_sec"], 0.0)


class TestFieldPartition(unittest.TestCase):
    def test_detection_and_generation_fields_partition_schema(self):
        self.assertEqual(
            set(DETECTION_ONLY_FIELDS) | set(GENERATION_DEPENDENT_FIELDS),
            set(DIAGNOSTIC_FIELDS),
        )
        self.assertEqual(set(DETECTION_ONLY_FIELDS) & set(GENERATION_DEPENDENT_FIELDS), set())


if __name__ == "__main__":
    unittest.main()
