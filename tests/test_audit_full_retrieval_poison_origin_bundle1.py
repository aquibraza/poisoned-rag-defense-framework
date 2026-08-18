"""Tests for scripts/audit_full_retrieval_poison_origin_bundle1.py -- the
self-query vs cross-query poison-origin audit of the 3-query full-retrieval
pilot.

Fully offline and model-free: every test exercises the pure classification/
aggregation logic with synthetic fixtures. No Contriever/sentence-transformers/
flan-t5/distilgpt2 model is downloaded or run.

Run with: python -m unittest tests.test_audit_full_retrieval_poison_origin_bundle1 -v
"""
import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from defense.passages import RetrievedPassage  # noqa: E402

import audit_full_retrieval_poison_origin_bundle1 as audit  # noqa: E402


FULL_POOL = ["q0", "q1", "q2", "q3", "q4"]  # 5 pool positions x N=5 -> 25-slot pool


def _adv(qid: str, global_idx: int) -> str:
    return f"adv::LM_targeted::{qid}::{global_idx}"


class TestClassifyPassageOrigin(unittest.TestCase):
    def test_clean_source_is_always_clean(self):
        out = audit.classify_passage_origin(
            doc_id="clean123", source="corpus", current_qid="q1",
            full_pool_query_ids=FULL_POOL, mutated_self_global_indices=set(), n_adv_per_query=5,
        )
        self.assertEqual(out["origin_label"], audit.ORIGIN_CLEAN)
        self.assertIsNone(out["true_global_index"])
        self.assertIsNone(out["true_owning_query_id"])

    def test_self_query_mutated_slot(self):
        # q1 owns global indices 5..9.
        out = audit.classify_passage_origin(
            doc_id=_adv("q1", 7), source="adversarial", current_qid="q1",
            full_pool_query_ids=FULL_POOL, mutated_self_global_indices={5, 6, 7, 8, 9}, n_adv_per_query=5,
        )
        self.assertEqual(out["origin_label"], audit.ORIGIN_MUTATED_SELF)
        self.assertEqual(out["true_global_index"], 7)
        self.assertEqual(out["true_owning_query_id"], "q1")

    def test_self_query_but_not_replaced_slot_is_anomaly(self):
        # q1 owns 5..9, but only 5,6,7,8 were (hypothetically) replaced -- 9 wasn't.
        out = audit.classify_passage_origin(
            doc_id=_adv("q1", 9), source="adversarial", current_qid="q1",
            full_pool_query_ids=FULL_POOL, mutated_self_global_indices={5, 6, 7, 8}, n_adv_per_query=5,
        )
        self.assertEqual(out["origin_label"], audit.ORIGIN_ORIGINAL_SELF)

    def test_cross_query_poison_detected_even_when_doc_id_qid_is_wrong(self):
        # merge_and_topk always stamps the CURRENTLY-retrieved-for qid onto every
        # adversarial doc_id, so this doc_id claims "q1" even though global_index=12
        # actually belongs to pool position 2 ("q2"). The true owner must come from
        # the index, not the doc_id's qid segment.
        out = audit.classify_passage_origin(
            doc_id=_adv("q1", 12), source="adversarial", current_qid="q1",
            full_pool_query_ids=FULL_POOL, mutated_self_global_indices={5, 6, 7, 8, 9}, n_adv_per_query=5,
        )
        self.assertEqual(out["origin_label"], audit.ORIGIN_CROSS_QUERY)
        self.assertEqual(out["true_global_index"], 12)
        self.assertEqual(out["true_owning_query_id"], "q2")

    def test_unknown_source_raises(self):
        with self.assertRaises(ValueError):
            audit.classify_passage_origin(
                doc_id="x", source="mystery", current_qid="q1",
                full_pool_query_ids=FULL_POOL, mutated_self_global_indices=set(),
            )

    def test_out_of_range_global_index_raises(self):
        with self.assertRaises(ValueError):
            audit.classify_passage_origin(
                doc_id=_adv("q1", 999), source="adversarial", current_qid="q1",
                full_pool_query_ids=FULL_POOL, mutated_self_global_indices=set(), n_adv_per_query=5,
            )


class TestBuildOriginRows(unittest.TestCase):
    def _passages(self):
        return [
            RetrievedPassage(doc_id=_adv("q1", 5), text="self0", source="adversarial", is_poison=True, rank=0, retrieval_score=0.9),
            RetrievedPassage(doc_id=_adv("q1", 12), text="cross0", source="adversarial", is_poison=True, rank=1, retrieval_score=0.8),
            RetrievedPassage(doc_id="clean_a", text="clean text", source="corpus", is_poison=False, rank=2, retrieval_score=0.5),
        ]

    def test_rows_have_correct_origin_and_removed_flags(self):
        passages = self._passages()
        removed = {"ragdefender": {_adv("q1", 5), "clean_a"}, "filterrag_semantic": {_adv("q1", 12)}}
        rows = audit.build_origin_rows(
            qid="q1", k=10, passages=passages, full_pool_query_ids=FULL_POOL,
            mutated_self_global_indices={5, 6, 7, 8, 9}, removed_doc_ids_by_defense=removed,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["origin_label"], audit.ORIGIN_MUTATED_SELF)
        self.assertEqual(rows[0]["rank"], 1)
        self.assertTrue(rows[0]["removed_by_ragdefender"])
        self.assertFalse(rows[0]["removed_by_filterrag_semantic"])
        self.assertEqual(rows[1]["origin_label"], audit.ORIGIN_CROSS_QUERY)
        self.assertEqual(rows[1]["true_owning_query_id"], "q2")
        self.assertTrue(rows[1]["removed_by_filterrag_semantic"])
        self.assertEqual(rows[2]["origin_label"], audit.ORIGIN_CLEAN)
        self.assertTrue(rows[2]["removed_by_ragdefender"])
        # Defenses not present in removed_doc_ids_by_defense default to False, never KeyError.
        for name in audit.DEFENSE_NAMES:
            self.assertIn(f"removed_by_{name}", rows[0])


class TestVerifyReplacementBudget(unittest.TestCase):
    def test_exactly_five_distinct_slots_passes(self):
        class _Slot:
            def __init__(self, gi):
                self.global_index = gi

        plan = {"q1": {i: _Slot(5 + i) for i in range(5)}}
        out = audit.verify_replacement_budget(plan, ["q1"])
        self.assertEqual(out["q1"]["n_slots_replaced"], 5)
        self.assertEqual(out["q1"]["n_distinct_global_indices"], 5)
        self.assertTrue(out["q1"]["exactly_5_replaced"])

    def test_fewer_than_five_fails(self):
        class _Slot:
            def __init__(self, gi):
                self.global_index = gi

        plan = {"q1": {i: _Slot(5 + i) for i in range(4)}}
        out = audit.verify_replacement_budget(plan, ["q1"])
        self.assertFalse(out["q1"]["exactly_5_replaced"])

    def test_missing_query_reports_zero_not_crash(self):
        out = audit.verify_replacement_budget({}, ["q_missing"])
        self.assertEqual(out["q_missing"]["n_slots_replaced"], 0)
        self.assertFalse(out["q_missing"]["exactly_5_replaced"])


class TestCountOriginsAndDuplicatesAndMaxFive(unittest.TestCase):
    def _rows(self):
        return [
            {"query_id": "q1", "origin_label": audit.ORIGIN_MUTATED_SELF, "true_global_index": 5},
            {"query_id": "q1", "origin_label": audit.ORIGIN_MUTATED_SELF, "true_global_index": 6},
            {"query_id": "q1", "origin_label": audit.ORIGIN_CROSS_QUERY, "true_global_index": 12},
            {"query_id": "q1", "origin_label": audit.ORIGIN_CLEAN, "true_global_index": None},
            {"query_id": "q2", "origin_label": audit.ORIGIN_ORIGINAL_SELF, "true_global_index": 9},
        ]

    def test_count_origins_per_query(self):
        counts = audit.count_origins_per_query(self._rows(), ["q1", "q2"])
        self.assertEqual(counts["q1"][audit.ORIGIN_MUTATED_SELF], 2)
        self.assertEqual(counts["q1"][audit.ORIGIN_CROSS_QUERY], 1)
        self.assertEqual(counts["q1"]["total_poison"], 3)
        self.assertEqual(counts["q1"]["total_clean"], 1)
        self.assertEqual(counts["q2"][audit.ORIGIN_ORIGINAL_SELF], 1)
        self.assertEqual(counts["q2"]["total_poison"], 1)

    def test_verify_no_original_self_poison_duplicate_finds_anomaly(self):
        dup = audit.verify_no_original_self_poison_duplicate(self._rows())
        self.assertEqual(len(dup), 1)
        self.assertEqual(dup[0]["query_id"], "q2")

    def test_verify_no_original_self_poison_duplicate_empty_when_clean(self):
        rows = [r for r in self._rows() if r["origin_label"] != audit.ORIGIN_ORIGINAL_SELF]
        self.assertEqual(audit.verify_no_original_self_poison_duplicate(rows), [])

    def test_max_five_mutated_self_not_exceeded(self):
        out = audit.verify_max_five_mutated_self_per_query(self._rows(), ["q1", "q2"])
        self.assertEqual(out["q1"]["n_mutated_self_retrieved"], 2)
        self.assertFalse(out["q1"]["exceeds_budget_of_5"])
        self.assertEqual(out["q2"]["n_mutated_self_retrieved"], 0)

    def test_max_five_mutated_self_flags_impossible_over_budget_case(self):
        rows = [
            {"query_id": "q1", "origin_label": audit.ORIGIN_MUTATED_SELF, "true_global_index": i}
            for i in range(6)
        ]
        out = audit.verify_max_five_mutated_self_per_query(rows, ["q1"])
        self.assertEqual(out["q1"]["n_mutated_self_retrieved"], 6)
        self.assertTrue(out["q1"]["exceeds_budget_of_5"])


class TestSummarizeRemovedByOrigin(unittest.TestCase):
    def test_tally_matches_manual_count(self):
        rows = [
            {"query_id": "q1", "origin_label": audit.ORIGIN_MUTATED_SELF, "removed_by_ragdefender": True, **{f"removed_by_{d}": False for d in audit.DEFENSE_NAMES if d != "ragdefender"}},
            {"query_id": "q1", "origin_label": audit.ORIGIN_CROSS_QUERY, "removed_by_ragdefender": True, **{f"removed_by_{d}": False for d in audit.DEFENSE_NAMES if d != "ragdefender"}},
            {"query_id": "q1", "origin_label": audit.ORIGIN_CLEAN, "removed_by_ragdefender": False, **{f"removed_by_{d}": False for d in audit.DEFENSE_NAMES if d != "ragdefender"}},
        ]
        out = audit.summarize_removed_by_origin(rows, ["q1"])
        self.assertEqual(out[("q1", "ragdefender")][audit.ORIGIN_MUTATED_SELF], 1)
        self.assertEqual(out[("q1", "ragdefender")][audit.ORIGIN_CROSS_QUERY], 1)
        self.assertEqual(out[("q1", "ragdefender")][audit.ORIGIN_CLEAN], 0)
        self.assertEqual(out[("q1", "filterrag_semantic")][audit.ORIGIN_MUTATED_SELF], 0)


class TestNoForbiddenCalls(unittest.TestCase):
    def test_no_forbidden_api_modules_imported(self):
        import ast

        tree = ast.parse(inspect.getsource(audit))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for forbidden in ("openai", "anthropic", "google.generativeai"):
            self.assertFalse(any(n == forbidden or n.startswith(forbidden + ".") for n in imported))

    def test_no_fit_or_llm_query_or_requests_post_calls(self):
        import ast

        tree = ast.parse(inspect.getsource(audit))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            attr = node.func
            receiver = attr.value.id if isinstance(attr.value, ast.Name) else None
            self.assertFalse(attr.attr == "post" and receiver == "requests")
            self.assertFalse(attr.attr == "query" and receiver == "llm")
            self.assertNotEqual(attr.attr, "fit")

    def test_offline_env_vars_forced(self):
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")

    def test_reuses_pilot_selected_query_ids(self):
        import run_full_retrieval_pilot_bundle1 as pilot

        self.assertIs(audit.SELECTED_QUERY_IDS, pilot.SELECTED_QUERY_IDS)


if __name__ == "__main__":
    unittest.main()
