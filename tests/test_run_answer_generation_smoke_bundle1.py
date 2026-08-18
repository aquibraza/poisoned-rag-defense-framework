"""Tests for scripts/run_answer_generation_smoke_bundle1.py.

Fully offline: no GPT/API call, no llm.query(), no Contriever, no defense
model load. Reconstruction, condition enumeration, ASR fields, and report
logic are exercised with fixtures (plus a light read of the published
full-retrieval artifacts when they are present).

Run with: python -m unittest tests.test_run_answer_generation_smoke_bundle1 -v
"""
import ast
import csv
import inspect
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from defense.passages import RetrievedPassage  # noqa: E402

import run_answer_generation_smoke_bundle1 as m  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_RET_DIR = os.path.join(
    REPO_ROOT,
    "manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/full_retrieval_pilot",
)


def _p(doc_id, text, *, poison=True, rank=0):
    return RetrievedPassage(
        doc_id=doc_id, text=text,
        source="adversarial" if poison else "corpus",
        is_poison=poison, rank=rank,
    )


class TestConditionBudget(unittest.TestCase):
    def test_exactly_ten_conditions_and_thirty_calls(self):
        conds = m.enumerate_conditions()
        self.assertEqual(len(conds), 10)
        self.assertEqual(m.n_generation_calls(), 30)
        self.assertEqual(len(m.ORIGINAL_CONDITIONS), 4)
        self.assertEqual(len(m.MUTATED_CONDITIONS), 6)
        self.assertIn(("original", "ml_filterrag", 0.4), conds)
        self.assertNotIn(("original", "ml_filterrag", 0.35), conds)
        self.assertNotIn(("original", "ml_filterrag", 0.5), conds)
        for t in (0.35, 0.4, 0.5):
            self.assertIn(("mutated", "ml_filterrag", t), conds)

    def test_only_the_three_named_queries(self):
        self.assertEqual(
            list(m.SELECTED_QUERY_IDS),
            [
                "5a8e068b5542995085b37384",
                "5ae224da554299234fd043ee",
                "5ae22b8d554299234fd0440f",
            ],
        )


class TestLmTargetedTemplate(unittest.TestCase):
    def test_matches_attacker_get_attack_concatenation(self):
        answers = {
            "q_owner": {
                "question": "Are both Dictyosperma, and Huernia described as a genus?",
                "adv_texts": ["body0", "body1", "body2", "body3", "body4"],
            }
        }
        self.assertEqual(
            m.lm_targeted_text(answers, "q_owner", 2),
            "Are both Dictyosperma, and Huernia described as a genus?.body2",
        )

    def test_owning_query_and_slot(self):
        pool = ["q0", "q1", "q2"]
        self.assertEqual(m.owning_query_and_slot(12, pool), ("q2", 2))
        self.assertEqual(m.owning_query_and_slot(0, pool), ("q0", 0))


class TestOriginalReconstruction(unittest.TestCase):
    def test_inserts_missing_cross_query_rank(self):
        poison = [
            {
                "doc_id": "adv::LM_targeted::qA::10",
                "original_poison_text": "self poison",
                "retrieved_rank": "1",
                "poison_slot": "0",
            }
        ]
        clean = [
            {"doc_id": "c1", "clean_text": "clean one", "retrieved_rank": "2"},
        ]
        # archived is 2 passages; audit says rank 3 is a cross-query poison at global 12
        audit = {
            "qA": [
                {"rank": "1", "doc_id": "adv::LM_targeted::qA::10", "is_poison": "True", "true_global_index": "10"},
                {"rank": "2", "doc_id": "c1", "is_poison": "False", "true_global_index": ""},
                {"rank": "3", "doc_id": "adv::LM_targeted::qA::12", "is_poison": "True", "true_global_index": "12"},
            ]
        }
        # pad to k=3 for this fixture by monkeypatching K
        answers = {
            "q2": {"question": "other q", "adv_texts": ["a0", "a1", "a2", "a3", "a4"]},
        }
        pool = ["q0", "q1", "q2"]
        orig_k = m.K
        try:
            m.K = 3
            out = m.reconstruct_original_contexts(
                selected_query_ids=["qA"],
                poison_by_query={"qA": poison},
                clean_by_query={"qA": clean},
                audit_rows_by_query=audit,
                incorrect_answers=answers,
                full_pool_query_ids=pool,
            )
        finally:
            m.K = orig_k
        passages = out["qA"]
        self.assertEqual(len(passages), 3)
        self.assertEqual(passages[2].doc_id, "adv::LM_targeted::qA::12")
        self.assertTrue(passages[2].is_poison)
        self.assertEqual(passages[2].text, "other q.a2")


class TestMutatedReconstruction(unittest.TestCase):
    def test_uses_mutated_text_and_falls_back_to_template(self):
        audit = {
            "qA": [
                {
                    "rank": "1", "doc_id": "adv::LM_targeted::qA::10",
                    "is_poison": "True", "source": "adversarial",
                    "true_global_index": "10", "retrieval_score": "1.5",
                },
                {
                    "rank": "2", "doc_id": "c1",
                    "is_poison": "False", "source": "corpus",
                    "true_global_index": "", "retrieval_score": "0.9",
                },
            ]
        }
        lookup = {
            "adv::LM_targeted::qA::10": "mutated self text",
            "c1": "clean text",
        }
        orig_k = m.K
        try:
            m.K = 2
            out = m.reconstruct_mutated_contexts(
                selected_query_ids=["qA"],
                audit_rows_by_query=audit,
                lookup=lookup,
                incorrect_answers={},
                full_pool_query_ids=["q0", "q1", "qA"],
            )
        finally:
            m.K = orig_k
        self.assertEqual(out["qA"][0].text, "mutated self text")
        self.assertTrue(out["qA"][0].is_poison)
        self.assertEqual(out["qA"][1].text, "clean text")
        self.assertFalse(out["qA"][1].is_poison)


class TestAuditRemovals(unittest.TestCase):
    def test_none_keeps_all(self):
        passages = [_p("a", "t1", rank=0), _p("b", "t2", poison=False, rank=1)]
        kept = m.apply_audit_removals(passages, [], "none", None)
        self.assertEqual([p.doc_id for p in kept], ["a", "b"])

    def test_flag_filters_removed_doc_ids(self):
        passages = [_p("a", "t1", rank=0), _p("b", "t2", poison=False, rank=1)]
        audit = [
            {"doc_id": "a", "removed_by_ragdefender": "True"},
            {"doc_id": "b", "removed_by_ragdefender": "False"},
        ]
        kept = m.apply_audit_removals(passages, audit, "ragdefender", None)
        self.assertEqual([p.doc_id for p in kept], ["b"])


class TestAsrFields(unittest.TestCase):
    def test_strict_no_does_not_match_not(self):
        out = m.asr_fields(
            generated_answer="They are not both plants.",
            target_wrong_answer="no",
            correct_answer="yes",
        )
        self.assertFalse(out["strict_asr_success"])
        self.assertTrue(out["contains_target_wrong_answer"])  # legacy substring
        self.assertFalse(out["contains_correct_answer"])

    def test_strict_yes_standalone(self):
        out = m.asr_fields(
            generated_answer="Yes, both drinks contain gin.",
            target_wrong_answer="yes",
            correct_answer="no",
        )
        self.assertTrue(out["strict_asr_success"])
        self.assertTrue(out["contains_target_wrong_answer"])

    def test_multiword_target(self):
        out = m.asr_fields(
            generated_answer="He was voted World's Best Defender in 1992.",
            target_wrong_answer="World's Best Defender",
            correct_answer="World's Best Goalkeeper",
        )
        self.assertTrue(out["strict_asr_success"])
        self.assertFalse(out["contains_correct_answer"])

    def test_dry_run_none(self):
        out = m.asr_fields(
            generated_answer=None, target_wrong_answer="no", correct_answer="yes"
        )
        self.assertIsNone(out["strict_asr_success"])
        self.assertIsNone(out["normalized_output"])


class TestReportAnswers(unittest.TestCase):
    def _row(self, **kwargs):
        base = {
            "query_id": "q1",
            "question": "Q?",
            "target_wrong_answer": "no",
            "correct_answer": "yes",
            "context_type": "mutated",
            "defense_name": "none",
            "threshold": None,
            "retrieved_poison_count": 5,
            "removed_poison": 0,
            "remaining_poison_count": 5,
            "remaining_clean_count": 5,
            "residual_poison_fraction": 1.0,
            "generator_model": "gpt-3.5-turbo",
            "generated_answer": "No",
            "strict_asr_success": False,
            "contains_target_wrong_answer": False,
            "contains_correct_answer": False,
            "notes": "",
        }
        base.update(kwargs)
        return base

    def test_report_says_no_when_no_defended_hit(self):
        rows = [
            self._row(query_id="5a8e068b5542995085b37384", defense_name="none", strict_asr_success=True),
            self._row(query_id="5a8e068b5542995085b37384", defense_name="ragdefender", strict_asr_success=False, remaining_poison_count=1, residual_poison_fraction=0.2, removed_poison=4),
            self._row(query_id="5ae224da554299234fd043ee", defense_name="ml_filterrag", threshold=0.4, removed_poison=0, remaining_poison_count=5, residual_poison_fraction=1.0, strict_asr_success=False),
            self._row(query_id="5ae22b8d554299234fd0440f", defense_name="filterrag_semantic", threshold=0.2, strict_asr_success=False),
        ]
        report = m.build_report(
            summary_rows=rows, generator_model="gpt-3.5-turbo",
            n_calls=30, dry_run=False, out_dir="tmp",
        )
        self.assertIn("**1. Did any mutated defended context produce the target wrong answer?** No", report)
        self.assertIn("**4. Did ML-FilterRAG full evasion produce downstream wrong answers?** No", report)

    def test_report_says_yes_on_defended_hit(self):
        rows = [
            self._row(
                query_id="5ae224da554299234fd043ee",
                defense_name="ml_filterrag", threshold=0.4,
                removed_poison=0, remaining_poison_count=5,
                residual_poison_fraction=1.0, strict_asr_success=True,
                generated_answer="Yes",
            )
        ]
        report = m.build_report(
            summary_rows=rows, generator_model="gpt-3.5-turbo",
            n_calls=30, dry_run=False, out_dir="tmp",
        )
        self.assertIn("**1. Did any mutated defended context produce the target wrong answer?** Yes", report)
        self.assertIn("**4. Did ML-FilterRAG full evasion produce downstream wrong answers?** Yes", report)
        self.assertIn("clear hits", report)
        self.assertIn("Scale to the remaining", report)


class TestNoApiAtImportAndNoDefenseEdits(unittest.TestCase):
    def test_module_source_has_no_top_level_create_model_or_llm_query(self):
        src = inspect.getsource(m)
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                text = ast.get_source_segment(src, node) or ""
                self.assertNotIn("create_model", text)
                self.assertNotIn("openai", text.lower())

    def test_create_model_is_lazy_inside_main(self):
        src = inspect.getsource(m.main)
        self.assertIn("create_model", src)
        self.assertIn("dry_run", src)

    def test_does_not_import_defense_package_for_mutation(self):
        # The script may import defense helpers, but must not write defense/*.py.
        defense_dir = os.path.join(REPO_ROOT, "defense")
        # Sanity: this test file itself is not a defense edit.
        self.assertTrue(os.path.isdir(defense_dir))


@unittest.skipUnless(
    os.path.exists(os.path.join(FULL_RET_DIR, "full_retrieval_poison_origin_breakdown.csv")),
    "full-retrieval pilot artifacts not present",
)
class TestRealArtifactReconstruction(unittest.TestCase):
    def test_mutated_membership_matches_audit_and_published_poison_counts(self):
        audit_path = os.path.join(FULL_RET_DIR, "full_retrieval_poison_origin_breakdown.csv")
        scores_path = os.path.join(FULL_RET_DIR, "full_retrieval_defense_scores.csv")
        cand_path = os.path.join(FULL_RET_DIR, "full_retrieval_candidate_inputs.jsonl")
        pilot_dir = os.path.join(REPO_ROOT, "manual_text_mutation_pilot/hotpotqa_50q_k10")
        with open(audit_path, newline="", encoding="utf-8") as f:
            audit_rows = list(csv.DictReader(f))
        with open(scores_path, newline="", encoding="utf-8") as f:
            scores = list(csv.DictReader(f))
        candidates = m.load_jsonl(cand_path)
        clean_by_query = {}
        with open(os.path.join(pilot_dir, "clean_context_passages.csv"), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                clean_by_query.setdefault(r["query_id"], []).append(r)
        answers = m.load_json(os.path.join(REPO_ROOT, "results/adv_targeted_results/hotpotqa.json"))
        pool = m.load_full_pool_query_ids(
            os.path.join(REPO_ROOT, "results/diagnostics/ml_filterrag_dataset_hotpotqa_50q/dataset_config.json")
        )
        audit_by_query = m.group_by_query(audit_rows)
        lookup = m.mutated_text_lookup(
            candidate_rows=candidates,
            clean_by_query=clean_by_query,
            incorrect_answers=answers,
            full_pool_query_ids=pool,
        )
        mutated = m.reconstruct_mutated_contexts(
            selected_query_ids=m.SELECTED_QUERY_IDS,
            audit_rows_by_query=audit_by_query,
            lookup=lookup,
            incorrect_answers=answers,
            full_pool_query_ids=pool,
        )
        published = {
            r["query_id"]: int(float(r["N_retrieved_poison"]))
            for r in scores
            if r["condition"] == "mutated" and r["family"] == "filterrag_targeted"
        }
        for qid in m.SELECTED_QUERY_IDS:
            n_poison, n_clean = m.count_poison_clean(mutated[qid])
            self.assertEqual(n_poison + n_clean, 10)
            self.assertEqual(n_poison, published[qid])
            self.assertTrue(all(p.text.strip() for p in mutated[qid]))
            # All 5 mutated self-query slots must be present.
            self.assertGreaterEqual(
                sum(1 for r in audit_by_query[qid] if r["origin_label"] == "mutated_self_query_poison"),
                5,
            )


if __name__ == "__main__":
    unittest.main()
