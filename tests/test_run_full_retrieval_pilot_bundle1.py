"""Tests for scripts/run_full_retrieval_pilot_bundle1.py -- the full-retrieval
(real Contriever embedding + dot-product top-k) rerun of the 3 selected
normalized `filterrag_targeted` mutation cases.

Fully offline: no real Contriever/sentence-transformers/flan-t5/distilgpt2
model is downloaded or run, and no LLM/GPT/API call is ever made anywhere in
this file. Every test exercises pure logic (pool bookkeeping, replacement-
plan construction, budget assertions, merge/rank, survival stats) with
synthetic fixtures -- no heavy dependency is imported at test time beyond
what `scripts/run_full_retrieval_pilot_bundle1.py` itself already imports
at module scope (which does not include `torch`/`transformers`/`src.attack`/
`src.utils` -- those are imported lazily, inside `main()`/helper functions,
specifically so this test file can exercise the pure logic without ever
loading a real model).

Run with: python -m unittest tests.test_run_full_retrieval_pilot_bundle1 -v
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

import run_full_retrieval_pilot_bundle1 as m  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _poison_rows(qid, global_indices):
    """Mirrors mutation_input_passages.csv's row shape for one query_id's 5
    poison rows, `poison_slot` 0..4, doc_id encoding `global_indices[slot]`."""
    return [
        {
            "query_id": qid, "poison_slot": str(slot), "retrieved_rank": str(slot + 1),
            "doc_id": f"adv::LM_targeted::{qid}::{global_indices[slot]}",
            "original_poison_text": f"question for {qid}.original body {slot}",
        }
        for slot in range(5)
    ]


def _normalized_record_line(qid, global_indices, *, family="filterrag_targeted", mutate_slot_text=None):
    mutate_slot_text = mutate_slot_text or {}
    return json.dumps({
        "query_id": qid, "k": 10, "family": family, "intended_defense": "filterrag",
        "selection_role": "primary", "question": f"question for {qid}?",
        "target_wrong_answer": "wrong",
        "mutated_passages": [
            {
                "poison_slot": slot,
                "doc_id": f"adv::LM_targeted::{qid}::{global_indices[slot]}",
                "source_file_doc_id": f"adv::LM_targeted::{qid}::{slot}",
                "mutated_text": mutate_slot_text.get(slot, f"mutated body {slot} for {qid}"),
                "doc_id_mismatch": False, "quality_flags": [],
            }
            for slot in range(5)
        ],
    }) + "\n"


def _write_temp(contents: str, suffix: str = ".jsonl") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(contents)
    return path


# ---------------------------------------------------------------------------
# 1. load_full_pool_query_ids / extract_global_index
# ---------------------------------------------------------------------------

class TestPoolAndDocIdHelpers(unittest.TestCase):
    def test_load_full_pool_query_ids_returns_ordered_list(self):
        path = _write_temp(json.dumps({"target_query_ids": ["qA", "qB", "qC"]}), suffix=".json")
        try:
            pool = m.load_full_pool_query_ids(path)
        finally:
            os.remove(path)
        self.assertEqual(pool, ["qA", "qB", "qC"])

    def test_load_full_pool_query_ids_raises_on_missing_key(self):
        path = _write_temp(json.dumps({"other_key": []}), suffix=".json")
        try:
            with self.assertRaises(ValueError):
                m.load_full_pool_query_ids(path)
        finally:
            os.remove(path)

    def test_extract_global_index_parses_trailing_int(self):
        self.assertEqual(m.extract_global_index("adv::LM_targeted::q1::201"), 201)
        self.assertEqual(m.extract_global_index("adv::LM_targeted::q1::0"), 0)

    def test_extract_global_index_raises_on_bad_shape(self):
        with self.assertRaises(ValueError):
            m.extract_global_index("not-an-adv-doc-id")
        with self.assertRaises(ValueError):
            m.extract_global_index("adv::LM_targeted::q1::not_an_int")


# ---------------------------------------------------------------------------
# 2. load_normalized_family_file
# ---------------------------------------------------------------------------

class TestLoadNormalizedFamilyFile(unittest.TestCase):
    def test_parses_valid_jsonl(self):
        contents = _normalized_record_line("q1", [10, 11, 12, 13, 14]) + _normalized_record_line("q2", [20, 21, 22, 23, 24])
        path = _write_temp(contents)
        try:
            by_qid = m.load_normalized_family_file(path)
        finally:
            os.remove(path)
        self.assertEqual(set(by_qid), {"q1", "q2"})
        self.assertEqual(by_qid["q1"].family, "filterrag_targeted")
        self.assertEqual(by_qid["q1"].slots[0].mutated_text, "mutated body 0 for q1")
        self.assertEqual(by_qid["q1"].slots[0].doc_id, "adv::LM_targeted::q1::10")

    def test_wrong_slot_count_raises(self):
        rec = json.loads(_normalized_record_line("q1", [10, 11, 12, 13, 14]))
        rec["mutated_passages"].pop()
        path = _write_temp(json.dumps(rec) + "\n")
        try:
            with self.assertRaises(ValueError):
                m.load_normalized_family_file(path)
        finally:
            os.remove(path)

    def test_duplicate_poison_slot_raises(self):
        rec = json.loads(_normalized_record_line("q1", [10, 11, 12, 13, 14]))
        rec["mutated_passages"][1]["poison_slot"] = 0
        path = _write_temp(json.dumps(rec) + "\n")
        try:
            with self.assertRaises(ValueError):
                m.load_normalized_family_file(path)
        finally:
            os.remove(path)

    def test_empty_mutated_text_raises(self):
        rec = json.loads(_normalized_record_line("q1", [10, 11, 12, 13, 14]))
        rec["mutated_passages"][0]["mutated_text"] = "   "
        path = _write_temp(json.dumps(rec) + "\n")
        try:
            with self.assertRaises(ValueError):
                m.load_normalized_family_file(path)
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
# 3. build_replacement_plan -- doc_id/poison_slot/query_id mapping preserved.
# ---------------------------------------------------------------------------

class TestBuildReplacementPlan(unittest.TestCase):
    def _poison_by_query(self):
        return {
            "q1": _poison_rows("q1", [10, 11, 12, 13, 14]),
            "q2": _poison_rows("q2", [20, 21, 22, 23, 24]),
        }

    def _normalized_by_qid(self):
        contents = (
            _normalized_record_line("q1", [10, 11, 12, 13, 14])
            + _normalized_record_line("q2", [20, 21, 22, 23, 24])
        )
        path = _write_temp(contents)
        try:
            return m.load_normalized_family_file(path)
        finally:
            os.remove(path)

    def test_plan_preserves_query_id_slot_doc_id_mapping(self):
        plan = m.build_replacement_plan(["q1", "q2"], self._poison_by_query(), self._normalized_by_qid())
        self.assertEqual(set(plan), {"q1", "q2"})
        for qid, global_indices in (("q1", [10, 11, 12, 13, 14]), ("q2", [20, 21, 22, 23, 24])):
            for slot in range(5):
                r = plan[qid][slot]
                self.assertEqual(r.query_id, qid)
                self.assertEqual(r.poison_slot, slot)
                self.assertEqual(r.original_doc_id, f"adv::LM_targeted::{qid}::{global_indices[slot]}")
                self.assertEqual(r.global_index, global_indices[slot])
                self.assertEqual(r.mutation_family, "filterrag_targeted")
                self.assertEqual(r.bundle_id, "filterrag_targeted")

    def test_missing_selected_query_id_raises(self):
        with self.assertRaises(ValueError):
            m.build_replacement_plan(["q1", "q_missing"], self._poison_by_query(), self._normalized_by_qid())

    def test_doc_id_mismatch_between_csv_and_normalized_raises(self):
        poison_by_query = self._poison_by_query()
        normalized_by_qid = self._normalized_by_qid()
        # Corrupt the normalized record's doc_id for one slot so it no longer
        # matches the CSV's canonical doc_id -- mapping preservation must be
        # enforced, not silently accepted.
        normalized_by_qid["q1"].slots[0].doc_id = "adv::LM_targeted::q1::999"
        with self.assertRaises(ValueError):
            m.build_replacement_plan(["q1"], poison_by_query, normalized_by_qid)

    def test_wrong_family_raises(self):
        poison_by_query = {"q1": _poison_rows("q1", [10, 11, 12, 13, 14])}
        contents = _normalized_record_line("q1", [10, 11, 12, 13, 14], family="ragdefender_targeted")
        path = _write_temp(contents)
        try:
            normalized_by_qid = m.load_normalized_family_file(path)
        finally:
            os.remove(path)
        with self.assertRaises(ValueError):
            m.build_replacement_plan(["q1"], poison_by_query, normalized_by_qid)


# ---------------------------------------------------------------------------
# 4. apply_replacements / assert_budget_preserved -- the core budget /
#    no-duplication / exact-insertion guarantees required by the task.
# ---------------------------------------------------------------------------

class TestApplyReplacementsAndBudget(unittest.TestCase):
    def _plan_and_pool(self):
        poison_by_query = {
            "q1": _poison_rows("q1", [10, 11, 12, 13, 14]),
            "q2": _poison_rows("q2", [20, 21, 22, 23, 24]),
        }
        contents = (
            _normalized_record_line("q1", [10, 11, 12, 13, 14])
            + _normalized_record_line("q2", [20, 21, 22, 23, 24])
        )
        path = _write_temp(contents)
        try:
            normalized_by_qid = m.load_normalized_family_file(path)
        finally:
            os.remove(path)
        plan = m.build_replacement_plan(["q1", "q2"], poison_by_query, normalized_by_qid)

        # Baseline pool: 5 dummy texts per query id "q1".."q5" (250-like pool
        # scaled down), with q1/q2's own poison text matching what
        # build_replacement_plan recorded as original_poison_text (required
        # for apply_replacements' internal cross-check to pass).
        baseline_pool = ["clean-pool-filler"] * 30
        for qid, indices in (("q1", [10, 11, 12, 13, 14]), ("q2", [20, 21, 22, 23, 24])):
            for slot, idx in enumerate(indices):
                baseline_pool[idx] = f"question for {qid}.original body {slot}"
        return plan, baseline_pool

    def test_replacement_does_not_increase_poison_budget(self):
        plan, baseline_pool = self._plan_and_pool()
        mutated_pool, replaced = m.apply_replacements(baseline_pool, plan)
        self.assertEqual(len(mutated_pool), len(baseline_pool))
        self.assertEqual(len(replaced), 10)  # 2 queries x 5 slots, never more

    def test_original_and_mutated_poison_are_not_both_present(self):
        plan, baseline_pool = self._plan_and_pool()
        mutated_pool, replaced = m.apply_replacements(baseline_pool, plan)
        for idx in replaced:
            self.assertNotEqual(mutated_pool[idx], baseline_pool[idx])
            self.assertNotIn(baseline_pool[idx], mutated_pool)  # original text is gone, not duplicated

    def test_mutated_text_inserted_exactly(self):
        plan, baseline_pool = self._plan_and_pool()
        mutated_pool, _ = m.apply_replacements(baseline_pool, plan)
        for qid, indices in (("q1", [10, 11, 12, 13, 14]), ("q2", [20, 21, 22, 23, 24])):
            for slot, idx in enumerate(indices):
                self.assertEqual(mutated_pool[idx], plan[qid][slot].mutated_text)
                self.assertEqual(mutated_pool[idx], f"mutated body {slot} for {qid}")

    def test_other_pool_entries_untouched(self):
        plan, baseline_pool = self._plan_and_pool()
        mutated_pool, replaced = m.apply_replacements(baseline_pool, plan)
        replaced_set = set(replaced)
        for idx in range(len(baseline_pool)):
            if idx not in replaced_set:
                self.assertEqual(mutated_pool[idx], baseline_pool[idx])

    def test_assert_budget_preserved_passes_on_correct_replacement(self):
        plan, baseline_pool = self._plan_and_pool()
        mutated_pool, replaced = m.apply_replacements(baseline_pool, plan)
        m.assert_budget_preserved(baseline_pool, mutated_pool, replaced, n_selected_queries=2)  # no raise

    def test_assert_budget_preserved_catches_augmentation(self):
        plan, baseline_pool = self._plan_and_pool()
        mutated_pool, replaced = m.apply_replacements(baseline_pool, plan)
        augmented_pool = mutated_pool + ["extra poison text"]  # pool grew -> augmentation
        with self.assertRaises(AssertionError):
            m.assert_budget_preserved(baseline_pool, augmented_pool, replaced, n_selected_queries=2)

    def test_assert_budget_preserved_catches_collateral_edit(self):
        plan, baseline_pool = self._plan_and_pool()
        mutated_pool, replaced = m.apply_replacements(baseline_pool, plan)
        mutated_pool[0] = "unexpectedly edited filler text"  # not part of the plan
        with self.assertRaises(AssertionError):
            m.assert_budget_preserved(baseline_pool, mutated_pool, replaced, n_selected_queries=2)

    def test_assert_budget_preserved_catches_missing_replacement(self):
        plan, baseline_pool = self._plan_and_pool()
        mutated_pool, replaced = m.apply_replacements(baseline_pool, plan)
        mutated_pool[replaced[0]] = baseline_pool[replaced[0]]  # revert one replacement
        with self.assertRaises(AssertionError):
            m.assert_budget_preserved(baseline_pool, mutated_pool, replaced, n_selected_queries=2)

    def test_apply_replacements_detects_pool_reconstruction_drift(self):
        plan, baseline_pool = self._plan_and_pool()
        baseline_pool[10] = "this does not match the archived original_poison_text"
        with self.assertRaises(AssertionError):
            m.apply_replacements(baseline_pool, plan)


# ---------------------------------------------------------------------------
# 5. merge_and_topk -- pure retrieval merge/rank logic.
# ---------------------------------------------------------------------------

class TestMergeAndTopk(unittest.TestCase):
    def test_merges_clean_and_adversarial_sorted_by_score(self):
        clean_entries = [
            {"score": 0.5, "context": "clean A", "doc_id": "c1"},
            {"score": 0.3, "context": "clean B", "doc_id": "c2"},
        ]
        adv_texts = ["adv0", "adv1", "adv2"]
        adv_scores = [0.9, 0.1, 0.6]
        topk = m.merge_and_topk(clean_entries, adv_texts, adv_scores, qid="q1", k=4)
        self.assertEqual(len(topk), 4)
        self.assertEqual([e["score"] for e in topk], sorted([0.5, 0.3, 0.9, 0.1, 0.6], reverse=True)[:4])
        self.assertEqual(topk[0]["doc_id"], "adv::LM_targeted::q1::0")  # score 0.9, is_poison
        self.assertTrue(topk[0]["is_poison"])

    def test_doc_id_uses_global_index_and_qid(self):
        adv_texts = ["a", "b"]
        adv_scores = [1.0, 2.0]
        topk = m.merge_and_topk([], adv_texts, adv_scores, qid="qX", k=2)
        doc_ids = {e["doc_id"] for e in topk}
        self.assertEqual(doc_ids, {"adv::LM_targeted::qX::0", "adv::LM_targeted::qX::1"})

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            m.merge_and_topk([], ["a", "b"], [1.0], qid="q1", k=5)

    def test_truncates_to_k(self):
        clean_entries = [{"score": float(i), "context": f"c{i}", "doc_id": f"c{i}"} for i in range(20)]
        topk = m.merge_and_topk(clean_entries, [], [], qid="q1", k=3)
        self.assertEqual(len(topk), 3)


# ---------------------------------------------------------------------------
# 6. retrieval_survival_stats
# ---------------------------------------------------------------------------

class TestRetrievalSurvivalStats(unittest.TestCase):
    def _passages(self, poison_doc_ids_present, n_clean=5):
        passages = []
        rank = 0
        for d in poison_doc_ids_present:
            passages.append(RetrievedPassage(doc_id=d, text="poison", source="adversarial", is_poison=True, rank=rank))
            rank += 1
        for i in range(n_clean):
            passages.append(RetrievedPassage(doc_id=f"clean{i}", text="clean", source="corpus", is_poison=False, rank=rank))
            rank += 1
        return passages

    def test_all_5_survive(self):
        canonical = [f"p{i}" for i in range(5)]
        passages = self._passages(canonical)
        stats = m.retrieval_survival_stats(passages, canonical)
        self.assertEqual(stats["canonical_poison_survived_count"], 5)
        self.assertTrue(stats["all_5_poison_survive"])
        self.assertEqual(stats["retrieval_survival_rate"], 1.0)
        self.assertEqual(stats["canonical_poison_survived_ranks"], "1;2;3;4;5")

    def test_partial_survival(self):
        canonical = [f"p{i}" for i in range(5)]
        passages = self._passages(canonical[:2], n_clean=8)  # only 2 of 5 survived
        stats = m.retrieval_survival_stats(passages, canonical)
        self.assertEqual(stats["canonical_poison_survived_count"], 2)
        self.assertFalse(stats["all_5_poison_survive"])
        self.assertAlmostEqual(stats["retrieval_survival_rate"], 0.4)

    def test_no_survival(self):
        canonical = [f"p{i}" for i in range(5)]
        passages = self._passages([], n_clean=10)
        stats = m.retrieval_survival_stats(passages, canonical)
        self.assertEqual(stats["canonical_poison_survived_count"], 0)
        self.assertIsNone(stats["mean_poison_retrieval_rank"])
        self.assertEqual(stats["retrieval_survival_rate"], 0.0)


# ---------------------------------------------------------------------------
# 7. build_full_pool_adv_text_list -- pool order/indexing (mocked Attacker,
#    no real model/tokenizer).
# ---------------------------------------------------------------------------

class TestBuildFullPoolAdvTextList(unittest.TestCase):
    def test_scope_is_exactly_the_given_pool_and_index_matches_pool_position(self):
        """Retrieval/pool construction must only ever be built for the
        query_ids explicitly passed in -- never silently expanded to "all
        queries in incorrect_answers"."""
        incorrect_answers = {
            "qA": {"question": "Question A?"},
            "qB": {"question": "Question B?"},
            "qC": {"question": "Question C? (not in pool)"},
        }
        full_pool_query_ids = ["qA", "qB"]  # qC intentionally excluded

        class _FakeAttacker:
            def __init__(self, args, **kwargs):
                self.adv_per_query = args.adv_per_query

            def get_attack(self, target_queries):
                # Mirrors Attacker.get_attack's LM_targeted branch: pure
                # string templating, no model/tokenizer call.
                return [
                    [f"{tq['query']}.body{i}" for i in range(self.adv_per_query)]
                    for tq in target_queries
                ]

        import unittest.mock as mock

        with mock.patch("src.attack.Attacker", _FakeAttacker):
            adv_text_list = m.build_full_pool_adv_text_list(
                full_pool_query_ids, incorrect_answers,
                model=None, c_model=None, tokenizer=None, get_emb=None,
            )

        self.assertEqual(len(adv_text_list), 2 * 5)
        self.assertTrue(adv_text_list[0].startswith("Question A?"))
        self.assertTrue(adv_text_list[5].startswith("Question B?"))
        self.assertFalse(any("Question C?" in t for t in adv_text_list))


# ---------------------------------------------------------------------------
# 8. Static checks: no GPT/API/training calls, retrieval scope limited to
#    SELECTED_QUERY_IDS.
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

    def test_no_requests_post_or_llm_query_or_fit_calls(self):
        """AST check: no `requests.post(...)`, no `llm.query(...)`, and no
        `.fit(...)` call anywhere (this script must never train/retrain the
        ML-FilterRAG classifier or any other model -- it only ever calls
        `.predict_proba(...)` via the reused, unmodified
        `defense.ml_filterrag` / `defense.dispatch` code)."""
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
            self.assertNotEqual(method_name, "fit", msg="found a `.fit(...)` call -- no model may be trained here")

    def test_offline_env_vars_are_forced(self):
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")

    def test_selected_query_ids_are_exactly_the_3_task_queries(self):
        self.assertEqual(
            set(m.SELECTED_QUERY_IDS),
            {
                "5a8e068b5542995085b37384",
                "5ae224da554299234fd043ee",
                "5ae22b8d554299234fd0440f",
            },
        )
        self.assertEqual(len(m.SELECTED_QUERY_IDS), 3)

    def test_main_only_iterates_selected_query_ids_for_retrieval_and_scoring(self):
        """Structural check: the per-query retrieval/scoring loop in main()
        iterates `for qid in SELECTED_QUERY_IDS`, never over
        `full_pool_query_ids` or any other larger collection -- i.e.
        retrieval is rerun (and defenses scored) only for the 3 selected
        queries, not the full 50-query pool."""
        source = inspect.getsource(m.main)
        # Every top-level "for qid in ..." loop in main() (validation loops
        # plus the retrieval/scoring loop) iterates SELECTED_QUERY_IDS --
        # never the full 50-query pool or any other larger collection.
        self.assertGreaterEqual(source.count("for qid in SELECTED_QUERY_IDS:"), 2)
        self.assertNotIn("for qid in full_pool_query_ids:", source)


if __name__ == "__main__":
    unittest.main()
