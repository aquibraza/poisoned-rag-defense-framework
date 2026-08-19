"""Tests for defense/robustrag_kw.py (RobustRAG-KW proxy).

Dependency-free by construction: `defense/robustrag_kw.py` is stdlib-only and
takes its generator by injection (`generate_fn`), so every test here runs on a
bare system `python3` -- no torch, no transformers, no sentence-transformers,
and crucially **no LLM/API call of any kind**. Generators in these tests are
either canned lists or a stub that raises if it is ever invoked.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import unittest
from collections import OrderedDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from defense.asr_match import _legacy_clean_str, strict_match  # noqa: E402
from defense.passages import RetrievedPassage  # noqa: E402
from defense.robustrag_kw import (  # noqa: E402
    ABSTAIN_ANSWER,
    ABSTENTION_PHRASES,
    AGGREGATION_MODES,
    CacheKey,
    GenerationCache,
    IsolatedAnswer,
    NORMALIZATION_MODES,
    RobustRagKwConfig,
    aggregate_isolated,
    aggregate_votes,
    build_groups,
    decide,
    extract_short_answer,
    is_abstention,
    normalize_answer,
    prompt_hash,
    raising_generate_fn,
    robustrag_kw_answer,
)

DEFENSE_MODULE = os.path.join(REPO_ROOT, "defense", "robustrag_kw.py")
PILOT_SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_robustrag_kw_pilot_bundle1.py")


def make_passage(doc_id, text, is_poison=False, rank=None):
    return RetrievedPassage(
        doc_id=doc_id,
        text=text,
        source="adversarial" if is_poison else "corpus",
        is_poison=is_poison,
        retrieval_score=None,
        rank=rank,
    )


def make_isolated(normalized, abstain=False, extracted=None, **kwargs):
    """Minimal IsolatedAnswer for aggregation-only tests."""
    defaults = dict(
        group_index=kwargs.pop("group_index", 0),
        doc_ids=kwargs.pop("doc_ids", ["d"]),
        ranks=kwargs.pop("ranks", [0]),
        prompt="p",
        prompt_hash="h",
        model_name="fake",
        query_id="q",
        context_type="mutated",
        cache_hit=False,
        raw_answer=kwargs.pop("raw_answer", normalized),
        extracted_answer=extracted if extracted is not None else normalized,
        normalized_answer=normalized,
        is_clean=kwargs.pop("is_clean", True),
        is_poison=kwargs.pop("is_poison", False),
        is_self_query_poison=kwargs.pop("is_self_query_poison", False),
        is_cross_query_poison=kwargs.pop("is_cross_query_poison", False),
        matches_target_wrong_answer_strict=None,
        matches_correct_answer_strict=None,
        is_abstain=abstain,
    )
    defaults.update(kwargs)
    return IsolatedAnswer(**defaults)


class TestNormalization(unittest.TestCase):
    def test_all_modes_on_a_fixed_table(self):
        text = "  The World's Best Defender.  "
        self.assertEqual(normalize_answer(text, "raw"), text)
        self.assertEqual(normalize_answer(text, "legacy_clean"),
                         "the world's best defender")
        self.assertEqual(normalize_answer(text, "squad"), "worlds best defender")
        self.assertEqual(normalize_answer(text, "token"),
                         "the world's best defender")

    def test_none_passes_through_in_every_mode(self):
        for mode in NORMALIZATION_MODES:
            self.assertIsNone(normalize_answer(None, mode))

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            normalize_answer("x", "not_a_mode")

    def test_legacy_clean_is_byte_identical_to_asr_match(self):
        for sample in ["Yes.", "  NO  ", "World's Best Defender.", "a.b.", ""]:
            self.assertEqual(normalize_answer(sample, "legacy_clean"),
                             _legacy_clean_str(sample))

    def test_squad_mode_agrees_with_contriever_normalize_answer(self):
        try:
            from src.contriever_src.evaluation import normalize_answer as ref
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"src.contriever_src.evaluation unavailable: {exc}")
        for sample in ["The World's Best Defender.", "a Gibson", "AN apple",
                       "yes", "No, not really!", "  spaced   out  "]:
            self.assertEqual(normalize_answer(sample, "squad"), ref(sample))

    def test_squad_drops_articles_but_token_does_not(self):
        self.assertEqual(normalize_answer("the answer", "squad"), "answer")
        self.assertEqual(normalize_answer("the answer", "token"), "the answer")


class TestExtraction(unittest.TestCase):
    def test_strips_answer_label(self):
        self.assertEqual(extract_short_answer("Answer: Paris"), "Paris")
        self.assertEqual(extract_short_answer("Final Answer - Paris"), "Paris")

    def test_keeps_leading_yes_no_token(self):
        got = extract_short_answer("No, Ferocactus is a plant. But Silene is not.")
        self.assertTrue(got.lower().startswith("no,"))
        self.assertIn("no", normalize_answer(got, "token").split())

    def test_truncates_at_max_answer_tokens(self):
        long_answer = " ".join(str(i) for i in range(50))
        got = extract_short_answer(long_answer, max_answer_tokens=5)
        self.assertEqual(got, "0 1 2 3 4")

    def test_empty_and_none_return_none(self):
        self.assertIsNone(extract_short_answer(None))
        self.assertIsNone(extract_short_answer(""))
        self.assertIsNone(extract_short_answer("   \n  "))

    def test_takes_first_nonempty_line(self):
        self.assertEqual(extract_short_answer("\n\n  Paris\nsomething else"), "Paris")

    def test_three_way_storage_is_reproducible_from_raw(self):
        """raw / extracted / normalized are distinct and re-derivable."""
        raw = "Answer: The World's Best Defender. Additional prose here."
        extracted = extract_short_answer(raw)
        normalized = normalize_answer(extracted, "squad")
        self.assertNotEqual(raw, extracted)
        self.assertNotEqual(extracted, normalized)
        # Re-deriving from raw alone reproduces both downstream fields.
        self.assertEqual(extract_short_answer(raw), extracted)
        self.assertEqual(normalize_answer(extract_short_answer(raw), "squad"),
                         normalized)


class TestAbstention(unittest.TestCase):
    def test_empty_and_none_are_abstentions(self):
        self.assertTrue(is_abstention(None))
        self.assertTrue(is_abstention(""))
        self.assertTrue(is_abstention("   "))

    def test_phrases_detected(self):
        self.assertTrue(is_abstention("I don't know"))
        self.assertTrue(is_abstention("I do not know the answer."))
        self.assertTrue(is_abstention("The context does not mention it"))

    def test_real_answers_are_not_abstentions(self):
        self.assertFalse(is_abstention("Yes."))
        self.assertFalse(is_abstention("World's Best Goalkeeper"))

    def test_covers_every_phrase_the_smoke_script_recognizes(self):
        """The canonical detector must never be weaker than the published one.

        Direction matters: the smoke script is a published artifact and is not
        being changed, so the guard is that everything *it* calls a non-answer
        is also an abstention here, not the reverse.
        """
        try:
            import run_answer_generation_smoke_bundle1 as smoke
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"smoke script unavailable: {exc}")
        for phrase in tuple(smoke._UNCERTAIN_PHRASES) + tuple(smoke._UNCERTAIN_EXACT):
            self.assertTrue(
                is_abstention(phrase),
                f"is_abstention() misses {phrase!r}, which the smoke script "
                "treats as a non-answer",
            )

    def test_none_differs_from_smoke_script_by_design(self):
        try:
            import run_answer_generation_smoke_bundle1 as smoke
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"smoke script unavailable: {exc}")
        self.assertTrue(is_abstention(None))
        self.assertFalse(smoke.is_no_answer_or_uncertain(None))

    def test_bare_unknown_does_not_fire_inside_a_factual_sentence(self):
        self.assertTrue(is_abstention("unknown"))
        self.assertFalse(is_abstention("The cause is unknown to historians"))


class TestGrouping(unittest.TestCase):
    def setUp(self):
        self.passages = [
            make_passage(f"d{i}", f"text {i}", is_poison=(i < 5), rank=i)
            for i in range(10)
        ]

    def test_group_sizes(self):
        for size, expected in ((1, 10), (2, 5), (3, 4)):
            groups = build_groups(self.passages, size)
            self.assertEqual(len(groups), expected)
            flat = [p for g in groups for p in g]
            self.assertEqual([p.doc_id for p in flat],
                             [p.doc_id for p in self.passages],
                             "grouping must not drop, duplicate, or reorder")

    def test_final_group_may_be_short(self):
        groups = build_groups(self.passages, 3)
        self.assertEqual(len(groups[-1]), 1)

    def test_group_size_at_or_above_n_yields_one_group(self):
        self.assertEqual(len(build_groups(self.passages, 10)), 1)
        self.assertEqual(len(build_groups(self.passages, 99)), 1)

    def test_non_positive_group_size_raises(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                build_groups(self.passages, bad)
            with self.assertRaises(ValueError):
                RobustRagKwConfig(group_size=bad)

    def test_max_isolated_calls_raises_and_does_not_truncate(self):
        cfg = RobustRagKwConfig(group_size=1, max_isolated_calls=4)
        with self.assertRaises(ValueError) as ctx:
            robustrag_kw_answer(
                "q", self.passages, generate_fn=raising_generate_fn, config=cfg
            )
        self.assertIn("max_isolated_calls", str(ctx.exception))

    def test_group_level_poison_flag_is_disjunctive(self):
        cfg = RobustRagKwConfig(group_size=2, max_isolated_calls=16)
        result = robustrag_kw_answer(
            "q", self.passages, generate_fn=lambda p: "yes", config=cfg
        )
        # Passages 0-4 poison, 5-9 clean -> group 2 spans (d4 poison, d5 clean).
        mixed = result.isolated_answers[2]
        self.assertTrue(mixed.is_poison)
        self.assertFalse(mixed.is_clean)
        self.assertEqual(mixed.doc_ids, ["d4", "d5"])
        self.assertEqual(mixed.ranks, [4, 5])


class TestVoteAggregation(unittest.TestCase):
    def test_clear_majority_and_unanimous(self):
        votes = aggregate_votes(["no", "no", "yes"], [False] * 3)
        self.assertEqual(list(votes.items()), [("no", 2), ("yes", 1)])
        votes = aggregate_votes(["no"] * 4, [False] * 4)
        self.assertEqual(list(votes.items()), [("no", 4)])

    def test_abstentions_and_empties_are_not_counted(self):
        votes = aggregate_votes(["no", None, "", "yes"], [False, True, False, False])
        self.assertEqual(list(votes.items()), [("no", 1), ("yes", 1)])

    def test_empty_vote_set_leads_to_abstention(self):
        votes = aggregate_votes([None, None], [True, True])
        self.assertEqual(len(votes), 0)
        final, abstained, winner, count, share, margin, denom = decide(
            votes, n_counted=0, n_groups=2, config=RobustRagKwConfig()
        )
        self.assertEqual(final, ABSTAIN_ANSWER)
        self.assertTrue(abstained)
        self.assertIsNone(winner)

    def test_exact_mode_does_not_merge_substring_answers(self):
        votes = aggregate_votes(
            ["best goalkeeper", "world's best goalkeeper"], [False, False],
            aggregation_mode="exact",
        )
        self.assertEqual(len(votes), 2)

    def test_keyword_mode_merges_token_subsets(self):
        votes = aggregate_votes(
            ["best goalkeeper", "world's best goalkeeper"], [False, False],
            aggregation_mode="keyword",
        )
        self.assertEqual(list(votes.items()), [("world's best goalkeeper", 2)])

    def test_keyword_mode_does_not_merge_opposite_answers(self):
        """The Schmeichel case: near-identical strings, opposite meanings."""
        votes = aggregate_votes(
            ["worlds best defender", "worlds best goalkeeper"], [False, False],
            aggregation_mode="keyword",
        )
        self.assertEqual(len(votes), 2, "defender and goalkeeper must not merge")

    def test_vote_margin_for_majority_near_tie_and_tie(self):
        cfg = RobustRagKwConfig(tie_breaker="first_rank")
        votes = aggregate_votes(["a"] * 3 + ["b"], [False] * 4)
        _f, _ab, _w, _c, _s, margin, _d = decide(
            votes, n_counted=4, n_groups=4, config=cfg)
        self.assertAlmostEqual(margin, 0.5)

        votes = aggregate_votes(["a", "a", "b"], [False] * 3)
        _f, _ab, _w, _c, _s, margin, _d = decide(
            votes, n_counted=3, n_groups=3, config=cfg)
        self.assertAlmostEqual(margin, 1 / 3)

        votes = aggregate_votes(["a", "b"], [False] * 2)
        _f, _ab, _w, _c, _s, margin, _d = decide(
            votes, n_counted=2, n_groups=2, config=cfg)
        self.assertAlmostEqual(margin, 0.0)

    def test_ordering_is_deterministic(self):
        answers = ["b", "a", "b", "c", "a", "b"]
        first = aggregate_votes(answers, [False] * len(answers))
        for _ in range(20):
            self.assertEqual(list(aggregate_votes(answers, [False] * len(answers)).items()),
                             list(first.items()))
        self.assertEqual(list(first.items()), [("b", 3), ("a", 2), ("c", 1)])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            aggregate_votes(["a"], [False, False])

    def test_invalid_aggregation_mode_raises(self):
        with self.assertRaises(ValueError):
            aggregate_votes(["a"], [False], aggregation_mode="semantic")


class TestAbstentionPolicies(unittest.TestCase):
    """Both policies from the plan, including the central §6.2 fixture."""

    def test_vote_threshold_boundary_is_inclusive(self):
        votes = aggregate_votes(["a", "a", "b", "b"], [False] * 4)
        cfg = RobustRagKwConfig(vote_threshold=0.5, tie_breaker="first_rank")
        final, abstained, *_ = decide(votes, n_counted=4, n_groups=4, config=cfg)
        self.assertFalse(abstained)
        self.assertEqual(final, "a")

    def test_five_wrong_five_abstaining_diverges_between_policies(self):
        """The plan's central claim, pinned.

        5 poisoned passages answer with the target wrong answer; 5 clean
        passages abstain. Discarding abstentions makes the attacker unanimous;
        including them puts the winner at 5/10, below a 0.6 abstain threshold.
        """
        normalized = ["no"] * 5 + [None] * 5
        abstentions = [False] * 5 + [True] * 5
        votes = aggregate_votes(normalized, abstentions)
        self.assertEqual(list(votes.items()), [("no", 5)])

        discard = RobustRagKwConfig(
            abstention_policy="discard_abstentions", abstain_threshold=0.6)
        final, abstained, _w, count, share, _m, denom = decide(
            votes, n_counted=5, n_groups=10, config=discard)
        self.assertEqual(denom, 5)
        self.assertAlmostEqual(share, 1.0)
        self.assertFalse(abstained)
        self.assertEqual(final, "no")

        include = RobustRagKwConfig(
            abstention_policy="include_abstentions", abstain_threshold=0.6)
        final, abstained, _w, count, share, _m, denom = decide(
            votes, n_counted=5, n_groups=10, config=include)
        self.assertEqual(denom, 10)
        self.assertAlmostEqual(share, 0.5)
        self.assertTrue(abstained)
        self.assertEqual(final, ABSTAIN_ANSWER)

    def test_abstain_threshold_gates_a_six_of_ten_winner(self):
        votes = aggregate_votes(["no"] * 6 + ["yes"] * 4, [False] * 10)
        strict = RobustRagKwConfig(
            abstention_policy="include_abstentions", abstain_threshold=0.7)
        _f, abstained, *_ = decide(votes, n_counted=10, n_groups=10, config=strict)
        self.assertTrue(abstained)

        lenient = RobustRagKwConfig(
            abstention_policy="include_abstentions", abstain_threshold=0.0)
        final, abstained, *_ = decide(votes, n_counted=10, n_groups=10, config=lenient)
        self.assertFalse(abstained)
        self.assertEqual(final, "no")

    def test_tie_breaker_abstain_vs_first_rank(self):
        votes = aggregate_votes(["a"] * 5 + ["b"] * 5, [False] * 10)
        abstain_cfg = RobustRagKwConfig(tie_breaker="abstain")
        final, abstained, *_ = decide(votes, n_counted=10, n_groups=10,
                                      config=abstain_cfg)
        self.assertTrue(abstained)
        self.assertEqual(final, ABSTAIN_ANSWER)

        rank_cfg = RobustRagKwConfig(tie_breaker="first_rank")
        final, abstained, *_ = decide(votes, n_counted=10, n_groups=10,
                                      config=rank_cfg)
        self.assertFalse(abstained)
        self.assertEqual(final, "a")

    def test_invalid_policy_values_raise(self):
        for kwargs in (
            {"abstention_policy": "drop"},
            {"aggregation_mode": "embedding"},
            {"tie_breaker": "random"},
            {"normalization_mode": "nfkc"},
        ):
            with self.assertRaises(ValueError):
                RobustRagKwConfig(**kwargs)


class TestSelfCrossQueryAccounting(unittest.TestCase):
    QID = "5a8e068b5542995085b37384"
    CROSS_DOC = "adv::LM_targeted::5a8e068b5542995085b37384::52"
    CROSS_OWNER = "5abd259d55429924427fcf1a"

    def _run(self, passages, origin):
        return robustrag_kw_answer(
            "q", passages,
            generate_fn=lambda p: "yes",
            config=RobustRagKwConfig(group_size=1),
            query_id=self.QID,
            context_type="mutated",
            origin_by_doc_id=origin,
        )

    def test_flags_are_mutually_exclusive_and_consistent(self):
        passages = [
            make_passage("self", "t", is_poison=True, rank=0),
            make_passage(self.CROSS_DOC, "t", is_poison=True, rank=1),
            make_passage("clean", "t", is_poison=False, rank=2),
        ]
        origin = {
            "self": {"origin_label": "mutated_self_query_poison",
                     "true_owning_query_id": self.QID, "true_global_index": 98},
            self.CROSS_DOC: {"origin_label": "cross_query_poison",
                             "true_owning_query_id": self.CROSS_OWNER,
                             "true_global_index": 52},
            "clean": {"origin_label": "clean"},
        }
        recs = self._run(passages, origin).isolated_answers

        self.assertTrue(recs[0].is_poison and recs[0].is_self_query_poison)
        self.assertFalse(recs[0].is_cross_query_poison)

        self.assertTrue(recs[1].is_poison and recs[1].is_cross_query_poison)
        self.assertFalse(recs[1].is_self_query_poison)
        self.assertEqual(recs[1].true_owning_query_id, self.CROSS_OWNER)

        self.assertTrue(recs[2].is_clean)
        self.assertFalse(recs[2].is_poison)
        self.assertFalse(recs[2].is_self_query_poison)
        self.assertFalse(recs[2].is_cross_query_poison)

        for rec in recs:
            rec.validate()
            self.assertEqual(rec.is_clean, not rec.is_poison)

    def test_validate_rejects_impossible_flag_combinations(self):
        rec = make_isolated("a")
        rec.is_poison = True  # is_clean also True -> contradiction
        with self.assertRaises(ValueError):
            rec.validate()

        rec = make_isolated("a", is_clean=False, is_poison=True)
        rec.is_self_query_poison = True
        rec.is_cross_query_poison = True
        with self.assertRaises(ValueError):
            rec.validate()

    def test_recomputed_labels_match_published_origin_breakdown(self):
        """Cross-check against the real full-retrieval audit CSV."""
        csv_path = os.path.join(
            REPO_ROOT, "manual_text_mutation_pilot", "hotpotqa_50q_k10",
            "mutation_bundle_1", "full_retrieval_pilot",
            "full_retrieval_poison_origin_breakdown.csv")
        if not os.path.exists(csv_path):
            self.skipTest("published origin-breakdown CSV not available")
        try:
            import csv as _csv
            import run_answer_generation_smoke_bundle1 as smoke
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"smoke script unavailable: {exc}")

        config_path = os.path.join(REPO_ROOT, smoke.DEFAULT_DATASET_CONFIG)
        if not os.path.exists(config_path):
            self.skipTest("ml_filterrag dataset_config.json not available")
        pool = smoke.load_full_pool_query_ids(config_path)

        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            rows = list(_csv.DictReader(fh))
        checked = 0
        for row in rows:
            if row["is_poison"].strip().lower() != "true":
                continue
            gidx = smoke.extract_global_index(row["doc_id"])
            owner, _slot = smoke.owning_query_and_slot(gidx, pool)
            expected_self = (owner == row["query_id"])
            actual_self = (row["origin_label"] == "mutated_self_query_poison")
            self.assertEqual(
                expected_self, actual_self,
                f"origin label mismatch for {row['doc_id']}")
            self.assertEqual(owner, row["true_owning_query_id"])
            checked += 1
        self.assertGreater(checked, 0)

    def test_ferocactus_rank9_is_cross_query_in_published_audit(self):
        csv_path = os.path.join(
            REPO_ROOT, "manual_text_mutation_pilot", "hotpotqa_50q_k10",
            "mutation_bundle_1", "full_retrieval_pilot",
            "full_retrieval_poison_origin_breakdown.csv")
        if not os.path.exists(csv_path):
            self.skipTest("published origin-breakdown CSV not available")
        import csv as _csv
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            rows = [r for r in _csv.DictReader(fh)
                    if r["query_id"] == self.QID and r["rank"] == "9"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["origin_label"], "cross_query_poison")
        self.assertEqual(rows[0]["true_owning_query_id"], self.CROSS_OWNER)


class TestGenerationCache(unittest.TestCase):
    def test_prompt_hash_is_stable_and_discriminating(self):
        self.assertEqual(prompt_hash("p", "m"), prompt_hash("p", "m"))
        self.assertNotEqual(prompt_hash("p", "m"), prompt_hash("p", "m2"))
        self.assertNotEqual(prompt_hash("p", "m"), prompt_hash("p2", "m"))
        # Pinned literal: stable across processes and interpreter restarts,
        # unlike Python's salted builtin hash().
        self.assertEqual(
            prompt_hash("hello", "gpt-3.5-turbo"),
            "fbe3e955368f8d40430de3a6e6efa19622787b4c63fef095b69cefd4680d0ece",
        )

    def test_nul_separator_prevents_concatenation_collisions(self):
        self.assertNotEqual(prompt_hash("b", "a"), prompt_hash("", "ab"))

    def test_roundtrip_preserves_every_metadata_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.jsonl")
            cache = GenerationCache(path)
            key = CacheKey(prompt_hash=prompt_hash("p", "m"), model_name="m")
            meta = {
                "query_id": "q1", "context_type": "mutated", "group_index": 3,
                "doc_ids": ["d1"], "ranks": [3],
                "origin_label": "cross_query_poison",
                "true_owning_query_id": "other", "true_global_index": 52,
                "mutation_family": "filterrag_targeted", "is_mutated": True,
            }
            cache.put(key, "an answer", meta)
            self.assertEqual(cache.flush(), 1)

            reloaded = GenerationCache(path).load()
            self.assertEqual(len(reloaded), 1)
            self.assertEqual(reloaded.get(key), "an answer")
            with open(path, "r", encoding="utf-8") as fh:
                rec = json.loads(fh.readline())
            for k, v in meta.items():
                self.assertEqual(rec[k], v)
            self.assertIn("created_at", rec)
            self.assertEqual(rec["prompt_hash"], key.prompt_hash)
            self.assertEqual(rec["model_name"], "m")

    def test_absent_cache_file_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = GenerationCache(os.path.join(tmp, "missing.jsonl")).load()
            self.assertEqual(len(cache), 0)

    def test_second_run_makes_zero_generator_calls(self):
        passages = [make_passage(f"d{i}", f"t{i}", rank=i) for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.jsonl")
            calls = []

            def counting_fn(prompt):
                calls.append(prompt)
                return "yes"

            cache = GenerationCache(path).load()
            first = robustrag_kw_answer(
                "q", passages, generate_fn=counting_fn, cache=cache,
                model_name="fake", query_id="q1", context_type="mutated")
            cache.flush()
            self.assertEqual(len(calls), 4)
            self.assertEqual(first.n_isolated_calls, 4)
            self.assertEqual(first.n_cache_hits, 0)

            cache2 = GenerationCache(path).load()
            second = robustrag_kw_answer(
                "q", passages, generate_fn=raising_generate_fn, cache=cache2,
                model_name="fake", query_id="q1", context_type="mutated")
            self.assertEqual(second.n_isolated_calls, 0)
            self.assertEqual(second.n_cache_hits, 4)
            self.assertEqual(first.final_answer, second.final_answer)
            self.assertEqual(list(first.vote_counts.items()),
                             list(second.vote_counts.items()))

    def test_cache_collision_on_different_query_raises(self):
        """A prompt embeds its question, so a query_id collision is a bug."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.jsonl")
            cache = GenerationCache(path)
            key = CacheKey(prompt_hash=prompt_hash("p", "m"), model_name="m")
            cache.put(key, "a", {"query_id": "q1", "context_type": "mutated"})
            with self.assertRaises(ValueError):
                cache.get(key, query_id="q2", context_type="mutated")

    def test_identical_prompt_across_context_types_is_reused(self):
        """Only self-query poison was mutated, so clean passages produce a
        byte-identical isolated prompt under both contexts. Reuse is correct."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.jsonl")
            cache = GenerationCache(path)
            key = CacheKey(prompt_hash=prompt_hash("p", "m"), model_name="m")
            cache.put(key, "a clean answer",
                      {"query_id": "q1", "context_type": "original"})
            self.assertEqual(
                cache.get(key, query_id="q1", context_type="mutated"),
                "a clean answer")
            cache.flush()
            with open(path, encoding="utf-8") as fh:
                rec = json.loads(fh.readline())
            self.assertIn("original", rec["context_types"])

    def test_sweep_over_cache_makes_no_calls(self):
        """All configurations re-aggregate from cached answers, zero API."""
        isolated = [make_isolated("no", is_clean=False, is_poison=True) for _ in range(5)]
        isolated += [make_isolated(None, abstain=True) for _ in range(5)]
        seen = []
        for policy in ("discard_abstentions", "include_abstentions"):
            for threshold in (0.0, 0.5, 0.6, 0.7):
                for norm in ("squad", "token"):
                    for agg in AGGREGATION_MODES:
                        cfg = RobustRagKwConfig(
                            abstention_policy=policy, abstain_threshold=threshold,
                            normalization_mode=norm, aggregation_mode=agg)
                        seen.append(aggregate_isolated(isolated, config=cfg))
        self.assertEqual(len(seen), 32)
        # The policy split must actually change outcomes somewhere in the grid.
        self.assertTrue(any(r.abstained for r in seen))
        self.assertTrue(any(not r.abstained for r in seen))

    def test_raising_generate_fn_raises(self):
        with self.assertRaises(AssertionError):
            raising_generate_fn("anything")


class TestStrictAsrIntegration(unittest.TestCase):
    def _passages(self):
        return [make_passage(f"d{i}", f"t{i}", is_poison=(i < 5), rank=i)
                for i in range(10)]

    def test_all_target_wrong_gives_strict_asr_true(self):
        result = robustrag_kw_answer(
            "Are Ferocactus and Silene both types of plant?", self._passages(),
            generate_fn=lambda p: "No.",
            config=RobustRagKwConfig(), target_wrong_answer="no",
            correct_answer="yes")
        self.assertFalse(result.abstained)
        self.assertTrue(strict_match("no", result.final_answer))
        self.assertTrue(all(r.matches_target_wrong_answer_strict
                            for r in result.isolated_answers))

    def test_gold_winner_gives_strict_asr_false_and_correct_match(self):
        result = robustrag_kw_answer(
            "Are Ferocactus and Silene both types of plant?", self._passages(),
            generate_fn=lambda p: "Yes.",
            config=RobustRagKwConfig(), target_wrong_answer="no",
            correct_answer="yes")
        self.assertFalse(result.abstained)
        self.assertFalse(strict_match("no", result.final_answer))
        self.assertTrue(strict_match("yes", result.final_answer))

    def test_abstained_result_matches_neither(self):
        result = robustrag_kw_answer(
            "q", self._passages(), generate_fn=lambda p: "I don't know",
            config=RobustRagKwConfig(), target_wrong_answer="no",
            correct_answer="yes")
        self.assertTrue(result.abstained)
        self.assertEqual(result.final_answer, ABSTAIN_ANSWER)
        self.assertFalse(strict_match("no", result.final_answer))
        self.assertFalse(strict_match("yes", result.final_answer))

    def test_deterministic_result_with_canned_answers(self):
        canned = ["no", "no", "no", "yes", "I don't know", "yes",
                  "no", "yes", "no", "I don't know"]
        passages = self._passages()

        def make_fn():
            it = iter(canned)
            return lambda p: next(it)

        first = robustrag_kw_answer(
            "q", passages, generate_fn=make_fn(), config=RobustRagKwConfig(),
            target_wrong_answer="no", correct_answer="yes")
        for _ in range(5):
            again = robustrag_kw_answer(
                "q", passages, generate_fn=make_fn(), config=RobustRagKwConfig(),
                target_wrong_answer="no", correct_answer="yes")
            self.assertEqual(first.final_answer, again.final_answer)
            self.assertEqual(list(first.vote_counts.items()),
                             list(again.vote_counts.items()))
            self.assertEqual(first.n_abstentions, again.n_abstentions)
        self.assertEqual(first.n_abstentions, 2)
        self.assertEqual(list(first.vote_counts.items()), [("no", 5), ("yes", 3)])


class TestSafetyGuarantees(unittest.TestCase):
    """Structural guarantees: no API calls, no dispatch changes, no reruns."""

    @staticmethod
    def _tree(path):
        with open(path, "r", encoding="utf-8") as fh:
            return ast.parse(fh.read())

    @staticmethod
    def _imported_modules(tree):
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module)
        return mods

    @staticmethod
    def _called_attrs(tree):
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute):
                    names.add(fn.attr)
                elif isinstance(fn, ast.Name):
                    names.add(fn.id)
        return names

    def test_defense_module_never_imports_models_or_calls_query(self):
        tree = self._tree(DEFENSE_MODULE)
        mods = self._imported_modules(tree)
        for banned in ("src.models", "openai", "google.generativeai"):
            self.assertNotIn(banned, mods)
        self.assertFalse(any(m.startswith("src.models") for m in mods))
        called = self._called_attrs(tree)
        self.assertNotIn("create_model", called)
        self.assertNotIn("query", called,
                         "defense/robustrag_kw.py must never call .query()")

    def test_defense_module_is_stdlib_only(self):
        tree = self._tree(DEFENSE_MODULE)
        mods = self._imported_modules(tree)
        heavy = {"torch", "transformers", "sentence_transformers", "numpy",
                 "sklearn", "beir", "regex"}
        self.assertEqual(mods & heavy, set(),
                         f"unexpected heavy dependency in {DEFENSE_MODULE}")

    def test_dispatch_is_unmodified_and_excludes_robustrag_kw(self):
        """Pins the standalone-defense decision."""
        try:
            from defense.dispatch import DEFENSE_CHOICES
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"defense.dispatch needs heavy deps: {exc}")
        self.assertNotIn("robustrag_kw", DEFENSE_CHOICES)
        self.assertEqual(DEFENSE_CHOICES, (
            "none", "ragdefender", "ragdefender_original", "ragdefender_paper",
            "oracle_remove_all_poison", "random_remove_same_count",
            "filterrag", "filterrag_query_only", "ml_filterrag",
        ))

    def test_pilot_script_does_not_rerun_retrieval_or_change_poison_budget(self):
        if not os.path.exists(PILOT_SCRIPT):
            self.skipTest("pilot script not present yet")
        with open(PILOT_SCRIPT, "r", encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source)
        mods = self._imported_modules(tree)
        for banned in ("beir", "src.attack"):
            self.assertFalse(any(m == banned or m.startswith(banned + ".")
                                 for m in mods),
                             f"pilot script must not import {banned}")
        called = self._called_attrs(tree)
        for banned in ("get_attack", "load_beir_datasets"):
            self.assertNotIn(banned, called,
                             f"pilot script must not call {banned}()")

        # The poison budget may be *read* (for correct/incorrect answers) but
        # never written: no open() in a write/append mode may target it.
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "open"):
                continue
            literals = [a.value for a in node.args
                        if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            path_arg = literals[0] if literals else ""
            modes = [v for v in literals[1:] if set(v) & {"w", "a", "x", "+"}]
            if modes:
                self.assertNotIn("adv_targeted_results", path_arg,
                                 "pilot script must not write to the poison budget")


if __name__ == "__main__":
    unittest.main()
