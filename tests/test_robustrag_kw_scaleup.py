#!/usr/bin/env python3
"""Tests for the RobustRAG-KW scale-up
(`scripts/robustrag_kw_scaleup_lib.py` + `scripts/run_robustrag_kw_scaleup_bundle1.py`).

Non-negotiable for this suite: **no GPT/API call is ever made**. That is not
left to convention -- `TestNoApiCallsInScaleup` walks both modules' ASTs to
prove the pure stages cannot reach a generator, and every generation exercised
here is a local stub or a cache hit.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import robustrag_kw_scaleup_lib as lib  # noqa: E402
import run_robustrag_kw_scaleup_bundle1 as scaleup  # noqa: E402

from defense.robustrag_kw import (  # noqa: E402
    CacheKey,
    GenerationCache,
    RobustRagKwConfig,
    aggregate_isolated,
    prompt_hash,
    raising_generate_fn,
    robustrag_kw_answer,
)
from defense.passages import RetrievedPassage  # noqa: E402
from defense.asr_match import strict_match  # noqa: E402

SCALEUP_SCRIPT = os.path.join(SCRIPTS_DIR, "run_robustrag_kw_scaleup_bundle1.py")
SCALEUP_LIB = os.path.join(SCRIPTS_DIR, "robustrag_kw_scaleup_lib.py")
OUT_DIR = os.path.join(REPO_ROOT, "results", "diagnostics", "robustrag_kw_scaleup")


def make_case(family, qid, *, self_retrieved=5, poison=5, clean=5,
              residual=(0, 0, 0), removed=(5, 5, 5), baseline=(5, 5, 5)):
    names = lib.FILTER_DEFENSES
    return lib.CaseStats(
        family=family, query_id=qid,
        n_mutated_self_retrieved=self_retrieved,
        n_retrieved_poison=poison, n_retrieved_clean=clean,
        residual_poison_by_defense=dict(zip(names, residual)),
        removed_poison_by_defense=dict(zip(names, removed)),
        baseline_removed_poison_by_defense=dict(zip(names, baseline)),
    )


class TestCandidateSelectionRule(unittest.TestCase):
    """The shortlist decides where API budget goes, so its rule is pinned."""

    def test_retrieval_gate_blocks_low_poison_survival(self):
        # Fails the gate at 3/5 even though a defense leaves 3 poison behind.
        case = make_case("filterrag_targeted", "q1", self_retrieved=3, residual=(3, 0, 0))
        row = lib.select_candidates([case])[0]
        self.assertFalse(row["selected"])
        self.assertFalse(row["retrieval_gate_pass"])
        self.assertIn("gate_failed", row["selection_reason"])

    def test_gate_boundary_is_inclusive_at_four(self):
        for n, expected in ((3, False), (4, True), (5, True)):
            case = make_case("filterrag_targeted", "q1", self_retrieved=n, residual=(2, 0, 0))
            row = lib.select_candidates([case])[0]
            self.assertEqual(row["retrieval_gate_pass"], expected, f"n={n}")

    def test_criterion_a_needs_two_residual_poison(self):
        one = lib.select_candidates([make_case("f", "q", residual=(1, 0, 0))])[0]
        two = lib.select_candidates([make_case("f", "q", residual=(2, 0, 0))])[0]
        self.assertFalse(one["criterion_a_residual_poison"])
        self.assertTrue(two["criterion_a_residual_poison"])
        self.assertEqual(two["criterion_a_defenses"], "ragdefender")

    def test_criterion_b_compares_against_that_querys_baseline(self):
        # Removed 3 where the unmutated baseline removed 5 -> a drop of 2.
        case = make_case("f", "q", removed=(3, 5, 5), baseline=(5, 5, 5), residual=(0, 0, 0))
        row = lib.select_candidates([case])[0]
        self.assertTrue(row["criterion_b_removed_poison_drop"])
        self.assertEqual(row["criterion_b_defenses"], "ragdefender")

    def test_criterion_b_ignores_a_one_passage_drop(self):
        case = make_case("f", "q", removed=(4, 5, 5), baseline=(5, 5, 5))
        self.assertFalse(lib.select_candidates([case])[0]["criterion_b_removed_poison_drop"])

    def test_criterion_c_fires_once_per_distinct_failure_signature(self):
        a = make_case("filterrag_targeted", lib.BUNDLE_QUERY_IDS[0], residual=(1, 0, 0))
        b = make_case("filterrag_targeted", lib.BUNDLE_QUERY_IDS[1], residual=(1, 0, 0))
        rows = {r["query_id"]: r for r in lib.select_candidates([a, b])}
        self.assertTrue(rows[lib.BUNDLE_QUERY_IDS[0]]["selected"])
        self.assertFalse(rows[lib.BUNDLE_QUERY_IDS[1]]["selected"])
        self.assertIn("signature_already_covered",
                      rows[lib.BUNDLE_QUERY_IDS[1]]["selection_reason"])

    def test_gate_failing_case_never_claims_a_signature(self):
        """A case that failed the retrieval gate must not consume the
        signature slot, or it would silently suppress a later eligible case
        exhibiting the same defense-failure mode."""
        blocked = make_case("filterrag_targeted", lib.BUNDLE_QUERY_IDS[0],
                            self_retrieved=1, residual=(1, 0, 0))
        eligible = make_case("filterrag_targeted", lib.BUNDLE_QUERY_IDS[1],
                             self_retrieved=5, residual=(1, 0, 0))
        rows = {r["query_id"]: r for r in lib.select_candidates([blocked, eligible])}
        self.assertFalse(rows[lib.BUNDLE_QUERY_IDS[0]]["selected"])
        self.assertTrue(rows[lib.BUNDLE_QUERY_IDS[1]]["selected"])

    def test_selection_is_order_independent(self):
        cases = [
            make_case("mlfilterrag_targeted", lib.BUNDLE_QUERY_IDS[2], residual=(2, 1, 0)),
            make_case("filterrag_targeted", lib.BUNDLE_QUERY_IDS[0], residual=(1, 0, 0)),
            make_case("ragdefender_targeted", lib.BUNDLE_QUERY_IDS[1], residual=(1, 0, 0)),
        ]
        forward = lib.select_candidates(cases)
        backward = lib.select_candidates(list(reversed(cases)))
        self.assertEqual(forward, backward)

    def test_selection_is_repeatable(self):
        cases = [make_case("filterrag_targeted", q, residual=(i % 3, 0, 0))
                 for i, q in enumerate(lib.BUNDLE_QUERY_IDS)]
        self.assertEqual(lib.select_candidates(cases), lib.select_candidates(cases))

    def test_output_order_follows_canonical_family_then_query(self):
        cases = [make_case(f, q) for f in reversed(lib.MUTATION_FAMILIES)
                 for q in reversed(lib.BUNDLE_QUERY_IDS)]
        rows = lib.select_candidates(cases)
        got = [(r["family"], r["query_id"]) for r in rows]
        want = [(f, q) for f in lib.MUTATION_FAMILIES for q in lib.BUNDLE_QUERY_IDS]
        self.assertEqual(got, want)

    def test_every_case_produces_exactly_one_decision_row(self):
        cases = [make_case(f, q) for f in lib.MUTATION_FAMILIES for q in lib.BUNDLE_QUERY_IDS]
        self.assertEqual(len(lib.select_candidates(cases)), 18)


class TestSelfVersusCrossQueryAccounting(unittest.TestCase):
    def test_origin_groups_are_distinct_buckets(self):
        self.assertEqual(lib.origin_group(lib.ORIGIN_CLEAN), "clean")
        self.assertEqual(lib.origin_group(lib.ORIGIN_MUTATED_SELF), "self_query_poison")
        self.assertEqual(lib.origin_group(lib.ORIGIN_CROSS_QUERY), "cross_query_poison")

    def test_original_self_poison_is_surfaced_not_folded_in(self):
        """Under a replacement-only budget this label must never appear; if it
        does it is a budget violation, so it must not be quietly merged into
        self_query_poison where it would go unnoticed."""
        self.assertEqual(lib.origin_group(lib.ORIGIN_ORIGINAL_SELF), lib.ORIGIN_ORIGINAL_SELF)

    def test_unknown_label_is_not_silently_clean(self):
        self.assertEqual(lib.origin_group("something_else"), "unknown")
        self.assertEqual(lib.origin_group(None), "unknown")

    def test_published_isolated_answers_have_consistent_origin_flags(self):
        path = os.path.join(OUT_DIR, "robustrag_kw_scaleup_isolated_answers.jsonl")
        if not os.path.exists(path):
            self.skipTest("scale-up artifacts not present")
        with open(path, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        self.assertTrue(rows)
        for r in rows:
            self.assertNotEqual(r["is_clean"], r["is_poison"], r["doc_id"])
            if r["is_clean"]:
                self.assertFalse(r["is_self_query_poison"])
                self.assertFalse(r["is_cross_query_poison"])
                self.assertEqual(r["origin_label"], lib.ORIGIN_CLEAN)
            else:
                self.assertNotEqual(r["is_self_query_poison"], r["is_cross_query_poison"],
                                    f"{r['doc_id']} is neither or both")

    def test_cross_query_poison_is_owned_by_a_different_query(self):
        path = os.path.join(OUT_DIR, "robustrag_kw_scaleup_isolated_answers.jsonl")
        if not os.path.exists(path):
            self.skipTest("scale-up artifacts not present")
        with open(path, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        cross = [r for r in rows if r["is_cross_query_poison"]]
        self.assertTrue(cross, "expected at least one cross-query passage in the bundle")
        for r in cross:
            self.assertNotEqual(r["true_owning_query_id"], r["query_id"])
        for r in [r for r in rows if r["is_self_query_poison"]]:
            self.assertEqual(r["true_owning_query_id"], r["query_id"])


class TestSweepGrid(unittest.TestCase):
    def test_grid_covers_both_abstention_denominators(self):
        policies = {g["abstention_policy"] for g in lib.sweep_grid()}
        self.assertEqual(policies, {"discard_abstentions", "include_abstentions"})

    def test_grid_includes_majority_vote_and_threshold_variants(self):
        thresholds = {g["vote_threshold"] for g in lib.sweep_grid()}
        self.assertIn(0.5, thresholds)          # plain majority
        self.assertTrue({0.6, 0.7} <= thresholds)

    def test_grid_includes_an_abstain_if_low_margin_variant(self):
        self.assertTrue(any(g["abstain_threshold"] > 0 for g in lib.sweep_grid()))
        self.assertTrue(any(g["abstain_threshold"] == 0 for g in lib.sweep_grid()))

    def test_grid_size_and_order_are_stable(self):
        grid = lib.sweep_grid()
        self.assertEqual(len(grid), 2 * 2 * 2 * 4 * 2)
        self.assertEqual(grid, lib.sweep_grid())

    def test_sweeps_never_generate(self):
        """Replaying the grid over already-cached answers must not call a
        generator: `aggregate_isolated` gets isolated answers, never a
        generate_fn, so there is no path to an API call."""
        passages = [
            RetrievedPassage(text="Paris is the capital.", doc_id="d1", is_poison=False,
                             source="corpus", rank=0),
            RetrievedPassage(text="London is the capital.", doc_id="d2", is_poison=True,
                             source="adversarial", rank=1),
        ]
        calls = []

        def stub(prompt):
            calls.append(prompt)
            return "Paris" if "Paris" in prompt else "London"

        result = robustrag_kw_answer(
            "What is the capital of France?", passages, generate_fn=stub,
            config=RobustRagKwConfig(group_size=1), model_name="stub", query_id="q")
        self.assertEqual(len(calls), 2)
        before = len(calls)
        for grid in lib.sweep_grid():
            aggregate_isolated(result.isolated_answers, config=RobustRagKwConfig(
                group_size=1, vote_threshold=grid["vote_threshold"],
                abstain_threshold=grid["abstain_threshold"],
                normalization_mode=grid["normalization_mode"],
                abstention_policy=grid["abstention_policy"],
                aggregation_mode=grid["aggregation_mode"]))
        self.assertEqual(len(calls), before, "the sweep generated new answers")


class TestCacheReuse(unittest.TestCase):
    def test_prompt_hash_is_stable_and_model_scoped(self):
        h1 = prompt_hash("some prompt", "gpt-3.5-turbo")
        self.assertEqual(h1, prompt_hash("some prompt", "gpt-3.5-turbo"))
        self.assertNotEqual(h1, prompt_hash("some prompt", "gpt-4"))
        self.assertNotEqual(h1, prompt_hash("other prompt", "gpt-3.5-turbo"))

    def test_second_pass_is_served_entirely_from_cache(self):
        passages = [
            RetrievedPassage(text="A clean passage about Paris.", doc_id="d1",
                             is_poison=False, source="corpus", rank=0),
            RetrievedPassage(text="A poisoned passage about London.", doc_id="d2",
                             is_poison=True, source="adversarial", rank=1),
        ]
        calls = []

        def stub(prompt):
            calls.append(prompt)
            return "Paris"

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.jsonl")
            cache = GenerationCache(path)
            first = robustrag_kw_answer("Q?", passages, generate_fn=stub, cache=cache,
                                        model_name="m", query_id="q")
            cache.flush()
            self.assertEqual(first.n_isolated_calls, 2)
            self.assertEqual(len(calls), 2)

            reloaded = GenerationCache(path).load()
            second = robustrag_kw_answer("Q?", passages, generate_fn=raising_generate_fn,
                                         cache=reloaded, model_name="m", query_id="q")
            self.assertEqual(second.n_isolated_calls, 0)
            self.assertEqual(second.n_cache_hits, 2)
            self.assertEqual(len(calls), 2, "a cached prompt was regenerated")
            self.assertEqual(first.final_answer, second.final_answer)

    def test_cache_miss_raises_instead_of_silently_generating(self):
        cache = GenerationCache().load()
        passages = [RetrievedPassage(text="t", doc_id="d", is_poison=False,
                                     source="corpus", rank=0)]
        with self.assertRaises(AssertionError) as ctx:
            robustrag_kw_answer("Q?", passages, generate_fn=raising_generate_fn,
                                cache=cache, model_name="m", query_id="q")
        self.assertIn("generate_fn was invoked", str(ctx.exception))

    def test_published_scaleup_cache_is_deduplicated_and_self_contained(self):
        cache_path = os.path.join(OUT_DIR, "robustrag_kw_scaleup_generation_cache.jsonl")
        iso_path = os.path.join(OUT_DIR, "robustrag_kw_scaleup_isolated_answers.jsonl")
        if not (os.path.exists(cache_path) and os.path.exists(iso_path)):
            self.skipTest("scale-up artifacts not present")
        with open(cache_path, "r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        hashes = [r["prompt_hash"] for r in records]
        self.assertEqual(len(hashes), len(set(hashes)),
                         "duplicate prompt_hash rows make the replay non-reproducible")

        cached = {(r["model_name"], r["prompt_hash"]) for r in records}
        with open(iso_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.assertIn((row["model_name"], row["prompt_sha256"]), cached,
                              "published answer is not backed by the published cache")

    def test_every_published_generation_carries_a_session_id(self):
        cache_path = os.path.join(OUT_DIR, "robustrag_kw_scaleup_generation_cache.jsonl")
        if not os.path.exists(cache_path):
            self.skipTest("scale-up artifacts not present")
        with open(cache_path, "r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        for r in records:
            self.assertTrue(r.get("generation_session_id"),
                            f"prompt {r['prompt_hash'][:12]} has no session provenance")


class TestBaselineGenerationCache(unittest.TestCase):
    """Baseline answers are content-addressed by (model_name, prompt) exactly
    like the isolated calls.

    Regression suite for a real incident: baselines used to be skipped by
    consulting the output JSONL, which a run only wrote at stage exit, so
    overlapping approval-gated runs each saw no prior output and each paid for
    the same prompts -- 69 useful generations billed as 122. Every test here
    counts generator invocations; none of them can reach an API.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = self.tmp.name
        self.calls = []
        self.cases = {
            ("filterrag_targeted", "q1"): self._case("filterrag_targeted", "q1"),
            ("ragdefender_targeted", "q2"): self._case("ragdefender_targeted", "q2"),
        }
        self.keys = list(self.cases.keys())

    def _case(self, family, qid, *, removals=None):
        """Each defense removes a different passage by default, so the four
        baseline conditions are four distinct contexts and therefore four
        distinct prompts. Pass `removals` to make two conditions coincide."""
        removals = removals or {"ragdefender": {1}, "filterrag_semantic": {2},
                                "ml_filterrag": {3}}
        passages = [{"doc_id": f"{qid}-d0", "context": f"Clean text for {qid}.",
                     "is_poison": False, "source": "corpus", "rank": 1,
                     "origin_label": lib.ORIGIN_CLEAN}]
        for i in (1, 2, 3):
            passages.append({
                "doc_id": f"{qid}-p{i}", "context": f"Poison {i} for {qid}.",
                "is_poison": True, "source": "adversarial", "rank": i + 1,
                "origin_label": lib.ORIGIN_MUTATED_SELF})
        for p_i, p in enumerate(passages):
            for defense, removed in removals.items():
                p[scaleup.REMOVAL_FLAG[defense]] = p_i in removed
        return {
            "family": family, "query_id": qid, "question": f"Question for {qid}?",
            "correct_answer": "Paris", "target_wrong_answer": "London",
            "passages": passages, "n_retrieved_poison": 3, "n_retrieved_clean": 1,
            "defense_counts": {d: {"removed_poison": 1, "remaining_poison": 2}
                               for d in ("ragdefender", "filterrag_semantic",
                                         "ml_filterrag_t04")},
        }

    def _stub(self, prompt):
        self.calls.append(prompt)
        return f"answer::{len(self.calls)}"

    def _run(self, cache, *, generate_fn=None, session_id="s1", model="gpt-3.5-turbo"):
        return scaleup.generate_baselines(
            self.cases, self.keys, cache=cache,
            generate_fn=generate_fn or self._stub, generator_model=model,
            session_id=session_id, out_dir=self.out_dir, smoke_dir=None, verbose=False)

    def _cache(self):
        return GenerationCache(os.path.join(self.out_dir, "cache.jsonl")).load()

    def test_repeat_run_regenerates_nothing(self):
        rows, n = self._run(self._cache())
        expected = len(self.keys) * len(scaleup.BASELINE_CONDITIONS)
        self.assertEqual(n, expected)
        self.assertEqual(len(self.calls), expected)

        rows2, n2 = self._run(self._cache(), generate_fn=raising_generate_fn)
        self.assertEqual(n2, 0, "a repeated run paid for prompts it already had")
        self.assertEqual(len(self.calls), expected)
        self.assertEqual([r["raw_output"] for r in rows],
                         [r["raw_output"] for r in rows2],
                         "replayed answers differ from the ones first generated")
        self.assertTrue(all(r["cache_hit"] for r in rows2))

    def test_overlapping_run_sees_partial_work_flushed_by_the_other(self):
        """The incident, reproduced: a second run starts while the first is
        mid-stage. It must pick up what the first has already flushed, which
        only works because flushes are per-generation rather than at exit."""
        first = self._cache()
        one_key = self.keys[:1]
        scaleup.generate_baselines(
            self.cases, one_key, cache=first, generate_fn=self._stub,
            generator_model="gpt-3.5-turbo", session_id="run_a",
            out_dir=self.out_dir, smoke_dir=None, verbose=False)
        done = len(self.calls)
        self.assertEqual(done, len(scaleup.BASELINE_CONDITIONS))

        _, n = self._run(self._cache(), session_id="run_b")
        self.assertEqual(n, len(scaleup.BASELINE_CONDITIONS),
                         "run B repaid for prompts run A had already flushed")
        self.assertEqual(len(self.calls), done + n)

    def test_run_started_before_the_other_flushed_still_reuses_its_work(self):
        """Worst case: run B loaded the cache while it was still empty. The
        pre-generation disk re-check is the only thing standing between it and
        a duplicate bill."""
        stale = self._cache()
        self.assertEqual(len(stale), 0)

        scaleup.generate_baselines(
            self.cases, self.keys, cache=self._cache(), generate_fn=self._stub,
            generator_model="gpt-3.5-turbo", session_id="run_a",
            out_dir=self.out_dir, smoke_dir=None, verbose=False)
        paid = len(self.calls)

        _, n = self._run(stale, generate_fn=raising_generate_fn, session_id="run_b")
        self.assertEqual(n, 0, "a stale in-memory cache caused a duplicate bill")
        self.assertEqual(len(self.calls), paid)

    def test_backfill_adopts_published_answers_that_predate_the_cache(self):
        """Answers written before baselines were cached must be re-keyed by
        prompt hash, not regenerated."""
        rows, _ = self._run(self._cache())
        os.remove(os.path.join(self.out_dir, "cache.jsonl"))
        self.assertEqual(len(self._cache()), 0)

        rows2, n = self._run(self._cache(), generate_fn=raising_generate_fn)
        self.assertEqual(n, 0, "published baselines were regenerated, not adopted")
        self.assertEqual([r["raw_output"] for r in rows], [r["raw_output"] for r in rows2])

    def test_cache_key_distinguishes_prompts_and_models(self):
        cache = self._cache()
        self._run(cache)
        paid = len(self.calls)

        _, n = self._run(self._cache(), model="gpt-4")
        self.assertEqual(n, paid, "a different model reused another model's answers")

        hashes = {scaleup.prompt_hash(
            scaleup.baseline_prompt(self.cases[k], d), "gpt-3.5-turbo")
            for k in self.keys for d, _ in scaleup.BASELINE_CONDITIONS}
        self.assertEqual(len(hashes), paid, "distinct baseline contexts collided")

    def test_conditions_with_an_identical_context_share_one_generation(self):
        """A filter that removed nothing is asking the generator the same
        question as no filter at all. Paying twice for that buys only a
        nondeterministic paraphrase, so the two conditions share a generation
        and are reported with the same answer."""
        self.cases = {("filterrag_targeted", "q1"): self._case(
            "filterrag_targeted", "q1",
            removals={"ragdefender": {1}, "filterrag_semantic": set(),
                      "ml_filterrag": {2}})}
        self.keys = list(self.cases.keys())

        rows, n = self._run(self._cache())
        self.assertEqual(n, len(scaleup.BASELINE_CONDITIONS) - 1,
                         "an empty filter was billed as a separate condition")
        by_defense = {r["defense_name"]: r for r in rows}
        self.assertEqual(by_defense["none"]["raw_output"],
                         by_defense["filterrag_semantic"]["raw_output"])
        self.assertTrue(by_defense["filterrag_semantic"]["cache_hit"])
        counts = self.cases[self.keys[0]]["defense_counts"]
        for defense, row in by_defense.items():
            expected = counts.get(
                "ml_filterrag_t04" if defense == "ml_filterrag" else defense, {})
            self.assertEqual(row["removed_poison"], expected.get("removed_poison", 0),
                             "sharing a generation merged the defenses' own counts")

    def test_metadata_differences_do_not_force_regeneration(self):
        """`defense_name`, session and threshold are descriptive, not part of
        the key: two labels for one prompt must share one generation."""
        cache = self._cache()
        case = self.cases[self.keys[0]]
        prompt = scaleup.baseline_prompt(case, "none")
        key = CacheKey(prompt_hash=scaleup.prompt_hash(prompt, "gpt-3.5-turbo"),
                       model_name="gpt-3.5-turbo")
        cache.put(key, "seeded answer", scaleup._baseline_cache_meta(
            case, "some_other_label", 0.99, "an_older_session", "elsewhere"))
        cache.flush()

        rows, _ = self._run(self._cache(), session_id="new_session")
        row = next(r for r in rows
                   if r["query_id"] == case["query_id"] and r["defense_name"] == "none")
        self.assertEqual(row["raw_output"], "seeded answer")
        self.assertNotIn(prompt, self.calls)
        self.assertEqual(row["generation_session_id"], "an_older_session",
                         "replaying a generation rewrote the session that made it")

    def test_row_shape_stays_compatible_with_the_report_reader(self):
        rows, _ = self._run(self._cache())
        required = {"family", "query_id", "bundle_id", "context_type", "defense_name",
                    "threshold", "raw_output", "retrieved_poison_count",
                    "removed_poison", "remaining_poison", "generation_session_id",
                    "model_name", "source"}
        for r in rows:
            self.assertTrue(required.issubset(r.keys()))
            self.assertEqual(r["prompt_sha256"],
                             scaleup.prompt_hash(
                                 scaleup.baseline_prompt(
                                     self.cases[(r["family"], r["query_id"])],
                                     r["defense_name"]), r["model_name"]))

    def _replay_published(self):
        src = os.path.join(OUT_DIR, "robustrag_kw_scaleup_baseline_answers.jsonl")
        retrieval = os.path.join(OUT_DIR, "robustrag_kw_scaleup_retrieval.jsonl")
        if not (os.path.exists(src) and os.path.exists(retrieval)):
            self.skipTest("scale-up artifacts not present")
        published = scaleup.load_jsonl(src)
        cases = {scaleup.case_key(c): c for c in scaleup.load_jsonl(retrieval)}
        keys = sorted({(r["family"], r["query_id"]) for r in published})
        scaleup.write_jsonl(
            os.path.join(self.out_dir, scaleup.BASELINE_ANSWERS_FILE), published)
        rows, n = scaleup.generate_baselines(
            cases, keys, cache=self._cache(), generate_fn=raising_generate_fn,
            generator_model="gpt-3.5-turbo", session_id="replay",
            out_dir=self.out_dir, smoke_dir=None, verbose=False)
        return cases, published, rows, n

    def test_published_baselines_all_replay_from_cache_without_generating(self):
        """The published run, re-derived: with the fix in place, not one of the
        36 baseline answers costs a call the second time around."""
        _, published, rows, n = self._replay_published()
        self.assertEqual(n, 0, "re-deriving the published baselines would cost API calls")
        self.assertEqual(len(rows), len(published))

    def test_replaying_published_baselines_changes_no_verdict(self):
        """Content addressing merges baseline conditions whose kept context is
        identical -- a filter that removed nothing answers the same prompt as
        no filter at all, and one query shared across mutation families answers
        the same prompt in each. Merging them may swap a nondeterministic
        paraphrase for its twin, so what has to hold is that no strict-ASR or
        gold-match verdict moves, and that no answer changes except where two
        published rows really were the same prompt."""
        cases, published, rows, _ = self._replay_published()
        by_key = {(r["family"], r["query_id"], r["defense_name"]): r for r in published}
        shared = self._prompts_shared_by_multiple_rows(cases, published)
        for row in rows:
            k = (row["family"], row["query_id"], row["defense_name"])
            old, new = by_key[k]["raw_output"], row["raw_output"]
            case = cases[(row["family"], row["query_id"])]
            for label, ref in (("strict ASR", case["target_wrong_answer"]),
                               ("gold match", case["correct_answer"])):
                self.assertEqual(strict_match(old, ref), strict_match(new, ref),
                                 f"{label} verdict moved for {k}")
            if old != new:
                self.assertIn(row["prompt_sha256"], shared,
                              f"{k} changed answer without sharing a prompt")

    @staticmethod
    def _prompts_shared_by_multiple_rows(cases, published):
        seen, shared = {}, set()
        for r in published:
            h = scaleup.prompt_hash(
                scaleup.baseline_prompt(cases[(r["family"], r["query_id"])],
                                        r["defense_name"]), "gpt-3.5-turbo")
            if h in seen:
                shared.add(h)
            seen[h] = r
        return shared


class TestOutputSchemas(unittest.TestCase):
    """Pinned column orders: a reshaped published CSV should fail here rather
    than silently break a downstream reader."""

    REQUIRED_ISOLATED_FIELDS = (
        "query_id", "mutation_family", "bundle_id", "context_type", "doc_id",
        "retrieved_rank", "is_clean", "is_poison", "is_self_query_poison",
        "is_cross_query_poison", "raw_answer", "extracted_answer",
        "normalized_answer", "is_abstain", "matches_target_wrong_answer_strict",
        "matches_correct_answer_strict", "prompt_sha256", "model_name",
        "generation_session_id",
    )

    def test_isolated_schema_contains_every_required_diagnostic_field(self):
        for field in self.REQUIRED_ISOLATED_FIELDS:
            self.assertIn(field, lib.ISOLATED_ANSWER_FIELDS)

    def test_schemas_have_no_duplicate_columns(self):
        for name in ("CANDIDATE_SELECTION_FIELDS", "ISOLATED_ANSWER_FIELDS",
                     "VOTE_SUMMARY_FIELDS", "GENERATION_RESULTS_FIELDS",
                     "VS_EXISTING_DEFENSES_FIELDS", "ABSTENTION_SWEEP_FIELDS",
                     "ORIGIN_BREAKDOWN_FIELDS"):
            fields = getattr(lib, name)
            self.assertEqual(len(fields), len(set(fields)), name)

    def test_raw_extracted_and_normalized_answers_are_separate_columns(self):
        for field in ("raw_answer", "extracted_answer", "normalized_answer"):
            self.assertIn(field, lib.ISOLATED_ANSWER_FIELDS)

    def test_published_files_match_the_pinned_schemas(self):
        import csv
        expected = {
            "robustrag_kw_scaleup_candidate_selection.csv": lib.CANDIDATE_SELECTION_FIELDS,
            "robustrag_kw_scaleup_vote_summary.csv": lib.VOTE_SUMMARY_FIELDS,
            "robustrag_kw_scaleup_generation_results.csv": lib.GENERATION_RESULTS_FIELDS,
            "robustrag_kw_scaleup_vs_existing_defenses.csv": lib.VS_EXISTING_DEFENSES_FIELDS,
            "robustrag_kw_scaleup_abstention_sweep.csv": lib.ABSTENTION_SWEEP_FIELDS,
            "robustrag_kw_scaleup_origin_breakdown.csv": lib.ORIGIN_BREAKDOWN_FIELDS,
        }
        for name, fields in expected.items():
            path = os.path.join(OUT_DIR, name)
            if not os.path.exists(path):
                self.skipTest("scale-up artifacts not present")
            with open(path, "r", encoding="utf-8", newline="") as fh:
                header = next(csv.reader(fh))
            self.assertEqual(header, list(fields), name)

    def test_published_isolated_rows_expose_every_required_field(self):
        path = os.path.join(OUT_DIR, "robustrag_kw_scaleup_isolated_answers.jsonl")
        if not os.path.exists(path):
            self.skipTest("scale-up artifacts not present")
        with open(path, "r", encoding="utf-8") as fh:
            row = json.loads(next(line for line in fh if line.strip()))
        for field in self.REQUIRED_ISOLATED_FIELDS:
            self.assertIn(field, row)


class TestNoApiCallsInScaleup(unittest.TestCase):
    """Structural guarantees, checked by AST rather than by reading the code."""

    @staticmethod
    def _tree(path):
        with open(path, "r", encoding="utf-8") as fh:
            return ast.parse(fh.read())

    def test_pure_lib_imports_nothing_that_can_generate(self):
        tree = self._tree(SCALEUP_LIB)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for banned in ("src", "openai", "torch", "transformers"):
            self.assertNotIn(banned, imported,
                             f"the pure selection lib must not import {banned}")

    def test_generator_is_only_constructed_inside_the_generate_stage(self):
        """`src.models.create_model` and `llm.query()` may appear only within
        `stage_generate` -- including its nested `generate_fn` closure, which
        is the single place a prompt reaches the API."""
        tree = self._tree(SCALEUP_SCRIPT)
        stage_generate = next(n for n in ast.walk(tree)
                              if isinstance(n, ast.FunctionDef) and n.name == "stage_generate")
        allowed = {n for n in ast.walk(stage_generate)}

        offenders = []
        for node in ast.walk(tree):
            is_create = (isinstance(node, ast.ImportFrom) and node.module == "src.models")
            is_query = (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "query")
            if (is_create or is_query) and node not in allowed:
                offenders.append(ast.dump(node)[:80])
        self.assertEqual(offenders, [],
                         "only stage_generate may construct or call the generator")

    def test_report_stage_installs_a_raising_generate_fn(self):
        tree = self._tree(SCALEUP_SCRIPT)
        report = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == "stage_report")
        names = {n.id for n in ast.walk(report) if isinstance(n, ast.Name)}
        self.assertIn("raising_generate_fn", names)

    def test_retrieval_and_select_stages_cannot_generate(self):
        tree = self._tree(SCALEUP_SCRIPT)
        for stage in ("stage_retrieval", "stage_select"):
            func = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and n.name == stage)
            names = {n.id for n in ast.walk(func) if isinstance(n, ast.Name)}
            self.assertNotIn("robustrag_kw_answer", names, stage)
            for node in ast.walk(func):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotEqual(node.func.attr, "query", stage)

    def test_scaleup_does_not_register_a_new_defense_or_touch_dispatch(self):
        from defense.dispatch import DEFENSE_CHOICES

        self.assertNotIn("robustrag_kw", DEFENSE_CHOICES)

        # The name may be discussed in prose, but must never be imported,
        # assigned or mutated by the scale-up.
        tree = self._tree(SCALEUP_SCRIPT)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn("DEFENSE_CHOICES", [a.name for a in node.names])
            if isinstance(node, ast.Name) and node.id == "DEFENSE_CHOICES":
                self.fail("scale-up references DEFENSE_CHOICES in executable code")

        with open(SCALEUP_SCRIPT, "r", encoding="utf-8") as fh:
            source = fh.read()
        for banned in ("import main", "from main import"):
            self.assertNotIn(banned, source)


class TestBudgetAndProvenance(unittest.TestCase):
    def test_published_cases_preserve_the_replacement_only_budget(self):
        path = os.path.join(OUT_DIR, "robustrag_kw_scaleup_retrieval.jsonl")
        if not os.path.exists(path):
            self.skipTest("scale-up artifacts not present")
        with open(path, "r", encoding="utf-8") as fh:
            cases = [json.loads(line) for line in fh if line.strip()]
        self.assertEqual(len(cases), len(lib.MUTATION_FAMILIES) * len(lib.BUNDLE_QUERY_IDS))
        for case in cases:
            self.assertEqual(len(case["mutated_self_global_indices"]), 5, case["query_id"])
            self.assertLessEqual(case["n_mutated_self_retrieved"], 5)
            labels = [p["origin_label"] for p in case["passages"]]
            self.assertNotIn(lib.ORIGIN_ORIGINAL_SELF, labels,
                             "original poison survived alongside its mutation: budget inflated")
            self.assertEqual(len(case["passages"]), 10)

    def test_every_family_and_query_is_covered_exactly_once(self):
        path = os.path.join(OUT_DIR, "robustrag_kw_scaleup_retrieval.jsonl")
        if not os.path.exists(path):
            self.skipTest("scale-up artifacts not present")
        with open(path, "r", encoding="utf-8") as fh:
            keys = [(json.loads(line)["family"], json.loads(line)["query_id"])
                    for line in fh if line.strip()]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys),
                         {(f, q) for f in lib.MUTATION_FAMILIES for q in lib.BUNDLE_QUERY_IDS})


if __name__ == "__main__":
    unittest.main()
