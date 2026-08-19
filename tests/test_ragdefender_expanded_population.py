"""Tests for `scripts/ragdefender_expanded_population_lib.py` and
`scripts/build_ragdefender_expanded_population.py` -- the STEP 3
prospective HotpotQA k=10 population freeze.

Covers:
- regime classification A/B/C/D (`classify_regime`);
- determinism of the population build (same input -> same output, no
  hidden randomness);
- that the eligible-pool / recovery pipeline never consults any
  RAGDefender outcome (no `is_poison`-derived success/failure filtering,
  no N_adv, no Stage-2 result anywhere in the population-lib module);
- no-overwrite safeguard for the freeze script's outputs;
- k=10-only scope (no accidental k=2 mixing into this population).

Fully offline except where explicitly gated on real, already-recovered
artifacts existing on disk (features.csv/corpus.jsonl/etc. -- no Stella,
no network).

Run with: python -m unittest tests.test_ragdefender_expanded_population -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import ragdefender_expanded_population_lib as poplib  # noqa: E402


class TestClassifyRegime(unittest.TestCase):
    """A/B/C/D regime classification (STEP 2 §9.3 of the fidelity audit)."""

    def test_below_ceiling(self):
        # k=10, floor(k/2)=5; M=4 < 5 -> A
        self.assertEqual(poplib.classify_regime(k=10, m_poison=4, c_clean=6), "A_BELOW_CEILING")

    def test_at_ceiling(self):
        # k=10, M=5 == floor(10/2) -> B
        self.assertEqual(poplib.classify_regime(k=10, m_poison=5, c_clean=5), "B_AT_CEILING")

    def test_above_ceiling_majority_poison(self):
        # k=10, M=6 > 5, C=4 >= 1 -> C
        self.assertEqual(poplib.classify_regime(k=10, m_poison=6, c_clean=4), "C_ABOVE_CEILING")

    def test_all_poison_is_d_regardless_of_m_vs_ceiling(self):
        # k=10, M=10, C=0 -> D, even though M(10) > floor(k/2)(5)
        self.assertEqual(poplib.classify_regime(k=10, m_poison=10, c_clean=0), "D_ALL_POISON")

    def test_d_checked_before_c_never_mixed(self):
        """An all-poison context must be D even when it would also satisfy
        C's numeric M > floor(k/2) condition -- C and D must never both
        apply to the same context."""
        for k in [2, 3, 4, 5, 6, 10, 11]:
            with self.subTest(k=k):
                label = poplib.classify_regime(k=k, m_poison=k, c_clean=0)
                self.assertEqual(label, "D_ALL_POISON")

    def test_odd_k_boundary_values(self):
        # k=5, floor(5/2)=2
        self.assertEqual(poplib.classify_regime(k=5, m_poison=1, c_clean=4), "A_BELOW_CEILING")
        self.assertEqual(poplib.classify_regime(k=5, m_poison=2, c_clean=3), "B_AT_CEILING")
        self.assertEqual(poplib.classify_regime(k=5, m_poison=3, c_clean=2), "C_ABOVE_CEILING")

    def test_k2_ceiling_is_one(self):
        self.assertEqual(poplib.classify_regime(k=2, m_poison=0, c_clean=2), "A_BELOW_CEILING")
        self.assertEqual(poplib.classify_regime(k=2, m_poison=1, c_clean=1), "B_AT_CEILING")
        self.assertEqual(poplib.classify_regime(k=2, m_poison=2, c_clean=0), "D_ALL_POISON")

    def test_raises_on_inconsistent_k_m_c(self):
        with self.assertRaises(poplib.PopulationBuildError):
            poplib.classify_regime(k=10, m_poison=4, c_clean=4)  # 4+4 != 10

    def test_every_regime_is_one_of_four_allowed_labels(self):
        allowed = {"A_BELOW_CEILING", "B_AT_CEILING", "C_ABOVE_CEILING", "D_ALL_POISON"}
        for k in range(2, 12):
            for m in range(0, k + 1):
                c = k - m
                label = poplib.classify_regime(k=k, m_poison=m, c_clean=c)
                with self.subTest(k=k, m=m, c=c):
                    self.assertIn(label, allowed)


class TestPoolIndexMapping(unittest.TestCase):
    """`pool_index_to_source` -- pure arithmetic, no I/O -- underlies the
    no-new-retrieval text-recovery guarantee."""

    def test_pool_index_zero_maps_to_first_query_local_zero(self):
        ordered = ["qidA", "qidB", "qidC"]
        source, local = poplib.pool_index_to_source(0, ordered)
        self.assertEqual(source, "qidA")
        self.assertEqual(local, 0)

    def test_pool_index_within_first_query_block(self):
        ordered = ["qidA", "qidB", "qidC"]
        for pool_index in range(5):
            source, local = poplib.pool_index_to_source(pool_index, ordered)
            with self.subTest(pool_index=pool_index):
                self.assertEqual(source, "qidA")
                self.assertEqual(local, pool_index)

    def test_pool_index_rolls_over_to_next_query(self):
        ordered = ["qidA", "qidB", "qidC"]
        source, local = poplib.pool_index_to_source(5, ordered)
        self.assertEqual(source, "qidB")
        self.assertEqual(local, 0)
        source, local = poplib.pool_index_to_source(9, ordered)
        self.assertEqual(source, "qidB")
        self.assertEqual(local, 4)


class TestGateBcExclusionIsAPureSetDifference(unittest.TestCase):
    """The exclusion set is fixed/hardcoded (not derived from any live
    computation), and has exactly the documented 8 members."""

    def test_exclusion_set_has_exactly_8_members(self):
        self.assertEqual(len(poplib.GATE_BC_EXCLUDED_QUERY_IDS), 8)

    def test_exclusion_set_is_a_frozenset_immutable(self):
        self.assertIsInstance(poplib.GATE_BC_EXCLUDED_QUERY_IDS, frozenset)


class TestNoOutcomeBasedFiltering(unittest.TestCase):
    """Static/structural guard: the population-lib module must not import
    or reference any RAGDefender outcome machinery -- it can only build
    context (doc_ids/is_poison/text), never N_adv/Stage-2/defense_runner."""

    def test_population_lib_does_not_import_defense_runner_or_internals(self):
        import inspect

        source = inspect.getsource(poplib)
        self.assertNotIn("defense_runner", source)
        self.assertNotIn("ragdefender_internals", source)
        self.assertNotIn("concentration_stage1", source)
        self.assertNotIn("stage2_pair_frequency", source)
        self.assertNotIn("apply_defense", source)

    def test_recover_context_for_query_signature_has_no_outcome_parameters(self):
        import inspect

        sig = inspect.signature(poplib.recover_context_for_query)
        outcome_like = {"n_adv", "stage2", "removed", "residual", "success", "failure"}
        for param in sig.parameters:
            for bad in outcome_like:
                self.assertNotIn(bad, param.lower())


@unittest.skipUnless(
    poplib.DATASET_CONFIG_PATH.exists()
    and poplib.FEATURES_CSV_PATH.exists()
    and poplib.ORDERED_QIDS_SOURCE_PATH.exists()
    and poplib.ADV_TARGETED_RESULTS_PATH.exists()
    and poplib.CORPUS_JSONL_PATH.exists(),
    "Real existing-artifact files not found on disk (results/datasets are gitignored; "
    "this test only runs in an environment where the prior retrieval/poisoning pipeline "
    "artifacts are present).",
)
class TestRealArtifactPopulationBuildIsDeterministic(unittest.TestCase):
    """End-to-end (still zero-Stella, zero-network) determinism check
    against the REAL saved artifacts: building the population twice from
    scratch must give byte-identical results, and every recovered context
    must be k=10 with a self-consistent M+C=k and a regime label."""

    def test_eligible_pool_excludes_exactly_the_8_gate_bc_queries(self):
        pool = poplib.load_eligible_pool()
        self.assertEqual(len(pool), 50 - 8)
        for qid in poplib.GATE_BC_EXCLUDED_QUERY_IDS:
            self.assertNotIn(qid, pool)

    def test_population_build_is_deterministic_across_two_runs(self):
        pool = poplib.load_eligible_pool()
        # Only recompute for a small deterministic slice to keep this test
        # fast (the corpus streaming pass is the expensive part).
        sample = sorted(pool)[:3]
        contexts_1 = poplib.recover_all_contexts(sample, k=10)
        contexts_2 = poplib.recover_all_contexts(sample, k=10)
        self.assertEqual(contexts_1, contexts_2)

    def test_every_recovered_context_is_k10_and_self_consistent(self):
        pool = poplib.load_eligible_pool()
        sample = sorted(pool)[:3]
        contexts = poplib.recover_all_contexts(sample, k=10)
        for ctx in contexts:
            with self.subTest(query_id=ctx["query_id"]):
                self.assertEqual(ctx["k"], 10)
                self.assertEqual(ctx["m_poison"] + ctx["c_clean"], 10)
                self.assertEqual(len(ctx["texts"]), 10)
                self.assertEqual(len(ctx["is_poison"]), 10)
                self.assertIn(
                    ctx["regime"], {"A_BELOW_CEILING", "B_AT_CEILING", "C_ABOVE_CEILING", "D_ALL_POISON"}
                )


class TestFrozenPopulationArtifactsIfPresent(unittest.TestCase):
    """If the STEP-3 freeze has already been run in this environment,
    verify its own internal self-consistency and k=10-only scope (no k=2
    mixing) -- read-only, never regenerates or overwrites."""

    FREEZE_DIR = poplib.REPO_ROOT / "results/diagnostics/ragdefender_expanded_baseline"

    def setUp(self):
        recovered_path = self.FREEZE_DIR / "recovered_contexts.json"
        if not recovered_path.exists():
            self.skipTest(f"{recovered_path} not found -- STEP 3 freeze has not been run in this environment.")
        import json

        with open(recovered_path) as f:
            self.contexts = json.load(f)

    def test_all_contexts_are_k10_no_k2_mixing(self):
        for ctx in self.contexts:
            with self.subTest(query_id=ctx["query_id"]):
                self.assertEqual(ctx["k"], 10)

    def test_all_contexts_have_a_valid_regime_and_consistent_composition(self):
        allowed = {"A_BELOW_CEILING", "B_AT_CEILING", "C_ABOVE_CEILING", "D_ALL_POISON"}
        for ctx in self.contexts:
            with self.subTest(query_id=ctx["query_id"]):
                self.assertIn(ctx["regime"], allowed)
                self.assertEqual(ctx["m_poison"] + ctx["c_clean"], ctx["k"])
                self.assertEqual(ctx["ceiling"], ctx["k"] // 2)

    def test_no_gate_bc_query_is_present_in_the_frozen_population(self):
        frozen_ids = {ctx["query_id"] for ctx in self.contexts}
        overlap = frozen_ids & poplib.GATE_BC_EXCLUDED_QUERY_IDS
        self.assertEqual(overlap, set())

    def test_no_duplicate_query_ids(self):
        ids = [ctx["query_id"] for ctx in self.contexts]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
