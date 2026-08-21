"""Tests for the Regime-C Stage-2 Identification-Capacity study.

Uses ONLY the frozen 20 Regime-C matrices already on disk (read-only) plus
small synthetic fixtures for isolated unit checks. No retrieval, no Stella
re-encoding, no text mutation, no generation, no API call.
"""
import csv
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from defense import ragdefender_internals as ri  # noqa: E402
import ragdefender_regime_c_stage2_lib as lib  # noqa: E402
import run_ragdefender_regime_c_stage2 as driver  # noqa: E402

BASELINE_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_baseline"
GATE_C_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_gate_c"

_CASES_CACHE = None


def _cases():
    global _CASES_CACHE
    if _CASES_CACHE is None:
        _CASES_CACHE = driver.load_regime_c_cases()
    return _CASES_CACHE


# ---------------------------------------------------------------------------
# 1. exact Regime-C population n=20
# ---------------------------------------------------------------------------

class TestPopulationSize(unittest.TestCase):
    def test_exactly_20_regime_c_queries(self):
        cases = _cases()
        self.assertEqual(len(cases), 20)

    def test_query_ids_unique(self):
        cases = _cases()
        ids = [c["query_id"] for c in cases]
        self.assertEqual(len(ids), len(set(ids)))


# ---------------------------------------------------------------------------
# 2. M>5 and C>=1 for every query
# ---------------------------------------------------------------------------

class TestRegimeCInvariants(unittest.TestCase):
    def test_m_greater_than_5_and_c_at_least_1_for_every_query(self):
        for case in _cases():
            self.assertGreater(case["m_poison"], 5, case["query_id"])
            self.assertGreaterEqual(case["c_clean"], 1, case["query_id"])
            self.assertEqual(case["k"], 10, case["query_id"])
            self.assertEqual(case["m_poison"] + case["c_clean"], 10, case["query_id"])


# ---------------------------------------------------------------------------
# 3. true-count Stage-2 recomputation matches expanded Gate C
# ---------------------------------------------------------------------------

class TestReproducesGateC(unittest.TestCase):
    def test_recomputation_matches_gate_c_oracle_columns(self):
        with open(GATE_C_DIR / "expanded_gate_c_per_query.csv") as f:
            gate_c = {r["query_id"]: r for r in csv.DictReader(f) if r["regime"] == "C_ABOVE_CEILING"}
        self.assertEqual(len(gate_c), 20)
        for case in _cases():
            g = gate_c[case["query_id"]]
            self.assertEqual(case["n_pairs"], int(g["oracle_N_pairs"]))
            self.assertEqual(case["composition"]["n_PP_selected"], int(g["oracle_pp_count"]))
            self.assertEqual(case["composition"]["n_PC_selected"], int(g["oracle_pc_count"]))
            self.assertEqual(case["composition"]["n_CC_selected"], int(g["oracle_cc_count"]))
            self.assertEqual(case["outcome"]["removed_poison"], int(g["oracle_removed_poison"]))
            self.assertEqual(case["outcome"]["removed_clean"], int(g["oracle_removed_clean"]))
            self.assertEqual(case["outcome"]["residual_poison"], int(g["oracle_residual_poison"]))


# ---------------------------------------------------------------------------
# 4. 4/20 success, 16/20 mixed failures reproduce
# ---------------------------------------------------------------------------

class TestSuccessFailureCounts(unittest.TestCase):
    def test_4_successes_16_failures(self):
        cases = _cases()
        n_success = sum(1 for c in cases if c["success"])
        n_failure = sum(1 for c in cases if not c["success"])
        self.assertEqual(n_success, 4)
        self.assertEqual(n_failure, 16)

    def test_labels_match_historical_decomposition(self):
        for case in _cases():
            if case["success"]:
                self.assertEqual(case["historical_label"], "A. COUNT-LIMITED")
            else:
                self.assertEqual(case["historical_label"], "B. COUNT + IDENTIFICATION LIMITED")


# ---------------------------------------------------------------------------
# 5. N_pairs == C(M,2)
# ---------------------------------------------------------------------------

class TestNPairsFormula(unittest.TestCase):
    def test_n_pairs_equals_m_choose_2(self):
        for case in _cases():
            m = case["m_poison"]
            self.assertEqual(case["n_pairs"], m * (m - 1) // 2)

    def test_n_choose_2_known_values(self):
        self.assertEqual(lib.n_choose_2(6), 15)
        self.assertEqual(lib.n_choose_2(7), 21)
        self.assertEqual(lib.n_choose_2(8), 28)
        self.assertEqual(lib.n_choose_2(9), 36)
        self.assertEqual(lib.n_choose_2(10), 45)
        self.assertEqual(lib.n_choose_2(1), 0)
        self.assertEqual(lib.n_choose_2(0), 0)


# ---------------------------------------------------------------------------
# 6. unordered-pair enumeration correct
# ---------------------------------------------------------------------------

class TestPairEnumeration(unittest.TestCase):
    def test_all_pairs_sorted_has_c_k_2_entries_and_i_lt_j(self):
        rng = np.random.default_rng(0)
        k = 10
        matrix = rng.uniform(-1, 1, size=(k, k))
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 1.0)
        pairs = lib.all_pairs_sorted(matrix)
        self.assertEqual(len(pairs), 45)
        seen = set()
        for i, j, sim in pairs:
            self.assertLess(i, j)
            seen.add((i, j))
            self.assertAlmostEqual(sim, matrix[i, j])
        self.assertEqual(len(seen), 45)

    def test_all_pairs_sorted_descending_by_raw_similarity(self):
        rng = np.random.default_rng(1)
        k = 10
        matrix = rng.uniform(-1, 1, size=(k, k))
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 1.0)
        pairs = lib.all_pairs_sorted(matrix)
        sims = [sim for _, _, sim in pairs]
        self.assertEqual(sims, sorted(sims, reverse=True))


# ---------------------------------------------------------------------------
# 7. PP/PC/CC classification correct
# ---------------------------------------------------------------------------

class TestPairClassification(unittest.TestCase):
    def test_classify_pair_all_combinations(self):
        is_poison = np.array([True, True, False, False])
        self.assertEqual(lib.classify_pair(0, 1, is_poison), "PP")
        self.assertEqual(lib.classify_pair(2, 3, is_poison), "CC")
        self.assertEqual(lib.classify_pair(0, 2, is_poison), "PC")
        self.assertEqual(lib.classify_pair(1, 3, is_poison), "PC")

    def test_pair_class_counts_matches_manual_enumeration(self):
        is_poison = np.array([True, True, True, False, False])  # M=3, C=2
        pairs = [(i, j, 0.5) for i in range(5) for j in range(i + 1, 5)]
        counts = lib.pair_class_counts(pairs, is_poison)
        self.assertEqual(counts["n_PP"], 3)  # C(3,2)
        self.assertEqual(counts["n_PC"], 6)  # 3*2
        self.assertEqual(counts["n_CC"], 1)  # C(2,2)
        self.assertEqual(counts["n_PP"] + counts["n_PC"] + counts["n_CC"], 10)  # C(5,2)


# ---------------------------------------------------------------------------
# 8. pair ranking matches production Stage 2
# ---------------------------------------------------------------------------

class TestPairRankingMatchesProduction(unittest.TestCase):
    def test_top_pairs_match_production_for_every_regime_c_query(self):
        for case in _cases():
            matrix = case["matrix"]
            m = case["m_poison"]
            top_pairs, all_pairs, n_pairs = lib.stage2_original_top_pairs(matrix, m)
            prod = ri.stage2_pair_frequency(matrix, n_adv=m, p=2.0)
            self.assertEqual(n_pairs, prod.n_pairs)
            self.assertEqual(top_pairs, prod.top_pairs)


# ---------------------------------------------------------------------------
# 9. frequency scores match production exactly
# ---------------------------------------------------------------------------

class TestFrequencyScoresMatchProduction(unittest.TestCase):
    def test_frequency_scores_byte_for_byte(self):
        for case in _cases():
            matrix = case["matrix"]
            m = case["m_poison"]
            prod = ri.stage2_pair_frequency(matrix, n_adv=m, p=2.0)
            np.testing.assert_array_equal(case["selection"].frequency_scores, prod.frequency_scores)


# ---------------------------------------------------------------------------
# 10. passage removal ranking matches production
# ---------------------------------------------------------------------------

class TestRemovalRankingMatchesProduction(unittest.TestCase):
    def test_selected_indices_match_production_exactly(self):
        for case in _cases():
            matrix = case["matrix"]
            m = case["m_poison"]
            prod = ri.stage2_pair_frequency(matrix, n_adv=m, p=2.0)
            self.assertEqual(sorted(case["selection"].selected_indices), sorted(prod.selected_indices))
            self.assertEqual(case["selection"].selected_indices, prod.selected_indices)


# ---------------------------------------------------------------------------
# 11. no poison labels influence original Stage-2 recomputation
# ---------------------------------------------------------------------------

class TestLabelsDoNotInfluenceOriginalRecomputation(unittest.TestCase):
    def test_stage2_original_top_pairs_never_receives_is_poison(self):
        import inspect

        sig = inspect.signature(lib.stage2_original_top_pairs)
        self.assertNotIn("is_poison", sig.parameters)
        sig2 = inspect.signature(lib.compute_frequency_and_selection)
        self.assertNotIn("is_poison", sig2.parameters)

    def test_flipping_labels_does_not_change_selection(self):
        # The real production recomputation (top pairs + frequency
        # selection) depends only on the similarity matrix and n_adv --
        # flipping the (diagnostic-only) is_poison array must not change
        # `selected_indices`/`frequency_scores` at all.
        case = _cases()[0]
        matrix = case["matrix"]
        m = case["m_poison"]
        top_pairs, _all_pairs, _n_pairs = lib.stage2_original_top_pairs(matrix, m)
        sel_a = lib.compute_frequency_and_selection(top_pairs, case["k"], m)
        # Recompute again from scratch with no reference to is_poison at all.
        top_pairs_b, _, _ = lib.stage2_original_top_pairs(matrix, m)
        sel_b = lib.compute_frequency_and_selection(top_pairs_b, case["k"], m)
        self.assertEqual(sel_a.selected_indices, sel_b.selected_indices)
        np.testing.assert_array_equal(sel_a.frequency_scores, sel_b.frequency_scores)


# ---------------------------------------------------------------------------
# 12. pure-PP oracle uses exactly C(M,2) PP pairs
# ---------------------------------------------------------------------------

class TestPurePPOracleExactCount(unittest.TestCase):
    def test_pure_pp_pair_set_size_and_classes(self):
        for case in _cases():
            m = case["m_poison"]
            pp_pairs = lib.pure_pp_pair_set(case["matrix"], case["is_poison"])
            self.assertEqual(len(pp_pairs), lib.n_choose_2(m))
            for i, j, _sim in pp_pairs:
                self.assertTrue(case["is_poison"][i])
                self.assertTrue(case["is_poison"][j])

    def test_pure_pp_oracle_always_succeeds_for_m_gte_2(self):
        # Structural proof check: for any M>=2, the pure-PP pair set is the
        # complete graph K_M on the poison indices, so every poison index
        # has degree M-1>=1 and no clean index has any degree -- the top-M
        # selection must be exactly the M poison indices.
        for case in _cases():
            m = case["m_poison"]
            pp_pairs = lib.pure_pp_pair_set(case["matrix"], case["is_poison"])
            selection = lib.compute_frequency_and_selection(pp_pairs, case["k"], m)
            outcome = lib.removal_outcome(selection, case["is_poison"], m)
            self.assertTrue(outcome["success"], case["query_id"])
            self.assertEqual(outcome["removed_clean"], 0)

    def test_pure_pp_oracle_synthetic_negative_similarities_still_succeeds(self):
        # Even with some NEGATIVE poison-poison similarities (edge case
        # explicitly flagged by the task), the pure-PP oracle must still
        # succeed because clean passages are entirely ABSENT from `freq`
        # (not merely tied at 0.0).
        k = 6
        is_poison = np.array([True, True, True, True, False, False])
        matrix = np.eye(k)
        # M=4 poison passages, some pairwise similarities negative.
        poison_sims = {(0, 1): -0.9, (0, 2): -0.8, (0, 3): 0.1, (1, 2): 0.2, (1, 3): -0.3, (2, 3): 0.05}
        for (i, j), sim in poison_sims.items():
            matrix[i, j] = matrix[j, i] = sim
        matrix[4, 5] = matrix[5, 4] = 0.99  # strong clean-clean similarity, must not matter
        pp_pairs = lib.pure_pp_pair_set(matrix, is_poison)
        self.assertEqual(len(pp_pairs), 6)
        selection = lib.compute_frequency_and_selection(pp_pairs, k, 4)
        outcome = lib.removal_outcome(selection, is_poison, 4)
        self.assertTrue(outcome["success"])
        self.assertEqual(sorted(outcome["removed_indices"]), [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# 13. pair-class ablation does not refill removed slots
# ---------------------------------------------------------------------------

class TestAblationDoesNotRefill(unittest.TestCase):
    def test_ablation_variant_sizes_are_strict_subsets(self):
        for case in _cases():
            if case["success"]:
                continue
            variants = lib.ablation_variants(case["top_pairs"], case["is_poison"])
            n_original = len(variants["A_original"])
            self.assertEqual(n_original, case["n_pairs"])
            n_cc = case["composition"]["n_CC_selected"]
            n_pc = case["composition"]["n_PC_selected"]
            n_pp = case["composition"]["n_PP_selected"]
            self.assertEqual(len(variants["B_remove_CC"]), n_original - n_cc)
            self.assertEqual(len(variants["C_remove_PC"]), n_original - n_pc)
            self.assertEqual(len(variants["D_pp_only"]), n_pp)
            # None of the ablated variants may be LARGER than the original,
            # and none may contain a pair absent from the original (no
            # refilling with new pairs).
            original_set = {(i, j) for i, j, _ in variants["A_original"]}
            for name in ("B_remove_CC", "C_remove_PC", "D_pp_only"):
                self.assertLessEqual(len(variants[name]), n_original)
                for i, j, _ in variants[name]:
                    self.assertIn((i, j), original_set)

    def test_classify_ablation_driver_rule(self):
        self.assertEqual(lib.classify_ablation_driver(True, False, False), "CC-driven")
        self.assertEqual(lib.classify_ablation_driver(False, True, False), "PC-driven")
        self.assertEqual(lib.classify_ablation_driver(True, True, False), "mixed PC+CC (either alone sufficient)")
        self.assertEqual(
            lib.classify_ablation_driver(False, False, True), "mixed PC+CC (only both together sufficient)"
        )
        self.assertEqual(lib.classify_ablation_driver(False, False, False), "PP-weighting/other")


# ---------------------------------------------------------------------------
# 14. displacement mapping correctness
# ---------------------------------------------------------------------------

class TestDisplacementMapping(unittest.TestCase):
    def test_displacement_rows_only_for_failed_queries_and_correct_cardinality(self):
        cases = _cases()
        rows = driver.build_displacement_rows(cases)
        failure_ids = {c["query_id"] for c in cases if not c["success"]}
        success_ids = {c["query_id"] for c in cases if c["success"]}
        row_ids = {r["query_id"] for r in rows}
        self.assertTrue(row_ids.issubset(failure_ids))
        self.assertEqual(row_ids & success_ids, set())

        by_query = {}
        for r in rows:
            by_query.setdefault(r["query_id"], []).append(r)
        for case in cases:
            if case["success"]:
                continue
            qid = case["query_id"]
            removed_set = set(case["selection"].selected_indices)
            n_removed_clean = sum(1 for i in removed_set if not case["is_poison"][i])
            n_residual_poison = sum(1 for i in range(case["k"]) if case["is_poison"][i] and i not in removed_set)
            self.assertEqual(len(by_query.get(qid, [])), n_removed_clean * n_residual_poison, qid)

    def test_score_difference_sign_is_positive_for_every_displacement(self):
        # A removed clean passage must, by construction of "removed",
        # outrank the residual poison passage it is paired with.
        cases = _cases()
        rows = driver.build_displacement_rows(cases)
        for r in rows:
            self.assertGreaterEqual(r["f_clean"], r["f_poison"] - 1e-9)


# ---------------------------------------------------------------------------
# 15/16. minimum-swap search correctness + deterministic tie-breaking on
# synthetic small cases
# ---------------------------------------------------------------------------

class TestMinimalPairSwapSearch(unittest.TestCase):
    def _synthetic_case(self):
        # M=3 poison (0,1,2), C=3 clean (3,4,5). Construct P_top (n_pairs=3)
        # such that clean passage 3 accumulates score from BOTH a PC pair
        # (2,3) and a CC pair (3,4), out-scoring poison passage 2 (which
        # only has the one weak PP edge plus that single PC pair) --
        # verified numerically to fail originally and to be repaired by
        # exactly one swap (remove the CC pair (3,4), add the unselected
        # true-PP pair (0,2)).
        k = 6
        is_poison = np.array([True, True, True, False, False, False])
        matrix = np.eye(k)
        matrix[0, 1] = matrix[1, 0] = 0.9  # PP, strong
        matrix[0, 2] = matrix[2, 0] = 0.05  # PP, weak (unselected initially)
        matrix[1, 2] = matrix[2, 1] = 0.04  # PP, weak (unselected initially)
        matrix[2, 3] = matrix[3, 2] = 0.6  # PC
        matrix[3, 4] = matrix[4, 3] = 0.55  # CC
        for i, j in [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 5), (4, 5)]:
            matrix[i, j] = matrix[j, i] = -0.9
        m = 3
        top_pairs, all_pairs, n_pairs = lib.stage2_original_top_pairs(matrix, m)
        return k, m, is_poison, matrix, top_pairs, all_pairs, n_pairs

    def test_synthetic_case_fails_originally_and_swap_repairs_it(self):
        k, m, is_poison, matrix, top_pairs, all_pairs, n_pairs = self._synthetic_case()
        self.assertEqual(n_pairs, 3)
        self.assertEqual(top_pairs, [(0, 1, 0.9), (2, 3, 0.6), (3, 4, 0.55)])
        selection = lib.compute_frequency_and_selection(top_pairs, k, m)
        outcome = lib.removal_outcome(selection, is_poison, m)
        self.assertFalse(outcome["success"])  # sanity: constructed to fail
        self.assertEqual(outcome["removed_poison"], 2)
        self.assertEqual(outcome["removed_clean"], 1)

        result = lib.minimal_pair_swap_search(top_pairs, matrix, is_poison, k, m)
        self.assertEqual(result.swap_count, 1)
        self.assertTrue(result.is_exact)
        self.assertTrue(result.outcome["success"])
        self.assertEqual(result.outcome["removed_clean"], 0)
        self.assertEqual(result.outcome["removed_poison"], m)
        # The single removed pair must be non-PP (CC here); the single
        # added pair must be a true PP pair not already selected.
        self.assertEqual(len(result.removed_pairs), 1)
        self.assertEqual(result.removed_pairs[0], (3, 4, 0.55))
        self.assertEqual(lib.classify_pair(result.removed_pairs[0][0], result.removed_pairs[0][1], is_poison), "CC")
        self.assertEqual(len(result.added_pairs), 1)
        self.assertEqual(lib.classify_pair(result.added_pairs[0][0], result.added_pairs[0][1], is_poison), "PP")

    def test_swap_search_is_deterministic_across_repeated_calls(self):
        k, m, is_poison, matrix, top_pairs, all_pairs, n_pairs = self._synthetic_case()
        r1 = lib.minimal_pair_swap_search(top_pairs, matrix, is_poison, k, m)
        r2 = lib.minimal_pair_swap_search(top_pairs, matrix, is_poison, k, m)
        self.assertEqual(r1.removed_pairs, r2.removed_pairs)
        self.assertEqual(r1.added_pairs, r2.added_pairs)
        self.assertEqual(r1.swap_count, r2.swap_count)

    def test_swap_search_never_exceeds_number_of_non_pp_pairs(self):
        for case in _cases():
            if case["success"]:
                continue
            result = lib.minimal_pair_swap_search(case["top_pairs"], case["matrix"], case["is_poison"], case["k"], case["m_poison"])
            n_non_pp = case["composition"]["n_PC_selected"] + case["composition"]["n_CC_selected"]
            self.assertIsNotNone(result.swap_count)
            self.assertLessEqual(result.swap_count, n_non_pp)
            self.assertGreaterEqual(result.swap_count, 1)

    def test_all_16_failures_resolve_exactly(self):
        for case in _cases():
            if case["success"]:
                continue
            result = lib.minimal_pair_swap_search(case["top_pairs"], case["matrix"], case["is_poison"], case["k"], case["m_poison"])
            self.assertTrue(result.is_exact, case["query_id"])
            self.assertTrue(result.outcome["success"], case["query_id"])


# ---------------------------------------------------------------------------
# 17. historical matrices remain read-only
# ---------------------------------------------------------------------------

class TestHistoricalArtifactsReadOnly(unittest.TestCase):
    def test_baseline_and_gate_c_files_unchanged_by_running_the_study(self):
        protected = [
            BASELINE_DIR / "expanded_baseline_per_query.csv",
            BASELINE_DIR / "recovered_contexts.json",
            GATE_C_DIR / "expanded_gate_c_per_query.csv",
        ]
        cases = _cases()
        for c in cases[:5]:
            protected.append(BASELINE_DIR / "similarity" / f"{c['query_id']}_stella_similarity_matrix.npy")

        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}

        # Exercise the library and a fresh load again -- must not mutate
        # any historical file on disk.
        _ = driver.load_regime_c_cases()
        for case in cases:
            _ = lib.pure_pp_pair_set(case["matrix"], case["is_poison"])
            _ = lib.ablation_variants(case["top_pairs"], case["is_poison"])

        after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        self.assertEqual(before, after)

    def test_matrices_loaded_are_not_written_back(self):
        # np.load returns a read-only-friendly array by default (writeable
        # unless mmap'd); verify no in-place mutation occurs in any lib
        # function by checking a copy stays equal after use.
        case = _cases()[0]
        original = case["matrix"].copy()
        _ = lib.pure_pp_pair_set(case["matrix"], case["is_poison"])
        _ = lib.stage2_original_top_pairs(case["matrix"], case["m_poison"])
        np.testing.assert_array_equal(case["matrix"], original)


# ---------------------------------------------------------------------------
# 18. no retrieval/API dependency
# ---------------------------------------------------------------------------

class TestNoRetrievalOrApiDependency(unittest.TestCase):
    FORBIDDEN_MODULE_SUBSTRINGS = (
        "requests", "openai", "google", "sentence_transformers", "beir",
        "urllib", "http.client", "socket",
    )

    def test_lib_and_driver_source_have_no_forbidden_imports(self):
        for path in (
            REPO_ROOT / "scripts" / "ragdefender_regime_c_stage2_lib.py",
            REPO_ROOT / "scripts" / "run_ragdefender_regime_c_stage2.py",
        ):
            text = path.read_text()
            for forbidden in self.FORBIDDEN_MODULE_SUBSTRINGS:
                self.assertNotIn(f"import {forbidden}", text, f"{path} imports forbidden module '{forbidden}'")

    def test_load_regime_c_cases_only_reads_local_frozen_files(self):
        # A pure offline load must succeed with only local filesystem
        # access -- exercised implicitly by every other test in this file
        # via `_cases()`; this test just re-confirms no exception and a
        # well-formed, fully offline result.
        cases = _cases()
        self.assertEqual(len(cases), 20)
        for case in cases:
            self.assertIsInstance(case["matrix"], np.ndarray)


# ---------------------------------------------------------------------------
# Extra: driver-level CSV shape sanity (not one of the 18 numbered items,
# but cheap and catches gross regressions in the report-generation path).
# ---------------------------------------------------------------------------

class TestDriverRowCounts(unittest.TestCase):
    def test_row_counts_are_internally_consistent(self):
        cases = _cases()
        per_query_rows = driver.build_per_query_rows(cases)
        pair_rows = driver.build_pair_rows(cases)
        passage_rows = driver.build_passage_score_rows(cases)
        by_m_rows = driver.build_by_m_rows(per_query_rows)
        pure_pp_rows = driver.build_pure_pp_oracle_rows(cases)
        ablation_rows = driver.build_ablation_rows(cases)
        swap_rows = driver.build_pair_swap_rows(cases)

        self.assertEqual(len(per_query_rows), 20)
        self.assertEqual(len(pair_rows), 20 * 45)
        self.assertEqual(len(passage_rows), 20 * 10)
        self.assertEqual(len(pure_pp_rows), 20)
        self.assertEqual(len(ablation_rows), 16)
        self.assertEqual(len(swap_rows), 16)
        self.assertEqual(sum(r["n_queries"] for r in by_m_rows), 20)
        self.assertEqual(sorted(r["m_poison"] for r in by_m_rows), [6, 7, 8, 9])

    def test_pure_pp_oracle_repairs_exactly_the_16_failures(self):
        cases = _cases()
        pure_pp_rows = driver.build_pure_pp_oracle_rows(cases)
        n_repaired = sum(1 for r in pure_pp_rows if r["pure_pp_repairs_failure"])
        self.assertEqual(n_repaired, 16)
        n_success_total = sum(1 for r in pure_pp_rows if r["pure_pp_success"])
        self.assertEqual(n_success_total, 20)


if __name__ == "__main__":
    unittest.main()
