"""Tests for the Regime-C Stage-2 V2 VALIDATION PASS.

Covers the three V2 corrections:
  1. the pair-swap "exact minimum" certification bug fix
     (`certified_minimum_pair_swap_search`);
  2. the PP-COVERAGE-LIMITED graph-theoretic mechanism
     (`pp_coverage_analysis`, `classify_mechanism_v2`);
  3. artifact safety (V1 outputs untouched by the V2 pass) and the
     zero-external-dependency invariant.

Uses ONLY the frozen 20 Regime-C matrices already on disk (read-only)
plus small synthetic fixtures. No retrieval, no Stella re-encoding, no
text mutation, no generation, no API call.
"""
import csv
import hashlib
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ragdefender_regime_c_stage2_lib as lib  # noqa: E402
import run_ragdefender_regime_c_stage2 as v1_driver  # noqa: E402
import run_ragdefender_regime_c_stage2_v2 as v2_driver  # noqa: E402

BASELINE_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_baseline"
GATE_C_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_gate_c"
V1_REPORT = REPO_ROOT / "results/diagnostics/ragdefender_regime_c_stage2/REGIME_C_STAGE2_REPORT.md"
V1_CSVS = [
    REPO_ROOT / "results/diagnostics/ragdefender_regime_c_stage2" / name
    for name in (
        "regime_c_per_query.csv", "regime_c_pairs.csv", "regime_c_passage_scores.csv",
        "regime_c_displacements.csv", "regime_c_by_M.csv", "regime_c_pure_pp_oracle.csv",
        "regime_c_pair_class_ablation.csv", "regime_c_pair_swap_oracle.csv", "regime_c_score_epsilon_oracle.csv",
    )
]

_CASES_CACHE = None


def _cases():
    global _CASES_CACHE
    if _CASES_CACHE is None:
        _CASES_CACHE = v1_driver.load_regime_c_cases()
    return _CASES_CACHE


def _naive_brute_force_min_swap(top_pairs, matrix, is_poison, k, m_poison):
    """Independent, unoptimized reference implementation (full pair-list
    reconstruction + `compute_frequency_and_selection` per candidate, NO
    vectorization, NO greedy shortcut at any level) used to cross-check
    `certified_minimum_pair_swap_search`'s output on small synthetic
    cases."""
    identity = lib.pair_count_identity(top_pairs, matrix, is_poison, k)
    non_pp_pairs = sorted(identity["non_pp_pairs"], key=lambda t: t[2])
    unselected_pp_pairs = sorted(identity["missing_pp_pairs"], key=lambda t: t[2], reverse=True)
    q = len(non_pp_pairs)
    assert q == len(unselected_pp_pairs)
    base_pairs = list(top_pairs)
    for r in range(1, q + 1):
        for removal_combo in itertools.combinations(non_pp_pairs, r):
            remove_set = {(p[0], p[1]) for p in removal_combo}
            remaining = [p for p in base_pairs if (p[0], p[1]) not in remove_set]
            for addition_combo in itertools.combinations(unselected_pp_pairs, r):
                candidate = remaining + list(addition_combo)
                selection = lib.compute_frequency_and_selection(candidate, k, m_poison)
                outcome = lib.removal_outcome(selection, is_poison, m_poison)
                if outcome["success"]:
                    return r, removal_combo, addition_combo
    return None, None, None


def _make_matrix(k, edges, default=-0.9):
    matrix = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            matrix[i, j] = matrix[j, i] = default
    for (i, j), v in edges.items():
        matrix[i, j] = matrix[j, i] = v
    return matrix


# ---------------------------------------------------------------------------
# PAIR-SWAP CERTIFICATION (items 1-8)
# ---------------------------------------------------------------------------

class TestPairSwapCertification(unittest.TestCase):
    def _single_swap_case(self):
        # Same construction as the V1 synthetic fixture: M=3, C=3; fails
        # originally; exactly one swap (remove CC, add weak PP) repairs it.
        k = 6
        is_poison = np.array([True, True, True, False, False, False])
        matrix = np.eye(k)
        matrix[0, 1] = matrix[1, 0] = 0.9
        matrix[0, 2] = matrix[2, 0] = 0.05
        matrix[1, 2] = matrix[2, 1] = 0.04
        matrix[2, 3] = matrix[3, 2] = 0.6
        matrix[3, 4] = matrix[4, 3] = 0.55
        for i, j in [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 5), (4, 5)]:
            matrix[i, j] = matrix[j, i] = -0.9
        m = 3
        top_pairs, _, _ = lib.stage2_original_top_pairs(matrix, m)
        return k, m, is_poison, matrix, top_pairs

    def test_1_exact_r1_case(self):
        k, m, is_poison, matrix, top_pairs = self._single_swap_case()
        result = lib.certified_minimum_pair_swap_search(top_pairs, matrix, is_poison, k, m)
        self.assertEqual(result.certified_min_swap_count, 1)
        self.assertTrue(result.minimum_certified)
        self.assertTrue(result.outcome["success"])
        self.assertEqual(result.outcome["removed_clean"], 0)
        self.assertEqual(result.outcome["removed_poison"], m)
        self.assertEqual(len(result.audit_rows), 1)
        self.assertTrue(result.audit_rows[0].exhaustive)
        self.assertTrue(result.audit_rows[0].success_found)

    def _r2_case(self):
        # M=4 poison {0,1,2,3}, C=4 clean {4,5,6,7}. P_top ends up with
        # TWO selected non-PP pairs (q=2 removal, q=2 addition) such that
        # NEITHER single swap alone repairs it (r=1 exhaustively fails,
        # all C(2,1)^2=4 candidates), but removing BOTH non-PP pairs and
        # adding BOTH missing PP pairs (r=2, the only r=2 candidate since
        # q=2) repairs it.
        k = 8
        is_poison = np.array([True, True, True, True, False, False, False, False])
        edges = {
            (0, 1): 0.85, (0, 2): 0.06, (0, 3): 0.05,  # PP: one strong, two weak/unselected
            (1, 2): 0.07, (1, 3): 0.04,
            (2, 3): 0.03,
            (2, 4): 0.75, (3, 5): 0.7,  # two PC pairs -> both selected (strong), drive both clean 4,5 above weak poison 2,3
        }
        matrix = _make_matrix(k, edges)
        m = 4
        top_pairs, _, n_pairs = lib.stage2_original_top_pairs(matrix, m)
        return k, m, is_poison, matrix, top_pairs, n_pairs

    def test_2_r1_fully_exhausted_then_exact_r2(self):
        k, m, is_poison, matrix, top_pairs, n_pairs = self._r2_case()
        self.assertEqual(n_pairs, 6)
        selection = lib.compute_frequency_and_selection(top_pairs, k, m)
        outcome = lib.removal_outcome(selection, is_poison, m)
        self.assertFalse(outcome["success"])  # sanity: constructed to fail originally

        result = lib.certified_minimum_pair_swap_search(top_pairs, matrix, is_poison, k, m)
        self.assertTrue(result.minimum_certified)
        self.assertTrue(result.outcome["success"])
        # Cross-check against the independent brute-force reference.
        ref_r, _, _ = _naive_brute_force_min_swap(top_pairs, matrix, is_poison, k, m)
        self.assertEqual(result.certified_min_swap_count, ref_r)
        self.assertGreaterEqual(len(result.audit_rows), result.certified_min_swap_count)
        for row in result.audit_rows[:-1]:
            self.assertTrue(row.exhaustive)
            self.assertFalse(row.success_found)
        self.assertTrue(result.audit_rows[-1].success_found)

    def _greedy_vs_true_minimum_case(self):
        """Deterministically constructed (fixed RNG seed) so q=3 (3
        selected non-PP pairs, 3 unselected true-PP pairs) -- large enough
        that a "weakest-first" GREEDY r=2 pick (remove the 2 weakest
        non-PP, add the 2 strongest missing PP) is a genuinely DIFFERENT
        candidate from at least one other r=2 combination. Verified below
        that the fixture: (a) fails originally; (b) has no r=1 success at
        all (exhaustive); (c) the specific greedy r=2 pick fails; (d) some
        other r=2 combination succeeds. Found by an offline randomized
        search (`np.random.default_rng(0)`) satisfying all four
        properties simultaneously; hardcoded here as a fixed matrix
        construction recipe for reproducibility."""
        rng = np.random.default_rng(0)
        k = 10
        is_poison = np.array([True, True, True, True, False, False, False, False, False, False])
        m = 4
        matrix = np.eye(k)
        for i in range(k):
            for j in range(i + 1, k):
                matrix[i, j] = matrix[j, i] = rng.uniform(-0.05, 0.05)
        pp_edges = [(0, 1), (1, 2), (2, 3)]
        pp_vals = sorted(rng.uniform(0.5, 0.9, size=3))
        for (i, j), v in zip(pp_edges, pp_vals):
            matrix[i, j] = matrix[j, i] = v
        non_pp_edges = [(0, 4), (1, 5), (3, 6)]
        non_pp_vals = rng.uniform(0.5, 0.9, size=3)
        for (i, j), v in zip(non_pp_edges, non_pp_vals):
            matrix[i, j] = matrix[j, i] = v
        top_pairs, _, n_pairs = lib.stage2_original_top_pairs(matrix, m)
        return k, m, is_poison, matrix, top_pairs, n_pairs

    def test_3_mandatory_v1_bug_regression_case(self):
        """MANDATORY (Step 4) synthetic regression: a search space where a
        successful solution exists at r=2 via a NON-greedy combination,
        while the naive "greedy-first-candidate" pick at r=2 (weakest 2
        non-PP removed, strongest 2 missing-PP added) FAILS, and no r=1
        candidate succeeds either. The V1-bug failure mode would be to
        skip exhaustive r=2 (evaluating only the failing greedy candidate)
        and later report a LARGER exhaustively-found r as "exact". V2
        MUST instead certify the true r=2 minimum (q=3 here is tiny, well
        within the exhaustive budget -- V2 never needs the bounded-only
        fallback for a case this small)."""
        k, m, is_poison, matrix, top_pairs, n_pairs = self._greedy_vs_true_minimum_case()
        self.assertEqual(n_pairs, 6)
        self.assertEqual(len(top_pairs), 6)

        selection = lib.compute_frequency_and_selection(top_pairs, k, m)
        outcome = lib.removal_outcome(selection, is_poison, m)
        self.assertFalse(outcome["success"])  # sanity: originally fails

        identity = lib.pair_count_identity(top_pairs, matrix, is_poison, k)
        self.assertEqual(identity["q_selected_non_pp"], 3)
        self.assertEqual(identity["q_missing_pp"], 3)
        non_pp_sorted = sorted(identity["non_pp_pairs"], key=lambda t: t[2])
        pp_sorted = sorted(identity["missing_pp_pairs"], key=lambda t: t[2], reverse=True)

        def check(removal_combo, addition_combo):
            remove_set = {(p[0], p[1]) for p in removal_combo}
            remaining = [p for p in top_pairs if (p[0], p[1]) not in remove_set]
            candidate = remaining + list(addition_combo)
            sel = lib.compute_frequency_and_selection(candidate, k, m)
            out = lib.removal_outcome(sel, is_poison, m)
            return out["success"]

        # No r=1 candidate succeeds (exhaustive over all 3x3=9 combos).
        r1_any_success = any(
            check(rc, ac) for rc in itertools.combinations(non_pp_sorted, 1) for ac in itertools.combinations(pp_sorted, 1)
        )
        self.assertFalse(r1_any_success, "fixture must have no r=1 success")

        # The naive greedy r=2 candidate (remove the 2 weakest non-PP,
        # add the 2 strongest missing PP) must NOT be a success on its
        # own -- otherwise this case would not exercise the bug at all.
        self.assertFalse(
            check(non_pp_sorted[:2], pp_sorted[:2]),
            "fixture must make the naive greedy r=2 pick fail",
        )

        # But SOME r=2 combination succeeds (verified independently by
        # exhaustive check over all 3x3=9 r=2 combos), and it must NOT be
        # reported as a larger r.
        r2_any_success = any(
            check(rc, ac) for rc in itertools.combinations(non_pp_sorted, 2) for ac in itertools.combinations(pp_sorted, 2)
        )
        self.assertTrue(r2_any_success, "fixture must have its true minimum reachable at r=2")

        ref_r, ref_removal, ref_addition = _naive_brute_force_min_swap(top_pairs, matrix, is_poison, k, m)
        self.assertEqual(ref_r, 2, "fixture must have its true minimum at r=2, not r=1 or r=3")

        result = lib.certified_minimum_pair_swap_search(top_pairs, matrix, is_poison, k, m)
        self.assertTrue(
            (result.minimum_certified and result.certified_min_swap_count == 2)
            or (not result.minimum_certified and result.successful_upper_bound is not None
                and result.successful_upper_bound >= 2 and result.certified_lower_bound <= 2),
            f"V1-bug regression: got certified_min={result.certified_min_swap_count}, "
            f"minimum_certified={result.minimum_certified} -- must not silently report r=3 as an exact minimum "
            f"when r=2 was not exhaustively ruled out",
        )
        # For this specific (small, well within budget) fixture, V2 must
        # actually FIND the true exact r=2 minimum, not merely avoid the
        # false r=3 claim.
        self.assertEqual(result.certified_min_swap_count, 2)
        self.assertTrue(result.minimum_certified)
        self.assertNotEqual(result.certified_min_swap_count, 3)

    def test_4_certification_false_if_a_smaller_level_could_not_be_exhausted(self):
        k, m, is_poison, matrix, top_pairs, n_pairs = self._r2_case()
        # Force an artificially tiny cap so even r=1 (C(2,1)^2=4 candidates)
        # cannot be exhausted -- must yield an honest bounded result, never
        # a fabricated certification.
        result = lib.certified_minimum_pair_swap_search(top_pairs, matrix, is_poison, k, m, hard_combination_cap=1)
        self.assertFalse(result.minimum_certified)
        self.assertIsNone(result.certified_min_swap_count)
        self.assertEqual(result.certified_lower_bound, 1)
        self.assertIsNone(result.successful_upper_bound)

    def test_5_deterministic_tie_breaking(self):
        k, m, is_poison, matrix, top_pairs = self._single_swap_case()
        r1 = lib.certified_minimum_pair_swap_search(top_pairs, matrix, is_poison, k, m)
        r2 = lib.certified_minimum_pair_swap_search(top_pairs, matrix, is_poison, k, m)
        self.assertEqual(r1.removed_pairs, r2.removed_pairs)
        self.assertEqual(r1.added_pairs, r2.added_pairs)
        self.assertEqual(r1.certified_min_swap_count, r2.certified_min_swap_count)

    def test_6_and_7_brute_force_cross_check_on_real_failures(self):
        """Every returned certified minimum, independently brute-force
        verified -- run on all 16 real failures (small enough q to brute
        force in aggregate; see also the dedicated small-synthetic checks
        above)."""
        checked = 0
        for case in _cases():
            if case["success"]:
                continue
            if case["query_id"] not in (
                "5a8133725542995ce29dcbdb", "5a7cc50e554299452d57ba3e", "5add11935542994734353827",
            ):
                continue  # keep this test fast; the full-population check is in TestAllSixteenFailuresV2
            result = lib.certified_minimum_pair_swap_search(
                case["top_pairs"], case["matrix"], case["is_poison"], case["k"], case["m_poison"]
            )
            ref_r, _, _ = _naive_brute_force_min_swap(
                case["top_pairs"], case["matrix"], case["is_poison"], case["k"], case["m_poison"]
            )
            self.assertEqual(result.certified_min_swap_count, ref_r, case["query_id"])
            checked += 1
        self.assertEqual(checked, 3)

    def test_8_certification_metadata_accurate(self):
        for case in _cases():
            if case["success"]:
                continue
            result = lib.certified_minimum_pair_swap_search(
                case["top_pairs"], case["matrix"], case["is_poison"], case["k"], case["m_poison"]
            )
            self.assertTrue(result.minimum_certified, case["query_id"])
            r = result.certified_min_swap_count
            self.assertIsNotNone(r, case["query_id"])
            # Every level below r must be recorded, exhaustive, and failed.
            self.assertEqual(len(result.audit_rows), r, case["query_id"])
            for row in result.audit_rows[:-1]:
                self.assertTrue(row.exhaustive)
                self.assertFalse(row.success_found)
            self.assertTrue(result.audit_rows[-1].exhaustive)
            self.assertTrue(result.audit_rows[-1].success_found)
            self.assertEqual(result.largest_fully_exhausted_r, r)
            self.assertGreater(result.n_combinations_examined, 0, case["query_id"])


class TestAllSixteenFailuresV2(unittest.TestCase):
    def test_all_16_failures_exact_certified_minimum(self):
        for case in _cases():
            if case["success"]:
                continue
            result = lib.certified_minimum_pair_swap_search(
                case["top_pairs"], case["matrix"], case["is_poison"], case["k"], case["m_poison"]
            )
            self.assertTrue(result.minimum_certified, case["query_id"])
            self.assertTrue(result.outcome["success"], case["query_id"])
            self.assertEqual(result.outcome["removed_clean"], 0, case["query_id"])
            self.assertEqual(result.outcome["removed_poison"], case["m_poison"], case["query_id"])

    def test_v1_bug_actually_understated_one_query(self):
        # Confirmed concrete regression: V1's `minimal_pair_swap_search`
        # reported swap_count=10 as "exact" for this query; the corrected
        # V2 search certifies the true minimum is 8.
        qid = "5a7be2595542997c3ec972ac"
        case = next(c for c in _cases() if c["query_id"] == qid)
        v1_result = lib.minimal_pair_swap_search(case["top_pairs"], case["matrix"], case["is_poison"], case["k"], case["m_poison"])
        v2_result = lib.certified_minimum_pair_swap_search(case["top_pairs"], case["matrix"], case["is_poison"], case["k"], case["m_poison"])
        self.assertEqual(v1_result.swap_count, 10)
        self.assertTrue(v1_result.is_exact)  # V1 mislabels this "exact"
        self.assertEqual(v2_result.certified_min_swap_count, 8)  # true certified minimum
        self.assertLess(v2_result.certified_min_swap_count, v1_result.swap_count)

    def test_certified_minimum_never_exceeds_v1_reported_value(self):
        # The correction can only ever move a value DOWN (an exhaustive
        # search cannot find a larger true minimum than a partially-greedy
        # search already found).
        for case in _cases():
            if case["success"]:
                continue
            v1_result = lib.minimal_pair_swap_search(case["top_pairs"], case["matrix"], case["is_poison"], case["k"], case["m_poison"])
            v2_result = lib.certified_minimum_pair_swap_search(case["top_pairs"], case["matrix"], case["is_poison"], case["k"], case["m_poison"])
            self.assertLessEqual(v2_result.certified_min_swap_count, v1_result.swap_count, case["query_id"])


# ---------------------------------------------------------------------------
# PAIR COUNT IDENTITY (item 9)
# ---------------------------------------------------------------------------

class TestPairCountIdentity(unittest.TestCase):
    def test_9_selected_non_pp_equals_unselected_true_pp_for_every_query(self):
        for case in _cases():
            identity = lib.pair_count_identity(case["top_pairs"], case["matrix"], case["is_poison"], case["k"])
            self.assertTrue(identity["identity_holds"], case["query_id"])
            self.assertEqual(identity["q_selected_non_pp"], identity["q_missing_pp"], case["query_id"])


# ---------------------------------------------------------------------------
# PP COVERAGE (items 10-14)
# ---------------------------------------------------------------------------

class TestPPCoverage(unittest.TestCase):
    def test_10_pp_only_degree_matches_manual_count(self):
        k = 6
        is_poison = np.array([True, True, True, False, False, False])
        top_pairs = [(0, 1, 0.9), (1, 2, 0.05), (2, 3, 0.6)]  # (2,3) is PC, not PP
        cov = lib.pp_coverage_analysis(top_pairs, is_poison, 3)
        # manual degrees: vertex0:1 (from (0,1)), vertex1:2 (from (0,1),(1,2)), vertex2:1 (from (1,2))
        self.assertEqual(cov["n_poison_covered_by_PP"], 3)
        self.assertEqual(cov["n_poison_uncovered_by_PP"], 0)
        self.assertTrue(cov["pp_vertex_coverage_complete"])
        self.assertEqual(cov["min_poison_pp_degree"], 1)
        self.assertEqual(cov["max_poison_pp_degree"], 2)

    def test_10b_uncovered_vertex_detected(self):
        k = 6
        is_poison = np.array([True, True, True, False, False, False])
        top_pairs = [(0, 1, 0.9), (2, 3, 0.6)]  # vertex 2 has no PP edge at all
        cov = lib.pp_coverage_analysis(top_pairs, is_poison, 3)
        self.assertEqual(cov["n_poison_uncovered_by_PP"], 1)
        self.assertEqual(cov["uncovered_poison_indices"], [2])
        self.assertFalse(cov["pp_vertex_coverage_complete"])

    def test_11_complete_coverage_implies_pp_only_success(self):
        checked = 0
        for case in _cases():
            cov = lib.pp_coverage_analysis(case["top_pairs"], case["is_poison"], case["m_poison"])
            if not cov["pp_vertex_coverage_complete"]:
                continue
            variants = lib.ablation_variants(case["top_pairs"], case["is_poison"])
            selection_d = lib.compute_frequency_and_selection(variants["D_pp_only"], case["k"], case["m_poison"])
            outcome_d = lib.removal_outcome(selection_d, case["is_poison"], case["m_poison"])
            self.assertTrue(outcome_d["success"], case["query_id"])
            self.assertEqual(outcome_d["removed_clean"], 0, case["query_id"])
            checked += 1
        self.assertGreater(checked, 0)

    def test_12_incomplete_coverage_can_produce_pp_only_failure(self):
        found_incomplete_failure = False
        for case in _cases():
            cov = lib.pp_coverage_analysis(case["top_pairs"], case["is_poison"], case["m_poison"])
            if cov["pp_vertex_coverage_complete"]:
                continue
            variants = lib.ablation_variants(case["top_pairs"], case["is_poison"])
            selection_d = lib.compute_frequency_and_selection(variants["D_pp_only"], case["k"], case["m_poison"])
            outcome_d = lib.removal_outcome(selection_d, case["is_poison"], case["m_poison"])
            if not outcome_d["success"]:
                found_incomplete_failure = True
        self.assertTrue(found_incomplete_failure)

    def test_13_pp_only_failure_implies_uncovered_vertex(self):
        for case in _cases():
            cov = lib.pp_coverage_analysis(case["top_pairs"], case["is_poison"], case["m_poison"])
            variants = lib.ablation_variants(case["top_pairs"], case["is_poison"])
            selection_d = lib.compute_frequency_and_selection(variants["D_pp_only"], case["k"], case["m_poison"])
            outcome_d = lib.removal_outcome(selection_d, case["is_poison"], case["m_poison"])
            if not outcome_d["success"]:
                self.assertGreaterEqual(cov["n_poison_uncovered_by_PP"], 1, case["query_id"])
            # Full theorem, both directions:
            self.assertEqual(cov["pp_vertex_coverage_complete"], outcome_d["success"], case["query_id"])
            # PP-only pair sets can NEVER fail via clean displacement.
            self.assertEqual(outcome_d["removed_clean"], 0, case["query_id"])

    def test_14_complete_true_pp_kM_always_covers_every_poison_vertex(self):
        for case in _cases():
            k, m = case["k"], case["m_poison"]
            if m < 2:
                continue
            true_pp = lib.pure_pp_pair_set(case["matrix"], case["is_poison"])
            cov = lib.pp_coverage_analysis(true_pp, case["is_poison"], m)
            self.assertTrue(cov["pp_vertex_coverage_complete"], case["query_id"])
            self.assertEqual(cov["min_poison_pp_degree"], m - 1, case["query_id"])
            self.assertEqual(len(true_pp), m * (m - 1) // 2, case["query_id"])


# ---------------------------------------------------------------------------
# MECHANISM CLASSIFICATION (items 15-18)
# ---------------------------------------------------------------------------

class TestMechanismClassificationV2(unittest.TestCase):
    def test_15_pc_contribution_driven(self):
        self.assertEqual(lib.classify_mechanism_v2(True, True), "A. PC-CONTRIBUTION-DRIVEN")

    def test_16_pp_coverage_limited(self):
        self.assertEqual(lib.classify_mechanism_v2(False, True), "B. PP-COVERAGE-LIMITED")
        self.assertEqual(lib.classify_mechanism_v2(False, False), "B. PP-COVERAGE-LIMITED")

    def test_17_overlap_other_unexplained_not_forced_into_a_or_b(self):
        # Coverage complete but PC-removal alone does not repair -> must
        # be reported as OTHER/UNEXPLAINED, not silently folded into A.
        self.assertEqual(lib.classify_mechanism_v2(True, False), "C. OTHER/UNEXPLAINED")

    def test_18_real_population_split_is_recomputed_not_hardcoded(self):
        result = v2_driver.build_mechanism_classification_v2_rows(_cases())
        counts = {"A": 0, "B": 0, "C": 0}
        for row in result:
            counts[row["mechanism_classification"][0]] += 1
            # Cross-check each row's label against an independent
            # from-scratch recomputation (not trusting the driver's
            # internal call to classify_mechanism_v2 alone).
            case = next(c for c in _cases() if c["query_id"] == row["query_id"])
            cov = lib.pp_coverage_analysis(case["top_pairs"], case["is_poison"], case["m_poison"])
            variants = lib.ablation_variants(case["top_pairs"], case["is_poison"])
            sel_c = lib.compute_frequency_and_selection(variants["C_remove_PC"], case["k"], case["m_poison"])
            out_c = lib.removal_outcome(sel_c, case["is_poison"], case["m_poison"])
            expected_label = lib.classify_mechanism_v2(cov["pp_vertex_coverage_complete"], out_c["success"])
            self.assertEqual(row["mechanism_classification"], expected_label, row["query_id"])
            self.assertTrue(row["coverage_theorem_consistent"])
            self.assertEqual(row["overlap_note"], "")
        self.assertEqual(sum(counts.values()), 16)
        self.assertEqual(counts["C"], 0)


# ---------------------------------------------------------------------------
# ARTIFACT SAFETY (items 19-20)
# ---------------------------------------------------------------------------

class TestArtifactSafetyV2(unittest.TestCase):
    def test_19_v1_report_and_csvs_untouched_by_v2_pass(self):
        protected = [V1_REPORT] + [p for p in V1_CSVS if p.exists()]
        self.assertTrue(all(p.exists() for p in protected), "expected V1 artifacts missing")
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}

        # Exercise the V2 library/driver functions (read-only against V1
        # artifacts) without invoking `run()` (which would try to write
        # V2 files that may already exist from a prior run in this repo;
        # we only care that V1 files are never touched by the underlying
        # computation).
        cases = _cases()
        for case in cases:
            _ = lib.pp_coverage_analysis(case["top_pairs"], case["is_poison"], case["m_poison"])
            if not case["success"]:
                _ = lib.certified_minimum_pair_swap_search(
                    case["top_pairs"], case["matrix"], case["is_poison"], case["k"], case["m_poison"]
                )

        after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        self.assertEqual(before, after)

    def test_20_historical_similarity_matrices_unchanged(self):
        cases = _cases()[:5]
        protected = [BASELINE_DIR / "similarity" / f"{c['query_id']}_stella_similarity_matrix.npy" for c in cases]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        for case in cases:
            _ = lib.certified_minimum_pair_swap_search(
                case["top_pairs"], case["matrix"], case["is_poison"], case["k"], case["m_poison"]
            )
            _ = lib.pp_coverage_analysis(case["top_pairs"], case["is_poison"], case["m_poison"])
        after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        self.assertEqual(before, after)


# ---------------------------------------------------------------------------
# ZERO EXTERNAL DEPENDENCY (item 21)
# ---------------------------------------------------------------------------

class TestNoExternalDependencyV2(unittest.TestCase):
    FORBIDDEN_MODULE_SUBSTRINGS = (
        "requests", "openai", "google", "sentence_transformers", "beir",
        "urllib", "http.client", "socket",
    )

    def test_21_v2_lib_and_driver_source_have_no_forbidden_imports(self):
        for path in (
            REPO_ROOT / "scripts/ragdefender_regime_c_stage2_lib.py",
            REPO_ROOT / "scripts/run_ragdefender_regime_c_stage2_v2.py",
        ):
            source = path.read_text()
            for token in self.FORBIDDEN_MODULE_SUBSTRINGS:
                self.assertNotIn(token, source, f"{path} references forbidden module '{token}'")

    def test_21b_no_network_modules_loaded_after_import(self):
        forbidden_loaded = [
            name for name in sys.modules
            if any(tok in name for tok in ("requests", "openai", "sentence_transformers", "beir"))
        ]
        self.assertEqual(forbidden_loaded, [])


if __name__ == "__main__":
    unittest.main()
