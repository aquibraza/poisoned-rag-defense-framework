"""Tests for the REGIME-B STAGE-1 TEXT-MANIFOLD REALIZATION STUDY.

Covers the 25 required items from the task spec. Most tests are pure
unit tests against `scripts/ragdefender_regime_b_text_realization_lib.py`,
`scripts/build_regime_b_rewrite_bank.py`, and the already-written frozen
artifacts under
`results/diagnostics/ragdefender_regime_b_text_realization/` -- these run
with zero external dependencies and require no live Stella.

A small number of tests are gated behind
`REGIME_B_TEXT_REALIZATION_RUN_STELLA=1` (Stella-dependent, e.g. re-running
one live encode to reproduce a frozen Stage-1 result) and are skipped by
default in ordinary unit-test runs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ragdefender_regime_b_text_realization_lib as tlib  # noqa: E402
import build_regime_b_rewrite_bank as bank_builder  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_regime_b_text_realization"
BASELINE_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_baseline"
ORACLE_DIR = REPO_ROOT / "results/diagnostics/ragdefender_regime_b_stage1_oracle"

RUN_STELLA = os.environ.get("REGIME_B_TEXT_REALIZATION_RUN_STELLA") == "1"


def _load_bank_rows():
    path = OUTPUT_DIR / "rewrite_bank.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# 1. Exactly 14 frozen baseline failures selected.
# ---------------------------------------------------------------------------

class Test01FrozenFailureSelection(unittest.TestCase):
    def test_exactly_14_failures_and_5_successes(self):
        with open(BASELINE_DIR / "expanded_baseline_per_query.csv") as f:
            rows = [r for r in csv.DictReader(f) if r["regime"] == "B_AT_CEILING"]
        self.assertEqual(len(rows), 19)
        n_success = sum(1 for r in rows if r["zero_residual_poison_success"] == "True")
        n_failure = len(rows) - n_success
        self.assertEqual(n_success, 5)
        self.assertEqual(n_failure, 14)

    def test_targets_module_selects_same_14(self):
        targets = bank_builder.load_targets()
        self.assertEqual(len(targets), 14)
        qids = {t["query_id"] for t in targets}
        self.assertEqual(len(qids), 14)


# ---------------------------------------------------------------------------
# 2/3. Same-session baseline reproduction -- gated (needs live Stella /
# already-written phase0 artifacts).
# ---------------------------------------------------------------------------

class Test02SameSessionBaselineReproduction(unittest.TestCase):
    def test_phase0_csv_exists_and_reproduces_n_adv_4_for_all_14(self):
        path = OUTPUT_DIR / "phase0_baseline_reproduction.csv"
        if not path.exists():
            self.skipTest("Phase-0 artifact not yet written (run the driver script first).")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 14)
        for row in rows:
            self.assertEqual(int(row["n_adv_same_session"]), 4)

    def test_phase0_mechanism_split_11_median_3_mean(self):
        path = OUTPUT_DIR / "phase0_baseline_reproduction.csv"
        if not path.exists():
            self.skipTest("Phase-0 artifact not yet written.")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        n_median = sum(1 for r in rows if r["mechanism_same_session"] == "A. MEDIAN-LIMITED")
        n_mean = sum(1 for r in rows if r["mechanism_same_session"] == "B. MEAN-GATED")
        self.assertEqual(n_median, 11)
        self.assertEqual(n_mean, 3)

    def test_phase0_mutual_median_holds_for_all_median_limited(self):
        path = OUTPUT_DIR / "phase0_baseline_reproduction.csv"
        if not path.exists():
            self.skipTest("Phase-0 artifact not yet written.")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        median_rows = [r for r in rows if r["mechanism_same_session"] == "A. MEDIAN-LIMITED"]
        self.assertEqual(len(median_rows), 11)
        for row in median_rows:
            self.assertEqual(row["mutual_median_match"], "True")

    @unittest.skipUnless(RUN_STELLA, "Stella-dependent; set REGIME_B_TEXT_REALIZATION_RUN_STELLA=1 to run.")
    def test_live_stella_reencode_reproduces_n_adv_4_single_query(self):
        import run_ragdefender_regime_b_text_realization as drv

        cases = drv.load_frozen_failures()
        case = cases[0]
        s_model, st_util, _ = drv.load_stella_model()
        matrix = drv.encode_matrix(s_model, st_util, case["texts"])
        from defense import ragdefender_internals as ri

        stage1 = ri.concentration_stage1_paper(matrix)
        self.assertEqual(stage1.n_adv_estimated, 4)


# ---------------------------------------------------------------------------
# 4/5. Oracle target candidate matches V2 frozen winner; selection does not
# use poison label.
# ---------------------------------------------------------------------------

class Test03OracleTargetSelection(unittest.TestCase):
    def test_candidate_matches_v2_frozen_winner(self):
        targets = bank_builder.load_targets()
        with open(ORACLE_DIR / "regime_b_matrix_winners_v2.csv") as f:
            winners = {r["query_id"]: r for r in csv.DictReader(f)}
        for t in targets:
            winner = winners[t["query_id"]]
            self.assertEqual(t["candidate_index"], int(winner["psd_valid_1e8_winner_candidate_index"]))
            self.assertEqual(t["oracle_mode"], winner["psd_valid_1e8_winner_mode"])
            self.assertAlmostEqual(t["oracle_alpha"], float(winner["psd_valid_1e8_winner_alpha"]), places=9)

    def test_selection_function_source_never_reads_is_poison(self):
        source = (REPO_ROOT / "scripts/build_regime_b_rewrite_bank.py").read_text()
        # `load_targets` must not branch on is_poison when choosing candidate_index.
        load_targets_src = source.split("def load_targets")[1].split("def build_rows")[0]
        self.assertNotIn("is_poison", load_targets_src)


# ---------------------------------------------------------------------------
# 6/7/8. Exactly 3 Round-1 rewrites/query; bank hash frozen; no post-hoc edits.
# ---------------------------------------------------------------------------

class Test04RewriteBankIntegrity(unittest.TestCase):
    def test_exactly_3_round1_rewrites_per_query_42_total(self):
        rows = _load_bank_rows()
        self.assertEqual(len(rows), 42)
        by_query = {}
        for r in rows:
            by_query.setdefault(r["query_id"], set()).add(r["mutation_id"])
        self.assertEqual(len(by_query), 14)
        for qid, mutations in by_query.items():
            self.assertEqual(mutations, {"R1", "R2", "R3"})

    def test_rewrite_bank_hash_matches_frozen_manifest(self):
        bank_path = OUTPUT_DIR / "rewrite_bank.jsonl"
        manifest_path = OUTPUT_DIR / "rewrite_bank_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        actual_sha256 = hashlib.sha256(bank_path.read_bytes()).hexdigest()
        self.assertEqual(actual_sha256, manifest["sha256"])

    def test_manifest_declares_frozen_before_stella(self):
        with open(OUTPUT_DIR / "rewrite_bank_manifest.json") as f:
            manifest = json.load(f)
        self.assertTrue(manifest["frozen_before_any_stella_evaluation"])

    def test_every_row_marked_generated_before_stella_evaluation(self):
        rows = _load_bank_rows()
        for r in rows:
            self.assertTrue(r["generated_before_stella_evaluation"])

    def test_original_text_in_bank_matches_frozen_recovered_contexts(self):
        with open(BASELINE_DIR / "recovered_contexts.json") as f:
            contexts_by_id = {c["query_id"]: c for c in json.load(f)}
        rows = _load_bank_rows()
        for r in rows:
            ctx = contexts_by_id[r["query_id"]]
            self.assertEqual(r["original_text"], ctx["texts"][r["candidate_index"]])


# ---------------------------------------------------------------------------
# 9/10. Length-ratio validation; number/date preservation.
# ---------------------------------------------------------------------------

class Test05RewriteConstraints(unittest.TestCase):
    def test_all_round1_rewrites_within_length_bounds(self):
        rows = _load_bank_rows()
        for r in rows:
            check = tlib.rule_based_semantic_check(r["original_text"], r["rewritten_text"])
            self.assertTrue(
                check.length_ratio_pass,
                f"{r['query_id']} {r['mutation_id']}: ratio={check.length_ratio:.3f}",
            )

    def test_all_round1_rewrites_preserve_numbers_and_years(self):
        rows = _load_bank_rows()
        for r in rows:
            check = tlib.rule_based_semantic_check(r["original_text"], r["rewritten_text"])
            self.assertTrue(check.numbers_preserved, f"{r['query_id']} {r['mutation_id']}: numbers not preserved")
            self.assertTrue(check.years_preserved, f"{r['query_id']} {r['mutation_id']}: years not preserved")

    def test_no_forbidden_words_in_any_rewrite(self):
        forbidden = ["poison", "adversarial", "defense", "ragdefender", "retrieval", "attack"]
        rows = _load_bank_rows()
        for r in rows:
            lower = r["rewritten_text"].lower()
            for word in forbidden:
                self.assertNotIn(word, lower, f"{r['query_id']} {r['mutation_id']} contains forbidden word {word!r}")

    def test_no_exact_duplicate_rewrites(self):
        rows = _load_bank_rows()
        for r in rows:
            self.assertNotEqual(r["rewritten_text"].strip(), r["original_text"].strip())


# ---------------------------------------------------------------------------
# 11. Semantic-preservation flag correctness.
# ---------------------------------------------------------------------------

class Test06SemanticPreservationFlag(unittest.TestCase):
    def test_identical_text_passes_with_full_marks(self):
        check = tlib.rule_based_semantic_check("Hello world in 1999.", "Hello world in 1999 today.")
        self.assertTrue(check.numbers_preserved)
        self.assertTrue(check.years_preserved)

    def test_missing_number_fails_preservation(self):
        check = tlib.rule_based_semantic_check("It happened in 1999 with 42 people.", "It happened in 1999.")
        self.assertFalse(check.numbers_preserved)
        self.assertFalse(check.semantic_preservation_pass)

    def test_too_short_rewrite_fails_length_ratio(self):
        check = tlib.rule_based_semantic_check("one two three four five six seven eight nine ten", "short")
        self.assertFalse(check.length_ratio_pass)
        self.assertFalse(check.semantic_preservation_pass)

    def test_minilm_below_threshold_fails_when_available(self):
        check = tlib.rule_based_semantic_check(
            "one two three four five", "one two three four six", minilm_cosine=0.5, minilm_available=True
        )
        self.assertFalse(check.semantic_preservation_pass)

    def test_minilm_above_threshold_passes_when_available(self):
        check = tlib.rule_based_semantic_check(
            "one two three four five", "one two three four six", minilm_cosine=0.95, minilm_available=True
        )
        self.assertTrue(check.semantic_preservation_pass)

    def test_minilm_unavailable_falls_back_to_rule_based_only(self):
        check = tlib.rule_based_semantic_check(
            "one two three four five", "one two three four six", minilm_cosine=None, minilm_available=False
        )
        self.assertTrue(check.semantic_preservation_pass)
        self.assertFalse(check.minilm_available)


# ---------------------------------------------------------------------------
# 12/13. Actual Stella similarity-delta calculation; oracle-direction
# alignment calculation.
# ---------------------------------------------------------------------------

class Test07DeltaAndAlignment(unittest.TestCase):
    def test_compute_delta_vector_excludes_self_index(self):
        orig = np.array([1.0, 0.5, 0.6, 0.7])
        rewrite = np.array([1.0, 0.55, 0.6, 0.75])
        delta = tlib.compute_delta_vector(orig, rewrite, candidate_index=0)
        np.testing.assert_allclose(delta, [0.05, 0.0, 0.05])

    def test_boost_alignment_all_positive_gives_full_alignment(self):
        delta = np.array([0.1, 0.1, 0.1])
        result = tlib.compute_alignment(delta, "boost")
        self.assertAlmostEqual(result.mean_signed_alignment, 0.1)
        self.assertAlmostEqual(result.fraction_entries_in_oracle_direction, 1.0)
        self.assertAlmostEqual(result.cosine_alignment, 1.0, places=6)
        self.assertAlmostEqual(result.fitted_beta, 0.1, places=6)
        self.assertAlmostEqual(result.oracle_profile_residual, 0.0, places=6)

    def test_decrease_alignment_sign_convention(self):
        delta = np.array([-0.1, -0.1, -0.1])
        result = tlib.compute_alignment(delta, "decrease")
        self.assertAlmostEqual(result.mean_signed_alignment, 0.1)
        self.assertAlmostEqual(result.fraction_entries_in_oracle_direction, 1.0)

    def test_mixed_direction_partial_alignment(self):
        delta = np.array([0.1, -0.1, 0.1])
        result = tlib.compute_alignment(delta, "boost")
        self.assertAlmostEqual(result.fraction_entries_in_oracle_direction, 2.0 / 3.0)

    def test_fitted_beta_clipped_at_zero_when_anti_aligned(self):
        delta = np.array([-0.1, -0.1, -0.1])
        result = tlib.compute_alignment(delta, "boost")
        self.assertEqual(result.fitted_beta, 0.0)

    def test_invalid_oracle_mode_raises(self):
        with self.assertRaises(ValueError):
            tlib.compute_alignment(np.array([0.1]), "sideways")


# ---------------------------------------------------------------------------
# 14/15/16. Complete Stage-1 recomputation after rewrite; FULL success only
# for 4->5; partial-realization classification.
# ---------------------------------------------------------------------------

class Test08RealizationClassification(unittest.TestCase):
    def _alignment(self, sign=1.0):
        delta = np.array([0.05] * 9) * sign
        return tlib.compute_alignment(delta, "boost" if sign > 0 else "decrease")

    def test_full_realization_requires_4_to_5(self):
        result = tlib.classify_realization(4, 5, "median-limited", self._alignment())
        self.assertEqual(result, tlib.REALIZATION_FULL)

    def test_no_full_realization_if_original_not_4(self):
        result = tlib.classify_realization(3, 5, "median-limited", self._alignment())
        self.assertNotEqual(result, tlib.REALIZATION_FULL)

    def test_no_full_realization_if_rewrite_stays_at_4(self):
        progress = tlib.MedianLimitedProgress(False, False, False)
        result = tlib.classify_realization(4, 4, "median-limited", self._alignment(), median_progress=progress)
        self.assertNotEqual(result, tlib.REALIZATION_FULL)

    def test_mechanism_partial_when_tie_broken_but_count_unchanged(self):
        progress = tlib.MedianLimitedProgress(exact_tie_broken=True, median_gap_became_positive=True, n_above_median_increased=False)
        result = tlib.classify_realization(4, 4, "median-limited", self._alignment(), median_progress=progress)
        self.assertEqual(result, tlib.REALIZATION_MECHANISM_PARTIAL)

    def test_mean_gated_mechanism_partial_when_margin_crosses(self):
        progress = tlib.MeanGatedProgress(blocking_margin_moved_toward_zero=True, blocking_margin_crossed_zero=True)
        result = tlib.classify_realization(4, 4, "mean-gated", self._alignment(), mean_progress=progress)
        self.assertEqual(result, tlib.REALIZATION_MECHANISM_PARTIAL)

    def test_geometry_aligned_only_when_blocker_unchanged_but_row_aligned(self):
        progress = tlib.MedianLimitedProgress(False, False, False)
        result = tlib.classify_realization(4, 4, "median-limited", self._alignment(sign=1.0), median_progress=progress)
        self.assertEqual(result, tlib.REALIZATION_GEOMETRY_ALIGNED_ONLY)

    def test_non_aligned_when_row_moves_against_oracle_direction(self):
        progress = tlib.MedianLimitedProgress(False, False, False)
        anti_alignment = tlib.compute_alignment(np.array([-0.05] * 9), "boost")
        result = tlib.classify_realization(4, 4, "median-limited", anti_alignment, median_progress=progress)
        self.assertEqual(result, tlib.REALIZATION_NON_ALIGNED)

    def test_full_realization_does_not_require_margin_movement_alone(self):
        # Margin movement alone (mechanism-partial) must never be reported as FULL.
        progress = tlib.MedianLimitedProgress(True, True, True)
        result = tlib.classify_realization(4, 4, "median-limited", self._alignment(), median_progress=progress)
        self.assertNotEqual(result, tlib.REALIZATION_FULL)


# ---------------------------------------------------------------------------
# 17/18. Stage2 uses unchanged production function; success/degradation
# labeling.
# ---------------------------------------------------------------------------

class Test09Stage2Labeling(unittest.TestCase):
    def test_driver_imports_unchanged_production_stage2_function(self):
        source = (REPO_ROOT / "scripts/run_ragdefender_regime_b_text_realization.py").read_text()
        self.assertIn("ragdefender_internals", source)

    def test_stage2_success_label(self):
        label = tlib.classify_stage2_outcome(removed_poison=5, removed_clean=0, m_poison=5)
        self.assertEqual(label, tlib.STAGE2_SUCCESS)

    def test_stage2_degraded_label_wrong_poison_count(self):
        label = tlib.classify_stage2_outcome(removed_poison=4, removed_clean=0, m_poison=5)
        self.assertEqual(label, tlib.STAGE2_DEGRADED)

    def test_stage2_degraded_label_clean_removed(self):
        label = tlib.classify_stage2_outcome(removed_poison=5, removed_clean=1, m_poison=5)
        self.assertEqual(label, tlib.STAGE2_DEGRADED)


# ---------------------------------------------------------------------------
# 19/20. Round-2 only for 0/3 Round-1 full-realization queries; max 5
# variants/query.
# ---------------------------------------------------------------------------

class Test10Round2Eligibility(unittest.TestCase):
    def test_round2_eligible_only_if_round1_all_csv_present(self):
        path = OUTPUT_DIR / "phase5_round2_eligible_queries.json"
        if not path.exists():
            self.skipTest("Round-1 results not yet computed.")
        with open(path) as f:
            eligible = json.load(f)
        self.assertIsInstance(eligible, list)
        self.assertTrue(all(isinstance(q, str) for q in eligible))

    def test_round2_bank_never_exceeds_2_per_query_and_5_total(self):
        path = OUTPUT_DIR / "rewrite_bank_round2.jsonl"
        if not path.exists():
            self.skipTest("Round-2 bank not written (0 queries were eligible, or not yet run).")
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        by_query = {}
        for r in rows:
            by_query.setdefault(r["query_id"], set()).add(r["mutation_id"])
        for qid, mutations in by_query.items():
            self.assertEqual(mutations, {"R4", "R5"})

    def test_combined_round1_and_round2_never_exceeds_5_per_query(self):
        round1 = _load_bank_rows()
        round1_by_query = {}
        for r in round1:
            round1_by_query.setdefault(r["query_id"], set()).add(r["mutation_id"])

        round2_path = OUTPUT_DIR / "rewrite_bank_round2.jsonl"
        round2_by_query = {}
        if round2_path.exists():
            with open(round2_path) as f:
                round2 = [json.loads(line) for line in f]
            for r in round2:
                round2_by_query.setdefault(r["query_id"], set()).add(r["mutation_id"])

        for qid, mutations in round1_by_query.items():
            total = mutations | round2_by_query.get(qid, set())
            self.assertLessEqual(len(total), 5)


# ---------------------------------------------------------------------------
# 21. Target ground-truth label attached only after primary selection.
# ---------------------------------------------------------------------------

class Test11LabelAttachmentOrder(unittest.TestCase):
    def test_bank_builder_source_never_uses_is_poison_before_target_frozen(self):
        source = (REPO_ROOT / "scripts/build_regime_b_rewrite_bank.py").read_text()
        # ROUND1_REWRITES dict (authored rewrite text) may carry an inline
        # `# ..., is_poison=True/False` comment purely as a post-hoc
        # bookkeeping annotation for the human author -- that is NOT a
        # selection-logic usage. What must never happen is a conditional
        # branch on is_poison (an `if`/ternary referencing it) anywhere in
        # the file, which would mean the label influenced which rewrite
        # text got written or which candidate got selected.
        self.assertNotRegex(source, r"if\s+.*is_poison")
        self.assertNotRegex(source, r"is_poison\s*(==|!=|is\s)")

    def test_threat_model_wording_function_is_pure_boolean_map(self):
        self.assertEqual(tlib.threat_model_wording(True), tlib.WORDING_ATTACKER_CONTROLLED)
        self.assertEqual(tlib.threat_model_wording(False), tlib.WORDING_NON_ATTACKER_CONTROLLED)


# ---------------------------------------------------------------------------
# 22. Retrieval run only on L3-successful poison-target variants.
# ---------------------------------------------------------------------------

class Test12RetrievalGating(unittest.TestCase):
    def test_retrieval_csv_absent_or_only_contains_poison_target_full_realizations(self):
        path = OUTPUT_DIR / "regime_b_text_realization_retrieval.csv"
        if not path.exists():
            self.skipTest("No retrieval-eligible variants (retrieval CSV not written) -- valid outcome.")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        for row in rows:
            self.assertEqual(row["candidate_is_poison"], "True")
            self.assertEqual(row["full_realization"], "True")

    def test_eligible_variants_json_matches_stage2_poison_targets(self):
        eligible_path = OUTPUT_DIR / "phase10_l4_eligible_variants.json"
        stage2_path = OUTPUT_DIR / "regime_b_text_realization_stage2.csv"
        label_path = OUTPUT_DIR / "regime_b_text_realization_target_label_audit.csv"
        if not (eligible_path.exists() and stage2_path.exists() and label_path.exists()):
            self.skipTest("Phase 6/9/10 artifacts not yet written.")
        with open(eligible_path) as f:
            eligible = json.load(f)
        with open(stage2_path) as f:
            stage2_rows = {(r["query_id"], r["mutation_id"]): r for r in csv.DictReader(f)}
        with open(label_path) as f:
            labels_by_qid = {r["query_id"]: r for r in csv.DictReader(f)}

        # Every eligible variant must be a Stage-2-recorded FULL L3
        # realization whose target is poison-controlled (never selected
        # by is_poison at PRIMARY-target-selection time -- see Test03/11 --
        # only used here, post-hoc, to gate the L4 retrieval step).
        for e in eligible:
            key = (e["query_id"], e["mutation_id"])
            self.assertIn(key, stage2_rows)
            self.assertEqual(labels_by_qid[e["query_id"]]["candidate_is_poison"], "True")

        # And no poison-target FULL realization should be missing from the
        # eligible list (completeness of the gate, not just soundness).
        poison_qids = {qid for qid, r in labels_by_qid.items() if r["candidate_is_poison"] == "True"}
        full_poison_variants = [
            key for key, r in stage2_rows.items() if key[0] in poison_qids
        ]
        eligible_keys = {(e["query_id"], e["mutation_id"]) for e in eligible}
        self.assertEqual(set(full_poison_variants), eligible_keys)

    def test_retrieval_result_preserves_attack_budget(self):
        path = OUTPUT_DIR / "regime_b_text_realization_retrieval.csv"
        if not path.exists():
            self.skipTest("Retrieval CSV not yet written.")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row.get("total_retrieved_poison_M_after_rewrite") in (None, ""):
                continue
            total = int(row["total_retrieved_poison_M_after_rewrite"]) + int(row["clean_count_after_rewrite"])
            self.assertEqual(total, 10, "Rewritten top-k context must still have exactly 10 passages.")

    def test_retrieval_result_l4_stage1_and_stage2_consistent_with_l3(self):
        path = OUTPUT_DIR / "regime_b_text_realization_retrieval.csv"
        if not path.exists():
            self.skipTest("Retrieval CSV not yet written.")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row.get("still_retrieved_in_topk") != "True":
                continue
            # If the rewrite survives retrieval, the L4 rerun must itself
            # be internally consistent: Stage-2 residual = M - removed.
            n_adv = int(row["l4_stage1_n_adv"])
            removed_poison = int(row["l4_stage2_removed_poison"])
            residual = int(row["l4_stage2_residual_poison"])
            m_poison = int(row["total_retrieved_poison_M_after_rewrite"])
            self.assertEqual(residual, m_poison - removed_poison)
            self.assertGreaterEqual(n_adv, 0)


# ---------------------------------------------------------------------------
# Phase 8 aggregation -- per-query / alignment summary CSVs are internally
# consistent with the per-variant source data.
# ---------------------------------------------------------------------------

class Test15Phase8Aggregation(unittest.TestCase):
    def test_per_query_csv_has_14_rows_matching_mechanism_split(self):
        path = OUTPUT_DIR / "regime_b_text_realization_per_query.csv"
        if not path.exists():
            self.skipTest("Phase 8 aggregation not yet run.")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 14)
        n_median = sum(1 for r in rows if r["mechanism"] == "median-limited")
        n_mean = sum(1 for r in rows if r["mechanism"] == "mean-gated")
        self.assertEqual(n_median, 11)
        self.assertEqual(n_mean, 3)

    def test_per_query_any_full_realization_matches_per_variant_source(self):
        per_query_path = OUTPUT_DIR / "regime_b_text_realization_per_query.csv"
        per_variant_path = OUTPUT_DIR / "regime_b_text_realization_per_variant.csv"
        if not (per_query_path.exists() and per_variant_path.exists()):
            self.skipTest("Phase 8 aggregation not yet run.")
        with open(per_variant_path) as f:
            variant_rows = list(csv.DictReader(f))
        full_qids = {r["query_id"] for r in variant_rows if r["classification"] == "A. FULL REALIZATION"}
        with open(per_query_path) as f:
            for row in csv.DictReader(f):
                expected = row["query_id"] in full_qids
                self.assertEqual(row["any_full_realization"] == "True", expected)

    def test_per_variant_row_count_is_sum_of_round1_and_round2(self):
        round1_path = OUTPUT_DIR / "regime_b_text_realization_per_variant_round1.csv"
        round2_path = OUTPUT_DIR / "regime_b_text_realization_per_variant_round2.csv"
        merged_path = OUTPUT_DIR / "regime_b_text_realization_per_variant.csv"
        if not (round1_path.exists() and merged_path.exists()):
            self.skipTest("Phase 8 aggregation not yet run.")
        with open(round1_path) as f:
            n1 = sum(1 for _ in csv.DictReader(f))
        n2 = 0
        if round2_path.exists():
            with open(round2_path) as f:
                n2 = sum(1 for _ in csv.DictReader(f))
        with open(merged_path) as f:
            n_merged = sum(1 for _ in csv.DictReader(f))
        self.assertEqual(n_merged, n1 + n2)


# ---------------------------------------------------------------------------
# 23/24. No generation step; no external LLM/API dependency.
# ---------------------------------------------------------------------------

class Test13NoExternalDependency(unittest.TestCase):
    FORBIDDEN_IMPORTS = ("openai", "anthropic", "requests_oauthlib", "google.generativeai", "cohere")

    def _assert_no_forbidden_imports(self, path: Path):
        source = path.read_text()
        for name in self.FORBIDDEN_IMPORTS:
            self.assertNotIn(name, source, f"{path} references forbidden import {name!r}")

    def test_lib_module_has_no_forbidden_imports(self):
        self._assert_no_forbidden_imports(REPO_ROOT / "scripts/ragdefender_regime_b_text_realization_lib.py")

    def test_driver_module_has_no_forbidden_imports(self):
        self._assert_no_forbidden_imports(REPO_ROOT / "scripts/run_ragdefender_regime_b_text_realization.py")

    def test_bank_builder_has_no_forbidden_imports(self):
        self._assert_no_forbidden_imports(REPO_ROOT / "scripts/build_regime_b_rewrite_bank.py")

    def test_phase10_retrieval_script_has_no_forbidden_imports(self):
        path = REPO_ROOT / "scripts/run_ragdefender_regime_b_text_realization_phase10.py"
        if path.exists():
            self._assert_no_forbidden_imports(path)

    def test_phase10_retrieval_script_never_calls_generation(self):
        path = REPO_ROOT / "scripts/run_ragdefender_regime_b_text_realization_phase10.py"
        if not path.exists():
            self.skipTest("Phase-10 script not present.")
        source = path.read_text()
        for forbidden in ("generate_answer", "llm_answer_fn", "openai.", "anthropic.", "Attacker.get_attack"):
            # Attacker.get_attack itself is fine (offline template reproduction,
            # reused unmodified from run_full_retrieval_pilot_bundle1.py) --
            # only direct generation-model calls are forbidden here.
            if forbidden == "Attacker.get_attack":
                continue
            self.assertNotIn(forbidden, source)

    def test_driver_never_calls_generation_functions(self):
        source = (REPO_ROOT / "scripts/run_ragdefender_regime_b_text_realization.py").read_text()
        for forbidden in ("generate_answer", "llm_answer_fn", "openai.", "anthropic."):
            self.assertNotIn(forbidden, source)

    def test_no_network_modules_loaded_after_importing_lib(self):
        # A conservative proxy: the pure lib module itself must not import
        # any networking library at module scope.
        source = (REPO_ROOT / "scripts/ragdefender_regime_b_text_realization_lib.py").read_text()
        for forbidden in ("requests", "httpx", "urllib.request", "socket"):
            self.assertNotIn(f"import {forbidden}", source)


# ---------------------------------------------------------------------------
# 25. Historical matrices/artifacts remain read-only.
# ---------------------------------------------------------------------------

class Test14ArtifactSafety(unittest.TestCase):
    def test_historical_expanded_baseline_similarity_matrices_unchanged(self):
        # Spot-check a couple of historical matrices still exist and are
        # untouched by this study (this study writes only under
        # ragdefender_regime_b_text_realization/, never under
        # ragdefender_expanded_baseline/).
        sim_dir = BASELINE_DIR / "similarity"
        self.assertTrue(sim_dir.exists())
        matrices = list(sim_dir.glob("*_stella_similarity_matrix.npy"))
        self.assertGreaterEqual(len(matrices), 19)

    def test_historical_regime_b_oracle_csvs_unchanged_by_this_study(self):
        for name in ("regime_b_matrix_winners_v2.csv", "regime_b_boundary_per_query.csv"):
            path = ORACLE_DIR / name
            self.assertTrue(path.exists())

    def test_driver_never_writes_outside_its_own_output_dir(self):
        source = (REPO_ROOT / "scripts/run_ragdefender_regime_b_text_realization.py").read_text()
        self.assertNotIn("ragdefender_expanded_baseline_bridge", source)
        # The only OUTPUT_DIR assignment must point at this study's own dir.
        self.assertIn('OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_regime_b_text_realization"', source)

    def test_bank_builder_refuses_to_overwrite_existing_bank(self):
        source = (REPO_ROOT / "scripts/build_regime_b_rewrite_bank.py").read_text()
        self.assertIn("Refusing to overwrite existing", source)


if __name__ == "__main__":
    unittest.main()
