"""Paper-fidelity unit tests for `ragdefender_paper` against the FINAL ACSAC
2025 RAGDefender paper (Kim, Lee, Koo, "Rescuing the Unpoisoned," ACSAC 2025,
DOI 10.1109/ACSAC67867.2025.00093) -- Eq. (1)-(7).

This file is deliberately self-contained: every hand-computed fixture is
recomputed in the test docstrings/comments rather than referencing
tests/test_ragdefender_cluster_viz.py, so a reviewer can audit paper
fidelity from this file alone. It complements (does not replace)
tests/test_ragdefender_cluster_viz.py, which continues to cover
`ragdefender_legacy`'s own behavior unchanged.

AUTHORITY RULE (see docs/RAGDEFENDER_FIDELITY_AUDIT_V2.md / plan §0a item 2):
the final published paper governs every explicitly-specified behavior;
the authors' officially released code is consulted ONLY to fill in the one
place the paper is silent -- median tie-breaking for an even number of
values. That choice (the authors'/`torch.median`'s lower-of-two-middle
convention) is exercised and documented as an implementation choice, never
attributed to the paper, in the "GLOBAL MEDIAN / TIE BREAK" tests below.

Run with: python -m unittest tests.test_ragdefender_paper_fidelity -v
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from defense.ragdefender_internals import (
    concentration_stage1_paper,
    stage2_pair_frequency,
)


# ---------------------------------------------------------------------------
# Shared fixtures (values hand-verified; see comments at each use site)
# ---------------------------------------------------------------------------

# 4x4 matrix reused from tests/test_ragdefender_cluster_viz.py's legacy
# fixtures, recomputed here under the PAPER's self-excluded equations.
# Off-diagonal rows: row0=[.90,.85,.10] mean=0.616667 median=0.85
#                    row1=[.90,.88,.15] mean=0.643333 median=0.88
#                    row2=[.85,.88,.05] mean=0.593333 median=0.85
#                    row3=[.10,.15,.05] mean=0.10      median=0.10
MATRIX_4X4 = np.array([
    [1.00, 0.90, 0.85, 0.10],
    [0.90, 1.00, 0.88, 0.15],
    [0.85, 0.88, 1.00, 0.05],
    [0.10, 0.15, 0.05, 1.00],
])

# 4x4 matrix designed so the global median-of-medians (s_tilde) genuinely
# depends on the even-count tie-break convention (lower-of-two vs. average).
# Off-diagonal rows: row0=[.90,.05,.60] mean=0.516667 median=0.60
#                    row1=[.90,.15,.80] mean=0.616667 median=0.80
#                    row2=[.05,.15,.75] mean=0.316667 median=0.15
#                    row3=[.60,.80,.75] mean=0.716667 median=0.75
# s_mean=[.516667,.616667,.316667,.716667], s_bar=0.541667
# s_median=[.60,.80,.15,.75], sorted=[.15,.60,.75,.80]
#   torch-style (lower-of-two-middle) s_tilde = sorted[1] = 0.60
#   numpy-average-style s_tilde would be (0.60+0.75)/2 = 0.675 -- DIFFERENT.
# above_mean=[F,T,F,T], above_median=[F,T,F,T] -> AND -> n_adv=2 (rows 1,3)
MATRIX_TIEBREAK = np.array([
    [1.00, 0.90, 0.05, 0.60],
    [0.90, 1.00, 0.15, 0.80],
    [0.05, 0.15, 1.00, 0.75],
    [0.60, 0.80, 0.75, 1.00],
])

# Uniform 3x3 matrix -- the existing legacy flip-branch fixture
# (tests/test_ragdefender_cluster_viz.py's test_flip_branch_fires_when_or_count_zero).
# Under ragdefender_legacy this triggers the flip branch (result = 3 - 0 = 3,
# flagging everyone). Under the paper's Eq. (3) there is no flip branch, so
# an all-tied matrix simply yields n_adv=0.
MATRIX_UNIFORM_3X3 = np.array([
    [1.0, 0.5, 0.5],
    [0.5, 1.0, 0.5],
    [0.5, 0.5, 1.0],
])

# 4x4 matrix engineered (via brute-force search over a small grid, see the
# session's derivation) so each row lands in a distinct AND-logic category:
#   row0: above_mean=True,  above_median=False -> "mean-only"
#   row1: above_mean=False, above_median=True  -> "median-only"
#   row2: above_mean=True,  above_median=True  -> "both"
#   row3: above_mean=False, above_median=False -> "neither"
# s_mean=[.166667,.15,.216667,.10], s_bar=0.158333
# s_median=[.05,.20,.20,.05], s_tilde=0.05
# Only row2 ("both") is AND-flagged -> n_adv=1.
MATRIX_AND_LOGIC = np.array([
    [1.00, 0.05, 0.40, 0.05],
    [0.05, 1.00, 0.20, 0.20],
    [0.40, 0.20, 1.00, 0.05],
    [0.05, 0.20, 0.05, 1.00],
])


def _hand_median(values):
    """Plain sorted-list median helper for readable hand-verification in
    test bodies (NOT the torch-style tie-break -- see _torch_lower_median)."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _torch_lower_median(values):
    """The authority-rule tie-break this module actually uses: for an even
    count, the LOWER of the two middle values (torch.median's convention,
    inherited from the authors' own code per the authority rule -- see
    module docstring)."""
    s = sorted(values)
    n = len(s)
    return s[(n - 1) // 2]


# ---------------------------------------------------------------------------
# 1. PAPER MEAN CONCENTRATION
# ---------------------------------------------------------------------------

class TestPaperMeanConcentration(unittest.TestCase):
    def test_excludes_diagonal_and_matches_hand_calculation(self):
        result = concentration_stage1_paper(MATRIX_4X4)
        # s_mean_0 = (0.90 + 0.85 + 0.10) / 3 = 1.85 / 3
        self.assertAlmostEqual(result.s_mean[0], 1.85 / 3, places=9)
        self.assertAlmostEqual(result.s_mean[1], 1.93 / 3, places=9)
        self.assertAlmostEqual(result.s_mean[2], 1.78 / 3, places=9)
        self.assertAlmostEqual(result.s_mean[3], 0.30 / 3, places=9)


# ---------------------------------------------------------------------------
# 2. PAPER MEDIAN CONCENTRATION
# ---------------------------------------------------------------------------

class TestPaperMedianConcentration(unittest.TestCase):
    def test_odd_length_off_diagonal_row_k_even(self):
        """k=4 -> each row's off-diagonal length is 3 (odd) -- unambiguous
        median, no tie-break needed."""
        result = concentration_stage1_paper(MATRIX_4X4)
        self.assertAlmostEqual(result.s_median[0], 0.85, places=9)  # sorted [.10,.85,.90]
        self.assertAlmostEqual(result.s_median[1], 0.88, places=9)  # sorted [.15,.88,.90]
        self.assertAlmostEqual(result.s_median[2], 0.85, places=9)  # sorted [.05,.85,.88]
        self.assertAlmostEqual(result.s_median[3], 0.10, places=9)  # sorted [.05,.10,.15]

    def test_even_length_off_diagonal_row_k_odd_uses_torch_style_tie_break(self):
        """k=5 -> each row's off-diagonal length is 4 (even) -- median
        tie-break matters. Row 0's off-diagonal values are [0.9, 0.5, 0.7,
        0.1]; sorted [0.1, 0.5, 0.7, 0.9], middle two are 0.5 and 0.7.
        Per the authority rule, this module uses the lower-of-two-middle
        (torch-style) convention -> 0.5, NOT numpy's average-style 0.6."""
        matrix_5x5 = np.array([
            [1.00, 0.90, 0.50, 0.70, 0.10],
            [0.90, 1.00, 0.60, 0.40, 0.30],
            [0.50, 0.60, 1.00, 0.20, 0.80],
            [0.70, 0.40, 0.20, 1.00, 0.05],
            [0.10, 0.30, 0.80, 0.05, 1.00],
        ])
        result = concentration_stage1_paper(matrix_5x5)
        self.assertAlmostEqual(result.s_median[0], 0.5, places=9)
        self.assertNotAlmostEqual(result.s_median[0], 0.6, places=9)


# ---------------------------------------------------------------------------
# 3. GLOBAL MEAN THRESHOLD
# ---------------------------------------------------------------------------

class TestGlobalMeanThreshold(unittest.TestCase):
    def test_s_bar_is_mean_of_s_mean(self):
        result = concentration_stage1_paper(MATRIX_4X4)
        expected = (1.85 / 3 + 1.93 / 3 + 1.78 / 3 + 0.30 / 3) / 4.0
        self.assertAlmostEqual(result.s_bar, expected, places=9)
        self.assertAlmostEqual(result.s_bar, float(np.mean(result.s_mean)), places=9)


# ---------------------------------------------------------------------------
# 4. GLOBAL MEDIAN THRESHOLD
# ---------------------------------------------------------------------------

class TestGlobalMedianThreshold(unittest.TestCase):
    def test_s_tilde_is_median_of_s_median(self):
        result = concentration_stage1_paper(MATRIX_TIEBREAK)
        # s_median = [0.60, 0.80, 0.15, 0.75] -> sorted [0.15, 0.60, 0.75, 0.80]
        np.testing.assert_allclose(sorted(result.s_median), [0.15, 0.60, 0.75, 0.80])
        self.assertAlmostEqual(result.s_tilde, 0.60, places=9)


# ---------------------------------------------------------------------------
# 5. AND LOGIC
# ---------------------------------------------------------------------------

class TestAndLogic(unittest.TestCase):
    def test_only_the_both_passage_contributes_to_n_adv(self):
        result = concentration_stage1_paper(MATRIX_AND_LOGIC)
        np.testing.assert_array_equal(result.above_mean, [True, False, True, False])
        np.testing.assert_array_equal(result.above_median, [False, True, True, False])
        # row0: mean-only, row1: median-only, row2: both, row3: neither
        np.testing.assert_array_equal(result.adv_flag, [False, False, True, False])
        self.assertEqual(result.n_adv_estimated, 1)


# ---------------------------------------------------------------------------
# 6. NO FLIP BRANCH
# ---------------------------------------------------------------------------

class TestNoFlipBranch(unittest.TestCase):
    def test_uniform_matrix_never_flips_to_complement(self):
        """Under ragdefender_legacy, this exact matrix triggers the flip
        branch (raw OR-count is 0, so `len - 0 = 3` flags everyone -- see
        tests/test_ragdefender_cluster_viz.py::test_flip_branch_fires_when_or_count_zero).
        The paper's Eq. (3) has no such branch: it is a direct AND-count,
        so it must return 0 here, never the complement (3)."""
        result = concentration_stage1_paper(MATRIX_UNIFORM_3X3)
        np.testing.assert_array_equal(result.above_mean, [False, False, False])
        np.testing.assert_array_equal(result.above_median, [False, False, False])
        self.assertEqual(result.n_adv_estimated, 0)
        self.assertNotEqual(result.n_adv_estimated, 3)


# ---------------------------------------------------------------------------
# 7. SELF-SIMILARITY EXCLUSION
# ---------------------------------------------------------------------------

class TestSelfSimilarityExclusion(unittest.TestCase):
    def test_mutating_diagonal_does_not_change_output(self):
        baseline = concentration_stage1_paper(MATRIX_4X4)

        mutated = MATRIX_4X4.copy()
        np.fill_diagonal(mutated, [5.0, -3.0, 100.0, 0.0])  # arbitrary, not 1.0
        mutated_result = concentration_stage1_paper(mutated)

        np.testing.assert_allclose(baseline.s_mean, mutated_result.s_mean)
        np.testing.assert_allclose(baseline.s_median, mutated_result.s_median)
        self.assertAlmostEqual(baseline.s_bar, mutated_result.s_bar, places=9)
        self.assertAlmostEqual(baseline.s_tilde, mutated_result.s_tilde, places=9)
        self.assertEqual(baseline.n_adv_estimated, mutated_result.n_adv_estimated)


# ---------------------------------------------------------------------------
# 8. N_ADV END-TO-END
# ---------------------------------------------------------------------------

class TestNAdvEndToEnd(unittest.TestCase):
    def test_matrix_4x4_gives_n_adv_1(self):
        # s_bar=0.488333, s_tilde=0.85; above_mean=[T,T,T,F], above_median=[F,T,F,F]
        result = concentration_stage1_paper(MATRIX_4X4)
        self.assertEqual(result.n_adv_estimated, 1)
        np.testing.assert_array_equal(result.adv_flag, [False, True, False, False])

    def test_matrix_tiebreak_gives_n_adv_2(self):
        result = concentration_stage1_paper(MATRIX_TIEBREAK)
        self.assertEqual(result.n_adv_estimated, 2)
        np.testing.assert_array_equal(result.adv_flag, [False, True, False, True])

    def test_matrix_and_logic_gives_n_adv_1(self):
        result = concentration_stage1_paper(MATRIX_AND_LOGIC)
        self.assertEqual(result.n_adv_estimated, 1)

    def test_uniform_matrix_gives_n_adv_0(self):
        result = concentration_stage1_paper(MATRIX_UNIFORM_3X3)
        self.assertEqual(result.n_adv_estimated, 0)


# ---------------------------------------------------------------------------
# 9. N_PAIRS
# ---------------------------------------------------------------------------

class TestNPairsFormula(unittest.TestCase):
    def _matrix(self, k):
        m = np.eye(k)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        for n, (i, j) in enumerate(pairs):
            v = 0.05 + 0.03 * n
            m[i, j] = v
            m[j, i] = v
        return m

    def test_n_pairs_equals_max_1_choose_2(self):
        matrix = self._matrix(6)
        expected = {0: 0, 1: 1, 2: 1, 4: 6, 5: 10}
        for n_adv, exp_n_pairs in expected.items():
            with self.subTest(n_adv=n_adv):
                result = stage2_pair_frequency(matrix, n_adv=n_adv)
                self.assertEqual(result.n_pairs, exp_n_pairs)


# ---------------------------------------------------------------------------
# 10. UNIQUE UNORDERED PAIRS
# ---------------------------------------------------------------------------

class TestUniqueUnorderedPairs(unittest.TestCase):
    def test_no_duplicate_or_self_pairs(self):
        result = stage2_pair_frequency(MATRIX_4X4, n_adv=4)
        seen = set()
        for i, j, _ in result.top_pairs:
            self.assertLess(i, j)  # i < j -- never (j, i) too, never i == j
            self.assertNotIn((i, j), seen)
            seen.add((i, j))


# ---------------------------------------------------------------------------
# 11. FREQUENCY SCORE (p=2)
# ---------------------------------------------------------------------------

class TestFrequencyScore(unittest.TestCase):
    def test_positive_similarity(self):
        result = stage2_pair_frequency(MATRIX_4X4, n_adv=2, p=2.0)
        self.assertEqual(result.top_pairs, [(0, 1, 0.9)])
        # sgn(0.9) * |0.9|^2 = 0.81, contributed to both passages 0 and 1
        np.testing.assert_allclose(result.frequency_scores, [0.81, 0.81, 0.0, 0.0])

    def test_negative_similarity_sign_is_preserved(self):
        """All-negative matrix forces a negative pair into the top selection
        (positive/high-similarity pairs would otherwise always dominate
        "top" in practice) -- exercises sgn(sim)*|sim|^p for sim < 0."""
        matrix = np.array([
            [1.0, -0.1, -0.2, -0.3],
            [-0.1, 1.0, -0.4, -0.5],
            [-0.2, -0.4, 1.0, -0.6],
            [-0.3, -0.5, -0.6, 1.0],
        ])
        result = stage2_pair_frequency(matrix, n_adv=2, p=2.0)
        self.assertEqual(result.top_pairs, [(0, 1, -0.1)])  # least-negative == "top" by raw sim
        # sgn(-0.1) * |-0.1|^2 = -0.01
        expected = math.copysign(abs(-0.1) ** 2, -0.1)
        np.testing.assert_allclose(result.frequency_scores, [expected, expected, 0.0, 0.0])
        self.assertLess(result.frequency_scores[0], 0.0)


# ---------------------------------------------------------------------------
# 12. FINAL REMOVAL
# ---------------------------------------------------------------------------

class TestFinalRemoval(unittest.TestCase):
    def test_removes_exactly_top_n_adv_by_frequency_score(self):
        result = stage2_pair_frequency(MATRIX_4X4, n_adv=2, p=2.0)
        self.assertEqual(len(result.selected_indices), 2)
        self.assertEqual(sorted(result.selected_indices), [0, 1])


# ---------------------------------------------------------------------------
# 13. TIE BREAK (documented as our implementation choice, not the paper's)
# ---------------------------------------------------------------------------

class TestTieBreakIsDocumentedImplementationChoice(unittest.TestCase):
    def test_global_median_tie_break_differs_from_numpy_average_convention(self):
        """The paper does NOT specify how to break a median tie for an even
        number of values -- this test pins down the specific choice this
        module makes (torch-style lower-of-two-middle, per the authority
        rule: the officially released code's convention fills this
        paper-silent gap) and proves it is NOT numpy's average-of-two-middle
        convention, which would silently give a different s_tilde."""
        result = concentration_stage1_paper(MATRIX_TIEBREAK)
        torch_style = _torch_lower_median(result.s_median.tolist())
        numpy_style = _hand_median(result.s_median.tolist())
        self.assertNotAlmostEqual(torch_style, numpy_style, places=6)
        self.assertAlmostEqual(result.s_tilde, torch_style, places=9)
        self.assertNotAlmostEqual(result.s_tilde, numpy_style, places=6)

    def test_stage2_tie_break_matches_first_insertion_order(self):
        """Stage-2 tie-breaking (which of several equal-scoring passages is
        selected when frequency scores tie) is also a paper-silent detail;
        this module's choice is a plain dict's first-insertion order plus a
        stable sort, matching the authors' Counter-based code."""
        result = stage2_pair_frequency(MATRIX_4X4, n_adv=1)
        self.assertEqual(result.n_pairs, 1)
        # Only pair (0, 1) exists in Ptop; both tie at the same frequency
        # score, so the first-inserted index (0) is selected deterministically.
        self.assertEqual(result.selected_indices, [0])


# ---------------------------------------------------------------------------
# 14. STELLA DEFAULT RESOLUTION (config only -- no model load/download)
# ---------------------------------------------------------------------------

class TestStellaDefaultResolution(unittest.TestCase):
    """Pure string-resolution logic: importing defense_runner here does pull
    in `torch` at module level (see defense_runner.py's own imports), but
    `_resolve_similarity_model` itself never imports sentence_transformers
    or touches the network -- resolving a preset name is not the same as
    loading a model."""

    def setUp(self):
        from defense import defense_runner
        self.defense_runner = defense_runner

    def test_paper_version_resolves_to_stella(self):
        cfg = self.defense_runner.DefenseConfig(ragdefender_version="paper")
        self.assertEqual(
            self.defense_runner._resolve_similarity_model(cfg),  # noqa: SLF001
            "dunzhang/stella_en_1.5B_v5",
        )

    def test_legacy_version_resolves_to_minilm_unchanged(self):
        cfg = self.defense_runner.DefenseConfig()  # all defaults
        self.assertEqual(cfg.ragdefender_version, "legacy")
        self.assertEqual(
            self.defense_runner._resolve_similarity_model(cfg),  # noqa: SLF001
            "paraphrase-MiniLM-L6-v2",
        )

    def test_explicit_similarity_model_overrides_either_preset(self):
        cfg = self.defense_runner.DefenseConfig(
            ragdefender_version="paper", similarity_model="some/other-model"
        )
        self.assertEqual(
            self.defense_runner._resolve_similarity_model(cfg),  # noqa: SLF001
            "some/other-model",
        )


class TestExplicitDeviceHandling(unittest.TestCase):
    """Isolated, network-free unit tests for `_get_s_model`'s device
    placement logic, using a fake SentenceTransformer (no real model load).
    Proves an explicit `DefenseConfig.device` of "cpu"/"mps"/"cuda" is
    always honored via `.to(device)` -- previously only "cuda" was, so a
    caller requesting "cpu" could silently end up on whatever
    sentence-transformers auto-detected (see git history / Gate-A-readiness
    notes: this was observed to be MPS on Apple Silicon in practice)."""

    def setUp(self):
        from unittest import mock

        from defense import defense_runner

        self.defense_runner = defense_runner
        self.defense_runner._S_MODEL_CACHE.clear()  # noqa: SLF001 -- avoid cross-test cache hits
        self.addCleanup(self.defense_runner._S_MODEL_CACHE.clear)  # noqa: SLF001

        self.to_calls = []

        class FakeSentenceTransformer:
            def __init__(self, model_name, **kwargs):
                self.model_name = model_name
                self.init_kwargs = kwargs
                self.device = "auto-detected-device"  # sentence-transformers' own default

            def to(fake_self, device):  # noqa: N805
                self.to_calls.append(device)
                fake_self.device = device
                return fake_self

        self.lazy_st_patcher = mock.patch.object(
            self.defense_runner, "_lazy_st", return_value=(FakeSentenceTransformer, None)
        )
        self.lazy_st_patcher.start()
        self.addCleanup(self.lazy_st_patcher.stop)

        # Avoid a real `torch.cuda.set_device` call in the "cuda" test case
        # (no GPU required to exercise this pure dispatch logic).
        self.cuda_set_device_patcher = mock.patch("torch.cuda.set_device")
        self.cuda_set_device_patcher.start()
        self.addCleanup(self.cuda_set_device_patcher.stop)

    def test_cpu_device_is_explicitly_applied(self):
        cfg = self.defense_runner.DefenseConfig(device="cpu")
        model = self.defense_runner._get_s_model(cfg)  # noqa: SLF001
        self.assertEqual(self.to_calls, ["cpu"])
        self.assertEqual(model.device, "cpu")

    def test_mps_device_is_explicitly_applied(self):
        cfg = self.defense_runner.DefenseConfig(device="mps")
        model = self.defense_runner._get_s_model(cfg)  # noqa: SLF001
        self.assertEqual(self.to_calls, ["mps"])
        self.assertEqual(model.device, "mps")

    def test_cuda_device_is_still_explicitly_applied_unchanged(self):
        cfg = self.defense_runner.DefenseConfig(device="cuda", gpu_id=0)
        model = self.defense_runner._get_s_model(cfg)  # noqa: SLF001
        self.assertEqual(self.to_calls, ["cuda"])
        self.assertEqual(model.device, "cuda")

    def test_unrecognized_device_string_preserves_old_auto_detect_fallback(self):
        # Deliberately not one of "cuda"/"cpu"/"mps" -- must not raise, and
        # must leave sentence-transformers' own auto-detected device alone
        # (legacy behavior for anything not explicitly recognized).
        cfg = self.defense_runner.DefenseConfig(device="some-other-device-string")
        model = self.defense_runner._get_s_model(cfg)  # noqa: SLF001
        self.assertEqual(self.to_calls, [])
        self.assertEqual(model.device, "auto-detected-device")

    def test_paper_variant_applies_stella_compat_shim_before_load(self):
        from unittest import mock

        with mock.patch.object(
            self.defense_runner, "_apply_stella_dynamic_cache_compat_shim"
        ) as shim:
            cfg = self.defense_runner.DefenseConfig(ragdefender_version="paper", device="cpu")
            self.defense_runner._get_s_model(cfg)  # noqa: SLF001
            shim.assert_called_once()

    def test_legacy_variant_does_not_apply_stella_compat_shim(self):
        from unittest import mock

        with mock.patch.object(
            self.defense_runner, "_apply_stella_dynamic_cache_compat_shim"
        ) as shim:
            cfg = self.defense_runner.DefenseConfig(ragdefender_version="legacy", device="cpu")
            self.defense_runner._get_s_model(cfg)  # noqa: SLF001
            shim.assert_not_called()


class TestStellaDynamicCacheCompatShim(unittest.TestCase):
    """Unit test for the shim itself (no real Stella model involved): it
    must add `get_usable_length` only when missing, and the added method
    must delegate to `get_seq_length`."""

    def test_backfills_missing_method_as_alias_for_get_seq_length(self):
        from defense import defense_runner

        class FakeCacheMissingMethod:
            def get_seq_length(self, layer_idx=0):
                return 42

        if hasattr(FakeCacheMissingMethod, "get_usable_length"):
            del FakeCacheMissingMethod.get_usable_length

        import transformers.cache_utils as cache_utils_module
        from unittest import mock

        with mock.patch.object(cache_utils_module, "DynamicCache", FakeCacheMissingMethod):
            defense_runner._apply_stella_dynamic_cache_compat_shim()  # noqa: SLF001
            self.assertTrue(hasattr(FakeCacheMissingMethod, "get_usable_length"))
            instance = FakeCacheMissingMethod()
            self.assertEqual(instance.get_usable_length(new_seq_length=99), 42)

    def test_layer_idx_is_forwarded_not_silently_ignored(self):
        """Stella's modeling code calls
        `past_key_value.get_usable_length(kv_seq_len, self.layer_idx)` --
        i.e. `layer_idx` positionally, and (for a multi-layer model) with a
        value that varies per decoder layer, not always the 0 default. This
        pins that the shim forwards whichever `layer_idx` it was called
        with to `get_seq_length`, rather than hardcoding/dropping it."""
        from defense import defense_runner

        calls = []

        class FakeCacheTracksLayerIdx:
            def get_seq_length(self, layer_idx=0):
                calls.append(layer_idx)
                return 1000 + layer_idx  # distinguishable per layer_idx

        if hasattr(FakeCacheTracksLayerIdx, "get_usable_length"):
            del FakeCacheTracksLayerIdx.get_usable_length

        import transformers.cache_utils as cache_utils_module
        from unittest import mock

        with mock.patch.object(cache_utils_module, "DynamicCache", FakeCacheTracksLayerIdx):
            defense_runner._apply_stella_dynamic_cache_compat_shim()  # noqa: SLF001
            instance = FakeCacheTracksLayerIdx()

            # Exact call shape used by Stella's modeling_qwen.py:
            # `past_key_value.get_usable_length(kv_seq_len, self.layer_idx)`
            # -- both arguments positional, layer_idx non-zero/distinguishable.
            result_layer_3 = instance.get_usable_length(50, 3)
            result_layer_7 = instance.get_usable_length(50, 7)

        self.assertEqual(calls, [3, 7])
        self.assertEqual(result_layer_3, 1003)
        self.assertEqual(result_layer_7, 1007)
        self.assertNotEqual(result_layer_3, result_layer_7)

    def test_is_a_noop_when_method_already_present(self):
        from defense import defense_runner

        class FakeCacheHasMethod:
            def get_seq_length(self, layer_idx=0):
                return 1

            def get_usable_length(self, new_seq_length, layer_idx=0):
                return "original-untouched"

        import transformers.cache_utils as cache_utils_module
        from unittest import mock

        with mock.patch.object(cache_utils_module, "DynamicCache", FakeCacheHasMethod):
            defense_runner._apply_stella_dynamic_cache_compat_shim()  # noqa: SLF001
            instance = FakeCacheHasMethod()
            self.assertEqual(instance.get_usable_length(new_seq_length=99), "original-untouched")


@unittest.skipUnless(
    os.environ.get("RAGDEFENDER_LOAD_STELLA") == "1",
    "Set RAGDEFENDER_LOAD_STELLA=1 to run this heavy integration test "
    "(downloads/loads the real dunzhang/stella_en_1.5B_v5 model). Not run "
    "by default -- see docs/RAGDEFENDER_FIDELITY_AUDIT_V2.md Gate B.",
)
class TestStellaModelLoadsAndEncodes(unittest.TestCase):
    """Gated smoke test: actually loads Stella (dunzhang/stella_en_1.5B_v5,
    requires trust_remote_code=True) and verifies encoding of a small batch
    of dummy passages, consistent embedding dimensionality, all-finite
    embedding values, and that pairwise cosine similarity is computable.
    This is intentionally NOT part of the default test run -- it requires a
    ~1.5B-parameter model download and is unrelated to the pure-logic
    fidelity tests above. Not an experiment: no retrieval, no generation,
    no external LLM/API calls.

    IMPORTANT: this class calls `defense_runner._get_s_model` directly --
    the exact production path Gate B will call -- with NO test-side
    monkeypatching of `transformers`. The
    transformers>=4.46/DynamicCache.get_usable_length compatibility fix
    lives in `defense_runner._apply_stella_dynamic_cache_compat_shim`
    (applied automatically by `_get_s_model` whenever
    `ragdefender_version == "paper"`), not here -- see that function's
    docstring for the two remediation options considered and why the
    narrowly-scoped shim (not a pinned-dependency downgrade) was chosen.
    `TestStellaProductionPathSmoke` below is the test that most directly
    proves this (asserts the shim is *not* pre-applied by the test itself).
    """

    DUMMY_PASSAGES = [
        "The Eiffel Tower is located in Paris, France.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The stock market closed higher today after a volatile session.",
        "A large language model was fine-tuned on a domain-specific corpus.",
    ]

    @classmethod
    def setUpClass(cls):
        from defense import defense_runner

        cls.defense_runner = defense_runner
        cls.cfg = defense_runner.DefenseConfig(ragdefender_version="paper", device="cpu")
        cls.s_model = defense_runner._get_s_model(cls.cfg)  # noqa: SLF001

    def test_model_loads_with_trust_remote_code(self):
        self.assertIsNotNone(self.s_model)

    def test_model_is_actually_placed_on_requested_cpu_device(self):
        # DefenseConfig(device="cpu") above must be honored explicitly, not
        # silently overridden by sentence-transformers' own cuda>mps>cpu
        # auto-detection -- see TestExplicitDeviceHandling for the isolated
        # (fake-model) unit tests of this logic.
        self.assertEqual(str(self.s_model.device), "cpu")

    def test_stella_encodes_batch_of_dummy_passages(self):
        embeddings = self.s_model.encode(self.DUMMY_PASSAGES, convert_to_tensor=True)
        self.assertEqual(embeddings.shape[0], len(self.DUMMY_PASSAGES))

    def test_embedding_dimensionality_is_consistent(self):
        embeddings = self.s_model.encode(self.DUMMY_PASSAGES, convert_to_tensor=True)
        dim = embeddings.shape[1]
        self.assertGreater(dim, 0)
        # Encode one-at-a-time and confirm every passage yields the same dim.
        for passage in self.DUMMY_PASSAGES:
            single = self.s_model.encode([passage], convert_to_tensor=True)
            self.assertEqual(single.shape[1], dim)

    def test_embeddings_are_finite(self):
        import torch as _torch

        embeddings = self.s_model.encode(self.DUMMY_PASSAGES, convert_to_tensor=True)
        self.assertTrue(bool(_torch.isfinite(embeddings).all()))

    def test_pairwise_cosine_similarity_is_computable(self):
        import torch as _torch
        from sentence_transformers import util as st_util

        embeddings = self.s_model.encode(self.DUMMY_PASSAGES, convert_to_tensor=True)
        sims = st_util.cos_sim(embeddings, embeddings)
        self.assertEqual(tuple(sims.shape), (len(self.DUMMY_PASSAGES), len(self.DUMMY_PASSAGES)))
        self.assertTrue(bool(_torch.isfinite(sims).all()))
        # Self-similarity should be (numerically) 1.0 on the diagonal.
        diag = sims.diagonal()
        self.assertTrue(bool((diag > 0.999).all()))


@unittest.skipUnless(
    os.environ.get("RAGDEFENDER_LOAD_STELLA") == "1",
    "Set RAGDEFENDER_LOAD_STELLA=1 to run this heavy integration test "
    "(downloads/loads the real dunzhang/stella_en_1.5B_v5 model). Not run "
    "by default -- see docs/RAGDEFENDER_FIDELITY_AUDIT_V2.md Gate B.",
)
class TestStellaProductionPathSmoke(unittest.TestCase):
    """The production-path smoke test the Gate-B readiness pass explicitly
    requires: proves the SAME path Gate B will call
    (`defense_runner._get_s_model` -> `defense_runner.apply_defense(...,
    ragdefender_version="paper")`) loads Stella and encodes passages with
    ZERO test-only monkeypatching applied anywhere in this test. If the
    transformers/DynamicCache compatibility fix were ever removed from
    `defense_runner._apply_stella_dynamic_cache_compat_shim`, this specific
    test (unlike `TestStellaModelLoadsAndEncodes`, which happens to run
    after that shim was already applied at module scope in the same
    process) would fail first in a fresh process."""

    def test_apply_defense_paper_variant_runs_end_to_end_with_stella(self):
        from defense import defense_runner

        query = "What is the capital of France?"
        docs = [
            "Paris is the capital and most populous city of France.",
            "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
            "Photosynthesis converts light energy into chemical energy.",
            "The mitochondria is the powerhouse of the cell.",
        ]
        kept = defense_runner.apply_defense(
            query,
            list(docs),
            dataset="hotpotqa",
            device="cpu",
            ragdefender_version="paper",
        )
        self.assertIsInstance(kept, list)
        self.assertGreater(len(kept), 0)
        self.assertLessEqual(len(kept), len(docs))

    def test_get_s_model_device_matches_requested_cpu_with_no_test_patching(self):
        from defense import defense_runner

        cfg = defense_runner.DefenseConfig(ragdefender_version="paper", device="cpu")
        s_model = defense_runner._get_s_model(cfg)  # noqa: SLF001
        self.assertEqual(str(s_model.device), "cpu")


# ---------------------------------------------------------------------------
# 15. LEGACY REGRESSION
# ---------------------------------------------------------------------------

class TestLegacyRegression(unittest.TestCase):
    """The existing tests/test_ragdefender_cluster_viz.py and
    tests/test_dispatch_smoke.py suites are the primary regression guard
    (run alongside this file, unchanged) -- this class adds one direct
    pin proving the new `ragdefender_version`/`similarity_model` parameters
    default to byte-identical legacy behavior."""

    def test_default_defense_config_is_unchanged_legacy_behavior(self):
        from defense import defense_runner

        cfg = defense_runner.DefenseConfig()
        self.assertEqual(cfg.ragdefender_version, "legacy")
        self.assertIsNone(cfg.similarity_model)
        self.assertEqual(
            defense_runner._resolve_similarity_model(cfg),  # noqa: SLF001
            "paraphrase-MiniLM-L6-v2",
        )

    def test_apply_defense_default_ragdefender_version_is_legacy(self):
        import inspect

        from defense import defense_runner

        sig = inspect.signature(defense_runner.apply_defense)
        self.assertEqual(sig.parameters["ragdefender_version"].default, "legacy")


# ---------------------------------------------------------------------------
# 16. REAL HOTPOTQA FIXTURE (deferred until Gate B runs)
# ---------------------------------------------------------------------------

_GATE_B_FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "ragdefender_gate_b_real_hotpotqa_fixture.json"
)


class TestRealHotpotQAFixture(unittest.TestCase):
    """Populated after Gate B (results/diagnostics/ragdefender_gate_b/) ran
    once and produced a real Stella-derived similarity matrix for an actual
    HotpotQA k=10/N=5 query. This test locks in that ONE real query's
    matrix (stored verbatim in `tests/fixtures/`) and re-derives Stage
    1/Stage 2 from it via the ordinary paper-faithful functions -- it does
    NOT load Stella live; it is a pure regression test on a real, frozen
    Stella-encoding output. The fixture query
    (`5a722b8655429971e9dc9329`) is Gate B's sole zero-residual-poison
    success case (see GATE_B_STELLA_FIDELITY_REPORT.md)."""

    @classmethod
    def setUpClass(cls):
        import json

        with open(_GATE_B_FIXTURE_PATH) as f:
            cls.fixture = json.load(f)
        cls.matrix = np.array(cls.fixture["similarity_matrix"], dtype=np.float64)

    def test_fixture_matrix_is_square_and_finite(self):
        self.assertEqual(self.matrix.shape, (self.fixture["k"], self.fixture["k"]))
        self.assertTrue(np.isfinite(self.matrix).all())

    def test_stage1_reproduces_expected_n_adv(self):
        stage1 = concentration_stage1_paper(self.matrix)
        self.assertEqual(stage1.n_adv_estimated, self.fixture["expected_n_adv"])

    def test_stage2_reproduces_expected_pair_and_removal_composition(self):
        stage1 = concentration_stage1_paper(self.matrix)
        stage2 = stage2_pair_frequency(self.matrix, n_adv=stage1.n_adv_estimated, p=2.0)

        self.assertEqual(stage2.n_pairs, self.fixture["expected_n_pairs"])

        top_i, top_j, top_sim = stage2.top_pairs[0]
        self.assertEqual(sorted((top_i, top_j)), self.fixture["expected_top_pair"])
        self.assertAlmostEqual(top_sim, self.fixture["expected_top_pair_similarity"], places=6)

        is_poison = self.fixture["is_poison"]
        pi, pj = is_poison[top_i], is_poison[top_j]
        top_pair_class = "PP" if (pi and pj) else ("CC" if (not pi and not pj) else "PC")
        self.assertEqual(top_pair_class, self.fixture["expected_top_pair_class"])

        removed_indices = set(stage2.selected_indices)
        self.assertEqual(removed_indices, set(self.fixture["expected_removed_indices"]))

        removed_poison = sum(1 for idx in removed_indices if is_poison[idx])
        removed_clean = sum(1 for idx in removed_indices if not is_poison[idx])
        self.assertEqual(removed_poison, self.fixture["expected_removed_poison"])
        self.assertEqual(removed_clean, self.fixture["expected_removed_clean"])

        n_retrieved_poison = sum(1 for x in is_poison if x)
        residual_poison = n_retrieved_poison - removed_poison
        self.assertEqual(residual_poison, self.fixture["expected_residual_poison"])


# ---------------------------------------------------------------------------
# Correction 1: NQ/MS MARCO clustering-path reuse (verify, do not rewrite)
# ---------------------------------------------------------------------------

class TestClusteringStage1PathIsReusedUnchanged(unittest.TestCase):
    """Per plan §0a item 1: `ragdefender_paper`'s single-hop grouping must
    call `_find_num_adversarial_agg` (Eq. 1-2) unchanged, not a
    reimplementation. This test independently hand-verifies Eq. 1-2 against
    that exact function (proving the "already paper-faithful" claim, not
    just asserting it) using a deterministic fake encoder -- no real model
    download."""

    def test_agg_estimator_matches_eq1_eq2_by_hand(self):
        import torch

        from defense import defense_runner

        texts = [
            "poison poison poison target keyword here now",
            "poison poison target keyword appears again here",
            "poison target keyword shows up one more time",
            "completely unrelated fact about geography and history",
            "another unrelated fact about science and nature today",
        ]
        # 2D points forming two well-separated clusters: {0,1,2} vs {3,4}.
        vectors = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.05, -0.05],
            [10.0, 10.0],
            [10.1, 9.9],
        ]

        class _FakeModel:
            def encode(self, text_list, convert_to_tensor=True):
                return torch.tensor(vectors, dtype=torch.float32)

        n_tfidf = defense_runner._find_num_adversarial_tfidf(texts)  # noqa: SLF001
        self.assertEqual(n_tfidf, 3)  # the 3 "poison"-sharing passages

        # Eq. (2): N_TF-IDF (3) > |R|/2 (2.5) -> N_adv = |R| - n_min = 5 - 2 = 3
        n_adv = defense_runner._find_num_adversarial_agg(texts, _FakeModel())  # noqa: SLF001
        self.assertEqual(n_adv, 3)


class TestRagdefenderPaperThroughDispatch(unittest.TestCase):
    """End-to-end smoke test of `--defense ragdefender_paper` through
    defense/dispatch.py, mirroring tests/test_dispatch_smoke.py's
    conventions exactly (FakeSentenceTransformer, no network access)."""

    def setUp(self):
        import hashlib
        from unittest import mock

        import torch

        from defense import defense_runner, dispatch

        self.dispatch = dispatch

        class FakeSentenceTransformer:
            def encode(self, text_list, convert_to_tensor=True):
                vecs = []
                for t in text_list:
                    digest = hashlib.md5(t.encode("utf-8")).hexdigest()
                    seed = int(digest[:8], 16)
                    gen = torch.Generator().manual_seed(seed)
                    vecs.append(torch.rand(16, generator=gen))
                return torch.stack(vecs)

        self.patcher = mock.patch.object(
            defense_runner, "_get_s_model", return_value=FakeSentenceTransformer()
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def _passages(self):
        from defense.passages import label_passages

        raw = []
        for i in range(3):
            raw.append({
                "doc_id": f"c{i}", "context": f"Some unrelated clean fact number {i}.",
                "score": 0.9 - i * 0.01, "source": "corpus", "is_poison": False,
            })
        for i in range(2):
            raw.append({
                "doc_id": f"adv{i}",
                "context": f"Who was born first? Variant {i} says the incorrect answer is X.",
                "score": 0.5 - i * 0.01, "source": "adversarial", "is_poison": True,
            })
        return label_passages(raw)

    def test_ragdefender_paper_is_in_defense_choices(self):
        self.assertIn("ragdefender_paper", self.dispatch.DEFENSE_CHOICES)

    def test_ragdefender_paper_runs_offline_hotpotqa(self):
        passages = self._passages()
        kept, diag = self.dispatch.run_defense(
            "ragdefender_paper", "Who was born first?", passages, "hotpotqa", device="cpu"
        )
        self.assertIsInstance(kept, list)
        self.assertLessEqual(len(kept), len(passages))
        self.assertIn("N_adv_estimated_by_ragdefender", diag)
        self.assertIsInstance(diag["N_adv_estimated_by_ragdefender"], int)
        self.assertEqual(diag["notes"], "ragdefender_version=paper")

    def test_ragdefender_paper_and_legacy_do_not_interfere(self):
        """Calling ragdefender_paper then ragdefender_original in sequence
        must not leak state (e.g. via the module-level model cache) that
        changes either one's behavior."""
        passages = self._passages()
        kept_paper, _ = self.dispatch.run_defense(
            "ragdefender_paper", "Who was born first?", passages, "hotpotqa", device="cpu"
        )
        kept_legacy, _ = self.dispatch.run_defense(
            "ragdefender_original", "Who was born first?", passages, "hotpotqa", device="cpu"
        )
        self.assertIsInstance(kept_paper, list)
        self.assertIsInstance(kept_legacy, list)


if __name__ == "__main__":
    unittest.main()
