"""Tests for `scripts/stress_ml_filterrag_feature_oracle.py` -- the
ML-FilterRAG-top-k **feature-space oracle stress test**.

Fully offline and dependency-free of any real model artifact: a synthetic
CSV (built in-memory) plus a small fake classifier that mimics
`MLFilterRAGClassifier`'s `predict_proba`/`predict`/`feature_names`/
`threshold_default` surface stand in for the real trained
`hotpotqa_50q_mlfilterrag_topk_rf.joblib` artifact and
`ml_filterrag_dataset_hotpotqa_50q/features.csv`. No GPT/API import, no
`llm.query()` call, no retrieval, no passage text anywhere in this file or
in the module under test.

Run with:
    python -m unittest tests.test_stress_ml_filterrag_feature_oracle -v
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.stress_ml_filterrag_feature_oracle as oracle  # noqa: E402

FEATURE_NAMES = ("freq_density_score", "matched_freq_sum", "perplexity", "slm_answer_logprob")


class FakeClassifier:
    """Minimal stand-in for `defense.ml_filterrag.MLFilterRAGClassifier`:
    only the surface `build_targets`/`run_sweep`/`classify` actually touch
    (`feature_names`, `threshold_default`, `predict_proba`) -- no joblib
    artifact, no sklearn model, ever loaded."""

    def __init__(self, feature_names=FEATURE_NAMES, threshold_default=0.5, proba_fn=None):
        self.feature_names = tuple(feature_names)
        self.threshold_default = threshold_default
        # Default rule: "poison-looking" iff freq_density_score > 0.5 --
        # simple, deterministic, and trivially invertible for tests below.
        self._proba_fn = proba_fn or (lambda X: (X[:, 0] > 0.5).astype(float))

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self._proba_fn(X)

    def predict(self, X, threshold=None):
        t = self.threshold_default if threshold is None else threshold
        return (self.predict_proba(X) >= t).astype(int)


def make_df(rows):
    """`rows`: list of dicts with at least `is_poison` + FEATURE_NAMES
    keys; `query_id`/`k`/`split` are optional per-row overrides."""
    return pd.DataFrame(rows)


def poison_row(freq=0.9, matched=9.0, perp=50.0, logprob=-1.0, **overrides):
    row = {
        "is_poison": True, "freq_density_score": freq, "matched_freq_sum": matched,
        "perplexity": perp, "slm_answer_logprob": logprob,
    }
    row.update(overrides)
    return row


def clean_row(freq=0.1, matched=1.0, perp=5.0, logprob=-0.1, **overrides):
    row = {
        "is_poison": False, "freq_density_score": freq, "matched_freq_sum": matched,
        "perplexity": perp, "slm_answer_logprob": logprob,
    }
    row.update(overrides)
    return row


class TestAlphaOneLeavesFeaturesUnchanged(unittest.TestCase):
    def test_poison_rows_unchanged_at_alpha_1(self):
        df = make_df([
            poison_row(k=5, query_id="q1"), poison_row(freq=0.8, matched=7.0, k=5, query_id="q2"),
            clean_row(k=5, query_id="q3"), clean_row(freq=0.2, k=5, query_id="q4"),
        ])
        X = oracle.feature_matrix(df, FEATURE_NAMES)
        is_poison = df["is_poison"].to_numpy(dtype=bool)
        for strategy in oracle.ALL_STRATEGIES:
            targets, _meta = oracle.build_targets(strategy, df, X, FEATURE_NAMES)
            modified = oracle.interpolate_poison_features(X, targets, is_poison, alpha=1.0)
            np.testing.assert_array_equal(modified, X, err_msg=f"strategy={strategy!r} alpha=1.0 must be a no-op")


class TestAlphaZeroMapsPoisonToTarget(unittest.TestCase):
    def test_poison_rows_equal_target_at_alpha_0(self):
        df = make_df([
            poison_row(k=5, query_id="q1"), poison_row(freq=0.8, matched=7.0, k=10, query_id="q2"),
            clean_row(k=5, query_id="q3"), clean_row(freq=0.2, k=10, query_id="q4"),
        ])
        X = oracle.feature_matrix(df, FEATURE_NAMES)
        is_poison = df["is_poison"].to_numpy(dtype=bool)
        for strategy in oracle.ALL_STRATEGIES:
            targets, _meta = oracle.build_targets(strategy, df, X, FEATURE_NAMES)
            modified = oracle.interpolate_poison_features(X, targets, is_poison, alpha=0.0)
            np.testing.assert_allclose(
                modified[is_poison], targets[is_poison], err_msg=f"strategy={strategy!r} alpha=0.0 mismatch",
            )


class TestCleanRowsNeverModified(unittest.TestCase):
    def test_clean_rows_identical_across_every_alpha_and_strategy(self):
        df = make_df([
            poison_row(k=5, query_id="q1"), poison_row(freq=0.95, matched=12.0, k=10, query_id="q2"),
            clean_row(k=5, query_id="q3"), clean_row(freq=0.3, matched=2.0, k=10, query_id="q4"),
            clean_row(freq=0.05, perp=3.0, k=10, query_id="q5"),
        ])
        X = oracle.feature_matrix(df, FEATURE_NAMES)
        is_poison = df["is_poison"].to_numpy(dtype=bool)
        is_clean = ~is_poison
        for strategy in oracle.ALL_STRATEGIES:
            targets, _meta = oracle.build_targets(strategy, df, X, FEATURE_NAMES)
            for alpha in oracle.DEFAULT_ALPHA_SWEEP:
                modified = oracle.interpolate_poison_features(X, targets, is_poison, alpha)
                np.testing.assert_array_equal(
                    modified[is_clean], X[is_clean],
                    err_msg=f"clean rows changed for strategy={strategy!r} alpha={alpha!r}",
                )


class TestCleanCentroidTargetIsCorrect(unittest.TestCase):
    def test_matches_manual_mean_of_clean_rows(self):
        df = make_df([
            poison_row(k=5, query_id="q1"), poison_row(freq=0.8, k=5, query_id="q2"),
            clean_row(freq=0.1, matched=1.0, perp=5.0, logprob=-0.1, k=5, query_id="q3"),
            clean_row(freq=0.3, matched=3.0, perp=9.0, logprob=-0.3, k=5, query_id="q4"),
        ])
        X = oracle.feature_matrix(df, FEATURE_NAMES)
        targets, meta = oracle.build_targets_clean_centroid(df, X, FEATURE_NAMES)
        expected = np.array([0.2, 2.0, 7.0, -0.2])  # manual mean of the two clean rows
        np.testing.assert_allclose(targets[0], expected)
        np.testing.assert_allclose(targets[1], expected)
        # Every row (poison or clean) gets the same broadcast global centroid.
        np.testing.assert_allclose(targets[2], expected)
        self.assertEqual(meta["n_clean_rows_used"], 2)

    def test_raises_when_no_clean_rows_present(self):
        df = make_df([poison_row(), poison_row(freq=0.7)])
        X = oracle.feature_matrix(df, FEATURE_NAMES)
        with self.assertRaises(ValueError):
            oracle.build_targets_clean_centroid(df, X, FEATURE_NAMES)


class TestSameKCleanCentroidFallsBackToGlobal(unittest.TestCase):
    def test_k_group_with_no_clean_rows_uses_global_centroid(self):
        df = make_df([
            poison_row(k=5, query_id="q1"),           # k=5 has NO clean rows
            clean_row(freq=0.4, matched=4.0, perp=8.0, logprob=-0.4, k=10, query_id="q2"),
            poison_row(freq=0.6, k=10, query_id="q3"),
        ])
        X = oracle.feature_matrix(df, FEATURE_NAMES)
        targets, meta = oracle.build_targets_same_k_clean_centroid(df, X, FEATURE_NAMES)
        global_centroid = np.array([0.4, 4.0, 8.0, -0.4])  # only one clean row overall
        np.testing.assert_allclose(targets[0], global_centroid)  # k=5 poison row escalated to global
        np.testing.assert_allclose(targets[2], global_centroid)  # k=10's own (only) clean row == global anyway
        self.assertIn("5", meta["k_values_with_no_clean_rows_fell_back_to_global"])


class TestSameQueryCleanCentroidFallback(unittest.TestCase):
    def test_falls_back_to_same_k_when_query_has_no_clean_rows(self):
        df = make_df([
            poison_row(k=5, query_id="q_poison_only"),                       # q1: no clean rows at all
            clean_row(freq=0.2, matched=2.0, perp=6.0, logprob=-0.2, k=5, query_id="q_clean_A"),
            clean_row(freq=0.4, matched=4.0, perp=10.0, logprob=-0.4, k=5, query_id="q_clean_B"),
        ])
        X = oracle.feature_matrix(df, FEATURE_NAMES)
        same_k_targets, _ = oracle.build_targets_same_k_clean_centroid(df, X, FEATURE_NAMES)
        targets, meta = oracle.build_targets_same_query_clean_centroid(df, X, FEATURE_NAMES)
        # q_poison_only has zero clean rows -> must equal the (global, since
        # only one k group exists) same_k_clean_centroid target, not some
        # other query's centroid and not the raw poison features.
        np.testing.assert_allclose(targets[0], same_k_targets[0])
        self.assertIn("q_poison_only", meta["query_ids_with_no_clean_rows_fell_back_to_same_k"])
        self.assertEqual(meta["n_query_ids_fell_back"], 1)

    def test_uses_own_query_centroid_when_available(self):
        df = make_df([
            poison_row(k=5, query_id="qA"),
            clean_row(freq=0.2, matched=2.0, perp=6.0, logprob=-0.2, k=5, query_id="qA"),
            clean_row(freq=0.6, matched=6.0, perp=20.0, logprob=-0.6, k=5, query_id="qB"),  # different query
        ])
        X = oracle.feature_matrix(df, FEATURE_NAMES)
        targets, meta = oracle.build_targets_same_query_clean_centroid(df, X, FEATURE_NAMES)
        own_query_centroid = np.array([0.2, 2.0, 6.0, -0.2])
        np.testing.assert_allclose(targets[0], own_query_centroid)
        self.assertEqual(meta["query_ids_with_no_clean_rows_fell_back_to_same_k"], [])

    def test_falls_back_entirely_when_no_query_id_column(self):
        df = make_df([
            {"is_poison": True, "freq_density_score": 0.9, "matched_freq_sum": 9.0, "perplexity": 50.0, "slm_answer_logprob": -1.0, "k": 5},
            {"is_poison": False, "freq_density_score": 0.1, "matched_freq_sum": 1.0, "perplexity": 5.0, "slm_answer_logprob": -0.1, "k": 5},
        ])
        self.assertNotIn("query_id", df.columns)
        X = oracle.feature_matrix(df, FEATURE_NAMES)
        same_k_targets, _ = oracle.build_targets_same_k_clean_centroid(df, X, FEATURE_NAMES)
        targets, meta = oracle.build_targets_same_query_clean_centroid(df, X, FEATURE_NAMES)
        np.testing.assert_array_equal(targets, same_k_targets)
        self.assertFalse(meta["has_query_id_column"])


class TestNearestCleanBijection(unittest.TestCase):
    def test_every_poison_row_gets_a_real_clean_rows_features_as_target(self):
        df = make_df([
            poison_row(freq=0.91, matched=9.1, perp=51.0, logprob=-1.1, k=5, query_id="q1"),
            poison_row(freq=0.2, matched=2.0, perp=6.0, logprob=-0.2, k=5, query_id="q2"),  # deliberately clean-like
            clean_row(freq=0.1, matched=1.0, perp=5.0, logprob=-0.1, k=5, query_id="q3"),
            clean_row(freq=0.9, matched=9.0, perp=50.0, logprob=-1.0, k=5, query_id="q4"),  # deliberately poison-like
        ])
        X = oracle.feature_matrix(df, FEATURE_NAMES)
        targets, meta = oracle.build_targets_nearest_clean_bijection(df, X, FEATURE_NAMES)
        is_poison = df["is_poison"].to_numpy(dtype=bool)
        is_clean = ~is_poison
        clean_rows = X[is_clean]
        for row in targets[is_poison]:
            # Every poison row's target must be an exact copy of some clean row's own feature vector.
            self.assertTrue(any(np.allclose(row, c) for c in clean_rows))
        self.assertEqual(meta["strategy"], oracle.NEAREST_CLEAN_BIJECTION)

    def test_deterministic_across_repeated_calls(self):
        df = make_df([
            poison_row(freq=0.9, k=10, query_id="q1"), poison_row(freq=0.7, matched=5.0, k=10, query_id="q2"),
            poison_row(freq=0.6, matched=4.0, perp=30.0, k=10, query_id="q3"),
            clean_row(freq=0.1, k=10, query_id="q4"), clean_row(freq=0.3, matched=2.0, k=10, query_id="q5"),
        ])
        X = oracle.feature_matrix(df, FEATURE_NAMES)
        targets1, _ = oracle.build_targets_nearest_clean_bijection(df, X, FEATURE_NAMES, seed=12)
        targets2, _ = oracle.build_targets_nearest_clean_bijection(df, X, FEATURE_NAMES, seed=12)
        np.testing.assert_array_equal(targets1, targets2)

    def test_escalates_to_global_pool_when_k_group_has_no_clean_rows(self):
        df = make_df([
            poison_row(k=5, query_id="q1"),  # k=5: no clean rows locally
            clean_row(freq=0.15, matched=1.5, perp=5.5, logprob=-0.15, k=10, query_id="q2"),
            poison_row(freq=0.5, k=10, query_id="q3"),
        ])
        X = oracle.feature_matrix(df, FEATURE_NAMES)
        targets, meta = oracle.build_targets_nearest_clean_bijection(df, X, FEATURE_NAMES)
        only_clean_row = X[1]
        np.testing.assert_allclose(targets[0], only_clean_row)
        self.assertIn("5", meta["k_values_escalated_to_global_clean_pool"])

    def test_brute_force_matches_greedy_on_small_equal_groups(self):
        """A tiny, hand-checkable 2-poison/2-clean group: the optimal
        bijection should NOT be the naive index-order pairing when that
        pairing isn't actually the nearest one."""
        poison_positions = [0, 1]
        clean_positions = [2, 3]
        X = np.array([
            [0.0, 0.0, 0.0, 0.0],   # poison 0 -- nearest to clean at index 3
            [10.0, 10.0, 10.0, 10.0],  # poison 1 -- nearest to clean at index 2
            [9.0, 9.0, 9.0, 9.0],    # clean 2
            [1.0, 1.0, 1.0, 1.0],    # clean 3
        ])
        assignment = oracle._brute_force_match(poison_positions, clean_positions, X)
        self.assertEqual(assignment, {0: 3, 1: 2})


class TestClassifierPredictionsRecomputedFromModifiedMatrix(unittest.TestCase):
    def test_prediction_reflects_modified_features_not_original(self):
        df = make_df([
            poison_row(freq=0.9, k=5, query_id="q1"),
            clean_row(freq=0.1, k=5, query_id="q2"),
        ])
        clf = FakeClassifier()  # poison-looking iff freq_density_score > 0.5
        X = oracle.feature_matrix(df, FEATURE_NAMES)
        is_poison = df["is_poison"].to_numpy(dtype=bool)
        targets, _ = oracle.build_targets_clean_centroid(df, X, FEATURE_NAMES)  # target freq = 0.1

        pred_unmodified, _ = oracle.classify(clf, X, threshold=0.5)
        self.assertEqual(list(pred_unmodified), [1, 0])  # poison row flagged, clean row not

        modified_full_jump = oracle.interpolate_poison_features(X, targets, is_poison, alpha=0.0)
        pred_after, _ = oracle.classify(clf, modified_full_jump, threshold=0.5)
        self.assertEqual(list(pred_after), [0, 0])  # poison row now evades detection


class TestMetricsComputedCorrectly(unittest.TestCase):
    def test_recall_fpr_residuals_and_displacement_match_manual_computation(self):
        # 3 poison rows (2 detected, 1 evades), 2 clean rows (1 false-positive).
        is_poison = np.array([True, True, True, False, False])
        pred = np.array([1, 1, 0, 1, 0])
        X_original = np.zeros((5, 4))
        X_modified = np.array(X_original, copy=True)
        X_modified[0] = [1.0, 0.0, 0.0, 0.0]  # displacement sqrt(1)=1 for poison row 0
        X_modified[1] = [3.0, 4.0, 0.0, 0.0]  # displacement 5 for poison row 1 (max)
        # poison row 2 and both clean rows unmodified (displacement 0 / not counted for clean).

        metrics = oracle.compute_alpha_metrics(
            X_original=X_original, X_modified=X_modified, is_poison=is_poison, pred=pred,
            feature_names=FEATURE_NAMES,
        )
        self.assertEqual(metrics["n_poison"], 3)
        self.assertEqual(metrics["n_clean"], 2)
        self.assertEqual(metrics["removed_poison"], 2)
        self.assertEqual(metrics["removed_clean"], 1)
        self.assertAlmostEqual(metrics["poison_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["clean_false_positive_rate"], 1 / 2)
        self.assertEqual(metrics["residual_poison_count"], 1)  # 1 poison row survived (pred==0)
        self.assertEqual(metrics["residual_clean_count"], 1)   # 1 clean row survived (pred==0)
        self.assertAlmostEqual(metrics["residual_poison_fraction"], 1 / 2)
        self.assertAlmostEqual(metrics["mean_poison_l2_displacement"], (1.0 + 5.0 + 0.0) / 3)
        self.assertAlmostEqual(metrics["max_poison_l2_displacement"], 5.0)
        self.assertAlmostEqual(metrics["mean_abs_change__freq_density_score"], (1.0 + 3.0 + 0.0) / 3)
        self.assertAlmostEqual(metrics["mean_abs_change__matched_freq_sum"], (0.0 + 4.0 + 0.0) / 3)

    def test_zero_poison_rows_reports_none_recall_and_none_displacement(self):
        is_poison = np.array([False, False])
        pred = np.array([0, 1])
        X = np.zeros((2, 4))
        metrics = oracle.compute_alpha_metrics(
            X_original=X, X_modified=X, is_poison=is_poison, pred=pred, feature_names=FEATURE_NAMES,
        )
        self.assertIsNone(metrics["poison_recall"])
        self.assertIsNone(metrics["mean_poison_l2_displacement"])
        self.assertEqual(metrics["clean_false_positive_rate"], 0.5)

    def test_first_break_alphas_reports_first_alpha_below_each_threshold(self):
        alpha_to_recall = {1.0: 1.0, 0.9: 0.95, 0.8: 0.85, 0.7: 0.6, 0.6: 0.4, 0.5: 0.1}
        breaks = oracle.first_break_alphas(alpha_to_recall, thresholds=(0.9, 0.8, 0.5))
        self.assertEqual(breaks[0.9], 0.8)   # first alpha (desc order) with recall < 0.9
        self.assertEqual(breaks[0.8], 0.7)
        self.assertEqual(breaks[0.5], 0.6)

    def test_first_break_alphas_none_when_recall_never_drops_below(self):
        alpha_to_recall = {1.0: 1.0, 0.5: 0.95, 0.0: 0.91}
        breaks = oracle.first_break_alphas(alpha_to_recall, thresholds=(0.9,))
        self.assertIsNone(breaks[0.9])


class TestRunSweepEndToEnd(unittest.TestCase):
    def _fixture_df(self):
        return make_df([
            poison_row(freq=0.9, matched=9.0, perp=50.0, logprob=-1.0, k=5, query_id="q1", split="train"),
            poison_row(freq=0.85, matched=8.0, perp=45.0, logprob=-0.9, k=5, query_id="q2", split="test"),
            clean_row(freq=0.1, matched=1.0, perp=5.0, logprob=-0.1, k=5, query_id="q3", split="train"),
            clean_row(freq=0.15, matched=1.5, perp=6.0, logprob=-0.15, k=10, query_id="q4", split="test"),
            poison_row(freq=0.6, matched=5.0, perp=30.0, logprob=-0.5, k=10, query_id="q5", split="train"),
        ])

    def test_run_sweep_produces_one_row_per_strategy_alpha_and_recomputes_predictions(self):
        df = self._fixture_df()
        clf = FakeClassifier()
        alphas = (1.0, 0.5, 0.0)
        strategies = oracle.ALL_STRATEGIES
        rows, strategy_meta = oracle.run_sweep(
            df, clf, FEATURE_NAMES, threshold=0.5, alphas=alphas, strategies=strategies,
        )
        self.assertEqual(len(rows), len(alphas) * len(strategies))
        for strategy in strategies:
            self.assertIn("first_break_alphas", strategy_meta[strategy])
        # At alpha=1.0 (no modification), poison_recall must equal the
        # unmodified classifier's own recall for every strategy.
        X = oracle.feature_matrix(df, FEATURE_NAMES)
        is_poison = df["is_poison"].to_numpy(dtype=bool)
        pred_unmodified, _ = oracle.classify(clf, X, threshold=0.5)
        expected_recall = pred_unmodified[is_poison].sum() / is_poison.sum()
        for row in rows:
            if row["alpha"] == 1.0:
                self.assertAlmostEqual(row["poison_recall"], expected_recall)


class TestOutputFilesWritten(unittest.TestCase):
    def test_csv_report_and_config_are_written(self):
        df = make_df([
            poison_row(freq=0.9, k=5, query_id="q1"),
            clean_row(freq=0.1, k=5, query_id="q2"),
        ])
        clf = FakeClassifier()
        alphas = (1.0, 0.5, 0.0)
        strategies = (oracle.CLEAN_CENTROID, oracle.SAME_K_CLEAN_CENTROID)
        rows, strategy_meta = oracle.run_sweep(
            df, clf, FEATURE_NAMES, threshold=0.5, alphas=alphas, strategies=strategies,
        )
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "FEATURE_ORACLE_SWEEP.csv")
            report_path = os.path.join(tmp, "FEATURE_ORACLE_REPORT.md")
            config_path = os.path.join(tmp, "run_config.json")

            oracle.write_sweep_csv(rows, csv_path, FEATURE_NAMES, oracle.DEFAULT_RECALL_BREAK_THRESHOLDS)
            oracle.write_report_md(
                report_path, model_path="FAKE_MODEL.joblib", features_csv="FAKE_FEATURES.csv",
                feature_names=FEATURE_NAMES, threshold=0.5, n_rows=len(df),
                n_poison=int(df["is_poison"].sum()), n_clean=int((~df["is_poison"]).sum()),
                split_counts=None, alphas=alphas, strategies=strategies, sweep_rows=rows,
                strategy_meta=strategy_meta, recall_break_thresholds=oracle.DEFAULT_RECALL_BREAK_THRESHOLDS,
            )
            oracle.write_run_config(config_path, model_path="FAKE_MODEL.joblib", n_rows=len(df))

            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(report_path))
            self.assertTrue(os.path.exists(config_path))

            written = pd.read_csv(csv_path)
            self.assertEqual(len(written), len(alphas) * len(strategies))

            with open(report_path, encoding="utf-8") as f:
                report_text = f.read()
            self.assertIn("feature-space oracle", report_text.lower())
            self.assertIn("not a text-realizable attack", report_text.lower())
            self.assertIn("top-s Algorithm 2".lower(), report_text.lower())

            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            self.assertEqual(config["model_path"], "FAKE_MODEL.joblib")


class TestResolveFeatureNames(unittest.TestCase):
    def test_default_uses_classifier_feature_names(self):
        clf = FakeClassifier(feature_names=("a", "b"))
        self.assertEqual(oracle.resolve_feature_names(clf), ("a", "b"))

    def test_override_takes_precedence(self):
        clf = FakeClassifier(feature_names=("a", "b"))
        self.assertEqual(oracle.resolve_feature_names(clf, override=["c", "d"]), ("c", "d"))


class TestNoGptApiImportsOrLlmQueryCalls(unittest.TestCase):
    def test_no_banned_api_imports_in_module_source(self):
        import inspect

        source = inspect.getsource(oracle)
        banned = ["openai", "google.generativeai", "anthropic", "cohere"]
        for name in banned:
            self.assertNotIn(name, source, f"found banned API import/reference {name!r}")

    def test_no_llm_query_calls_outside_documentation(self):
        import inspect

        source = inspect.getsource(oracle)
        # Every literal "llm.query(" occurrence must be inside a backtick-quoted
        # documentation mention (`llm.query()`), never an actual call.
        self.assertEqual(source.count("llm.query("), source.count("`llm.query()`"))

    def test_module_never_imports_torch_transformers_or_retrieval_harness(self):
        import inspect

        source = inspect.getsource(oracle)
        for banned_import in ("import torch", "from transformers", "from src.attack", "load_beir_datasets"):
            self.assertNotIn(banned_import, source)

    def test_module_never_imports_or_calls_into_training_or_defense_modules_beyond_the_classifier(self):
        import inspect

        source = inspect.getsource(oracle)
        self.assertNotIn("defense.filterrag", source)
        self.assertNotIn("defense.cluster_normalized_poisoning", source)
        self.assertNotIn("defense.ragdefender_internals", source)
        self.assertIn("from defense.ml_filterrag import", source)


if __name__ == "__main__":
    unittest.main()
