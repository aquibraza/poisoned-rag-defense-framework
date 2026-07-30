"""Tests for scripts/build_formal_oracle_comparison.py -- the consolidated
formal oracle comparison report (E1 vs CORAL-PCA vs CORAL-ridge vs MMD).

This script reads only already-existing artifacts (never reruns any
intervention), so most tests here use small synthetic fixture CSVs with
known, hand-crafted failure patterns to verify the summarization logic
deterministically and independently of whatever real runs happen to exist
on disk. A separate test class checks the specific known values from the
real runs already on disk (skipped gracefully if those `results/` "
artifacts are not present, e.g. a fresh checkout, since `results/` is
gitignored).

Run with: python -m unittest tests.test_build_formal_oracle_comparison -v
"""
import ast
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "build_formal_oracle_comparison.py")
_spec = importlib.util.spec_from_file_location("build_formal_oracle_comparison", _SCRIPT_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------

def _make_e1_df() -> pd.DataFrame:
    """2 queries x 2 strategies x 3 alphas (1.0 baseline, 0.6 mid, 0.3 max
    perturbation). `q1` fails (residual poison) at alpha<=0.6 for both
    strategies; `q2` never fails."""
    rows = []
    for strategy in ("rank_aligned", "nearest_bijection"):
        for qid, fails_at in [("q1", 0.6), ("q2", None)]:
            for alpha, top_pair_pp in [(1.0, 8), (0.6, 4), (0.3, 1)]:
                fails = fails_at is not None and alpha <= fails_at
                removed_poison = 2 if fails else 5
                rows.append({
                    "query_id": qid, "alpha": alpha, "anchor_strategy": strategy,
                    "N_retrieved_poison": 5, "N_retrieved_clean": 5,
                    "top_pair_pp": top_pair_pp, "top_pair_pc": 10 - top_pair_pp,
                    "removed_poison": removed_poison, "removed_clean": 0,
                    "residual_poison_fraction": (5 - removed_poison) / 5,
                    "decision_label": "residual_poison_failure" if fails else "over_removal_success",
                })
    return pd.DataFrame(rows)


def _make_perturbation_sweep_df(param_col: str, extra_param: str = None,
                                 extra_values=(None,), fail_at_max=False) -> pd.DataFrame:
    """2 queries x (extra_values) x [0, 1] `param_col` (0 = identity
    baseline, 1 = max perturbation). If `fail_at_max`, `q1` fails
    (`removed_poison < N_retrieved_poison`) at `param_col == 1` for every
    `extra_values` entry equal to the *first* one only (to allow crafting
    a controlled per-(extra_value) failure count)."""
    rows = []
    for extra in extra_values:
        for qid in ("q1", "q2"):
            for val in (0, 1):
                fails = fail_at_max and val == 1 and qid == "q1" and extra == extra_values[0]
                removed_poison = 2 if fails else 5
                row = {
                    "query_id": qid, param_col: val,
                    "N_retrieved_poison": 5, "N_retrieved_clean": 5,
                    "top_pair_pp": 10 if val == 0 else (9 if not fails else 1),
                    "removed_poison": removed_poison, "removed_clean": 0,
                    "residual_poison_fraction": (5 - removed_poison) / 5,
                    "mmd_distance_before": 0.8, "mmd_distance_after": 0.8 - 0.1 * val,
                    "mmd_distance_reduction": 0.1 * val,
                    "coral_distance_before": 0.2, "coral_distance_after": 0.2 - 0.02 * val,
                    "coral_distance_reduction": 0.02 * val,
                    "mean_poison_l2_displacement": 0.5 * val, "max_poison_l2_displacement": 0.6 * val,
                    "mean_poison_original_cosine": 1.0 - 0.3 * val, "min_poison_original_cosine": 1.0 - 0.4 * val,
                }
                if extra_param is not None:
                    row[extra_param] = extra
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Artifact discovery
# --------------------------------------------------------------------------

class TestArtifactDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _make_run(self, name: str, filename: str, content: str = "x"):
        run_dir = Path(self.tmpdir) / name
        run_dir.mkdir(parents=True)
        (run_dir / filename).write_text(content)
        return run_dir / filename

    def test_picks_latest_coral_pca_run(self):
        self._make_run("20260101_000000_coral_pca_hotpotqa_k10_N5", "CORAL_PCA_SWEEP.csv", "old")
        newest = self._make_run("20260201_000000_coral_pca_hotpotqa_k10_N5", "CORAL_PCA_SWEEP.csv", "new")
        found = mod.discover_latest_artifact(self.tmpdir, "*_coral_pca_*", "CORAL_PCA_SWEEP.csv")
        self.assertEqual(found, newest)

    def test_picks_latest_coral_ridge_run(self):
        self._make_run("20260101_000000_coral_ridge_hotpotqa_k10_N5", "CORAL_RIDGE_SWEEP.csv", "old")
        self._make_run("20260115_000000_coral_ridge_hotpotqa_k10_N5", "CORAL_RIDGE_SWEEP.csv", "mid")
        newest = self._make_run("20260201_000000_coral_ridge_hotpotqa_k10_N5", "CORAL_RIDGE_SWEEP.csv", "new")
        found = mod.discover_latest_artifact(self.tmpdir, "*_coral_ridge_*", "CORAL_RIDGE_SWEEP.csv")
        self.assertEqual(found, newest)

    def test_picks_latest_mmd_run(self):
        self._make_run("20260101_000000_mmd_hotpotqa_k10_N5", "MMD_SWEEP.csv", "old")
        newest = self._make_run("20260301_000000_mmd_hotpotqa_k10_N5", "MMD_SWEEP.csv", "new")
        found = mod.discover_latest_artifact(self.tmpdir, "*_mmd_*", "MMD_SWEEP.csv")
        self.assertEqual(found, newest)

    def test_returns_none_when_no_match(self):
        found = mod.discover_latest_artifact(self.tmpdir, "*_coral_pca_*", "CORAL_PCA_SWEEP.csv")
        self.assertIsNone(found)

    def test_does_not_pick_unrelated_intervention_directory(self):
        """A `*_coral_ridge_*` run directory must never be picked up by the
        `*_coral_pca_*` glob, even though both live under the same parent."""
        self._make_run("20260301_000000_coral_ridge_hotpotqa_k10_N5", "CORAL_RIDGE_SWEEP.csv", "ridge")
        found = mod.discover_latest_artifact(self.tmpdir, "*_coral_pca_*", "CORAL_PCA_SWEEP.csv")
        self.assertIsNone(found)


# --------------------------------------------------------------------------
# Summarization logic (synthetic fixtures -- deterministic, no real artifacts)
# --------------------------------------------------------------------------

class TestSummarizeE1(unittest.TestCase):
    def test_query_level_failure_count_and_row_counts(self):
        e1_df = _make_e1_df()
        summary = mod.summarize_e1("rank_aligned", e1_df)
        self.assertEqual(summary["method"], "E1_rank_aligned")
        self.assertEqual(summary["n_tested_rows"], 6)  # 2 queries x 3 alphas
        self.assertEqual(summary["n_query_level_units"], 2)
        self.assertEqual(summary["query_level_failure_count"], 1)  # only q1 fails
        self.assertEqual(summary["first_failure_perturbation"], 0.6)
        self.assertEqual(summary["top_pair_pp_baseline"], 8.0)
        self.assertEqual(summary["top_pair_pp_at_max_perturbation"], 1.0)
        self.assertAlmostEqual(summary["top_pair_pp_reduction"], 7.0)
        self.assertIsNone(summary["mmd_distance_before"])
        self.assertIsNone(summary["mean_poison_l2_displacement"])


class TestSummarizePerturbationSweptMethod(unittest.TestCase):
    def test_zero_failure_case(self):
        df = _make_perturbation_sweep_df("beta", fail_at_max=False)
        summary = mod.summarize_perturbation_swept_method("CORAL_PCA", df, param_col="beta")
        self.assertEqual(summary["n_tested_rows"], 4)  # 2 queries x 2 betas
        self.assertEqual(summary["n_query_level_units"], 2)
        self.assertEqual(summary["residual_poison_failure_row_count"], 0)
        self.assertEqual(summary["query_level_failure_count"], 0)
        self.assertIsNone(summary["first_failure_perturbation"])
        self.assertAlmostEqual(summary["mmd_distance_reduction"], 0.1)
        self.assertAlmostEqual(summary["mean_poison_l2_displacement"], 0.5)

    def test_one_query_failure_case(self):
        df = _make_perturbation_sweep_df("beta", fail_at_max=True)
        summary = mod.summarize_perturbation_swept_method("CORAL_PCA", df, param_col="beta")
        self.assertEqual(summary["query_level_failure_count"], 1)
        self.assertEqual(summary["first_failure_perturbation"], 1.0)
        self.assertGreaterEqual(summary["residual_poison_failure_row_count"], 1)

    def test_mmd_style_two_lambda_preserve_grid(self):
        """Mirrors the real MMD sweep's structure: two `lambda_preserve`
        groups, one that fails and one that doesn't, summed to a combined
        query-lambda failure count (the same aggregation the real
        MMD_lambda_preserve rows use)."""
        df_fail = _make_perturbation_sweep_df("steps", extra_param="lambda_preserve",
                                               extra_values=(0.01,), fail_at_max=True)
        df_ok = _make_perturbation_sweep_df("steps", extra_param="lambda_preserve",
                                             extra_values=(1.0,), fail_at_max=False)
        summary_fail = mod.summarize_perturbation_swept_method("MMD_lambda_preserve=0.01", df_fail, param_col="steps")
        summary_ok = mod.summarize_perturbation_swept_method("MMD_lambda_preserve=1.00", df_ok, param_col="steps")
        self.assertEqual(summary_fail["query_level_failure_count"], 1)
        self.assertEqual(summary_ok["query_level_failure_count"], 0)
        total_failures = summary_fail["query_level_failure_count"] + summary_ok["query_level_failure_count"]
        total_units = summary_fail["n_query_level_units"] + summary_ok["n_query_level_units"]
        self.assertEqual((total_failures, total_units), (1, 4))


# --------------------------------------------------------------------------
# End-to-end: main() over synthetic fixtures written to a temp dir
# --------------------------------------------------------------------------

class TestEndToEndSynthetic(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.e1_csv = Path(self.tmpdir) / "BATCH_COMPARISON_SUCCESS_CASES.csv"
        _make_e1_df().to_csv(self.e1_csv, index=False)

        self.formal_dir = Path(self.tmpdir) / "formal"
        pca_dir = self.formal_dir / "20260101_000000_coral_pca_hotpotqa_k10_N5"
        pca_dir.mkdir(parents=True)
        _make_perturbation_sweep_df("beta", fail_at_max=False).to_csv(pca_dir / "CORAL_PCA_SWEEP.csv", index=False)

        ridge_dir = self.formal_dir / "20260101_000000_coral_ridge_hotpotqa_k10_N5"
        ridge_dir.mkdir(parents=True)
        ridge_df = pd.concat([
            _make_perturbation_sweep_df("beta", extra_param="lambda", extra_values=(0.1,), fail_at_max=False),
            _make_perturbation_sweep_df("beta", extra_param="lambda", extra_values=(0.01,), fail_at_max=False),
            _make_perturbation_sweep_df("beta", extra_param="lambda", extra_values=(0.001,), fail_at_max=False),
        ])
        ridge_df.to_csv(ridge_dir / "CORAL_RIDGE_SWEEP.csv", index=False)

        mmd_dir = self.formal_dir / "20260101_000000_mmd_hotpotqa_k10_N5"
        mmd_dir.mkdir(parents=True)
        mmd_df = pd.concat([
            _make_perturbation_sweep_df("steps", extra_param="lambda_preserve", extra_values=(0.01,), fail_at_max=True),
            _make_perturbation_sweep_df("steps", extra_param="lambda_preserve", extra_values=(0.10,), fail_at_max=True),
            _make_perturbation_sweep_df("steps", extra_param="lambda_preserve", extra_values=(1.00,), fail_at_max=False),
        ])
        mmd_df.to_csv(mmd_dir / "MMD_SWEEP.csv", index=False)

        self.output_dir = Path(self.tmpdir) / "output"

    def test_main_writes_csv_and_report_with_expected_rows(self):
        md_path = mod.main([
            "--e1_csv", str(self.e1_csv),
            "--formal_dir", str(self.formal_dir),
            "--output_dir", str(self.output_dir),
        ])
        self.assertTrue(md_path.exists())
        csv_path = self.output_dir / "FORMAL_ORACLE_COMPARISON.csv"
        self.assertTrue(csv_path.exists())

        df = pd.read_csv(csv_path)
        expected_methods = [
            "E1_rank_aligned", "E1_nearest_bijection", "CORAL_PCA",
            "CORAL_RIDGE_lambda=0.1", "CORAL_RIDGE_lambda=0.01", "CORAL_RIDGE_lambda=0.001",
            "MMD_lambda_preserve=0.01", "MMD_lambda_preserve=0.10", "MMD_lambda_preserve=1.00",
        ]
        self.assertEqual(list(df["method"]), expected_methods)
        for col in mod.REQUIRED_COLUMNS:
            self.assertIn(col, df.columns)

        # Known synthetic failure counts (mirrors the real-run pattern).
        pca_row = df[df["method"] == "CORAL_PCA"].iloc[0]
        self.assertEqual(pca_row["query_level_failure_count"], 0)
        ridge_failures = df[df["method"].str.startswith("CORAL_RIDGE")]["query_level_failure_count"].sum()
        self.assertEqual(ridge_failures, 0)
        mmd_failures = df[df["method"].str.startswith("MMD_")]["query_level_failure_count"].sum()
        self.assertEqual(mmd_failures, 2)  # 1 (lp=0.01) + 1 (lp=0.10) + 0 (lp=1.00)

        report_text = md_path.read_text(encoding="utf-8")
        self.assertIn("MMD is **stronger than CORAL but weaker than E1 in coverage**", report_text)
        self.assertIn("MMD distance decreases under MMD optimization; CORAL distance is reported "
                       "separately and is not the primary optimized objective", report_text)
        self.assertNotIn("comparable to E1.", report_text)
        self.assertIn("Limitations", report_text)
        self.assertIn("oracle embedding-space stress test", report_text)

    def test_raises_when_e1_csv_missing(self):
        with self.assertRaises(FileNotFoundError):
            mod.main([
                "--e1_csv", str(Path(self.tmpdir) / "missing.csv"),
                "--formal_dir", str(self.formal_dir),
                "--output_dir", str(self.output_dir),
            ])

    def test_raises_when_no_pca_artifact_found(self):
        empty_formal_dir = Path(self.tmpdir) / "empty_formal"
        empty_formal_dir.mkdir()
        with self.assertRaises(FileNotFoundError):
            mod.main([
                "--e1_csv", str(self.e1_csv),
                "--formal_dir", str(empty_formal_dir),
                "--output_dir", str(self.output_dir),
            ])


# --------------------------------------------------------------------------
# Known values against the real artifacts already on disk (skipped if absent)
# --------------------------------------------------------------------------

class TestKnownValuesAgainstRealArtifacts(unittest.TestCase):
    def setUp(self):
        self.e1_csv = Path(REPO_ROOT) / mod.DEFAULT_E1_CSV
        self.formal_dir = Path(REPO_ROOT) / mod.DEFAULT_FORMAL_DIR
        self.pca_csv = mod.discover_latest_artifact(str(self.formal_dir), "*_coral_pca_*", "CORAL_PCA_SWEEP.csv")
        self.ridge_csv = mod.discover_latest_artifact(str(self.formal_dir), "*_coral_ridge_*", "CORAL_RIDGE_SWEEP.csv")
        self.mmd_csv = mod.discover_latest_artifact(str(self.formal_dir), "*_mmd_*", "MMD_SWEEP.csv")
        if not (self.e1_csv.exists() and self.pca_csv and self.ridge_csv and self.mmd_csv):
            self.skipTest(
                "Real cluster_normalized_poisoning(_formal) artifacts not found on disk "
                "(results/ is gitignored) -- skipping known-value regression check."
            )

    def test_known_failure_counts(self):
        e1_df = pd.read_csv(self.e1_csv)
        pca_df = pd.read_csv(self.pca_csv)
        ridge_df = pd.read_csv(self.ridge_csv)
        mmd_df = pd.read_csv(self.mmd_csv)
        comparison_df = mod.build_comparison_table(e1_df, pca_df, ridge_df, mmd_df)

        pca_row = comparison_df[comparison_df["method"] == "CORAL_PCA"].iloc[0]
        self.assertEqual(pca_row["n_query_level_units"], 6)
        self.assertEqual(pca_row["query_level_failure_count"], 0)  # CORAL-PCA: 0/6 query failures

        ridge_rows = comparison_df[comparison_df["method"].str.startswith("CORAL_RIDGE")]
        self.assertEqual(int(ridge_rows["n_query_level_units"].sum()), 18)
        self.assertEqual(int(ridge_rows["query_level_failure_count"].sum()), 0)  # CORAL-ridge: 0/18

        mmd_rows = comparison_df[comparison_df["method"].str.startswith("MMD_")]
        self.assertEqual(int(mmd_rows["n_query_level_units"].sum()), 18)
        self.assertEqual(int(mmd_rows["query_level_failure_count"].sum()), 10)  # MMD: 10/18

    def test_real_report_contains_required_wording(self):
        md_path = mod.main([
            "--e1_csv", str(self.e1_csv),
            "--formal_dir", str(self.formal_dir),
            "--pca_csv", str(self.pca_csv),
            "--ridge_csv", str(self.ridge_csv),
            "--mmd_csv", str(self.mmd_csv),
            "--output_dir", tempfile.mkdtemp(),
        ])
        report_text = md_path.read_text(encoding="utf-8")
        self.assertIn("MMD is **stronger than CORAL but weaker than E1 in coverage**", report_text)


# --------------------------------------------------------------------------
# No GPT/API calls / forbidden imports
# --------------------------------------------------------------------------

class TestNoForbiddenImports(unittest.TestCase):
    def test_no_forbidden_imports(self):
        with open(_SCRIPT_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])
        forbidden = {"openai", "requests", "httpx", "urllib", "sentence_transformers", "torch"}
        self.assertEqual(imported_modules & forbidden, set())

    def test_only_allowed_imports(self):
        with open(_SCRIPT_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        allowed = {
            "argparse", "os", "sys", "pathlib", "typing",
            "numpy", "pandas", "build_batch_comparison_success_cases", "__future__",
        }
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])
        self.assertTrue(imported_modules.issubset(allowed),
                        f"unexpected imports: {imported_modules - allowed}")

    def test_script_never_imports_generation_or_embedder_modules(self):
        """This report reads CSVs only -- it must never import the
        embedder-loading helpers (`visualize_ragdefender_clusters`) or any
        defense/runner module that could trigger generation."""
        with open(_SCRIPT_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        for forbidden_name in ("visualize_ragdefender_clusters", "defense_runner", "main.py", "dispatch"):
            self.assertNotIn(forbidden_name, source)


if __name__ == "__main__":
    unittest.main()
