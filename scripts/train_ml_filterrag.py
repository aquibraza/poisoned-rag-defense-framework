#!/usr/bin/env python3
"""Train an `MLFilterRAGClassifier` ("ML-FilterRAG-top-k") on a feature
dataset built by `scripts/build_ml_filterrag_dataset.py`.

Trains on `split == "train"` rows only, using `DEFAULT_FEATURE_NAMES`
(exactly the 4 paper-cited features: `freq_density_score`,
`matched_freq_sum`, `perplexity`, `slm_answer_logprob`) unless
`--feature_set all` is explicitly passed, in which case the classifier is
trained on `DEFAULT_FEATURE_NAMES + AUXILIARY_FEATURE_NAMES` and every
artifact/report is explicitly labeled a **repo-augmented ML-FilterRAG-top-k
variant**, never plain "ML-FilterRAG" (see
`docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md` section 3).

Evaluates on `split == "test"` rows only (held-out `query_id`s -- the
train/test `query_id` sets are re-checked for zero overlap before training,
mirroring `defense.ml_filterrag.assert_no_query_id_leakage()`'s use in
`scripts/build_ml_filterrag_dataset.py`), and reports precision/recall/F1/
confusion matrix (always) and ROC-AUC (only when the test split contains
both classes -- otherwise skipped with an explicit message, never a crash
or a fabricated 0.5).

**Per-dataset paper-aligned classifier choice** (Appendix C, Table VI):
Random Forest for HotpotQA/MS-MARCO, XGBoost for NQ. If `--model_type` is
omitted, it defaults via `defense.ml_filterrag.paper_aligned_model_type
(--dataset)`. Training NQ with Random Forest (because `xgboost` isn't
installed, or by explicit request) requires `--allow_proxy_classifier` and
stamps `training_meta["proxy_classifier"] = True` plus a proxy label into
the artifact filename/report header -- it is never a silent substitution.

**No GPT/API call is ever made, no `llm.query()` call, no live
generation** -- this script only reads a pre-built feature CSV and fits a
local `sklearn`/`xgboost` classifier on it.

Usage:
    python scripts/train_ml_filterrag.py \\
        --dataset_csv results/diagnostics/ml_filterrag_dataset_hotpotqa/features.csv \\
        --dataset hotpotqa --out_dir models/ml_filterrag
"""
import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from defense.ml_filterrag import (  # noqa: E402
    ALL_FEATURE_NAMES,
    AUXILIARY_FEATURE_NAMES,
    DEFAULT_FEATURE_NAMES,
    MLFilterRAGClassifier,
    VALID_MODEL_TYPES,
    assert_no_query_id_leakage,
    paper_aligned_model_type,
)

FEATURE_SETS = {
    "default": DEFAULT_FEATURE_NAMES,
    "all": DEFAULT_FEATURE_NAMES + AUXILIARY_FEATURE_NAMES,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset_csv", nargs="+", required=True,
        help="One or more feature CSV(s) written by scripts/build_ml_filterrag_dataset.py.",
    )
    parser.add_argument(
        "--dataset", default=None, choices=["hotpotqa", "msmarco", "nq"],
        help="Dataset name, used for the per-dataset paper-aligned --model_type default "
             "(Appendix C, Table VI). Inferred from the CSV's 'dataset' column if omitted "
             "and every row agrees; required otherwise.",
    )
    parser.add_argument(
        "--attack_methods", nargs="+", default=None,
        help="Restrict training/eval to these attack_method value(s) (must be stated "
             "explicitly in the report per docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md sec 2 "
             "-- default: use every attack_method present in --dataset_csv).",
    )
    parser.add_argument(
        "--model_type", default=None, choices=list(VALID_MODEL_TYPES),
        help="Default: paper-aligned choice for --dataset (random_forest for "
             "hotpotqa/msmarco, xgboost for nq).",
    )
    parser.add_argument(
        "--allow_proxy_classifier", action="store_true",
        help="Required to train --dataset nq with --model_type random_forest (or the "
             "xgboost-unavailable fallback) -- stamps a 'proxy_classifier' label into the "
             "artifact/report rather than silently substituting.",
    )
    parser.add_argument(
        "--feature_set", default="default", choices=list(FEATURE_SETS),
        help="'default' (paper-aligned, exactly DEFAULT_FEATURE_NAMES) or 'all' "
             "(DEFAULT_FEATURE_NAMES + AUXILIARY_FEATURE_NAMES -- a repo-augmented "
             "ML-FilterRAG-top-k variant, never plain 'ML-FilterRAG'; must be reported as "
             "such).",
    )
    parser.add_argument("--threshold_default", type=float, default=0.5)
    parser.add_argument(
        "--eval_threshold", type=float, default=None,
        help="Threshold used for the test-split precision/recall/F1/confusion-matrix report "
             "(default: --threshold_default).",
    )
    parser.add_argument("--n_estimators", type=int, default=100, help="RandomForestClassifier n_estimators.")
    parser.add_argument("--out_dir", default="models/ml_filterrag")
    parser.add_argument("--report_dir", default="results/diagnostics/ml_filterrag")
    parser.add_argument("--run_name", default=None, help="Default: auto-generated from dataset/attack/model_type/timestamp.")
    return parser.parse_args()


def load_rows(csv_paths):
    import pandas as pd  # noqa: PLC0415

    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    # is_poison round-trips through CSV as the strings "True"/"False" (or
    # already-bool if pandas infers a bool dtype from a single-source CSV);
    # normalize to bool explicitly rather than relying on pandas' dtype
    # inference, which is not guaranteed stable across concatenated CSVs.
    if df["is_poison"].dtype != bool:
        df["is_poison"] = df["is_poison"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False}
        )
    return df


def resolve_dataset_name(args, df) -> str:
    if args.dataset is not None:
        return args.dataset
    unique_datasets = sorted(df["dataset"].dropna().unique().tolist())
    if len(unique_datasets) == 1:
        return unique_datasets[0]
    raise ValueError(
        f"--dataset_csv contains {len(unique_datasets)} distinct 'dataset' value(s) "
        f"{unique_datasets} -- pass --dataset explicitly to choose the paper-aligned "
        "--model_type default (Appendix C, Table VI)."
    )


def _xgboost_available() -> bool:
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("xgboost") is not None


def resolve_model_type(args, dataset_name: str) -> str:
    """Resolve the --model_type to actually train.

    If --model_type is explicit, it is used as-is (still subject to the
    proxy-labeling check in is_proxy_classifier() below). Otherwise this
    defaults to the paper-aligned choice (Appendix C, Table VI); if that
    choice is 'xgboost' and xgboost is not installed, this either falls
    back to 'random_forest' (only when --allow_proxy_classifier was
    passed) or raises a clear, actionable error -- it never silently
    substitutes.
    """
    aligned = paper_aligned_model_type(dataset_name)
    if args.model_type is not None:
        return args.model_type
    if aligned == "xgboost" and not _xgboost_available():
        if args.allow_proxy_classifier:
            print(
                f"[train_ml_filterrag] WARNING: xgboost is not installed; falling back to "
                f"model_type='random_forest' for --dataset {dataset_name!r} because "
                "--allow_proxy_classifier was passed. This is a labeled proxy result, NOT "
                "a paper-faithful NQ reproduction (Appendix C, Table VI)."
            )
            return "random_forest"
        raise ValueError(
            f"--dataset {dataset_name!r}'s paper-aligned classifier is 'xgboost' (Appendix "
            "C, Table VI), but the 'xgboost' package is not installed in this environment. "
            "Refusing to silently substitute Random Forest. Either `pip install xgboost` "
            "for a paper-faithful NQ classifier, or pass --allow_proxy_classifier to "
            "proceed with a Random Forest substitute (the artifact/report will be labeled "
            "'NQ Random-Forest proxy result')."
        )
    return aligned


def is_proxy_classifier(dataset_name: str, model_type: str) -> bool:
    """True iff this (dataset, model_type) combination deviates from the
    paper-aligned default in Appendix C, Table VI (currently only the
    NQ + random_forest substitution, since XGBoost isn't installed)."""
    try:
        aligned = paper_aligned_model_type(dataset_name)
    except ValueError:
        return False
    return model_type != aligned


def compute_metrics(y_true, y_pred, y_proba):
    """precision/recall/F1/confusion matrix always; ROC-AUC only if both
    classes are present in y_true (sklearn raises otherwise) -- returns
    `roc_auc=None` with an explanatory `roc_auc_skipped_reason` instead of
    crashing or reporting a fabricated value."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score  # noqa: PLC0415

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    roc_auc = None
    roc_auc_skipped_reason = None
    unique_classes = set(int(v) for v in y_true)
    if len(unique_classes) < 2:
        roc_auc_skipped_reason = (
            f"test split contains only class(es) {sorted(unique_classes)} -- ROC-AUC is "
            "undefined with a single class present, skipped rather than fabricated."
        )
    else:
        roc_auc = float(roc_auc_score(y_true, y_proba))

    n_test_poison = int(sum(int(v) for v in y_true))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "roc_auc_skipped_reason": roc_auc_skipped_reason,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test_rows": int(len(y_true)),
        "n_test_poison": n_test_poison,
        "n_test_clean": int(len(y_true) - n_test_poison),
    }


def write_report_md(path, *, run_name, dataset_name, model_type, proxy, feature_set_name,
                     feature_names, attack_methods, n_train_rows, n_test_rows,
                     n_train_query_ids, n_test_query_ids, metrics, model_path,
                     threshold_default, eval_threshold, n_estimators, dataset_csvs):
    status_line = (
        "**NQ Random-Forest proxy result -- NOT a paper-faithful NQ reproduction** "
        "(xgboost unavailable/not requested; --allow_proxy_classifier was set)."
        if proxy else
        f"Paper-aligned classifier choice for {dataset_name} (Appendix C, Table VI)."
    )
    feature_set_line = (
        "Paper-aligned: exactly the 4 `DEFAULT_FEATURE_NAMES`."
        if feature_set_name == "default" else
        "**Repo-augmented ML-FilterRAG-top-k variant** -- trained on "
        "`DEFAULT_FEATURE_NAMES + AUXILIARY_FEATURE_NAMES`, NOT the paper-aligned baseline."
    )
    roc_auc_line = (
        f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None
        else f"skipped ({metrics['roc_auc_skipped_reason']})"
    )
    lines = [
        f"# ML-FilterRAG-top-k Training Report: {run_name}",
        "",
        "**Status: ML-FilterRAG-top-k (MVP)** -- this repo's harness retrieves `top_k` "
        "directly, not the paper's `top-s -> filter -> top-k` pipeline. See "
        "`docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md`.",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Attack method(s): `{attack_methods}`",
        f"- Model type: `{model_type}` ({'random_forest n_estimators=' + str(n_estimators) if model_type == 'random_forest' else 'xgboost'})",
        f"- {status_line}",
        f"- Feature set (`{feature_set_name}`): {feature_set_line}",
        f"- Feature names (order matters): `{list(feature_names)}`",
        f"- Source CSV(s): `{dataset_csvs}`",
        f"- Train rows: {n_train_rows} (across {n_train_query_ids} distinct held-out-from-test query_id(s))",
        f"- Test rows: {n_test_rows} (across {n_test_query_ids} distinct held-out-from-train query_id(s))",
        f"- No GPT/API call made. No `llm.query()` call made. No live generation run.",
        "",
        "## Test-split metrics (held-out query_ids, never seen during training)",
        "",
        f"- Threshold used for precision/recall/F1/confusion-matrix: `{eval_threshold}` "
        f"(artifact's stored `threshold_default`: `{threshold_default}`)",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        f"- ROC-AUC: {roc_auc_line}",
        f"- Confusion matrix (rows=true, cols=pred; label 1 = poison/adversarial): "
        f"TN={metrics['confusion_matrix']['tn']} FP={metrics['confusion_matrix']['fp']} "
        f"FN={metrics['confusion_matrix']['fn']} TP={metrics['confusion_matrix']['tp']}",
        f"- Test rows: {metrics['n_test_rows']} ({metrics['n_test_poison']} poison / "
        f"{metrics['n_test_clean']} clean)",
        "",
        f"## Artifact",
        "",
        f"- Model path: `{model_path}`",
        "",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote training report to {path}")


def main():
    args = parse_args()
    print("[train_ml_filterrag] No GPT/API call will be made; no llm.query() call will be made.")

    df = load_rows(args.dataset_csv)
    if args.attack_methods:
        df = df[df["attack"].isin(args.attack_methods)].reset_index(drop=True)
        print(f"[train_ml_filterrag] restricted to attack_method(s) {args.attack_methods}: {len(df)} row(s)")
    attack_methods_used = sorted(df["attack"].dropna().unique().tolist())

    dataset_name = resolve_dataset_name(args, df)
    model_type = resolve_model_type(args, dataset_name)
    proxy = is_proxy_classifier(dataset_name, model_type)

    if proxy and not args.allow_proxy_classifier:
        raise ValueError(
            f"--dataset {dataset_name!r} with --model_type {model_type!r} deviates from the "
            f"paper-aligned classifier choice ({paper_aligned_model_type(dataset_name)!r}, "
            "Appendix C, Table VI) -- refusing to train a silently-substituted proxy. Pass "
            "--allow_proxy_classifier to proceed anyway (the artifact/report will be "
            "labeled 'NQ Random-Forest proxy result'), or `pip install xgboost` for a "
            "paper-faithful NQ classifier."
        )

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    if len(train_df) == 0:
        raise ValueError("No rows with split=='train' found in --dataset_csv -- nothing to train on.")
    if len(test_df) == 0:
        raise ValueError("No rows with split=='test' found in --dataset_csv -- nothing to evaluate on.")

    train_query_ids = set(train_df["query_id"].unique().tolist())
    test_query_ids = set(test_df["query_id"].unique().tolist())
    assert_no_query_id_leakage(train_query_ids, test_query_ids)

    feature_names = FEATURE_SETS[args.feature_set]
    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        raise ValueError(f"--dataset_csv is missing required feature column(s): {missing_cols}")

    X_train = train_df[list(feature_names)].to_numpy(dtype=float)
    y_train = train_df["is_poison"].astype(int).to_numpy()
    X_test = test_df[list(feature_names)].to_numpy(dtype=float)
    y_test = test_df["is_poison"].astype(int).to_numpy()

    model_kwargs = {"n_estimators": args.n_estimators} if model_type == "random_forest" else {}

    training_meta = {
        "status": "ML-FilterRAG-top-k (MVP)",
        "dataset": dataset_name,
        "attack_methods": attack_methods_used,
        "feature_set": args.feature_set,
        "proxy_classifier": bool(proxy),
        "proxy_reason": (
            f"model_type={model_type!r} deviates from paper-aligned "
            f"{paper_aligned_model_type(dataset_name)!r} for dataset={dataset_name!r} "
            "(Appendix C, Table VI); xgboost likely unavailable in this environment."
        ) if proxy else None,
        "source_csvs": list(args.dataset_csv),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_query_ids": len(train_query_ids),
        "n_test_query_ids": len(test_query_ids),
        "trained_at_unix": time.time(),
    }

    print(
        f"[train_ml_filterrag] training model_type={model_type} on {len(train_df)} row(s) "
        f"({len(train_query_ids)} query_ids), feature_set={args.feature_set} "
        f"({list(feature_names)})"
    )
    clf = MLFilterRAGClassifier(
        model_type=model_type,
        feature_names=feature_names,
        threshold_default=args.threshold_default,
        training_meta=training_meta,
        **model_kwargs,
    )
    clf.train(X_train, y_train)

    eval_threshold = args.eval_threshold if args.eval_threshold is not None else args.threshold_default
    y_proba = clf.predict_proba(X_test)
    y_pred = (y_proba >= eval_threshold).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    proxy_tag = "_proxy" if proxy else ""
    attack_tag = attack_methods_used[0] if len(attack_methods_used) == 1 else "multi_attack"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{dataset_name}_{attack_tag}_{model_type}{proxy_tag}_{timestamp}"

    model_path = os.path.join(args.out_dir, f"{run_name}.joblib")
    clf.save(model_path)
    print(f"[train_ml_filterrag] wrote model artifact to {model_path}")

    report_path = os.path.join(args.report_dir, f"{run_name}_TRAIN_REPORT.md")
    write_report_md(
        report_path,
        run_name=run_name,
        dataset_name=dataset_name,
        model_type=model_type,
        proxy=proxy,
        feature_set_name=args.feature_set,
        feature_names=feature_names,
        attack_methods=attack_methods_used,
        n_train_rows=len(train_df),
        n_test_rows=len(test_df),
        n_train_query_ids=len(train_query_ids),
        n_test_query_ids=len(test_query_ids),
        metrics=metrics,
        model_path=model_path,
        threshold_default=args.threshold_default,
        eval_threshold=eval_threshold,
        n_estimators=args.n_estimators,
        dataset_csvs=args.dataset_csv,
    )

    metrics_csv_path = os.path.join(args.report_dir, f"{run_name}_metrics.csv")
    os.makedirs(os.path.dirname(metrics_csv_path) or ".", exist_ok=True)
    import csv as _csv  # noqa: PLC0415

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in ("precision", "recall", "f1", "roc_auc"):
            writer.writerow([key, metrics[key]])
        for key, value in metrics["confusion_matrix"].items():
            writer.writerow([f"confusion_matrix_{key}", value])
        writer.writerow(["n_test_rows", metrics["n_test_rows"]])
        writer.writerow(["n_test_poison", metrics["n_test_poison"]])
        writer.writerow(["n_test_clean", metrics["n_test_clean"]])
    print(f"[train_ml_filterrag] wrote metrics CSV to {metrics_csv_path}")

    summary = {
        "run_name": run_name,
        "model_path": model_path,
        "report_path": report_path,
        "metrics_csv_path": metrics_csv_path,
        "proxy_classifier": proxy,
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
