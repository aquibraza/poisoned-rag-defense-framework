"""REGIME-B STAGE-1 TEXT-MANIFOLD REALIZATION STUDY -- driver.

Bounded "realizability bridge" between the closed Regime-B matrix-space
Stage-1 oracle (L2, offline matrix/statistic analysis, see
`scripts/run_ragdefender_regime_b_stage1_oracle_v2.py`) and the actual
Stella text-embedding geometry (L3, fixed-context text realization).

Implements:
  PHASE 0  -- same-session baseline re-encoding + reproduction check.
  PHASE 1  -- freeze oracle targets (delegated to
              `scripts/build_regime_b_rewrite_bank.py`, which reads the
              frozen V2 winner CSV -- this script re-derives the same
              targets for cross-checking, never a different selection).
  PHASE 2  -- actual Stella delta-vector + oracle-alignment metrics.
  PHASE 4  -- semantic-preservation checks (rule-based + MiniLM if cached).
  PHASE 5  -- fixed-context L3 realization + Stage-1 recomputation.
  PHASE 6  -- unchanged Stage-2 check for every FULL L3 realization.
  PHASE 7  -- predeclared Round-2 (R4/R5) for 0/3-FULL queries only.
  PHASE 9  -- ground-truth label attach (poison/clean), AFTER selection.

Writes ONLY to `results/diagnostics/ragdefender_regime_b_text_realization/`.
Never touches `results/diagnostics/ragdefender_expanded_baseline/` or
`results/diagnostics/ragdefender_regime_b_stage1_oracle/` (read-only).

Zero retrieval, zero generation, zero E1/CORAL/MMD, zero external LLM/API
calls. Uses ONLY the already-cached `dunzhang/stella_en_1.5B_v5` and (if
cached) `paraphrase-MiniLM-L6-v2` -- both loaded with
HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE to guarantee no network fetch.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from defense import defense_runner, ragdefender_internals as ri  # noqa: E402
import ragdefender_regime_b_stage1_oracle_lib as blib  # noqa: E402
import ragdefender_regime_b_text_realization_lib as tlib  # noqa: E402

BASELINE_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_baseline"
ORACLE_DIR = REPO_ROOT / "results/diagnostics/ragdefender_regime_b_stage1_oracle"
OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_regime_b_text_realization"
REQUESTED_DEVICE = "cpu"


class TextRealizationStopCondition(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# PHASE 0 -- load frozen population + same-session Stella baseline.
# ---------------------------------------------------------------------------

def load_frozen_failures() -> List[dict]:
    per_query_csv = BASELINE_DIR / "expanded_baseline_per_query.csv"
    with open(per_query_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["regime"] == "B_AT_CEILING"]
    with open(BASELINE_DIR / "recovered_contexts.json") as f:
        contexts_by_id = {c["query_id"]: c for c in json.load(f)}
    with open(ORACLE_DIR / "regime_b_matrix_winners_v2.csv") as f:
        winners_by_id = {r["query_id"]: r for r in csv.DictReader(f)}
    with open(ORACLE_DIR / "regime_b_boundary_per_query.csv") as f:
        boundary_by_id = {r["query_id"]: r for r in csv.DictReader(f)}

    failures = [r for r in rows if r["zero_residual_poison_success"] == "False"]
    if len(failures) != 14:
        raise TextRealizationStopCondition(f"Expected 14 Regime-B failures, found {len(failures)}.")

    cases = []
    for row in failures:
        qid = row["query_id"]
        ctx = contexts_by_id[qid]
        winner = winners_by_id[qid]
        boundary = boundary_by_id[qid]
        cases.append(
            {
                "query_id": qid,
                "texts": list(ctx["texts"]),
                "is_poison": list(ctx["is_poison"]),
                "m_poison": ctx["m_poison"],
                "candidate_index": int(winner["psd_valid_1e8_winner_candidate_index"]),
                "oracle_mode": winner["psd_valid_1e8_winner_mode"],
                "oracle_alpha": float(winner["psd_valid_1e8_winner_alpha"]),
                "psd_min_eigenvalue": float(winner["psd_valid_1e8_winner_min_eigenvalue"]),
                "mechanism": "median-limited" if boundary["binding_classification"] == "A. MEDIAN-LIMITED" else "mean-gated",
                "historical_n_adv": int(row["n_adv"]),
            }
        )
    return cases


def load_stella_model():
    cfg = defense_runner.DefenseConfig(ragdefender_version="paper", device=REQUESTED_DEVICE)
    s_model = defense_runner._get_s_model(cfg)  # noqa: SLF001
    _, st_util = defense_runner._lazy_st()  # noqa: SLF001
    actual_device = str(s_model.device)
    if actual_device != REQUESTED_DEVICE:
        raise TextRealizationStopCondition(
            f"Requested device {REQUESTED_DEVICE!r} != actual device {actual_device!r}."
        )
    return s_model, st_util, actual_device


def encode_matrix(s_model, st_util, texts: List[str]) -> np.ndarray:
    import torch as _torch

    embeddings = s_model.encode(texts, convert_to_tensor=True)
    if not bool(_torch.isfinite(embeddings).all()):
        raise TextRealizationStopCondition("Non-finite Stella embeddings observed.")
    matrix = st_util.cos_sim(embeddings, embeddings).cpu().numpy().astype(np.float64)
    return matrix


def environment_record(actual_device: str) -> dict:
    import sentence_transformers
    import torch
    import transformers

    return {
        "stella_model_id": "dunzhang/stella_en_1.5B_v5",
        "stella_revision": "same as production ragdefender_paper preset (no pinned revision override)",
        "transformers_version": transformers.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "requested_device": REQUESTED_DEVICE,
        "actual_device": actual_device,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
    }


def phase0_baseline(cases: List[dict], s_model, st_util) -> Dict[str, dict]:
    """Same-session re-encoding of all 14 queries' ORIGINAL 10-passage
    contexts. Returns {query_id: {"matrix": S_current, "stage1": ..., ...}}.
    STOPS (raises) on any N_adv or mechanism-classification mismatch."""
    results = {}
    for case in cases:
        qid = case["query_id"]
        t0 = time.time()
        matrix = encode_matrix(s_model, st_util, case["texts"])
        elapsed = time.time() - t0
        stage1 = ri.concentration_stage1_paper(matrix)

        if stage1.n_adv_estimated != 4:
            raise TextRealizationStopCondition(
                f"STOP: {qid} same-session N_adv={stage1.n_adv_estimated}, expected 4."
            )

        n_above_median = int(np.sum(stage1.above_median))
        n_and = stage1.n_adv_estimated
        binding = blib.classify_binding_condition(n_above_median, n_and, ceiling=5)
        expected_mechanism = "median-limited" if case["mechanism"] == "median-limited" else "mean-gated"
        actual_mechanism = "median-limited" if binding == blib.BINDING_MEDIAN_LIMITED else (
            "mean-gated" if binding == blib.BINDING_MEAN_GATED else binding
        )
        if actual_mechanism != expected_mechanism:
            raise TextRealizationStopCondition(
                f"STOP: {qid} mechanism classification changed: expected {expected_mechanism}, got {actual_mechanism} ({binding})."
            )

        mutual = None
        if case["mechanism"] == "median-limited":
            mutual = blib.mutual_median_validation_for_query(matrix, stage1.s_median)
            if not mutual["mutual_median_match"]:
                raise TextRealizationStopCondition(
                    f"STOP: {qid} lost its mutual-median boundary mechanism under same-session re-encoding."
                )

        results[qid] = {
            "matrix": matrix,
            "stage1": stage1,
            "binding": binding,
            "mutual_median": mutual,
            "encode_seconds": elapsed,
        }
        print(f"[phase0] {qid}: N_adv={stage1.n_adv_estimated} mechanism={actual_mechanism} ({elapsed:.1f}s)", flush=True)
    return results


def run_phase0(cases: List[dict]) -> None:
    s_model, st_util, actual_device = load_stella_model()
    env = environment_record(actual_device)
    print("Environment:", json.dumps(env, indent=2))

    results = phase0_baseline(cases, s_model, st_util)

    n_median = sum(1 for c in cases if c["mechanism"] == "median-limited")
    n_mean = sum(1 for c in cases if c["mechanism"] == "mean-gated")
    if n_median != 11 or n_mean != 3:
        raise TextRealizationStopCondition(f"Expected 11/3 median/mean split, found {n_median}/{n_mean}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "matrices").mkdir(exist_ok=True)
    for qid, r in results.items():
        np.save(OUTPUT_DIR / "matrices" / f"{qid}_ORIGINAL_same_session.npy", r["matrix"])

    env_path = OUTPUT_DIR / "phase0_environment.json"
    with open(env_path, "w") as f:
        json.dump(env, f, indent=2)

    summary_path = OUTPUT_DIR / "phase0_baseline_reproduction.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "n_adv_same_session", "mechanism_same_session", "mutual_median_match", "encode_seconds"])
        for qid, r in results.items():
            writer.writerow([
                qid,
                r["stage1"].n_adv_estimated,
                r["binding"],
                r["mutual_median"]["mutual_median_match"] if r["mutual_median"] else "N/A (mean-gated)",
                round(r["encode_seconds"], 2),
            ])

    print(f"PHASE 0 PASSED: 14/14 N_adv=4 reproduced, mechanism split 11/3 confirmed.")
    print(f"Wrote {env_path}")
    print(f"Wrote {summary_path}")
    return s_model, st_util, results


def load_minilm_model():
    cfg = defense_runner.DefenseConfig(ragdefender_version="legacy", device=REQUESTED_DEVICE)
    try:
        model = defense_runner._get_s_model(cfg)  # noqa: SLF001
        return model
    except Exception as exc:  # noqa: BLE001
        print(f"MiniLM unavailable locally ({exc!r}); falling back to rule-based-only semantic checks.")
        return None


def minilm_cosine(minilm_model, st_util, text_a: str, text_b: str) -> float:
    emb = minilm_model.encode([text_a, text_b], convert_to_tensor=True)
    sim = st_util.cos_sim(emb[0:1], emb[1:2])
    return float(sim.item())


# ---------------------------------------------------------------------------
# PHASE 4 -- semantic-preservation checks for every rewrite (Round-1).
# ---------------------------------------------------------------------------

def run_phase4(rewrite_rows: List[dict], minilm_model, st_util) -> List[dict]:
    out = []
    minilm_available = minilm_model is not None
    for row in rewrite_rows:
        cos = None
        if minilm_available:
            cos = minilm_cosine(minilm_model, st_util, row["original_text"], row["rewritten_text"])
        check = tlib.rule_based_semantic_check(
            row["original_text"], row["rewritten_text"], minilm_cosine=cos, minilm_available=minilm_available
        )
        merged = dict(row)
        merged["semantic_check"] = check
        out.append(merged)
        print(
            f"[phase4] {row['query_id']} {row['mutation_id']}: "
            f"len_ratio={check.length_ratio:.2f} minilm={cos} pass={check.semantic_preservation_pass}",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# PHASE 5 -- fixed-context L3 realization for every accepted rewrite.
# ---------------------------------------------------------------------------

def run_phase5(cases_by_qid: Dict[str, dict], phase0_results: Dict[str, dict], rewrite_rows: List[dict], s_model, st_util) -> List[dict]:
    out = []
    for row in rewrite_rows:
        qid = row["query_id"]
        case = cases_by_qid[qid]
        p0 = phase0_results[qid]
        idx = row["candidate_index"]
        mode = row["oracle_mode"]
        mechanism = case["mechanism"]

        texts_rewrite = list(case["texts"])
        texts_rewrite[idx] = row["rewritten_text"]

        t0 = time.time()
        matrix_rewrite = encode_matrix(s_model, st_util, texts_rewrite)
        elapsed = time.time() - t0
        stage1_rewrite = ri.concentration_stage1_paper(matrix_rewrite)

        stage1_orig = p0["stage1"]
        matrix_orig = p0["matrix"]

        delta = tlib.compute_delta_vector(matrix_orig[idx, :], matrix_rewrite[idx, :], idx)
        alignment = tlib.compute_alignment(delta, mode)

        n_above_median_orig = int(np.sum(stage1_orig.above_median))
        n_above_median_rewrite = int(np.sum(stage1_rewrite.above_median))
        gap_orig = blib.median_rank_gap_analysis(stage1_orig.s_median)
        gap_rewrite = blib.median_rank_gap_analysis(stage1_rewrite.s_median)

        median_progress = None
        mean_progress = None
        if mechanism == "median-limited":
            tie_orig = blib.identify_tied_boundary_passages(stage1_orig.s_median)
            tie_rewrite = blib.identify_tied_boundary_passages(stage1_rewrite.s_median)
            exact_tie_broken = bool(tie_orig["is_tied"]) and not bool(tie_rewrite["is_tied"])
            gap_before = gap_orig["median_gap"] or 0.0
            gap_after = gap_rewrite["median_gap"] or 0.0
            median_progress = tlib.MedianLimitedProgress(
                exact_tie_broken=exact_tie_broken,
                median_gap_became_positive=(gap_before <= 1e-12 and gap_after > 1e-12),
                n_above_median_increased=(n_above_median_rewrite > n_above_median_orig),
            )
        else:
            mgc_orig = blib.mean_gate_candidates(stage1_orig)
            if mgc_orig:
                margins_orig = [abs(stage1_orig.s_mean[i] - stage1_orig.s_bar) for i in mgc_orig]
                margins_after = [abs(stage1_rewrite.s_mean[i] - stage1_rewrite.s_bar) for i in mgc_orig]
                crossed = any(bool(stage1_rewrite.above_mean[i]) for i in mgc_orig)
                moved_toward_zero = bool(min(margins_after) < min(margins_orig))
            else:
                crossed, moved_toward_zero = False, False
            mean_progress = tlib.MeanGatedProgress(
                blocking_margin_moved_toward_zero=bool(moved_toward_zero),
                blocking_margin_crossed_zero=bool(crossed),
            )

        classification = tlib.classify_realization(
            n_adv_original=stage1_orig.n_adv_estimated,
            n_adv_rewrite=stage1_rewrite.n_adv_estimated,
            mechanism=mechanism,
            alignment=alignment,
            median_progress=median_progress,
            mean_progress=mean_progress,
        )

        result_row = dict(row)
        result_row.update(
            {
                "mechanism": mechanism,
                "n_adv_original_current": stage1_orig.n_adv_estimated,
                "n_adv_rewrite": stage1_rewrite.n_adv_estimated,
                "full_realization": classification == tlib.REALIZATION_FULL,
                "classification": classification,
                "s_bar_original": float(stage1_orig.s_bar),
                "s_bar_rewrite": float(stage1_rewrite.s_bar),
                "s_tilde_original": float(stage1_orig.s_tilde),
                "s_tilde_rewrite": float(stage1_rewrite.s_tilde),
                "candidate_s_mean_original": float(stage1_orig.s_mean[idx]),
                "candidate_s_mean_rewrite": float(stage1_rewrite.s_mean[idx]),
                "candidate_s_median_original": float(stage1_orig.s_median[idx]),
                "candidate_s_median_rewrite": float(stage1_rewrite.s_median[idx]),
                "n_above_median_original": n_above_median_orig,
                "n_above_median_rewrite": n_above_median_rewrite,
                "median_gap_original": gap_orig["median_gap"],
                "median_gap_rewrite": gap_rewrite["median_gap"],
                "rank5_equals_rank6_original": gap_orig["rank5_equals_rank6"],
                "rank5_equals_rank6_rewrite": gap_rewrite["rank5_equals_rank6"],
                "mean_signed_alignment": alignment.mean_signed_alignment,
                "median_signed_alignment": alignment.median_signed_alignment,
                "fraction_entries_in_oracle_direction": alignment.fraction_entries_in_oracle_direction,
                "max_delta": alignment.max_delta,
                "min_delta": alignment.min_delta,
                "cosine_alignment": alignment.cosine_alignment,
                "fitted_beta": alignment.fitted_beta,
                "oracle_profile_residual": alignment.oracle_profile_residual,
                "delta_vector": "|".join(f"{v:.8f}" for v in delta),
                "oracle_alpha": case["oracle_alpha"],
                "median_progress_exact_tie_broken": median_progress.exact_tie_broken if median_progress else None,
                "median_progress_gap_became_positive": median_progress.median_gap_became_positive if median_progress else None,
                "median_progress_n_above_median_increased": median_progress.n_above_median_increased if median_progress else None,
                "mean_progress_moved_toward_zero": mean_progress.blocking_margin_moved_toward_zero if mean_progress else None,
                "mean_progress_crossed_zero": mean_progress.blocking_margin_crossed_zero if mean_progress else None,
                "encode_seconds": elapsed,
            }
        )
        out.append(result_row)
        np.save(OUTPUT_DIR / "matrices" / f"{qid}_{row['mutation_id']}.npy", matrix_rewrite)
        print(
            f"[phase5] {qid} {row['mutation_id']}: N_adv {stage1_orig.n_adv_estimated}->{stage1_rewrite.n_adv_estimated} "
            f"class={classification} mean_align={alignment.mean_signed_alignment:.4f} ({elapsed:.1f}s)",
            flush=True,
        )
    return out


def write_variant_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = [k for k in rows[0].keys() if k != "semantic_check"]
    if "semantic_check" in rows[0]:
        sc_fields = list(rows[0]["semantic_check"].__dict__.keys())
        fieldnames = fieldnames + [f"sc_{f}" for f in sc_fields]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {k: v for k, v in row.items() if k != "semantic_check"}
            if "semantic_check" in row:
                for k, v in row["semantic_check"].__dict__.items():
                    flat[f"sc_{k}"] = v
            writer.writerow(flat)


if __name__ == "__main__":
    cases = load_frozen_failures()
    cases_by_qid = {c["query_id"]: c for c in cases}

    s_model, st_util, phase0_results = run_phase0(cases)
    minilm_model = load_minilm_model()

    bank_path = OUTPUT_DIR / "rewrite_bank.jsonl"
    with open(bank_path) as f:
        round1_rows = [json.loads(line) for line in f]

    phase4_rows = run_phase4(round1_rows, minilm_model, st_util)
    phase5_rows = run_phase5(cases_by_qid, phase0_results, phase4_rows, s_model, st_util)

    write_variant_csv(phase5_rows, OUTPUT_DIR / "regime_b_text_realization_per_variant_round1.csv")
    print(f"Wrote {OUTPUT_DIR / 'regime_b_text_realization_per_variant_round1.csv'}")

    n_full = sum(1 for r in phase5_rows if r["full_realization"])
    print(f"ROUND 1 SUMMARY: {n_full}/{len(phase5_rows)} variants FULL realization.")
    by_query: Dict[str, int] = {}
    for r in phase5_rows:
        by_query.setdefault(r["query_id"], 0)
        if r["full_realization"]:
            by_query[r["query_id"]] += 1
    zero_full_queries = [qid for qid, n in by_query.items() if n == 0]
    print(f"Queries with 0/3 FULL realization (Round-2 eligible): {len(zero_full_queries)}")
    for qid in zero_full_queries:
        print(f"  {qid}")

    with open(OUTPUT_DIR / "phase5_round2_eligible_queries.json", "w") as f:
        json.dump(zero_full_queries, f, indent=2)
