"""Historical n=42 expanded-baseline 5-query reproducibility bridge.

==========================================================================
SCOPE
==========================================================================
Uses ONLY already-frozen artifacts from
`results/diagnostics/ragdefender_expanded_baseline/`:
  - `prospective_population.csv` / `recovered_contexts.json` (STEP 3 freeze,
    read-only, NEVER re-derived or re-selected here)
  - `expanded_baseline_per_query.csv` (STEP 4 historical Stage-1/2 result)
  - `similarity/{query_id}_stella_similarity_matrix.npy` (STEP 4 historical
    Stella cosine matrix, READ-ONLY -- never written to by this script)
  - `results/diagnostics/ragdefender_expanded_gate_c/expanded_gate_c_per_query.csv`
    (historical oracle-count decomposition label, for the optional STEP 6
    recheck)

Selects exactly 5 queries via a DETERMINISTIC, pre-declared rule (see
`select_five_queries`), computed purely from frozen-population ORDER and
frozen LABELS -- never from whether re-encoding reproduces the historical
result. Re-encodes those 5 queries' exact frozen passage texts with Stella
through the production `ragdefender_paper` model-loading path, and compares
the resulting matrix/Stage-1/Stage-2/outcome against the historical
records.

Zero retrieval, zero generation, zero E1/CORAL/MMD, zero LLM/API calls.
Never overwrites `expanded_baseline_per_query.csv`,
`expanded_baseline_by_regime.csv`, any historical `.npy` matrix, any
expanded Gate-C CSV, or any Gate-B artifact -- writes ONLY
`expanded_baseline_bridge_5q.csv` under
`results/diagnostics/ragdefender_environment_bridge/`.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from defense import defense_runner, ragdefender_internals as ri  # noqa: E402

BASELINE_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_baseline"
GATE_C_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_gate_c"
OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_environment_bridge"
OUTPUT_CSV = OUTPUT_DIR / "expanded_baseline_bridge_5q.csv"

PROSPECTIVE_POPULATION_CSV = BASELINE_DIR / "prospective_population.csv"
RECOVERED_CONTEXTS_JSON = BASELINE_DIR / "recovered_contexts.json"
BASELINE_PER_QUERY_CSV = BASELINE_DIR / "expanded_baseline_per_query.csv"
GATE_C_PER_QUERY_CSV = GATE_C_DIR / "expanded_gate_c_per_query.csv"

STELLA_MODEL_NAME = "dunzhang/stella_en_1.5B_v5"
REQUESTED_DEVICE = "cpu"

STRICT_TOL = {"atol": 1e-8, "rtol": 1e-7}
LOOSE_TOL = {"atol": 1e-6, "rtol": 1e-5}


class BridgeStopCondition(RuntimeError):
    """Raised for any of the task's documented STOP conditions."""


# ---------------------------------------------------------------------------
# STEP 2 -- deterministic 5-query selection (pre-declared rule, computed
# PURELY from frozen population order + frozen labels -- never from
# reproduction outcome).
# ---------------------------------------------------------------------------

SELECTION_RULE_KEYS = ["b_success", "b_failure", "c_m6", "c_m8plus", "d_allpoison"]


def _load_frozen_order_and_labels() -> List[dict]:
    """Reads `prospective_population.csv` for frozen ORDER and regime/M,
    and `expanded_baseline_per_query.csv` for the frozen baseline success
    label -- both already-written, read-only artifacts. Returns rows in
    the frozen population's own order (verified below to match)."""
    if not PROSPECTIVE_POPULATION_CSV.exists():
        raise BridgeStopCondition(f"{PROSPECTIVE_POPULATION_CSV} not found.")
    if not BASELINE_PER_QUERY_CSV.exists():
        raise BridgeStopCondition(f"{BASELINE_PER_QUERY_CSV} not found.")

    with open(PROSPECTIVE_POPULATION_CSV) as f:
        pop_rows = list(csv.DictReader(f))
    with open(BASELINE_PER_QUERY_CSV) as f:
        baseline_rows = list(csv.DictReader(f))

    pop_order = [r["query_id"] for r in pop_rows]
    baseline_order = [r["query_id"] for r in baseline_rows]
    if pop_order != baseline_order:
        raise BridgeStopCondition(
            "prospective_population.csv and expanded_baseline_per_query.csv "
            "orderings disagree -- the frozen-order precondition this "
            "selection rule depends on does not hold."
        )

    baseline_by_id = {r["query_id"]: r for r in baseline_rows}
    merged = []
    for qid in pop_order:
        b = baseline_by_id[qid]
        merged.append(
            {
                "query_id": qid,
                "regime": b["regime"],
                "m_poison": int(b["m_poison"]),
                "zero_residual_poison_success": b["zero_residual_poison_success"] == "True",
            }
        )
    return merged


def select_five_queries(rows: List[dict]) -> Dict[str, str]:
    """Pure function of frozen-population ORDER + already-frozen LABELS
    (regime, m_poison, historical baseline success/failure) -- NEVER a
    function of whether re-encoding reproduces anything. Selection must be
    computed and recorded before any Stella call in `main()`.

    Rule (first appearance in frozen order):
      1. first Regime-B baseline SUCCESS
      2. first Regime-B baseline FAILURE
      3. first Regime-C query with M=6
      4. first Regime-C query with M>=8
      5. first Regime-D query
    """
    selected: Dict[str, str] = {}
    for row in rows:
        regime = row["regime"]
        m = row["m_poison"]
        success = row["zero_residual_poison_success"]

        if "b_success" not in selected and regime == "B_AT_CEILING" and success:
            selected["b_success"] = row["query_id"]
        if "b_failure" not in selected and regime == "B_AT_CEILING" and not success:
            selected["b_failure"] = row["query_id"]
        if "c_m6" not in selected and regime == "C_ABOVE_CEILING" and m == 6:
            selected["c_m6"] = row["query_id"]
        if "c_m8plus" not in selected and regime == "C_ABOVE_CEILING" and m >= 8:
            selected["c_m8plus"] = row["query_id"]
        if "d_allpoison" not in selected and regime == "D_ALL_POISON":
            selected["d_allpoison"] = row["query_id"]

    missing = [k for k in SELECTION_RULE_KEYS if k not in selected]
    if missing:
        raise BridgeStopCondition(f"Selection rule could not find a match for: {missing}")
    return selected


# ---------------------------------------------------------------------------
# STEP 3 -- re-encode with the CURRENT Stella environment
# ---------------------------------------------------------------------------

def _current_environment() -> dict:
    import torch as _torch
    import transformers
    import sentence_transformers

    stella_revision = None
    snapshot_root = (
        Path.home()
        / ".cache/huggingface/hub/models--dunzhang--stella_en_1.5B_v5/refs/main"
    )
    if snapshot_root.exists():
        stella_revision = snapshot_root.read_text().strip()

    return {
        "transformers_version": transformers.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": _torch.__version__,
        "stella_model_id": STELLA_MODEL_NAME,
        "stella_revision": stella_revision,
        "requested_device": REQUESTED_DEVICE,
    }


def _load_stella():
    cfg = defense_runner.DefenseConfig(ragdefender_version="paper", device=REQUESTED_DEVICE)
    s_model = defense_runner._get_s_model(cfg)  # noqa: SLF001
    actual_device = str(s_model.device)
    if actual_device != REQUESTED_DEVICE:
        raise BridgeStopCondition(
            f"Requested device {REQUESTED_DEVICE!r} != actual device {actual_device!r}."
        )
    _, st_util = defense_runner._lazy_st()  # noqa: SLF001
    return s_model, st_util, actual_device


def _load_frozen_context(query_id: str) -> dict:
    with open(RECOVERED_CONTEXTS_JSON) as f:
        contexts = json.load(f)
    ctx = next((c for c in contexts if c["query_id"] == query_id), None)
    if ctx is None:
        raise BridgeStopCondition(f"{query_id}: not found in {RECOVERED_CONTEXTS_JSON}.")
    return ctx


def _load_historical_matrix(query_id: str) -> np.ndarray:
    path = BASELINE_DIR / "similarity" / f"{query_id}_stella_similarity_matrix.npy"
    if not path.exists():
        raise BridgeStopCondition(f"{query_id}: historical matrix missing at {path}.")
    # Read-only load -- this function must never write to `path`.
    return np.load(path)


def _classify_pair(i: int, j: int, is_poison: np.ndarray) -> str:
    pi, pj = bool(is_poison[i]), bool(is_poison[j])
    if pi and pj:
        return "PP"
    if pi != pj:
        return "PC"
    return "CC"


def _pair_composition(top_pairs, is_poison: np.ndarray) -> Dict[str, int]:
    counts = {"PP": 0, "PC": 0, "CC": 0}
    for i, j, _sim in top_pairs:
        counts[_classify_pair(i, j, is_poison)] += 1
    return counts


# ---------------------------------------------------------------------------
# STEP 4 -- comparison metrics
# ---------------------------------------------------------------------------

def _stage1_stage2_from_matrix(matrix: np.ndarray, is_poison: np.ndarray) -> dict:
    stage1 = ri.concentration_stage1_paper(matrix)
    stage2 = ri.stage2_pair_frequency(matrix, n_adv=stage1.n_adv_estimated, p=2.0)
    removed = set(stage2.selected_indices)
    removed_poison = sum(1 for i in removed if is_poison[i])
    removed_clean = sum(1 for i in removed if not is_poison[i])
    n_poison = int(is_poison.sum())
    residual_poison = n_poison - removed_poison
    return {
        "s_bar": float(stage1.s_bar),
        "s_tilde": float(stage1.s_tilde),
        "n_adv": int(stage1.n_adv_estimated),
        "and_flags": tuple(bool(x) for x in stage1.adv_flag),
        "n_pairs": stage2.n_pairs,
        "top_pairs": stage2.top_pairs,
        "frequency_scores": stage2.frequency_scores,
        "removed_indices": tuple(sorted(removed)),
        "removed_poison": removed_poison,
        "removed_clean": removed_clean,
        "residual_poison": residual_poison,
        "zero_residual_poison_success": residual_poison == 0,
    }


def _oracle_from_matrix(matrix: np.ndarray, m_poison: int, is_poison: np.ndarray) -> dict:
    """STEP 6: oracle-count Stage 2 -- supplied count is `m_poison` ONLY
    (the true retrieved poison COUNT, an integer), never the poison
    identity/index labels themselves; `is_poison` is used ONLY afterwards,
    read-only, to score the already-fixed `stage2.selected_indices`
    output. This mirrors `scripts/run_ragdefender_expanded_gate_c.py`
    exactly."""
    stage2 = ri.stage2_pair_frequency(matrix, n_adv=m_poison, p=2.0)
    removed = set(stage2.selected_indices)
    removed_poison = sum(1 for i in removed if is_poison[i])
    removed_clean = sum(1 for i in removed if not is_poison[i])
    n_poison = int(is_poison.sum())
    residual_poison = n_poison - removed_poison
    return {
        "removed_poison": removed_poison,
        "removed_clean": removed_clean,
        "residual_poison": residual_poison,
    }


def _classify_decomposition_label(estimated: dict, oracle: dict, count_error: int) -> str:
    """Byte-for-byte mirror of
    `scripts/run_ragdefender_expanded_gate_c.py::_classify_decomposition`
    (priority-ordered decision tree over exactly the four allowed labels)."""
    if estimated["residual_poison"] == 0:
        return "D. BASELINE SUCCESS"
    if count_error == 0:
        return "C. IDENTIFICATION LIMITED"
    if oracle["residual_poison"] == 0 and oracle["removed_clean"] == 0:
        return "A. COUNT-LIMITED"
    return "B. COUNT + IDENTIFICATION LIMITED"


def _matrix_comparison(historical: np.ndarray, current: np.ndarray) -> dict:
    diff = np.abs(historical - current)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "frobenius_norm_diff": float(np.linalg.norm(historical - current)),
        "allclose_strict": bool(np.allclose(historical, current, **STRICT_TOL)),
        "allclose_loose": bool(np.allclose(historical, current, **LOOSE_TOL)),
    }


def _classify_query(matrix_cmp: dict, hist_s1s2: dict, cur_s1s2: dict) -> str:
    decision_identical = (
        hist_s1s2["n_adv"] == cur_s1s2["n_adv"]
        and hist_s1s2["removed_indices"] == cur_s1s2["removed_indices"]
        and hist_s1s2["zero_residual_poison_success"] == cur_s1s2["zero_residual_poison_success"]
    )
    if matrix_cmp["max_abs_diff"] == 0.0 and decision_identical:
        return "A. BYTE/NUMERICALLY IDENTICAL"
    if decision_identical:
        return "B. NUMERIC DRIFT, DECISION-STABLE"
    return "C. DECISION DRIFT"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bridge() -> List[dict]:
    if OUTPUT_CSV.exists():
        raise BridgeStopCondition(f"Refusing to overwrite existing output: {OUTPUT_CSV}")

    frozen_rows = _load_frozen_order_and_labels()
    selected = select_five_queries(frozen_rows)  # computed BEFORE any Stella call
    print("Selected 5 queries (pre-declared rule, before any encoding):")
    for key in SELECTION_RULE_KEYS:
        print(f"  {key}: {selected[key]}")

    with open(GATE_C_PER_QUERY_CSV) as f:
        gate_c_by_id = {r["query_id"]: r for r in csv.DictReader(f)}

    env = _current_environment()
    print("Current environment:", env)

    s_model, st_util, actual_device = _load_stella()

    rows = []
    for role in SELECTION_RULE_KEYS:
        query_id = selected[role]
        ctx = _load_frozen_context(query_id)
        texts = ctx["texts"]
        is_poison = np.array(ctx["is_poison"], dtype=bool)
        m_poison = int(is_poison.sum())

        historical_matrix = _load_historical_matrix(query_id)

        import torch as _torch

        embeddings = s_model.encode(texts, convert_to_tensor=True)
        if not bool(_torch.isfinite(embeddings).all()):
            raise BridgeStopCondition(f"{query_id}: non-finite Stella embeddings observed.")
        current_matrix = st_util.cos_sim(embeddings, embeddings).cpu().numpy().astype(np.float64)

        matrix_cmp = _matrix_comparison(historical_matrix, current_matrix)
        hist_s1s2 = _stage1_stage2_from_matrix(historical_matrix, is_poison)
        cur_s1s2 = _stage1_stage2_from_matrix(current_matrix, is_poison)

        hist_comp = _pair_composition(hist_s1s2["top_pairs"], is_poison)
        cur_comp = _pair_composition(cur_s1s2["top_pairs"], is_poison)

        classification = _classify_query(matrix_cmp, hist_s1s2, cur_s1s2)

        # STEP 6: optional oracle-count decomposition recheck (count only).
        hist_oracle = _oracle_from_matrix(historical_matrix, m_poison, is_poison)
        cur_oracle = _oracle_from_matrix(current_matrix, m_poison, is_poison)
        hist_count_error = hist_s1s2["n_adv"] - m_poison
        cur_count_error = cur_s1s2["n_adv"] - m_poison
        hist_label_recomputed = _classify_decomposition_label(hist_s1s2, hist_oracle, hist_count_error)
        cur_label_recomputed = _classify_decomposition_label(cur_s1s2, cur_oracle, cur_count_error)
        historical_label_on_disk = gate_c_by_id.get(query_id, {}).get("decomposition_label")

        rows.append(
            {
                "role": role,
                "query_id": query_id,
                "regime": ctx.get("regime") or next(r["regime"] for r in frozen_rows if r["query_id"] == query_id),
                "m_poison": m_poison,
                "max_abs_diff": matrix_cmp["max_abs_diff"],
                "mean_abs_diff": matrix_cmp["mean_abs_diff"],
                "frobenius_norm_diff": matrix_cmp["frobenius_norm_diff"],
                "allclose_strict_1e8_1e7": matrix_cmp["allclose_strict"],
                "allclose_loose_1e6_1e5": matrix_cmp["allclose_loose"],
                "hist_s_bar": hist_s1s2["s_bar"],
                "cur_s_bar": cur_s1s2["s_bar"],
                "hist_s_tilde": hist_s1s2["s_tilde"],
                "cur_s_tilde": cur_s1s2["s_tilde"],
                "hist_n_adv": hist_s1s2["n_adv"],
                "cur_n_adv": cur_s1s2["n_adv"],
                "n_adv_changed": hist_s1s2["n_adv"] != cur_s1s2["n_adv"],
                "hist_and_flags": "|".join(str(int(x)) for x in hist_s1s2["and_flags"]),
                "cur_and_flags": "|".join(str(int(x)) for x in cur_s1s2["and_flags"]),
                "hist_count_error": hist_s1s2["n_adv"] - m_poison,
                "cur_count_error": cur_s1s2["n_adv"] - m_poison,
                "hist_n_pairs": hist_s1s2["n_pairs"],
                "cur_n_pairs": cur_s1s2["n_pairs"],
                "hist_pp_pc_cc": f"{hist_comp['PP']}|{hist_comp['PC']}|{hist_comp['CC']}",
                "cur_pp_pc_cc": f"{cur_comp['PP']}|{cur_comp['PC']}|{cur_comp['CC']}",
                "hist_removed_indices": "|".join(map(str, hist_s1s2["removed_indices"])),
                "cur_removed_indices": "|".join(map(str, cur_s1s2["removed_indices"])),
                "removed_set_changed": hist_s1s2["removed_indices"] != cur_s1s2["removed_indices"],
                "hist_removed_poison": hist_s1s2["removed_poison"],
                "cur_removed_poison": cur_s1s2["removed_poison"],
                "hist_removed_clean": hist_s1s2["removed_clean"],
                "cur_removed_clean": cur_s1s2["removed_clean"],
                "hist_residual_poison": hist_s1s2["residual_poison"],
                "cur_residual_poison": cur_s1s2["residual_poison"],
                "hist_zero_residual_poison_success": hist_s1s2["zero_residual_poison_success"],
                "cur_zero_residual_poison_success": cur_s1s2["zero_residual_poison_success"],
                "baseline_outcome_changed": (
                    hist_s1s2["zero_residual_poison_success"] != cur_s1s2["zero_residual_poison_success"]
                ),
                "classification": classification,
                "gate_c_label_on_disk": historical_label_on_disk,
                "hist_label_recomputed_from_hist_matrix": hist_label_recomputed,
                "cur_label_recomputed_from_cur_matrix": cur_label_recomputed,
                "gate_c_label_changed": hist_label_recomputed != cur_label_recomputed,
                "gate_c_label_matches_on_disk_record": hist_label_recomputed == historical_label_on_disk,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")
    for row in rows:
        print(
            f"  {row['role']:12s} {row['query_id']}: max_abs_diff={row['max_abs_diff']:.3e} "
            f"classification={row['classification']} gate_c_label_changed={row['gate_c_label_changed']}"
        )
    return rows


if __name__ == "__main__":
    run_bridge()
