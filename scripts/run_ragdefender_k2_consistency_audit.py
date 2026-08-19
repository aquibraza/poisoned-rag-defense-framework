"""Nominal HotpotQA k=2 consistency audit -- synthetic/mathematical pass.

==========================================================================
SCOPE
==========================================================================
This script performs ZERO retrieval, ZERO generation, ZERO Stella/network
calls, and ZERO E1/CORAL/MMD experiments. It is a pure mathematical/code
audit of what `ragdefender_paper` (final ACSAC 2025 paper Eq. 3) and
`ragdefender_legacy` (the authors' released `RAGDefender/artifacts/main.py
::find_num_adv`, observationally reproduced via the UNMODIFIED local
`ragdefender_internals.concentration_stage1` port) each compute on a
literal 2-element (`k=2`) symmetric similarity matrix

    [[1, s],
     [s, 1]]

for a sweep of representative `s` values, plus the two variants' Stage 2
and final-returned-context behavior at that same `k=2` input.

`ragdefender_legacy` itself is NEVER modified by this script -- it is
called read-only, exactly as Gate A/B/C called it.

==========================================================================
OUTPUTS
==========================================================================
results/diagnostics/ragdefender_k2_consistency/k2_synthetic_results.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from defense import ragdefender_internals as ri  # noqa: E402
from run_ragdefender_median_sensitivity import (  # noqa: E402
    _concentration_stage1_average_median,
)

OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_k2_consistency"
OUTPUT_CSV = OUTPUT_DIR / "k2_synthetic_results.csv"

# Representative s values requested by the task, plus a denser sweep and
# the two mathematical boundary values (-1, 1) for completeness.
REQUIRED_S_VALUES = [-0.5, 0.0, 0.3, 0.8, 0.99]
SWEEP_S_VALUES = list(np.round(np.linspace(-0.99, 0.99, 23), 4))
BOUNDARY_S_VALUES = [-1.0, 1.0]

ALL_S_VALUES = sorted(set(REQUIRED_S_VALUES + SWEEP_S_VALUES + BOUNDARY_S_VALUES))


def _row_for_s(s: float) -> dict:
    matrix = np.array([[1.0, s], [s, 1.0]])

    # --- paper Eq. (3), production median convention (lower-of-two-middle) ---
    paper = ri.concentration_stage1_paper(matrix)

    # --- paper Eq. (3), diagnostic median convention (average-of-two-middle) ---
    paper_avg_median = _concentration_stage1_average_median(matrix)

    # --- legacy (authors' released find_num_adv, observationally reproduced) ---
    legacy = ri.concentration_stage1(matrix)

    # --- Stage 2, run with each Stage-1's own n_adv ---
    stage2_paper = ri.stage2_pair_frequency(matrix, n_adv=paper.n_adv_estimated, p=2.0)
    stage2_legacy = ri.stage2_pair_frequency(matrix, n_adv=legacy.n_adv_estimated, p=2.0)

    # --- final returned context (2-passage placeholder doc list) ---
    doc_list = ["passage_0", "passage_1"]

    paper_removed = set(stage2_paper.selected_indices)
    paper_final = [d for i, d in enumerate(doc_list) if i not in paper_removed]
    # ragdefender_paper: NO restore-all fallback (see defense_runner._apply_defense_paper).

    legacy_removed = set(stage2_legacy.selected_indices)
    legacy_final = [d for i, d in enumerate(doc_list) if i not in legacy_removed]
    if not legacy_final:
        legacy_final = doc_list  # ragdefender_legacy's historical restore-all fallback.

    return {
        "s": s,
        "paper_s_mean": round(float(paper.s_mean[0]), 6),
        "paper_s_bar": round(float(paper.s_bar), 6),
        "paper_s_median": round(float(paper.s_median[0]), 6),
        "paper_s_tilde": round(float(paper.s_tilde), 6),
        "paper_above_mean": bool(paper.above_mean[0]),
        "paper_above_median": bool(paper.above_median[0]),
        "paper_n_adv": int(paper.n_adv_estimated),
        "paper_avg_median_convention_n_adv": int(paper_avg_median.n_adv_estimated),
        "legacy_avg_0": round(float(legacy.avg[0]), 6),
        "legacy_avg_1": round(float(legacy.avg[1]), 6),
        "legacy_avg_avg": round(float(legacy.avg_avg), 6),
        "legacy_median_0": round(float(legacy.median[0]), 6),
        "legacy_median_1": round(float(legacy.median[1]), 6),
        "legacy_avg_median": round(float(legacy.avg_median), 6),
        "legacy_combined_threshold": round(float(legacy.combined_threshold), 6),
        "legacy_above_avg_0": bool(legacy.above_avg[0]),
        "legacy_above_avg_1": bool(legacy.above_avg[1]),
        "legacy_above_median_0": bool(legacy.above_median[0]),
        "legacy_above_median_1": bool(legacy.above_median[1]),
        "legacy_or_count": int(legacy.raw_or_flag.sum()),
        "legacy_flipped": bool(legacy.flipped),
        "legacy_n_adv": int(legacy.n_adv_estimated),
        "stage2_paper_n_pairs": stage2_paper.n_pairs,
        "stage2_paper_selected_indices": "|".join(map(str, sorted(stage2_paper.selected_indices))),
        "stage2_legacy_n_pairs": stage2_legacy.n_pairs,
        "stage2_legacy_selected_indices": "|".join(map(str, sorted(stage2_legacy.selected_indices))),
        "final_context_paper_size": len(paper_final),
        "final_context_legacy_size": len(legacy_final),
        "final_context_paper_equals_input": paper_final == doc_list,
        "final_context_legacy_equals_input": legacy_final == doc_list,
        "final_contexts_agree_paper_vs_legacy": paper_final == legacy_final,
    }


def main() -> None:
    if OUTPUT_CSV.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {OUTPUT_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [_row_for_s(s) for s in ALL_S_VALUES]

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

    # Sanity summary printed to stdout (not written elsewhere).
    n_paper_zero = sum(1 for r in rows if r["paper_n_adv"] == 0)
    n_legacy_k = sum(1 for r in rows if r["legacy_n_adv"] == 2)
    n_final_contexts_agree = sum(1 for r in rows if r["final_contexts_agree_paper_vs_legacy"])
    print(f"paper_n_adv == 0 for {n_paper_zero}/{len(rows)} s values")
    print(f"legacy_n_adv == k(=2) for {n_legacy_k}/{len(rows)} s values")
    print(f"final contexts (paper vs legacy) agree for {n_final_contexts_agree}/{len(rows)} s values")


if __name__ == "__main__":
    main()
