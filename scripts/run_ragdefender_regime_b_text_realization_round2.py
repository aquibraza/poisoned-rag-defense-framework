"""PHASE 7 (continued) -- Round-2 (R4/R5) fixed-context L3 realization.

Reuses the same Phase 0 same-session original matrices (already written
by `run_ragdefender_regime_b_text_realization.py` under
`matrices/{query_id}_ORIGINAL_same_session.npy`) -- does NOT re-encode the
originals a second time, so Round-2 deltas are computed against the exact
same original embeddings Round 1 used (same-session discipline).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np

import run_ragdefender_regime_b_text_realization as drv  # noqa: E402
from defense import ragdefender_internals as ri  # noqa: E402

OUTPUT_DIR = drv.OUTPUT_DIR


def main() -> None:
    cases = drv.load_frozen_failures()
    cases_by_qid = {c["query_id"]: c for c in cases}

    eligible = json.load(open(OUTPUT_DIR / "phase5_round2_eligible_queries.json"))

    phase0_results = {}
    for qid in eligible:
        matrix = np.load(OUTPUT_DIR / "matrices" / f"{qid}_ORIGINAL_same_session.npy")
        stage1 = ri.concentration_stage1_paper(matrix)
        phase0_results[qid] = {"matrix": matrix, "stage1": stage1}

    s_model, st_util, _ = drv.load_stella_model()
    minilm_model = drv.load_minilm_model()

    with open(OUTPUT_DIR / "rewrite_bank_round2.jsonl") as f:
        round2_rows = [json.loads(line) for line in f]

    phase4_rows = drv.run_phase4(round2_rows, minilm_model, st_util)
    phase5_rows = drv.run_phase5(cases_by_qid, phase0_results, phase4_rows, s_model, st_util)

    drv.write_variant_csv(phase5_rows, OUTPUT_DIR / "regime_b_text_realization_per_variant_round2.csv")
    print(f"Wrote {OUTPUT_DIR / 'regime_b_text_realization_per_variant_round2.csv'}")

    n_full = sum(1 for r in phase5_rows if r["full_realization"])
    print(f"ROUND 2 SUMMARY: {n_full}/{len(phase5_rows)} variants FULL realization.")


if __name__ == "__main__":
    main()
