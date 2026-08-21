"""PHASE 6 + PHASE 9 -- unchanged Stage-2 check for every FULL L3
realization, and ground-truth threat-model label attach.

Reads the already-written Round-1/Round-2 per-variant CSVs and the
already-saved `matrices/{query_id}_{mutation_id}.npy` rewrite matrices
(no re-encoding). For every row with `full_realization == True`, reruns
the UNCHANGED production `stage2_pair_frequency(matrix, n_adv=5, p=2.0)`
and records removal outcomes (Phase 6), then attaches each target's
already-known `is_poison` ground-truth label (Phase 9) -- purely for
post-hoc interpretation, never for selection (selection happened in
Phase 1, before any text was written).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np

from defense import ragdefender_internals as ri  # noqa: E402
import ragdefender_regime_b_text_realization_lib as tlib  # noqa: E402
import run_ragdefender_regime_b_text_realization as drv  # noqa: E402

OUTPUT_DIR = drv.OUTPUT_DIR


def load_all_variant_rows():
    rows = []
    for name in ("regime_b_text_realization_per_variant_round1.csv", "regime_b_text_realization_per_variant_round2.csv"):
        path = OUTPUT_DIR / name
        if not path.exists():
            continue
        with open(path) as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def main() -> None:
    cases = drv.load_frozen_failures()
    cases_by_qid = {c["query_id"]: c for c in cases}

    all_rows = load_all_variant_rows()
    full_rows = [r for r in all_rows if r["full_realization"] == "True"]
    print(f"Found {len(full_rows)} FULL L3 realizations across both rounds.")

    stage2_rows = []
    for row in full_rows:
        qid = row["query_id"]
        mutation_id = row["mutation_id"]
        case = cases_by_qid[qid]
        idx = case["candidate_index"]
        is_poison_arr = case["is_poison"]
        m_poison = case["m_poison"]
        candidate_is_poison = bool(is_poison_arr[idx])

        matrix_path = OUTPUT_DIR / "matrices" / f"{qid}_{mutation_id}.npy"
        matrix = np.load(matrix_path)
        stage2 = ri.stage2_pair_frequency(matrix, n_adv=5, p=2.0)

        removed_indices = stage2.selected_indices
        removed_poison = sum(1 for i in removed_indices if is_poison_arr[i])
        removed_clean = sum(1 for i in removed_indices if not is_poison_arr[i])
        residual_poison = m_poison - removed_poison

        stage2_label = tlib.classify_stage2_outcome(removed_poison, removed_clean, m_poison=m_poison)
        threat_wording = tlib.threat_model_wording(candidate_is_poison)

        stage2_rows.append(
            {
                "query_id": qid,
                "mutation_id": mutation_id,
                "mechanism": row["mechanism"],
                "oracle_mode": row["oracle_mode"],
                "candidate_index": idx,
                "candidate_is_poison": candidate_is_poison,
                "threat_model_wording": threat_wording,
                "n_adv_used_for_stage2": 5,
                "removed_indices": "|".join(str(i) for i in removed_indices),
                "removed_poison": removed_poison,
                "removed_clean": removed_clean,
                "residual_poison": residual_poison,
                "m_poison": m_poison,
                "stage2_label": stage2_label,
            }
        )
        print(
            f"[phase6] {qid} {mutation_id}: removed_poison={removed_poison}/{m_poison} "
            f"removed_clean={removed_clean} label={stage2_label} candidate_is_poison={candidate_is_poison} "
            f"({threat_wording})"
        )

    out_path = OUTPUT_DIR / "regime_b_text_realization_stage2.csv"
    if stage2_rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(stage2_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stage2_rows)
        print(f"Wrote {out_path}")

    # Phase 9: full per-target threat-model label audit (all 14 targets,
    # not just FULL realizations) -- for the report's target-control table.
    label_rows = []
    for case in cases:
        idx = case["candidate_index"]
        candidate_is_poison = bool(case["is_poison"][idx])
        label_rows.append(
            {
                "query_id": case["query_id"],
                "candidate_index": idx,
                "mechanism": case["mechanism"],
                "oracle_mode": case["oracle_mode"],
                "candidate_is_poison": candidate_is_poison,
                "threat_model_wording": tlib.threat_model_wording(candidate_is_poison),
            }
        )
    label_path = OUTPUT_DIR / "regime_b_text_realization_target_label_audit.csv"
    with open(label_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(label_rows[0].keys()))
        writer.writeheader()
        writer.writerows(label_rows)
    print(f"Wrote {label_path}")

    n_poison_targets = sum(1 for r in label_rows if r["candidate_is_poison"])
    print(f"Target-control audit: {n_poison_targets}/14 targets are poison-controlled (diagnostic).")

    eligible_l4 = [r for r in full_rows if bool(cases_by_qid[r["query_id"]]["is_poison"][cases_by_qid[r["query_id"]]["candidate_index"]])]
    print(f"L4 retrieval-eligible variants (FULL + candidate_is_poison=True): {len(eligible_l4)}")
    for r in eligible_l4:
        print(f"  {r['query_id']} {r['mutation_id']}")
    with open(OUTPUT_DIR / "phase10_l4_eligible_variants.json", "w") as f:
        json.dump([{"query_id": r["query_id"], "mutation_id": r["mutation_id"]} for r in eligible_l4], f, indent=2)


if __name__ == "__main__":
    main()
