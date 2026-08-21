"""PHASE 8 -- aggregate the primary L3 results across Round 1 + Round 2.

Reads the two per-variant CSVs already written by the driver
(`regime_b_text_realization_per_variant_round1.csv`,
`..._round2.csv`), merges them into a single
`regime_b_text_realization_per_variant.csv`, and derives:

  - `regime_b_text_realization_per_query.csv` -- one row per one of the
    14 frozen failures: best classification achieved across all attempted
    variants (max 5), whether >=1 FULL realization occurred, mechanism,
    oracle mode, oracle alpha, variants attempted.

  - `regime_b_text_realization_alignment.csv` -- Phase 2 oracle-alignment
    diagnostics aggregated per query (mean/median across that query's
    variants) plus the single frozen oracle_alpha, for the
    alpha-vs-success relationship table in the report.

Pure post-processing over already-written CSVs -- no Stella, no
retrieval, no new text generation.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_regime_b_text_realization"

FULL = "A. FULL REALIZATION"
MECH_PARTIAL = "B. MECHANISM-PARTIAL"
GEOM_ONLY = "C. GEOMETRY-ALIGNED ONLY"
NON_ALIGNED = "D. NON-ALIGNED"
RANK = {FULL: 3, MECH_PARTIAL: 2, GEOM_ONLY: 1, NON_ALIGNED: 0}


def load_variants():
    rows = []
    for name in ("regime_b_text_realization_per_variant_round1.csv", "regime_b_text_realization_per_variant_round2.csv"):
        path = OUTPUT_DIR / name
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                r["round"] = "1" if "round1" in name else "2"
                rows.append(r)
    return rows


def write_merged_per_variant(rows):
    out_path = OUTPUT_DIR / "regime_b_text_realization_per_variant.csv"
    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} variant rows, Round1+Round2).")
    return out_path


def build_per_query(rows):
    by_qid = {}
    for r in rows:
        by_qid.setdefault(r["query_id"], []).append(r)

    per_query_rows = []
    for qid, variants in by_qid.items():
        best = max(variants, key=lambda r: RANK[r["classification"]])
        any_full = any(r["classification"] == FULL for r in variants)
        full_variants = [r["mutation_id"] for r in variants if r["classification"] == FULL]
        per_query_rows.append(
            {
                "query_id": qid,
                "mechanism": variants[0]["mechanism"],
                "oracle_mode": variants[0]["oracle_mode"],
                "oracle_alpha": variants[0]["oracle_alpha"],
                "candidate_index": variants[0]["candidate_index"],
                "n_variants_attempted": len(variants),
                "used_round2": any(r["round"] == "2" for r in variants),
                "best_classification": best["classification"],
                "any_full_realization": any_full,
                "full_realization_mutation_ids": "|".join(full_variants),
                "n_full": len(full_variants),
                "n_mechanism_partial": sum(1 for r in variants if r["classification"] == MECH_PARTIAL),
                "n_geometry_aligned_only": sum(1 for r in variants if r["classification"] == GEOM_ONLY),
                "n_non_aligned": sum(1 for r in variants if r["classification"] == NON_ALIGNED),
            }
        )
    per_query_rows.sort(key=lambda r: (r["mechanism"], r["query_id"]))
    out_path = OUTPUT_DIR / "regime_b_text_realization_per_query.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_query_rows)
    print(f"Wrote {out_path} ({len(per_query_rows)} query rows).")
    return per_query_rows


def build_alignment_summary(rows):
    by_qid = {}
    for r in rows:
        by_qid.setdefault(r["query_id"], []).append(r)

    align_rows = []
    for qid, variants in by_qid.items():
        mean_signed = [float(r["mean_signed_alignment"]) for r in variants]
        median_signed = [float(r["median_signed_alignment"]) for r in variants]
        frac_dir = [float(r["fraction_entries_in_oracle_direction"]) for r in variants]
        cos_align = [float(r["cosine_alignment"]) for r in variants]
        fitted_beta = [float(r["fitted_beta"]) for r in variants]
        align_rows.append(
            {
                "query_id": qid,
                "mechanism": variants[0]["mechanism"],
                "oracle_mode": variants[0]["oracle_mode"],
                "oracle_alpha": float(variants[0]["oracle_alpha"]),
                "any_full_realization": any(r["classification"] == FULL for r in variants),
                "mean_of_mean_signed_alignment": statistics.mean(mean_signed),
                "mean_of_median_signed_alignment": statistics.mean(median_signed),
                "mean_fraction_entries_in_oracle_direction": statistics.mean(frac_dir),
                "mean_cosine_alignment": statistics.mean(cos_align),
                "mean_fitted_beta": statistics.mean(fitted_beta),
                "n_variants_mean_signed_positive": sum(1 for v in mean_signed if v > 0),
                "n_variants_total": len(variants),
            }
        )
    align_rows.sort(key=lambda r: (r["mechanism"], r["query_id"]))
    out_path = OUTPUT_DIR / "regime_b_text_realization_alignment.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(align_rows[0].keys()))
        writer.writeheader()
        writer.writerows(align_rows)
    print(f"Wrote {out_path} ({len(align_rows)} query rows).")
    return align_rows


def main() -> None:
    rows = load_variants()
    write_merged_per_variant(rows)
    per_query = build_per_query(rows)
    align = build_alignment_summary(rows)

    n_queries = len(per_query)
    n_median = sum(1 for r in per_query if r["mechanism"] == "median-limited")
    n_mean = sum(1 for r in per_query if r["mechanism"] == "mean-gated")
    n_full_queries = sum(1 for r in per_query if r["any_full_realization"])
    n_full_median = sum(1 for r in per_query if r["mechanism"] == "median-limited" and r["any_full_realization"])
    n_full_mean = sum(1 for r in per_query if r["mechanism"] == "mean-gated" and r["any_full_realization"])

    print(f"\n=== PHASE 8 SUMMARY ===")
    print(f"Total queries: {n_queries} (median-limited={n_median}, mean-gated={n_mean})")
    print(f"Queries with >=1 FULL realization: {n_full_queries}/{n_queries}")
    print(f"  median-limited FULL: {n_full_median}/{n_median}")
    print(f"  mean-gated FULL: {n_full_mean}/{n_mean}")

    variants = rows
    n_variants = len(variants)
    by_class = {}
    for r in variants:
        by_class[r["classification"]] = by_class.get(r["classification"], 0) + 1
    print(f"Total variants attempted: {n_variants}")
    for cls, n in sorted(by_class.items()):
        print(f"  {cls}: {n}")

    by_family = {}
    for r in variants:
        by_family.setdefault(r["mutation_id"], []).append(r["classification"])
    print("By mutation family:")
    for fam in sorted(by_family):
        classes = by_family[fam]
        n_f = sum(1 for c in classes if c == FULL)
        print(f"  {fam}: n={len(classes)} FULL={n_f}")


if __name__ == "__main__":
    main()
