"""STEP 3 -- Prospective HotpotQA k=10 population definition (freeze),
BEFORE inspecting any `ragdefender_paper` outcome.

==========================================================================
SCOPE
==========================================================================
Defines and FREEZES the query-id list for the expanded paper-faithful
baseline (STEP 4) and expanded Gate-C oracle-count decomposition (STEP 5).
Recovers each query's k=10 retrieved passage TEXT and observed
poison/clean composition (M, C) from EXISTING artifacts only -- see
`scripts/ragdefender_expanded_population_lib.py`'s module docstring for
the exact no-new-retrieval recovery mechanism.

This script NEVER calls Stella, NEVER runs `ragdefender_paper`/
`ragdefender_legacy`, and NEVER computes `N_adv` or any Stage-1/2
quantity. It exists specifically so the population can be frozen and
written to disk BEFORE any defense-outcome-dependent quantity exists,
satisfying the "do not inspect outcomes before freezing this list"
requirement. Observed context composition (M, C, rho, regime) is NOT an
"outcome" of `ragdefender_paper` -- it is a property of the ALREADY-FIXED
retrieved context, recorded here for planning/regime-design purposes
only, matching the explicit instruction that composition inspection is
required, not disallowed.

==========================================================================
ELIGIBLE POOL / EXCLUSIONS / SAMPLING RULE
==========================================================================
- Eligible pool: the 50 `target_query_ids` in
  `results/diagnostics/ml_filterrag_dataset_hotpotqa_50q/dataset_config.json`
  -- itself a pre-existing split frozen for an UNRELATED purpose
  (ML-FilterRAG train/test partitioning), long before this RAGDefender
  task, and never touched by any `ragdefender_legacy`/`ragdefender_paper`
  outcome.
- Exclusions: the 8 Gate-A/B/C queries (already-used diagnostic
  development sample; see `GATE_BC_EXCLUDED_QUERY_IDS`), leaving 42
  eligible queries.
- Sampling rule: OPTION A -- ALL 42 eligible queries are included (no
  sub-sampling; compute budget for 42 live-Stella encodings at k=10 is
  feasible, matching Gate B's own per-query cost). No random seed is
  used because no sampling was performed.

Writes to `results/diagnostics/ragdefender_expanded_baseline/` ONLY:
`PROSPECTIVE_POPULATION_FREEZE.md`, `prospective_population.csv`,
`recovered_contexts.json`. Refuses to overwrite if any already exist.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ragdefender_expanded_population_lib as poplib  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "results/diagnostics/ragdefender_expanded_baseline"
K_STRESS = 10


class PopulationFreezeStopCondition(RuntimeError):
    pass


def _check_no_overwrite(paths: List[Path]) -> None:
    existing = [p for p in paths if p.exists()]
    if existing:
        raise PopulationFreezeStopCondition(f"Refusing to overwrite existing artifact(s): {existing}")


def build_population() -> List[dict]:
    eligible_pool = poplib.load_eligible_pool()
    contexts = poplib.recover_all_contexts(eligible_pool, k=K_STRESS)
    return contexts


def write_population_csv(contexts: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query_id", "k", "m_poison", "c_clean", "rho", "ceiling", "regime", "doc_ids", "is_poison"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ctx in contexts:
            writer.writerow({
                "query_id": ctx["query_id"],
                "k": ctx["k"],
                "m_poison": ctx["m_poison"],
                "c_clean": ctx["c_clean"],
                "rho": ctx["rho"],
                "ceiling": ctx["ceiling"],
                "regime": ctx["regime"],
                "doc_ids": "|".join(ctx["doc_ids"]),
                "is_poison": "|".join(str(int(x)) for x in ctx["is_poison"]),
            })


def write_recovered_contexts_json(contexts: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(contexts, f, indent=2)


def write_freeze_report(contexts: List[dict], eligible_pool: List[str], path: Path) -> None:
    regime_counts = {}
    for ctx in contexts:
        regime_counts[ctx["regime"]] = regime_counts.get(ctx["regime"], 0) + 1

    lines: List[str] = []
    lines.append("# Prospective HotpotQA k=10 Population Freeze (STEP 3)")
    lines.append("")
    lines.append(
        "> Frozen BEFORE any `ragdefender_paper`/`ragdefender_legacy` run on these queries. This "
        "document records the eligible pool, exclusions, and sampling rule, and the OBSERVED RETRIEVED "
        "COMPOSITION (M, C, rho, regime) of each already-fixed context -- composition is a property of "
        "the existing retrieval, not a defense outcome, and its inspection here does not violate the "
        "no-outcome-based-selection requirement. No `N_adv`, Stage-2, or any other RAGDefender-derived "
        "quantity exists yet for any of these queries at the time this document was written."
    )
    lines.append("")
    lines.append("## Source split")
    lines.append("")
    lines.append(
        "- Pool source: `results/diagnostics/ml_filterrag_dataset_hotpotqa_50q/dataset_config.json` "
        "(`target_query_ids`), a pre-existing 50-query HotpotQA split frozen for an UNRELATED purpose "
        "(ML-FilterRAG train/test partitioning), never touched by any RAGDefender outcome."
    )
    lines.append(f"- Total eligible pool (`target_query_ids`): **50**.")
    lines.append("")
    lines.append("## Exclusions")
    lines.append("")
    lines.append(
        f"- Excluded: the **8** Gate-A/B/C diagnostic-development queries (already used for Gates A-C; "
        "kept as a clearly-labeled exploratory/development population, not double-counted here): "
        + ", ".join(f"`{q}`" for q in sorted(poplib.GATE_BC_EXCLUDED_QUERY_IDS))
    )
    lines.append(f"- Remaining eligible pool after exclusion: **{len(eligible_pool)}**.")
    lines.append("")
    lines.append("## Sampling rule")
    lines.append("")
    lines.append(
        f"- **OPTION A**: all {len(eligible_pool)} eligible queries are included (no sub-sampling). "
        "No random seed was used because no sampling was performed."
    )
    lines.append(
        "- The sample was NOT selected by legacy/paper RAGDefender success or failure, `N_adv` outcome, "
        "Stage-2 outcome, similarity geometry, `top_pair_pp`, or existing oracle behavior -- none of "
        "these quantities were computed for any of these queries before this freeze."
    )
    lines.append("")
    lines.append(f"## Selected query IDs (n={len(contexts)})")
    lines.append("")
    lines.append("| query_id | k | M (poison) | C (clean) | rho | floor(k/2) | regime |")
    lines.append("|---|---|---|---|---|---|---|")
    for ctx in contexts:
        lines.append(
            f"| `{ctx['query_id']}` | {ctx['k']} | {ctx['m_poison']} | {ctx['c_clean']} | "
            f"{ctx['rho']:.2f} | {ctx['ceiling']} | {ctx['regime']} |"
        )
    lines.append("")
    lines.append("## Regime distribution (observed composition, NOT an outcome)")
    lines.append("")
    for regime in ("A_BELOW_CEILING", "B_AT_CEILING", "C_ABOVE_CEILING", "D_ALL_POISON"):
        lines.append(f"- {regime}: **{regime_counts.get(regime, 0)}/{len(contexts)}**")
    lines.append("")
    if regime_counts.get("A_BELOW_CEILING", 0) == 0:
        lines.append(
            "**Note:** Regime A (below ceiling, M < floor(k/2)) has **zero** representation in this "
            "population. This is an honest property of the existing attack configuration (every query's "
            "retrieval was tested against an N=5-candidate poisoning pool, and at k=10 the retrieved "
            "poison count is empirically never below 5 = floor(10/2) in this dataset) -- it is reported "
            "as a limitation, not engineered around."
        )
        lines.append("")
    lines.append("## Text-recovery mechanism (no new retrieval)")
    lines.append("")
    lines.append(
        "See `scripts/ragdefender_expanded_population_lib.py` module docstring for the full mechanism: "
        "clean passage text is looked up (not re-retrieved) from `datasets/hotpotqa/corpus.jsonl` by the "
        "already-retrieved `doc_id`; poisoned passage text is reconstructed as "
        "`question + \".\" + adv_texts[pool_index % 5]` using the source query resolved from "
        "`pool_index // 5` into the flattened 100-query adversarial-text pool -- both purely deterministic "
        "lookups into already-computed artifacts, not a new retrieval or poisoning-generation pass."
    )
    lines.append("")
    lines.append("## Data files")
    lines.append("")
    lines.append("- `prospective_population.csv` -- one row per selected query (identity + observed "
                  "composition + doc_ids/is_poison), written BEFORE any Stella/RAGDefender run.")
    lines.append("- `recovered_contexts.json` -- the same population plus full recovered passage TEXT per "
                  "query, consumed verbatim by `scripts/run_ragdefender_expanded_baseline.py` (STEP 4) so "
                  "the baseline evaluates EXACTLY this frozen population, not a re-derived one.")
    lines.append("")
    lines.append(
        "No retrieval, generation, E1, CORAL, MMD, or LLM/API experiment was run to produce this "
        "document. No new retrieval was run."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    out_csv = OUTPUT_DIR / "prospective_population.csv"
    out_json = OUTPUT_DIR / "recovered_contexts.json"
    out_report = OUTPUT_DIR / "PROSPECTIVE_POPULATION_FREEZE.md"
    _check_no_overwrite([out_csv, out_json, out_report])

    eligible_pool = poplib.load_eligible_pool()
    contexts = build_population()

    write_population_csv(contexts, out_csv)
    write_recovered_contexts_json(contexts, out_json)
    write_freeze_report(contexts, eligible_pool, out_report)

    print(f"Population freeze complete: {len(contexts)} queries frozen (eligible pool: {len(eligible_pool)}).")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_report}")


if __name__ == "__main__":
    main()
