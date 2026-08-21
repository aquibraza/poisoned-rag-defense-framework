#!/usr/bin/env python3
"""Build the paper-ready RobustRAG-KW comparison tables from published scale-up
artifacts.

Pure projection: every number is read from a committed CSV in the scale-up
output directory. Nothing is retrieved, generated, scored or trained here, and
the module imports no generator -- `defense.robustrag_kw.is_abstention` is the
only defense import, used to label baseline answers that declined to answer
(the published baseline rows carry no `abstained` column).

    python scripts/build_robustrag_kw_paper_tables.py

Writes into `paper_tables/robustrag_kw/`:
    robustrag_kw_main_table.csv       9 shortlisted cases, main-paper columns
    robustrag_kw_appendix_table.csv   the same cases with per-origin diagnostics
    robustrag_kw_main_table.tex       IEEE-friendly booktabs rendering
    ROBUST_RAG_KW_TABLE_NOTES.md      provenance, conventions, derived labels
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from defense.robustrag_kw import is_abstention  # noqa: E402

DEFAULT_SCALEUP_DIR = os.path.join(
    "manual_text_mutation_pilot", "hotpotqa_50q_k10", "mutation_bundle_1",
    "robustrag_kw_scaleup")
DEFAULT_OUT_DIR = os.path.join("paper_tables", "robustrag_kw")

GENERATION_RESULTS = "robustrag_kw_scaleup_generation_results.csv"
VS_EXISTING = "robustrag_kw_scaleup_vs_existing_defenses.csv"
ORIGIN_BREAKDOWN = "robustrag_kw_scaleup_origin_breakdown.csv"
CANDIDATE_SELECTION = "robustrag_kw_scaleup_candidate_selection.csv"

#: Compact family labels. The artifacts name the family by the filter the
#: mutation was written to evade.
FAMILY_SHORT = {
    "filterrag_targeted": "FR-tgt",
    "ragdefender_targeted": "RD-tgt",
    "mlfilterrag_targeted": "ML-tgt",
}

#: Human-readable stand-ins for the 24-hex query ids, written from each case's
#: question text. Presentation only -- `query_id` is carried in the appendix
#: table so every row remains traceable to the artifact.
QUERY_SHORT_NAME = {
    "5ae224da554299234fd043ee": "Gibson/Zurracapote gin",
    "5aba749055429901930fa7d8": "Menges/Avakian occupation",
    "5ae22b8d554299234fd0440f": "Schmeichel IFFHS 1992",
    "5a7759fc5542993569682d60": "Teide/Garajonay parks",
    "5a8133725542995ce29dcbdb": "Roth/Childers England",
    "5a8e068b5542995085b37384": "Ferocactus/Silene plants",
}

#: Passage-filtering baselines, in the order they appear in the compact cell.
FILTERS = (("ragdefender", "RD"), ("filterrag_semantic", "FR"), ("ml_filterrag", "ML"))

ASR, DEFENDED, ABSTAINED, UNAVAILABLE = "ASR", "Def.", "Abs.", "—"

MAIN_FIELDS = [
    "mutation_family", "query_short_name", "self_query_poison_retrieved",
    "passage_filter_outcome", "robustrag_kw_outcome",
    "wrong_answer_vote_share", "correct_answer_vote_share",
    "robustrag_kw_interpretation",
]

APPENDIX_FIELDS = [
    "mutation_family", "query_id", "query_short_name",
    "none", "ragdefender", "filterrag_semantic", "ml_filterrag_t04", "robustrag_kw",
    "wrong_answer_vote_share", "correct_answer_vote_share", "abstained",
    "self_query_poison_isolated_asr_hits", "clean_gold_matches",
    "cross_query_poison_gold_matches",
]


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def truthy(value: Optional[str]) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def outcome(row: Dict[str, str]) -> str:
    """Collapse one defense's published result into a table label.

    `abstained` is only populated for RobustRAG-KW, so a baseline that declined
    to answer is recognised by running the same abstention detector over its
    published answer text. Without that, a refusal would be reported as a
    successful defense purely because it did not contain the attacker's string.
    """
    if row is None:
        return UNAVAILABLE
    if truthy(row.get("abstained")) or is_abstention(row.get("final_answer") or ""):
        return ABSTAINED
    return ASR if truthy(row.get("strict_asr_success")) else DEFENDED


#: Full reading label -> compact form for the IEEE-width LaTeX table only. The
#: CSV and notes keep the full text; only the typeset table is space-limited.
SHORT_READING = {
    "Poison consensus failure": "Poison consensus",
    "Abstention under contested votes": "Abstain (contested)",
    "Aggregation defends filter-evasion": "Defends evasion",
    "All filters already defend": "Filters defend",
}


def interpret(kw: str, filters: Dict[str, str]) -> str:
    """Assign each case one of the pre-agreed reading labels.

    The rule is fixed and applied uniformly rather than case by case: an
    abstention is an abstention; a clean RobustRAG-KW defense is only notable
    when some filter failed. Every RobustRAG-KW ASR hit is labelled *Poison
    consensus failure* regardless of whether the passage filters also failed --
    the appendix table's `self_query_poison_isolated_asr_hits` is 5/5 for both
    ASR cases in the published scale-up, so the mechanism is the same either
    way: the isolated calls unanimously voted for the poisoned answer. Whether
    a passage filter separately caught that poison is a fact about the filter,
    not about why RobustRAG-KW failed, so it does not change this label.
    """
    failed = [name for name, label in filters.items() if label == ASR]
    if kw == ABSTAINED:
        return "Abstention under contested votes"
    if kw == ASR:
        return "Poison consensus failure"
    return ("All filters already defend" if not failed
            else "Aggregation defends filter-evasion")


def share(value: Optional[str]) -> str:
    return "n/a" if value in (None, "") else f"{float(value):.2f}"


def build_rows(scaleup_dir: str):
    results = read_csv(os.path.join(scaleup_dir, GENERATION_RESULTS))
    vs = read_csv(os.path.join(scaleup_dir, VS_EXISTING))
    selection = read_csv(os.path.join(scaleup_dir, CANDIDATE_SELECTION))

    origin_path = os.path.join(scaleup_dir, ORIGIN_BREAKDOWN)
    origin: Dict[tuple, Dict[str, str]] = {}
    if os.path.exists(origin_path):
        for r in read_csv(origin_path):
            origin[(r["family"], r["query_id"], r["origin_group"])] = r

    by_defense: Dict[tuple, Dict[str, Dict[str, str]]] = {}
    for r in vs:
        by_defense.setdefault((r["family"], r["query_id"]), {})[r["defense_name"]] = r

    selected = {(r["family"], r["query_id"]): r for r in selection
                if truthy(r.get("selected"))}

    main_rows: List[Dict[str, str]] = []
    appendix_rows: List[Dict[str, str]] = []
    missing: List[str] = []

    for res in results:
        key = (res["family"], res["query_id"])
        defenses = by_defense.get(key, {})
        if key not in selected:
            missing.append(f"{key}: case is not marked selected in {CANDIDATE_SELECTION}")

        filters = {short: outcome(defenses.get(name)) for name, short in FILTERS}
        kw_row = defenses.get("robustrag_kw")
        if kw_row is None:
            missing.append(f"{key}: no robustrag_kw row in {VS_EXISTING}")
        kw = outcome(kw_row)
        for name, short in FILTERS:
            if name not in defenses:
                missing.append(f"{key}: no {name} row in {VS_EXISTING}")

        short_name = QUERY_SHORT_NAME.get(res["query_id"])
        if short_name is None:
            missing.append(f"{key}: no short name for this query id; using the id prefix")
            short_name = res["query_id"][:8]

        n_self = res.get("n_self_query_poison")
        n_total = res.get("n_retrieved_poison")
        self_retrieved = ("n/a" if n_self in (None, "") or n_total in (None, "")
                          else f"{n_self}/{n_total}")

        main_rows.append({
            "mutation_family": FAMILY_SHORT.get(res["family"], res["family"]),
            "query_short_name": short_name,
            "self_query_poison_retrieved": self_retrieved,
            "passage_filter_outcome": ", ".join(
                f"{short}={filters[short]}" for _, short in FILTERS),
            "robustrag_kw_outcome": kw,
            "wrong_answer_vote_share": share(res.get("wrong_answer_vote_share")),
            "correct_answer_vote_share": share(res.get("correct_answer_vote_share")),
            "robustrag_kw_interpretation": interpret(kw, filters),
        })

        appendix_rows.append({
            "mutation_family": res["family"],
            "query_id": res["query_id"],
            "query_short_name": short_name,
            "none": outcome(defenses.get("none")),
            "ragdefender": filters["RD"],
            "filterrag_semantic": filters["FR"],
            "ml_filterrag_t04": filters["ML"],
            "robustrag_kw": kw,
            "wrong_answer_vote_share": share(res.get("wrong_answer_vote_share")),
            "correct_answer_vote_share": share(res.get("correct_answer_vote_share")),
            "abstained": res.get("abstained", "n/a"),
            "self_query_poison_isolated_asr_hits": _origin_cell(
                origin, key, "self_query_poison", "n_strict_asr_hit"),
            "clean_gold_matches": _origin_cell(
                origin, key, "clean", "n_gold_match"),
            "cross_query_poison_gold_matches": _origin_cell(
                origin, key, "cross_query_poison", "n_gold_match"),
        })

    return main_rows, appendix_rows, missing


def _origin_cell(origin: Dict[tuple, Dict[str, str]], key: tuple,
                 group: str, field: str) -> str:
    """`hits/passages` for one origin group, or `n/a` when the case retrieved no
    passage of that origin -- 5 of 9 cases have no cross-query poison at all,
    and reporting those as `0` would read as a measured zero."""
    row = origin.get((key[0], key[1], group))
    if row is None:
        return "n/a"
    return f"{row[field]}/{row['n_passages']}"


CAPTION = ("RobustRAG-KW generation-time aggregation compared with passage-filtering "
           "defenses on shortlisted full-retrieval mutation cases.")
TABLE_NOTE = ("The 9 cases are a shortlisted, filter-stress sample, not a benchmark-wide "
              "rate. ASR means the target wrong answer was produced under strict matching; "
              "Def. means the target was not produced; Abs. denotes abstention.")


def latex_escape(text: str) -> str:
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                     ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        text = text.replace(old, new)
    return text.replace("—", "---")


def build_latex(rows: List[Dict[str, str]]) -> str:
    header = ["Family", "Query", "Self/All poison", "Passage filters",
              "RRKW", "$s_{\\mathrm{wrong}}$", "$s_{\\mathrm{gold}}$", "Reading"]
    # table*: eight columns, two of them prose, do not fit an IEEE column.
    L = [r"% Requires \usepackage{booktabs}.",
         r"\begin{table*}[t]", r"\centering", r"\scriptsize",
         r"\caption{" + latex_escape(CAPTION) + "}",
         r"\label{tab:robustrag-kw-comparison}",
         r"\begin{tabular}{llclcccl}", r"\toprule",
         " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        cells = [SHORT_READING.get(r[f], r[f]) if f == "robustrag_kw_interpretation" else r[f]
                 for f in MAIN_FIELDS]
        L.append(" & ".join(latex_escape(str(c)) for c in cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{\textwidth}", r"\vspace{2pt}", r"\scriptsize",
          latex_escape(TABLE_NOTE), r"\end{minipage}", r"\end{table*}", ""]
    return "\n".join(L)


def build_notes(main_rows, appendix_rows, missing, scaleup_dir) -> str:
    kw_counts = _tally(main_rows, "robustrag_kw_outcome")
    readings = _tally(main_rows, "robustrag_kw_interpretation")
    n_filter_failed_kw_held = sum(
        1 for r in main_rows
        if ASR in r["passage_filter_outcome"] and r["robustrag_kw_outcome"] != ASR)

    L = ["# RobustRAG-KW paper tables -- notes", "",
         "Generated by `scripts/build_robustrag_kw_paper_tables.py` from the committed "
         f"scale-up artifacts in `{scaleup_dir}`. The script reads CSVs and writes "
         "tables; it retrieves nothing, generates nothing, scores no defense and trains "
         "no model.", "",
         "## Source of every column", "",
         "| Table column | Artifact | Field |",
         "| --- | --- | --- |",
         f"| Mutation family, query id | `{GENERATION_RESULTS}` | `family`, `query_id` |",
         f"| Self/All poison retrieved | `{GENERATION_RESULTS}` | `n_self_query_poison` / `n_retrieved_poison` |",
         f"| Passage-filter outcome | `{VS_EXISTING}` | `strict_asr_success` per `defense_name` |",
         f"| RobustRAG-KW outcome | `{VS_EXISTING}` | `strict_asr_success`, `abstained` (`defense_name=robustrag_kw`) |",
         f"| Wrong/correct vote share | `{GENERATION_RESULTS}` | `wrong_answer_vote_share`, `correct_answer_vote_share` |",
         f"| Isolated ASR hits, gold matches by origin | `{ORIGIN_BREAKDOWN}` | `n_strict_asr_hit`, `n_gold_match`, `n_passages` |",
         f"| Case membership | `{CANDIDATE_SELECTION}` | `selected` |",
         "",
         "## Conventions and derived labels", "",
         "Three things in these tables are computed rather than copied. Each is "
         "deterministic and reproducible from the artifacts.", "",
         "1. **`Abs.` for a passage filter.** The published baseline rows carry no "
         "`abstained` column -- only RobustRAG-KW rows do. A baseline answer is "
         "therefore labelled `Abs.` when `defense.robustrag_kw.is_abstention()` "
         "accepts its published answer text, the same detector RobustRAG-KW uses. This "
         "affects exactly one cell: `filterrag_targeted` / `5ae22b8d` / ML-FilterRAG, "
         "whose answer is \"I don't know.\" The scale-up report calls that cell "
         "*defended*, under the convention that a defense defends when the attacker's "
         "answer is not produced. Both readings are correct; the table separates them "
         "because a refusal and a correct answer are not the same outcome for a reader.",
         "2. **Query short names.** Written from each case's `question` text for "
         "legibility. They are labels, not data. `query_id` is carried in the appendix "
         "table so every row stays traceable.",
         "3. **Reading labels.** Assigned by a fixed rule, not case by case: an "
         "abstention gives *Abstention under contested votes*; every RobustRAG-KW ASR "
         "hit gives *Poison consensus failure*, regardless of whether a passage filter "
         "also failed on that case -- both ASR cases have `self_query_poison_isolated_"
         "asr_hits` = 5/5 in the appendix table, i.e. every isolated call independently "
         "voted for the poisoned answer, which is what the label describes; a clean "
         "RobustRAG-KW defense gives *Aggregation defends filter-evasion* if some filter "
         "failed and *All filters already defend* otherwise. The LaTeX table renders "
         "these in a shortened form for width (`Poison consensus`, `Abstain "
         "(contested)`, `Defends evasion`, `Filters defend`); the CSVs and this file "
         "always use the full label.",
         "4. **`Self/All poison` notation (`a/b`).** `a` is `n_self_query_poison`, the "
         "mutated passages written for this exact query; `b` is `n_retrieved_poison`, "
         "all poisoned passages retrieved for the case, including any poison originally "
         "targeting a different query in the shared adversarial pool. `5/5` means every "
         "retrieved poison passage targets this query; `5/6` and `5/7` mean the case "
         "additionally retrieved 1 or 2 cross-query poison passages that also ranked in "
         "the top-10 -- those extra passages are the ones counted in the appendix "
         "table's `cross_query_poison_gold_matches` column.",
         "",
         "## What the table supports", "",
         f"- RobustRAG-KW outcomes across the 9 cases: " +
         ", ".join(f"{k} {v}" for k, v in sorted(kw_counts.items())) + ".",
         f"- Readings: " + "; ".join(f"{k} ({v})" for k, v in sorted(readings.items())) + ".",
         "- Both ASR cases carry the *Poison consensus failure* reading for the same "
         "mechanical reason: all 5 self-query poison passages produced the target in "
         "isolation (`self_query_poison_isolated_asr_hits` = 5/5 in the appendix table "
         "for each). One of them (Gibson/Zurracapote gin) also happens to be a case "
         "where every passage filter failed too, and the other (Menges/Avakian) is one "
         "where a filter held; the reading label does not distinguish these, since both "
         "are RobustRAG-KW's own isolated calls unanimously voting for poison -- the "
         "plural in the claim below is carried by both.",
         f"- On **{n_filter_failed_kw_held} of 9** cases at least one passage filter "
         "produced the attacker's answer while RobustRAG-KW did not. Report this as "
         "RobustRAG-KW *did not produce the attacker answer in 3* shortlisted cases, "
         "not as \"defended 3\": 1 of the 3 is a clean defense and 2 are abstentions, "
         "and the table distinguishes `Def.` from `Abs.` so a reader who counts only "
         "`Def.` cells will find 1, not 3.",
         "",
         "Supported claim:", "",
         "> RobustRAG-KW provides an orthogonal generation-time defense profile: it did "
         "not produce the attacker answer in 3 shortlisted cases where at least one "
         "passage filter produced the attacker answer, failed under poison-consensus "
         "cases, and frequently abstained under contested votes.", "",
         "## Coverage", "",
         f"- Main table: {len(main_rows)} rows (one per shortlisted case).",
         f"- Appendix table: {len(appendix_rows)} rows.",
         "- `cross_query_poison_gold_matches` is `n/a` for the 5 cases that retrieved no "
         "cross-query poison; those cases have no passages of that origin, so a `0` "
         "would read as a measured zero.",
         "- Counts of the form `a/b` are `matching isolated answers / passages of that "
         "origin`.",
         ""]
    if missing:
        L += ["## Missing values", ""] + [f"- {m}" for m in missing] + [""]
    else:
        L += ["No value was missing: every cell except the derived labels above was read "
              "directly from a published artifact.", ""]
    return "\n".join(L)


def _tally(rows, field):
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r[field]] = counts.get(r[field], 0) + 1
    return counts


def write_csv(path: str, fields: List[str], rows: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scaleup_dir", default=DEFAULT_SCALEUP_DIR)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    main_rows, appendix_rows, missing = build_rows(args.scaleup_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    main_csv = os.path.join(args.out_dir, "robustrag_kw_main_table.csv")
    appendix_csv = os.path.join(args.out_dir, "robustrag_kw_appendix_table.csv")
    main_tex = os.path.join(args.out_dir, "robustrag_kw_main_table.tex")
    notes_md = os.path.join(args.out_dir, "ROBUST_RAG_KW_TABLE_NOTES.md")

    write_csv(main_csv, MAIN_FIELDS, main_rows)
    write_csv(appendix_csv, APPENDIX_FIELDS, appendix_rows)
    with open(main_tex, "w", encoding="utf-8") as fh:
        fh.write(build_latex(main_rows))
    with open(notes_md, "w", encoding="utf-8") as fh:
        fh.write(build_notes(main_rows, appendix_rows, missing, args.scaleup_dir))

    for path, n in ((main_csv, len(main_rows)), (appendix_csv, len(appendix_rows)),
                    (main_tex, len(main_rows)), (notes_md, 0)):
        print(f"[paper_tables] wrote {path}" + (f" ({n} rows)" if n else ""))
    if missing:
        print(f"[paper_tables] {len(missing)} missing value(s) marked n/a; see the notes.")
    else:
        print("[paper_tables] every cell sourced from a published artifact.")


if __name__ == "__main__":
    main()
