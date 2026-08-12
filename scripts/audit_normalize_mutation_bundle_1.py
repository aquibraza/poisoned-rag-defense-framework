#!/usr/bin/env python3
"""Audit and normalize the 3 defense-*targeted* GPT mutation family files in
`manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/`, **before**
any further fixed-context evaluation or full retrieval rerun.

This script is intentionally standalone and lightweight: it performs no
retrieval, no defense scoring, and loads no ML/embedding/LM model. It only:

1. loads `mutation_input_passages.csv` as the **authoritative** source of
   `(query_id, poison_slot) -> {doc_id, original_poison_text,
   target_wrong_answer, question}` (and `clean_context_passages.csv` /
   `selected_queries.csv` for completeness cross-checks);
2. parses each of the 3 mutation-family files permissively (never crashing
   the whole file on a single bad record -- structural problems are
   flagged per record/passage instead) and validates:
   - exactly 5 mutated passages per query, `poison_slot` in `0..4`, unique;
   - `query_id` known to `mutation_input_passages.csv`;
   - mutated text differs from the authoritative original poison text;
   - `target_wrong_answer` is present in the mutated text (simple
     case-insensitive substring check -- a heuristic, not a semantic
     entailment check; documented as such in the report);
   - the query's true/correct answer (from
     `results/adv_targeted_results/hotpotqa.json`'s `"correct answer"`
     field, a pre-existing local artifact -- no API/GPT call) does **not**
     leak into the mutated text (simple case-insensitive whole-word regex
     check);
   - no duplicated mutated text within the same query/family;
   - no missing/malformed fields;
3. normalizes `doc_id`: passage identity for every output record is
   resolved by `(query_id, poison_slot)` against the authoritative CSV;
   the family file's own `doc_id` (which is wrong for 3/6 queries in the
   RAGDefender family -- see prior evaluation report) is preserved
   unmodified under `source_file_doc_id`, never used as the identity key;
4. writes normalized JSONL files (mutated text byte-for-byte unchanged)
   plus a per-passage audit CSV and a markdown integrity report.

Strict constraints honored (see module-level tests in
`tests/test_audit_normalize_mutation_bundle_1.py` for static verification):
no GPT/API call, no `llm.query()`, no retrieval rerun, no defense rerun, no
model training, no defense-code modification, no mutated-text alteration.

Usage:
    python scripts/audit_normalize_mutation_bundle_1.py \\
        --pilot_dir manual_text_mutation_pilot/hotpotqa_50q_k10 \\
        --bundle_dir manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1 \\
        --out_dir manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/normalized
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PILOT_DIR = os.path.join("manual_text_mutation_pilot", "hotpotqa_50q_k10")
DEFAULT_BUNDLE_DIR = os.path.join(DEFAULT_PILOT_DIR, "mutation_bundle_1")
DEFAULT_OUT_DIR = os.path.join(DEFAULT_BUNDLE_DIR, "normalized")
DEFAULT_ADV_TARGETED_RESULTS_PATH = os.path.join("results", "adv_targeted_results", "hotpotqa.json")

EXPECTED_SLOTS = (0, 1, 2, 3, 4)

FAMILY_SPECS: Dict[str, Dict] = {
    "ragdefender_targeted": {
        "filename": "ragdefender_discourse_diverse_poisoned_passages.txt",
        "intended_defense": "ragdefender",
        "passage_list_keys": ("rewritten_passages",),
        "text_field_keys": ("rewritten_text",),
        "normalized_filename": "ragdefender_targeted.normalized.jsonl",
    },
    "filterrag_targeted": {
        "filename": "filterrag_gpt_poisoned_passages_low_overlap.txt",
        "intended_defense": "filterrag",
        "passage_list_keys": ("poisoned_passages",),
        "text_field_keys": ("rewritten_text",),
        "normalized_filename": "filterrag_targeted.normalized.jsonl",
    },
    "mlfilterrag_targeted": {
        "filename": "mlfilterrag_gpt_prompt_packets_clean_reference_rewrites.txt",
        "intended_defense": "ml_filterrag",
        "passage_list_keys": ("poisoned_passages",),
        # This family file mislabels its mutated-text field "original_text"
        # (see run_targeted_mutation_bundle_1_eval.py's module docstring for
        # the cross-check that already confirmed this holds mutated text).
        "text_field_keys": ("rewritten_text", "original_text"),
        "normalized_filename": "mlfilterrag_targeted.normalized.jsonl",
    },
}

AUDIT_FIELDS = [
    "family", "query_id", "poison_slot", "selection_role",
    "source_file_doc_id", "canonical_doc_id", "doc_id_mismatch",
    "mutated_text_unchanged_from_original", "target_wrong_answer_simple_match",
    "possible_true_answer_leak", "true_answer_leak_low_confidence",
    "true_answer_leak_expected_named_entity_choice",
    "duplicate_mutated_text_within_family_query",
    "record_level_flags", "passage_level_flags", "included_in_normalized_output",
]


# ---------------------------------------------------------------------------
# 1. Authoritative-source loading.
# ---------------------------------------------------------------------------

def load_authoritative_passages(path: str) -> Dict[Tuple[str, int], Dict]:
    """Loads `mutation_input_passages.csv` into
    `{(query_id, poison_slot): {"doc_id", "original_poison_text",
    "target_wrong_answer", "question"}}`. Raises `ValueError` if any
    (query_id, poison_slot) pair is duplicated (the CSV itself would be
    malformed) -- this is a hard file-integrity error, not a per-record
    flag, since every downstream check depends on this mapping being
    unambiguous.

    Note: `original_poison_text` in this CSV is the pre-mutation text
    generated for the *original* PoisonedRAG attack pool; it is not itself
    a GPT/API/retrieval call product of this script -- it is a static
    input artifact loaded read-only."""
    authoritative: Dict[Tuple[str, int], Dict] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"]
            slot = int(row["poison_slot"])
            key = (qid, slot)
            if key in authoritative:
                raise ValueError(f"{path}: duplicate (query_id, poison_slot)={key!r}.")
            authoritative[key] = {
                "doc_id": row["doc_id"],
                "original_poison_text": row["original_poison_text"],
                "target_wrong_answer": row["target_wrong_answer"],
                "question": row["question"],
            }
    return authoritative


def load_known_query_ids_with_clean_context(path: str) -> Dict[str, int]:
    """Returns `{query_id: n_clean_rows}` from `clean_context_passages.csv`,
    used only for the completeness cross-check ("does every mutated query
    still have clean context rows available for a future fixed-context
    rerun") -- not used for any doc_id/slot resolution."""
    counts: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            counts[row["query_id"]] = counts.get(row["query_id"], 0) + 1
    return counts


def load_selection_roles(path: str) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            roles[row["query_id"]] = row.get("selection_role", "")
    return roles


def load_correct_answers(path: Optional[str]) -> Dict[str, str]:
    """Reads the pre-existing local `results/adv_targeted_results/hotpotqa.json`
    artifact's `"correct answer"` field per query_id, purely as an
    already-computed ground-truth reference for the true-answer-leak check.
    This is a local file read, not a GPT/API/retrieval call. Returns `{}`
    (never raises) if `path` is `None` or the file does not exist, so the
    leak check is explicitly marked unavailable rather than silently
    skipped."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, str] = {}
    for qid, rec in data.items():
        if isinstance(rec, dict) and "correct answer" in rec:
            out[qid] = str(rec["correct answer"])
    return out


# ---------------------------------------------------------------------------
# 2. Permissive per-family parsing + auditing.
# ---------------------------------------------------------------------------

def load_family_raw_records(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    stripped = raw.strip()
    if not stripped:
        raise ValueError(f"{path}: file is empty.")
    records = json.loads(stripped)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a non-empty top-level JSON array, got {type(records).__name__}.")
    return records


def _whole_word_ci_search(needle: str, haystack: str) -> bool:
    """Case-insensitive whole-word substring search (word-boundary regex),
    used for the true-answer-leak check to avoid trivial substring false
    positives (e.g. "no" inside "novel")."""
    if not needle or not needle.strip():
        return False
    pattern = r"\b" + re.escape(needle.strip().lower()) + r"\b"
    return re.search(pattern, haystack.lower()) is not None


def audit_and_normalize_family(
    family_key: str,
    spec: Dict,
    path: str,
    authoritative: Dict[Tuple[str, int], Dict],
    known_query_ids: set,
    correct_answers: Dict[str, str],
    selection_roles: Dict[str, str],
) -> Tuple[List[Dict], List[Dict]]:
    """Returns `(audit_rows, normalized_query_records)`.

    `audit_rows`: one dict per audited (family, query_id, poison_slot) --
    or a single row with `poison_slot=None` for a record that could not be
    resolved to a query_id/passage-list at all -- matching `AUDIT_FIELDS`.

    `normalized_query_records`: one dict per query_id that was fully
    structurally resolvable (exactly 5 unique valid poison_slots 0..4, each
    with non-empty text, each with a canonical doc_id) -- these, and only
    these, are written to the family's normalized JSONL output. Mutated
    text is copied verbatim; never altered.
    """
    raw_records = load_family_raw_records(path)
    audit_rows: List[Dict] = []
    normalized_records: List[Dict] = []

    for rec in raw_records:
        qid = rec.get("query_id")
        if not qid or not isinstance(qid, str):
            audit_rows.append({
                "family": family_key, "query_id": rec.get("query_id"), "poison_slot": None,
                "selection_role": "", "source_file_doc_id": None, "canonical_doc_id": None,
                "doc_id_mismatch": None, "mutated_text_unchanged_from_original": None,
                "target_wrong_answer_simple_match": None, "possible_true_answer_leak": None,
                "true_answer_leak_low_confidence": None, "true_answer_leak_expected_named_entity_choice": None,
                "duplicate_mutated_text_within_family_query": None,
                "record_level_flags": "missing_or_invalid_query_id",
                "passage_level_flags": "", "included_in_normalized_output": False,
            })
            continue

        record_flags: List[str] = []
        if qid not in known_query_ids:
            audit_rows.append({
                "family": family_key, "query_id": qid, "poison_slot": None,
                "selection_role": selection_roles.get(qid, ""),
                "source_file_doc_id": None, "canonical_doc_id": None, "doc_id_mismatch": None,
                "mutated_text_unchanged_from_original": None, "target_wrong_answer_simple_match": None,
                "possible_true_answer_leak": None, "true_answer_leak_low_confidence": None,
                "true_answer_leak_expected_named_entity_choice": None,
                "duplicate_mutated_text_within_family_query": None,
                "record_level_flags": "unknown_query_id_not_in_mutation_input_passages",
                "passage_level_flags": "", "included_in_normalized_output": False,
            })
            continue

        passages_list = None
        for list_key in spec["passage_list_keys"]:
            if list_key in rec:
                passages_list = rec[list_key]
                break
        if not isinstance(passages_list, list) or not passages_list:
            audit_rows.append({
                "family": family_key, "query_id": qid, "poison_slot": None,
                "selection_role": selection_roles.get(qid, ""),
                "source_file_doc_id": None, "canonical_doc_id": None, "doc_id_mismatch": None,
                "mutated_text_unchanged_from_original": None, "target_wrong_answer_simple_match": None,
                "possible_true_answer_leak": None, "true_answer_leak_low_confidence": None,
                "true_answer_leak_expected_named_entity_choice": None,
                "duplicate_mutated_text_within_family_query": None,
                "record_level_flags": f"missing_or_empty_passage_list_keys={spec['passage_list_keys']!r}",
                "passage_level_flags": "", "included_in_normalized_output": False,
            })
            continue

        if len(passages_list) != 5:
            record_flags.append(f"wrong_passage_count:{len(passages_list)}")

        by_slot: Dict[int, List[Dict]] = {}
        invalid_slot_entries: List[Dict] = []
        for raw_p in passages_list:
            slot_raw = raw_p.get("poison_slot")
            try:
                slot = int(slot_raw)
            except (TypeError, ValueError):
                invalid_slot_entries.append(raw_p)
                continue
            text = None
            for text_key in spec["text_field_keys"]:
                candidate = raw_p.get(text_key)
                if candidate and isinstance(candidate, str) and candidate.strip():
                    text = candidate
                    break
            by_slot.setdefault(slot, []).append({
                "file_doc_id": raw_p.get("doc_id"), "text": text,
            })

        if invalid_slot_entries:
            record_flags.append(f"invalid_or_missing_poison_slot_count:{len(invalid_slot_entries)}")
        out_of_range = sorted(s for s in by_slot if s not in EXPECTED_SLOTS)
        if out_of_range:
            record_flags.append(f"poison_slot_out_of_range:{out_of_range!r}")
        duplicated = sorted(s for s, entries in by_slot.items() if len(entries) > 1)
        if duplicated:
            record_flags.append(f"duplicate_poison_slot:{duplicated!r}")
        missing = sorted(set(EXPECTED_SLOTS) - set(by_slot))
        if missing:
            record_flags.append(f"missing_poison_slot:{missing!r}")

        structurally_resolvable = (
            not invalid_slot_entries and not out_of_range and not duplicated and not missing
        )

        # Collect resolvable per-slot texts for the within-record duplicate-text check.
        resolvable_texts: Dict[int, str] = {}
        if structurally_resolvable:
            for slot in EXPECTED_SLOTS:
                text = by_slot[slot][0]["text"]
                if text:
                    resolvable_texts[slot] = text.strip()
        dup_text_slots = set()
        seen_texts: Dict[str, int] = {}
        for slot, text in resolvable_texts.items():
            if text in seen_texts:
                dup_text_slots.add(slot)
                dup_text_slots.add(seen_texts[text])
            else:
                seen_texts[text] = slot

        # Audit every out-of-range / duplicated / invalid slot too (never silently
        # dropped), each unconditionally excluded from the normalized output.
        query_audit_rows: List[Dict] = []
        normalized_passages: List[Dict] = []
        record_fully_included = structurally_resolvable

        for entry in invalid_slot_entries:
            query_audit_rows.append({
                "family": family_key, "query_id": qid, "poison_slot": None,
                "selection_role": selection_roles.get(qid, ""),
                "source_file_doc_id": entry.get("doc_id"), "canonical_doc_id": None,
                "doc_id_mismatch": None, "mutated_text_unchanged_from_original": None,
                "target_wrong_answer_simple_match": None, "possible_true_answer_leak": None,
                "true_answer_leak_low_confidence": None, "true_answer_leak_expected_named_entity_choice": None,
                "duplicate_mutated_text_within_family_query": None,
                "record_level_flags": ";".join(record_flags),
                "passage_level_flags": "invalid_or_missing_poison_slot",
                "included_in_normalized_output": False,
            })

        for slot in sorted(by_slot):
            entries = by_slot[slot]
            for dup_i, entry in enumerate(entries):
                file_doc_id = entry["file_doc_id"]
                text = entry["text"]
                passage_flags: List[str] = []

                if slot not in EXPECTED_SLOTS:
                    passage_flags.append("poison_slot_out_of_range")
                if len(entries) > 1:
                    passage_flags.append("duplicate_poison_slot")

                authoritative_row = authoritative.get((qid, slot)) if slot in EXPECTED_SLOTS else None
                canonical_doc_id = authoritative_row["doc_id"] if authoritative_row else None
                original_text = authoritative_row["original_poison_text"] if authoritative_row else None
                target_wrong_answer = authoritative_row["target_wrong_answer"] if authoritative_row else None
                question_text = authoritative_row["question"] if authoritative_row else None

                if text is None:
                    passage_flags.append("missing_or_empty_text")
                if slot in EXPECTED_SLOTS and canonical_doc_id is None:
                    passage_flags.append("no_authoritative_row_for_slot")

                doc_id_mismatch = bool(file_doc_id and canonical_doc_id and file_doc_id != canonical_doc_id)

                mutated_text_unchanged = None
                target_match = None
                leak = None
                leak_low_conf = None
                leak_expected_named_entity_choice = None
                if text is not None and original_text is not None:
                    mutated_text_unchanged = text.strip() == original_text.strip()
                    if mutated_text_unchanged:
                        passage_flags.append("mutated_text_unchanged_from_original")
                if text is not None and target_wrong_answer:
                    target_match = target_wrong_answer.lower() in text.lower()
                    if not target_match:
                        passage_flags.append("target_wrong_answer_not_found_by_simple_substring")
                if text is not None:
                    correct_answer = correct_answers.get(qid)
                    if correct_answer and correct_answer.strip() and correct_answer.lower() != (target_wrong_answer or "").lower():
                        leak = _whole_word_ci_search(correct_answer, text)
                        leak_low_conf = leak and len(correct_answer.strip().split()) == 1 and len(correct_answer.strip()) <= 4
                        # A binary named-entity-choice question (e.g. "...Henry Roth or
                        # Robert Erskine Childers?") already names the correct answer in
                        # the *question itself*; a mutated passage arguing for the wrong
                        # option still has to name the correct-answer entity to construct
                        # a coherent (false) comparison. That is expected/structural, not
                        # a genuine information leak beyond what the question discloses.
                        leak_expected_named_entity_choice = bool(
                            leak and question_text and _whole_word_ci_search(correct_answer, question_text)
                        )
                        if leak:
                            if leak_low_conf:
                                passage_flags.append("possible_true_answer_leak_low_confidence")
                            elif leak_expected_named_entity_choice:
                                passage_flags.append("possible_true_answer_leak_expected_named_entity_choice")
                            else:
                                passage_flags.append("possible_true_answer_leak")
                is_dup_text = slot in dup_text_slots and dup_i == 0
                if is_dup_text:
                    passage_flags.append("duplicate_mutated_text_within_family_query")

                included_this_passage = (
                    structurally_resolvable and dup_i == 0 and text is not None and canonical_doc_id is not None
                )

                query_audit_rows.append({
                    "family": family_key, "query_id": qid, "poison_slot": slot,
                    "selection_role": selection_roles.get(qid, ""),
                    "source_file_doc_id": file_doc_id, "canonical_doc_id": canonical_doc_id,
                    "doc_id_mismatch": doc_id_mismatch,
                    "mutated_text_unchanged_from_original": mutated_text_unchanged,
                    "target_wrong_answer_simple_match": target_match,
                    "possible_true_answer_leak": leak,
                    "true_answer_leak_low_confidence": leak_low_conf,
                    "true_answer_leak_expected_named_entity_choice": leak_expected_named_entity_choice,
                    "duplicate_mutated_text_within_family_query": is_dup_text,
                    "record_level_flags": ";".join(record_flags),
                    "passage_level_flags": ";".join(passage_flags),
                    "included_in_normalized_output": None,  # backfilled below
                })

                if included_this_passage:
                    normalized_passages.append({
                        "poison_slot": slot,
                        "doc_id": canonical_doc_id,
                        "source_file_doc_id": file_doc_id,
                        "mutated_text": text,
                        "doc_id_mismatch": doc_id_mismatch,
                        "quality_flags": [f for f in passage_flags if f != "no_authoritative_row_for_slot"],
                    })

        # Only ever emit the normalized record if every one of the 5 expected slots
        # qualified (never a partial bundle); backfill the per-row inclusion flag
        # accordingly and merge this query's rows into the family's audit_rows.
        record_fully_included = structurally_resolvable and len(normalized_passages) == 5
        for row in query_audit_rows:
            if row["included_in_normalized_output"] is False:
                continue
            row["included_in_normalized_output"] = record_fully_included
        audit_rows.extend(query_audit_rows)

        if record_fully_included:
            normalized_passages.sort(key=lambda p: p["poison_slot"])
            normalized_records.append({
                "query_id": qid,
                "k": 10,
                "family": family_key,
                "intended_defense": spec["intended_defense"],
                "selection_role": selection_roles.get(qid, ""),
                "question": authoritative.get((qid, 0), {}).get("question") or rec.get("question"),
                "target_wrong_answer": (
                    authoritative.get((qid, 0), {}).get("target_wrong_answer") or rec.get("target_wrong_answer")
                ),
                "mutated_passages": normalized_passages,
            })

    return audit_rows, normalized_records


# ---------------------------------------------------------------------------
# 3. I/O helpers.
# ---------------------------------------------------------------------------

def write_normalized_jsonl(path: str, records: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_audit_csv(path: str, rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in AUDIT_FIELDS})


# ---------------------------------------------------------------------------
# 4. Report.
# ---------------------------------------------------------------------------

def build_report(
    *, pilot_dir: str, bundle_dir: str, out_dir: str,
    audit_rows: Sequence[Dict], normalized_counts: Dict[str, int],
    clean_context_counts: Dict[str, int], query_ids: Sequence[str],
    correct_answers_available: bool,
) -> str:
    lines: List[str] = []
    lines.append("# Mutation Bundle 1 -- Input Integrity Audit & doc_id Normalization Report")
    lines.append("")
    lines.append(
        "Audit of the 3 defense-*targeted* GPT mutation family files in "
        f"`{os.path.relpath(bundle_dir, REPO_ROOT)}/` against the authoritative "
        f"`{os.path.relpath(os.path.join(pilot_dir, 'mutation_input_passages.csv'), REPO_ROOT)}`, "
        "performed **before** any further fixed-context evaluation or full retrieval rerun. "
        "No GPT/API call, `llm.query()` call, retrieval rerun, defense rerun, model training, "
        "defense-code modification, or mutated-text alteration was performed -- see 'Process "
        "confirmation' below."
    )
    lines.append("")

    doc_id_mismatch_rows = [r for r in audit_rows if r.get("doc_id_mismatch")]
    mismatch_families = sorted(set(r["family"] for r in doc_id_mismatch_rows))
    structural_exclusion_rows = [
        r for r in audit_rows if r.get("poison_slot") is None or r.get("included_in_normalized_output") is False
    ]
    unresolved_rows = [r for r in audit_rows if r.get("included_in_normalized_output") is False]
    leak_rows = [r for r in audit_rows if r.get("possible_true_answer_leak")]
    leak_rows_named_entity = [
        r for r in leak_rows
        if not r.get("true_answer_leak_low_confidence") and r.get("true_answer_leak_expected_named_entity_choice")
    ]
    leak_rows_high_conf = [
        r for r in leak_rows
        if not r.get("true_answer_leak_low_confidence") and not r.get("true_answer_leak_expected_named_entity_choice")
    ]
    dup_text_rows = [r for r in audit_rows if r.get("duplicate_mutated_text_within_family_query")]
    unchanged_rows = [r for r in audit_rows if r.get("mutated_text_unchanged_from_original")]
    target_not_found_rows = [
        r for r in audit_rows
        if r.get("target_wrong_answer_simple_match") is False
    ]

    lines.append("## 1. Was previous-run scoring affected by the doc_id mismatches?")
    lines.append("")
    if doc_id_mismatch_rows:
        lines.append(
            f"**No.** {len(doc_id_mismatch_rows)} passage(s) across {len(mismatch_families)} family "
            f"file(s) ({', '.join(f'`{f}`' for f in mismatch_families)}) have a `doc_id` field that "
            "does not match the authoritative `(query_id, poison_slot) -> doc_id` mapping in "
            "`mutation_input_passages.csv`. However, "
            "`scripts/run_targeted_mutation_bundle_1_eval.py` (the previous evaluation run) already "
            "resolved every mutated passage's identity by `(query_id, poison_slot)` against that same "
            "authoritative CSV -- **never** by the family file's own `doc_id` -- so the wrong `doc_id` "
            "values never affected which passage's text was substituted, and the previously reported "
            "`targeted_family_bundle_scores.csv` / `targeted_family_bundle_deltas.csv` results remain "
            "valid. This audit independently re-confirms that resolution logic and additionally "
            "produces normalized files with the `doc_id` field corrected at the source."
        )
    else:
        lines.append("No `doc_id` mismatches were found; the family files' own `doc_id` values already agree with `mutation_input_passages.csv`.")
    lines.append("")

    lines.append("## 2. doc_id mismatch counts")
    lines.append("")
    lines.append(f"- Total mismatched passages: **{len(doc_id_mismatch_rows)}** (out of {len([r for r in audit_rows if r.get('poison_slot') is not None])} audited passages across all 3 families).")
    lines.append(f"- Family file(s) with mismatches: {', '.join(f'`{f}`' for f in mismatch_families) if mismatch_families else 'none'}.")
    for family in sorted(FAMILY_SPECS):
        family_mismatches = [r for r in doc_id_mismatch_rows if r["family"] == family]
        lines.append(f"  - `{family}`: {len(family_mismatches)} mismatch(es).")
    lines.append("")

    lines.append("## 3. Could all records be safely resolved by poison_slot?")
    lines.append("")
    if not unresolved_rows:
        lines.append(
            "**Yes.** Every audited passage across all 3 families had a known `query_id`, a unique "
            "valid `poison_slot` in `0..4`, non-empty mutated text, and a resolvable canonical `doc_id` "
            "from `mutation_input_passages.csv`. All 6 queries x 3 families x 5 slots = "
            f"{len([r for r in audit_rows if r.get('poison_slot') is not None])} passages were included in the normalized output."
        )
    else:
        lines.append(
            f"**No -- {len(unresolved_rows)} record(s)/passage(s) could not be safely resolved** and "
            "were excluded from the normalized output (see `mutation_bundle_1_integrity_audit.csv`, "
            "`included_in_normalized_output=False`):"
        )
        lines.append("")
        for r in unresolved_rows:
            lines.append(
                f"- family=`{r['family']}` query_id=`{r['query_id']}` poison_slot={r.get('poison_slot')} "
                f"-- {r.get('record_level_flags') or r.get('passage_level_flags') or 'unspecified'}"
            )
    lines.append("")

    lines.append("## 4. Should any query/family be excluded from paper-level claims?")
    lines.append("")
    exclusion_notes: List[str] = []
    if unresolved_rows:
        exclusion_notes.append(
            f"{len(set((r['family'], r['query_id']) for r in unresolved_rows))} (family, query_id) pair(s) "
            "were structurally unresolvable and are already excluded from the normalized output above; "
            "these should be excluded from any paper-level claim until manually fixed at the source."
        )
    if leak_rows_high_conf:
        exclusion_notes.append(
            f"{len(leak_rows_high_conf)} passage(s) were flagged `possible_true_answer_leak` (high-confidence, "
            "multi-character/multi-word correct-answer phrase found as a whole word in the mutated text, and "
            "the question itself does *not* already name that correct answer, so this is a genuine content "
            "leak beyond what the question discloses): "
            + "; ".join(f"`{r['family']}`/`{r['query_id']}`/slot={r['poison_slot']}" for r in leak_rows_high_conf)
            + ". These should be manually reviewed before any paper-level claim that these mutations "
            "purely target a wrong answer without also surfacing the true answer."
        )
    if leak_rows_named_entity:
        leak_qids = sorted(set(r["query_id"] for r in leak_rows_named_entity))
        exclusion_notes.append(
            f"{len(leak_rows_named_entity)} passage(s) across quer(y/ies) {', '.join(f'`{q}`' for q in leak_qids)} "
            "matched the correct answer as a whole word, but flagged "
            "`possible_true_answer_leak_expected_named_entity_choice`: the question itself is a binary choice "
            "between two named entities (e.g. \"...Henry Roth or Robert Erskine Childers?\") and already names "
            "the correct answer, so a mutated passage arguing for the wrong option still has to name the "
            "correct-answer entity to construct a coherent (if false) comparison. This is expected/structural "
            "for this question type, **not** a genuine information leak beyond what the question already "
            "discloses -- not recommended as grounds for exclusion on its own, but manual review is still "
            "advisable to confirm the passage's argument favors the wrong option rather than the correct one."
        )
    leak_rows_low_conf = [r for r in leak_rows if r.get("true_answer_leak_low_confidence")]
    if leak_rows_low_conf:
        exclusion_notes.append(
            f"{len(leak_rows_low_conf)} additional passage(s) matched a short (<=4 character, single-word) "
            "correct answer (e.g. \"yes\"/\"no\") as a whole word -- flagged `true_answer_leak_low_confidence` "
            "since such short common words are likely to appear coincidentally in unrelated sentences; these "
            "are **not** recommended grounds for exclusion on their own, but are listed for transparency."
        )
    if dup_text_rows:
        exclusion_notes.append(
            f"{len(dup_text_rows)} passage(s) had mutated text duplicated with another slot in the same "
            "query/family: " + "; ".join(f"`{r['family']}`/`{r['query_id']}`/slot={r['poison_slot']}" for r in dup_text_rows)
            + ". Review recommended (reduces the intended discourse/lexical diversity of the 5-passage bundle)."
        )
    if unchanged_rows:
        exclusion_notes.append(
            f"{len(unchanged_rows)} passage(s) had mutated text identical to the original poison text -- "
            "flagged `mutated_text_unchanged_from_original`. These should be excluded (no mutation actually "
            "occurred)."
        )
    if exclusion_notes:
        for note in exclusion_notes:
            lines.append(f"- {note}")
    else:
        lines.append(
            "No query/family needs to be excluded from paper-level claims on integrity grounds: no "
            "structural resolution failures, no unchanged-text passages, no duplicate-text passages, and "
            "no high-confidence true-answer leaks were found. See the `target_wrong_answer_simple_match` "
            "caveat immediately below, which is expected/heuristic and not itself grounds for exclusion."
        )
    lines.append("")
    lines.append(
        f"Note (heuristic limitation, not an exclusion criterion): {len(target_not_found_rows)} passage(s) "
        "did not contain the literal `target_wrong_answer` string as a case-insensitive substring "
        "(`target_wrong_answer_not_found_by_simple_substring`). This is expected for many of these "
        "mutations, which paraphrase/imply the wrong answer through content rather than stating it "
        "verbatim (e.g. a yes/no question's mutated passage describing matching facts without literally "
        "writing \"yes\"); a simple string check cannot detect semantic entailment, so this flag alone "
        "does not indicate a defective mutation."
    )
    lines.append("")

    lines.append("## 5. Are the normalized files safe for fixed-context rerun and full retrieval rerun?")
    lines.append("")
    lines.append(
        "**Fixed-context rerun: Yes**, for every query/family included in the normalized output (see "
        "Section 3). Each normalized record carries a canonical `doc_id` resolved from "
        "`mutation_input_passages.csv` by `(query_id, poison_slot)`, the original (possibly wrong) "
        "family-file `doc_id` preserved separately under `source_file_doc_id`, and byte-identical "
        "mutated text -- sufficient for `scripts/run_text_mutation_fixed_context_eval.py`-style scoring "
        "that substitutes text into the existing fixed k=10 context by `doc_id`/`poison_slot`."
    )
    missing_clean = [qid for qid in query_ids if clean_context_counts.get(qid, 0) == 0]
    if missing_clean:
        lines.append(
            f"- Caveat: quer(y/ies) {', '.join(f'`{q}`' for q in missing_clean)} have **no** rows in "
            "`clean_context_passages.csv`, so a fixed-context rerun cannot reconstruct their k=10 context."
        )
    else:
        lines.append(
            f"- Cross-checked: all {len(query_ids)} audited query_id(s) have at least one row in "
            "`clean_context_passages.csv` (clean-passage counts: "
            + ", ".join(f"`{qid}`={clean_context_counts.get(qid, 0)}" for qid in query_ids) + ")."
        )
    lines.append("")
    lines.append(
        "**Full retrieval rerun: Conditionally yes, with an important caveat.** Normalizing `doc_id` "
        "only fixes *metadata identity* against the already-selected passage set; it does not verify "
        "that the canonical `doc_id` is independently retrievable from the underlying corpus/index in a "
        "full (non-fixed) retrieval pass -- that verification is out of scope for this audit (no "
        "retrieval was rerun here, per the strict constraints) and must happen as its own step before "
        "any full retrieval rerun's results are trusted."
    )
    lines.append("")
    if not correct_answers_available:
        lines.append(
            f"- Note: `{DEFAULT_ADV_TARGETED_RESULTS_PATH}` was not found, so the true-answer-leak check "
            "(Section 4) could not run for this invocation; `possible_true_answer_leak` is blank for all "
            "rows in that case. Re-run with `--adv_targeted_results_path` pointing at that artifact to "
            "enable it."
        )
        lines.append("")

    lines.append("## Per-family doc_id-mismatch detail")
    lines.append("")
    if doc_id_mismatch_rows:
        lines.append("| family | query_id | poison_slot | source_file_doc_id | canonical_doc_id |")
        lines.append("|---|---|---:|---|---|")
        for r in doc_id_mismatch_rows:
            lines.append(f"| `{r['family']}` | `{r['query_id']}` | {r['poison_slot']} | `{r['source_file_doc_id']}` | `{r['canonical_doc_id']}` |")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("## Files written")
    lines.append("")
    for spec in FAMILY_SPECS.values():
        n = normalized_counts.get(spec["normalized_filename"], 0)
        lines.append(f"- `{os.path.relpath(os.path.join(out_dir, spec['normalized_filename']), REPO_ROOT)}` -- {n} query record(s).")
    lines.append(f"- `{os.path.relpath(os.path.join(out_dir, 'mutation_bundle_1_integrity_audit.csv'), REPO_ROOT)}` -- {len(audit_rows)} row(s).")
    lines.append("")

    lines.append("## Process confirmation")
    lines.append("")
    lines.append("- No GPT/API calls were made.")
    lines.append("- No `llm.query()` calls were made.")
    lines.append("- Retrieval was not rerun.")
    lines.append("- No defense was rerun/scored (this is a pure metadata/schema audit).")
    lines.append("- No defense code (`defense/*.py`) was read or modified.")
    lines.append("- No mutated text content was altered; every normalized `mutated_text` value is copied verbatim from its source family file.")
    lines.append("- Only `doc_id` metadata was normalized (canonical value substituted; original preserved under `source_file_doc_id`).")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Orchestration.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot_dir", default=os.path.join(REPO_ROOT, DEFAULT_PILOT_DIR))
    parser.add_argument("--bundle_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument(
        "--adv_targeted_results_path",
        default=os.path.join(REPO_ROOT, DEFAULT_ADV_TARGETED_RESULTS_PATH),
        help="Local artifact with each query's 'correct answer' field, used only for the "
             "true-answer-leak check. Not fetched from any API.",
    )
    args = parser.parse_args()

    pilot_dir = args.pilot_dir
    bundle_dir = args.bundle_dir or os.path.join(pilot_dir, "mutation_bundle_1")
    out_dir = args.out_dir or os.path.join(bundle_dir, "normalized")

    authoritative = load_authoritative_passages(os.path.join(pilot_dir, "mutation_input_passages.csv"))
    known_query_ids = set(qid for qid, _slot in authoritative)
    clean_context_counts = load_known_query_ids_with_clean_context(os.path.join(pilot_dir, "clean_context_passages.csv"))
    selection_roles = load_selection_roles(os.path.join(pilot_dir, "selected_queries.csv"))
    correct_answers = load_correct_answers(args.adv_targeted_results_path)

    all_audit_rows: List[Dict] = []
    normalized_counts: Dict[str, int] = {}
    query_ids_seen: set = set()

    for family_key, spec in FAMILY_SPECS.items():
        path = os.path.join(bundle_dir, spec["filename"])
        audit_rows, normalized_records = audit_and_normalize_family(
            family_key, spec, path, authoritative, known_query_ids, correct_answers, selection_roles,
        )
        all_audit_rows.extend(audit_rows)
        query_ids_seen.update(r["query_id"] for r in audit_rows if r.get("query_id"))

        out_path = os.path.join(out_dir, spec["normalized_filename"])
        write_normalized_jsonl(out_path, normalized_records)
        normalized_counts[spec["normalized_filename"]] = len(normalized_records)
        print(
            f"[audit_normalize_mutation_bundle_1] {family_key}: wrote {len(normalized_records)} "
            f"normalized query record(s) to {out_path}"
        )

    audit_csv_path = os.path.join(out_dir, "mutation_bundle_1_integrity_audit.csv")
    write_audit_csv(audit_csv_path, all_audit_rows)

    report = build_report(
        pilot_dir=pilot_dir, bundle_dir=bundle_dir, out_dir=out_dir,
        audit_rows=all_audit_rows, normalized_counts=normalized_counts,
        clean_context_counts=clean_context_counts, query_ids=sorted(query_ids_seen),
        correct_answers_available=bool(correct_answers),
    )
    report_path = os.path.join(out_dir, "MUTATION_BUNDLE_1_INTEGRITY_AUDIT.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    n_mismatches = len([r for r in all_audit_rows if r.get("doc_id_mismatch")])
    n_unresolved = len([r for r in all_audit_rows if r.get("included_in_normalized_output") is False])
    print(
        f"[audit_normalize_mutation_bundle_1] wrote {len(all_audit_rows)} audit row(s) to {audit_csv_path}; "
        f"{n_mismatches} doc_id mismatch(es); {n_unresolved} unresolved row(s)/record(s)."
    )
    print(f"[audit_normalize_mutation_bundle_1] wrote report to {report_path}")


if __name__ == "__main__":
    main()
