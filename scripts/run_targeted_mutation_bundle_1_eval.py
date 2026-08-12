#!/usr/bin/env python3
"""Fixed-context evaluation of the three defense-*targeted* GPT mutation
families in `manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/`,
against the original k=10 baseline and against all three defenses (not only
each family's intended target), to detect unexpected cross-defense failures.

**Fixed retrieval only** -- identical guarantees to
`scripts/run_text_mutation_fixed_context_eval.py` (reused directly, not
reimplemented, for every defense-scoring/context-reconstruction primitive):
this script never reruns retrieval, never trains/retrains any model, never
calls an LLM/GPT/API, and never calls `llm.query()`. It only:

1. reconstructs each selected query's *exact* original k=10 retrieved
   context from already-exported pilot artifacts
   (`mutation_input_passages.csv` + `clean_context_passages.csv`);
2. for each of the 3 mutation families, replaces *only* the 5 poisoned
   passages' text (matched by `poison_slot`) with that family's rewritten
   text, leaving every clean passage and k=10 membership untouched;
3. re-scores that fixed passage list with the same already-trained defense
   scorers `run_text_mutation_fixed_context_eval.py` uses (RAGDefender,
   semantic-mode FilterRAG, ML-FilterRAG-top-k) -- imported and reused
   unmodified, not copied.

Mutation family input files (each a JSON array of 6 per-query records --
4 primary + 2 backup selected queries -- with exactly 5 rewritten passages
per query, keyed by `poison_slot` 0..4):

- `ragdefender_discourse_diverse_poisoned_passages.txt` (RAGDefender-targeted;
  key `rewritten_passages` / text field `rewritten_text`)
- `filterrag_gpt_poisoned_passages_low_overlap.txt` (FilterRAG-targeted;
  key `poisoned_passages` / text field `rewritten_text`)
- `mlfilterrag_gpt_prompt_packets_clean_reference_rewrites.txt`
  (ML-FilterRAG-targeted; key `poisoned_passages` / text field
  `original_text` -- NOTE: despite the field's name, this file's own
  `mutation_instructions` ("Rewrite the five poisoned passages...")
  confirm it holds the *mutated* text, not the true original poison text;
  this is cross-checked against `mutation_input_passages.csv` at parse
  time, which holds the real original text and differs from this field).

Known data-quality issue handled explicitly (not silently): in
`ragdefender_discourse_diverse_poisoned_passages.txt`, the per-passage
`doc_id` field is wrong (placeholder indices `0`..`4`) for 3 of the 6
queries (`5a7759fc5542993569682d60`, `5a8133725542995ce29dcbdb`,
`5a8e068b5542995085b37384`). Passage *identity* is resolved by
`(query_id, poison_slot)` against `mutation_input_passages.csv` (the
authoritative source, already used to build the original context) rather
than trusting each family file's own `doc_id`; every such mismatch is
still detected, counted, and reported (see `doc_id_mismatches` in the
report and console output) rather than silently ignored.

Usage:
    python scripts/run_targeted_mutation_bundle_1_eval.py \\
        --pilot_dir manual_text_mutation_pilot/hotpotqa_50q_k10 \\
        --bundle_dir manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1 \\
        --out_dir manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/evaluation
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Belt-and-suspenders: forced offline before importing anything that may
# touch huggingface_hub/sentence-transformers, mirroring
# run_text_mutation_fixed_context_eval.py's own convention (which sets the
# same two variables again via setdefault -- no conflict).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import run_text_mutation_fixed_context_eval as base_eval  # noqa: E402

DEFAULT_PILOT_DIR = base_eval.DEFAULT_PILOT_DIR
DEFAULT_BUNDLE_DIR = os.path.join(DEFAULT_PILOT_DIR, "mutation_bundle_1")
DEFAULT_ML_MODEL_PATH = base_eval.DEFAULT_ML_MODEL_PATH

FORBIDDEN_SOURCE_SNIPPETS = (
    "openai", "google.generativeai", "llm.query", "anthropic", "requests.post",
)

# ---------------------------------------------------------------------------
# Mutation-family specs: 3 defense-targeted families, each a single bundle
# (one rewrite per poison_slot, no gpt_b01/b02/b03 alternatives) per query.
# ---------------------------------------------------------------------------

FAMILY_SPECS: Dict[str, Dict] = {
    "ragdefender_targeted": {
        "filename": "ragdefender_discourse_diverse_poisoned_passages.txt",
        "intended_defense": "ragdefender",
        "passage_list_keys": ("rewritten_passages",),
        "text_field_keys": ("rewritten_text",),
    },
    "filterrag_targeted": {
        "filename": "filterrag_gpt_poisoned_passages_low_overlap.txt",
        "intended_defense": "filterrag",
        "passage_list_keys": ("poisoned_passages",),
        "text_field_keys": ("rewritten_text",),
    },
    "mlfilterrag_targeted": {
        "filename": "mlfilterrag_gpt_prompt_packets_clean_reference_rewrites.txt",
        "intended_defense": "ml_filterrag",
        "passage_list_keys": ("poisoned_passages",),
        # "original_text" is this file's actual mutated-text field name (see
        # module docstring); "rewritten_text" is also accepted defensively
        # in case a future export of this family fixes the field name.
        "text_field_keys": ("rewritten_text", "original_text"),
    },
}

DEFENSE_NAMES = ("ragdefender", "filterrag", "ml_filterrag")


# ---------------------------------------------------------------------------
# 1. Parsing the 3 targeted mutation-family files.
# ---------------------------------------------------------------------------

def parse_family_file(path: str, spec: Dict) -> Dict[str, Dict]:
    """Parse one targeted-mutation-family JSON array file into
    `{query_id: {"query_id", "question", "target_wrong_answer", "passages": [...]}}`.

    Each passage dict has `poison_slot` (int), `file_doc_id` (str or None,
    the family file's own -- possibly wrong, see module docstring --
    `doc_id`), and `text` (str, non-empty). Raises `ValueError` naming the
    offending record on any schema violation (missing query_id, wrong
    passage count, duplicate/missing poison_slot, empty text, no
    recognized passage-list/text-field key present).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    stripped = raw.strip()
    if not stripped:
        raise ValueError(f"{path}: file is empty.")
    records = json.loads(stripped)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a non-empty top-level JSON array, got {type(records).__name__}.")

    out: Dict[str, Dict] = {}
    for rec in records:
        qid = rec.get("query_id")
        if not qid or not isinstance(qid, str):
            raise ValueError(f"{path}: record missing a valid 'query_id': {rec!r}")

        passages_list = None
        for list_key in spec["passage_list_keys"]:
            if list_key in rec:
                passages_list = rec[list_key]
                break
        if passages_list is None:
            raise ValueError(
                f"{path}: query_id={qid!r} has none of the expected passage-list keys "
                f"{spec['passage_list_keys']!r}."
            )
        if not isinstance(passages_list, list) or len(passages_list) != 5:
            raise ValueError(
                f"{path}: query_id={qid!r} has "
                f"{len(passages_list) if isinstance(passages_list, list) else 'no'} passages "
                "(expected exactly 5)."
            )

        parsed_passages: List[Dict] = []
        seen_slots = set()
        for p in passages_list:
            slot = p.get("poison_slot")
            if slot is None:
                raise ValueError(f"{path}: query_id={qid!r} has a passage missing 'poison_slot'.")
            slot = int(slot)
            if slot in seen_slots:
                raise ValueError(f"{path}: query_id={qid!r} has a duplicate poison_slot={slot!r}.")
            seen_slots.add(slot)

            text = None
            for text_key in spec["text_field_keys"]:
                candidate = p.get(text_key)
                if candidate and isinstance(candidate, str) and candidate.strip():
                    text = candidate
                    break
            if not text:
                raise ValueError(
                    f"{path}: query_id={qid!r} poison_slot={slot!r} has no non-empty text in any of "
                    f"{spec['text_field_keys']!r}."
                )
            parsed_passages.append({"poison_slot": slot, "file_doc_id": p.get("doc_id"), "text": text})

        if sorted(seen_slots) != [0, 1, 2, 3, 4]:
            raise ValueError(
                f"{path}: query_id={qid!r} has poison_slot values {sorted(seen_slots)!r} "
                "(expected exactly [0, 1, 2, 3, 4])."
            )
        parsed_passages.sort(key=lambda x: x["poison_slot"])

        out[qid] = {
            "query_id": qid,
            "question": rec.get("question"),
            "target_wrong_answer": rec.get("target_wrong_answer"),
            "passages": parsed_passages,
        }
    return out


def check_doc_id_consistency(
    family_key: str, family_records: Dict[str, Dict], poison_by_query: Dict[str, List[Dict]]
) -> List[Dict]:
    """Cross-checks each family passage's *own* `doc_id` field against the
    authoritative `(query_id, poison_slot) -> doc_id` mapping in
    `mutation_input_passages.csv`. Returns a list of mismatch dicts (never
    raises); passage identity for scoring always uses the CSV mapping via
    `poison_slot`, never the family file's `doc_id`, so a mismatch here is
    informational/diagnostic only -- it does not change which passage gets
    mutated (see module docstring)."""
    mismatches: List[Dict] = []
    for qid, rec in family_records.items():
        doc_by_slot = {int(r["poison_slot"]): r["doc_id"] for r in poison_by_query.get(qid, [])}
        for p in rec["passages"]:
            expected = doc_by_slot.get(p["poison_slot"])
            if p["file_doc_id"] and expected and p["file_doc_id"] != expected:
                mismatches.append({
                    "family": family_key, "query_id": qid, "poison_slot": p["poison_slot"],
                    "file_doc_id": p["file_doc_id"], "csv_doc_id": expected,
                })
    return mismatches


def family_record_to_bundle(rec: Dict) -> Dict:
    """Adapts a parsed family record into the `{"mutated_passages": [...]}`
    shape `run_text_mutation_fixed_context_eval.build_mutated_context`
    expects (`poison_rank` == `poison_slot`, `mutated_text` == `text`), so
    that function is reused completely unmodified."""
    return {
        "mutated_passages": [
            {"poison_rank": p["poison_slot"], "mutated_text": p["text"]} for p in rec["passages"]
        ]
    }


# ---------------------------------------------------------------------------
# 2. Aggregation helpers.
# ---------------------------------------------------------------------------

def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(statistics.fmean(values)) if values else None


def summarize_family_defense(
    family_key: str, intended_defense: str, defense: str,
    bundle_rows: Sequence[Dict], delta_rows: Sequence[Dict],
) -> Dict:
    """One row of `targeted_family_summary_by_defense.csv`: means across
    every query evaluated for `family_key`, restricted to `defense`'s own
    metrics (other defenses' columns are left blank on this row -- no
    invented cross-defense numbers)."""
    row: Dict = {
        "family": family_key,
        "intended_target_defense": intended_defense,
        "defense": defense,
        "is_intended_target": defense == intended_defense,
        "n_queries": len(bundle_rows),
    }
    if defense == "ragdefender":
        row.update({
            "mean_removed_poison": _mean([r["ragdefender_removed_poison"] for r in bundle_rows]),
            "mean_removed_clean": _mean([r["ragdefender_removed_clean"] for r in bundle_rows]),
            "mean_residual_poison_fraction": _mean([r["ragdefender_residual_poison_fraction"] for r in bundle_rows]),
            "mean_delta_removed_poison": _mean([d["delta_ragdefender_removed_poison"] for d in delta_rows]),
            "mean_delta_removed_clean": _mean([d["delta_ragdefender_removed_clean"] for d in delta_rows]),
            "mean_delta_residual_poison_fraction": _mean([d["delta_ragdefender_residual_poison_fraction"] for d in delta_rows]),
            "mean_top_pair_pp": _mean([r["ragdefender_top_pair_pp"] for r in bundle_rows]),
            "mean_delta_top_pair_pp": _mean([d["delta_ragdefender_top_pair_pp"] for d in delta_rows]),
            "mean_pp_cosine": _mean([r["ragdefender_mean_pp_cosine"] for r in bundle_rows]),
            "mean_pc_cosine": _mean([r["ragdefender_mean_pc_cosine"] for r in bundle_rows]),
        })
    elif defense == "filterrag":
        row.update({
            "mean_removed_poison": _mean([r["filterrag_removed_poison"] for r in bundle_rows]),
            "mean_removed_clean": _mean([r["filterrag_removed_clean"] for r in bundle_rows]),
            "mean_residual_poison_fraction": _mean([r["filterrag_residual_poison_fraction"] for r in bundle_rows]),
            "mean_delta_removed_poison": _mean([d["delta_filterrag_removed_poison"] for d in delta_rows]),
            "mean_delta_removed_clean": _mean([d["delta_filterrag_removed_clean"] for d in delta_rows]),
            "mean_delta_residual_poison_fraction": _mean([d["delta_filterrag_residual_poison_fraction"] for d in delta_rows]),
            "mean_freq_density": _mean([r["filterrag_mean_freq_density_poison"] for r in bundle_rows]),
            "mean_delta_freq_density": _mean([d["delta_filterrag_mean_freq_density_poison"] for d in delta_rows]),
            "mean_matched_freq_sum": _mean([r["filterrag_mean_matched_freq_sum_poison"] for r in bundle_rows]),
            "mean_delta_matched_freq_sum": _mean([d["delta_filterrag_mean_matched_freq_sum_poison"] for d in delta_rows]),
        })
    else:  # ml_filterrag
        row.update({
            "mean_removed_poison": _mean([r["ml_removed_poison_t04"] for r in bundle_rows]),
            "mean_removed_clean": _mean([r["ml_removed_clean_t04"] for r in bundle_rows]),
            "mean_residual_poison_fraction": _mean([r["ml_residual_poison_fraction_t04"] for r in bundle_rows]),
            "mean_delta_removed_poison": _mean([d["delta_ml_removed_poison_t04"] for d in delta_rows]),
            "mean_delta_removed_clean": _mean([d["delta_ml_removed_clean_t04"] for d in delta_rows]),
            "mean_delta_residual_poison_fraction": _mean([d["delta_ml_residual_poison_fraction_t04"] for d in delta_rows]),
            "mean_removed_poison_t035": _mean([r["ml_removed_poison_t035"] for r in bundle_rows]),
            "mean_removed_poison_t05": _mean([r["ml_removed_poison_t05"] for r in bundle_rows]),
            "mean_poison_probability": _mean([r["ml_mean_poison_probability"] for r in bundle_rows]),
            "mean_delta_poison_probability": _mean([d["delta_ml_mean_poison_probability"] for d in delta_rows]),
            "mean_freq_density": _mean([r["ml_mean_freq_density_poison"] for r in bundle_rows]),
            "mean_delta_freq_density": _mean([d["delta_ml_mean_freq_density_poison"] for d in delta_rows]),
            "mean_matched_freq_sum": _mean([r["ml_mean_matched_freq_sum_poison"] for r in bundle_rows]),
            "mean_delta_matched_freq_sum": _mean([d["delta_ml_mean_matched_freq_sum_poison"] for d in delta_rows]),
            "mean_perplexity": _mean([r["ml_mean_perplexity_poison"] for r in bundle_rows]),
            "mean_slm_answer_logprob": _mean([r["ml_mean_slm_answer_logprob_poison"] for r in bundle_rows]),
        })
    return row


SUMMARY_FIELDS = [
    "family", "intended_target_defense", "defense", "is_intended_target", "n_queries",
    "mean_removed_poison", "mean_removed_clean", "mean_residual_poison_fraction",
    "mean_delta_removed_poison", "mean_delta_removed_clean", "mean_delta_residual_poison_fraction",
    "mean_top_pair_pp", "mean_delta_top_pair_pp", "mean_pp_cosine", "mean_pc_cosine",
    "mean_freq_density", "mean_delta_freq_density", "mean_matched_freq_sum", "mean_delta_matched_freq_sum",
    "mean_poison_probability", "mean_delta_poison_probability",
    "mean_removed_poison_t035", "mean_removed_poison_t05",
    "mean_perplexity", "mean_slm_answer_logprob",
]


def build_cross_defense_failure_matrix(summary_rows: Sequence[Dict]) -> List[Dict]:
    """One row per family: for each of the 3 defenses, the mean
    `delta_removed_poison` and a `weakened` flag (`True` iff that mean delta
    is strictly negative, i.e. the mutated passages evaded *more* removal
    than the unmutated baseline for that defense). `any_cross_defense_failure`
    is `True` iff a defense *other than* the family's own intended target is
    flagged `weakened`."""
    by_family: Dict[str, Dict[str, Dict]] = {}
    for row in summary_rows:
        by_family.setdefault(row["family"], {})[row["defense"]] = row

    matrix: List[Dict] = []
    for family, defenses in by_family.items():
        intended = next(iter(defenses.values()))["intended_target_defense"]
        out: Dict = {"family": family, "intended_target_defense": intended}
        weakened_others = []
        for defense in DEFENSE_NAMES:
            d = defenses.get(defense, {})
            delta = d.get("mean_delta_removed_poison")
            weakened = bool(delta is not None and delta < 0)
            out[f"{defense}_mean_delta_removed_poison"] = delta
            out[f"{defense}_weakened"] = weakened
            if weakened and defense != intended:
                weakened_others.append(defense)
        out["any_cross_defense_failure"] = len(weakened_others) > 0
        out["cross_defense_failure_defenses"] = ";".join(weakened_others)
        matrix.append(out)
    return matrix


CROSS_MATRIX_FIELDS = [
    "family", "intended_target_defense",
    "ragdefender_mean_delta_removed_poison", "ragdefender_weakened",
    "filterrag_mean_delta_removed_poison", "filterrag_weakened",
    "ml_filterrag_mean_delta_removed_poison", "ml_filterrag_weakened",
    "any_cross_defense_failure", "cross_defense_failure_defenses",
]


# ---------------------------------------------------------------------------
# 3. Comparison against the first (generic) mutation pilot.
# ---------------------------------------------------------------------------

def load_first_pilot_deltas(path: str) -> Optional[List[Dict]]:
    """Reads the already-computed `mutation_bundle_deltas.csv` from the
    first (generic gpt_b01/b02/b03) fixed-context pilot, if present, purely
    to aggregate already-computed numbers for a report comparison -- no
    model is re-run to produce this. Returns `None` if the file is absent
    (comparison section is then explicitly marked unavailable, never
    fabricated)."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if v in ("", None):
                r[k] = None
            else:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


# ---------------------------------------------------------------------------
# 4. Report.
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    return base_eval._fmt(v)  # noqa: SLF001 -- shared formatting helper, not private state


def build_report(
    *, pilot_dir: str, bundle_dir: str, ml_model_path: str, query_ids: Sequence[str],
    baseline_rows: Sequence[Dict], bundle_rows: Sequence[Dict], delta_rows: Sequence[Dict],
    summary_rows: Sequence[Dict], cross_matrix: Sequence[Dict], doc_id_mismatches: Sequence[Dict],
    selected_queries: Dict[str, Dict], first_pilot_deltas: Optional[List[Dict]],
) -> str:
    lines: List[str] = []
    lines.append("# Targeted Mutation Bundle 1 -- Fixed-Context Cross-Defense Evaluation Report")
    lines.append("")
    lines.append(
        "Fixed-retrieval evaluation of the 3 defense-*targeted* GPT mutation families in "
        f"`{os.path.relpath(bundle_dir, REPO_ROOT)}/`, each evaluated against **all three** "
        "defenses (RAGDefender, semantic FilterRAG epsilon=0.2, ML-FilterRAG-top-k at "
        "t in {0.35, 0.4, 0.5}) -- not only its own intended target -- to detect unexpected "
        "cross-defense failures. Retrieval membership/order is identical between the baseline "
        "and every mutated context for a given query; only the 5 poisoned passages' text "
        "differs per family; clean passages are byte-identical everywhere."
    )
    lines.append("")
    lines.append("## Artifact paths used")
    lines.append("")
    for key, spec in FAMILY_SPECS.items():
        lines.append(f"- `{os.path.relpath(os.path.join(bundle_dir, spec['filename']), REPO_ROOT)}` ({key}, intended target: {spec['intended_defense']})")
    lines.append(f"- `{os.path.join(pilot_dir, 'selected_queries.csv')}`")
    lines.append(f"- `{os.path.join(pilot_dir, 'mutation_input_passages.csv')}`")
    lines.append(f"- `{os.path.join(pilot_dir, 'clean_context_passages.csv')}`")
    lines.append(f"- `{ml_model_path}` (trained ML-FilterRAG-top-k random-forest classifier, loaded read-only)")
    lines.append("")

    if doc_id_mismatches:
        lines.append("## Data-integrity note: family-file `doc_id` mismatches")
        lines.append("")
        lines.append(
            f"{len(doc_id_mismatches)} passage(s) across the family files have a `doc_id` field "
            "that does not match `mutation_input_passages.csv`'s authoritative "
            "`(query_id, poison_slot) -> doc_id` mapping. Passage identity for every mutated "
            "context in this evaluation is resolved by `poison_slot` against that CSV (never by "
            "the family file's own possibly-wrong `doc_id`), so this does **not** affect which "
            "passage's text was mutated -- it is reported here purely as a data-quality flag on "
            "the input files:"
        )
        lines.append("")
        lines.append("| family | query_id | poison_slot | file doc_id | csv doc_id |")
        lines.append("|---|---|---:|---|---|")
        for m in doc_id_mismatches:
            lines.append(f"| `{m['family']}` | `{m['query_id']}` | {m['poison_slot']} | `{m['file_doc_id']}` | `{m['csv_doc_id']}` |")
        lines.append("")

    lines.append("## Queries evaluated")
    lines.append("")
    for qid in query_ids:
        q = selected_queries.get(qid, {})
        lines.append(
            f"- `{qid}` ({q.get('selection_role', 'n/a')}) -- {q.get('question', 'n/a')} "
            f"(target wrong answer: {q.get('target_wrong_answer', 'n/a')})"
        )
    lines.append("")

    lines.append("## Summary by family x defense")
    lines.append("")
    lines.append("| family | intended target | defense | mean removed_poison | mean delta_removed_poison | mean residual_poison_fraction |")
    lines.append("|---|---|---|---:|---:|---:|")
    for r in summary_rows:
        marker = " **(target)**" if r["is_intended_target"] else ""
        lines.append(
            f"| `{r['family']}` | {r['intended_target_defense']} | {r['defense']}{marker} | "
            f"{_fmt(r['mean_removed_poison'])} | {_fmt(r['mean_delta_removed_poison'])} | "
            f"{_fmt(r['mean_residual_poison_fraction'])} |"
        )
    lines.append("")

    lines.append("## Cross-defense failure matrix")
    lines.append("")
    lines.append("| family | intended target | RAGDefender delta (weakened?) | FilterRAG delta (weakened?) | ML-FilterRAG delta (weakened?) | any cross-defense failure |")
    lines.append("|---|---|---|---|---|---|")
    for r in cross_matrix:
        lines.append(
            f"| `{r['family']}` | {r['intended_target_defense']} | "
            f"{_fmt(r['ragdefender_mean_delta_removed_poison'])} ({r['ragdefender_weakened']}) | "
            f"{_fmt(r['filterrag_mean_delta_removed_poison'])} ({r['filterrag_weakened']}) | "
            f"{_fmt(r['ml_filterrag_mean_delta_removed_poison'])} ({r['ml_filterrag_weakened']}) | "
            f"{r['any_cross_defense_failure']} |"
        )
    lines.append("")

    # ---- Derived facts for the answers section --------------------------
    summary_by_family_defense = {(r["family"], r["defense"]): r for r in summary_rows}

    def _best_attack_on_own_target(family_key: str) -> Dict:
        intended = FAMILY_SPECS[family_key]["intended_defense"]
        return summary_by_family_defense[(family_key, intended)]

    own_target_rows = {fk: _best_attack_on_own_target(fk) for fk in FAMILY_SPECS}
    best_family_for_own_target = min(
        own_target_rows.items(),
        key=lambda kv: (kv[1]["mean_delta_removed_poison"] if kv[1]["mean_delta_removed_poison"] is not None else 0.0),
    )

    any_cross_failure = any(r["any_cross_defense_failure"] for r in cross_matrix)
    cross_failure_families = [r["family"] for r in cross_matrix if r["any_cross_defense_failure"]]

    def _largest_drop_for_defense(defense: str, delta_key: str) -> Optional[Dict]:
        candidates = [r for r in delta_rows if r.get(delta_key) is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r[delta_key])

    largest_drop_ragdefender = _largest_drop_for_defense("ragdefender", "delta_ragdefender_removed_poison")
    largest_drop_filterrag = _largest_drop_for_defense("filterrag", "delta_filterrag_removed_poison")
    largest_drop_ml = _largest_drop_for_defense("ml_filterrag", "delta_ml_removed_poison_t04")

    best_rerun_candidate = min(
        bundle_rows,
        key=lambda r: (
            r["ragdefender_removed_poison"] + r["filterrag_removed_poison"] + r["ml_removed_poison_t04"],
            r["ml_mean_poison_probability"] if r["ml_mean_poison_probability"] is not None else 1.0,
        ),
    )

    any_reduced_top_pair_pp = any(
        d.get("delta_ragdefender_top_pair_pp") is not None and d["delta_ragdefender_top_pair_pp"] < 0
        for d in delta_rows
    )
    min_top_pair_pp_delta = min(
        (d["delta_ragdefender_top_pair_pp"] for d in delta_rows if d.get("delta_ragdefender_top_pair_pp") is not None),
        default=None,
    )

    freq_density_values = [r["filterrag_mean_freq_density_poison"] for r in bundle_rows if r["filterrag_mean_freq_density_poison"] is not None]
    min_freq_density = min(freq_density_values) if freq_density_values else None
    freq_density_at_or_below_epsilon = min_freq_density is not None and min_freq_density <= base_eval.FILTERRAG_EPSILON

    proba_values = [r["ml_mean_poison_probability"] for r in bundle_rows if r["ml_mean_poison_probability"] is not None]
    min_proba = min(proba_values) if proba_values else None
    any_mean_proba_below_t04 = min_proba is not None and min_proba < base_eval.ML_PRIMARY_THRESHOLD
    any_individual_below_t04 = any(r["ml_removed_poison_t04"] < r["N_retrieved_poison"] for r in bundle_rows)

    lines.append("## Answers")
    lines.append("")
    lines.append(
        f"**1. Which mutation family best attacked its intended defense?** "
        f"`{best_family_for_own_target[0]}` (intended target: {FAMILY_SPECS[best_family_for_own_target[0]]['intended_defense']}) "
        f"-- mean delta_removed_poison on its own target = {_fmt(best_family_for_own_target[1]['mean_delta_removed_poison'])} "
        f"(most negative / largest reduction among the 3 families' own-target deltas: "
        + ", ".join(f"`{fk}`={_fmt(r['mean_delta_removed_poison'])}" for fk, r in own_target_rows.items())
        + ")."
    )
    lines.append("")
    lines.append(
        f"**2. Did any mutation family unexpectedly weaken another defense?** "
        f"{'Yes' if any_cross_failure else 'No'} -- "
        + (
            f"family/families {', '.join(f'`{f}`' for f in cross_failure_families)} showed a negative mean "
            "delta_removed_poison on a defense other than their own intended target (see cross-defense "
            "failure matrix above for which defense(s))."
            if any_cross_failure else
            "every family's mean delta_removed_poison was >= 0 on every defense other than its own intended "
            "target; no family reduced a non-target defense's removal count on average."
        )
    )
    lines.append("")
    lines.append("**3. Which individual bundle caused the largest drop in removed_poison for each defense?**")
    lines.append("")
    for label, row, key in (
        ("RAGDefender", largest_drop_ragdefender, "delta_ragdefender_removed_poison"),
        ("FilterRAG", largest_drop_filterrag, "delta_filterrag_removed_poison"),
        ("ML-FilterRAG (t=0.4)", largest_drop_ml, "delta_ml_removed_poison_t04"),
    ):
        if row is None:
            lines.append(f"- {label}: n/a (no non-null deltas).")
        else:
            lines.append(
                f"- {label}: `{row['query_id']}` / `{row['family']}` "
                f"(delta_removed_poison = {_fmt(row[key])})."
            )
    lines.append("")
    lines.append(
        f"**4. Which bundle is the best candidate for a full retrieval rerun?** "
        f"`{best_rerun_candidate['query_id']}` / `{best_rerun_candidate['family']}` -- lowest combined "
        f"removal across the three defenses (RAGDefender={best_rerun_candidate['ragdefender_removed_poison']}, "
        f"FilterRAG={best_rerun_candidate['filterrag_removed_poison']}, "
        f"ML-FilterRAG t0.4={best_rerun_candidate['ml_removed_poison_t04']}, out of "
        f"{best_rerun_candidate['N_retrieved_poison']} retrieved poison passages)."
    )
    lines.append("")
    lines.append(
        f"**5. Did any bundle reduce RAGDefender `top_pair_pp`?** "
        f"{'Yes' if any_reduced_top_pair_pp else 'No'} -- "
        f"most negative delta_top_pair_pp observed = {_fmt(min_top_pair_pp_delta)}."
    )
    lines.append("")
    lines.append(
        f"**6. Did any bundle push FilterRAG Freq-Density close to or below epsilon={base_eval.FILTERRAG_EPSILON}?** "
        f"{'Yes' if freq_density_at_or_below_epsilon else 'No'} -- "
        f"lowest mean Freq-Density (poison passages) observed across all family/query rows = {_fmt(min_freq_density)}."
    )
    lines.append("")
    if any_mean_proba_below_t04:
        proba_detail = f"at least one row's *mean* predicted poison probability fell below t={base_eval.ML_PRIMARY_THRESHOLD}."
    elif any_individual_below_t04:
        proba_detail = (
            "no row's mean fell below the threshold, but at least one row had "
            "ml_removed_poison_t04 < N_retrieved_poison (an individual poison passage scored below "
            f"t={base_eval.ML_PRIMARY_THRESHOLD} even though the mean stayed above it)."
        )
    else:
        proba_detail = f"every row had ml_removed_poison_t04 == N_retrieved_poison (no individual poison passage fell below t={base_eval.ML_PRIMARY_THRESHOLD})."
    lines.append(
        f"**7. Did any bundle push ML-FilterRAG poison probability below t={base_eval.ML_PRIMARY_THRESHOLD}?** "
        f"{'Yes' if (any_mean_proba_below_t04 or any_individual_below_t04) else 'No'} -- "
        f"lowest mean predicted poison probability observed = {_fmt(min_proba)}; {proba_detail}"
    )
    lines.append("")

    if first_pilot_deltas is None:
        lines.append(
            "**8. Are the results stronger than the first generic mutation pilot?** "
            "Comparison unavailable -- the first pilot's `mutation_bundle_deltas.csv` was not found "
            "at the expected path; no comparison was fabricated."
        )
    else:
        first_removed = [d["delta_removed_poison"] for d in first_pilot_deltas if d.get("delta_removed_poison") is not None]
        first_proba = [d["delta_mean_poison_probability"] for d in first_pilot_deltas if d.get("delta_mean_poison_probability") is not None]
        this_removed = [d["delta_ml_removed_poison_t04"] for d in delta_rows if d.get("delta_ml_removed_poison_t04") is not None]
        this_proba = [d["delta_ml_mean_poison_probability"] for d in delta_rows if d.get("delta_ml_mean_poison_probability") is not None]
        first_mean_removed = _mean(first_removed)
        this_mean_removed = _mean(this_removed)
        first_mean_proba = _mean(first_proba)
        this_mean_proba = _mean(this_proba)
        stronger = (
            this_mean_removed is not None and first_mean_removed is not None and this_mean_removed < first_mean_removed
        )
        lines.append(
            f"**8. Are the results stronger than the first generic mutation pilot?** "
            f"Mean delta_removed_poison (ML-FilterRAG t=0.4): first pilot (generic gpt_b01/b02/b03, "
            f"4 primary queries) = {_fmt(first_mean_removed)}, this targeted pilot (3 families, "
            f"{len(query_ids)} queries) = {_fmt(this_mean_removed)}. Mean delta_mean_poison_probability: "
            f"first pilot = {_fmt(first_mean_proba)}, this targeted pilot = {_fmt(this_mean_proba)}. "
            f"{'Yes -- the targeted families achieved a more negative (larger) mean reduction in ML-FilterRAG removed_poison than the first generic pilot.' if stronger else 'No / marginal -- the targeted families did not achieve a larger mean reduction in ML-FilterRAG removed_poison than the first generic pilot on this comparison basis.'}"
        )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Each family here is a *single* rewrite per poison_slot (unlike the first pilot's 3 "
        "alternative gpt_b01/b02/b03 bundles per query/condition), so per-family statistics in this "
        "report are means over 6 queries, not over multiple independent bundle attempts per query."
    )
    lines.append(
        "- See the data-integrity note above regarding `ragdefender_discourse_diverse_poisoned_passages.txt`'s "
        "incorrect `doc_id` field for 3 of 6 queries; passage identity was resolved by `poison_slot` "
        "against `mutation_input_passages.csv` in every case, so scoring is unaffected."
    )
    lines.append(
        "- `mlfilterrag_gpt_prompt_packets_clean_reference_rewrites.txt` stores its mutated text under "
        "the field name `original_text` (a mislabeling in the source file); this was verified against "
        "the file's own `mutation_instructions` and against `mutation_input_passages.csv`'s true "
        "original text (which differs) before being treated as mutated text here."
    )
    lines.append(
        "- All models (SLM, LM, RAGDefender embedder, ML-FilterRAG classifier) run on "
        f"device=`{base_eval.DEVICE}` for determinism, matching the first pilot's configuration; "
        "baseline metrics here are a fresh, independent re-scoring of the fixed passages for this "
        "evaluation's own internal reference, not a claim of bit-identical reproduction of any other run."
    )
    lines.append("")
    lines.append("## Process confirmation")
    lines.append("")
    lines.append("- No GPT/API calls were made.")
    lines.append("- No `llm.query()` calls were made.")
    lines.append("- Retrieval was not rerun (k=10 membership reconstructed verbatim from existing pilot CSV artifacts).")
    lines.append("- No model was trained or retrained (every model loaded read-only for inference).")
    lines.append("- No defense code (`defense/*.py`) was modified; every defense function used here is called unmodified via `scripts/run_text_mutation_fixed_context_eval.py`.")
    lines.append("- Only text substitution on already-provided mutation family files was applied; no mutations were generated by this script.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Orchestration.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot_dir", default=os.path.join(REPO_ROOT, DEFAULT_PILOT_DIR))
    parser.add_argument("--bundle_dir", default=None)
    parser.add_argument("--ml_model_path", default=os.path.join(REPO_ROOT, DEFAULT_ML_MODEL_PATH))
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    pilot_dir = args.pilot_dir
    bundle_dir = args.bundle_dir or os.path.join(pilot_dir, "mutation_bundle_1")
    out_dir = args.out_dir or os.path.join(bundle_dir, "evaluation")

    selected_queries = base_eval.load_selected_queries(os.path.join(pilot_dir, "selected_queries.csv"))
    poison_by_query = base_eval.load_mutation_input_passages(os.path.join(pilot_dir, "mutation_input_passages.csv"))
    clean_by_query = base_eval.load_clean_context_passages(os.path.join(pilot_dir, "clean_context_passages.csv"))

    family_records: Dict[str, Dict[str, Dict]] = {}
    doc_id_mismatches: List[Dict] = []
    for family_key, spec in FAMILY_SPECS.items():
        path = os.path.join(bundle_dir, spec["filename"])
        records = parse_family_file(path, spec)
        family_records[family_key] = records
        doc_id_mismatches.extend(check_doc_id_consistency(family_key, records, poison_by_query))

    query_ids = sorted(set.union(*(set(r) for r in family_records.values())))
    for qid in query_ids:
        if qid not in selected_queries or qid not in poison_by_query or qid not in clean_by_query:
            raise ValueError(f"query_id={qid!r} from a mutation family file is missing required pilot rows.")
    for family_key, records in family_records.items():
        missing = set(query_ids) - set(records)
        if missing:
            print(f"[run_targeted_mutation_bundle_1_eval] WARNING: family={family_key!r} is missing query_id(s) {sorted(missing)!r}; skipping those for this family.")

    if doc_id_mismatches:
        print(f"[run_targeted_mutation_bundle_1_eval] WARNING: {len(doc_id_mismatches)} family-file doc_id mismatch(es) detected (see report); scoring used poison_slot-based identity from mutation_input_passages.csv regardless.")

    print(f"[run_targeted_mutation_bundle_1_eval] loading models (device={base_eval.DEVICE})...")
    models = base_eval.load_models(args.ml_model_path)

    baseline_by_qid: Dict[str, Dict] = {}
    baseline_rows: List[Dict] = []
    bundle_rows: List[Dict] = []
    delta_rows: List[Dict] = []

    for qid in query_ids:
        q = selected_queries[qid]
        question = q["question"]
        target_wrong_answer = q["target_wrong_answer"]

        original_context = base_eval.build_original_context(poison_by_query[qid], clean_by_query[qid])
        print(f"[run_targeted_mutation_bundle_1_eval] scoring baseline for {qid} ({len(original_context)} passages)...")
        baseline_metrics = base_eval.score_context(question, original_context, models)
        baseline_by_qid[qid] = baseline_metrics
        baseline_rows.append({
            "query_id": qid, "k": 10, "selection_role": q.get("selection_role", ""),
            "question": question, "target_wrong_answer": target_wrong_answer,
            **baseline_metrics,
        })

        for family_key, spec in FAMILY_SPECS.items():
            rec = family_records[family_key].get(qid)
            if rec is None:
                continue

            bundle = family_record_to_bundle(rec)
            mutated_context = base_eval.build_mutated_context(original_context, poison_by_query[qid], bundle)
            base_eval.assert_same_k10_membership(original_context, mutated_context)
            for orig, mut in zip(original_context, mutated_context):
                if not orig.is_poison and orig.text != mut.text:
                    raise AssertionError(
                        f"{qid}/{family_key}: clean passage doc_id={orig.doc_id!r} text changed "
                        "(must remain unchanged)."
                    )

            print(f"[run_targeted_mutation_bundle_1_eval] scoring {qid} / {family_key}...")
            bundle_metrics = base_eval.score_context(question, mutated_context, models)
            bundle_row = {
                "query_id": qid, "k": 10, "family": family_key, "bundle_id": family_key,
                "intended_target_defense": spec["intended_defense"],
                "selection_role": q.get("selection_role", ""),
                "question": question, "target_wrong_answer": target_wrong_answer,
                **bundle_metrics,
            }
            bundle_rows.append(bundle_row)

            deltas = base_eval.compute_deltas(baseline_metrics, bundle_metrics)
            delta_rows.append({
                "query_id": qid, "k": 10, "family": family_key, "bundle_id": family_key,
                "intended_target_defense": spec["intended_defense"],
                **deltas,
            })

    summary_rows: List[Dict] = []
    for family_key, spec in FAMILY_SPECS.items():
        family_bundle_rows = [r for r in bundle_rows if r["family"] == family_key]
        family_delta_rows = [d for d in delta_rows if d["family"] == family_key]
        for defense in DEFENSE_NAMES:
            summary_rows.append(
                summarize_family_defense(family_key, spec["intended_defense"], defense, family_bundle_rows, family_delta_rows)
            )

    cross_matrix = build_cross_defense_failure_matrix(summary_rows)

    baseline_fields = [
        "query_id", "k", "selection_role", "question", "target_wrong_answer",
        "N_retrieved_poison", "N_retrieved_clean",
    ] + list(base_eval._NUMERIC_METRIC_KEYS)
    bundle_fields = ["family", "bundle_id", "intended_target_defense"] + baseline_fields
    delta_fields = (
        ["query_id", "k", "family", "bundle_id", "intended_target_defense"]
        + [f"delta_{k}" for k in base_eval._NUMERIC_METRIC_KEYS]
        + list(base_eval.DELTA_ALIASES.keys())
    )

    base_eval.write_csv(os.path.join(out_dir, "targeted_family_bundle_scores.csv"), bundle_fields, bundle_rows)
    base_eval.write_csv(os.path.join(out_dir, "targeted_family_bundle_deltas.csv"), delta_fields, delta_rows)
    base_eval.write_csv(os.path.join(out_dir, "targeted_family_summary_by_defense.csv"), SUMMARY_FIELDS, summary_rows)
    base_eval.write_csv(os.path.join(out_dir, "cross_defense_failure_matrix.csv"), CROSS_MATRIX_FIELDS, cross_matrix)

    first_pilot_deltas_path = os.path.join(pilot_dir, "evaluation", "mutation_bundle_deltas.csv")
    first_pilot_deltas = load_first_pilot_deltas(first_pilot_deltas_path)

    report = build_report(
        pilot_dir=pilot_dir, bundle_dir=bundle_dir, ml_model_path=args.ml_model_path,
        query_ids=query_ids, baseline_rows=baseline_rows, bundle_rows=bundle_rows,
        delta_rows=delta_rows, summary_rows=summary_rows, cross_matrix=cross_matrix,
        doc_id_mismatches=doc_id_mismatches, selected_queries=selected_queries,
        first_pilot_deltas=first_pilot_deltas,
    )
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "TARGETED_MUTATION_BUNDLE_1_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(
        f"[run_targeted_mutation_bundle_1_eval] wrote {len(baseline_rows)} baseline row(s), "
        f"{len(bundle_rows)} bundle row(s), {len(delta_rows)} delta row(s), "
        f"{len(summary_rows)} summary row(s) to {out_dir}"
    )
    print(
        f"[run_targeted_mutation_bundle_1_eval] SLM generation calls: "
        f"{models.memo_slm_answer_fn.calls} (cache hits: {models.memo_slm_answer_fn.cache_hits})"
    )
    print(
        f"[run_targeted_mutation_bundle_1_eval] LM perplexity calls: "
        f"{models.memo_causal_scorer.calls} (cache hits: {models.memo_causal_scorer.cache_hits})"
    )


if __name__ == "__main__":
    main()
