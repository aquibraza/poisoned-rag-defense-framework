# Mutation Bundle 1 -- Input Integrity Audit & doc_id Normalization Report

Audit of the 3 defense-*targeted* GPT mutation family files in `manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/` against the authoritative `manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_input_passages.csv`, performed **before** any further fixed-context evaluation or full retrieval rerun. No GPT/API call, `llm.query()` call, retrieval rerun, defense rerun, model training, defense-code modification, or mutated-text alteration was performed -- see 'Process confirmation' below.

## 1. Was previous-run scoring affected by the doc_id mismatches?

**No.** 15 passage(s) across 1 family file(s) (`ragdefender_targeted`) have a `doc_id` field that does not match the authoritative `(query_id, poison_slot) -> doc_id` mapping in `mutation_input_passages.csv`. However, `scripts/run_targeted_mutation_bundle_1_eval.py` (the previous evaluation run) already resolved every mutated passage's identity by `(query_id, poison_slot)` against that same authoritative CSV -- **never** by the family file's own `doc_id` -- so the wrong `doc_id` values never affected which passage's text was substituted, and the previously reported `targeted_family_bundle_scores.csv` / `targeted_family_bundle_deltas.csv` results remain valid. This audit independently re-confirms that resolution logic and additionally produces normalized files with the `doc_id` field corrected at the source.

## 2. doc_id mismatch counts

- Total mismatched passages: **15** (out of 90 audited passages across all 3 families).
- Family file(s) with mismatches: `ragdefender_targeted`.
  - `filterrag_targeted`: 0 mismatch(es).
  - `mlfilterrag_targeted`: 0 mismatch(es).
  - `ragdefender_targeted`: 15 mismatch(es).

## 3. Could all records be safely resolved by poison_slot?

**Yes.** Every audited passage across all 3 families had a known `query_id`, a unique valid `poison_slot` in `0..4`, non-empty mutated text, and a resolvable canonical `doc_id` from `mutation_input_passages.csv`. All 6 queries x 3 families x 5 slots = 90 passages were included in the normalized output.

## 4. Should any query/family be excluded from paper-level claims?

- 14 passage(s) across quer(y/ies) `5a8133725542995ce29dcbdb` matched the correct answer as a whole word, but flagged `possible_true_answer_leak_expected_named_entity_choice`: the question itself is a binary choice between two named entities (e.g. "...Henry Roth or Robert Erskine Childers?") and already names the correct answer, so a mutated passage arguing for the wrong option still has to name the correct-answer entity to construct a coherent (if false) comparison. This is expected/structural for this question type, **not** a genuine information leak beyond what the question already discloses -- not recommended as grounds for exclusion on its own, but manual review is still advisable to confirm the passage's argument favors the wrong option rather than the correct one.

Note (heuristic limitation, not an exclusion criterion): 7 passage(s) did not contain the literal `target_wrong_answer` string as a case-insensitive substring (`target_wrong_answer_not_found_by_simple_substring`). This is expected for many of these mutations, which paraphrase/imply the wrong answer through content rather than stating it verbatim (e.g. a yes/no question's mutated passage describing matching facts without literally writing "yes"); a simple string check cannot detect semantic entailment, so this flag alone does not indicate a defective mutation.

## 5. Are the normalized files safe for fixed-context rerun and full retrieval rerun?

**Fixed-context rerun: Yes**, for every query/family included in the normalized output (see Section 3). Each normalized record carries a canonical `doc_id` resolved from `mutation_input_passages.csv` by `(query_id, poison_slot)`, the original (possibly wrong) family-file `doc_id` preserved separately under `source_file_doc_id`, and byte-identical mutated text -- sufficient for `scripts/run_text_mutation_fixed_context_eval.py`-style scoring that substitutes text into the existing fixed k=10 context by `doc_id`/`poison_slot`.
- Cross-checked: all 6 audited query_id(s) have at least one row in `clean_context_passages.csv` (clean-passage counts: `5a7759fc5542993569682d60`=5, `5a8133725542995ce29dcbdb`=3, `5a8e068b5542995085b37384`=4, `5aba749055429901930fa7d8`=5, `5ae224da554299234fd043ee`=5, `5ae22b8d554299234fd0440f`=5).

**Full retrieval rerun: Conditionally yes, with an important caveat.** Normalizing `doc_id` only fixes *metadata identity* against the already-selected passage set; it does not verify that the canonical `doc_id` is independently retrievable from the underlying corpus/index in a full (non-fixed) retrieval pass -- that verification is out of scope for this audit (no retrieval was rerun here, per the strict constraints) and must happen as its own step before any full retrieval rerun's results are trusted.

## Per-family doc_id-mismatch detail

| family | query_id | poison_slot | source_file_doc_id | canonical_doc_id |
|---|---|---:|---|---|
| `ragdefender_targeted` | `5a7759fc5542993569682d60` | 0 | `adv::LM_targeted::5a7759fc5542993569682d60::0` | `adv::LM_targeted::5a7759fc5542993569682d60::187` |
| `ragdefender_targeted` | `5a7759fc5542993569682d60` | 1 | `adv::LM_targeted::5a7759fc5542993569682d60::1` | `adv::LM_targeted::5a7759fc5542993569682d60::189` |
| `ragdefender_targeted` | `5a7759fc5542993569682d60` | 2 | `adv::LM_targeted::5a7759fc5542993569682d60::2` | `adv::LM_targeted::5a7759fc5542993569682d60::186` |
| `ragdefender_targeted` | `5a7759fc5542993569682d60` | 3 | `adv::LM_targeted::5a7759fc5542993569682d60::3` | `adv::LM_targeted::5a7759fc5542993569682d60::188` |
| `ragdefender_targeted` | `5a7759fc5542993569682d60` | 4 | `adv::LM_targeted::5a7759fc5542993569682d60::4` | `adv::LM_targeted::5a7759fc5542993569682d60::185` |
| `ragdefender_targeted` | `5a8133725542995ce29dcbdb` | 0 | `adv::LM_targeted::5a8133725542995ce29dcbdb::0` | `adv::LM_targeted::5a8133725542995ce29dcbdb::22` |
| `ragdefender_targeted` | `5a8133725542995ce29dcbdb` | 1 | `adv::LM_targeted::5a8133725542995ce29dcbdb::1` | `adv::LM_targeted::5a8133725542995ce29dcbdb::21` |
| `ragdefender_targeted` | `5a8133725542995ce29dcbdb` | 2 | `adv::LM_targeted::5a8133725542995ce29dcbdb::2` | `adv::LM_targeted::5a8133725542995ce29dcbdb::24` |
| `ragdefender_targeted` | `5a8133725542995ce29dcbdb` | 3 | `adv::LM_targeted::5a8133725542995ce29dcbdb::3` | `adv::LM_targeted::5a8133725542995ce29dcbdb::23` |
| `ragdefender_targeted` | `5a8133725542995ce29dcbdb` | 4 | `adv::LM_targeted::5a8133725542995ce29dcbdb::4` | `adv::LM_targeted::5a8133725542995ce29dcbdb::20` |
| `ragdefender_targeted` | `5a8e068b5542995085b37384` | 0 | `adv::LM_targeted::5a8e068b5542995085b37384::0` | `adv::LM_targeted::5a8e068b5542995085b37384::96` |
| `ragdefender_targeted` | `5a8e068b5542995085b37384` | 1 | `adv::LM_targeted::5a8e068b5542995085b37384::1` | `adv::LM_targeted::5a8e068b5542995085b37384::95` |
| `ragdefender_targeted` | `5a8e068b5542995085b37384` | 2 | `adv::LM_targeted::5a8e068b5542995085b37384::2` | `adv::LM_targeted::5a8e068b5542995085b37384::99` |
| `ragdefender_targeted` | `5a8e068b5542995085b37384` | 3 | `adv::LM_targeted::5a8e068b5542995085b37384::3` | `adv::LM_targeted::5a8e068b5542995085b37384::98` |
| `ragdefender_targeted` | `5a8e068b5542995085b37384` | 4 | `adv::LM_targeted::5a8e068b5542995085b37384::4` | `adv::LM_targeted::5a8e068b5542995085b37384::97` |

## Files written

- `manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/normalized/ragdefender_targeted.normalized.jsonl` -- 6 query record(s).
- `manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/normalized/filterrag_targeted.normalized.jsonl` -- 6 query record(s).
- `manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/normalized/mlfilterrag_targeted.normalized.jsonl` -- 6 query record(s).
- `manual_text_mutation_pilot/hotpotqa_50q_k10/mutation_bundle_1/normalized/mutation_bundle_1_integrity_audit.csv` -- 90 row(s).

## Process confirmation

- No GPT/API calls were made.
- No `llm.query()` calls were made.
- Retrieval was not rerun.
- No defense was rerun/scored (this is a pure metadata/schema audit).
- No defense code (`defense/*.py`) was read or modified.
- No mutated text content was altered; every normalized `mutated_text` value is copied verbatim from its source family file.
- Only `doc_id` metadata was normalized (canonical value substituted; original preserved under `source_file_doc_id`).
