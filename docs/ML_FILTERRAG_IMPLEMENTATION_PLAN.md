# ML-FilterRAG Implementation Plan

**Status: approved and implemented as "ML-FilterRAG-top-k" (MVP).** This
document is the tracked repo copy of the plan approved before
implementation (originally drafted/revised as a Cursor plan,
`ml-filterrag_implementation_plan_619b3b99.plan.md`). It is reproduced here
essentially verbatim as the implementation's design record; see
`defense/ml_filterrag.py`'s module docstring and this repo's tests for the
implementation itself.

This plan is grounded in a fresh read of `defense/filterrag.py`,
`defense/dispatch.py`, `defense/diagnostics.py`, `defense/passages.py`,
`main.py`, `docs/FILTERRAG_FIDELITY_AUDIT.md`, `docs/FILTERRAG_BASELINE.md`,
`tests/README.md`, and the actual paper text (Edemacu et al. 2025, `docs/Edemacu
et al. - 2025 - ...pdf`, Section III-B2 "ML-FilterRAG", Algorithm 2, Appendix
C).

## 1. Paper fidelity audit for ML-FilterRAG

Algorithm 2 (paper, verbatim structure):

```
Input: query qi, poisoned DB D∪D̃, trained model M, top-s, SLM, LLM, Retriever
1. RetrievedItems ← Retriever(qi, D∪D̃, top-s)
2. for each dj in RetrievedItems:
     aj ← SLM(qi, dj)
     features[dj] ← Feature(qi, dj)
3. for each dj in RetrievedItems:
     pred ← M(features[dj])
     if pred == "non-adversarial": keep dj else discard dj
4. From survivors, top-k -> LLM context
```

**Paper-faithful components** (Section III-B2 + Appendix C, cited precisely):

- Reuses Freq-Density (Eq. 4) exactly as Algorithm 1 -- "Similar to the extraction method discussed in III-B1, we employ an SLM that takes the target query qi and a retrieved text dj as input and subsequently outputs aj. However, on top of the Freq-Density..." This is the *semantic*-matching Freq-Density (`matching_mode="semantic"`, threshold 0.6), not the `exact` legacy mode.
- Three additional features, verbatim: **"perplexity, joint log probability of the SLM's output aj, and sum of frequencies of semantically similar words between (qi⊕aj) and dj."**
  - **Perplexity**: attributed to "Inspired by the work in [5] [PoisonedRAG/Zou et al.], which used perplexity to attempt to filter out adversarial texts" -- i.e. perplexity of the *passage* `dj` itself (paper Fig. 1 caption: "Pair Plot for Freq-Density vs Perplexity", x-axis Freq-Density, y-axis Perplexity per data point = per passage).
  - **Joint log-probability**: explicitly "of the SLM's output aj" -- i.e. score under the SLM that generated it, not a separate model.
  - **"Sum of frequencies of semantically similar words between (qi⊕aj) and dj"**: this is the Freq-Density *numerator* alone (before dividing by `UniqueWords(dj)`) -- a distinct feature from the Freq-Density ratio itself. `freq_density_detailed()` in `defense/filterrag.py` already computes this internally as `total_freq` but currently only returns the divided ratio (`freq_density_score`) plus `matched_keyword_count` (a *distinct-keyword* count, not a frequency sum) -- see §3 below for the one small additive change needed.
- Classifier choice **per dataset** (Appendix C, Table VI): **XGBoost for NQ**, **Random Forest for HotpotQA and MS-MARCO** (reported train/test accuracy: MS-MARCO RF 0.860/0.850, NQ XGBoost 0.990/0.970, HotpotQA RF 0.990/0.982).
- Supervised training: "we use the supervised learning approach in which the labeled datasets of retrieved texts for each target query are annotated as adversarial and non-adversarial." Labels come from ground truth, never inferred from text.
- Inference: classify each retrieved item independently (`Feature(qi, dj) -> M -> predict`), then keep only items predicted "non-adversarial", then take top-k of survivors.

**Repo-practical proxies** (deviations, documented the same way `docs/FILTERRAG_FIDELITY_AUDIT.md` documents FilterRAG's):

- **SLM**: paper uses LLaMA-2/3; this repo already substitutes `google/flan-t5-small` for FilterRAG's `aj` generation (`DEFAULT_SLM_MODEL`). ML-FilterRAG reuses the exact same substitution -- no new deviation, just inherits the existing one.
- **Perplexity LM**: the paper doesn't state which model scores perplexity(dj) (only that it's "inspired by" PoisonedRAG's use of perplexity, which itself used GPT-2). This repo has no GPU-scale option, so the proxy is a small local **causal** HF LM (`distilgpt2`, ~82M params, CPU-friendly, same class of proxy as `flan-t5-small`), configurable via `--ml_filterrag_lm_model`.
- **Training data construction**: the paper trains on GPT-4o-generated *additional* adversarial texts (5 more per question, filtered to rank top-15) balanced 5-poison/5-clean per question, with the rest held out for eval. This repo **must not call GPT/API** per this task's constraints, so the practical proxy is: build the labeled dataset entirely from this repo's **existing** ground-truth `is_poison` labels (the original PoisonedRAG-style adversarial texts already generated offline by `Attacker.get_attack()` for `LM_targeted`/`hotflip`, exactly as `scripts/filterrag_score_inspection.py` already does with zero GPT calls) rather than synthesizing new poison texts. This is a real fidelity gap (smaller / less diverse poison-text distribution than the paper's GPT-4o-augmented set) that must be called out, not hidden.
- **top-s vs top-k (strengthened)**: paper retrieves an oversized `top-s` candidate pool then filters to `top-k`; this repo's harness (`main.py`) has no `--top_s`, only `--top_k` (same gap already flagged in `docs/FILTERRAG_FIDELITY_AUDIT.md` §3.3 for threshold FilterRAG). **The MVP therefore uses the current top-k harness and is an "ML-FilterRAG-top-k" implementation, not a full Algorithm-2 `top-s -> filter -> top-k` reproduction.** This must be named as such everywhere results are reported (docstrings, training/eval reports, this document itself) -- not silently equated with "ML-FilterRAG." Before any final paper-level evaluation/comparison against the paper's published numbers, a `top-s` retrieval + filter + `top-k` harness must be added or evaluated; this MVP does not attempt that harness change.

**Unknowns / assumptions requiring an explicit call-out:**

- Whether "perplexity" is of `dj` alone or of `(qi ⊕ dj)`/prompt+passage -- assumption: `dj` alone (matches Fig. 1's framing and the PoisonedRAG-perplexity precedent).
- Whether "joint log probability of aj" is under the SLM used to *generate* it (assumption: yes, most literal reading, and reusable for free from the already-loaded `flan-t5-small` pipeline via teacher-forced scoring) vs. some independent LM. See §3 for the exact loss-to-joint-logprob conversion this requires (not a mean-loss approximation).
- Exact feature vector order/normalization used by the paper's classifiers is unspecified -- the MVP documents its own explicit `DEFAULT_FEATURE_NAMES` ordering, restricted to the 4 paper-cited features (see §3's default-vs-auxiliary split).
- Paper's per-dataset classifier choice (Random Forest for HotpotQA/MS-MARCO, XGBoost for NQ) is followed as the *paper-aligned default* per dataset; if `xgboost` isn't installed, NQ results produced with a Random Forest substitute must be explicitly labeled **proxy results**, not a paper-faithful NQ reproduction (see §4, §9, and the summary table in §10).

## 2. Data construction plan

**Ground-truth-only labels.** Every training/eval row's label comes from `RetrievedPassage.is_poison` (propagated from attack-injection ground truth in `defense/passages.py::label_passages`, itself set at construction time in `main.py`/`scripts/filterrag_score_inspection.py` via `source`/`is_poison` keys attached from `Attacker.get_attack()` output) -- never inferred from passage text, matching the repo's existing invariant (`defense/passages.py` module docstring).

**Leakage prevention -- query-level splits:**

- Split assignment (`train`/`test`) is decided **per `query_id`, globally within a dataset**, before any passage rows are generated, via a deterministic seeded shuffle of the *set* of `query_id`s for that dataset (not passage rows, not per-attack subsets). All passage rows for a given `query_id` -- across **every** `k`, **every** `N`, and **every** `attack_method` used to build that dataset's CSV -- go entirely into one split. **The same `query_id` must never appear in `train` under one `attack_method` and `test` under another** -- the split key is `(dataset, query_id)`, full stop; `attack`/`k`/`N` are never part of the split decision.
- Because the paper trains one model **per dataset**, the split is computed once per `dataset` (not per `(dataset, attack_method)` group). The dataset CSV still records `dataset`, `attack`, `k`, `N` (adv_per_query), `query_id`, `doc_id`, `is_poison`, `split` for every row, so downstream scripts can verify/re-derive the split rather than trust a single boolean blindly.
- **If training/evaluation is intentionally restricted to a single `attack_method`** (e.g. train and evaluate only on `LM_targeted` rows, or train on one attack and evaluate cross-attack generalization on another), that restriction must be stated explicitly in the script's docstring/report -- it is a named experimental setting, not an accident of how the CSV happened to be filtered -- and the same no-query_id-overlap assertion below still applies *within* that restricted setting.
- The dataset-builder script asserts (fails loudly, not silently) that `set(train.query_id) & set(test.query_id) == set()` before writing output, and this same assertion is re-run by `scripts/train_ml_filterrag.py` against whatever subset of rows it's actually about to train/evaluate on.
- Labels are only ever used offline (during dataset construction / training / evaluation-report generation) -- never as an input feature, and never available to `ml_filterrag_defense()` at inference time (it only ever sees `RetrievedPassage.text`/`query`, exactly like every other defense in `defense/dispatch.py`).

## 3. Feature extraction design -- `defense/ml_filterrag.py`

One small **additive, backward-compatible** change first: `freq_density_detailed()` in `defense/filterrag.py` currently computes `total_freq` internally but discards it in favor of the ratio; a new `"matched_freq_sum"` key (the raw numerator, i.e. paper's "sum of frequencies of semantically similar words") is added to its returned dict. This is purely additive (no existing key removed/renamed), so no existing caller (`freq_density()`, `score_passages()`, all pre-existing tests) breaks.

`defense/ml_filterrag.py` computes a **superset** feature dict per passage (all features below, always computed, always logged), but only a **strict subset of 4** feeds the default classifier -- this split is load-bearing for paper fidelity and is explicit in code (`DEFAULT_FEATURE_NAMES` vs. `AUXILIARY_FEATURE_NAMES`), not just in comments:

**`DEFAULT_FEATURE_NAMES` (exactly these 4, paper-cited, used by the default classifier):**

- `freq_density_score` -- via `defense.filterrag.freq_density_detailed(..., matching_mode="semantic", semantic_threshold=0.6, semantic_matcher=get_semantic_word_matcher())` (paper feature, Eq. 4).
- `matched_freq_sum` -- from the same call's new key (paper feature: "sum of frequencies of semantically similar words between (qi⊕aj) and dj").
- `perplexity` -- of `dj` alone, via a lazily-loaded local **causal** LM (`CausalLMScorer`, mirroring `SemanticWordMatcher`'s lazy-load/caching pattern): `--ml_filterrag_lm_model` (default `distilgpt2`), computed as `exp(mean_token_nll)` over `dj`'s tokens (paper feature: "perplexity", inspired by [Zou et al./PoisonedRAG]'s use of perplexity).
- `slm_answer_logprob` -- **joint** log-probability of `aj` under the SLM that generated it (paper feature: "joint log probability of the SLM's output aj") -- see the dedicated conversion note immediately below; this is **not** the raw mean training loss.

**`AUXILIARY_FEATURE_NAMES` (always computed + logged in diagnostics/dataset CSVs, but excluded from the default classifier's feature vector):**

- `exact_freq_density_score` -- a second call to `freq_density_detailed()` with `matching_mode="exact"`. Not a paper feature.
- `matched_keyword_count`, `unique_word_count` -- already-existing `freq_density_detailed()` keys; useful diagnostics, not paper-cited ML-FilterRAG features.
- `slm_answer_length` -- token count of `aj` (reuses `defense.filterrag._tokenize`). Not paper-cited as an ML-FilterRAG feature.
- `passage_length` -- token count of `dj`. Not a paper feature.
- `query_passage_lexical_overlap` -- simple Jaccard/overlap ratio between `tokenize(query)` and `tokenize(dj.text)`. Repo-only diagnostic, not a paper feature.
- `retrieval_score` (**repo-only, not in the paper**) -- `passage.retrieval_score` when present.

**Explicit statement: feeding any `AUXILIARY_FEATURE_NAMES` entry into the classifier -- via a CLI flag, a script default, or a future edit -- produces a *repo-augmented ML-FilterRAG variant*, not the paper-aligned baseline, and any report of results built that way must say so explicitly rather than calling it "ML-FilterRAG."** The MVP's default classifier training/inference path uses `DEFAULT_FEATURE_NAMES` only; auxiliary features are opt-in via an explicit, clearly-labeled ablation caller, never silently on by default.

**`slm_answer_logprob` -- joint log-probability, not mean loss (correction):** for an encoder-decoder SLM such as `flan-t5-small`, HuggingFace's `model(**encoder_inputs, labels=answer_ids)` returns a **mean** cross-entropy loss averaged over the non-pad answer tokens (`loss = -mean_i log P(token_i)`), not a sum. The paper's feature is the **joint** log-probability of the whole answer, `log P(aj) = sum_i log P(token_i) = -loss * n`, where `n` is the count of non-pad answer tokens actually scored (from the labels' attention mask, i.e. excluding any `-100`/pad-ignored positions). `defense/ml_filterrag.py` performs this `-loss * n` conversion explicitly (never returns the raw `loss` as-is, and never silently treats mean loss as if it were the joint log-probability) -- reusing the *same* cached pipeline `defense.filterrag._get_local_hf_slm_pipeline(model_name, device)` used for `aj` generation, via its public `.model`/`.tokenizer` attributes, so no extra model load is needed. A dedicated unit test (§8) verifies this exact `-loss * n` conversion against a fake model with a known loss and known token count.

All feature extraction is dry-run/no-generation-safe: it never calls `llm.query()` / any GPT/PaLM/Vicuna API -- only local HF models (`flan-t5-small` for `aj`, a causal LM for perplexity), exactly like `--defense filterrag` already does. `transformers`/`torch` stay lazily imported.

`extract_features(query, passages, *, slm_answer_fn, matching_mode="semantic", semantic_threshold=0.6, lm_model_name=DEFAULT_LM_MODEL) -> List[Dict]` is the main entry point, returning one dict per passage with **every** feature above (default + auxiliary) plus `doc_id` (so it can be joined back to a `RetrievedPassage`) -- auxiliary features are always computed for diagnostics/dataset-CSV completeness, they are just never implicitly fed to the classifier. `DEFAULT_FEATURE_NAMES` and `features_to_matrix(feature_dicts, feature_names)` produce the fixed-column-order `numpy` array the classifier actually consumes for the paper-aligned baseline; a caller wanting the repo-augmented ablation passes `feature_names=DEFAULT_FEATURE_NAMES + AUXILIARY_FEATURE_NAMES` (or a chosen subset) to the same helper, explicitly.

## 4. Classifier design

```
class MLFilterRAGClassifier:
    def __init__(self, model_type="random_forest", feature_names=DEFAULT_FEATURE_NAMES, **model_kwargs): ...
    def train(self, X, y) -> "MLFilterRAGClassifier": ...   # fits in place, returns self
    def predict_proba(self, X) -> np.ndarray: ...            # P(is_poison)
    def predict(self, X, threshold=0.5) -> np.ndarray: ...
    def save(self, path) -> None: ...                        # joblib, bundles model + feature_names + meta
    @classmethod
    def load(cls, path) -> "MLFilterRAGClassifier": ...
```

- **Paper-aligned classifier choice is per-dataset, not a single global default** (Appendix C, Table VI): **Random Forest is paper-aligned for HotpotQA and MS-MARCO**; **XGBoost is paper-aligned for NQ**. `scripts/train_ml_filterrag.py` defaults `--model_type` based on `--dataset` (`random_forest` for `hotpotqa`/`msmarco`, `xgboost` for `nq`) rather than one hardcoded global default.
- `RandomForestClassifier` (`sklearn.ensemble`, already a `requirements.txt` dependency: `scikit-learn==1.6.1`) -- always available.
- `XGBClassifier` (`xgboost`) -- **not currently installed** (`pip show xgboost` fails in this repo's environment). `model_type="xgboost"` raises a clear `ImportError`-derived message ("pip install xgboost to use model_type='xgboost'; falling back is not automatic") rather than making it a hard import-time dependency of `defense/ml_filterrag.py`.
- **Explicit proxy-labeling requirement for NQ**: if `xgboost` is not installed and NQ training/evaluation falls back to (or is explicitly run with) `model_type="random_forest"`, every artifact/report produced from that run (training report, model `training_meta`, evaluation comparison report) **must label the result "NQ Random-Forest proxy result," not a paper-faithful NQ reproduction** -- `scripts/train_ml_filterrag.py` refuses to silently substitute Random Forest for a requested-but-unavailable `xgboost` on NQ; it either fails with an actionable "pip install xgboost" message or proceeds only after an explicit `--allow_proxy_classifier` flag that also stamps the proxy label into the artifact's `training_meta` and the report filename/header.
- Artifact format: `joblib.dump({"model": ..., "model_type": ..., "feature_names": [...], "threshold_default": 0.5, "training_meta": {...}}, path)` -- one file, self-describing, no separate metadata sidecar needed. `training_meta` includes dataset/attack/k/N/feature-extractor config (matching-mode, semantic threshold, LM model names) used to build the training data, so a loaded model documents its own provenance.
- CLI-configurable inference threshold: `--ml_filterrag_threshold` (default 0.5), independent of whatever threshold was used to report training-time metrics.

## 5. Defense integration

`defense/dispatch.py`:

- `"ml_filterrag"` added to `DEFENSE_CHOICES`.
- `run_defense(...)` gains: `ml_filterrag_model_path`, `ml_filterrag_threshold=0.5`, `ml_filterrag_matching_mode="semantic"` (default **semantic**, not `exact` -- unlike `filterrag_matching_mode`, there is no pre-existing `ml_filterrag` behavior to preserve backward compatibility with, so it defaults straight to paper-faithful), `ml_filterrag_semantic_threshold=DEFAULT_SEMANTIC_THRESHOLD`, `ml_filterrag_lm_model` (perplexity LM, default `distilgpt2`) -- and reuses the existing `filterrag_slm_model`/`filterrag_slm_device` flags for `aj` generation (see §9).
- `name == "ml_filterrag"` branch: load-and-cache the classifier artifact (module-level cache keyed by resolved absolute path, mirroring `_SEMANTIC_MATCHER_CACHE`/`_SLM_PIPELINE_CACHE`), call `defense.ml_filterrag.extract_features(...)`, `classifier.predict_proba(...)`, threshold, drop predicted-poison passages, return `(kept_passages, diag_extra)`.
- Missing/invalid `--ml_filterrag_model_path` raises a clear `ValueError`/`FileNotFoundError` immediately (not a silent no-op or a cryptic joblib traceback).
- Does **not** touch `filterrag`/`filterrag_query_only`/any existing defense branch.

`main.py` CLI additions (mirrors the existing `--filterrag_*` flag block exactly in style/help-text density):

- `--ml_filterrag_model_path` (str, default `None`; required only when `--defense ml_filterrag`, validated inside `run_defense`/dispatch, not argparse, so `parse_args()` alone still round-trips cleanly in tests).
- `--ml_filterrag_threshold` (float, default 0.5).
- `--ml_filterrag_lm_model` (str, default `"distilgpt2"`) -- perplexity(dj) causal LM.
- `--ml_filterrag_matching_mode` (choices `exact`/`semantic`, default `"semantic"`).
- `--ml_filterrag_semantic_threshold` (float, default 0.6).

## 6. Training/evaluation scripts

- **`scripts/build_ml_filterrag_dataset.py`** -- structurally follows `scripts/filterrag_score_inspection.py`'s `build_passage_records()` pattern (reproduces `main.py`'s retrieval + offline `Attacker.get_attack()` injection, zero live GPT/API calls), but calls `defense.ml_filterrag.extract_features()` instead of just `score_passages()`. Sweeps multiple `k`/`N`/`attack_method` values for a larger sample. Computes the query-level train/test split (per §2) up front and writes one row per `(dataset, attack, k, query_id, doc_id)` with every feature column + `is_poison` + `split`, plus a companion JSON recording the exact feature-extractor config (matching mode, semantic threshold, SLM/LM model names, split seed) for provenance. Asserts no query_id leakage before writing.
- **`scripts/train_ml_filterrag.py`** -- reads the dataset CSV(s), filters `split=="train"`, trains `MLFilterRAGClassifier` on `DEFAULT_FEATURE_NAMES` only (default `--model_type` chosen per-dataset per §4, with the NQ proxy-labeling rule from §4 enforced when `xgboost` is unavailable; `--model_type` is still overridable), evaluates on `split=="test"` (held-out query_ids only), reports precision/recall/F1/ROC-AUC (guarded: skipped with a clear message if the test split has only one class present)/confusion matrix, writes the model artifact to `models/ml_filterrag/<dataset>_<attack>_<model_type>_<timestamp>.joblib`, and a human-readable report to `results/diagnostics/ml_filterrag/<run_name>_TRAIN_REPORT.md` (+ a metrics CSV).
- **`scripts/evaluate_ml_filterrag.py`** -- runs `defense.dispatch.run_defense("ml_filterrag", ...)` end-to-end against a trained artifact over held-out queries (reusing the same offline retrieval+injection harness), computing the identical detection-quality metrics `defense/diagnostics.py` already knows how to compute (poison recall, clean FPR, residual poison fraction), and writes a side-by-side comparison report against threshold FilterRAG (`--defense filterrag --filterrag_matching_mode semantic --filterrag_semantic_threshold 0.6`) on the same query set. Dry-run only by default -- no live generation, no `llm.query()`.

All three scripts explicitly print/assert "no GPT/API call made" in their own docstrings, and none call `llm.query()`.

## 7. Diagnostics

- **Per-passage** (analogous to `diag_extra["filterrag_scores"]`, a new `diag_extra["ml_filterrag_predictions"]` key -- not persisted to the main JSONL schema by default, same treatment as `filterrag_scores` today): per passage, `doc_id`, the full feature dict (§3), `predicted_proba`, `predicted_label`, `removed` (bool). `diag_extra` also records `model_path`, `model_artifact_hash` (md5 of the model file, so a diagnostic run is traceable to an exact artifact), `feature_names`, `model_type`, `threshold`, `matching_mode`, `semantic_threshold`, `slm_model`, `lm_model`, and `paper_aligned` (bool: whether this run used the paper-aligned defaults, or a proxy/ablation configuration).
- **Per-query**: no new schema changes needed -- `build_diagnostic_record()` already derives `removed_poison`, `removed_clean`, `poison_recall`, `clean_false_positive_rate`, `residual_poison_count/fraction`, and (when generation is run) ASR fields purely from `retrieved_passages`/`kept_passages`, identically for any `defense` name including `ml_filterrag`. `diag_extra["notes"]` for `ml_filterrag` records `model_path`, `threshold`, `model_type`, matching mode/threshold, LM model name -- mirroring `filterrag_defense()`'s `notes` string convention.

## 8. Tests

New `tests/test_ml_filterrag.py` (no third-party deps beyond what's already used by `test_filterrag.py`; reuses that file's `FakeSemanticMatcher` pattern and a fake/deterministic causal-LM double):

- Feature extractor returns finite (`math.isfinite`), non-`NaN` values for a normal passage, and doesn't crash on an empty passage/empty keyword set.
- Semantic Freq-Density feature is byte-identical to calling `defense.filterrag.freq_density_detailed(..., matching_mode="semantic")` directly (proves no reimplementation drift).
- `matched_freq_sum`'s new key is additive: existing `freq_density()`/`score_passages()`/`filterrag_defense()` callers and existing `tests/test_filterrag.py` cases are unaffected (no key removed/renamed).
- Perplexity/logprob features are deterministic given a fake/local model double.
- **Dedicated `slm_answer_logprob` conversion test**: given a fake encoder-decoder model that returns a known, fixed `loss` value and a known non-pad answer-token count `n`, asserts `slm_answer_logprob == -loss * n` exactly (not `loss`, not `-loss`, not a per-token average).
- `DEFAULT_FEATURE_NAMES` contains exactly `["freq_density_score", "matched_freq_sum", "perplexity", "slm_answer_logprob"]` and no auxiliary feature; a separate test asserts the default classifier's training matrix column count/order matches this 4-tuple exactly.
- Dataset-builder split logic: given a synthetic list of `query_id`s, asserts zero overlap between train/test query_id sets, and that the split is deterministic for a fixed seed.
- `MLFilterRAGClassifier.train()`/`predict_proba()`/`predict()` work end-to-end on small synthetic (X, y) data (no real corpus needed).
- `save()`/`load()` round-trip: predictions from a loaded artifact match the pre-save model's predictions exactly on the same input.
- `defense.dispatch.run_defense("ml_filterrag", ...)` removes exactly the passages predicted poison by a fake/mocked classifier.
- Missing/invalid `--ml_filterrag_model_path` raises a clear error both at the `defense/ml_filterrag.py` load layer and via `defense/dispatch.py`.
- No `openai`/`google.generativeai`/other GPT-API import anywhere in `defense/ml_filterrag.py` or the new scripts.
- `xgboost` stays optional: `model_type="random_forest"` works with no `xgboost` installed; `model_type="xgboost"` without `xgboost` installed raises the documented clear error.
- New `--ml_filterrag_*` CLI flags parse with correct defaults/choices and are forwarded into the `run_defense(...)` call site.
- Full existing suite (`python -m unittest discover -s tests -v`) still passes unmodified for every pre-existing test file.

## 9. Risks and decisions (resolved defaults used by the implementation)

- **Perplexity/logprob LM choice**: `distilgpt2` for perplexity(dj) (causal LM family, CPU-friendly, same size class as `flan-t5-small`), and reusing the *same* `flan-t5-small` SLM pipeline (via its `.model`/`.tokenizer`) for `slm_answer_logprob` rather than a second model.
- **All 7 `AUXILIARY_FEATURE_NAMES` entries**: compute-and-log-only (always present in the per-passage feature dict / dataset CSV for future ablation work), **excluded from the default classifier's feature vector**.
- **First classifier**: Random Forest for HotpotQA/MS-MARCO (paper-aligned, zero new hard dependencies); for NQ, `xgboost` first (paper-aligned) with a *labeled-proxy* Random Forest fallback only behind an explicit `--allow_proxy_classifier` flag.
- **Dataset source for MVP**: built fresh via `scripts/build_ml_filterrag_dataset.py` (sweep of `k`/`N`/`attack_method`), not mined from old ad hoc diagnostics directories.
- **top-s/top-k harness mismatch**: not built in this MVP (same deferral as threshold FilterRAG, `docs/FILTERRAG_FIDELITY_AUDIT.md` §3.3); every artifact/report from this MVP is labeled "ML-FilterRAG-top-k," not bare "ML-FilterRAG."
- **`ml_filterrag`'s own SLM flags**: reuses `--filterrag_slm_model`/`--filterrag_slm_device` for the `aj`-generation step, rather than introducing duplicate `--ml_filterrag_slm_*` flags; only the *new* perplexity LM (`--ml_filterrag_lm_model`) has its own flag.

## 10. Paper-faithful vs. repo-proxy summary

| Component | Paper-faithful implementation | Repo proxy/deviation |
|---|---|---|
| SLM (generates `aj`) | LLaMA-2 or LLaMA-3 | `google/flan-t5-small` (already the repo's existing FilterRAG substitution, inherited unchanged -- `defense/filterrag.py::DEFAULT_SLM_MODEL`) |
| Semantic word matching | `sentence-transformers/all-MiniLM-L6-v2`, cosine similarity, threshold 0.6 (Section IV-B2) | **Faithful as-is** -- reused directly from `defense/filterrag.py::SemanticWordMatcher`/`get_semantic_word_matcher()`, no repo deviation for this component |
| Classifier choice | Random Forest (HotpotQA, MS-MARCO); XGBoost (NQ) -- Appendix C, Table VI | Random Forest available by default (`scikit-learn`, already a dependency); XGBoost **not currently installed** -- NQ results using a Random Forest substitute must be labeled "NQ Random-Forest proxy result," not a paper-faithful NQ reproduction (§4) |
| Training data construction | GPT-4o generates 5 additional adversarial texts/question (filtered to top-15 rank), balanced 5-poison/5-clean/question for training, rest held out | **No GPT/API calls permitted in this repo per task constraints** -- dataset built entirely from existing ground-truth `is_poison` labels on the original offline-generated (`LM_targeted`/`hotflip`) adversarial texts; smaller/less diverse poison-text distribution than the paper's GPT-4o-augmented set |
| top-s / top-k | Retrieve oversized `top-s` candidate pool, filter, then take `top-k` survivors for the LLM (Algorithm 2) | This repo's harness (`main.py`) retrieves exactly `top_k` once, no `top-s` superset -- MVP is "ML-FilterRAG-top-k," not the full Algorithm-2 pipeline; a `top-s` harness is future work, not built here (§1, §9) |
| Perplexity LM | Unspecified in the paper (only "inspired by" PoisonedRAG's GPT-2-based perplexity use) | `distilgpt2` (small local causal LM, configurable via `--ml_filterrag_lm_model`) -- an assumption, not a stated paper choice |
| Feature vector fed to the classifier | Freq-Density, perplexity, joint log-probability of `aj`, sum of frequencies of semantically similar words (Section III-B2) -- exactly 4 features | `DEFAULT_FEATURE_NAMES` restricted to exactly these 4 (`freq_density_score`, `matched_freq_sum`, `perplexity`, `slm_answer_logprob`); 7 additional repo-only/diagnostic features are computed and logged but excluded from the default classifier -- adding any of them would produce a repo-augmented variant, not the paper baseline (§3) |

## Implementation sequence (as executed)

1. `docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md` (this document).
2. Additive change to `defense/filterrag.py`: `matched_freq_sum` added to `freq_density_detailed()`'s return dict; existing `tests/test_filterrag.py` confirmed passing with no regression.
3. `defense/ml_filterrag.py`: feature extraction (`extract_features`, `DEFAULT_FEATURE_NAMES`, `AUXILIARY_FEATURE_NAMES`, `features_to_matrix`), `CausalLMScorer` (perplexity), the `-loss * n` SLM-answer joint-logprob helper (`slm_answer_joint_logprob`), `MLFilterRAGClassifier` (train/predict_proba/predict/save/load, per-dataset paper-aligned model-type default via `paper_aligned_model_type`).
4. `defense/dispatch.py` (`ml_filterrag` branch, `DEFENSE_CHOICES`, `run_defense` kwargs) + `main.py` CLI flags.
5. `tests/test_ml_filterrag.py`, extended `tests/test_dispatch_smoke.py`, added `tests/test_main_cli_ml_filterrag.py`; full suite run.
6. `scripts/build_ml_filterrag_dataset.py`.
7. `scripts/train_ml_filterrag.py`.
8. `scripts/evaluate_ml_filterrag.py`.
9. `tests/README.md`, `docs/FILTERRAG_BASELINE.md` ("Deferred work" section) updated to reference `ml_filterrag`.

## Files created / modified

- Created: `docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md`, `defense/ml_filterrag.py`, `scripts/build_ml_filterrag_dataset.py`, `scripts/train_ml_filterrag.py`, `scripts/evaluate_ml_filterrag.py`, `tests/test_ml_filterrag.py`, `tests/test_main_cli_ml_filterrag.py`.
- Modified: `defense/filterrag.py` (additive `matched_freq_sum` key only), `defense/dispatch.py`, `main.py`, `tests/test_dispatch_smoke.py`, `tests/README.md`, `docs/FILTERRAG_BASELINE.md`, `.gitignore` (added `models/` output dir).

## Commands to run (no live GPT/API)

```bash
python -m unittest discover -s tests -v

python scripts/build_ml_filterrag_dataset.py \
  --eval_dataset hotpotqa --k_values 5 10 --N 5 --max_queries 20 \
  --out_dir results/diagnostics/ml_filterrag_dataset_hotpotqa

python scripts/train_ml_filterrag.py \
  --dataset_csv results/diagnostics/ml_filterrag_dataset_hotpotqa/features.csv \
  --model_type random_forest --out_dir models/ml_filterrag

python scripts/evaluate_ml_filterrag.py \
  --eval_dataset hotpotqa --ml_filterrag_model_path models/ml_filterrag/hotpotqa_random_forest.joblib \
  --dry_run True
```
