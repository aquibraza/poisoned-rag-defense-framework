# RAGDefender Fidelity Audit V2 -- Plan, Gate Status, and Claim-Impact Tracker

**Normative specification order (do not reverse):** FINAL PUBLISHED PAPER
(Kim, Lee, and Koo, "Rescuing the Unpoisoned: Efficient Defense against
Knowledge Corruption Attacks on RAG Systems," ACSAC 2025,
`docs/Rescuing_the_Unpoisoned_Efficient_Defense_Against_Knowledge_Corruption_Attacks_on_RAG_Systems_ACSAC.pdf`)
> AUTHORS' RELEASED CODE (`RAGDefender/` nested clone, `v0.2.0`) > our
LOCAL LEGACY PORT (`ragdefender_legacy` in `defense/defense_runner.py`).
**Authority rule:** the final paper governs explicit behavior; official
code fills paper-silent implementation details only (e.g. the even-length
median tie-break convention, §5).

This document supersedes prior ad-hoc fidelity notes for tracking
purposes. It does not rewrite the manuscript (`paper_latex/bigdata26_paper.tex`)
or `docs/PAPER_RESTRUCTURE_STRESS_TESTING_PLAN.md`'s legacy E1-E30
item list; see §6 for what is and is not yet reflected there.

---

## 1. Two implementations, deliberately coexisting

| | `ragdefender_legacy` | `ragdefender_paper` |
|---|---|---|
| Role | Authors'-code-faithful reproducibility comparator | FINAL-PAPER-faithful implementation (Eq. 1-7) |
| Embedder | `paraphrase-MiniLM-L6-v2` | `dunzhang/stella_en_1.5B_v5` (Stella) |
| Stage-1 combination | OR | AND |
| Stage-1 mean/median | diagonal-inclusive | self-excluded, `1/(k-1)` |
| Stage-1 threshold | hybrid `(avg_median + avg_avg)/2` | single `median(s_median)` (`s_tilde`) |
| Stage-1 flip branch | present | absent |
| Empty-safe-context behavior | historical restore-all fallback (**preserved**, §2) | `R_safe = R_tilde \ R_adv`, no fallback (**fixed**, §2) |
| `--defense` value | `ragdefender_original`/`ragdefender` (legacy alias) | `ragdefender_paper` |

Both are routed through `defense/defense_runner.py::apply_defense` via the
`ragdefender_version` parameter; `dispatch.py` exposes both as separate
`--defense` choices. Neither implementation imports the `RAGDefender/`
nested clone's `ragdefender` package at runtime (see
`docs/RAGDEFENDER_VERSION_AUDIT.md`); that package is a reproducibility/
documentation reference only.

---

## 2. STEP 1 (this session) -- empty safe-context fix

**Problem:** `_apply_defense_paper` inherited `ragdefender_legacy`'s
historical `if not clean_docs: clean_docs = doc_list` restore-all
fallback, silently returning every retrieved passage (including all
poison) whenever Stage 2 selected every passage for removal. The paper
defines the safe set conceptually as `R_safe = R_tilde \ R_adv`, with no
restore-all provision.

**Fix (`defense/defense_runner.py`):**
- **paper behavior: empty safe set remains empty.** `_apply_defense_paper`
  no longer has any restore-all fallback; if Stage 2 removes every
  passage, it returns `[]`.
- **legacy behavior: historical restore-all fallback preserved.**
  `apply_defense`'s `ragdefender_legacy` branch is byte-identical to
  before this fix.

**Tests (`tests/test_ragdefender_paper_fidelity.py::TestPaperEmptySafeContext`):**
A. synthetic case forcing Stage 2 to select every passage (via mocking
   the Stage-1 count estimator to return `len(doc_list)`) → paper variant
   returns `[]`. B. same assertion, explicit. C. same construction through
   the legacy path (`apply_defense(..., ragdefender_version="legacy")`)
   → returns the full, unmodified `doc_list`. D. an ordinary partial-removal
   case is unaffected (deterministic fixture: 4 passages, 2 removed, 2
   kept). **Status: COMPLETE, all 3 tests pass.**

---

## 3. STEP 2 (this session) -- Gate-B diagnostic taxonomy correction

**Problem:** the Gate-B diagnostic helper `_classify_query()` inferred a
`"clean-density / clean-top-pair failure"` label partly from Stage-1's
concentration AND-flags (`is_poison[adv_flag]` majority-clean check).
Stage-1 flags are **count-estimation indicator flags** -- they determine
only `N_adv` (an integer) -- not a predicted adversarial-passage subset;
Stage 2 independently decides which specific passages get removed via its
own frequency-score ranking.

**Fix (`scripts/run_ragdefender_gate_b_diagnostic.py`):**
- `_classify_query()`'s signature no longer accepts `is_poison`/`adv_flag`/
  `n_adv` at all -- it takes ONLY Stage-2 evidence (`removed_poison`,
  `removed_clean`, `residual_poison`, `top_pair_label`, `pp_count`,
  `pc_count`, `cc_count`).
- `"clean-density / clean-top-pair failure"` now fires only when: the top
  selected pair is CC, OR CC pairs are the plurality of the selected pair
  set, OR Stage 2 actually removed more clean than poison passages
  (equivalent to "clean passages dominate the top-`N_adv` frequency-score
  ranking," since the removed set *is* that ranking by construction).
- `above_mean_flags`/`above_median_flags`/`final_and_flags`/
  `final_adv_flag_indices` are retained in `gate_b_per_query.csv` as
  useful intermediate diagnostics, now explicitly documented (in the
  script's inline comments and the generated report's "Data files"
  section) as **Stage-1 count-estimation indicator flags**, not predicted
  adversarial passages.

**Verification that no existing artifact needed to change:** recomputing
`classification` for all 8 real, already-saved Gate-B rows with the
corrected function reproduces every value byte-for-byte identically (the
flawed inference path never actually fired on this dataset). **No
`gate_b_per_query.csv`/`.npy` file was touched.**

**Report wording correction (`results/diagnostics/ragdefender_gate_b/GATE_B_STELLA_FIDELITY_REPORT.md`):**
added a "Post-review correction" note and rewrote the Q4 answer to
explicitly separate the FAILURE (persists: `residual_poison=1` on
`5a8cb288...`) from the MECHANISM (does NOT reproduce: top pair is PP
under Stella, not CC as under legacy/MiniLM). No CSV/`.npy` regenerated.

**Tests:** `tests/test_run_ragdefender_gate_b_diagnostic.py::TestClassifyQuery`
(rewritten for the new signature) + new
`TestClassifyQueryIgnoresStage1Flags` (pins the signature has no
`is_poison`/`adv_flag`/`n_adv` parameter, and that a majority-clean
Stage-1-flag / all-PP Stage-2-removal shape is NOT labeled clean-density).
**Status: COMPLETE, 32 tests pass (31 passed + 1 skipped live-Stella
smoke test, as before).**

---

## 4. STEP 3 (this session) -- median-convention sensitivity (diagnostic only)

**Question:** the final paper is silent on the even-length median
tie-break convention. `ragdefender_paper` uses the authors' own
torch-style lower-of-two-middle convention under the authority rule
(§0 above) -- **unchanged, primary**. How much would the alternative
(conventional average-of-two-middle, NumPy's default) have changed Gate
B's outcomes?

**Method:** `scripts/run_ragdefender_median_sensitivity.py` implements the
average-of-two-middle variant (`_concentration_stage1_average_median`)
**entirely locally inside the script** -- it is not added to
`defense/ragdefender_internals.py`, `defense/defense_runner.py`, or any
`--defense` choice (guarded by
`tests/test_run_ragdefender_median_sensitivity.py::TestSensitivityVariantIsIsolated`).
It reuses the 8 already-saved Gate-B Stella similarity matrices; zero new
embeddings.

**Result (`results/diagnostics/ragdefender_median_sensitivity/`):**

| Metric | Value |
|---|---|
| Queries whose `N_adv` changes between conventions | **0/8** |
| Queries whose final Stage-2 removal set changes | **0/8** |
| Sole Gate-B success changes zero-residual-poison status | **No** |
| `N_adv=4` queries that become `N_adv=5` under the average convention | **0/7** |

**Interpretation:** the Gate-B count-underestimation observation is
**robust** to this specific paper-silent implementation ambiguity in the
evaluated n=8 sample (`s_tilde` shifts by up to ~0.003 on some queries,
but never crosses the AND-flag boundary for any passage). **Status:
COMPLETE**, 9 tests pass.

---

## 5. STEP 4 (this session) -- Gate C: ORACLE-COUNT decomposition

**Purpose:** separate RAGDefender error into (1) Stage-1 COUNT ESTIMATION
ERROR and (2) Stage-2 IDENTIFICATION ERROR conditional on a supplied
count, using ONLY the frozen Gate-B Stella matrices. The oracle supplies
**only the integer count** of poisoned passages (`N_poison`, observed),
never passage identities, to the unchanged `stage2_pair_frequency`.
Diagnostic control, not a deployable defense; not wired into
`defense/defense_runner.py` or `defense/dispatch.py`.

**Result (`results/diagnostics/ragdefender_gate_c_oracle_count/`):**

| | Estimated pipeline (Gate-B `N_adv`) | Oracle-count pipeline (`N_poison`) |
|---|---|---|
| Zero-residual-poison successes | 1/8 | **8/8** |
| Failures fixed by correcting only the count | -- | **7/7** |
| Additional clean passages removed by the oracle | -- | **0/8** |

**Decomposition labels:** A. COUNT-LIMITED = **7/8**; B. COUNT +
IDENTIFICATION LIMITED = 0/8; C. IDENTIFICATION LIMITED = 0/8; D.
BASELINE SUCCESS = 1/8 (the pre-existing Gate-B success,
`5a722b86...`, where `N_est` already equals the true count).

**Answers to the five Gate-C questions (full text in
`GATE_C_ORACLE_COUNT_REPORT.md`):**
1. 7/7 residual-poison failures become zero-residual-poison successes
   when only `N_adv` is corrected to the true poison count.
2. No -- oracle-count Stage 2 never removes more clean passages than the
   estimated pipeline did on this sample.
3. **Provisional (n=8):** in this eight-query diagnostic sample, residual
   poison was primarily associated with underestimation of `N_adv`;
   supplying the true poison count to the unchanged Stage-2 procedure
   eliminated 7/7 failures.
4. Stage 2 remains accurate conditional on the correct count on this
   sample (`n_label_C` = 0/8 -- no query with a correct count still
   failed).
5. `top_pair_pp` is **not** currently a discriminative success/failure
   variable: of the 8 estimated-pipeline queries whose selected-pair set
   is PP-leading, 1 is a success and 7 are failures -- PP-leading geometry
   occurs in both outcomes on this sample.

**Allowed wording (verbatim, matches the report):** "In this eight-query
diagnostic sample, residual poison was primarily associated with
underestimation of N_adv; supplying the true poison count to the
unchanged Stage-2 procedure eliminated 7/7 failures."
**NOT allowed:** "RAGDefender fails because Stage 1 is fundamentally
broken" -- this is an n=8, non-prospectively-sampled population (see §7).

**Status: COMPLETE.** `scripts/run_ragdefender_gate_c_oracle_count.py` +
`tests/test_run_ragdefender_gate_c_oracle_count.py` (17 tests, all pass).

---

## 6. Gate status tracker

| Gate | Status | Artifacts |
|---|---|---|
| Gate A (logic-isolation, MiniLM geometry) | **COMPLETE** | `results/diagnostics/ragdefender_gate_a/` |
| Gate-B readiness pass (Stella DynamicCache shim, explicit device handling) | **COMPLETE** | `defense/defense_runner.py`, `tests/test_ragdefender_paper_fidelity.py` |
| Gate B (Stella paper-fidelity gate) | **COMPLETE** | `results/diagnostics/ragdefender_gate_b/` |
| Success-case re-identification (paper-faithful, Stella) | **COMPLETE** | 1/8: `5a722b8655429971e9dc9329` (see Gate-B report §"Success-case re-identification") |
| Paper empty-safe-context fix | **COMPLETE** | §2 above |
| Gate-B taxonomy correction | **COMPLETE** | §3 above |
| Median-convention sensitivity (diagnostic) | **COMPLETE** | §4 above |
| Gate C (oracle-count decomposition) | **COMPLETE** | §5 above |
| E1 clean-anchor oracle / CORAL / MMD rerun on paper-faithful population | **PAUSED** -- gated on population expansion (§7); do not resume merely because it was the old plan | -- |
| Population expansion (prospective, unbiased HotpotQA k=10 sample) | **PLANNED, NOT EXECUTED** | §7 (design only) |
| `docs/PAPER_RESTRUCTURE_STRESS_TESTING_PLAN.md` update | **NOT YET DONE** -- that document still describes only the pre-fidelity-correction legacy E1-E30 narrative; it needs a new section describing `ragdefender_paper`/Gates A-C before it is consistent with this document | -- |
| `paper_latex/bigdata26_paper.tex` update | **NOT YET DONE, deliberately** -- manuscript Results section stays gated on the population-scale experiment (§7), per explicit instruction | -- |

---

## 7. Claim-impact updates (factual; does not rewrite the manuscript)

1. **The previous clean-density mechanism is legacy-MiniLM-specific and
   was not reproduced under Stella.** `5a8cb288...`'s failure persists
   (`residual_poison=1`) but its Stage-2 top pair is PP under Stella, not
   CC as under legacy/MiniLM (§3). Any manuscript text attributing this
   query's failure to "clean cluster denser than poison cluster" must be
   qualified as a legacy-implementation-specific observation, not a
   Stella/paper-faithful one.
2. **`top_pair_pp` is not currently a discriminative success/failure
   variable.** PP-leading Stage-2 geometry occurs in both the Gate-B
   success and all 7 Gate-B failures (Gate C Q5, §5). Prior manuscript
   framing that treated `top_pair_pp` collapse as *the* operative failure
   condition was established on the legacy/MiniLM E1-E30 population (see
   `docs/PAPER_RESTRUCTURE_STRESS_TESTING_PLAN.md` E16/C6) and has not
   been shown to transfer to the paper-faithful Stella population; it
   must not be presented as validated for `ragdefender_paper` without a
   dedicated causal experiment on the correct population.
3. **Stage-1 count underestimation is the leading candidate bottleneck on
   this n=8 sample, but remains provisional.** Gate C shows 7/7 Gate-B
   residual-poison failures are fully explained by count underestimation
   alone (§5) -- a striking, internally consistent result -- but this
   population is small (n=8) and was not prospectively/unbiasedly
   selected (it derives from the legacy cluster-viz instrumented case
   set, itself downstream of which queries had a recoverable saved
   MiniLM matrix and text). It must be tested on an expanded, bias-free
   population (§8) before being treated as an established property of
   `ragdefender_paper`.
4. **The original legacy E1/CORAL/MMD results remain historical/
   exploratory and should not be polished for the final manuscript.**
   They were computed against `ragdefender_legacy` + MiniLM geometry, not
   the paper-faithful implementation; per §6, that experimental sequence
   is PAUSED, not abandoned, pending the population-expansion decision in
   §8.

---

## 8. Population-expansion plan (DESIGN ONLY -- NOT EXECUTED)

**Motivation:** avoid basing mechanism conclusions (§7 item 3 in
particular) on the current n=8 population, which was selected via the
*legacy* RAGDefender cluster-visualization instrumentation, not
prospectively defined for `ragdefender_paper`.

**Eligible pool:** all HotpotQA multi-hop queries in the existing
retrieval/poisoning pipeline's held-out set that have a full k=10/N=5
retrieved-context record on disk (i.e. any query already run through the
"all-poison retrieved-context stress" configuration upstream of
`defense/`), excluding the 8 queries already used for Gates A-C (to avoid
double-counting the same population when reporting an expanded n).

**Inclusion/exclusion criteria (fixed BEFORE inspecting any
`ragdefender_paper` outcome, to avoid selection-on-the-dependent-variable
bias):**
- Include: any eligible query with a retrievable, non-corrupted passage
  text set (same text-recovery requirement Gate A/B already applied) and
  an observed retrieved composition that is not literally empty of
  poison or empty of clean (both-composition-types must exist to make
  `residual_poison`/`removed_clean` meaningful).
- Exclude: queries requiring new retrieval or new poisoning generation
  (out of scope per this task's constraints), and any query whose only
  available similarity geometry is MiniLM without a corresponding
  ability to re-encode with Stella from saved passage text (Gate B's own
  requirement).
- **Do not** filter by whether `ragdefender_legacy` succeeded or failed on
  a query -- that is exactly the bias Gate A/B/C's population inherited
  and this expansion must not repeat.

**Sampling approach:** prefer (A) all eligible held-out queries within
the available compute budget; if that is infeasible, (B) a
deterministically-seeded random sample fixed and recorded before any
`ragdefender_paper` run (e.g. `random.Random(seed).sample(...)` with the
seed and the resulting query-id list committed to the run's
`run_config.json` before Stage 1/2 are ever invoked on them). No
cherry-picking either way.

**Evaluation:** `ragdefender_paper` + Stella at the k=10/N=5 stress
setting (same setting as Gates A-C, for direct comparability), reporting
per query: true retrieved poison count, estimated `N_adv`, count error,
removed poison, removed clean, residual poison, Stage-2 removal
precision, Stage-2 poison recall, PP/PC/CC composition, and threshold
margins (mean and median, per Gate-B §6's caveat about the even-`k`
median-margin structural artifact).

**Separate, not mixed:** a small nominal HotpotQA **k=2** sanity-check
subset (the paper's own nominal multi-hop setting) should be run and
reported as an independent paper-configuration validation, never
aggregated into the same table/statistic as the k=10 stress-regime
results (different retrieval regime, different `N_pairs` scale).

**Decision gate after expansion:** only after this expanded baseline
exists should we decide whether the next oracle experiment should target
(A) Stage-1 count estimation specifically, (B) Stage-2 geometry
specifically (e.g. a targeted PP-vs-CC dispersion oracle), or (C) both
separately -- and whether E1/CORAL/MMD should be resumed on the expanded
paper-faithful population, redesigned, or retired in favor of a
count-estimation-targeted oracle. **Do not** default back to the old E1
plan merely because it existed before this decomposition.

**Explicitly not run by this document or by the current session:** no
new retrieval, no new poisoning generation, no new query selection beyond
this written plan, no `ragdefender_paper` run on any query outside the
existing 8 Gate-A/B/C queries.
