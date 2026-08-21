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
| Eq. (3) structural count-ceiling proof + validation | **COMPLETE** | §9 below |
| Prospective population freeze (STEP 3) | **COMPLETE** | §9 below, `results/diagnostics/ragdefender_expanded_baseline/PROSPECTIVE_POPULATION_FREEZE.md` |
| Expanded paper-faithful baseline (STEP 4) | **COMPLETE** (n=42) | `results/diagnostics/ragdefender_expanded_baseline/` |
| Expanded Gate-C oracle-count decomposition (STEP 5) | **COMPLETE** (n=42) | `results/diagnostics/ragdefender_expanded_gate_c/` |
| Mechanism decision report (STEP 6) | **COMPLETE** -- Decision C (two regime-conditioned failure modes); corrected re: Regime A (untested) / Regime D (degenerate oracle evidence) | `results/diagnostics/ragdefender_mechanism_decision/` |
| E1 clean-anchor oracle / CORAL / MMD rerun on paper-faithful population | **STILL PAUSED** -- the mechanism decision (§9.5) recommends two NEW Stage-1/Stage-2 oracles, not a resumption of E1/CORAL/MMD; do not resume merely because it was the old plan | -- |
| Population expansion (prospective, unbiased HotpotQA k=10 sample) | **COMPLETE** (n=42, superseded §8's design-only status; see §9) | §9 |
| Nominal HotpotQA k=2 mathematical/code/consistency audit (STEP 2-6 of the k2 audit task) | **COMPLETE** (math proof, authors'-code audit, 3-column consistency, synthetic tests) -- **real k=2 retrieval reproduction NOT run** | §9.6, `results/diagnostics/ragdefender_k2_consistency/` |
| Environment reproducibility bridge (Gate B + n=42 expanded-baseline 5-query sample) | **COMPLETE** -- Gate B fixture: byte-identical re-encoding, identical recorded env. n=42 STEP-4 run: package-version provenance **UNRESOLVED** (recorded env differs from current); 5-query sample: 0/5 byte-identical, 5/5 decision-stable (N_adv/removed-set/outcome/Gate-C label all unchanged) | §9.7, `results/diagnostics/ragdefender_environment_bridge/` |
| `docs/PAPER_RESTRUCTURE_STRESS_TESTING_PLAN.md` update | **NOT YET DONE** -- that document still describes only the pre-fidelity-correction legacy E1-E30 narrative; it needs a new section describing `ragdefender_paper`/Gates A-C before it is consistent with this document | -- |
| `paper_latex/bigdata26_paper.tex` update | **NOT YET DONE, deliberately** -- manuscript Results section stays gated on the population-scale experiment (§9), per explicit instruction | -- |

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
   this n=8 sample, but remains provisional -- UPDATE: tested and
   refined on the n=42 prospective population (§9.5).** Gate C shows 7/7
   Gate-B residual-poison failures are fully explained by count
   underestimation alone (§5) -- a striking, internally consistent result
   -- but this population is small (n=8) and was not
   prospectively/unbiasedly selected. The n=42 prospective expansion
   (§9.4-9.5) confirms that **at the ceiling (Regime B), Stage-1 count
   underestimation completely explains all observed baseline failures in
   this prospective sample** (14/14 failures fully resolved by count
   correction alone); **below-ceiling behavior (Regime A) remains
   untested, because the current population contains no Regime-A
   queries** (0/42). Above the ceiling (Regime C), count correction is
   only a PARTIAL explanation (only 4/20 failures fully resolved by count
   correction alone; the remaining 16/20 require additional Stage-2
   identification correction). See the mechanism decision report for the
   full, regime-conditioned claim, including the correction that the
   Regime-D (all-poison) oracle result is evidentially degenerate for
   Stage-2 identification quality, not usable as a "Stage 2 is accurate"
   claim.
4. **The original legacy E1/CORAL/MMD results remain historical/
   exploratory and should not be polished for the final manuscript.**
   They were computed against `ragdefender_legacy` + MiniLM geometry, not
   the paper-faithful implementation; per §6, that experimental sequence
   is PAUSED, not abandoned, pending the population-expansion decision in
   §8.

---

## 8. Population-expansion plan (historical design note -- superseded by §9)

> **Status update:** the design below was written before execution. §9
> records what was actually frozen/run (STEPs 3-6 of the population-expansion
> sequence) and takes precedence over any detail here that differs (e.g. §9's
> eligible pool is the 42-query set below, Option A, not a to-be-decided
> sample). This section is kept for its rationale/history, not as the current
> source of truth.



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

---

## 9. Stage-1 count ceiling and poison-fraction stress design

This section formalizes and operationalizes §7 item 3 / §8's plan: it makes
retrieved poison count/fraction an explicit, independent stress axis, and
records the actual (not merely planned) prospective population and its
regime distribution.

### 9.1 Structural Stage-1 count ceiling (Eq. 3)

**Claim:** `N_adv <= floor(k/2)` for the production `ragdefender_paper`
HotpotQA Stage-1 estimator (`concentration_stage1_paper`, Eq. 3's
mean-AND-median self-excluded concentration test), **unconditionally**, for
any `k >= 2` and any similarity matrix.

**Proof sketch (full proof: `results/diagnostics/ragdefender_count_ceiling/COUNT_CEILING_ANALYSIS.md`):**
Because `s_tilde` is the median of the `k` values `{s_median_i}`, at most
`floor(k/2)` of them can be strictly greater than their own median (this
holds under BOTH the primary lower-of-two-middle median convention and the
diagnostic average-of-two-middle convention -- same numeric ceiling either
way, proven by a sorted-order-statistic argument). Since
`N_adv = #{i : above_mean_i AND above_median_i} <= #{i : above_median_i}
<= floor(k/2)` (an intersection is never larger than either operand), the
bound follows without reference to Stella geometry, query semantics, poison
wording, or Stage 2 at all.

**Special case `k=2`:** `N_adv` is **provably always exactly 0** (strictly
stronger than the `floor(2/2)=1` ceiling), because matrix symmetry forces
both passages' self-excluded means to be identical, so neither can be
*strictly* greater than their shared mean. See analysis §5 for the full
argument; this is directly relevant to interpreting any future nominal
HotpotQA `k=2` sanity check.

**Programmatic validation:** `scripts/run_ragdefender_count_ceiling_validation.py`
tests `k in {2, 3, 4, 5, 6, 10, 11}` across random/edge-fixture symmetric
matrices and both median conventions -- **1414/1414 checks, 0 violations**
(see `results/diagnostics/ragdefender_count_ceiling/synthetic_validation.csv`).
Hand-computable edge fixtures reach the ceiling exactly for 6/7 required `k`
values (all except `k=2`, per the special case above).

**Tests:** `tests/test_ragdefender_stage1_count_ceiling.py` (17 tests,
production convention + diagnostic-convention comparison + AND-can-only-
reduce property + edge fixtures + `k=2` special case). **Status: COMPLETE.**

### 9.2 Threat-model consequence of the ceiling

Two structurally different situations, not to be conflated:

- **ALL-POISON RETRIEVED CONTEXT** (`C=0`, e.g. `k=5`, poison=5, clean=0):
  **violates** the paper's stated "at least one benign passage retrieved"
  assumption. This is Regime D below -- a threat-model-violation stress, not
  a within-assumption failure.
- **MAJORITY-POISON BUT THREAT-MODEL-COMPLIANT CONTEXT** (`C>=1`, e.g.
  `k=10`, poison=6, clean=4): still contains a benign passage, so it does
  **not** violate the stated assumption. But if the true poison count
  exceeds `floor(k/2)`, Eq. (3) **cannot** return the true count -- a
  structural estimator limitation, independent of Stella geometry, query
  semantics, poison wording, or Stage-2 quality. This is Regime C below.

**Allowed wording:** "Under Eq. (3), the estimated adversarial count cannot
exceed `floor(k/2)` under the declared median-threshold interpretation."
**Not allowed:** "RAGDefender cannot defend majority poisoning" (Stage-2/
system-level behavior is a separate question, analyzed independently via
the oracle-count decomposition).

### 9.3 Poison-count regimes (independent variable: M)

For a retrieved context of size `k`, define:

- `M` = observed retrieved poison count, `C` = observed retrieved clean
  count (`M + C = k`), `rho = M / k`, `count_error = N_adv - M`.

| Regime | Condition | Threat-model status |
|---|---|---|
| A -- BELOW CEILING | `M < floor(k/2)` | compliant |
| B -- AT CEILING | `M == floor(k/2)` | compliant |
| C -- ABOVE CEILING / MAJORITY POISON | `M > floor(k/2)` AND `C >= 1` | compliant, but Eq. (3) structurally cannot reach `M` |
| D -- ALL POISON | `C == 0` | **violates** the stated assumption |

`D` is checked first and is mutually exclusive with A/B/C (an all-poison
context is `D` regardless of how `M` compares to the ceiling; C and D are
never mixed).

### 9.4 Prospective population (STEP 3, executed)

**Eligible pool:** the 50 `target_query_ids` in
`results/diagnostics/ml_filterrag_dataset_hotpotqa_50q/dataset_config.json`
-- a pre-existing HotpotQA k=10/N=5 split frozen for an unrelated purpose
(ML-FilterRAG train/test partitioning) long before this task, never touched
by any `ragdefender_legacy`/`ragdefender_paper` outcome.

**Exclusions:** the 8 Gate-A/B/C diagnostic-development queries (listed in
`scripts/ragdefender_expanded_population_lib.py::GATE_BC_EXCLUDED_QUERY_IDS`),
leaving **42** eligible queries.

**Sampling rule:** Option A -- all 42 eligible queries included, no
sub-sampling, no random seed (none needed). The sample was **not** selected
by legacy/paper RAGDefender success or failure, `N_adv` outcome, Stage-2
outcome, similarity geometry, `top_pair_pp`, or existing oracle behavior --
none of these quantities were computed for any of these 42 queries before
the freeze.

**Text recovery (no new retrieval):** clean-passage text is looked up (not
re-retrieved) from `datasets/hotpotqa/corpus.jsonl` by the already-retrieved
`doc_id`; poisoned-passage text is reconstructed deterministically as
`question + "." + adv_texts[pool_index % 5]` (matches `src/attack.py`'s
`LM_targeted` construction), with the source query resolved from
`pool_index // 5` into the already-computed 100-query adversarial-text pool.
Full mechanism: `scripts/ragdefender_expanded_population_lib.py` module
docstring.

**Observed regime distribution (composition property of the already-fixed
retrieval, not a defense outcome -- see
`results/diagnostics/ragdefender_expanded_baseline/PROSPECTIVE_POPULATION_FREEZE.md`):**

| Regime | n (of 42) |
|---|---|
| A_BELOW_CEILING | 0 |
| B_AT_CEILING | 19 |
| C_ABOVE_CEILING | 20 |
| D_ALL_POISON | 3 |

Regime A has zero representation: at `k=10`, every retrieved context in this
attack configuration has an observed poison count `>= 5 = floor(10/2)`. This
is reported as an honest limitation of the existing N=5-candidate attack
configuration, not engineered around.

**Artifacts:** `results/diagnostics/ragdefender_expanded_baseline/{PROSPECTIVE_POPULATION_FREEZE.md,prospective_population.csv,recovered_contexts.json}`,
frozen strictly before any Stella/RAGDefender computation on these queries.
**Status: COMPLETE.**

### 9.5 Expanded baseline / expanded Gate-C / mechanism decision (STEPs 4-6, executed)

**Expanded baseline (n=42, `ragdefender_paper` + Stella, k=10):** zero-
residual-poison success rate **5/42 (11.9%)**; exact-count rate **5/42
(11.9%)**; undercount rate **37/42 (88.1%)**; overcount rate **0/42**.
Mean Stage-2 removal precision is **1.00 in every regime** -- Stage 2 never
removes a clean passage it wasn't asked to remove; all failure is
attributable to *how many* passages Stage 2 is asked to remove, and (in
Regime C only) *which* ones. Full detail:
`results/diagnostics/ragdefender_expanded_baseline/EXPANDED_BASELINE_REPORT.md`.

**Expanded Gate-C oracle-count decomposition (n=42, same matrices):** of
the 37 baseline failures, **21 (56.8%) are COUNT-LIMITED** (A),
**16 (43.2%) are COUNT + IDENTIFICATION LIMITED** (B), and **0 are pure
IDENTIFICATION-LIMITED** (C) -- i.e. a correct count with the unchanged
Stage-2 procedure is NEVER, by itself, insufficient in this sample UNLESS
the count itself is above the structural ceiling. Regime breakdown is the
decisive cut: **B_AT_CEILING: 14/14 failures (100%) fully fixed by the
oracle count alone, with zero additional clean removals** -- this is the
strongest available clean Stage-2-conditional-on-correct-count evidence
in this population; **C_ABOVE_CEILING: only 4/20 (20%)** fixed -- the
remaining 16/20 still show residual poison and/or extra clean removal
even when Stage 2 is handed the true (necessarily above-ceiling) count;
**D_ALL_POISON: 3/3 (100%)** fixed, but this result is **evidentially
degenerate for Stage-2 identification quality**: when `M=k`, supplying
the true count asks Stage 2 to remove every retrieved passage, so a
"success" here provides no information about poison-vs-clean ranking
ability and must not be read as corroborating Regime B's clean-Stage-2
evidence. Full detail:
`results/diagnostics/ragdefender_expanded_gate_c/EXPANDED_GATE_C_REPORT.md`.

**Mechanism decision (STEP 6):** **DECISION C -- two distinct failure
modes, separated by poison-count regime** (not the same "Decision A" story
uniformly). Precisely: **at the ceiling (Regime B), Stage-1 count
underestimation completely explains all observed baseline failures in
this prospective sample** (Stage 2 is accurate whenever it is given the
right count, evidenced by 14/14 zero-cost fixes); **below-ceiling behavior
(Regime A) remains untested, because the current population contains no
Regime-A queries**; **above the ceiling (Regime C)**, count correction is
necessary but insufficient for 80% of failures, and a genuine,
previously-unobserved Stage-2 identification-capacity cost co-occurs; and
**the all-poison Regime-D result is preserved only as a separate
threat-model-violation stress**, not as evidence of Stage-2 accuracy
(see above). Recommends TWO follow-up oracles (Stage-1 count-sensitivity
near the ceiling boundary, and a separate Stage-2 above-ceiling
identification-capacity oracle), neither implemented here. Full detail,
caveats, and exact manuscript-claim boundary:
`results/diagnostics/ragdefender_mechanism_decision/RAGDEFENDER_MECHANISM_DECISION_REPORT.md`.

### 9.6 Nominal HotpotQA k=2 mathematical/code/consistency audit (executed)

**Real k=2 retrieval was NOT run** -- no existing saved `k=2` HotpotQA
retrieval/poisoning artifact was found in this repository (§7 of the k=2
report), and new retrieval was out of scope for this task. What WAS
executed is a full mathematical, authors'-code, and specification-
consistency audit, since the entire question ("does Eq. (3) do anything at
`k=2`?") is answerable without any new retrieval or Stella call. Full
detail: `results/diagnostics/ragdefender_k2_consistency/RAGDEFENDER_K2_CONSISTENCY_REPORT.md`,
`results/diagnostics/ragdefender_k2_consistency/k2_synthetic_results.csv`.

**Structural result (corollary of Eq. (3), independent of similarity
geometry):** for a literal 2-element retrieved set, `s_mean_1 = s_mean_2 =
s = sim(r1,r2)` (self-excluded mean over the single other passage) and
`s_bar = s`, so `s_mean_i > s_bar` is false for both `i` (strict `>`
against an identical value); hence `above_mean = [False, False]`,
`adv_flag = [False, False]`, and **`N_adv = 0` always**, regardless of
Stella embeddings, poison wording, median convention (lower-of-two-middle
vs. average-of-two-middle — irrelevant here since the mean condition alone
already forces `N_adv=0`), or the actual value of `s`. Verified both
analytically and by direct call to `ragdefender_internals.concentration_stage1_paper`
on 2x2 matrices for `s in {-0.5, 0.0, 0.3, 0.8, 0.99}` (`n_adv_estimated=0`
in all 5 cases) plus a randomized synthetic sweep (see the CSV).

**Authors' released code at literal k=2 (`RAGDefender/artifacts/main.py::find_num_adv`,
observationally reproduced via the local `ragdefender_legacy` port,
UNMODIFIED):** the opposite structural degeneracy. `avg[i] == avg_avg`
and the median-threshold condition never fires either (for the same
self-referential reason, but on the *diagonal-inclusive* legacy
statistics), so `above_avg = above_median = [False, False]` and
`sum(final) = 0` -- which triggers the legacy flip branch
(`sum(final) > 0` is false), so **`n_adv = len(text_list) - 0 = k = 2`**:
the legacy estimator flags **both** passages as adversarial, for every
tested `s in {-0.5, 0.0, 0.3, 0.8, 0.99}`, independent of `s`. Stage 2
then selects both indices, so the pre-fallback "safe" set is empty --
however, `ragdefender_legacy`'s historical restore-all fallback
(`if not clean_docs: clean_docs = doc_list`, `defense/defense_runner.py`)
fires and the **final returned context is the full, unfiltered 2-passage
input** -- i.e. legacy is *also* a no-op at literal `k=2` in terms of
final output, but via a completely different internal path (Stage 1
flags everything, Stage 2 tries to remove everything, the restore-all
fallback undoes it) than `ragdefender_paper` (Stage 1 flags nothing,
Stage 2 never runs). Both variants are a no-op on the FINAL returned
context at literal `k=2`; their INTERNAL computations disagree completely
(`N_adv=0` vs. `N_adv=2`).

**Apparent specification/evaluation inconsistency (paper vs. code, cautious
wording; see the k2 report for the full three-column analysis):** the
final paper's own Hyperparameter Selection paragraph (§5) states, for
HotpotQA, "We set the top-k retrieval parameter to be `k = |R̃| = 2`" --
implying Eq. (3)'s sum (Stage 1) and Stage 2 both operate on a literal
2-element set. However, the authors' released `RAGDefender/artifacts/main.py`
constructs its actual Stage-1/2 input (`adv_text_now`) as up to
`A_N` (default 5) adversarial candidate texts **plus all ground-truth
gold passages** (`adv_text_now = adv_text_groups[...][:A_N] + ground_truth`),
and only AFTER Stage 1/2 filtering does it apply `args.top_k` (=2) as a
**separate, final truncation** of the *already-filtered* candidates,
ranked by direct query-passage similarity -- a step that has nothing to
do with RAGDefender's own Eq. (3)/Eq. (4-7) concentration-and-pair-
frequency logic. Under the released code's default hyperparameters, the
actual Stage-1/2 input size is therefore `5 + len(ground_truth)` (e.g. 7
for HotpotQA's stated 2-gold-passage baseline), not 2. **The published
top-k retrieval parameter and the authors' own released implementation's
actual Stage-1/2 input size require reconciliation**; this report does
not and cannot determine which one (if either) matches the exact
configuration used to produce the paper's Table 4/5/9 HotpotQA numbers,
and does not allege that those numbers are invalid or fabricated.

**Operation-order observation:** the released artifact differs from the
published pipeline not only in Stage-1/2 input cardinality, but also in
*operation order*: RAGDefender is applied to a constructed candidate pool
before the artifact's final top-k query-similarity selection, whereas the
paper describes RAGDefender as operating on the retrieved set `R̃`. That
is, in the released code, Stage 1/2 filtering happens first (on the
larger `adv_text_now` candidate pool), and the `args.top_k=2` truncation
by query-passage similarity happens second, on the already-filtered
output -- the reverse of a literal reading of the paper's "RAGDefender
operates on `R̃`, and `|R̃|=k=2`" framing, which would require the
truncation to `k=2` to happen first (defining `R̃`) and RAGDefender to
operate on that already-size-2 set second.

**Cautious cardinality observation:** a literal reading of the paper's
threat-model constraints implies `|R̃| >= M + 1` whenever all `M`
adversarial passages and at least one benign passage are retrieved (the
threat model requires at least one benign passage to survive retrieval).
This is incompatible with `|R̃|=2` whenever `M>=2` -- i.e. whenever two or
more adversarial passages are retrieved, a literal `|R̃|=2` would leave no
room for the required benign passage. This suggests that the published
notation or evaluation pipeline is underspecified with respect to how
`R̃`'s size interacts with the number of adversarial passages actually
retrieved, rather than that any reported result is fabricated or
invalid -- this report does not allege misconduct and does not have
evidence of any.

**Tests:** `tests/test_ragdefender_hotpotqa_k2_consistency.py` (see the k2
report for the full list). **Status: COMPLETE** (mathematical/code/test
audit); **real k=2 retrieval reproduction: NOT RUN, would require new
retrieval**.

### 9.7 Environment reproducibility bridge (executed)

Compared the exact Python/`transformers`/`sentence-transformers`/`torch`
versions and the cached Stella snapshot revision used for THIS session
against Gate B's recorded environment, and separately against the
n=42 expanded-baseline (STEP 4) run's own machine-written environment
record, then re-encoded frozen fixture/frozen-population queries with
Stella in the current environment to test reproducibility directly.

**CORRECTION (post-review):** an earlier version of this section stated
that Gate B and the expanded-baseline/current-session environments used
"identical package versions." That is only true for the Gate-B
comparison. It is **not** true for the expanded-baseline (STEP 4)
comparison: `EXPANDED_BASELINE_REPORT.md`'s own "Dependency / environment
record" (written dynamically at STEP-4 runtime via
`transformers.__version__`/`sentence_transformers.__version__`/
`torch.__version__`) records `transformers==4.57.0`,
`sentence-transformers==5.1.1`, `torch==2.13.0` -- which differ from this
session's currently-measured `4.57.6`/`5.1.2`/`2.8.0`. This is a
contemporaneous, machine-written artifact and must not be silently
overwritten or explained away. Two claims are kept explicitly separate:

- **(A) Historical package-version provenance for the n=42
  expanded-baseline run is UNRESOLVED.** We do not have a contemporaneous
  artifact establishing which package versions were *actually* active
  when STEP 4 ran beyond `EXPANDED_BASELINE_REPORT.md`'s own record,
  which differs from what this session measures now. This is not
  resolved by anything in this report.
- **(B) Computational reproducibility is directly testable from frozen
  texts/matrices, independent of (A), and IS tested here:**
  - **Gate-B fixture (1 query, `5a722b8655429971e9dc9329`):** re-encoding
    with Stella in the current environment reproduces the frozen cosine
    similarity matrix with max absolute difference **`0.0`** (byte-level),
    and Stage-1 `N_adv` / Stage-2 removed indices are also
    byte-identical. Gate B's OWN recorded package versions (`4.57.6`/
    `5.1.2`/`2.8.0`) do match this session's measured versions exactly --
    this one-query result is not in tension with (A) because Gate B's own
    record was never in question.
  - **Expanded-baseline 5-query bridge (STEP 8 of this task, NEW):** five
    queries selected from the n=42 population by a pre-declared
    deterministic rule (first Regime-B success, first Regime-B failure,
    first Regime-C M=6, first Regime-C M>=8, first Regime-D) were
    re-encoded with Stella in the current environment and compared
    against their historical saved matrices. Result: **0/5 byte-identical
    at strict tolerance** (max abs diff `~5e-7`-`~8e-7`, small but
    nonzero), **5/5 identical at loose tolerance**
    (`atol=1e-6, rtol=1e-5`), and **5/5 decision-stable**: `N_adv`,
    Stage-2 removed-index sets, zero-residual-poison outcomes, AND the
    Gate-C oracle-count decomposition labels (A/B/C/D) are all unchanged
    between the historical and re-encoded matrix for every one of the 5
    queries. Full detail:
    `results/diagnostics/ragdefender_environment_bridge/ENVIRONMENT_BRIDGE_REPORT.md`
    §4, `expanded_baseline_bridge_5q.csv`.

**Conclusion, stated at the correct scope:** "The sampled historical n=42
outputs are reproducible in the current environment at the decision
level" (5/5 sampled queries). This does **not** license "the historical
package versions were definitely X" -- that remains an open,
unresolved provenance question (A) -- and it does not certify
byte-identical geometry reproduction for the n=42 population (only the
single Gate-B fixture achieved that; the 5 expanded-baseline queries show
small numeric drift that does not change any decision). Scripts:
`scripts/run_ragdefender_expanded_baseline_bridge_5q.py`. Tests:
`tests/test_ragdefender_expanded_baseline_bridge_5q.py` (26 tests, all
pass; 1 gated live-Stella smoke test also passes with
`RAGDEFENDER_LOAD_STELLA=1`). **Status: COMPLETE** (as a reproducibility
bridge; historical package-version provenance for STEP 4 remains,
and is expected to remain, unresolved absent a new contemporaneous
artifact).

---

## 10. Regime-B Stage-1 boundary-sensitivity oracle (mechanism study)

**Scope note:** purely an OFFLINE matrix/statistic analysis over the 19
already-saved Regime-B (`k=10`, `M=5`) Stella similarity matrices from
`results/diagnostics/ragdefender_expanded_baseline/`. No retrieval, no
Stella re-encoding, no text mutation, no generation, no E1/CORAL/MMD, no
LLM/API call was run for either pass below. Scripts:
`scripts/ragdefender_regime_b_stage1_oracle_lib.py`,
`scripts/run_ragdefender_regime_b_stage1_oracle.py` (V1),
`scripts/run_ragdefender_regime_b_stage1_oracle_v2.py` (V2 correction).
Full reports:
`results/diagnostics/ragdefender_regime_b_stage1_oracle/REGIME_B_STAGE1_ORACLE_REPORT.md`
(V1, superseded for Phase 4/5) and
`REGIME_B_STAGE1_ORACLE_V2_REPORT.md` (V2, current).

### SUPPORTED — Phase 1–3 (unchanged by the V2 correction pass)

- Population: **n=19** Regime-B queries (5 successes, 14 failures);
  every failure has `N_adv=4`, every success `N_adv=5`.
- Boundary decomposition of the 14 failures: **11/14 MEDIAN-LIMITED**
  (`n_above_median<5`), **3/14 MEAN-GATED** (`n_above_median=5` but the
  mean test excludes one passage), **0/14 BOTH-LIMITED**.
- Exact `s_median` rank-5/rank-6 boundary tie (`median_gap==0.0`) in all
  11 MEDIAN-LIMITED failures; all 5 successes have `median_gap>0`
  (min observed positive success gap ≈0.0044).
- Statistic-space (idealized, not matrix-constrained) oracle: 14/14
  failures have a route to `N_adv=5` via a single-passage statistic
  perturbation (12 median-sensitive, 2 mean-sensitive).
- **V2 addition:** the mutual-median-match explanation is now explicitly
  verified, not just illustrated — **11/11** MEDIAN-LIMITED failures have
  their exact `s_median` tie caused by a provable mutual-median-match pair
  (`j in median_provider_set(i)` AND `i in median_provider_set(j)`, using
  full provider SETS to correctly handle within-row value ties). Claim:
  "In all 11 median-limited failures, the boundary tie is explained by at
  least one mutual-median provider relationship in the symmetric
  similarity matrix."

### V2 PHASE 4–5 — corrected matrix-space reachability (supersedes V1)

V1's shared alpha-search helper (`_monotonic_or_grid_search`) had a
false-negative bug: it decided reachability from the grid **endpoint**
only, silently discarding any transient success (predicate `True`
somewhere in the interior, `False` again at the endpoint). V2 rewrites
this to scan the full path, detect every contiguous success window, and
target the earliest one with deterministic local refinement
(`ALPHA_TOL=1e-6`), never assuming monotonicity.

| | V1 (superseded) | **V2 (corrected)** |
|---|---|---|
| any boost success | 0/14 | **14/14** (all transient) |
| any decrease success | 9/14 | **11/14** |
| reachable either mode | 9/14 | **14/14** |
| unreachable | 5/14 | **0/14** |
| winning-path mode split | 0 boost / 9 decrease | **12 boost / 2 decrease** |
| winning paths non-monotonic | "3/9" (provisional) | **12/14** |
| PSD-valid @1e-8 among reachable candidates | not measured | **72/80**; **14/14 winners** |
| Phase-5: count-fix+successful / degraded | 7/9 / 2/9 | **14/14 / 0/14** |

Terminology correction: V1's "realizable matrix oracle" wording is
withdrawn. Perturbations are symmetric, diagonal-preserving, and
`[-1,1]`-clipped by construction (**LEVEL 1: symmetric bounded
matrix-space oracle**) but this does not imply a valid cosine Gram matrix
(needs PSD, checked via `eigvalsh`; **LEVEL 2: abstract unit-vector-
compatible**) — and PSD validity does **not** imply Stella-embedding- or
text-realizability (LEVELS 3/4, not established by this or any prior
pass). In this population, the earliest-detected successful alpha (under
the specified grid-and-refinement search) winner happens to already be
PSD-valid in all 14 queries, so Levels 1 and 2 coincide for the winning
candidates specifically (not for the full achieving set: 8/80 non-winning
achieving candidates are not PSD-valid).

**Regime-B decision: A — sufficiently characterized.** The exact-tie
mechanism (mutual-median-match, 11/11) and mean-gate mechanism (3/3) are
both mechanistically precise; the corrected matrix-space analysis gives a
stable, exhaustive picture. Recommend proceeding to a Regime-C Stage-2
identification-capacity study (not run in this task).

## 11. Regime-C Stage-2 identification-capacity study (mechanism study)

Second mechanism study, run after Regime B closed. Pure OFFLINE Stage-2
diagnostic: the true poison count `M` is supplied to
`stage2_pair_frequency` and held FIXED throughout — Stage 1 is not
consulted. Full reports:
`results/diagnostics/ragdefender_regime_c_stage2/REGIME_C_STAGE2_REPORT.md`
(V1) and `REGIME_C_STAGE2_V2_REPORT.md` (V2 correction pass, this
section).

**POPULATION** (re-verified unchanged in V2): 20 queries, k=10, M>5, C>=1;
M distribution {6:8, 7:2, 8:6, 9:4}; true-count Stage 2 = **4/20 success,
16/20 failure**, reproducing expanded Gate C exactly.

**PAIR SELECTION** (unchanged in V2): PC pairs present in `P_top` **20/20**;
CC pairs present **8/20**; naturally pure-PP `P_top` **0/20**.

**MECHANISM** (V2 — replaces V1's vague "PP-weighting/other" with a
proved, computationally-verified graph-theoretic property):

| mechanism | n/16 |
|---|---|
| A. PC-contribution-driven (removing PC pairs alone repairs) | 7 |
| B. PP-coverage-limited (PP-only subgraph of original P_top leaves >=1 poison vertex with degree 0 — theorem, verified 20/20 exact) | 9 |
| C. other/unexplained | 0 |

**COMPLETE-PP ORACLE**: repairs 16/16, preserves 4/4 successes. Structural
proof retained (complete true-PP set = `K_M` on the M poison vertices;
for M>=2 every poison vertex has degree M-1>=1, no clean vertex has any
degree, so the top-M selection is necessarily exactly the M poison
passages). Corrected interpretation (V2): this proves complete true-PP
coverage is *sufficient* for exact identification — it does **not** prove
non-PP intrusion *alone* is necessary/sufficient for every failure (false
for the 9 PP-coverage-limited cases, which require restoring missing PP
edges, not merely removing non-PP ones).

**PAIR-SWAP CERTIFICATION** (V2 — Issue 1 fix): V1's search could silently
skip exhaustive search of a smaller swap count while mislabeling a larger,
exhaustively-found count as "exact minimum." Confirmed real impact: one
query's V1-reported minimum of **10** was actually **8**. V2's corrected,
vectorized, properly-certified search (cross-checked against an unbounded
brute-force reference, 16/16 exact agreement, 91s → 0.9s) certifies an
**exact minimum for all 16/16 failures**: median 4 swaps, range 1–9
(was 1–10), median `pair_swap_fraction` 0.258 (unchanged), range
0.048–0.333 (was 0.048–0.357).

**SCORE OVERLAP** (V2 — Issue 4 reframing): `score_overlap>=0` remains an
exceptionless outcome signature in this population (16/16 failures, 0/4
successes) but is now explicitly framed as a diagnostic ranking signature,
not a causal mechanism — the underlying causes are PC contribution (7/16)
and incomplete PP vertex coverage (9/16).

**Regime-C decision: A — sufficiently characterized.** PC-contribution
and PP-coverage-limited mechanisms jointly and exclusively explain all 16
failures (0 unexplained); the complete-PP oracle and certified pair-swap
oracle both give a full, non-ambiguous pair-set-level account. No
realizability claim is made — no text/embedding/retrieval experiment was
run in either the V1 or V2 pass. Recommended next step: cross-defense
paper synthesis, not another RAGDefender Stage-1/Stage-2 oracle.
