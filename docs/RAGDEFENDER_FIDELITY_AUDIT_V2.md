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
| Mechanism decision report (STEP 6) | **COMPLETE** -- Decision C (two regime-conditioned failure modes) | `results/diagnostics/ragdefender_mechanism_decision/` |
| E1 clean-anchor oracle / CORAL / MMD rerun on paper-faithful population | **STILL PAUSED** -- the mechanism decision (§9.5) recommends two NEW Stage-1/Stage-2 oracles, not a resumption of E1/CORAL/MMD; do not resume merely because it was the old plan | -- |
| Population expansion (prospective, unbiased HotpotQA k=10 sample) | **COMPLETE** (n=42, superseded §8's design-only status; see §9) | §9 |
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
   (§9.4-9.5) confirms count underestimation is the COMPLETE explanation
   at/below the structural ceiling (Regime B: 14/14 failures fully
   resolved by count correction alone) but reveals it is only a PARTIAL
   explanation above the ceiling (Regime C: only 4/20 failures fully
   resolved by count correction alone; the remaining 16/20 require
   additional Stage-2 identification correction). See the mechanism
   decision report for the full, regime-conditioned claim.
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
oracle count alone**; **D_ALL_POISON: 3/3 (100%)** fixed (threat-model
violation stress, interpret separately); **C_ABOVE_CEILING: only 4/20
(20%)** fixed -- the remaining 16/20 still show residual poison and/or
extra clean removal even when Stage 2 is handed the true (necessarily
above-ceiling) count. Full detail:
`results/diagnostics/ragdefender_expanded_gate_c/EXPANDED_GATE_C_REPORT.md`.

**Mechanism decision (STEP 6):** **DECISION C -- two distinct failure
modes, separated by poison-count regime** (not the same "Decision A" story
uniformly): at/below the ceiling, Stage-1 count estimation is the complete
explanation of failure (Stage 2 is always accurate given the right count);
above the ceiling, count correction is necessary but insufficient for 80%
of failures, and a genuine, previously-unobserved Stage-2
identification-capacity cost co-occurs. Recommends TWO follow-up oracles
(Stage-1 count-sensitivity near the ceiling boundary, and a separate
Stage-2 above-ceiling identification-capacity oracle), neither implemented
here. Full detail, caveats, and exact manuscript-claim boundary:
`results/diagnostics/ragdefender_mechanism_decision/RAGDEFENDER_MECHANISM_DECISION_REPORT.md`.

### 9.6 Nominal HotpotQA k=2 sanity check

**Not executed in this session** -- would require a fresh retrieval pass at
`k=2` (no existing saved `k=2` HotpotQA retrieval/poisoning artifacts were
found for this population), and new retrieval was not authorized for this
task. Recorded as a **planned follow-up**. Structural implication to keep in
mind when it is eventually run: `floor(2/2)=1`, and per §9.1's special case,
`N_adv` is provably always exactly `0` at `k=2` regardless of similarity
geometry -- so Eq. (3) can never flag a passage as adversarial under the
nominal paper `k=2` setting. Its results, if/when produced, must be kept in
a separate output directory and never aggregated with the k=10 stress-regime
results (different retrieval regime, different `N_pairs` scale).
