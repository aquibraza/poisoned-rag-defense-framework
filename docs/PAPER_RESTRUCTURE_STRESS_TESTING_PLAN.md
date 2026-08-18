# Paper Restructuring Plan: "The Anatomy of a RAG Defense"

**Target venue.** IEEE Big Data 2026, topic *Data Ecosystem: Trust, resilience, privacy and security issues*.
**Target title.** "The Anatomy of a RAG Defense: Stress Tests, Failure Modes, and Design Lessons for Knowledge Poisoning"
**Manuscript file.** `paper_latex/bigdata26_paper.tex` (single-file layout retained; no `sections/` split).
**Page limit.** 10 pages, references and any appendix inclusive.

**Status of this document.** Planning only. No LaTeX has been modified. No experiment has been run. Every number cited below is traced to an on-disk artifact; every number that is *not* yet supported is marked `PLANNED` or `VERIFY`.

> **Fidelity-correction update (post-Gate-B; see `docs/RAGDEFENDER_FIDELITY_AUDIT_V2.md` for full detail).** Everything below this line was written against `ragdefender_legacy` (MiniLM, OR logic) evaluation results. A FINAL-PAPER-faithful implementation, `ragdefender_paper` (Stella embedder, AND logic, self-excluded mean/median, no flip branch), now also exists and has been evaluated on the same 8 instrumented HotpotQA k=10/N=5 queries via Gates A (logic-isolation on MiniLM geometry), B (Stella re-encoding), and C (oracle-count decomposition). Headline result: on this n=8 sample, 7/8 residual-poison failures are fully explained by Stage-1 count underestimation alone (supplying the true poison count to the unchanged Stage-2 procedure eliminates 7/7 of them with zero added clean-passage removal); the legacy "clean-density" mechanism explanation (E12-E14 below) does **not** reproduce under Stella, and `top_pair_pp` (the E16/C6 "operative failure condition" below) is **not** discriminative between success and failure on the paper-faithful population. **The E1/CORAL/MMD sequence below is PAUSED, not superseded** -- it remains historically valid for `ragdefender_legacy`, but must not be presented as characterizing `ragdefender_paper` until it is rerun on a prospectively-expanded, paper-faithful population (design only, not yet executed; see the audit doc §8). This document's E1-E30 item list and manuscript-integration table have **not yet** been updated to reflect `ragdefender_paper`/Gates A-C; treat every claim below that depends on RAGDefender mechanism attribution as `ragdefender_legacy`-specific until this document is revised.

---

## 0. Core thesis and what changes

### 0.1 Thesis

The paper's contribution is an *evaluation methodology*, not a new attack and not a new defense. It states the assumption each published RAG poisoning defense actually relies on, builds a stress operator that targets that specific assumption, and then escalates the operator through six levels of realism — from an omniscient embedding-space oracle up to a full retrieval-plus-generation pipeline — reporting at each level exactly what the evidence does and does not license.

The single most defensible finding the current data supports is a *negative* one, and it should be positioned as the paper's intellectual anchor:

> Global distribution alignment of poisoned embeddings toward clean embeddings (CORAL, and MMD under strong preservation) reduces measured distribution distance substantially and displaces embeddings *further* than interventions that succeed, yet causes **zero** defense failures. Failure occurs only when the poison-poison top-pair dominance specifically collapses. A defense's stated assumption and its operative assumption are not the same thing, and only a mechanism-targeted operator finds the operative one.

That result (`results/diagnostics/cluster_normalized_poisoning_formal/FORMAL_ORACLE_COMPARISON.md`) is complete, is counter-intuitive, and reframes the whole paper from "we broke some defenses" to "here is how to test a defense properly." It is also the finding the current manuscript actively mis-states (see §2, item E16).

### 0.2 The framing shift, in one line per section

- Sections currently organized *per defense* (`\section{Baselines and Comparator Positioning}` with a `\subsection` per defense) become organized *per evidence level*. Defense mechanisms move into a mechanism/assumption section (IV); their results are redistributed across a progression (VI).
- The RAGDefender appendix derivation (Eq. 1-15, ~2 pages) compresses to ~6 equations inside IV.A; `\begin{appendices}` is deleted.
- Every result number acquires an explicit evidence tier, so the hierarchy in §7 is visible in the tables rather than only asserted in prose.

---

## 1. Target section structure with page budget

Budget assumes IEEE two-column `conference` `IEEEtran`. Float area is counted inside the owning section's allocation. Total sums to 10.00.

- **I. Introduction** — 0.90 pg
- **II. Background and Related Work** — 0.75 pg
- **III. Threat Model and Stress-Testing Framework** — 1.25 pg (contains Fig. 1, Table I)
  - A. PoisonedRAG attack setting
  - B. Retrieval regimes and hyperparameter rationale (N=5, k=5, k=10)
  - C. Defense stress-testing hierarchy (Levels 0-5)
  - D. Interpretation rules
- **IV. Defense Mechanisms and Stress Operators** — 1.50 pg (contains Fig. 2, Table II)
  - A. RAGDefender
  - B. FilterRAG
  - C. ML-FilterRAG
  - D. Additional defense family (placeholder)
- **V. Experimental Methodology** — 0.85 pg
  - A. Datasets · B. Poison generation · C. Retriever and retrieval settings · D. Baseline defense evaluation · E. Oracle stress-test protocols · F. Fixed-context mutation protocols · G. Full-retrieval mutation protocol · H. End-to-end generation protocol · I. Metrics
- **VI. Results** — 2.90 pg (contains Tables III-VI, Figs. 3, 6)
  - A. Baseline defense behavior and retrieval composition
  - B. RAGDefender oracle stress testing
  - C. FilterRAG / ML-FilterRAG baseline and oracle stress testing
  - D. Generic fixed-context mutation
  - E. Defense-targeted fixed-context mutation
  - F. Full-retrieval mutation
  - G. End-to-end target-answer preservation
  - H. Additional defense family (placeholder)
- **VII. Design Lessons for RAG Poisoning Defense Evaluation** — 0.50 pg
- **VIII. Limitations** — 0.35 pg
- **IX. Conclusion** — 0.15 pg
- **References** (~30 IEEE entries) — 0.85 pg

**No appendix.** Per the layout decision, the Stage-1/Stage-2 derivation is compressed into IV.A and `\usepackage{appendix}` plus the `\begin{appendices}...\end{appendices}` block are removed.

**Overflow release valves, in the order they should be used:** (1) drop Fig. 5, (2) drop Table VII (fidelity/proxy table, already optional), (3) fold Fig. 4's heatmap into Table VI as columns, (4) compress V.A-V.C into one paragraph, (5) reduce IV.A from 6 equations to 4 (drop the Stage-1 flip branch to prose).

---

## 2. Existing-material mapping

Every paragraph, equation, table, figure, and structural element of the current `paper_latex/bigdata26_paper.tex`, with a disposition. Line numbers refer to the file as it stands today.

### 2.1 Front matter and preamble

- **E1 — Title, L51-52.** "How Sharp Should the Swords Be? Stress-Testing Defenses Against RAG Knowledge Poisoning" → **REWRITE** to the target title. The sword metaphor implies an attack paper; the target title states the evaluation contribution.
- **E2 — Preamble, L1-47.** **RETAIN** with three fixes. (a) `\usepackage{float}` is *missing* but `[H]` is used at L110 and L129 — currently a compile failure or silently-ignored specifier; add the package or change to `[t]`. (b) `\usepackage{appendix}` becomes unused once the appendix is deleted — remove. (c) `\usepackage{tikz}` and `\usetikzlibrary{arrows.meta,positioning}` are currently unused but **will be needed** for Fig. 1 — keep.
- **E3 — Author block, L55-68.** **RETAIN** verbatim.
- **E4 — Abstract, L73-75 (empty).** **WRITE NEW**. Must lead with the methodology contribution and the CORAL negative result, and must state the evidence ceiling (pilot-scale end-to-end) in the abstract itself.
- **E5 — Keywords, L77-79 (empty).** **WRITE NEW**. Proposed: retrieval-augmented generation; knowledge poisoning; defense evaluation; adversarial robustness; data integrity; trustworthy AI.
- **E6 — `\section{Introduction}`, L86-87 (empty).** **WRITE NEW** as Section I.

### 2.2 Threat model section

- **E7 — `\section{Threat Model and Attack Setting}`, L92.** **RENAME** to "Threat Model and Stress-Testing Framework" (Section III). Section II (Background/Related Work) is inserted before it.
- **E8 — `\subsection{Hyoerparameter selection}` + paragraph, L94-95.** **MOVE to III.B, RETAIN near-verbatim.** This is the strongest existing paragraph in the manuscript: it already justifies N=5 as the PoisonedRAG default, k=5 as the poison-saturated regime where k=N, and k=10 as the mixed regime that PoisonedRAG's own ablation identifies, and it already explains why the mixed regime matters specifically for post-retrieval defenses. Edits needed: fix the "Hyoerparameter" typo; replace the Unicode right-quote in "PoisonedRAG's" with `'`; add `\cite{}` anchors for PoisonedRAG and for the ablation; add one sentence naming k=5 as Level 1 in the hierarchy of III.C so the paragraph connects forward.

### 2.3 Defense/baseline section (to be dissolved)

- **E9 — `\section{Baselines and Comparator Positioning}`, L102.** **DELETE the heading.** Its children split between Section IV (mechanism and assumption) and Section VI (results).
- **E10 — RAGDefender ¶1, L104.** "effective under its stated assumptions, but ... when top-k retrieval is fully saturated by poison, clean-anchor-based grouping cannot reliably rescue benign evidence because none is available." → **MOVE to VI.A, COMPRESS to one sentence.** Supported by `results/diagnostics/RAGDEFENDER_DIAGNOSTIC_REPORT.md` (k=5, residual poison fraction 1.000, poison recall 0.560, n=10).
- **E11 — RAGDefender ¶2, L106.** **SPLIT, and one clause must be REWRITTEN.**
  - "RAGDefender explicitly assumes at least one benign passage is retrieved" → **MOVE to IV.A** as the stated-assumption statement.
  - "When k is expanded from 5 to 10, restoring clean context, RAGDefender **substantially reduces ASR**" → **UNSUPPORTED. MUST REWRITE.** The k-sweep was executed in dry-run mode; `RAGDEFENDER_DIAGNOSTIC_REPORT.md` records `ASR_with_defense = n/a (dry-run)` in every row. Replace with the detection quantities that *are* measured: mean residual poison fraction 1.000 → 0.088, poison recall 0.560 → 0.880, clean FPR 0.200 at k=10. This is the single most important correction in the manuscript.
  - "query-level estimator failures remain" → **MOVE to VI.A**, supported by the one query where RAGDefender removed 0 poison and 3 clean (`5a8cb288554299585d9e3726`, k=10).
- **E12 — `\subsubsection{Stage 2 Successful Case}` + Figs. at L110-115 and L117-122.** **MOVE to Fig. 2 (mechanism panel) in IV.A/VI.A, MERGE the two floats into one.** Both floats carry the literal caption `Caption` and the label `\label{fig:placeholder}` — four duplicate labels exist across E12/E13 and will emit LaTeX multiply-defined warnings. Both must be captioned with the query id, the top-pair mix, and the removal counts.
- **E13 — `\subsubsection{Stage 2 Failure Case}` ¶ + Figs. at L129-134 and L136-141.** Same treatment; becomes the second half of Fig. 2. Supporting numbers exist in `docs/RAGDEFENDER_CLUSTER_DIAGNOSTIC_FINDINGS.md`: for `5a8cb288...`, mean clean-clean similarity 0.992 *exceeds* mean poison-poison 0.945, top-pair mix 0/0/3, removed 0 poison / 3 clean, residual poison fraction 0.714.
- **E14 — L127, "Stage 1 overestimates $N_{adv}$, so Stage 2 removes extra clean passages."** **FACTUAL MISMATCH — RECONCILE BEFORE REUSE.** This sentence describes the *over-removal* case `5a722b8655429971e9dc9329` (N_adv estimated 7 against a true 5, removing 5 poison and 2 clean), but it is placed under the figure for `5a8cb288...`, which is a *clean-density* failure with N_adv = 3 that removed 0 poison. These are two distinct failure modes and the paper needs both, separately labelled. `RAGDEFENDER_CLUSTER_DIAGNOSTIC_FINDINGS.md` defines four categories over 8 queries (2 poison-clique success, 4 success-with-clean-false-positives, 1 residual-poison failure, 1 clean-density failure) — use that taxonomy in VI.A rather than two ad-hoc cases.
- **E15 — L143, "The above case is a sample run where this defense technique lapsedl. but the following is where we carried out 'normalization'..."** **DELETE and REWRITE** as the Level-1 → Level-2 transition paragraph in VI.B. Contains a typo and first-person lab-notebook voice.
- **E16 — `\subsubsection{Fragility}`, L145-148.** **SPLIT, and the central claim must be RE-ATTRIBUTED.**
  - The conditional operator definition ("If a defense relies on poison-poison cosine-density being higher than clean-clean or poison-clean density, then we can test how fragile that assumption is by applying representation-level normalization...") → **MOVE to IV.A** as the representation-space oracle definition, and to III.C as the Level-2 definition. Excellent text; keep the logic.
  - "We use covariance alignment as an oracle diagnostic to test whether RAGDefender's reliance on concentrated adversarial passage geometry is robust" → **RETAIN as method** in IV.A.
  - "**By matching the pairwise similarity distribution of adversarial passages to benign retrieved passages, we show that RAGDefender's top-pair frequency heuristic is brittle to representation-level dispersion**" → **THIS ATTRIBUTION IS CONTRADICTED BY OUR OWN DATA. MUST REWRITE.** `FORMAL_ORACLE_COMPARISON.md` reports CORAL-PCA and all three CORAL-ridge settings at **0/24 query-level failures** at maximum perturbation, despite reducing MMD distance by 0.42-0.42 and displacing poison embeddings *more* (mean L2 0.947-0.979) than any failing configuration. Brittleness is caused by **E1 clean-anchor interpolation** (12/12 query-strategy units) and by **MMD at weak/mid preservation** (10/12), i.e. by interventions that collapse `top_pair_pp`, not by distribution matching per se. Rewrite as: covariance alignment is a *negative* result, and `top_pair_pp` collapse is the operative failure condition. Then promote this to the paper's headline methodological finding (§0.1).
- **E17 — L151, `****success figure*****`.** Placeholder marker. **REPLACE** with Fig. 3 (oracle dose-response).
- **E18 — L153, "We further approximate this oracle intervention using semantics-preserving passage rewrites that diversify adversarial evidence while preserving target-answer influence."** **MOVE to IV.A** as the sentence/text-space mutation objective, and forward-reference VI.E. Note that the *empirical* verdict on this sentence is mixed: the RAGDefender-targeted rewrite family produced a mean delta of **0.00** on RAGDefender, so "approximate" must be stated as an attempted approximation whose success was case-dependent (Schmeichel: PP pairs 10 → 3, removals 5 → 3; other cases: no effect).
- **E19 — L155, `**multiple success case figures*`.** Placeholder marker. **REPLACE** with Fig. 6 (text-space realization, two-case panel).
- **E20 — L157, "Across six originally successful HotpotQA k=10/N=5 RAGDefender cases, E1 clean-anchor oracle interpolation caused residual poisoned passages to survive in 22/24 query-strategy configurations. PP top-pair weakening preceded or coincided with residual-poison failure in all informative configurations..."** **MOVE to VI.B, RETAIN — but `VERIFY` the 22/24 figure.** `FORMAL_ORACLE_COMPARISON.md` reports 12/12 query-level units over two strategies at max perturbation; `BATCH_COMPARISON_SUCCESS_CASES.md` reports 6/6 queries failing by alpha ≤ 0.5 under at least one strategy and 5/6 under all four. 22/24 is presumably 4 strategies × 6 queries from `BATCH_COMPARISON_SUCCESS_CASES.csv`; recount from that CSV and state the denominator explicitly in the paper.
- **E21 — `\subsection{E1 causes MMD & CORAL variation?}` + ¶, L159-160.** **RETITLE and MOVE to VI.B.** Two issues: the bare `&` in a section title is a LaTeX error (needs `\&`), and the interrogative title is not publication voice. The content is sound and directly supports §0.1: CORAL distance and RBF-MMD both decreased monotonically as alpha decreased across 24 configurations, yet `top_pair_pp` was the specific failure predictor. Supported by `results/diagnostics/cluster_normalized_poisoning/DISTRIBUTION_METRICS_BATCH.md`.
- **E22 — `\subsection{CORAL impact}` ¶, L162-163.** **MOVE to IV.A and III.D, RETAIN near-verbatim.** "These are not text-space attacks or learned domain-adaptation models; they are formal embedding-space stress tests of RAGDefender's similarity-based decision rule." This is precisely the interpretation rule Section III.D needs; reuse the sentence there.
- **E23 — `\subsection{Semantics-preserving textual adversarial attacks}` ¶, L165-166.** **SPLIT and MOVE to Section II, strip the planning voice.** "Will need when I move from embedding-only stress tests to real passage mutation" is a note-to-self and must be deleted. The survey content (TextFooler semantic preservation and grammaticality; BERT-Attack masked-LM fluency; SemAttack's contextualized semantic spaces; TextAttack's SBERT/USE cosine constraints) becomes the Section II paragraph that motivates the Level-3 operator design, and the constraint list becomes the operator specification in IV.A.
- **E24 — `\subsection{Filter-RAG}`, L170 (empty heading).** **WRITE NEW** as IV.B. Source material exists in `docs/FILTERRAG_BASELINE.md` and `docs/FILTERRAG_FIDELITY_AUDIT.md`.
- **E25 — `\subsection{ML-Filter-RAG}` + ¶, L172-174.** Heading → IV.C. The paragraph ("At t=0.40, ML-FilterRAG-top-k reduces clean false positives from 78.1% to 15.6% relative to semantic threshold FilterRAG, while maintaining 94.9% poison recall and a lower residual poison fraction") → **MOVE to VI.C, RETAIN.** Supported by `results/diagnostics/ml_filterrag_eval_hotpotqa_50q_t04/`. This is also the paper's cleanest security-versus-utility datapoint and should be cited again in VII.
- **E26 — `\begin{table*}` `tab:mlfilterrag_threshold_sweep`, L176-200.** **RETAIN but COMPRESS and DEMOTE from `table*` to single-column `table`.** Becomes Table IV in VI.C. Drop the t=0.10, t=0.30, t=0.70 rows; keep the FilterRAG semantic ε=0.2 reference row, t=0.35, t=0.40, t=0.50, and t=0.90 as the utility endpoint where recall collapses to 0.678 at zero clean FPR. The caption already carries the correct denominators (10 held-out query ids, 20 (q,k) records, 118 retrieved poisoned, 32 retrieved clean) — keep them, they are what makes the row defensible at this scale.
- **E27 — `\subsubsection{ML-Filter-RAG Oracle Test}`, L202-207 (three paragraphs).** **SPLIT and COMPRESS from three overlapping paragraphs to two.**
  - The interpolation definition `z' = alpha·z_poison + (1-alpha)·z_target` and the sentence "This oracle does not rewrite passage text and therefore does not constitute a text-realizable attack" → **MOVE to IV.C** (nearest-clean feature-space oracle) and III.D.
  - The results ("reduces poison recall from 94.9% at alpha=1.0 to 67.8% at alpha=0.4, while increasing the residual poison fraction from 18.2% to 58.5%") → **MOVE to VI.C, RETAIN.**
  - "and below 50% by alpha=0.2" → **`VERIFY`** against `results/diagnostics/ml_filterrag_feature_oracle/FEATURE_ORACLE_SWEEP.csv`; recall at alpha=0.4 ranges 0.686-0.856 across strategies, so the alpha=0.2 figure must be stated per-strategy, not globally.
  - Also add from `FEATURE_ORACLE_REPORT.md`: the global clean centroid strategy is *much weaker* than nearest-clean and same-query clean centroid. That contrast is the ML-FilterRAG analogue of the CORAL negative result and strengthens §0.1.

### 2.4 Empty scaffolding

- **E28 — `\section{Experimental Setup}` L212, `\section{Metrics and Evaluation Protocol}` L216, `\section{Results}` L220, `\section{Limitations}` L223, `\section{Conclusion}` L227.** All empty. **Experimental Setup + Metrics MERGE into Section V** (Metrics becomes V.I). Results, Limitations, Conclusion become VI, VIII, IX. Section VII (Design Lessons) is entirely new.

### 2.5 Bibliography and appendix

- **E29 — `\bibliography{references}`, L235-236, and `paper_latex/references.bib`.** The `.bib` file is **0 bytes**, yet `\cite{kim2025ragdefender}` is used at L247. Citations currently cannot resolve. **P0 infrastructure task:** populate `references.bib`. A ready-made source list of ~31 references (attacks, defenses, distribution matching, clustering attacks, retrieval geometry, text-space attacks, similarity tooling) is enumerated across `docs/NORMALIZED_POISONING_LITERATURE_REVIEW.md` and `docs/CLUSTER_NORMALIZED_POISONING_EXECUTION_PLAN.md` — harvest from there rather than re-deriving.
- **E30 — `\clearpage` + `\begin{appendices}` block, L238-460.** **COMPRESS into IV.A and DELETE the appendix**, per the layout decision. Disposition of each equation:
  - **KEEP (6 equations in IV.A):** `eq:ragdefender_cosine` (pairwise cosine S_ij); a single `align` combining `eq:passage_mean_similarity` and `eq:passage_median_similarity` (mu_i, m_i); `eq:or_flag` (f_i = a_i OR b_i); `eq:adv_count_estimate` (the N_adv flip branch — essential, it is a documented deviation from the paper's AND rule and it is the source of one whole failure category); `eq:number_selected_pairs` (L = C(N_adv,2)); `eq:passage_frequency_score` (r_i, the pair-frequency suspicion score — the mathematical statement of the relative-concentration assumption the paper stress-tests).
  - **DEMOTE TO PROSE:** `eq:global_mean_similarity`, `eq:global_median_similarity`, `eq:mean_indicator`, `eq:median_threshold`, `eq:median_indicator`, `eq:raw_flag_count`, `eq:signed_squared_similarity`, `eq:removed_passages`, `eq:filtered_context`.
  - **RETAIN VERBATIM (prose, L444-459):** the paragraph beginning "The procedure relies on the assumption that poisoned passages form the densest high-similarity subgraph" — this *is* the assumption statement IV.A is built around — and the paragraph beginning "Our diagnostic implementation reproduces the baseline calculations without modifying the filtering behavior," which belongs in V.D/V.E as the instrumentation description.
  - Note for IV.A: the manuscript's own appendix already flags "In the evaluated implementation, these statistics include the diagonal self-similarity term S_ii = 1" and the OR-versus-AND deviation. `docs/RAGDEFENDER_VERSION_AUDIT.md` confirms the repo Stage 1 uses OR where the paper specifies AND, and that the embedder is `paraphrase-MiniLM-L6-v2` rather than the package default. These deviations must appear in V.C and VIII, not be quietly dropped with the appendix.

### 2.6 Structural defects to fix while restructuring

Independent of content, these are live compile or correctness problems in the current file:

- `[H]` float specifier used at L110 and L129 without `\usepackage{float}`.
- Four floats all labelled `\label{fig:placeholder}` (L114, L121, L133, L140) — multiply-defined.
- Four floats all captioned literally `Caption` (L113, L120, L132, L139).
- Literal unescaped `&` in the `\subsection` title at L159.
- Unicode right single quotes in body text (L95 "PoisonedRAG's", L106/L125/L146 "RAGDefender's") — render incorrectly under `[T1]{fontenc}` without `textcomp` handling; replace with ASCII `'`.
- `references.bib` empty while `\cite` is used.

---

## 3. Section-by-section implementation plan with content status

Status legend: **COMPLETED** = an on-disk artifact supports the text today. **PRELIMINARY** = an artifact exists but at pilot scale; text must be hedged and counts stated. **PLANNED** = no artifact; placeholder only. **OPTIONAL** = stretch, cut if page- or time-bound.

### Section I — Introduction (0.90 pg)

| Subsection | Status | Source |
|---|---|---|
| Problem framing: RAG poisoning defenses are evaluated only against the attack distribution they were designed for | COMPLETED (argument, not result) | `docs/ANALYSIS.md`, published defense papers |
| Contribution 1: an assumption-to-operator stress-testing hierarchy (Levels 0-5) | COMPLETED (this paper's framework) | §III |
| Contribution 2: the CORAL/MMD negative result and `top_pair_pp` as the operative failure condition | COMPLETED | `FORMAL_ORACLE_COMPARISON.md` |
| Contribution 3: a text-realizability ladder showing where oracle fragility does and does not survive contact with real rewrites, real retrieval, and real generation | PRELIMINARY | mutation pilots |
| Contribution 4: six design lessons for defense evaluation | COMPLETED (synthesis) | §VII |
| Explicit scope statement (HotpotQA, Contriever, N=5, pilot-scale Levels 4-5) | COMPLETED | — |

**Writing constraint.** The introduction must state the pilot scale of Levels 4-5 rather than deferring it to Limitations. The paper's credibility rests on the evidence hierarchy being visibly self-applied.

### Section II — Background and Related Work (0.75 pg)

| Content | Status | Source |
|---|---|---|
| RAG and dense retrieval; corpus poisoning attacks (PoisonedRAG, AGGD, unsupervised continuous-space poisoning) | PLANNED (writing) | refs in `NORMALIZED_POISONING_LITERATURE_REVIEW.md` |
| Post-retrieval filtering defenses: RAGDefender (cross-passage clustering), FilterRAG/ML-FilterRAG (per-passage statistics) | PLANNED (writing) | Kim et al. 2025; Edemacu et al. 2025 |
| Generation-side defenses (RobustRAG family) — positioned as the orthogonal family this paper argues must be included | PLANNED | §IV.D placeholder |
| Adversarial robustness of clustering and nearest-neighbor structures | PLANNED | Chhabra 2020; Cinà 2022; Villani 2026; Wang/Jha/Chaudhuri 2018; Sitawarin 2021; Wu 2022 |
| Distribution matching as a stress tool: CORAL, MMD, embedding post-processing | PLANNED | Sun 2016; Gretton 2012; Li 2020 (BERT-flow); Huang 2021; Mu & Viswanath 2018 |
| Semantics-preserving text attacks as the Level-3 operator family | COMPLETED (text exists) | **reuse E23** |
| Gap statement: no prior work separates oracle fragility from text-realizable evasion from retrieval-preserving evasion from end-to-end success | COMPLETED (argument) | — |

### Section III — Threat Model and Stress-Testing Framework (1.25 pg)

**III.A PoisonedRAG attack setting** — PLANNED (writing). Black-box LM-targeted poison generation, N poisoned passages per target query injected into the corpus, attacker goal is a specific target incorrect answer. Sources: `src/attack.py` (`Attacker`, `LM_targeted`), `results/adv_targeted_results/`. State explicitly that HotFlip exists in the codebase but is out of scope.

**III.B Retrieval regimes and hyperparameter rationale** — COMPLETED. **Reuse E8 near-verbatim** with the fixes noted. Add: k=5 is the saturated regime that violates RAGDefender's stated ≥1-benign-passage assumption; k=10 is the mixed regime where all three defenses are inside their intended operating envelope, which is why it is the regime used for Levels 2-5.

**III.C Defense stress-testing hierarchy** — COMPLETED (framework). This is new prose plus **Table I** and **Fig. 1**.

- **Level 0 — Nominal benchmark.** Defense evaluated on unmodified PoisonedRAG poison in the regime its authors used. Establishes: the defense works as published. Establishes nothing about robustness.
- **Level 1 — Operating-regime stress.** Same poison, hyperparameters moved off the published operating point (k=5 vs k=10; ε and t swept). Establishes: whether the published result is an operating-point artifact.
- **Level 2 — Defense-native oracle stress.** Poison representations or features are modified directly in the space the defense's decision rule reads, with retrieval membership and passage text held fixed. Establishes: which statistical assumption the decision rule *operatively* depends on. Establishes **no** attack.
- **Level 3 — Fixed-context text realization.** Poison passage *text* is rewritten; top-k membership is held fixed by construction. Establishes: the assumption is reachable by a real passage. Establishes **no** retrieval preservation.
- **Level 4 — Full-retrieval / retrieval-realizable stress.** Mutated poison replaces original poison in the corpus and retrieval is rerun. Establishes: the rewrite still retrieves. Establishes **no** generation outcome.
- **Level 5 — End-to-end adaptive poisoning.** Generation is run on the post-defense context and the target answer is scored. Establishes: an end-to-end poisoning claim.

**III.D Interpretation rules** — COMPLETED. Four rules, stated as constraints the paper imposes on itself:
1. *Oracle fragility* is a property of a decision rule, not a capability of an attacker. **Reuse E22's sentence verbatim** here.
2. *Text-realizable evasion* requires the mutated passage to be a real, fluent passage scored by the unmodified defense implementation.
3. *Retrieval preservation* requires rerun retrieval over the corpus with the mutated passage substituted under a preserved poison budget, not fixed-context scoring.
4. *End-to-end ASR* requires the target incorrect answer to be produced from the post-defense context, adjudicated so that a token-boundary match co-occurring with the correct answer is not counted as success.

Rule 4 is not a formality — it is a finding. `EXPANDED_ANSWER_GENERATION_REPORT.md` shows 25 strict-match hits resolving to 20 clear successes and 5 prefix/ambiguous false positives, and the ambiguous ones occur precisely when the defense *worked* and the model fell back to parametric knowledge.

### Section IV — Defense Mechanisms and Stress Operators (1.50 pg)

Each subsection follows a fixed five-part template so **Table II** can summarize it and so IV.D can be added later without disturbing the section: *decision mechanism → stated assumption → operative assumption → oracle operator → text-space operator*.

**IV.A RAGDefender** — COMPLETED.
- Decision mechanism: 6 compressed equations from E30 plus prose.
- Stated assumption: at least one benign passage is retrieved (from E11).
- Relative-concentration assumption: poisoned passages form the densest high-similarity subgraph (**retain E30's L444-451 paragraph verbatim**).
- Stage 1 / Stage 2 detail limited to what the stress argument needs: the flip branch (because it produces N_adv over-estimation failures independently of Stage 2) and the pair-frequency score (because it is what the oracle collapses).
- Representation-space oracle: **reuse E16's conditional formulation and E22's CORAL/MMD definitions**; specify E1 clean-anchor interpolation with the four anchor strategies (`rank_aligned`, `nearest_bijection`, `farthest_bijection`, `random`) and alpha ∈ [0.3, 1.0], plus CORAL-PCA, CORAL-ridge, MMD-minimize with a preservation penalty. Code: `defense/cluster_normalized_poisoning.py`, `defense/coral_mmd_intervention.py`.
- Sentence/text-space mutation objective: **reuse E18**, restated as an *objective* (reduce `top_pair_pp` subject to fluency, semantic preservation, and target-answer preservation) rather than as an achieved result.

**IV.B FilterRAG** — COMPLETED (mechanism); source `defense/filterrag.py`, `docs/FILTERRAG_BASELINE.md`, `docs/FILTERRAG_FIDELITY_AUDIT.md`.
- Decision mechanism: per-passage Freq-Density = (summed frequency of matched keywords from query ⊕ SLM answer) / unique words in passage; drop if score ≥ ε (default 0.2). Semantic matching via MiniLM cosine ≥ 0.6.
- Lexical/statistical concentration assumption: poisoned text is keyword-stuffed with the query and target answer relative to its own length. Crucially, this is a *per-passage* assumption, so unlike RAGDefender it needs no clean cluster — which is why it survives the k=5 saturation case that breaks RAGDefender. State this contrast explicitly; it is one of the paper's cleanest structural observations.
- Mutation objective: preserve the false claim while lowering query/answer keyword overlap and raising unique-word count.
- Must state proxy substitutions here (flan-t5-small for LLaMA-2/3; top-k rather than the paper's top-s → filter → top-k), not defer them.

**IV.C ML-FilterRAG** — COMPLETED (mechanism); source `defense/ml_filterrag.py`, `docs/ML_FILTERRAG_IMPLEMENTATION_PLAN.md`.
- Feature-based classifier: Random Forest over four paper-aligned features (Freq-Density, matched-frequency sum, perplexity, SLM answer joint log-probability); remove if P(poison) ≥ t.
- Feature-separability assumption: poison is linearly-or-tree separable in that 4-dimensional feature space.
- Nearest-clean feature-space oracle: **reuse E27's interpolation equation**; strategies nearest-clean, same-query clean centroid, global clean centroid.
- Text-space approximation: note that ML-FilterRAG inherits Freq-Density as a feature, which is why a FilterRAG-targeted rewrite transfers to it — the transfer is a *predicted consequence of shared features*, and the paper should present it as a prediction that the data confirms.

**IV.D Additional defense family** — **PLANNED / PLACEHOLDER.** No implementation exists in the repository (confirmed: no RobustRAG, TrustRAG, ReliabilityRAG, or InstructRAG code). The subsection exists as a labelled placeholder with selection criteria stated, and Section VI.H mirrors it. See §5 (P1-1) for the recommendation and rationale. **Do not write mechanism prose until the family is selected and implemented.**

### Section V — Experimental Methodology (0.85 pg)

| Subsection | Status | Source of truth |
|---|---|---|
| A. Datasets | COMPLETED for HotpotQA; PLANNED for a second dataset | `prepare_dataset.py`, BEIR HotpotQA; NQ/MS MARCO available but only n=1 defended coverage |
| B. Poison generation | COMPLETED | `src/attack.py` LM_targeted, N=5, `results/adv_targeted_results/hotpotqa.json` |
| C. Retriever and retrieval settings | COMPLETED | Contriever `facebook/contriever`, dot-product, `results/beir_results/hotpotqa-contriever.json`, k ∈ {5,10} |
| D. Baseline defense evaluation | COMPLETED | `defense/dispatch.py`; controls `oracle_remove_all_poison` and `random_remove_same_count` from `defense/controls.py` — state explicitly that these are diagnostic bounds, not deployable defenses |
| E. Oracle stress-test protocols | COMPLETED | `scripts/run_cluster_normalized_poisoning.py`, `run_coral_pca_*`, `run_coral_ridge_*`, `run_mmd_oracle_intervention.py`, `stress_ml_filterrag_feature_oracle.py` |
| F. Fixed-context mutation protocols | PRELIMINARY | `scripts/run_text_mutation_fixed_context_eval.py`, `run_normalized_targeted_mutation_bundle_1_eval.py`; **must disclose that mutations are hand-authored GPT rewrites, n=6 queries, three families** |
| G. Full-retrieval mutation protocol | PRELIMINARY | `scripts/run_full_retrieval_pilot_bundle1.py`; **must disclose the preserved 5-slot poison budget and the one cross-query contamination case** (`full_retrieval_budget_contamination_audit.md`) |
| H. End-to-end generation protocol | PRELIMINARY | `scripts/run_expanded_answer_generation_bundle1.py`, gpt-3.5-turbo, `wrap_prompt(prompt_id=4)`, 25 + 30 generations |
| I. Metrics | COMPLETED | definitions below |

**V.I Metrics — exact definitions to use (do not paraphrase loosely):**
- **ASR** — fraction of queries whose generated answer contains the attacker's target incorrect answer. Report under `strict_match` (token-boundary, `defense/asr_match.py`) *and* report the adjudication categories; state that the legacy substring matcher in `src/utils.py::clean_str` is retained only to expose the `no` ⊂ `not` class of false positive.
- **Poison recall / removal** — removed_poison / N_retrieved_poison.
- **Clean FPR / utility loss** — removed_clean / N_retrieved_clean. Add the FilterRAG k=5 empty-context case as a distinct utility failure mode that clean FPR cannot express.
- **Residual poison fraction** — residual_poison / (residual_poison + residual_clean), i.e. the poison share of the *surviving* context, not of the retrieved set. This definition differs from the intuitive reading and must be stated in the paper.
- **PP / PC / CC pair composition** — counts of poison-poison, poison-clean, clean-clean pairs among RAGDefender Stage-2's top C(N_adv,2) pairs. `top_pair_pp` is elevated to a first-class reported metric.
- **CORAL / MMD** — reported as *diagnostic distribution distances* (`defense/distribution_metrics.py`, RBF gamma=1.0) and separately as *intervention objectives* (`defense/coral_mmd_intervention.py`). Keep those two roles typographically distinct; conflating them is what produced the E16 mis-attribution.
- **Retrieval survival / rank** — count of mutated poison passages present in the rerun top-k, with rank range, under a preserved poison budget.
- **Target-answer preservation** — the mutated passage still asserts the attacker's target answer; verified by adjudicated end-to-end generation on the undefended mutated context (the `defense = none` rows), not assumed.

### Section VI — Results (2.90 pg)

Organized by evidence strength, not by defense. Each subsection opens with one sentence naming its level and what it therefore licenses.

**VI.A Baseline defense behavior and retrieval composition** (Levels 0-1) — **COMPLETED, detection-only, n=10.** Table III.
- RAGDefender k=5: residual poison fraction 1.000, poison recall 0.560 — the saturation failure (E10).
- RAGDefender k=10: residual 0.088, recall 0.880, clean FPR 0.200 (E11, corrected — **no ASR claim**).
- Failure taxonomy over 8 instrumented queries with the two concrete mechanism cases (E12-E14): clean-density failure (CC 0.992 > PP 0.945, 0 poison / 3 clean removed, residual 0.714) and N_adv over-estimation (7 vs 5, 5 poison + 2 clean removed). Fig. 2.
- FilterRAG k=5: context emptied on 100% of queries at every ε — clean FPR undefined, a Level-1 utility failure. k=10, ε=0.4-0.5: recall 1.0, residual 0, clean FPR 0.16-0.24.
- ML-FilterRAG nominal: test precision 0.956, recall 0.915, F1 0.935, AUC 0.962 (40/10 query split).
- **Framing decision:** VI.A reports *detection quality and retrieval composition*, not ASR. No nominal defended ASR exists for FilterRAG or ML-FilterRAG, and the RAGDefender k-sweep was dry-run. Deferring all ASR to VI.G is both honest and consistent with the evidence hierarchy. See P0-4 in §5 for the optional upgrade.

**VI.B RAGDefender oracle stress testing** (Level 2) — **COMPLETED, oracle, 6 queries.** Fig. 3, Table V.
- E1 clean-anchor interpolation induces residual poison on originally-successful cases (E20, denominator to be `VERIFY`ed).
- `top_pair_pp` 10 → 2.5 at max perturbation for both E1 strategies.
- **The negative result** (E16 corrected): CORAL-PCA and CORAL-ridge produce 0/24 failures despite reducing MMD distance by ~0.42 and displacing embeddings further (mean L2 0.947-0.979) than any failing configuration. MMD at weak/mid preservation: 10/12 failures; at `lambda_preserve=1.0`: 0/6.
- Monotone CORAL/MMD decrease with alpha across 24 configurations, with `top_pair_pp` the specific predictor (E21).
- Conclusion sentence: displacement magnitude and global distribution alignment are neither sufficient; direction relative to the poison-poison top-pair subgraph is what matters.

**VI.C FilterRAG / ML-FilterRAG baseline and oracle stress testing** (Levels 1-2) — **COMPLETED, oracle.** Table IV.
- Security/utility comparison at t=0.40 (E25): clean FPR 0.781 → 0.156 at recall 0.983 → 0.949.
- Threshold sweep (E26 compressed) showing the recall/FPR frontier through to t=0.90 (recall 0.678, FPR 0.000).
- Feature-space oracle (E27): recall 0.949 → 0.686 at alpha=0.4, residual poison 0.182 → 0.58; nearest-clean and same-query centroid effective, global centroid much weaker — the ML-FilterRAG analogue of VI.B's locality finding.

**VI.D Generic fixed-context mutation** (Level 3) — **PRELIMINARY, n=4 queries.**
- Generic paraphrase barely evades any defense; Freq-Density mean 0.983 → 0.424 and ML poison probability 0.898 → 0.701 without consistent removal changes. This is a necessary negative control: it shows Level-3 results are not an artifact of rewriting per se.

**VI.E Defense-targeted fixed-context mutation** (Level 3) — **PRELIMINARY, n=6 queries.** Fig. 4, Fig. 5, Fig. 6.
- Cross-defense delta matrix (mean change in poison removed, of 5): RAGDefender-targeted (0.00 / −0.17 / −0.67), FilterRAG-targeted (−0.33 / −1.17 / −2.17), ML-FilterRAG-targeted (+0.17 / 0.00 / −0.17).
- **The asymmetry is the finding:** the family aimed at lexical density is the only one that weakens all three defenses, and the family aimed at RAGDefender's clustering mechanism does not move RAGDefender at all. Represent as: the operative assumption of the density-based defenses is text-reachable; RAGDefender's is not, under hand-authored rewriting.
- Mechanism confirmation: lowest observed Freq-Density 0.184 (below ε=0.2); one query's mean ML poison probability 0.23 (below t=0.4).
- Fig. 6 is the paper's key qualitative panel: Schmeichel (PP top pairs 10 → 3, RAGDefender removals 5 → 3 — a text-space rewrite reproducing the oracle's mechanism signature) beside Gibson (poison clique intact, RAGDefender still removes 5, ML-FilterRAG 5 → 0 — evasion of a *different* mechanism without the oracle signature).

**VI.F Full-retrieval mutation** (Level 4) — **PRELIMINARY, n=3 (+3 in the expanded run).** Table VI (upper block).
- 15/15 mutated poison passages survive into a rerun Contriever top-10 at ranks 1-5, under a verified 5-slot preserved budget.
- Defense weakening largely reproduces the fixed-context result; RAGDefender remains the most robust.
- **Must disclose:** one of the three queries (`5a8e068b...`) retrieved a 6th poison passage authored for a different pool query; the per-origin column of `full_retrieval_poison_origin_breakdown.csv` is the precise figure for that query, and the aggregate `removed_poison` for it conflates two attack instances.

**VI.G End-to-end target-answer preservation** (Level 5) — **PRELIMINARY, n=6 queries, gpt-3.5-turbo.** Table VI (lower block).
- Report **counts, never rates**: 12 mutated-defended `clear_attack_success` conditions across 4 query ids; ML-FilterRAG t=0.4 highest at 4 clear of 6 mutated defended generations; RAGDefender 0 clear successes on mutated contexts despite leaving residual poison on two queries.
- Report the adjudication finding as a result in its own right: 25 strict hits → 20 clear / 5 prefix-ambiguous, with ambiguity concentrated where the defense succeeded and the model answered parametrically.
- **The dissociation is the scientific point:** residual poison after defense does not monotonically produce the target answer (RAGDefender left 1-2 residual poison passages with 0 clear successes), while full evasion does (ML-FilterRAG 5/5 residual → bare target token). Retrieval preservation is necessary but not sufficient for an end-to-end claim — the paper's own data demonstrates the hierarchy it advocates.

**VI.H Additional defense family** — **PLANNED / PLACEHOLDER.** Empty labelled subsection with a one-line statement of what will be reported (Levels 0-2 at minimum: nominal behavior plus whether the three existing stress operators are *applicable by construction*). No prose until P1-1 lands.

### Section VII — Design Lessons (0.50 pg)

Six lessons, each anchored to a specific result so none is an opinion. **COMPLETED** as synthesis.

1. **Explicitly state defense assumptions, and distinguish stated from operative.** RAGDefender states "≥1 benign passage"; its operative dependence is poison-poison top-pair dominance. Anchor: VI.B (CORAL 0/24 vs E1 12/12).
2. **Evaluate outside the original attack distribution.** ML-FilterRAG reaches F1 0.935 on PoisonedRAG text and loses recall to 0.686 under feature normalization it never saw. Anchor: VI.C.
3. **Distinguish oracle failure from realizable attack.** The oracle that most reliably breaks RAGDefender (E1) is the one whose text-space approximation moved RAGDefender least (delta 0.00 for the RAGDefender-targeted family). Anchor: VI.B vs VI.E.
4. **Require retrieval preservation for attack claims.** Fixed-context evasion is a lower bound on nothing until retrieval is rerun under a preserved poison budget; our own audit found cross-query contamination that inflates aggregate figures. Anchor: VI.F.
5. **Require generation preservation for end-to-end claims.** Residual poison after defense did not produce the target answer under RAGDefender; token-boundary ASR alone produced a 5-of-25 false-positive rate. Anchor: VI.G.
6. **Evaluate security and clean-context utility together.** FilterRAG at k=5 achieves perfect poison removal by emptying the context; ML-FilterRAG at t=0.9 reaches 0.000 clean FPR at 0.678 recall. Neither number is interpretable alone. Anchor: VI.A, VI.C.

**Big Data topic alignment.** Frame lessons 1-3 as *resilience* (does the defense degrade gracefully outside its design distribution), 4-5 as *data integrity and trust* (what evidence licenses a claim that corrupted data reached the answer), and 6 as *ecosystem cost* (a defense that empties the context has moved the failure rather than fixed it). **Do not force a privacy contribution** — no experiment in this repository measures privacy, and asserting one would contradict Section VII's own lesson 1.

### Section VIII — Limitations (0.35 pg)

**COMPLETED** — every item below is already documented in the repository and must be stated, not discovered by a reviewer.

- Scale: Levels 3-5 are pilots (6 fixed-context queries, 3+3 full-retrieval queries, 6 end-to-end queries); mutations are hand-authored, not algorithmically generated.
- Single dataset and retriever for Levels 2-5 (HotpotQA, Contriever); NQ and MS MARCO have only n=1 defended coverage.
- Proxy substitutions: flan-t5-small for LLaMA-2/3 as the FilterRAG SLM; distilgpt2 for perplexity; training poison is our own LM-targeted text rather than the paper's GPT-4o-augmented set; gpt-3.5-turbo generator rather than GPT-4/LLaMA.
- Implementation fidelity: ML-FilterRAG is a top-k proxy, not the paper's top-s → filter → top-k pipeline; our RAGDefender Stage 1 uses OR where the paper specifies AND, includes the S_ii = 1 diagonal, and contains a flip branch; the embedder is `paraphrase-MiniLM-L6-v2`.
- Oracle limits: interventions operate in a frozen embedding/feature space; no transformed embedding has been shown to correspond to any real passage; n=5 poison vectors in 384 dimensions makes unregularized covariance estimation ill-posed, and the MMD conclusion holds only within the swept `steps ≤ 100`, `lr = 0.05`, `gamma = 1.0` range.
- Our own nominal attack strength is below the published PoisonedRAG figures, so absolute defended numbers are not directly comparable to Kim et al. or Edemacu et al.
- ε for FilterRAG was not retuned for the proxy SLM.

### Section IX — Conclusion (0.15 pg)

**PLANNED (writing).** Restate the hierarchy, the CORAL negative result, and the asymmetry between text-reachable and text-unreachable assumptions. No new claims.

---

## 4. Claim-status table

Evidence tiers: **N** nominal, **O** oracle, **F** fixed-context, **R** retrieval-realizable, **E** end-to-end.

| ID | Claim | Tier | Status | Scale | Primary artifact |
|---|---|---|---|---|---|
| C1 | PoisonedRAG black-box poison is effective in this harness | N/E | PRELIMINARY | 10q JSON on disk; `ANALYSIS.md` cites 100q at 76-77% — **reconcile before citing** | `results/query_results/main/` |
| C2 | RAGDefender fails under top-k saturation (k=N=5): residual 1.000, recall 0.560 | N | COMPLETED (detection-only) | 10q HotpotQA | `RAGDEFENDER_DIAGNOSTIC_REPORT.md` |
| C3 | RAGDefender recovers at k=10: residual 0.088, recall 0.880, clean FPR 0.200 | N | COMPLETED (detection-only) | 10q | same |
| C4 | RAGDefender outcomes are mechanistically attributable to top-pair composition; four distinct failure modes | N | COMPLETED | 8q instrumented | `RAGDEFENDER_CLUSTER_DIAGNOSTIC_FINDINGS.md` |
| C5 | E1 clean-anchor interpolation induces residual poison on originally-successful cases | O | COMPLETED (`VERIFY` denominator) | 6q × 4 strategies | `BATCH_COMPARISON_SUCCESS_CASES.md`, `FORMAL_ORACLE_COMPARISON.md` |
| C6 | `top_pair_pp` collapse is the operative failure condition; global CORAL/MMD alignment is not sufficient (CORAL 0/24 despite larger displacement) | O | COMPLETED | 6q, 4 methods | `FORMAL_ORACLE_COMPARISON.md` |
| C7 | MMD causes failure at weak/mid preservation (10/12) but not at `lambda_preserve=1.0` (0/6) | O | COMPLETED (narrow sweep) | 6q × 3 lambda | same |
| C8 | ML-FilterRAG-top-k separates PoisonedRAG poison: F1 0.935, AUC 0.962 | N | COMPLETED | 40/10 query split | `hotpotqa_50q_..._TRAIN_REPORT.md` |
| C9 | ML-FilterRAG at t=0.40 dominates threshold FilterRAG on clean FPR (0.781 → 0.156) at comparable recall (0.983 → 0.949) | N | COMPLETED | 10 held-out q, 118 poison / 32 clean rows | `ml_filterrag_eval_hotpotqa_50q_t04/` |
| C10 | FilterRAG at k=5 empties the context on 100% of queries at every ε — a utility failure clean FPR cannot express | N | COMPLETED | 10q | `FILTER_RAG_RERUN_AFTER_SLM_FIX_REPORT.md` |
| C11 | ML-FilterRAG feature oracle: recall 0.949 → 0.686 at alpha=0.4; locality matters (global centroid much weaker) | O | COMPLETED (`VERIFY` the alpha=0.2 figure) | 10 test q, 150 rows | `FEATURE_ORACLE_REPORT.md` |
| C12 | Generic paraphrase mutation barely evades any of the three defenses | F | PRELIMINARY | 4q, 3 bundles | `TEXT_MUTATION_FIXED_CONTEXT_REPORT.md` |
| C13 | FilterRAG-targeted rewrites are the only family weakening all three defenses (−0.33 / −1.17 / −2.17); RAGDefender-targeted rewrites move RAGDefender by 0.00 | F | PRELIMINARY | 6q × 3 families | `NORMALIZED_TARGETED_MUTATION_BUNDLE_1_REPORT.md` |
| C14 | Mechanism confirmation: Freq-Density down to 0.184 (< ε=0.2); one query's mean ML poison probability 0.23 (< t=0.4) | F | PRELIMINARY | 6q | same |
| C15 | 15/15 mutated poison passages survive rerun Contriever top-10 at ranks 1-5 under a preserved budget | R | PRELIMINARY | 3q | `FULL_RETRIEVAL_PILOT_REPORT.md` + origin audit |
| C16 | Full-retrieval weakening largely reproduces fixed-context weakening; RAGDefender most robust | R | PRELIMINARY | 3q (+3) | same |
| C17 | 12 mutated-defended clear attack successes across 4 query ids; RAGDefender 0 clear successes despite residual poison | E | PRELIMINARY | 6q, gpt-3.5-turbo | `EXPANDED_ANSWER_GENERATION_REPORT.md` |
| C18 | Token-boundary ASR over-counts: 25 strict hits → 20 clear / 5 prefix-ambiguous, ambiguity concentrated where the defense succeeded | E | COMPLETED (as a methodological result) | 55 generations | same |
| C19 | Reduced Freq-Density transfers to ML-FilterRAG because the two share a feature | F | PRELIMINARY | 6q | C13 + C14 jointly |

### Claims that must remain tentative or be excluded

- **Any ASR figure for a defended nominal condition.** The k-sweep was dry-run; no FilterRAG or ML-FilterRAG end-to-end run exists at any scale outside the 6-query mutation pilot. The current manuscript's "substantially reduces ASR" (E11) must be removed.
- **Any attribution of RAGDefender brittleness to distribution matching.** Contradicted by C6. The current manuscript's Fragility paragraph (E16) must be rewritten.
- **Any end-to-end ASR *rate*.** n=6 queries, hand-authored mutations, a single generator, and conditions pre-filtered to those where the defense already weakened. Report counts and the filter criterion.
- **Any claim of reproducing Kim et al. or Edemacu et al.** Proxy models, top-k rather than top-s, OR rather than AND, unretuned ε, and a weaker nominal attack.
- **Cross-dataset or cross-retriever generality.** Not measured for Levels 2-5.
- **Any claim about RAGDefender being robust in general.** The correct statement is narrower: robust to *hand-authored* rewrites in *this* pilot, while remaining fragile to a mechanism-targeted oracle — which is exactly why P0-3 exists.
- **Any privacy contribution.**

---

## 5. Missing experiments, prioritized

### P0 — can materially change the paper's conclusions

**P0-0 · Manuscript infrastructure.** Populate `paper_latex/references.bib` (~31 entries already enumerated in `docs/NORMALIZED_POISONING_LITERATURE_REVIEW.md` and `docs/CLUSTER_NORMALIZED_POISONING_EXECUTION_PLAN.md`); fix the six structural defects in §2.6. Blocks compilation of everything else. No compute.

**P0-1 · Expand full-retrieval mutation beyond the 3-query pilot.** Target ≥ 20-25 queries with the `filterrag_targeted` family.
- *Blocking prerequisite:* mutations are currently **hand-authored** GPT rewrites for 6 queries. Scaling Level 4 requires scaling Level 3 first — either scripting the rewrite step from the existing `gpt_prompt_packets*.jsonl` prompt packets, or building the constrained automated mutation pipeline of P0-3 and reusing it.
- *Existing harness to extend:* `scripts/run_full_retrieval_pilot_bundle1.py` (generalize the hardcoded 3-query candidate list), with `scripts/audit_full_retrieval_poison_origin_bundle1.py` run on every new query to catch cross-query contamination.
- *Changes the paper if:* retrieval survival drops below ~100%, which would invalidate C15 as stated.

**P0-2 · End-to-end generation on all retrieval-surviving mutations.** Generalize `scripts/run_expanded_answer_generation_bundle1.py`.
- Add a paper-comparable generator for the 12 flagged clear-success conditions: GPT-4 (`model_configs/gpt4_config.template.json`) and/or LLaMA-2-7b-chat (`model_configs/llama7b_config.template.json`) — both already configured.
- Replace or supplement the strict-match adjudicator with a semantic judge to eliminate the yes/no prefix ambiguity documented in C18.
- Also report the undefended mutated condition as the target-answer-preservation control (already logged as the `defense = none` rows).
- *Changes the paper if:* clear-success counts do not reproduce under GPT-4/LLaMA, which would downgrade C17 from a Level-5 claim to a generator-specific observation.

**P0-3 · Mechanism-targeted sentence-space mutation against RAGDefender.** The current RAGDefender-targeted family produced a mean delta of 0.00; this is the largest open question in the paper because it is the gap between C5 (oracle succeeds) and C13 (text fails).
- Build an operator with an explicit objective on the Stage-2 quantity rather than on discourse diversity: minimize `top_pair_pp` (or the pair-frequency score r_i) over the N=5 poison set, subject to SBERT/USE cosine semantic-preservation constraints, a fluency constraint, and target-answer preservation.
- Three candidate mechanisms: (a) constrained TextAttack/TextFooler search with a custom goal function reading `defense/ragdefender_internals.py::stage2_pair_frequency`; (b) per-passage distinct-anchor conditioning — write each of the five poison passages in a different discourse frame or entity focus, mirroring E1's `nearest_bijection` anchor assignment in text space; (c) set-level rewriting with an explicit cross-passage diversity penalty.
- No code exists (no TextAttack dependency in the repo). New script required.
- *Changes the paper if:* it succeeds, C13's asymmetry becomes "reachable with a stronger operator" and the paper gains a positive Level-3/4/5 result against a clustering defense. If it fails after a genuine attempt, that negative result is itself publishable and strengthens Lesson 3 — **either outcome is usable**, which is why this is P0.

**P0-4 · Nominal defended ASR (not in the original priority list, but Section VI.A depends on it).** No FilterRAG or ML-FilterRAG end-to-end run exists at any scale. Two options: (a) run live generation on the 10 held-out queries × {none, RAGDefender, FilterRAG, ML-FilterRAG} at k=10 (~40-80 generations, cheap), giving Table III a nominal ASR column; or (b) accept the framing decision in VI.A and report Levels 0-1 as detection-quality-plus-composition only, deferring all ASR to VI.G. **Option (b) requires no new experiments and is internally consistent**; option (a) makes the paper markedly stronger for roughly one afternoon of compute. Recommend (a) if any budget exists, with (b) as the guaranteed fallback.

### P1 — strengthens the contribution

**P1-1 · One orthogonal defense family (Sections IV.D, VI.H).** Nothing is implemented. Recommendation: **RobustRAG** (isolate-then-aggregate: answer each passage independently, then aggregate by keyword or decoding vote). Rationale: (i) it moves the decision from *passage filtering* to *answer aggregation*, so all three of the paper's stress operators — representation density, lexical density, feature separability — are inapplicable *by construction*, which is the sharpest possible demonstration of Lesson 1; (ii) it is the baseline Kim et al. themselves compare against, so it is defensible as a comparator choice; (iii) its certification framing connects directly to the venue's resilience theme. Alternatives if compute-bound: TrustRAG (K-means plus self-assessment — but it is *another* clustering/statistical filter, so it is a weaker orthogonality claim) or the PoisonedRAG paper's own perplexity and duplicate-filter defenses (cheapest; perplexity scoring already exists in `defense/ml_filterrag.py`). Minimum deliverable: Levels 0-2 only.

**P1-2 · Repeat core stress tests on a second dataset.** NQ is cheapest — `results/beir_results/nq-contriever.json` and `results/adv_targeted_results/nq.json` already exist. It is also *scientifically* the better choice than MS MARCO, because `defense/defense_runner.py` uses a **different Stage-1 estimator for NQ** (`_find_num_adversarial_agg`, agglomerative clustering plus TF-IDF overlap) than for HotpotQA. Repeating Levels 0-2 on NQ therefore tests whether the concentration assumption holds across *estimator variants*, not merely across corpora. MS MARCO is the useful contrast case if a third is affordable (published poison-vs-benign pairwise similarity gap 0.976 vs 0.309).

### P2 — explicitly out of scope; list in Limitations/Future Work only

Second retriever (`contriever-msmarco` and `ance` are already wired into `src/utils.py::load_models`, so this is cheap but not decision-changing); multiple additional generators beyond the one paper-comparable model in P0-2; runtime and cost studies; additional oracle variants (optimal transport, whitening, BERT-flow); agentic or multi-hop-planning RAG; provenance and remediation.

---

## 6. Proposed figures and tables

### Figures that should survive (4 must-have, 2 conditional)

- **Fig. 1 — Stress-testing hierarchy and interpretation gates.** Single column, ~0.30 pg, new TikZ. Levels 0-5 as a pipeline with a gate between each level labelled by what that level does *not* establish. **MUST SURVIVE** — it carries the entire framing and it is what a reviewer will remember. Section III.C.
- **Fig. 2 — RAGDefender mechanism panel.** Double column, ~0.35 pg. 2×2 subfigure built from the four existing PNGs already in `paper_latex/figures/`: success top-pair graph, failure top-pair graph, and the two PCA scatters. **MUST SURVIVE** (this is the compressed replacement for E12/E13's four separate half-width floats). Section VI.A.
- **Fig. 3 — Oracle dose-response.** Single column, ~0.28 pg, two stacked panels: RAGDefender residual poison vs alpha across E1 strategies, and ML-FilterRAG poison recall vs alpha across feature-oracle strategies. Merges the E1 sweep with `docs/figures/supervisor_briefing/03_ml_feature_oracle_recall_vs_alpha.png`. **MUST SURVIVE.** Sections VI.B/VI.C.
- **Fig. 6 — Text-space realization of the oracle signature.** Double column, ~0.30 pg. Schmeichel before/after (`05_pairgraph_schmeichel_before_after.png`) beside Gibson before/after (`07_pairgraph_gibson_before_after.png`). **MUST SURVIVE** — it is the only figure showing a real rewrite reproducing the oracle's mechanism signature in one case and evading a different mechanism without it in another. Section VI.E.
- **Fig. 4 — Cross-defense mutation heatmap.** Single column, ~0.25 pg, from `01_cross_defense_heatmap.png`. **CONDITIONAL** — foldable into Table VI columns if over budget. Section VI.E.
- **Fig. 5 — Mechanism drop under rewrite (Freq-Density and ML poison probability, before/after).** Single column, ~0.25 pg, from `04_mechanism_freq_density_and_ml_prob.png`. **FIRST TO CUT.** Section VI.E.

**Figures to exclude.** The standalone PCA scatters `06_pca_schmeichel_before_after.png` and `08_pca_gibson_before_after.png` (redundant with the pair graphs in Fig. 6); `02_full_retrieval_before_after.png` (superseded by Table VI, which carries rank and per-origin information a bar chart cannot); all 58 per-query oracle sweep plots; all runtime plots.

### Tables that should survive (5 must-have, 1 optional)

- **Table I — Stress-testing hierarchy.** Single column. Columns: level, operator, what is held fixed, what it establishes, what it does *not* establish. **MUST SURVIVE** — this table *is* the evidence hierarchy. Section III.C.
- **Table II — Defense assumption matrix.** Single column. Rows are defenses (including a placeholder row for IV.D); columns are decision mechanism, stated assumption, operative assumption, oracle result, text-realizable result. **MUST SURVIVE** — this is the "anatomy" of the title, and it is where the additional defense family slots in as one row without disturbing anything else. Section IV or VII.
- **Table III — Levels 0-1: detection quality and retrieval composition.** Double column. Dataset, k, defense, poison recall, clean FPR, residual poison fraction, top-pair PP/PC/CC, plus diagnostic control rows. **MUST SURVIVE.** Section VI.A.
- **Table IV — ML-FilterRAG threshold sweep (compressed).** Single column, 5 rows. **MUST SURVIVE** — E26 compressed from `table*` to `table`. Section VI.C.
- **Table V — Oracle method comparison.** Single column. Method (E1×2, CORAL-PCA, CORAL-ridge, MMD×3) × query-level failures, `top_pair_pp` base→max, MMD-distance reduction, mean L2 displacement. **MUST SURVIVE** — this table carries the CORAL negative result, i.e. §0.1. Section VI.B.
- **Table VI — Levels 4-5 ladder.** Double column, and the paper's most important structural element. One row per (query, defense, threshold): mutated poison retrieved and rank range → removed by defense (original vs mutated) → remaining poison → strict ASR → adjudicated label. Designed to be **append-only**: P0-1 and P0-2 add rows, and no prose in VI.F or VI.G changes. **MUST SURVIVE.** Sections VI.F/VI.G.
- **Table VII — Fidelity and proxy substitutions** (paper component vs our implementation). Single column. **OPTIONAL, second to cut** — content moves into Section V/VIII prose if cut.

**Tables to exclude.** The full 8-row `table*` threshold sweep (compressed into Table IV); the k-sweep worst-query table; the full per-origin contamination breakdown (one sentence plus a footnote in VI.F).

---

## 7. Designing for later insertion

Concrete mechanisms so P0 results drop in without touching the argument.

**Count macros.** Define every pilot scale as a macro in the preamble so scaling means editing one line, not hunting through prose:

```latex
\newcommand{\NfixedQ}{6}       % fixed-context mutation queries
\newcommand{\NfullretQ}{3}     % full-retrieval queries
\newcommand{\NendtoendQ}{6}    % end-to-end generation queries
\newcommand{\Nclearsucc}{12}   % clear attack success conditions
```

**Evidence-tier notation.** A superscript on every reported number naming its tier (`O` oracle, `F` fixed-context, `R` retrieval-realizable, `E` end-to-end), defined once in Table I's caption. This makes the hierarchy visible in every table rather than only asserted in Section III.D, and it means a reader cannot mistake an oracle number for an attack result.

**Pending markers.** `\newcommand{\PENDING}[1]{\textbf{[PENDING: #1]}}` used in IV.D and VI.H, so an incomplete section is unmistakable in draft and is a single grep away from being finished.

**Artifact provenance comments.** Every table and figure gets a LaTeX comment naming the exact artifact path that generated it, so a number can be re-verified without archaeology. Example: `% source: results/diagnostics/cluster_normalized_poisoning_formal/FORMAL_ORACLE_COMPARISON.md`.

**Append-only result tables.** Table VI's row schema is fixed now and its prose describes *what the pattern shows*, not *how many rows there are* (counts come from the macros). Table II's row schema accommodates the additional defense family as one row.

**Argument invariance check.** None of Sections I, III, IV.A-C, V, VII, or VIII depends on the *size* of the Level 3-5 pilots — only VI.D-G do, and those are written as pattern claims with macro-driven counts. If P0-1 through P0-3 all fail, the paper still stands on C2-C11 plus Lessons 1, 2, 3, and 6.

---

## 8. Prioritized execution order

1. **P0-0** — populate `references.bib`, fix the six structural defects (§2.6). Unblocks compilation. No compute.
2. **Skeleton restructure** — insert all nine section headings and the placeholder subsections, delete the appendix, compress E30 to six equations in IV.A, move E8/E16/E22/E23/E27 to their new homes *as-is* without rewriting. Produces a compiling document with the target structure.
3. **Correct the three unsupported/mis-attributed claims** — E11 (ASR → detection metrics), E16 (re-attribute brittleness from distribution matching to `top_pair_pp` collapse), E14 (separate the two RAGDefender failure modes). Do this before any new writing so the new prose is built on correct statements.
4. **Verify the three flagged figures** — the 22/24 denominator (E20), the "below 50% by alpha=0.2" claim (E27), and the C1 nominal ASR scale (10q on disk vs 100q in `ANALYSIS.md`).
5. **Write Sections III and IV plus Tables I, II, V and Figs. 1, 2, 3** — everything here is COMPLETED-tier and needs no new experiments. At the end of this step the paper's framing and its strongest result are fully drafted.
6. **P0-4** — decide (a) or (b); if (a), run the nominal defended ASR sweep now, since it completes Table III.
7. **Write Sections V, VI.A-C, VII, VIII** — still all COMPLETED-tier.
8. **P0-3** — mechanism-targeted mutation operator. Highest scientific value, longest lead time, and usable whichever way it resolves. Start the implementation in parallel with step 5 if there is capacity.
9. **P0-1** — scale full-retrieval mutation (depends on either scripted rewrites or P0-3's operator).
10. **P0-2** — end-to-end generation on all surviving mutations, with a paper-comparable generator and a semantic adjudicator.
11. **Write VI.D-G, populate Table VI, Figs. 4/6** — fill from whatever P0-1/P0-2 produced; prose is already scale-agnostic.
12. **P1-1** — additional defense family; fill IV.D, VI.H, and one row of Table II.
13. **P1-2** — second dataset (NQ first, for the different Stage-1 estimator).
14. **Abstract, Introduction, Conclusion last** — written against the final result set.
15. **Page-budget pass** — apply the cut order in §1 until the document fits 10 pages.

---

## 9. Manuscript edit map

All edits land in the single file `paper_latex/bigdata26_paper.tex`. Line numbers are as of today.

| Action | Location | Detail |
|---|---|---|
| Add `\usepackage{float}`, remove `\usepackage{appendix}` | L9-33 | §2.6 |
| Add count macros, `\PENDING`, evidence-tier macro | after L41 | §7 |
| Rewrite title | L51-52 | E1 |
| Write abstract | L73-75 | E4 |
| Write keywords | L77-79 | E5 |
| Write Section I | L86-87 | E6 |
| **Insert new Section II** | after L87 | E23 relocated here |
| Rename section, restructure into III.A-D | L92-95 | E7, E8 |
| Insert Table I, Fig. 1 | in III.C | new |
| **Delete section heading** | L102 | E9 |
| **Insert new Section IV**, subsections A-D | replaces L103-207 | E10-E27 redistributed |
| Compress equations into IV.A | from L240-459 | E30: keep 6 of 15 |
| Insert Table II, Fig. 2 | in IV | new/merged |
| Merge Setup + Metrics into Section V | L212-217 | E28 |
| Write Section VI with subsections A-H | L220-221 | E10-E27 results |
| Demote `table*` to `table`, drop 3 rows | L176-200 → VI.C | E26 → Table IV |
| Insert Tables III, V, VI and Figs. 3, 4, 6 | in VI | new/merged |
| **Insert new Section VII** | after VI | new |
| Write Section VIII | L223-224 | E29 |
| Write Section IX | L227-228 | — |
| **Populate** `paper_latex/references.bib` | separate file | E29, ~31 entries |
| **Delete** `\clearpage` + `\begin{appendices}...\end{appendices}` | L238-460 | E30 |

**Figure assets.** Four PNGs already in `paper_latex/figures/` are reused by Fig. 2. Figs. 3-6 require copying from `docs/figures/supervisor_briefing/` into `paper_latex/figures/` and regenerating at publication resolution — `scripts/plot_supervisor_comparison_figures.py` is the existing generator and will need a paper-figure mode (single/double column widths, larger fonts, no briefing-style titles).
