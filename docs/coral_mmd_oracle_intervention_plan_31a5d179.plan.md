---
name: CORAL/MMD Oracle Intervention Plan
overview: "Design (do not implement) the next Cluster-Normalized Poisoning phase: formal CORAL-style covariance-alignment and MMD-minimizing oracle interventions on RAGDefender's embedding space, building on the E1 clean-anchor baseline, and write the design to docs/CORAL_MMD_ORACLE_INTERVENTION_PLAN.md."
todos: []
isProject: false
---


# CORAL/MMD Oracle Intervention Plan (design only)

Deliverable of this task: write `docs/CORAL_MMD_ORACLE_INTERVENTION_PLAN.md` containing the design below. No code, no `.gitignore` edit, no test files, and no runs are produced in this task — only the planning document.

## Grounding (Phase 1 recap, full findings above)

- Raw embeddings exist only transiently inside `scripts/run_cluster_normalized_poisoning.py::main()` (via `viz.encode_texts`); no run artifact on disk contains them, only `similarity_matrices/*.npy` cosine matrices.
- `defense/distribution_metrics.py` is diagnostic-only (Gram-matrix identities); it cannot produce new transformed poison coordinates. CORAL/MMD **transforms** need raw embeddings; CORAL/MMD **distance reporting** does not (existing Gram code is reused as-is).
- `defense/ragdefender_internals.py::concentration_stage1`/`stage2_pair_frequency`, `defense/cluster_normalized_poisoning.py::split_poison_clean`/`recombine_poison_clean`/`l2_normalize_rows`, `run_cluster_normalized_poisoning.py::decision_label`/`build_run_config`, and `summarize_cluster_normalized_poisoning.py::discover_run_dirs`/`latest_run_per_intervention`/`load_sweep` are all reused unmodified.

## New module: `defense/coral_mmd_intervention.py`

Kept separate from `defense/cluster_normalized_poisoning.py` (whose docstring scopes it to E0/E1 only) but following the same conventions (pure numpy/torch, no LLM/API calls, never imported by `defense_runner.py`/`dispatch.py`/`filterrag.py`/`main.py`).

**CORAL-style transform** (`coral_transform(z_poison, z_clean, lam, beta, rank=None)`):
1. Mean-center `Zp`, `Zc`.
2. Regularized covariances `Cp = cov(Zp) + lam*I`, `Cc = cov(Zc) + lam*I` (`lam > 0` required — see rank-deficiency discussion below).
3. Whiten via `Cp^{-1/2}` (eigendecomposition, clip tiny/negative eigenvalues from float noise to a small positive floor before taking `1/sqrt`), recolor via `Cc^{1/2}`.
4. Add back clean mean (or an interpolated mean, configurable but default = clean mean).
5. Mix: `Zp_final = normalize((1-beta)*Zp_original + beta*Zp_coral)`.

**Rank-deficiency discussion (must appear in the doc, not just here):**
- With `n=5` points in `d=384` dims, the centered covariance has rank `<= n-1 = 4`; 380 eigenvalues are exactly (or numerically) zero.
- Full-dimensional **unregularized** CORAL is invalid: `Cp^{-1/2}` is undefined (division by zero) in the null space.
- Ridge regularization (`+lam*I`) makes inversion well-defined everywhere, but in directions where the true signal variance is ~0, the correction is dominated entirely by `lam` (an invented value, not data) once inverted — this must be called out explicitly as a limitation of the full-384-dim ridge variant.
- **PCA/subspace variant** (`rank <= min(n_p-1, n_c-1) = 4`): SVD-project onto the top-`rank` singular directions of the centered poison data (the only directions with genuine signal), apply whiten/recolor only within that subspace, leave the orthogonal complement unchanged. This is the more defensible default; the full-dim ridge variant is retained only as a secondary point of comparison, with its limitation stated plainly in the report.
- Numerical stability: eigenvalue floor (e.g. `1e-8`), symmetrize covariance before eigendecomposition, and assert all outputs are finite before returning (feeds Test 2).

**Sweep:** `lambda in {1e-1, 1e-2, 1e-3}` x `beta in {0.0 (identity sanity), 0.25, 0.5, 0.75, 1.0}`.

**MMD-minimizing transform** (`mmd_minimize(z_poison, z_clean, lambda_preserve, gamma, steps, lr)`):
- Implemented with **PyTorch autograd** (torch is already a pipeline dependency; trivial compute at `n<=10`, `steps<=100`) rather than hand-derived analytic gradients — lower risk of a subtle math bug.
- Loss: `L = MMD_RBF(Zp_prime, Zc) + lambda_preserve * ||Zp_prime - Zp||^2`. L2-normalize `Zp_prime` by projection after every optimizer step (simpler than an added `lambda_norm` penalty term; the penalty-term alternative is documented as a later option, not implemented in this first pass, per the user's "keep it simple" instruction).
- `gamma` fixed at `DEFAULT_MMD_GAMMA=1.0` (reusing `defense/distribution_metrics.py`'s constant) for the first pass; median-heuristic bandwidth documented as a later option, not implemented.
- **Sweep:** `lambda_preserve in {0.01, 0.1, 1.0}`, `steps in {0 (identity sanity), 50, 100}`, conservative fixed `lr` (e.g. `0.05`, chosen so 100 steps converges without overshooting on toy fixtures — exact value pinned during implementation, not fabricated here).
- Trace every step: `step, loss, mmd, preserve_loss, mean_pp_sim, top_pair_pp, removed_poison` (the last three computed by recombining the current `Zp_prime` with `Zc`, recomputing cosine + Stage 1/2 at each step — trivial cost at `k=10`).

## New script: `scripts/run_coral_mmd_oracle_intervention.py`

A **batch** runner (unlike per-query `run_cluster_normalized_poisoning.py`) covering all 6 E1 success-case queries in one invocation, since Phase 5's output layout is one directory per batch run, not per query:

- Reuses `_force_offline_env`, `viz.recover_pre_defense_texts`/`load_embedder`/`encode_texts`/`cos_sim_from_embeddings`, `split_poison_clean`/`recombine_poison_clean`, `concentration_stage1`/`stage2_pair_frequency`, `decision_label` (imported from `run_cluster_normalized_poisoning.py`, not re-duplicated a third time), and `defense/distribution_metrics.py`'s Gram-based metrics for before/after CORAL/MMD distance reporting.
- Default query set = the 6 tested query_ids from `BATCH_COMPARISON_SUCCESS_CASES.md` (obtained via `build_batch_comparison_success_cases.discover_success_case_ids` + `check_text_recoverable`, not hardcoded).
- **E1 reference rows are never recomputed** — pulled from existing run directories via `summ.discover_run_dirs`/`latest_run_per_intervention`/`load_sweep`, defaulting to `rank_aligned` and `nearest_bijection` (configurable), with distribution metrics attached via `build_distribution_metrics_batch.attach_distribution_metrics` (also reused, not reimplemented). This directly satisfies "do not replace E1" and "do not rerun baseline retrieval."
- For CORAL and MMD, per query: run the sweep grid, save each transformed cosine matrix to `similarity_matrices/`, append one row per `(query_id, method, params)` to `CORAL_SWEEP.csv`/`MMD_SWEEP.csv` with columns: coral/MMD distance before/after, mean PP/PC/CC similarity, top_pair_pp/pc/cc, `N_adv`, selected indices, `removed_poison`, `removed_clean`, `residual_poison_fraction`, `decision_label`.
- `METHOD_COMPARISON.csv`: one row per `(query_id, method)` — E1 reference strategies, best CORAL config, best MMD config — comparing: which causes residual-poison failure at the least perturbation, which reduces `top_pair_pp` most efficiently, which reduces CORAL/MMD distance most, which best preserves closeness to the original poison embeddings (`||Zp_final - Zp_original||`).
- `run_config.json`/`manifest.json` follow `build_run_config`'s exact conventions (git commit/status, argv, dependency versions, extended `oracle_constraints` block with `coral_lambda`, `coral_beta`, `coral_rank`, `mmd_lambda_preserve`, `mmd_gamma`, `mmd_steps`, `mmd_lr`, plus the existing `claims_text_realizable_attack: false`, `gpt_or_api_calls_made: false`, `baseline_files_modified: []`).
- `CORAL_MMD_INTERVENTION_REPORT.md` renders the sweep tables, the method comparison, and verbatim Phase 7 claims/limitations text (extending the existing `LIMITATIONS_TEXT` pattern from `run_cluster_normalized_poisoning.py`/`build_distribution_metrics_batch.py`).

## Output layout (Phase 5, per the user's spec)

```
results/diagnostics/cluster_normalized_poisoning_formal/
  <timestamp>_coral_mmd_hotpotqa_k10_N5/
    run_config.json
    manifest.json
    CORAL_SWEEP.csv
    MMD_SWEEP.csv
    METHOD_COMPARISON.csv
    CORAL_MMD_INTERVENTION_REPORT.md
    traces/        (one CSV per (query_id, lambda_preserve, steps) MMD run)
    similarity_matrices/
```

`.gitignore` note: `results/` is already a blanket-ignored top-level entry (`.gitignore:181`), so this new directory is already excluded from git with no edit required. For documentation parity with the repo's existing convention of listing generated-artifact directories as commented-out entries (e.g. `.gitignore:186`, `# results/diagnostics/ragdefender_cluster_viz/`), a matching commented line for `cluster_normalized_poisoning_formal/` will be added under that same block — functionally redundant but consistent with repo style. This will be called out explicitly in the doc as a documentation-only addition, not a required gitignore fix.

## Tests to plan for (new `tests/test_coral_mmd_intervention.py`)

Mirroring `tests/test_cluster_normalized_poisoning.py`'s conventions (deterministic tiny fixtures, `FakeSentenceTransformer`, no real model/network):
1. CORAL transform shape preservation across the `lambda x beta` grid, both full-dim and PCA-rank variants.
2. CORAL transform returns finite values (no NaN/Inf) across the sweep, including `lambda=1e-3`.
3. `beta=0` reproduces original Stage 1/2 decisions exactly (identity check, same pattern as `TestIdentityRegressionAtAlphaOne`).
4. MMD loss strictly decreases over optimization steps on a toy well-separated poison/clean example.
5. `steps=0` reproduces original Stage 1/2 decisions exactly.
6. L2-normalized transformed embeddings (CORAL and MMD) have row norm ≈ 1.
7. Stage 1/2 recomputation runs successfully (valid `ConcentrationResult`/`Stage2Result`) after every transform in both sweeps.
8. No GPT/API calls: AST-based import guard (no `openai`/`requests`/`httpx`/`urllib`/`sentence_transformers` new imports beyond what's already offline-guarded) plus a `run_config.json["oracle_constraints"]["gpt_or_api_calls_made"] is False` assertion, matching existing test patterns.

An end-to-end smoke test for `scripts/run_coral_mmd_oracle_intervention.py` (faked embedder, tiny synthetic fixtures, tmp output dir) will also be planned, mirroring `tests/test_build_distribution_metrics_batch.py`'s end-to-end pattern.

## Claims and limitations (Phase 7, must appear verbatim in the doc and every run's report)

- CORAL/MMD interventions remain **oracle embedding-space interventions** — no natural-language rewrite of any poisoned passage is performed or implied.
- They are **stronger formal stress tests than E1** (distribution-alignment objectives rather than ad hoc interpolation) but still **not text-realizable attacks**.
- They test whether RAGDefender's similarity-based decision boundary is fragile under formal distribution alignment, not whether such alignment is reachable by rewriting text under the frozen encoder.
- Text-space approximation remains an explicitly later, out-of-scope phase.
- FilterRAG/ML-FilterRAG comparisons come only after RAGDefender formal oracle stress testing is complete.
- Small-sample caveats (`n=5` per group, `d=384`) apply to both CORAL and MMD and must be restated per-run, not just referenced.
- E1 is **not replaced**: it remains the empirical oracle baseline; CORAL/MMD are formal follow-ups compared against it, never supersede its findings.

## Todos for this task
</plan>
<todos>
<todo id="write-plan-doc">Write docs/CORAL_MMD_ORACLE_INTERVENTION_PLAN.md with the full design above (Phase 1 inspection summary, Phase 2 CORAL design incl. rank-deficiency/PCA discussion, Phase 3 MMD design, Phase 4 MVE scope, Phase 5 output structure, Phase 6 test plan, Phase 7 claims/limitations)</todo>
</todos>
