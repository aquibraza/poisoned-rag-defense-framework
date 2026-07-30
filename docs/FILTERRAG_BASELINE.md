# FilterRAG Baseline

> Status: **baseline implemented, diagnostics-first** (mirrors the
> RAGDefender diagnostics workflow in
> [docs/RAGDEFENDER_DIAGNOSTIC_PLAN.md](RAGDEFENDER_DIAGNOSTIC_PLAN.md)).
> This adds a second, independent defense family to compare against
> RAGDefender -- see [docs/ANALYSIS.md](ANALYSIS.md) §5c for the published
> numbers that motivated adding it.
>
> See [docs/FILTERRAG_FIDELITY_AUDIT.md](FILTERRAG_FIDELITY_AUDIT.md) for a
> detailed item-by-item audit against the paper's Algorithm 1, done prior to
> starting ML-FilterRAG. That audit found and fixed one load-bearing
> deviation not previously called out below: the keyword/passage-word
> matching inside Freq-Density was exact-token matching only, not the
> paper's semantic (cosine-similarity) word matching. `--filterrag_matching_mode
> semantic` (§2/§3 below) now implements the paper-faithful version;
> `exact` remains the default for backward compatibility.

## 1. What FilterRAG is

FilterRAG (Edemacu et al. 2025, *"Defending Against Knowledge Poisoning
Attacks During Retrieval-Augmented Generation"*) is a **per-passage
statistical filter**, unlike RAGDefender's cross-passage clustering. For
each retrieved passage `d` and query `q`:

1. Generate `SLM_answer` -- an answer to `q` produced by a **small** language
   model conditioned on `d` *alone* (not the full retrieved context).
2. Compute:
   ```
   Freq-Density(d) = sum(freq(w, d) for w in (q ⊕ SLM_answer) if w in d) / UniqueWords(d)
   ```
3. Drop `d` if `Freq-Density(d) >= epsilon` (paper default `epsilon = 0.2`).

Rationale: PoisonedRAG-style adversarial passages are deliberately stuffed
with query/answer keywords (to simultaneously win retrieval and steer
generation), which produces a measurably higher Freq-Density than ordinary
corpus text. Because this signal is computed **independently per passage**,
it doesn't require a "clean anchor cluster" the way RAGDefender's
concentration/clustering heuristics do -- which is exactly the assumption
that breaks at `k=N` (100% poisoned retrieved context); see
`docs/RAGDEFENDER_DIAGNOSTIC_PLAN.md` and `docs/ANALYSIS.md` §5b.

Published results (Table I, black-box attack, GPT-4): FilterRAG reduces ASR
from ~82-90% (no defense) to **2-9%** across NQ/HotpotQA/MS MARCO while
keeping ~81-90% accuracy -- see `docs/ANALYSIS.md` §5c for the full table.

**ML-FilterRAG** (Freq-Density + perplexity + log-probability -> a trained
XGBoost/Random Forest classifier) is **not implemented here**. It requires
labeled poison/clean training data and a training pipeline, which is a
separate, heavier lift; like `ragdefender_fixed`, it is deferred until after
the threshold-based `filterrag` baseline has been diagnosed. See "Deferred
work" below.

## 2. What's implemented in this repo

- `defense/filterrag.py`:
  - `freq_density(passage_text, keywords, matching_mode=..., semantic_threshold=..., semantic_matcher=...)`
    -- the pure scoring function; `matching_mode="exact"` (default) is
    dependency-free and byte-identical to the original implementation,
    `matching_mode="semantic"` requires a `SemanticWordMatcher` (lazily
    loaded on first use). Thin wrapper around `freq_density_detailed()`.
  - `freq_density_detailed(...)` -- same inputs, returns the full breakdown
    dict (`freq_density_score`, `unique_word_count`, `matched_keyword_count`,
    `matched_keywords`, `matching_mode`, `semantic_threshold`).
  - `score_passages(query, passages, slm_answer_fn=..., matching_mode=..., semantic_threshold=..., semantic_matcher=...)`
    -- per-passage score breakdown: `doc_id`, `freq_density_score`,
    `slm_answer` (unchanged keys) plus `matching_mode`, `semantic_threshold`,
    `unique_word_count`, `matched_keyword_count`, `matched_keywords_sample`
    (capped list, avoids huge diagnostic output).
  - `filterrag_defense(query, passages, epsilon=..., slm_answer_fn=..., matching_mode=..., semantic_threshold=..., semantic_matcher=...)`
    -- applies the `epsilon` threshold and returns `(kept_passages, diag_extra)`
    in the same shape every other defense in `defense/dispatch.py` returns,
    so it plugs into the existing diagnostics pipeline
    (`defense/diagnostics.py`) unchanged.
  - `SemanticWordMatcher` / `get_semantic_word_matcher(model_name=...)` --
    word-level cosine-similarity matcher backed by
    `sentence-transformers/all-MiniLM-L6-v2` (paper default, Section
    IV-B2), lazy-loaded and cached (both the HF model and per-word
    embeddings), used only when `matching_mode="semantic"`. See
    `docs/FILTERRAG_FIDELITY_AUDIT.md` §4.
  - `local_hf_slm_answer_fn(model_name=..., max_new_tokens=..., device=...)` --
    builds an `SlmAnswerFn` backed by a small local HuggingFace seq2seq model
    (lazy import + lazy load + cached, so importing `defense.filterrag`
    never requires `transformers`/`torch` unless this is actually called).
  - `resolve_slm_device(requested="auto")` -- picks Apple Silicon Metal/MPS
    if available, else CUDA, else CPU; an unavailable explicit request falls
    back to auto-detection with a logged warning instead of failing.
- `defense/dispatch.py`: two new `--defense` choices --
  - `filterrag` -- the full algorithm, using a local SLM (see "SLM choice"
    below) to generate `SLM_answer` per passage.
  - `filterrag_query_only` -- a **diagnostic ablation**, not in the paper:
    skips the SLM step entirely and scores passages using only the query's
    own keywords. This is a fast, fully-offline, zero-cost mode useful for
    smoke-testing the plumbing and for a first-pass sanity check, but it is
    *not* the full published algorithm -- it can't catch passages stuffed
    with the *answer* but not much of the question text, so it's expected to
    under-perform `filterrag` (higher residual poison / worse recall on
    passages that paraphrase the question).
- `main.py`: `--filterrag_epsilon` (default 0.2, matching the paper),
  `--filterrag_slm_model` (default `google/flan-t5-small`),
  `--filterrag_slm_device` (`auto`/`cpu`/`mps`/`cuda`, default `auto`),
  `--filterrag_matching_mode` (`exact`/`semantic`, default `exact` -- see
  §2.1 below), and `--filterrag_semantic_threshold` (default 0.6, matching
  the paper) CLI flags, wired straight through to `run_defense()`.

### 2.1 Matching mode: `exact` (default) vs. `semantic` (paper-faithful)

`--filterrag_matching_mode` controls how Freq-Density decides whether a
keyword (from `query ⊕ SLM_answer`) "matches" a word in the passage:

- `exact` (default): verbatim, case-folded string equality. This is the
  original implementation's behavior, unchanged, kept as the default so
  existing scripts/diagnostics runs (`scripts/filterrag_score_inspection.py`,
  `scripts/run_ragdefender_k_sweep.py`) are unaffected. **Not
  paper-faithful** -- see `docs/FILTERRAG_FIDELITY_AUDIT.md` §3.2: this is
  equivalent to the paper's own similarity-threshold ablation at
  threshold=1.0, the worst-performing setting in their Table II.
- `semantic`: cosine similarity of `sentence-transformers/all-MiniLM-L6-v2`
  word embeddings, thresholded by `--filterrag_semantic_threshold` (default
  0.6). This matches the paper's Section IV-B2 default exactly. Use this
  for paper-fidelity runs: `--filterrag_matching_mode semantic`.
  `sentence_transformers` is imported lazily -- only when this mode is
  actually exercised, same convention as `transformers`/`torch` for the SLM
  backend.

Both modes apply identically to `filterrag` and `filterrag_query_only`
(matching mode and SLM-vs-query-only-keywords are orthogonal); neither
makes `filterrag_query_only` paper-faithful, since that mode always skips
the SLM step regardless of matching mode.
- `scripts/run_ragdefender_k_sweep.py`: `filterrag`/`filterrag_query_only`
  added to `ALL_DEFENSE_CHOICES`; a `--quick_filterrag_hotpotqa` preset
  mirroring `--quick_hotpotqa` (same HotpotQA/max_queries=10/N=5/k=[5,10])
  but with `defenses=[none, filterrag_query_only, filterrag]`, for a direct,
  paired comparison against the RAGDefender quick-diagnostics results
  already collected.
- `scripts/summarize_ragdefender_diagnostics.py`: the detection-quality table
  (§1) and CSV output are defense-name-agnostic and already include
  `filterrag`/`filterrag_query_only` rows with no changes needed; the
  RAGDefender-vs-oracle-vs-random comparison table (§6) and `REMOVAL_DEFENSES`
  now explicitly list the two FilterRAG defense names too. The
  **interpretation decision tree (§3) remains RAGDefender-specific** for now
  (see "Deferred work").
- `defense/filterrag.py`'s `diag_extra["filterrag_scores"]` carries the full
  per-passage `(doc_id, freq_density_score, slm_answer)` breakdown beyond
  what the shared per-query diagnostic schema captures, in case deeper
  per-passage analysis is needed later (not currently persisted to the JSONL
  diagnostic record -- only `N_adv_estimated_by_ragdefender`/`notes`/removal
  counts are, same as every other defense).

## 3. Known deviations from the published method

- **SLM choice**: the paper uses LLaMA-2/3 as the SLM. This machine has no
  CUDA/NVIDIA GPU; it does have Apple Silicon Metal/MPS (an M4 MacBook Air),
  but a 7B model would still be far too slow to run once per retrieved
  passage per query at any useful scale on it. `local_hf_slm_answer_fn()`
  defaults to **`google/flan-t5-small`** (~80M params) as a small, fast
  proxy instead -- this default is unchanged regardless of which accelerator
  is available. This is a known fidelity tradeoff: a much smaller model
  likely produces lower-quality/less-specific `SLM_answer` text than the
  paper's LLaMA-2/3, which could under- or over-estimate Freq-Density
  relative to the published numbers. The SLM is fully pluggable
  (`slm_answer_fn` parameter / `--filterrag_slm_model` flag) so a larger
  model (e.g. the repo's existing `model_configs/llama7b_config.json`) can be
  substituted later, without changing `freq_density()`/`filterrag_defense()`
  at all.
- **Device placement**: `resolve_slm_device()` in `defense/filterrag.py`
  auto-selects the SLM's device -- Apple Silicon Metal/MPS if
  `torch.backends.mps.is_available()`, else CUDA if available, else CPU.
  Override with `--filterrag_slm_device {auto,cpu,mps,cuda}`; an explicit
  `mps`/`cuda` that isn't actually available on the machine logs a warning
  and falls back to auto-detection rather than failing the run.
  **In practice, on this repo's dev machine, `--filterrag_slm_device auto`
  first tries `mps` (it *is* available -- `torch.backends.mps.is_available()`
  is `True`) but `google/flan-t5-small` fails an immediate post-load
  smoke-test generation on it, so `_get_local_hf_slm_pipeline()` falls back
  to `cpu` before any real passage is ever scored -- `--defense filterrag`
  runs the SLM on CPU on this machine, not MPS** (see "Known issue" and
  §3.1 immediately below for why). The resolved *final* device is logged
  once (e.g. `[FilterRAG] SLM device: cpu (model=google/flan-t5-small)`,
  preceded by the fallback warning if one occurred), not per-passage/per-query.
  **Known issue (found + fixed during epsilon calibration, see §3.1
  below):** on this repo's dev environment (torch==1.13, transformers==4.30,
  Apple Silicon M4), `google/flan-t5-small` on `mps` fails *every single*
  generation call with `TypeError: Operation 'abs_out_mps()' does not
  support input type 'int64' in MPS backend` (torch 1.13's MPS backend
  doesn't implement `abs()` on int64 tensors, which T5's relative-position-
  bias attention needs). `_get_local_hf_slm_pipeline()` now runs one
  throwaway smoke-test generation immediately after loading any non-CPU
  pipeline and falls back to `cpu` (with a printed warning) if it fails,
  specifically so this doesn't silently degrade into "every SLM answer is
  empty" the way it did before this fix (see §3.1). `flan-t5-small` on CPU
  is fast enough in practice (~0.3-0.6s/passage) that this fallback has no
  material runtime cost at this repo's scale (10s of queries). A larger
  model, a newer torch version, or a non-T5 architecture could still
  legitimately run on `mps` -- the smoke test decides this per
  `(model_name, device)` pair, it is not a hardcoded "never use MPS" rule.
- **`filterrag_query_only`** (see above) is an added diagnostic ablation, not
  part of the paper. It exists purely so the pipeline (dispatch, diagnostics
  logging, k-sweep, summarizer) can be smoke-tested and run at zero cost
  before spending any time/compute on the SLM-backed `filterrag` mode -- the
  same "cheap diagnostics before expensive runs" pattern used throughout
  `docs/RAGDEFENDER_DIAGNOSTIC_PLAN.md`.
- **Keyword/passage-word matching defaults to `exact`, not the paper's
  `semantic`** (§2.1 above; found and fixed by
  `docs/FILTERRAG_FIDELITY_AUDIT.md`): the paper's Freq-Density intersection
  `(qi⊕aj) ∩ dj` is defined as a cosine-similarity match
  (`all-MiniLM-L6-v2`, threshold 0.6), not literal string equality.
  `--filterrag_matching_mode semantic` now implements this faithfully; the
  default stays `exact` for backward compatibility with runs/scripts
  written before this option existed, so paper-fidelity runs must opt in
  explicitly.
- **ML-FilterRAG is out of scope** (see "Deferred work").
- **Epsilon is not re-tuned for `flan-t5-small`**: the paper's `epsilon=0.2`
  was presumably tuned against LLaMA-2/3-generated `SLM_answer` text; with a
  much smaller substitute SLM, the "right" threshold for this repo's setup
  may differ. `--filterrag_epsilon` is exposed specifically so this can be
  swept/tuned once live diagnostics are available, rather than trusting the
  paper's value blindly.

### 3.1 Calibration finding: the SLM step was silently inert on MPS

The first detection-only diagnostic sweep (HotpotQA, 10 queries, k=5/10,
epsilon=0.2) found that `filterrag` (SLM-backed) and `filterrag_query_only`
(no SLM) produced **byte-for-byte identical removal decisions** on every
query. The root cause was not epsilon miscalibration: it was the MPS/T5
bug described in §3 above. `local_hf_slm_answer_fn()`'s per-passage
`try/except Exception: return None` swallowed that `TypeError` completely
silently, so `google/flan-t5-small` produced an **empty `SLM_answer` for
100% of the 150 passages scored** in that sweep -- `filterrag`'s Freq-Density
score was therefore mathematically identical to `filterrag_query_only`'s
(keywords = query tokens only, in both modes) even though the code path
*looked* like it was running the full SLM-backed algorithm.

`scripts/filterrag_score_inspection.py` was added to catch exactly this
class of failure: it computes both the full and query-only score for every
retrieved passage up front (rather than trusting one epsilon's removal
decision), so a "0% of scores differ" result is visible immediately instead
of looking like a plausible epsilon-tuning outcome. After the
`_get_local_hf_slm_pipeline()` smoke-test fallback (§3, "Device placement")
was added, `flan-t5-small` runs on CPU instead, producing real (often
multi-word) answers, and the two modes now diverge on **38/100 passages at
k=10** (10-query HotpotQA set) -- e.g. an SLM answer of "Neither are solely
used for real estate transactions." raises a poisoned passage's score from
0.893 (query-only) to 1.000 (full), and an answer of "Mrs. Tiggy-Winkle"
(matching a specific hedgehog character name absent from the query) raises
another poisoned passage from 0.552 to 0.655. See
`results/diagnostics/filterrag_calibration_10q/FILTERRAG_SCORE_EXAMPLES.md`
and `FILTERRAG_EPSILON_SWEEP_REPORT.md` for the full breakdown across
epsilon in [0.2, 0.8]. At low epsilon (<=0.5) both modes already saturate
poison recall to 1.0 on this 10-query set (the injected adversarial text is
heavily keyword-stuffed with the query itself, so query-only keywords alone
are already enough), so the SLM's incremental benefit shows up mainly at
higher epsilon (e.g. k=10, epsilon=0.6: poison recall 0.90 full vs. 0.84
query-only) and in the qualitative per-passage examples above, not yet in
the aggregate metrics at the paper's default epsilon=0.2 on this small
sample.

**Lesson for future baselines**: never trust a defense's own removal
decisions alone to validate that an optional sub-component (here, the SLM)
is actually contributing -- a component that fails 100% of the time can be
indistinguishable from "this ablation doesn't matter" unless the raw
per-item signal (here, `slm_answer` text and the two score variants) is
inspected directly.

## 4. How to run diagnostics

Exactly the same two-layer cost-control pattern as the RAGDefender
diagnostics (see `docs/RAGDEFENDER_DIAGNOSTIC_PLAN.md` §6):

```bash
# 1. Print commands only (no subprocess spawned):
python scripts/run_ragdefender_k_sweep.py --quick_filterrag_hotpotqa

# 2. Actually run them, but main.py --dry_run True (zero LLM API cost;
#    filterrag's own SLM calls to flan-t5-small are still made -- they are
#    free/local, not billed API calls):
python scripts/run_ragdefender_k_sweep.py --quick_filterrag_hotpotqa --execute

# 3. Summarize:
python scripts/summarize_ragdefender_diagnostics.py \
  --diagnostics_dir results/diagnostics/ragdefender

# 4. Only once diagnostics look sane, opt into real (billed) generation:
python scripts/run_ragdefender_k_sweep.py --quick_filterrag_hotpotqa --execute --live_generation
```

For epsilon calibration / inspecting per-passage scores directly (no
`main.py`/dispatch involved, no live GPT generation -- see §3.1 above):

```bash
python scripts/filterrag_score_inspection.py \
  --eval_dataset hotpotqa --k_values 5 10 --N 5 --max_queries 10 \
  --out_dir results/diagnostics/filterrag_calibration_10q
```

This writes a per-passage score CSV (`filterrag_score_inspection.csv`), an
epsilon-sweep summary CSV/report (`filterrag_epsilon_sweep.csv`,
`FILTERRAG_EPSILON_SWEEP_REPORT.md`), and a handful of worked examples
(`FILTERRAG_SCORE_EXAMPLES.md`) -- all computed from one retrieval +
one-pass-per-passage SLM scoring run, so sweeping `--epsilons` costs nothing
extra.

Note: unlike RAGDefender (which only needs `sentence-transformers` +
`torch`), `--defense filterrag` additionally downloads
`google/flan-t5-small` from the HuggingFace Hub on first use (a few hundred
MB, one-time, cached under `~/.cache/huggingface/hub`) and runs it locally
once per retrieved passage per query, on whichever device
`resolve_slm_device()`/`_get_local_hf_slm_pipeline()` resolves to at
runtime -- `auto` tries Apple Silicon Metal/MPS first on this repo's
development machine, but falls back to CPU there because
`google/flan-t5-small` fails its post-load smoke test on `mps` (see §3
"Device placement" and §3.1) -- this makes step 2 noticeably slower than
the equivalent RAGDefender run, especially at higher `k`.
`filterrag_query_only` has no such cost and is exactly as fast as `none`.

## 5. Deferred work

- **ML-FilterRAG**: needs a labeled poison/clean training set (can be built
  from this repo's own ground-truth `is_poison` labels) plus a
  perplexity/log-probability feature extractor and a trained
  XGBoost/Random Forest classifier. Left for a follow-up once the
  threshold-based `filterrag` baseline's diagnostics have been reviewed --
  same "diagnose before you build the next thing" discipline applied to
  RAGDefender's `ragdefender_fixed`.
- **Decision tree generalization**: `render_decision_tree()` in
  `scripts/summarize_ragdefender_diagnostics.py` is currently written
  specifically for RAGDefender's saturation hypothesis (H1-H5 in
  `docs/RAGDEFENDER_DIAGNOSTIC_PLAN.md`). FilterRAG's failure modes may
  differ (e.g. it doesn't rely on a clean-cluster assumption at all, so the
  k=N-vs-k>N comparison may be far less interesting for it, but SLM answer
  quality and epsilon miscalibration become the new candidate failure
  modes). A FilterRAG-specific interpretation section should be added once
  live diagnostics are available to write it against real numbers rather
  than speculation.
