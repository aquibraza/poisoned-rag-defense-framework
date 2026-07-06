# FilterRAG Baseline

> Status: **baseline implemented, diagnostics-first** (mirrors the
> RAGDefender diagnostics workflow in
> [docs/RAGDEFENDER_DIAGNOSTIC_PLAN.md](RAGDEFENDER_DIAGNOSTIC_PLAN.md)).
> This adds a second, independent defense family to compare against
> RAGDefender -- see [docs/ANALYSIS.md](ANALYSIS.md) §5c for the published
> numbers that motivated adding it.

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
  - `freq_density(passage_text, keywords)` -- the pure scoring function,
    fully deterministic and dependency-free.
  - `score_passages(query, passages, slm_answer_fn=...)` -- per-passage score
    breakdown (`doc_id`, `freq_density_score`, `slm_answer`).
  - `filterrag_defense(query, passages, epsilon=..., slm_answer_fn=...)` --
    applies the `epsilon` threshold and returns `(kept_passages, diag_extra)`
    in the same shape every other defense in `defense/dispatch.py` returns,
    so it plugs into the existing diagnostics pipeline
    (`defense/diagnostics.py`) unchanged.
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
  `--filterrag_slm_model` (default `google/flan-t5-small`), and
  `--filterrag_slm_device` (`auto`/`cpu`/`mps`/`cuda`, default `auto`) CLI
  flags, wired straight through to `run_defense()`.
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
  CUDA/NVIDIA GPU; Apple Silicon Metal/MPS may be available (the current
  development machine is an Apple Silicon M4 MacBook Air, which does have
  MPS), but a 7B model would still be far too slow to run once per retrieved
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
  and falls back to auto-detection rather than failing the run. The
  resolved device is logged once (e.g. `[FilterRAG] SLM device: mps
  (model=google/flan-t5-small)`), not per-passage/per-query.
- **`filterrag_query_only`** (see above) is an added diagnostic ablation, not
  part of the paper. It exists purely so the pipeline (dispatch, diagnostics
  logging, k-sweep, summarizer) can be smoke-tested and run at zero cost
  before spending any time/compute on the SLM-backed `filterrag` mode -- the
  same "cheap diagnostics before expensive runs" pattern used throughout
  `docs/RAGDEFENDER_DIAGNOSTIC_PLAN.md`.
- **ML-FilterRAG is out of scope** (see "Deferred work").
- **Epsilon is not re-tuned for `flan-t5-small`**: the paper's `epsilon=0.2`
  was presumably tuned against LLaMA-2/3-generated `SLM_answer` text; with a
  much smaller substitute SLM, the "right" threshold for this repo's setup
  may differ. `--filterrag_epsilon` is exposed specifically so this can be
  swept/tuned once live diagnostics are available, rather than trusting the
  paper's value blindly.

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

Note: unlike RAGDefender (which only needs `sentence-transformers` +
`torch`), `--defense filterrag` additionally downloads
`google/flan-t5-small` from the HuggingFace Hub on first use (a few hundred
MB, one-time, cached under `~/.cache/huggingface/hub`) and runs it locally
once per retrieved passage per query, on whichever device
`resolve_slm_device()` selects (Apple Silicon Metal/MPS on this repo's
development machine) -- this makes step 2 noticeably slower than the
equivalent RAGDefender run, especially at higher `k`. `filterrag_query_only`
has no such cost and is exactly as fast as `none`.

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
