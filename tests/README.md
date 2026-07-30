# Tests

Lightweight `unittest`-based tests for the RAGDefender diagnostics work
(`defense/passages.py`, `defense/diagnostics.py`, `defense/controls.py`,
`defense/dispatch.py`, `defense/asr_match.py`, `defense/filterrag.py`,
`scripts/summarize_ragdefender_diagnostics.py`, and `main.py`'s FilterRAG
CLI flags).
`pytest` is intentionally not used since it is not already a project
dependency; everything here runs with the Python standard library.

## Running all tests

From the repo root:

```bash
python -m unittest discover -s tests -v
```

Or run a single file:

```bash
python -m unittest tests.test_passages -v
```

## Dependency requirements per test file

Most test files (`test_passages.py`, `test_controls.py`,
`test_diagnostics_schema.py`, `test_asr_match.py`, `test_filterrag.py`,
`test_summarizer.py`, `test_existing_results_compat.py`) have **no
third-party dependencies** and run with any Python 3 interpreter, including
the system `python3`. `test_filterrag.py` in particular never imports
`transformers`/`sentence_transformers`/loads any real HF model --
`slm_answer_fn` is always a plain mock function (exactly like
`test_dispatch_smoke.py` mocks RAGDefender's encoder), and
`matching_mode="semantic"` is exercised via `FakeSemanticMatcher` (a
dependency-free test double) for matching-logic tests, or a fake
`sentence_transformers` module injected via `sys.modules` (mirroring the
existing fake-`transformers`/fake-`torch` pattern) for the handful of tests
that specifically verify the lazy-import/caching behavior of
`SemanticWordMatcher`/`get_semantic_word_matcher()`.

`test_dispatch_smoke.py` imports `defense.dispatch`, which imports
`defense.defense_runner`, which imports `torch` and (lazily)
`sentence-transformers`. If your environment doesn't have these installed
(e.g. a minimal system Python), that one file will fail to import. It does
**not** require network access or a real embedding-model download -- it
monkeypatches `defense_runner._get_s_model` with a deterministic fake
encoder, and (for its FilterRAG semantic-matching-mode tests)
`defense.filterrag.get_semantic_word_matcher` with a `FakeSemanticMatcher`
-- but it does need `torch` + `scikit-learn` + `sentence-transformers`
importable. If your project virtualenv (e.g. `PoisonedRAG_env`) can't load
`torch` due to macOS Gatekeeper quarantine, see `fix_venv_gatekeeper.sh` at
the repo root, or create a fresh venv and
`pip install torch scikit-learn sentence-transformers`. Its
`filterrag_query_only` cases need no such setup (no SLM, no
sentence-transformers) and would pass even without `defense_runner`'s heavy
deps, but the file as a whole still requires them to import at all.

`test_main_cli_filterrag.py` imports `main.py` directly (to call
`main.parse_args()`), which transitively imports `torch`, `transformers`,
`sentence-transformers`, and `beir` (via `src.utils`/`src.models`) at module
level -- same dependency footprint as `test_dispatch_smoke.py`. It does
**not** run `main()`'s actual retrieval/generation pipeline (no BEIR
dataset access, no dataset files on disk required) -- it only exercises
`argparse` plus a source-level string check that `main.py`'s
`run_defense(...)` call site forwards the new CLI args through.

None of the tests here make any LLM/API call (no GPT-4, no PaLM, no
Vicuna) or require GPU access.

## What's covered

- `test_passages.py` -- poison label propagation, filtering, and diffing
  stay faithful to ground-truth attack labels.
- `test_diagnostics_schema.py` -- the diagnostic JSONL schema is fully
  populated (detection fields) or explicitly `None` (generation fields)
  depending on whether generation ran; JSONL round-trips correctly; the
  strict/legacy ASR fields are correctly derived and can diverge (e.g.
  target "no" vs. a response containing "does not").
- `test_asr_match.py` -- `legacy_match` reproduces the repo's original
  substring-match ASR behavior byte-for-byte (including its "no" vs.
  "not"/"none"/"another"/"known" false positives); `strict_match`
  (strict token-boundary ASR: a standalone yes/no token or exact
  token-subsequence match) is verified against all of the required
  no/yes/texas cases, the denylist words, and its documented
  not-a-semantic-evaluator limitation (won't recognize "They are not in
  the same place" as "no" without a standalone "no" token).
- `test_controls.py` -- `oracle_remove_all_poison` removes exactly the
  poisoned passages and nothing else; `random_remove_same_count` removes
  exactly the requested count, deterministically per seed.
- `test_dispatch_smoke.py` -- `none`/`ragdefender_original`/
  `oracle_remove_all_poison`/`random_remove_same_count`/
  `filterrag_query_only` all run end-to-end through
  `defense.dispatch.run_defense()` fully offline; plus
  `filterrag_matching_mode`/`filterrag_semantic_threshold` are correctly
  forwarded through `run_defense()` for both `filterrag_query_only` and
  (mocked-SLM) `filterrag`, `exact` remains the dispatch-level default, and
  a synonym-stuffed passage is only caught in `semantic` mode (not `exact`).
- `test_filterrag.py` -- `freq_density()` scores keyword-stuffed passages
  higher than unrelated ones (case-insensitive, no double-counting of
  duplicate keywords); `score_passages()`/`filterrag_defense()` behave
  correctly both in the query-only ablation mode (`slm_answer_fn=None`) and
  with a mocked SLM answer function; threshold (`epsilon`) behavior at its
  extremes (0.0 removes everything, a very high value removes nothing); kept
  passages preserve their original metadata; `resolve_slm_device()`
  correctly prefers MPS > CUDA > CPU under `auto`, honors an explicit
  device when available, and falls back (with a logged warning) when an
  explicit `mps`/`cuda` isn't actually available -- exercised with a fake
  `torch` module injected via `sys.modules` so this needs no real torch
  install; `_get_local_hf_slm_pipeline()` smoke-tests a freshly-loaded
  non-CPU pipeline with one throwaway generation and falls back to CPU if it
  fails (reproducing, with a fake `transformers` module, the real
  `google/flan-t5-small` + torch==1.13 + MPS bug found during FilterRAG
  epsilon calibration -- see `docs/FILTERRAG_BASELINE.md` §3.1 -- where
  every SLM call failed and was silently swallowed, making `filterrag` and
  `filterrag_query_only` produce identical scores); a per-passage SLM
  failure degrades to "no answer" for just that passage but logs a warning
  at least once rather than failing completely silently. Also covers the
  `matching_mode="semantic"` option added in
  `docs/FILTERRAG_FIDELITY_AUDIT.md`: `exact` mode is unchanged/default and
  structurally cannot match a synonym (`"vehicle"` never matches
  `"car"`-only text); `semantic` mode (via `FakeSemanticMatcher`, no real
  embedding model) *does* match such synonym pairs, and the similarity
  threshold gates which pairs count; per-passage diagnostics report
  `matching_mode`/`semantic_threshold`/`unique_word_count`/
  `matched_keyword_count`/`matched_keywords_sample` (capped, so a
  heavily-matched passage doesn't blow up output size, while
  `matched_keyword_count` itself stays uncapped); `filterrag_query_only`
  still skips the SLM step and `filterrag` still calls it regardless of
  matching mode; invalid `matching_mode` values raise in
  `freq_density`/`score_passages`/`filterrag_defense`; and
  `sentence_transformers` is proven to be imported only when
  `matching_mode="semantic"` is actually exercised (never for `exact`), with
  both the loaded model and per-word embeddings cached across calls.
- `test_main_cli_filterrag.py` -- `main.py`'s `--filterrag_matching_mode`
  (default `exact`, choices `exact`/`semantic`, rejects invalid values) and
  `--filterrag_semantic_threshold` (default 0.6, matching the paper) parse
  correctly, and are forwarded into the `run_defense(...)` call site.
- `test_summarizer.py` -- aggregation, CSV/Markdown report rendering
  (including the interpretation decision tree) against fake JSONL fixtures.
- `test_existing_results_compat.py` -- pre-existing files under
  `results/query_results/main/` still parse and compute ASR correctly via
  `eval_asr.py`, proving this branch didn't silently change that schema.
