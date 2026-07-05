# Tests

Lightweight `unittest`-based tests for the RAGDefender diagnostics work
(`defense/passages.py`, `defense/diagnostics.py`, `defense/controls.py`,
`defense/dispatch.py`, `defense/asr_match.py`,
`scripts/summarize_ragdefender_diagnostics.py`).
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
`test_diagnostics_schema.py`, `test_asr_match.py`, `test_summarizer.py`,
`test_existing_results_compat.py`) have **no third-party dependencies** and
run with any Python 3 interpreter, including the system `python3`.

`test_dispatch_smoke.py` imports `defense.dispatch`, which imports
`defense.defense_runner`, which imports `torch` and (lazily)
`sentence-transformers`. If your environment doesn't have these installed
(e.g. a minimal system Python), that one file will fail to import. It does
**not** require network access or a real embedding-model download -- it
monkeypatches `defense_runner._get_s_model` with a deterministic fake
encoder -- but it does need `torch` + `scikit-learn` + `sentence-transformers`
importable. If your project virtualenv (e.g. `PoisonedRAG_env`) can't load
`torch` due to macOS Gatekeeper quarantine, see `fix_venv_gatekeeper.sh` at
the repo root, or create a fresh venv and
`pip install torch scikit-learn sentence-transformers`.

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
  `oracle_remove_all_poison`/`random_remove_same_count` all run end-to-end
  through `defense.dispatch.run_defense()` fully offline.
- `test_summarizer.py` -- aggregation, CSV/Markdown report rendering
  (including the interpretation decision tree) against fake JSONL fixtures.
- `test_existing_results_compat.py` -- pre-existing files under
  `results/query_results/main/` still parse and compute ASR correctly via
  `eval_asr.py`, proving this branch didn't silently change that schema.
