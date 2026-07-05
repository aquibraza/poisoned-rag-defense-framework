# RAGDefender Diagnostics: Research Plan

> Status: **diagnosis-only**. This document describes tooling that measures
> *why* RAGDefender under-performs/backfires; it does not implement a fixed
> defense. `ragdefender_fixed` is explicitly out of scope until these
> diagnostics have been run and reviewed. See [docs/ANALYSIS.md](ANALYSIS.md)
> §5b and §7.2 for the original hypothesis that motivated this work.

## 1. Research question

`docs/ANALYSIS.md` reports that RAGDefender (`defense/defense_runner.py`)
provides inconsistent protection against the PoisonedRAG LM-Targeted attack
and, on HotpotQA, sometimes *increases* ASR relative to no defense at all.
The existing experiments run at `k=5` (retrieved passages) with `N=5`
(injected adversarial passages) — i.e. the retrieved context is **100%
poisoned** by construction.

**Is RAGDefender's poor performance caused by:**

1. **Threat-model mismatch** — RAGDefender's source paper (Kim et al. 2025)
   explicitly assumes "at least one benign passage is retrieved." At k=N,
   that assumption is violated by construction, so there is no clean
   "anchor" for its clustering/concentration heuristics to compare against.
2. **Implementation mismatch** — a bug or deviation between
   `defense/defense_runner.py` and the algorithm described in Kim et al.
   (2025), independent of how poisoned the context is.
3. **Evaluation mismatch** — the defense is actually detecting/removing
   poison correctly, but something downstream (prompt construction, answer
   matching, residual-poison sensitivity) is what keeps ASR high.

This is not an attempt to make RAGDefender "look good." The goal is to
produce evidence that discriminates between these three explanations.

## 2. Hypotheses

- **H1 (threat-model mismatch):** RAGDefender's detection quality (poison
  recall, residual poison fraction) improves substantially once `k > N`
  (i.e. once clean passages are actually present in the retrieved set),
  even though the same `defense_runner.apply_defense` code runs unchanged.
- **H2 (implementation mismatch):** RAGDefender's detection quality stays
  poor even at `k > N`, suggesting a bug/deviation independent of context
  composition.
- **H3 (residual-poison sensitivity):** RAGDefender removes most poisoned
  passages (high poison recall) but ASR remains high, because even one
  surviving adversarial passage is often sufficient to control generation.
- **H4 (false-positive clean-evidence removal):** RAGDefender frequently
  removes more clean passages than poisoned ones, degrading answer quality
  independent of poison removal (this is the mechanism suspected for the
  HotpotQA ASR *increase* in `docs/ANALYSIS.md` §4.2).
- **H5 (evaluation/downstream mismatch):** Even *oracle* ground-truth
  removal of all poisoned passages fails to drop ASR meaningfully, meaning
  the problem is not detection at all, but prompt construction, answer
  matching, or the quality of the remaining clean evidence.

## 3. Experimental matrix

| Dimension | Values |
|---|---|
| Datasets | NQ, HotpotQA, MS MARCO |
| Retrieved context size `k` | 5, 10, 20, 50 (full sweep); 5, 10 (quick preset) |
| Injected adversarial passages `N` | 5 (fixed) |
| Defenses | `none`, `ragdefender_original`, `oracle_remove_all_poison`, `random_remove_same_count` |
| Attack | `LM_targeted` (matches existing stored adversarial corpus) |
| Max queries per run | 100 (full sweep, default); 10 (quick preset) |

Retrieval reuse: `results/beir_results/{dataset}-contriever.json` already
caches up to ~2000 candidates per query, so sweeping `k` up to 50 requires
**no new retrieval computation**. The pre-generated adversarial corpus in
`results/adv_targeted_results/{nq,hotpotqa,msmarco}.json` covers exactly 100
queries with 5 adversarial texts each, so `max_queries <= 100` requires
**no new `gen_adv.py` / GPT-4 calls**.

`oracle_remove_all_poison` and `random_remove_same_count` are **diagnostic
controls, not deployable defenses** (see `defense/controls.py`):
- Oracle removal uses ground-truth attack labels no real defense has access
  to at inference time. It is an upper bound: if it doesn't drop ASR, the
  problem isn't detection (see H5).
- Random removal deletes the same number of passages as RAGDefender's own
  estimate, with zero signal about which are poisoned. It is a chance
  baseline: if RAGDefender doesn't beat it, that itself is informative.

## 4. Metrics (see `defense/diagnostics.py` for the full JSONL schema)

**Detection-quality metrics (always computable, including under
`--dry_run`, i.e. with zero LLM API cost):**

- `N_retrieved_poison` / `N_retrieved_clean` — composition of the retrieved
  context before any defense runs.
- `N_adv_estimated_by_ragdefender` — RAGDefender's own estimate of how many
  passages are adversarial (captured independently of removal, so it's
  accurate even when an internal fallback causes zero removal).
- `removed_poison` / `removed_clean` — what was actually removed.
- `poison_recall` = removed_poison / N_retrieved_poison.
- `clean_false_positive_rate` = removed_clean / N_retrieved_clean.
- `residual_poison_count` / `residual_clean_count` / `residual_poison_fraction`
  — what's left in the context that actually reaches the LLM prompt.

**Generation-dependent metrics (only populated when `--dry_run False`,
i.e. real LLM calls were made):**

- `answer_no_defense` / `answer_with_defense`, `asr_no_defense` /
  `asr_with_defense`, `latency_generation_sec`.

## 5. Interpretation decision tree

Applied programmatically (with the actual observed numbers) by
`scripts/summarize_ragdefender_diagnostics.py`, in this exact order:

1. If RAGDefender's residual-poison/ASR **improves** at k=10/20 relative to
   k=5 → conclude **threat-model mismatch / retrieval-saturation failure**
   (supports H1).
2. If it **still fails** at k=10/20 too → flag for **implementation-mismatch
   investigation** (supports H2).
3. If RAGDefender **removes poison** (high `poison_recall`) but **ASR stays
   high** → conclude **residual-poison sensitivity** (supports H3; even one
   leftover adversarial passage can control generation).
4. If it **removes clean passages often** (high
   `clean_false_positive_rate`) → conclude **false-positive clean-evidence
   removal** (supports H4).
5. If **oracle removal itself fails** to drop ASR → conclude the issue lies
   in **prompt construction, answer matching, or clean-evidence quality**,
   not detection at all (supports H5).

When ASR is unavailable (i.e. all runs were `--dry_run`), the tree falls
back to `residual_poison_fraction` as a detection-quality proxy for steps
1–2 and explicitly notes that generation-dependent steps (3, 5) need a
`--dry_run False` re-run to confirm.

## 6. Expected outputs

- `results/diagnostics/ragdefender/*.jsonl` — one JSONL file per `main.py`
  run, one record per query (see `defense/diagnostics.py`).
- `results/diagnostics/ragdefender_summary.csv` — full aggregated CSV,
  grouped by `(dataset, model, defense, k, N_injected)`.
- `results/diagnostics/RAGDEFENDER_DIAGNOSTIC_REPORT.md` — human-readable
  report: detection-quality table first (ASR last), the interpretation
  decision tree with actual values, worst-10 queries by residual poison
  fraction, cases where more clean than poisoned passages were removed, and
  an oracle/random/RAGDefender comparison table.

## 7. How to run the quick diagnostics-only preset

The fastest way to test the core `k=N` saturation hypothesis (HotpotQA
only, 10 queries, k in `[5, 10]`, 4 defenses, **zero LLM API cost** by
default):

```bash
# Print the 8 commands this preset would run (no execution):
python scripts/run_ragdefender_k_sweep.py --quick_hotpotqa

# Actually run them. main.py still uses --dry_run True by default here,
# so this makes NO LLM API calls -- only retrieval + defense + detection
# diagnostics run and are logged.
python scripts/run_ragdefender_k_sweep.py --quick_hotpotqa --execute

# Only add this if you explicitly want to spend real, billed GPT-4 calls
# (2 k-values x 4 defenses x 10 queries; non-'none' defenses issue 2 LLM
# calls each):
python scripts/run_ragdefender_k_sweep.py --quick_hotpotqa --execute --live_generation
```

## 8. How to run the full k-sweep

```bash
# Print the full matrix of commands (3 datasets x 4 k-values x 2 defenses):
python scripts/run_ragdefender_k_sweep.py \
    --datasets nq hotpotqa msmarco --k_values 5 10 20 50 --N 5 \
    --max_queries 100 --defenses none ragdefender_original

# Execute with zero LLM API cost (main.py --dry_run True):
python scripts/run_ragdefender_k_sweep.py --execute --max_queries 10

# Include the diagnostic controls too:
python scripts/run_ragdefender_k_sweep.py --execute \
    --defenses none ragdefender_original oracle_remove_all_poison random_remove_same_count
```

Every invocation also writes a normal `results/query_results/<dir>/<name>.json`
file (via `main.py`'s existing, unmodified result format), in addition to
the JSONL diagnostics.

## 9. How to read the diagnostic report

```bash
python scripts/summarize_ragdefender_diagnostics.py
```

Reads every `*.jsonl` file under `results/diagnostics/ragdefender/` (or a
custom `--diagnostics_dir`) and writes the CSV + Markdown report described
in §6. Open `results/diagnostics/RAGDEFENDER_DIAGNOSTIC_REPORT.md` and start
with:

1. **Section 1 (detection-quality table)** — for a quick visual read: does
   `mean_residual_poison_fraction` drop as `k` increases?
2. **Section 3 (decision tree)** — the programmatic conclusion(s) drawn
   from your specific run.
3. **Sections 4–6** — the worst individual queries, the "removed more clean
   than poison" failure mode, and the oracle/random comparison.

## 10. What would support each conclusion

| Conclusion | Evidence to look for |
|---|---|
| Threat-model mismatch | `mean_residual_poison_fraction` (or ASR) improves clearly from k=5 to k=10/20 for `ragdefender_original`, with `mean_N_retrieved_clean > 0` at higher k. |
| Implementation mismatch | Poor detection persists at k=10/20 despite clean passages being present in the retrieved context. |
| Residual-poison sensitivity | High `mean_poison_recall` (e.g. ≥ 0.7) but ASR stays high (e.g. ≥ 0.3). |
| False-positive clean-evidence removal | High `mean_clean_false_positive_rate` (e.g. ≥ 0.3), especially co-occurring with an ASR *increase* relative to no defense (the HotpotQA backfire pattern in `docs/ANALYSIS.md` §4.2). |
| Evaluation/downstream mismatch | `oracle_remove_all_poison` (perfect ground-truth removal) still leaves ASR high. |
| Genuine RAGDefender limitation (no easy fix) | Detection stays poor even at large k with clean passages present, RAGDefender does no better than `random_remove_same_count`, *and* oracle removal does successfully drop ASR (ruling out H5). |

Only once these are reviewed should `ragdefender_fixed` be scoped, and only
to address the specific failure mode(s) the diagnostics actually identify —
not a generic "improve RAGDefender" pass.
