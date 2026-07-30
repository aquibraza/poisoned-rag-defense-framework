# FilterRAG Paper-Fidelity Audit

> Scope: audits `defense/filterrag.py` (threshold-based FilterRAG only) against
> Algorithm 1 / Section III-B1 of Edemacu et al. 2025/2026, *"Defending
> Against Knowledge Poisoning Attacks During Retrieval-Augmented
> Generation"* (arXiv:2508.02835). This is a **pre-implementation audit for
> ML-FilterRAG** -- it does not implement ML-FilterRAG (Section III-B2 /
> Algorithm 2), does not rerun experiments, and does not touch RAGDefender or
> Cluster-Normalized Poisoning code. See `docs/FILTERRAG_BASELINE.md` for the
> existing implementation writeup this audit supersedes in places.

## 1. Verdict

**Before this task, `defense/filterrag.py` was *not* paper-faithful for one
load-bearing piece of the algorithm: word matching in the Freq-Density
formula.** It implemented exact-token set membership instead of the paper's
semantic (cosine-similarity) word matching, which the paper's own ablation
(Table II) shows is the *worst-performing* configuration of the exact
mechanism the paper studies (equivalent to similarity threshold = 1.0: "high
ATR/ASR, low accuracy"). Every other structural piece of Algorithm 1 --
per-passage SLM answer generation, the `(query ⊕ SLM_answer)` keyword
construction, the `Freq-Density` numerator/denominator shape, and the
`>= epsilon` drop rule -- was already implemented correctly. The SLM model
itself (`flan-t5-small` vs. the paper's LLaMA-2/3) was already correctly
*documented* as a known, deliberate proxy, not a hidden fidelity gap.

This task adds a configurable **semantic matching mode**
(`all-MiniLM-L6-v2`, cosine threshold 0.6, matching the paper's Section
IV-B2 default) so paper-fidelity runs are possible, while leaving the
original exact-token behavior available and unchanged as a documented
legacy/fast mode. See §4 for what changed.

## 2. Paper's Algorithm 1 (FilterRAG), restated

```
Input: query qi, poisoned DB D ∪ D̃, top-s, epsilon, SLM, LLM, Retriever
1. RetrievedItems ← Retriever(qi, D ∪ D̃, top-s)
2. for each item dj in RetrievedItems:
     aj ← SLM(qi, dj)                         # SLM answer, conditioned on dj alone
     Freq-Density[dj] ← Compute(qi ⊕ aj, dj)  # Eq. 4, below
3. for each dj in RetrievedItems:
     if Freq-Density[dj] < epsilon: keep dj
     else: discard dj
4. From the survivors, take the top-k for LLM prompting.
```

Eq. 4 (Freq-Density):

```
Freq-Density(dj) = [ sum_{w in (qi⊕aj) ∩ dj} Freq(w, dj) ] / UniqueWords(dj)
```

where `(qi⊕aj) ∩ dj` is explicitly defined in the paper's prose (Section
III-B1, immediately under Eq. 4) as **"semantically similar words common to
`(qi⊕aj)` and `dj`, i.e. word pairs in `(qi⊕aj)` and `dj` whose computed
similarities exceed a predetermined threshold"** -- not literal/exact
string equality. Section IV-B2 fixes the mechanics of that similarity:

> "For semantic word similarity matching, we employ a huggingface sentence
> transformer\* and set a default similarity threshold value of 0.6 for
> cosine similarity."
>
> \*footnoted as `sentence-transformers/all-MiniLM-L6-v2`.

The Table II ablation (varying this threshold from 1.0 down to 0.4 on
MS-MARCO/GPT-4) is explicit about what threshold=1.0 means and why it's
bad: *"at the similarity threshold of 1.0, FilterRAG and ML-FilterRAG both
have high ATR and ASR, and low accuracies ... only query-answer combination
words that **perfectly match** the context text words are considered. This
level of perfection can allow attackers to evade our proposed methods
simply using modified word versions or synonyms."* Performance improves
monotonically from threshold 1.0 down to 0.6 (the paper's chosen default),
then degrades again below 0.6.

`epsilon = 0.2` is the paper's stated default for the threshold-based
FilterRAG (Section IV-B2, and used throughout Table I/IV).

## 3. Item-by-item comparison

| Algorithm 1 piece | Paper | This repo (before this task) | Faithful? |
|---|---|---|---|
| SLM answer per passage, `aj = SLM(qi, dj)` | LLaMA-2 or LLaMA-3, conditioned on `dj` alone | `local_hf_slm_answer_fn()`, prompts `google/flan-t5-small` with `dj` alone (`defense/filterrag.py:283-291`) | **Structurally yes** (correct per-passage-conditioning, correct prompt shape); **model choice no** (see §3.1) -- already documented as a deliberate, disclosed proxy, not hidden |
| Keyword set `(qi ⊕ aj)` | Concatenation of query tokens and SLM-answer tokens | `score_passages()`: `keywords = tokenize(query) + tokenize(slm_answer)` (`defense/filterrag.py:100`) | **Yes** |
| Freq-Density numerator: `sum_{w in (qi⊕aj)∩dj} Freq(w,dj)` | `∩` = semantically similar word pairs, cosine similarity ≥ threshold (default 0.6, all-MiniLM-L6-v2) | `∩` implemented as **exact lowercase string-set membership** (`freq_density()`, old `defense/filterrag.py:65-80`: `total_freq = sum(doc_counts[w] for w in keyword_set if w in doc_counts)`) | **No** -- this is the one load-bearing deviation; see §3.2 |
| Freq-Density denominator: `UniqueWords(dj)` | Count of unique words in `dj` | `len(set(doc_tokens))` | **Yes** |
| Filter rule: keep if `Freq-Density < epsilon`, else discard | `epsilon` default 0.2 | `filterrag_defense()`: `removed_doc_ids = {... if score >= epsilon}` (drop when `>= epsilon`, keep when `< epsilon`) with `DEFAULT_EPSILON = 0.2` | **Yes** -- logically equivalent to the paper's `Filter(dj)=1 if <epsilon else 0` |
| Retrieve top-s, filter, then take top-k of survivors | Two-stage: candidate pool `top-s` > final context `top-k` | Not modeled in `defense/filterrag.py` -- the surrounding harness (`main.py`) retrieves exactly `top_k` items and hands all of them to every defense (there is no `--top_s`/superset retrieval anywhere in `main.py`) | **No, but out of repo-harness scope for this task** -- see §3.3 |

### 3.1 SLM model choice (flan-t5-small vs. LLaMA-2/3)

Already correctly disclosed as a deviation, not a silent one. The module
docstring, `docs/FILTERRAG_BASELINE.md` §3, and the CLI help text for
`--filterrag_slm_model` all state this plainly, and the SLM is fully
pluggable (`slm_answer_fn` parameter / `--filterrag_slm_model`). **No change
needed here per this task's scope** (task 5 explicitly says: audit +
semantic matching only). Recommendation for future work: if a larger local
model becomes runnable at this repo's scale, swap `DEFAULT_SLM_MODEL`
without touching `freq_density()`/`filterrag_defense()` at all -- this is
already how the code is structured.

### 3.2 Exact lexical matching -- the actual deviation

The pre-existing `freq_density()` computed the intersection `(qi⊕aj) ∩ dj`
as `{w in keyword_set if w in doc_counts}` -- i.e., a keyword counts as
"matched" only if it appears **verbatim** (case-folded) in the passage. Per
§2 above, this is equivalent to the paper's own similarity-threshold
ablation at **threshold = 1.0**, which the paper explicitly identifies as
the *worst* setting they tested (Table II: ATR 0.415, ASR 0.260, accuracy
0.621 at threshold 1.0, vs. ATR 0.065, ASR 0.090, accuracy 0.840 at the
paper's chosen default of 0.6, MS-MARCO/GPT-4). Concretely, exact matching:

- **Cannot** catch an SLM answer or query that paraphrases passage content
  with synonyms/morphological variants (e.g. query word "disease", passage
  word "illness"; or "cause" vs. "causes" vs. "caused" -- no stemming
  either).
- Is exactly the mechanism the paper's Table II shows attackers can evade
  "simply using modified word versions or synonyms."
- Was not a documented deviation in `docs/FILTERRAG_BASELINE.md` prior to
  this audit -- it was an implicit simplification baked into `freq_density()`
  with no fidelity-tradeoff note, unlike the SLM-model deviation.

This is the deviation this task fixes (see §4).

### 3.3 top-s vs. top-k (not fixed in this task)

The paper's Algorithm 1 retrieves a *superset* `top-s` candidate pool,
filters it, then keeps the `top-k` survivors for the LLM. This repo's
`main.py` retrieves exactly `top_k` passages once (`--top_k`, no `--top_s`)
and hands that same list to every `--defense` mode uniformly (see
`main.py:286-287,323-324`, `defense/dispatch.py`). This means FilterRAG here
always operates on a context that's already been truncated to the final
`k`, not an oversized candidate pool it can freely shrink from -- so
"discarding" a passage here permanently reduces context size rather than
being backfilled from `top-s`'s extra candidates. This is a property of the
shared evaluation harness (affects every defense identically, not
FilterRAG-specific), and changing it would require restructuring
`main.py`'s retrieval call for *all* defenses, not just FilterRAG -- out of
scope for "the minimal missing pieces needed before ML-FilterRAG" in this
task. Flagged here for visibility; not changed.

## 4. What this task changes

Added a configurable matching mode to `defense/filterrag.py`, wired through
`defense/dispatch.py` and `main.py`:

- **`matching_mode="exact"`** (default, unchanged): byte-for-byte the
  original behavior (verbatim case-folded string matching). Kept as the
  default specifically for backward compatibility with existing diagnostics
  runs/scripts (`scripts/filterrag_score_inspection.py`,
  `scripts/run_ragdefender_k_sweep.py`) and because it has zero extra
  dependencies/runtime cost. **This mode is documented as the
  non-paper-faithful legacy mode** (equivalent to similarity threshold 1.0;
  see §3.2) -- callers who want the paper's default behavior must pass
  `matching_mode="semantic"` (`--filterrag_matching_mode semantic` at the
  CLI).
- **`matching_mode="semantic"`** (new, paper-faithful default candidate):
  word-pair matching by cosine similarity of
  `sentence-transformers/all-MiniLM-L6-v2` embeddings
  (`DEFAULT_SEMANTIC_MODEL`), threshold 0.6 (`DEFAULT_SEMANTIC_THRESHOLD`,
  `--filterrag_semantic_threshold`), matching Section IV-B2 exactly. For
  each unique word in the passage, its embedding is compared against every
  keyword's embedding; the passage word counts toward the Freq-Density
  numerator (contributing its raw frequency) if its best cosine similarity
  to any keyword is `>= threshold`. This generalizes exact matching (same
  word ⇒ cosine similarity 1.0 ⇒ always matches at any threshold `<= 1.0`).
  `sentence_transformers` is imported lazily, only the first time semantic
  mode is actually used (see `SemanticWordMatcher`/`get_semantic_word_matcher()`),
  matching the existing lazy-import convention for `transformers`/`torch` in
  this file.
- Default remains `exact` at every layer (function defaults,
  `defense/dispatch.py`, `--filterrag_matching_mode` CLI default) rather
  than silently switching the shipped default to `semantic`, since flipping
  a default changes the behavior of every existing script/experiment that
  calls `--defense filterrag`/`filterrag_query_only` without this new flag.
  Paper-fidelity runs must opt in with `--filterrag_matching_mode semantic`.
- Per-passage diagnostics (`score_passages()`) now also report
  `matching_mode`, `semantic_threshold` (`None` when `matching_mode="exact"`),
  `unique_word_count`, `matched_keyword_count`, and a length-capped
  `matched_keywords_sample` -- in addition to the pre-existing `doc_id`,
  `freq_density_score`, `slm_answer` (unchanged keys, so existing consumers
  of `score_passages()`/`filterrag_scores` are unaffected).

Not changed in this task (explicitly out of scope per the task's own
instructions): ML-FilterRAG, RAGDefender, Cluster-Normalized Poisoning,
top-s/top-k harness restructuring (§3.3), the SLM model default.

## 5. Recommendation for ML-FilterRAG (next task, not implemented here)

ML-FilterRAG (Algorithm 2) reuses this exact Freq-Density feature plus three
more: perplexity of `dj`, joint log-probability of the SLM's output `aj`,
and "sum of frequencies of semantically similar words between `(qi⊕aj)` and
`dj`" (Section III-B2) -- i.e. it needs the *same* semantic-matching
machinery this task adds, not a separate implementation. The
`SemanticWordMatcher` class and `freq_density_detailed()` added here are
intentionally reusable: `matched_keyword_count`/`matched_keywords` from
`freq_density_detailed()` already expose "sum of frequencies of
semantically similar words" as a byproduct, so ML-FilterRAG's feature
extractor should call into this module rather than reimplement word
matching. Perplexity and log-probability are new features not present in
`defense/filterrag.py` and still need their own extractor plus a trained
XGBoost/Random Forest classifier (per-dataset, as in the paper's Appendix C)
-- that remains the next task's actual scope of work.
