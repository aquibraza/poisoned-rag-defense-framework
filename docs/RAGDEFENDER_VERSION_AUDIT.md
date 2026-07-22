# RAGDefender Version Audit

**Date:** 2026-07-21
**Scope:** Read-only audit. No RAGDefender behavior, code, or baseline results were
modified while producing this report. No plots were implemented. Nothing was
committed.

**Trigger:** The local `RAGDefender/` clone was made on Feb 26 (this year). The
upstream `SecAI-Lab/RAGDefender` GitHub `main` branch now contains a large
"paper-aligned restructure" released as **v0.2.0** on **2026-05-12** (~2 months
before this audit). Before implementing any RAGDefender cluster visualization
that imports the `ragdefender` package, we need to know exactly which code
version the existing baseline results depend on.

---

## 1. Headline answer

**The existing PoisonedRAG baseline results (`results/query_results/main/*ragdefender*.json`,
dated Feb 26–27) do not depend on the `ragdefender` PyPI/GitHub package at all,
in either its old or new form.** `defense/defense_runner.py::apply_defense` is a
self-contained, hand-ported copy of the detection algorithm from
`RAGDefender/artifacts/main.py` (the paper's own reproduction script). It never
imports `ragdefender`. The package is not installed in the venv used for these
runs (`pip show ragdefender` → not found), and no file in the PoisonedRAG repo
(outside the `RAGDefender/` nested clone itself) contains `import ragdefender`
or `from ragdefender`.

Consequently:
- The upstream v0.1.1 → v0.2.0 breaking change to `RAGDefender.defend()`
  (see §3) **cannot have affected the existing baselines**, because those
  baselines never called `defend()` from the package in the first place.
- The local clone being 2 months stale is **not a reproducibility risk for
  existing results**, but it **is** a relevant decision point for any new code
  that imports `ragdefender` directly (e.g. a cluster-visualization script
  that wants `ragdefender.grouping.ConcentrationBasedGrouping` or
  `ragdefender.identification.IdentifyAdversarial` internals) — see §5.

---

## 2. Local clone vs. `origin/main`

Commands run in `/Users/araja2/Documents/PoisonedRAG/RAGDefender`.

| Item | Value |
|---|---|
| `git status --short` | *(empty — clean)* |
| `git rev-parse HEAD` | `a315e3e4c53a01e6a50c1805c4ae1f798730b6fe` |
| `git log --oneline -5` | `a315e3e Add CodeQL analysis workflow configuration` *(only 1 commit reachable — see below)* |
| `git branch --show-current` | `main` |
| `git tag --points-at HEAD` | *(none)* |
| `git remote -v` | `origin  https://github.com/SecAI-Lab/RAGDefender.git` (fetch/push) |
| Clone type | **Shallow, depth 1** (`.git/shallow` present; `git rev-parse --is-shallow-repository` → `true`) |
| `git rev-parse origin/main` (after `git fetch origin --tags`) | `ba2a17efba165d45409114df2d70b030ade1e1b8` |
| New tags fetched | `Release` → `bcc3f4f13cc39fdcb5ffd4402b950e4dcd707f1d`; `v0.2.0` → `1f32c868ac5d17151917377c178834a569728143` |

**Relationship between HEAD and origin/main:**
`git log --oneline --left-right --cherry-pick HEAD...origin/main` shows **20
commits, all on the right (`>`, i.e. only in `origin/main`) and none on the
left**. This means the local `HEAD` (`a315e3e`) is a strict ancestor of
`origin/main` — the local clone is not diverged or ahead in any way, it is
simply **behind**. The 20 upstream-only commits are the full "Phase 0–5"
`refactor/v0.2.0` branch plus its merge commit:

```
ba2a17e Merge refactor/v0.2.0 into main — paper-aligned restructure (released as v0.2.0 on PyPI)
2be0bb2 Phase 5 cleanup: address advisor findings before tag
eed5f42 Phase 5: documentation rewrite for v0.2.0
660c5a2 Phase 4: rename 'blind' → 'tan' (atomic) — paper-aligned naming
07fe058 Phase 2 follow-up: revert default embedder to v0.1.1 checkpoint
ed2645f Phase 2: package restructure to v0.2.0 (paper-aligned API)
85906e8 Phase 1: janitorial cleanup (no behavior change)
4387520 Phase 0: capture v0.1.1 legacy fixtures for byte-equivalence regression tests
bcc3f4f pip release update
b2ce6b9 Minor update (Logging)
b536aa5 Model quantization configuration update
0124415 Ablation study update
05f5573 Retriever update
bb60e92 Colab compatibility update
ca4d260 Readme update
9c17eb6 Readme update
a6bad98 Colab link, token changes
a1e0270 Shell files edit for compatibility with colab environment
1eead18 Initial commit: Add RAGDefender project files
e5ce19d Initial commit
```

`git diff --stat HEAD..origin/main` → **371 files changed, 3361
insertions(+), 1625 deletions(-)**. `git rev-parse Release v0.2.0
origin/main` confirms `origin/main` (`ba2a17e`) = the `v0.2.0` tag
(`1f32c868`) + one follow-up commit that only adds a CodeQL workflow file
(`.github/workflows/codeql.yml`, no behavior change) — i.e. **`origin/main`
today is, functionally, the v0.2.0 release**.

### Scoped diff: `ragdefender/`, `docs/`, `artifacts/`, `tests/`

`git diff --name-status HEAD..origin/main -- ragdefender docs artifacts tests`
(full output captured during the audit; summarized here — mechanical
`artifacts/blind/* → artifacts/tan/*` and `artifacts/blind/blind_*.json →
artifacts/tan/tan_*.json` renames, ~340 files, are omitted from this table):

| File | Change |
|---|---|
| `artifacts/eval.py`, `artifacts/main_abl.py`, `artifacts/run_garag.py` | Modified |
| `artifacts/run_blind.py` → `artifacts/run_tan.py` | Renamed (`blind` → `tan`, paper-aligned naming for the "TAN" baseline) |
| `artifacts/blind/**` → `artifacts/tan/**` | ~330 files renamed 1:1, no content change |
| `ragdefender/__init__.py` | Modified (new public API surface) |
| `ragdefender/_logging.py` | **Added** |
| `ragdefender/attacks/__init__.py`, `ragdefender/datasets/__init__.py`, `ragdefender/defenses/__init__.py`, `ragdefender/models/__init__.py` | **Deleted** (empty placeholder subpackages from v0.1.1) |
| `ragdefender/cli.py` | Modified (459-line rewrite) |
| `ragdefender/core/__init__.py`, `ragdefender/core/defender.py`, `ragdefender/core/evaluator.py` | Modified — gutted to thin back-compat shims (`core/defender.py` −226 lines, `core/evaluator.py` −247 lines) |
| `ragdefender/defender.py` | **Added** (new top-level `RAGDefender` class, replaces `core/defender.py`) |
| `ragdefender/evaluator.py` | **Added** (replaces `core/evaluator.py`) |
| `ragdefender/embedders.py` | **Added** (`load_embedder`, embedder presets) |
| `ragdefender/similarity.py` | **Added** (`cos_sim_matrix`, `top_similar_pairs`, `n_pairs_for`) |
| `ragdefender/grouping/__init__.py`, `ragdefender/grouping/base.py`, `ragdefender/grouping/clustering.py`, `ragdefender/grouping/concentration.py` | **Added** (Stage 1, paper §4.1, split out of `core/defender.py`) |
| `ragdefender/identification/__init__.py`, `ragdefender/identification/topk.py` | **Added** (Stage 2, paper §4.2 — new in v0.2.0) |
| `docs/algorithm.md`, `docs/migration-0.1-to-0.2.md`, `docs/reproducing-paper.md`, `docs/RELEASING.md` | **Added** |
| `tests/*` (9 new files, incl. `tests/fixtures/legacy_v011_*.json`) | **Added** (55 tests, incl. byte-equivalence regression vs. captured v0.1.1 outputs) |

---

## 3. What is actually imported by the PoisonedRAG environment

Environment: `/Users/araja2/Documents/PoisonedRAG/venv` (Python 3.9.6), the
venv present in the repo and the one `defense/defense_runner.py` runs under.

| Question | Answer |
|---|---|
| `python -c "import ragdefender"` | **`ModuleNotFoundError: No module named 'ragdefender'`** |
| `pip show ragdefender` | `WARNING: Package(s) not found: ragdefender` |
| Editable install (`pip list -e`) | *(empty — nothing installed editable)* |
| Any `import ragdefender` / `from ragdefender` in PoisonedRAG source (outside `RAGDefender/` itself) | **None found** (`rg`/`grep` across all `*.py`) |
| Source file for the `RAGDefender` class that PoisonedRAG's results actually depend on | **`defense/defense_runner.py`** (`apply_defense()`), a hand port of **`RAGDefender/artifacts/main.py`** (functions `find_num_adv`, `find_num_adv_tfidf`, `find_num_adv_agg`, and the pair-scoring block in `main()`) — *not* `ragdefender/core/defender.py` or `ragdefender/defender.py` |
| Package version actually used | N/A — no `ragdefender` package is loaded at runtime |

`defense/dispatch.py` documents this explicitly:

> `ragdefender` / `ragdefender_original` delegate to `defense_runner.apply_defense`
> completely unmodified — the original algorithm's behavior is preserved exactly.

"The original algorithm" here means the paper's `artifacts/main.py` script, not
the installable package.

---

## 4. Behavioral comparison: old package vs. current `origin/main` vs. what the baseline actually runs

| Question | Local clone (`ragdefender.core.defender`, pre-refactor / "v0.1.1") | `origin/main` today (v0.2.0) | PoisonedRAG baseline (`defense/defense_runner.py`) |
|---|---|---|---|
| Does `defend()` run Stage 2 identification? | **No.** `defend()` estimates `N_adv` then truncates `retrieved_docs[:len-N_adv]`, assuming poisoned docs are already at the end of the list. | **Yes.** `RAGDefender.defend()` now calls `IdentifyAdversarial.select()` (pair-frequency Top-K, paper §4.2, Eq. 4–7) to pick *which* indices to drop. CHANGELOG calls this "a correctness fix... F1 rises from 0.67→1.00 / 0.50→1.00" on captured fixtures, and flags it as a no-opt-out breaking change. | **Yes.** `apply_defense()` always performs the pair-frequency scoring step (`_top_similar_pairs` + `Counter` + `math.copysign(sim*sim, sim)`), matching `artifacts/main.py`'s `main()` loop, which is algorithmically the same computation `IdentifyAdversarial.select(..., p=2)` does. |
| Does multi-hop / HotpotQA use concentration-based grouping? | **Yes** — `_find_num_adversarial()` (mean/median cosine-similarity concentration heuristic). | **Yes** — moved verbatim into `ragdefender.grouping.concentration.ConcentrationBasedGrouping.estimate_n_adv()`. | **Yes** — `_find_num_adversarial()` in `defense_runner.py`, byte-identical logic. |
| OR vs AND in the concentration test? | **OR** (`above_avg[i] == 1 or above_median[i] == 1`) — the paper's text specifies AND. | **Still OR.** `grouping/concentration.py`'s docstring explicitly documents this as a known paper/implementation mismatch, preserved on purpose ("We deliberately do NOT silently rewrite this to match the paper text in v0.2.0... A paper-faithful implementation lands as part of Phase 6"). | **OR**, same expression, unchanged. |
| Combined median threshold `(avg_median + avg_avg) / 2`? | **Yes**, present. | **Yes**, present, unchanged (same docstring flags this as also non-paper-faithful, deferred to Phase 6). | **Yes**, present, unchanged. |
| Result-flipping branch (`sum(final) if sum(final)>0 and avg_avg<avg_median else len(R)-sum(final)`)? | **Yes**, present. | **Yes**, present, unchanged (third documented non-paper-faithful behavior, deferred to Phase 6). | **Yes**, present, unchanged. |
| Default embedder (package constructor default) | `sentence-transformers/all-MiniLM-L6-v2` (`similarity_model` default). | `DEFAULT_EMBEDDER = "minilm-all"` = `sentence-transformers/all-MiniLM-L6-v2` — **unchanged from v0.1.1 on purpose** (`embedders.py` docstring, and commit `07fe058 Phase 2 follow-up: revert default embedder to v0.1.1 checkpoint` — it was briefly changed mid-refactor and then reverted). | N/A — package embedder default is never used. |
| Embedder the baseline actually used | N/A (package not used). | N/A (package not used). | **`paraphrase-MiniLM-L6-v2`** — `DefenseConfig.similarity_model` default in `defense_runner.py`, matching `artifacts/main.py:142` (`SentenceTransformer('paraphrase-MiniLM-L6-v2', ...)`), which is a **different model** from either package version's default (`all-MiniLM-L6-v2`). |

**Net effect on the existing numbers:** the Stage‑1 estimator
(concentration/clustering `N_adv`) and Stage‑2 identification (pair-frequency
ranking) that the baseline runs actually execute are **numerically identical**
across `artifacts/main.py`, the old local-clone package, and `origin/main`'s
new `grouping`/`identification` modules — the only thing that changed upstream
is the *orchestration* inside the package's own `defend()` method, which
PoisonedRAG never calls. The embedder checkpoint (`paraphrase-MiniLM-L6-v2`)
the baseline uses has also been stable across both package versions (neither
version's default ever matched it; that's expected, since it mirrors the
*paper's script*, not the *packaged library's default*).

---

## 5. Did the baselines likely use "old" or "current" behavior?

**Neither, cleanly — they used a frozen, self-contained port that happens to
already match the "current" (Stage‑2‑complete) behavior for the parts that
matter, and was never exposed to either package release.**

- Timestamps: `results/query_results/main/*ragdefender*.json` are dated
  **Feb 26–27**, i.e. from this local clone's commit (`a315e3e`, pre-refactor)
  and ~2.5 months before v0.2.0 shipped (2026-05-12).
- But since `defense_runner.py` never imports `ragdefender`, upstream's
  Feb 26 → today changes (old or new) are irrelevant to reproducing these
  specific numbers — the numbers depend only on `defense_runner.py`'s own
  (unchanged, in-repo) code, `git status` shows it as currently modified in
  the working tree (see §7), so any future comparison must diff against the
  exact committed version used for the Feb runs, not against RAGDefender at
  all.
- If a *future* script instead imports the real `ragdefender` package (e.g.
  `from ragdefender import RAGDefender` for cluster visualization), the
  old‑vs‑new choice matters a lot, because the public API changed
  (`mode=` → `task_type=` (now required, no default), `similarity_model=` →
  `embedder=`, `ragdefender.core.defender` → `ragdefender.defender`, and
  `defend()`'s return semantics for *which* docs get removed differ between
  versions).

---

## 6. Recommendation

1. **Do not rerun the existing RAGDefender baselines.** They do not depend on
   the `ragdefender` package version (old or new) and are unaffected by the
   upstream v0.2.0 restructure. The thing to "freeze" for reproducibility is
   `defense/defense_runner.py` itself (already a static, version-controlled
   file in this repo), not the RAGDefender clone.
2. **For the planned cluster-visualization work, use `origin/main` /
   v0.2.0, not the stale Feb 26 clone**, for any code that imports the
   `ragdefender` package directly:
   - v0.2.0 exposes the Stage‑1 grouping (`ragdefender.grouping.
     ConcentrationBasedGrouping`, `ClusteringBasedGrouping`) and Stage‑2
     identification (`ragdefender.identification.IdentifyAdversarial`) as
     clean, independently importable classes with embeddings as a return
     value/shareable input — exactly the intermediate objects a cluster plot
     needs (pairwise similarities, per-passage concentration scores,
     frequency scores).
   - The old local clone's `core/defender.py` is a monolith that does not
     expose these intermediates as reusably, and is missing Stage 2 entirely
     in its public `defend()`.
   - The Stage‑1 math itself (concentration/clustering estimators) is
     unchanged between old and new, so switching does not change any
     estimated adversarial counts — it only changes *how the code is
     organized/imported*.
   - Pin the exact commit/tag (`v0.2.0`, `1f32c868ac5d17151917377c178834a569728143`,
     or the `origin/main` tip `ba2a17efba165d45409114df2d70b030ade1e1b8`) when
     updating the clone, so the visualization work has a reproducible
     RAGDefender version recorded, independent of whatever lands upstream next
     (a "Phase 6" paper-faithful AND/threshold rewrite is explicitly flagged
     as planned-but-not-yet-shipped).
3. **Do not change `defense/defense_runner.py`, `defense/dispatch.py`, or any
   existing `results/` artifact as part of this or the cluster-visualization
   work.** They are independent of the RAGDefender package and must stay
   frozen to keep the Feb 26–27 baseline numbers reproducible.
4. Updating the local `RAGDefender/` clone (`git pull` / re-clone to
   `origin/main`) is **safe and recommended** before starting the
   visualization work — it cannot silently change any existing PoisonedRAG
   result, because nothing in this repo currently reads from that clone at
   runtime.

---

## 7. Final `git status` (both repos, at end of this audit)

### `RAGDefender/` (nested clone)

```
$ git status --short
(clean — no output)
```

Only local ref-tracking metadata changed (`git fetch origin --tags` updated
`origin/main` and added the `Release`/`v0.2.0` tags); the working tree and
`HEAD` are untouched.

### `PoisonedRAG/` (top-level repo, unchanged by this audit except adding this report)

```
 M .gitignore
 M defense/filterrag.py
 D "docs/Edemacu et al. - 2025 - Defending Against Knowledge Poisoning Attacks During Retrieval-Augmented Generation.pdf"
 D "docs/Kim et al. - 2025 - Rescuing the Unpoisoned Efficient Defense against Knowledge Corruption Attacks on RAG Systems.pdf"
 D "docs/Zou and Geng - PoisonedRAG Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models.pdf"
 M tests/test_filterrag.py
?? docs/RAGDEFENDER_VERSION_AUDIT.md   <- this report (new, untracked)
?? docs/ara-proposal-template-2025.pdf
?? docs/research/
?? results/diagnostics/... (pre-existing untracked diagnostic artifacts, unrelated to this audit)
?? results/query_results/... (pre-existing untracked diagnostic artifacts, unrelated to this audit)
```

All modifications/deletions shown above (`.gitignore`, `defense/filterrag.py`,
`tests/test_filterrag.py`, the moved PDFs, the diagnostic result files) predate
this audit and are unrelated to RAGDefender version tracking — they were
present in `git status` before any command in this session was run. This
audit added exactly one new file: `docs/RAGDEFENDER_VERSION_AUDIT.md`.
Nothing was committed.

---

## Appendix: Clone update performed

Following this audit's recommendation (§6), the nested `RAGDefender/` clone was
subsequently updated in a controlled way, and a separate diagnostics environment
was set up. Full details, exact commands, and verification output are in
[`docs/RAGDEFENDER_CLONE_UPDATE.md`](RAGDEFENDER_CLONE_UPDATE.md). Summary:

- The old Feb 26 state (`a315e3e4c53a01e6a50c1805c4ae1f798730b6fe`) was preserved
  via `archive/feb26-a315e3e` (branch) and `local-feb26-baseline-a315e3e` (tag)
  before any further changes.
- The clone was unshallowed and pinned to a new local branch,
  `ragdefender-v0.2.0-for-diagnostics`, checked out at the `v0.2.0` tag
  (commit `2be0bb2f84b2bf4ef4a0f6d8387ad1c0d3df847a`) — not left tracking
  floating `origin/main`.
- `ragdefender` v0.2.0 was installed editable **only** in the diagnostics
  environment (`venv/`), never in `PoisonedRAG_env/` (the environment that
  actually produced the existing baselines, confirmed read-only in the update
  doc). `PoisonedRAG_env/` was not installed into, upgraded, or modified.
- This confirms and does not change the conclusion above: existing PoisonedRAG
  baselines remain unaffected, since they depend on `defense/defense_runner.py`
  (not the `ragdefender` package) and on `PoisonedRAG_env` (left untouched).
