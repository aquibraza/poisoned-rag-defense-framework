# RAGDefender Clone Update — Diagnostics Environment Setup

**Date/time:** 2026-07-22 (session spanning 2026-07-21 20:00 – 2026-07-22 01:38 UTC-5/UTC)
**Purpose:** Update the nested `RAGDefender/` clone in a controlled, reproducible way,
and set up an isolated diagnostics environment for future cluster-visualization work
— *without* touching the frozen baseline environment or any existing baseline
results. Follow-up to [`docs/RAGDEFENDER_VERSION_AUDIT.md`](RAGDEFENDER_VERSION_AUDIT.md).

Nothing in this session was committed. No RAGDefender behavior, `defense/defense_runner.py`,
`defense/dispatch.py`, `main.py`, or any existing `results/` artifact was modified.

---

## 1. Two environments — do not confuse them

This repo contains **two** Python environments, discovered and confirmed in this
session:

| | `PoisonedRAG_env/` | `venv/` |
|---|---|---|
| Role | **Frozen baseline environment.** Almost certainly what produced the existing `results/query_results/main/*.json` baselines (full ML stack, dated Oct 2025). | **Diagnostics-only environment** for future RAGDefender cluster-visualization work. Was nearly empty at the start of this session (only `pip`/`setuptools`/`wheel`). |
| Modified this session? | **No — read-only.** Only `pip freeze` and import checks were run. Nothing installed/upgraded/removed. | **Yes.** `pip`, `setuptools`, `wheel` upgraded; `ragdefender` installed editable along with its declared runtime dependencies. |
| Python | `PoisonedRAG_env/bin/python` → 3.9.6 | `venv/bin/python` → 3.9.6 |
| `ragdefender` importable? | No (`ModuleNotFoundError: No module named 'ragdefender'`) — expected, confirms baselines never depended on it. | Yes, from the nested clone (editable install). |
| Frozen dependency list | [`docs/pip_freeze_PoisonedRAG_env_frozen_baseline.txt`](pip_freeze_PoisonedRAG_env_frozen_baseline.txt) (140 packages, recorded read-only) | [`docs/pip_freeze_ragdefender_diag_env.txt`](pip_freeze_ragdefender_diag_env.txt) (36 packages, after the editable install) |

**Existing PoisonedRAG baselines remain fully unaffected by everything in this
document**, for two independent reasons:
1. `defense/defense_runner.py` never imports `ragdefender` (see
   [`RAGDEFENDER_VERSION_AUDIT.md`](RAGDEFENDER_VERSION_AUDIT.md) §1–3) — it is a
   self-contained hand-port of the paper's `artifacts/main.py` logic.
2. `PoisonedRAG_env/` — the environment that almost certainly ran the baselines —
   was not installed into, upgraded, or otherwise modified in this session. It
   was only read from (`pip freeze`, import checks).

### 1a. `PoisonedRAG_env` read-only verification (unmodified)

Commands run (`pip freeze` and plain `import` checks are read-only — neither
installs, upgrades, nor removes anything):

```
PoisonedRAG_env/bin/python -V
# Python 3.9.6

PoisonedRAG_env/bin/python -m pip freeze > docs/pip_freeze_PoisonedRAG_env_frozen_baseline.txt
# 140 packages recorded
```

Import check results:

| Package | Result |
|---|---|
| `torch` | OK — `1.13.0` |
| `transformers` | OK — `4.30.0` |
| `sentence_transformers` | **FAILED** — `ModuleNotFoundError("No module named 'huggingface_hub.snapshot_download'")` (pre-existing environment issue, unrelated to this session; not fixed, per "do not modify `PoisonedRAG_env`") |
| `numpy` | OK — `1.26.4` |
| `pandas` | OK — `2.3.3` |
| `sklearn` (scikit-learn) | OK — `1.6.1` |
| `ragdefender` | **FAILED** — `ModuleNotFoundError("No module named 'ragdefender'")` (expected — confirms baselines never imported it) |

This is consistent with `PoisonedRAG_env` being a real, populated ML environment
(unlike `venv/`, which had almost nothing installed before this session) — supporting
the premise that `PoisonedRAG_env` is the environment that actually executed the
Feb 26–27 baseline runs.

*Note:* the first attempt to `import torch` in `PoisonedRAG_env` under the
sandboxed shell segfaulted (exit 139). This was a sandbox restriction on native
extension init, not a problem with the environment — re-running the exact same
import with unrestricted (`all`) permissions succeeded immediately (`torch 1.13.0`).
No environment files were touched by either attempt.

---

## 2. `RAGDefender/` nested clone: git state (Phases 1–4, this update)

### 2a. Pre-update safety checks (Phase 1)

| Check | Result |
|---|---|
| `git status --short` (top-level) | Pre-existing, unrelated changes only (`.gitignore`, `defense/filterrag.py`, `tests/test_filterrag.py`, moved/deleted PDFs, untracked diagnostics artifacts) — none touched by this update |
| `git -C RAGDefender status --short` | Clean |
| `git -C RAGDefender rev-parse HEAD` (before update) | `a315e3e4c53a01e6a50c1805c4ae1f798730b6fe` — confirmed the expected Feb 26 commit |
| `git -C RAGDefender branch --show-current` (before update) | `main` |
| `git -C RAGDefender tag --points-at HEAD` (before update) | *(none)* |
| `git -C RAGDefender remote -v` | `origin  https://github.com/SecAI-Lab/RAGDefender.git` |
| `git -C RAGDefender rev-parse --is-shallow-repository` (before update) | `true` |

### 2b. Old clone state preserved (Phase 2)

```
git -C RAGDefender branch archive/feb26-a315e3e a315e3e4c53a01e6a50c1805c4ae1f798730b6fe
git -C RAGDefender tag    local-feb26-baseline-a315e3e a315e3e4c53a01e6a50c1805c4ae1f798730b6fe
```

Both created successfully (neither existed previously). The exact Feb 26 tree is
now permanently reachable via either `archive/feb26-a315e3e` (branch) or
`local-feb26-baseline-a315e3e` (tag), independent of what `main`/`origin/main`
does going forward.

### 2c. Unshallow fetch (Phase 3)

```
git -C RAGDefender fetch origin --tags --prune --unshallow
```

Succeeded on the first attempt (no fallback needed).

| Check | Result |
|---|---|
| `git -C RAGDefender rev-parse --is-shallow-repository` (after) | `false` |
| `git -C RAGDefender rev-parse v0.2.0` | `1f32c868ac5d17151917377c178834a569728143` (annotated tag object; peels to commit `2be0bb2f84b2bf4ef4a0f6d8387ad1c0d3df847a`) |
| `git -C RAGDefender rev-parse origin/main` | `ba2a17efba165d45409114df2d70b030ade1e1b8` |

### 2d. Pinned diagnostics branch (Phase 4)

```
git -C RAGDefender checkout -B ragdefender-v0.2.0-for-diagnostics v0.2.0
```

| Check | Result |
|---|---|
| `git -C RAGDefender rev-parse HEAD` | `2be0bb2f84b2bf4ef4a0f6d8387ad1c0d3df847a` (= the `v0.2.0` tag's commit) |
| `git -C RAGDefender tag --points-at HEAD` | `v0.2.0` |
| `git -C RAGDefender status --short` | Clean |
| `git -C RAGDefender branch --show-current` | `ragdefender-v0.2.0-for-diagnostics` |

The clone is **not** left tracking floating `origin/main` — it is pinned to a
named local branch created directly from the `v0.2.0` tag. Future upstream
commits on `origin/main` will not silently move this checkout.

**Consequence for the superproject:** `RAGDefender/` is tracked in the top-level
PoisonedRAG repo as a gitlink (no `.gitmodules`, so it's a loose nested-repo
pointer, not a real submodule). Since the nested clone's `HEAD` moved from
`a315e3e...` to `2be0bb2f...`, `git status --short` at the top level now shows
`M RAGDefender`. This is an intentional, expected side effect of pinning the
diagnostics branch — see §4 for the full top-level status. **Nothing has been
committed**; the superproject's recorded gitlink still points at `a315e3e` until
a commit is made (which requires your explicit approval, per your instructions).

---

## 3. Diagnostics environment (`venv/`) setup (this update)

### 3a. Import check before install

```python
import ragdefender  # -> ModuleNotFoundError: No module named 'ragdefender'
```

Confirmed not importable before install, in a fresh check this session.

### 3b. Editable install with dependency resolution

```
venv/bin/python -m pip install -e ./RAGDefender
```

This was **not** run with `--no-deps` this time (per your instruction — `venv/`
is sanctioned for RAGDefender's runtime dependencies). Resolution was scoped and
not excessive: `torch`, `numpy`, `tqdm`, `scikit-learn` were already present in
`venv/`; pip additionally installed `pandas`, `huggingface-hub`, `tokenizers`,
`transformers`, and `sentence-transformers` (5 new packages, all direct/transitive
requirements from `RAGDefender/pyproject.toml`'s `dependencies = [...]` list — no
unrelated packages pulled in).

*(Earlier in this session, before the two-environment distinction above was
established, `pip`, `setuptools`, and `wheel` were upgraded in `venv/` — required
because v0.2.0 dropped `setup.py` in favor of a `pyproject.toml`/PEP 660 build,
which the venv's original `pip 21.2.4` / `setuptools 58.0.4` could not perform.
This upgrade was done only after being explicitly approved, and only affects
`venv/`, never `PoisonedRAG_env/`.)*

### 3c. Post-install verification

```python
import ragdefender
import inspect
import importlib.metadata as md

ragdefender.__file__      # /Users/araja2/Documents/PoisonedRAG/RAGDefender/ragdefender/__init__.py
md.version("ragdefender") # "0.2.0"

from ragdefender import RAGDefender
inspect.getsourcefile(RAGDefender)  # /Users/araja2/Documents/PoisonedRAG/RAGDefender/ragdefender/defender.py

from ragdefender.grouping.concentration import ConcentrationBasedGrouping   # OK
from ragdefender.identification.topk import IdentifyAdversarial            # OK
from ragdefender.similarity import cos_sim_matrix, top_similar_pairs, n_pairs_for  # OK
```

All imports succeeded. `pip show ragdefender` confirms:

```
Name: ragdefender
Version: 0.2.0
Location: /Users/araja2/Documents/PoisonedRAG/venv/lib/python3.9/site-packages
Editable project location: /Users/araja2/Documents/PoisonedRAG/RAGDefender
Requires: numpy, pandas, scikit-learn, sentence-transformers, torch, tqdm, transformers
```

(full output saved to
[`docs/ragdefender_diag_pip_show.txt`](ragdefender_diag_pip_show.txt)) — and
`direct_url.json` in the install metadata records
`{"dir_info": {"editable": true}, "url": "file:///Users/araja2/Documents/PoisonedRAG/RAGDefender"}`,
confirming the installed package **is** the nested clone, not some unrelated
global/PyPI package.

Full dependency snapshot of `venv/` after this install: 36 packages, saved to
[`docs/pip_freeze_ragdefender_diag_env.txt`](pip_freeze_ragdefender_diag_env.txt).

### 3d. Tests (`RAGDefender/tests`)

`pytest` is **not installed** in `venv/` (`pip show pytest` → not found). Per
your instruction to report missing dependencies rather than blindly installing,
`RAGDefender/tests` was **not run** in this session (installing `pytest` was out
of scope for the steps requested this turn). This is a separate, deliberate gap
from Phases 1–5 above — flag if you'd like `pytest` added to `venv/` next.

---

## 4. Summary table

| Item | Value |
|---|---|
| Old local clone commit (before update) | `a315e3e4c53a01e6a50c1805c4ae1f798730b6fe` |
| Archive branch created | `archive/feb26-a315e3e` → `a315e3e4c53a01e6a50c1805c4ae1f798730b6fe` |
| Archive tag created | `local-feb26-baseline-a315e3e` → `a315e3e4c53a01e6a50c1805c4ae1f798730b6fe` |
| New checked-out branch | `ragdefender-v0.2.0-for-diagnostics` |
| New checked-out tag (pinned to) | `v0.2.0` (annotated tag `1f32c868ac5d17151917377c178834a569728143`) |
| New HEAD commit | `2be0bb2f84b2bf4ef4a0f6d8387ad1c0d3df847a` |
| `origin/main` commit | `ba2a17efba165d45409114df2d70b030ade1e1b8` (= `v0.2.0` + 1 no-op CodeQL-workflow commit) |
| Was clone shallow before update? | Yes (`.git/shallow` present, depth 1) |
| Was it unshallowed? | Yes — `git fetch origin --tags --prune --unshallow` succeeded on first attempt |
| Editable install performed? | Yes — in `venv/` **only**. Not performed, and not needed, in `PoisonedRAG_env/`. |
| `ragdefender` import path | `/Users/araja2/Documents/PoisonedRAG/RAGDefender/ragdefender/__init__.py` (via `venv/`'s editable install → resolves straight to the nested clone) |
| `RAGDefender` class source file | `/Users/araja2/Documents/PoisonedRAG/RAGDefender/ragdefender/defender.py` |
| Stage 1 grouping modules import? | Yes — `ragdefender.grouping.concentration.ConcentrationBasedGrouping` (and `ClusteringBasedGrouping`) import cleanly |
| Stage 2 identification modules import? | Yes — `ragdefender.identification.topk.IdentifyAdversarial` imports cleanly |
| Tests run? | **No** — `pytest` not installed in `venv/` in this session; not run |
| Existing PoisonedRAG baselines affected? | **No.** They depend only on `defense/defense_runner.py` (a static, in-repo, hand-ported implementation that never imports `ragdefender`), and `PoisonedRAG_env` — the environment that produced them — was not modified in any way. |

---

## 5. Recommendation

Any future RAGDefender cluster-visualization / diagnostics script that imports
the `ragdefender` package should:

1. Run under `venv/` (the diagnostics environment), never `PoisonedRAG_env/`.
2. Record the exact RAGDefender commit/tag it ran against in its
   `run_config.json` (or equivalent run-metadata file) — e.g.:
   ```json
   "ragdefender_commit": "2be0bb2f84b2bf4ef4a0f6d8387ad1c0d3df847a",
   "ragdefender_tag": "v0.2.0",
   "ragdefender_branch": "ragdefender-v0.2.0-for-diagnostics",
   "ragdefender_version": "0.2.0"
   ```
   so that any later re-run — including after upstream ships the "Phase 6"
   paper-faithful AND/threshold rewrite flagged in the audit — can tell exactly
   which RAGDefender behavior produced which plot.
3. Continue to treat `defense/defense_runner.py`, `defense/dispatch.py`,
   `main.py`, and everything under `results/` as frozen and untouched by any of
   this — they remain the sole source of truth for existing baselines.
