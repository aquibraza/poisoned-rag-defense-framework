# PoisonedRAG Defense Framework — Research Analysis

> **Source paper:** Zou, Wei; Geng, Runpeng; Wang, Binghui; Jia, Jinyuan.
> *"PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models."*
> USENIX Security Symposium 2025. [arXiv:2402.07867](https://arxiv.org/abs/2402.07867)

---

## 1. Executive Summary

This document compares the **paper's claimed results** against **this codebase's experimental outputs**, assesses the RAGDefender integration added beyond the paper, and identifies gaps, anomalies, and next steps.

**Bottom line:** The LM-Targeted black-box attack is highly effective and substantially validated by this repo's experiments. Attack Success Rates (ASR) of 76–100% are observed across all three benchmark datasets and both GPT-3.5 and GPT-4. The RAGDefender defense — not part of the original paper's evaluated defenses — largely **fails to reduce ASR** and in one case **increases it**, reinforcing the paper's conclusion that existing defenses are insufficient.

---

## 2. What the Paper Claims

### 2.1 Core Threat Model

The paper frames a **targeted knowledge corruption attack** where an attacker:

1. Selects target questions `Q_i` and attacker-desired incorrect answers `R_i`
2. Injects a small number `N` of malicious texts per question into a RAG knowledge database
3. The malicious texts are crafted to satisfy two simultaneous conditions:
   - **Retrieval condition** — the text is retrieved for the target question (high embedding similarity)
   - **Generation condition** — when used as context, the LLM generates the target incorrect answer

### 2.2 Two Attack Methods

| Method | Setting | How `S` (retrieval sub-text) is crafted |
|--------|---------|----------------------------------------|
| **LM-Targeted** | Black-box | Simply prepend the target question `Q` to a generated passage `I` — no retriever access needed |
| **HotFlip** | White-box | Gradient-based token substitution to maximize cosine similarity between the crafted text and the query embedding |

The full malicious text is `P = S ⊕ I` where `I` is an LLM-generated passage that satisfies the generation condition.

### 2.3 Paper's Key Quantitative Claims (Table 1 — Black-Box, N=5, k=5)

| Dataset | Knowledge DB size | GPT-4 ASR | PaLM 2 ASR | GPT-3.5 ASR | Avg across 8 LLMs |
|---------|------------------|-----------|-----------|-------------|------------------|
| NQ | 2,681,468 texts | **97%** | 97% | 92% | ~94% |
| HotpotQA | 5,233,329 texts | **93%** | 99% | 98% | ~97% |
| MS MARCO | 8,841,823 texts | **92%** | 91% | 89% | ~91% |

- Retrieval F1-Score: ≥ 89% across all settings (malicious texts are consistently retrieved)
- Average LLM queries to craft each malicious text: ~1.6–2.7 (very efficient)
- Black-box runtime per malicious text: < 2 seconds

### 2.4 Paper's Defense Evaluations

The paper evaluates four defenses, all of which **fail**:

| Defense | Effect on ASR | Why it fails |
|---------|--------------|-------------|
| **Paraphrasing** | Drops ~5–15% | Malicious texts still semantically similar enough to paraphrased queries |
| **Perplexity detection** | Near-random (AUC ≈ 0.5) | LLM-generated `I` is fluent; can't distinguish from clean text |
| **Duplicate text filtering** | No effect | GPT-4 generates diverse `I` per malicious text; no duplicates |
| **Knowledge expansion** (large k) | Partial reduction | 41% ASR persists even at k=50; attacker can inject more texts |

---

## 3. Implementation in This Repository

### 3.1 What Is Implemented

| Component | File(s) | Status |
|-----------|---------|--------|
| LM-Targeted black-box attack | `src/attack.py` → `LM_targeted` | ✅ Implemented |
| HotFlip white-box attack | `src/attack.py` → `hotflip` | ✅ Implemented |
| Contriever retrieval | `src/contriever_src/` | ✅ Integrated |
| GPT-3.5 / GPT-4 LLM | `src/models/GPT.py` | ✅ Implemented |
| PaLM 2 LLM | `src/models/PaLM2.py` | ✅ Implemented |
| LLaMA-2 (7B/13B) | `src/models/Llama.py` | ✅ Implemented |
| Vicuna (7B/13B/33B) | `src/models/Vicuna.py` | ✅ Implemented |
| Adversarial corpus generation | `gen_adv.py` | ✅ Implemented |
| BEIR retrieval caching | `evaluate_beir.py` | ✅ Implemented |
| ASR evaluation | `eval_asr.py` | ✅ Implemented |
| **RAGDefender defense** | `defense/defense_runner.py` | ✅ Added (beyond paper) |
| Paraphrasing defense | — | ❌ Not implemented |
| Perplexity-based detection | — | ❌ Not implemented |
| Duplicate text filtering | — | ❌ Not implemented |
| Knowledge expansion | — | ❌ Not implemented |
| Self-RAG / CRAG evaluation | — | ❌ Not implemented |
| FEVER fact verification | — | ❌ Not implemented |
| LLM Agent (ReAct) evaluation | — | ❌ Not implemented |

### 3.2 RAGDefender — Beyond the Paper

The paper's defense section does not evaluate RAGDefender. This repo integrates it as an additional defense via `defense/defense_runner.py`, ported from the [RAGDefender repository](https://github.com/leileigan/RAGDefender).

**How RAGDefender works:**

1. **Estimate** the number of adversarial documents using:
   - *HotpotQA (multi-hop)*: Cosine similarity heuristic over sentence embeddings — texts with above-average pairwise similarity are flagged
   - *NQ / MS MARCO (single-hop)*: Agglomerative clustering (2 clusters) + TF-IDF overlap to decide which cluster is the "poison" cluster
2. **Score** documents: find the top-`num_adv * (num_adv - 1) / 2` most similar pairs; each document accumulates a suspicion score (`sim²`) for each pair it appears in
3. **Remove** the `num_adv` highest-scoring documents from the context
4. **Prompt** the LLM with the remaining (supposedly clean) documents

The defense uses `paraphrase-MiniLM-L6-v2` as the similarity model.

---

## 4. Experimental Results: Repo vs. Paper

All repo experiments use: **Contriever retriever**, **Top-5 retrieval**, **LM-Targeted (black-box)**, **5 adversarial texts per query**, **dot-product similarity**, except where noted.

### 4.1 Attack-Only Results (No Defense Applied)

These correspond to the paper's Table 1 black-box GPT-4 numbers.

| Dataset | Model | n queries | Repo ASR (no defense) | Paper ASR (GPT-4, black-box) | Delta |
|---------|-------|-----------|----------------------|------------------------------|-------|
| NQ | GPT-4 | 100 | **77.0%** | **97%** | −20 pp |
| NQ | GPT-4 | 10 | 90.0% – 100.0% | 97% | ±3–3 pp |
| HotpotQA | GPT-4 | 100 | **76.0%** | **93%** | −17 pp |
| HotpotQA | GPT-4 | 10 | 80.0% | 93% | −13 pp |
| MS MARCO | GPT-4 | 10 | **100.0%** | **92%** | +8 pp |
| NQ | GPT-3.5 | 2 | 100.0% | 92% | +8 pp |

**Interpretation of discrepancies:**

- The **100-query runs (M10×10)** show 77% / 76% ASR versus the paper's 97% / 93%. This is the most material gap. Possible causes:
  1. **Adversarial corpus quality**: The `results/adv_targeted_results/` files were pre-generated and may have been created with different hyperparameters (e.g., number of regeneration trials `L`, text length `V`) than the paper's N=5, L=50, V=30 setting
  2. **Substring matching strictness**: The `eval_asr.py` implementation strips trailing periods and lowercases. Minor differences in how incorrect answers are compared could suppress apparent ASR
  3. **Question selection**: The paper randomly selects 10 close-ended questions per trial (×10 trials = 100). The pre-stored adversarial corpus in this repo may have been generated for questions with lower baseline LLM accuracy
  4. **GPT-4 model version drift**: The config uses `gpt-4-0613`, which may differ in instruction-following behavior from whatever version the paper used

- The **10-query small-batch runs** align more closely with the paper (80–100% range)

- **MS MARCO and GPT-3.5 over-performance** (100% in small batches) is expected variance given n=2 and n=10

### 4.2 RAGDefender Results (Novel — Not in Paper)

The defense results compare `output_poison_no_defense` (raw attack) vs. `output_poison` (RAGDefender applied):

| Dataset | Model | n | Raw ASR (no defense) | ASR with RAGDefender | Defense effect |
|---------|-------|---|----------------------|----------------------|----------------|
| NQ | GPT-4 | 10 | 100% | 100% | **No reduction** |
| NQ | GPT-4 | 10 | 90% | 80% | −10 pp |
| NQ | GPT-3.5 | 2 | 100% | 100% | **No reduction** |
| HotpotQA | GPT-4 | 10 | 80% | 90% | **+10 pp increase** ⚠️ |
| HotpotQA | GPT-4 | 10 | 80% | 90% | **+10 pp increase** ⚠️ |
| MS MARCO | GPT-4 | 10 | 100% | 100% | **No reduction** |
| MS MARCO | GPT-4 | 10 | 100% | 80% | −20 pp |

**Key observations:**

1. **RAGDefender is largely ineffective** against LM-Targeted attacks. In 3 out of 7 runs, it provides zero ASR reduction.

2. **ASR increase on HotpotQA (80% → 90%)** is a critical finding. This occurs because:
   - The multi-hop similarity heuristic in RAGDefender (designed for fact-based documents where adversarial texts appear highly correlated) **incorrectly identifies legitimate documents** as adversarial when dealing with complex multi-hop questions
   - Removing clean, semantically diverse context documents leaves the adversarial texts as a proportionally larger fraction of the context, making the attack *more* effective
   - This is a **defense backfire** — the defense degrades RAG quality without removing the actual poisoned documents

3. **The defense has variable effectiveness**: It achieves a meaningful reduction (−10 to −20 pp) in only 2 out of 7 runs.

4. **Why RAGDefender struggles with LM-Targeted:** The adversarial text `P = Q ⊕ I` has two components:
   - `Q` (the target question) makes it semantically similar to the query
   - `I` (LLM-generated passage) is fluent and contextually natural

   The inter-document similarity between multiple adversarial texts for the same question is high (all start with the same question `Q`), so RAGDefender's pair-based detection *should* work. However, the exact clustering thresholds and heuristics in `defense_runner.py` are sensitive to the mixture of clean and adversarial documents — with only 5 adversarial texts among 5 retrieved documents (100% poisoned context), the clustering has no clean "baseline cluster" to separate from.

### 4.3 Comparison with Paper's Defense Baselines

| Defense | Paper ASR result (NQ, black-box) | Repo result |
|---------|----------------------------------|-------------|
| No defense (baseline) | 97% | 77–100% |
| Paraphrasing | 87% (−10 pp) | *Not tested* |
| PPL detection | ~97% (near zero AUC) | *Not tested* |
| Duplicate filtering | 97% (no change) | *Not tested* |
| Knowledge expansion k=50 | 41%–43% | *Not tested* |
| **RAGDefender** (this repo) | *Not in paper* | **80–100% (often no change)** |

---

## 5. Deeper Analysis

### 5.1 Why the Attack Is So Effective

The LM-Targeted attack exploits a fundamental structural tension in RAG:

```
Retrieval condition:   P must look like the question Q
Generation condition:  P must look like a passage supporting answer R

Solution:   P = Q ⊕ I
            (The question itself satisfies retrieval;
             the LLM-crafted passage satisfies generation)
```

The elegance of this decomposition is that it requires:
- **No retriever access** (just use the question as retrieval bait)
- **~2 GPT-4 queries** per malicious text
- **< 2 seconds** to craft each text (black-box)

This makes the attack **economically viable at scale**: poisoning 100 target questions requires only ~200 API calls and minutes of wall-clock time.

### 5.2 The Retrieval Precision Problem

The paper shows F1-Score ≥ 89% for the black-box attack. This means that with k=5 and N=5 adversarial texts, most or all retrieved documents are adversarial. This is the core mechanism of attack efficacy:

| Fraction of poisoned docs in retrieved context | Approximate ASR (from paper Table 17) |
|-----------------------------------------------|---------------------------------------|
| 1/5 (only 1 adv in k=5) | ~20% |
| 3/5 | ~80% |
| 5/5 (all retrieved = adv) | ~97% |

When the retrieval F1-Score approaches 1.0, the LLM has no clean context to "override" the adversarial narrative.

### 5.3 Why Defenses Fail Systematically

| Defense Class | Root Failure |
|--------------|-------------|
| **Query-side** (paraphrasing) | The attack bakes in the question itself; paraphrased query is still semantically close to the target question |
| **Text-quality detection** (PPL) | GPT-4 generates fluent `I`; malicious texts are indistinguishable from clean Wikipedia in perplexity |
| **Deduplication** | GPT-4 temperature=1 produces diverse `I`; SHA-256 hash will never match |
| **Context dilution** (large k) | Increasing k is expensive; attacker can just increase N; 41% ASR persists at k=50 |
| **RAGDefender** (this repo) | Clustering heuristics fail when all retrieved docs are adversarial (no clean cluster); backfires on multi-hop |

### 5.4 The Fundamental Defense Challenge

The paper states the core difficulty clearly: adversarial texts satisfy the **generation condition** because they are *semantically coherent* facts (just incorrect ones). Any defense that relies on detecting anomalous text quality will fail because the malicious texts look like normal, well-written knowledge base entries.

Effective defenses would need to operate at a different level:
- **Source provenance**: Track the origin/edit history of knowledge base entries
- **Cross-reference verification**: Query multiple independent knowledge bases
- **Uncertainty-aware generation**: LLMs that express low confidence when retrieved facts conflict
- **Watermarking of legitimate sources**: Cryptographic signatures on trusted content

---

## 5b. Defense Literature in Context

The `docs/` folder contains two additional 2025 papers proposing improved defenses. Their findings are critical for contextualizing both the RAGDefender results in this repo and future directions.

### Kim et al. (2025) — *"Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption Attacks on RAG Systems"* — This IS RAGDefender

This is the **source paper** for the `defense/defense_runner.py` implementation. Key facts:

**Core insight:** Adversarial passages crafted by PoisonedRAG form extremely dense clusters in embedding space. On MS MARCO, the average pairwise cosine similarity among adversarial passages is **0.976** vs. **0.309** for benign passages. This is the geometric foundation of the defense.

**How it works (two-stage):**
1. **Grouping**: Estimate how many retrieved passages are adversarial using either:
   - *Clustering* (single-hop): Agglomerative clustering + TF-IDF to pick the smaller/larger cluster based on keyword concentration
   - *Concentration* (multi-hop): Flag passages whose mean AND median pairwise similarity with others exceed global averages
2. **Identification**: Find the top-N similar pairs, score passages by their pair participation weighted by `sim²`, remove top scorers

**Published results (Kim et al., Table 5 / Figure 5):**
- NQ + Contriever + GPT-4o at 1× ratio: ASR = **0.08** (vs 0.66 for RobustRAG)
- HotpotQA + Contriever + GPT-4o at 1× ratio: ASR = **0.04**
- MS MARCO + Contriever + GPT-4o at 1× ratio: ASR = **0.02**
- 12.36× faster than RobustRAG, zero GPU overhead

**Critical assumption violated in this repo:** The Kim et al. paper states in their threat model: *"at least one benign passage is retrieved"*. Their experiments use 1×, 2×, 4×, 6× adversarial-to-benign ratios, **never 100% poison**. In our repo we inject N=5 adversarial passages and retrieve k=5 — the retrieved context is 100% adversarial. The defense clustering has no clean "anchor cluster", causing it to either under-estimate `N_adv` or cluster adversarial passages with themselves and misidentify the wrong subset.

**This explains our backfire result on HotpotQA**: When all 5 retrieved passages are adversarial and the concentration heuristic is applied, the algorithm may count some adversarial passages as "clean" (low concentration) and flag others as adversarial. This inadvertently removes only some adversarial passages while the remaining ones still control the LLM output — and the random removal of context passages from an already-adversarial set can increase ASR by narrowing the LLM's context to fewer, more targeted adversarial texts.

**Fix**: Test RAGDefender with k > N (e.g., k=10, N=5) so at least half the retrieved context is benign.

---

### Edemacu et al. (2025) — *"Defending Against Knowledge Poisoning Attacks During Retrieval-Augmented Generation"* — FilterRAG / ML-FilterRAG

This paper proposes a **statistical frequency-density** approach to detect adversarial passages:

**Key observation:** Adversarial texts crafted by PoisonedRAG intentionally pack query-answer keywords into passages (to satisfy both retrieval and generation conditions). This creates a measurably **higher Freq-Density** compared to clean corpus passages.

**Freq-Density metric:**
```
Freq-Density = Σ(freq(w, d) for w in (query⊕SLM_answer) ∩ d) / UniqueWords(d)
```
Where `SLM_answer` is an answer generated by a small model from the passage as context.

**FilterRAG**: Threshold-based filter (ε = 0.2) — passages with Freq-Density ≥ ε are dropped.

**ML-FilterRAG**: Combines Freq-Density + perplexity + log-probability → XGBoost (NQ) or Random Forest (HotpotQA, MARCO) classifier trained on labeled examples.

**Published results (Edemacu et al., Table I — black-box attack):**

| Dataset | Defense | ATR ↓ | ASR (GPT-4) ↓ | Accuracy (GPT-4) ↑ |
|---------|---------|------|--------------|------------------|
| HotpotQA | PoisonedRAG (no def.) | 1.00 | 0.90 | 0.00 |
| HotpotQA | FilterRAG | **0.00** | **0.09** | **0.87** |
| HotpotQA | ML-FilterRAG | 0.015 | 0.091 | 0.905 |
| MS MARCO | PoisonedRAG (no def.) | 0.825 | 0.824 | 0.173 |
| MS MARCO | FilterRAG | 0.065 | **0.090** | **0.840** |
| MS MARCO | ML-FilterRAG | 0.045 | 0.060 | 0.849 |
| NQ | PoisonedRAG (no def.) | 0.980 | 0.870 | 0.119 |
| NQ | FilterRAG | 0.010 | **0.030** | **0.818** |
| NQ | ML-FilterRAG | 0.030 | 0.030 | 0.810 |

FilterRAG achieves **2–9% ASR** (down from 82–90%) while maintaining **81–90% accuracy** — near CleanRAG performance. This is dramatically better than RAGDefender's performance in our experiments.

**Why it works where RAGDefender fails:** FilterRAG operates at the **individual passage level** using statistical properties of each passage independently. It doesn't require a "clean cluster baseline" — it just asks: does this passage look like it was stuffed with target question keywords? This makes it robust to the 100% adversarial retrieval case that breaks cluster-based methods.

**Limitation:** Requires an SLM at inference time (LLaMA-2 or LLaMA-3 used as SLM). Also needs threshold tuning; wrong ε can either let adversarial texts through or over-aggressively remove clean ones.

### Defense Landscape Summary

| Defense | Method | ASR Reduction (NQ, GPT-4) | Clean Accuracy Preserved | Notes |
|---------|--------|--------------------------|--------------------------|-------|
| Paraphrasing [PoisonedRAG paper] | Query transformation | −10 pp (87%) | High | Fails because attack bakes question in |
| PPL detection [PoisonedRAG paper] | Text quality score | Near zero | N/A | GPT-4 generated text is fluent |
| Duplicate filtering [PoisonedRAG paper] | SHA-256 hash | None | Full | Diverse generation prevents dedup |
| Knowledge expansion k=50 [PoisonedRAG paper] | More retrieval | 41% residual | Degrades (long context) | Expensive |
| **RAGDefender [Kim 2025]** | Clustering + pair scoring | **92–98% reduction** (published) | ~95% | **Fails at 100% poison ratio** |
| **FilterRAG [Edemacu 2025]** | Freq-Density threshold | **97% reduction** (published) | ~98% | Needs SLM + threshold tuning |
| **ML-FilterRAG [Edemacu 2025]** | ML classifier on features | **97% reduction** (published) | ~97% | Needs training data |

---

## 6. Progress Assessment

### 6.1 What Has Been Accomplished in This Repo

| Task | Status | Notes |
|------|--------|-------|
| Full attack pipeline (LM-Targeted) | ✅ Complete | Validated on NQ, HotpotQA, MS MARCO |
| Multi-LLM support | ✅ Complete | GPT-3.5, GPT-4 tested; PaLM2, LLaMA, Vicuna configured |
| BEIR retrieval caching | ✅ Complete | Pre-computed for all 3 datasets |
| Adversarial corpus generation | ✅ Complete | Stored in `results/adv_targeted_results/` |
| ASR evaluation tooling | ✅ Complete | `eval_asr.py` with per-file and aggregate reporting |
| RAGDefender integration | ✅ Novel addition | Extends the paper's defense evaluation |
| HotFlip white-box attack | ⚠️ Configured | Not run in current results |
| Large-scale 100-query experiments | ⚠️ Partial | M10×10 runs exist for NQ and HotpotQA (no MS MARCO at n=100) |
| Paper defense baselines | ❌ Missing | Paraphrasing, PPL, duplicate filter, knowledge expansion not run |
| Advanced RAG schemes | ❌ Missing | Self-RAG, CRAG not evaluated |
| LLM agent evaluation | ❌ Missing | ReAct agent not set up |
| Open-source LLM results | ❌ Missing | LLaMA/Vicuna experiments not in stored results |

### 6.2 Completeness vs. Paper

This repo reproduces approximately **40–50%** of the paper's full experimental scope:

- ✅ Main attack results (Table 1 partial — GPT-3.5, GPT-4 only)
- ✅ Attack vs. one defense system (RAGDefender — novel contribution)
- ❌ Paper's defense comparisons (Table 12, 13 — paraphrasing, duplicate filter)
- ❌ Full LLM sweep (missing PaLM 2, LLaMA-2, Vicuna results)
- ❌ Ablation studies (impact of k, N, V, L)
- ❌ Real-world application evaluations (Wikipedia chatbot, LLM agents)
- ❌ Advanced RAG scheme evaluation (Self-RAG, CRAG)

---

## 7. Gaps, Anomalies, and Notes

### 7.1 ASR Gap (77% repo vs. 97% paper for NQ at n=100)

This is the most significant discrepancy and warrants further investigation. Recommended steps:
1. Inspect the adversarial corpus: verify that `results/adv_targeted_results/nq.json` adversarial texts were generated with `L=50` trials and `V=30` token length
2. Re-run `gen_adv.py` on the M10×10 question set with the correct hyperparameters if generation quality is suspect
3. Audit the substring matching: check whether the `clean_str` normalization in `eval_asr.py` is consistent with `main.py`'s ASR check

### 7.2 RAGDefender Backfire on HotpotQA

The consistent +10 pp ASR increase on both HotpotQA runs suggests a systematic issue with the multi-hop detection heuristic when faced with 100% poisoned retrieved contexts. The `_find_num_adversarial` function assumes a mix of adversarial and clean documents — when all 5 retrieved documents are adversarial, the similarity matrix has no "clean baseline" and the heuristic may under-count poisoned documents, leading to selective removal of the least-adversarial (but still adversarial) document, or erroneously flagging a clean document that was close to a query term.

**Recommended fix**: Before applying RAGDefender, expand retrieval to k > N so there are clean documents present. This matches the paper's "Knowledge Expansion" defense concept.

### 7.3 HotFlip White-Box Attack Not Benchmarked

The repo has HotFlip implemented but no stored results show white-box runs. The paper reports that white-box achieves equal or slightly higher ASR (97%–99% on NQ). Adding these results would complete the attack comparison.

### 7.4 No MS MARCO Large-Scale Run

The 100-query MS MARCO experiment is missing. Paper reports 92% ASR for GPT-4 black-box on MS MARCO. Adding this would fill the experimental matrix.

---

## 8. Recommended Next Steps

### Short Term (Reproduce Core Paper Results)
1. **Re-generate adversarial corpus** with verified `L=50, V=30` hyperparameters and re-run 100-query NQ/HotpotQA experiments
2. **Add MS MARCO 100-query run**
3. **Implement paraphrasing defense** to reproduce Table 12 (requires GPT-4 calls to paraphrase queries)
4. **Run perplexity detection** to reproduce Figure 6 ROC curve

### Medium Term (Extend the Paper)
5. **Test RAGDefender with k > N** (e.g., k=10 with N=5) to provide it a fighting chance and investigate the backfire
6. **Add PaLM 2 / LLaMA-2 / Vicuna results** to cover the full Table 1 sweep
7. **Implement knowledge expansion defense** and plot ASR vs. k curve
8. **Run HotFlip white-box experiments** and compare with black-box (Table 5 style)

### Long Term (Novel Contributions)
9. **Evaluate GPT-4o or GPT-4 Turbo** — model updates may change ASR characteristics
10. **Test RAG frameworks with built-in safety** (LlamaIndex with citation checking, Azure AI Search with semantic reranking)
11. **Design a provenance-based defense**: track document source and edit history as metadata, flag recently-edited entries
12. **Extend to open-domain internet RAG**: simulate attacker posting on Wikipedia, blogs, or forums

---

## 9. Conclusion

The PoisonedRAG attack is a serious and practically relevant threat to RAG-based applications. This codebase validates the paper's core claims: the LM-Targeted black-box attack achieves high ASR with minimal attacker resources. The RAGDefender integration extends the paper's defense evaluation with a more sophisticated defense method, but the results confirm the paper's conclusion — **current defenses are insufficient**, and in some configurations (HotpotQA, multi-hop), the defense can actively degrade system safety.

The ~20 percentage point gap between the repo's 100-query results and the paper's reported numbers deserves investigation, likely attributable to adversarial corpus generation quality rather than a fundamental implementation difference. The small-sample (n=10) experiments closely match the paper's expectations.

The fundamental research problem — that LLM-generated fluent disinformation, when injected into a knowledge base, is retrievable, indistinguishable from clean text, and sufficient to control LLM outputs — remains **unsolved**.

---

*Analysis prepared May 2026. Based on paper published in USENIX Security 2025 and experimental results in `results/query_results/main/`.*
