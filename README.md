# PoisonedRAG Defense Framework

> **Based on:** [PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models](https://arxiv.org/abs/2402.07867)
> — USENIX Security 2025

This repository extends the official PoisonedRAG codebase with an integrated defense framework (RAGDefender), additional evaluation tooling, and experiment results across multiple LLMs and datasets.

![Illustration of PoisonedRAG](PoisonedRAG.png)

---

## Overview

**Retrieval-Augmented Generation (RAG)** systems enhance LLMs by retrieving relevant passages from a knowledge base at query time. This project demonstrates — and defends against — **knowledge corruption attacks** that inject adversarially crafted documents into the corpus to manipulate LLM outputs.

### What This Framework Does

| Component | Description |
|-----------|-------------|
| **Attack** | Inject adversarial passages that cause a dense retriever (Contriever) to surface attacker-controlled content |
| **LM-Targeted Attack** | Black-box method: prepend the target question to a crafted incorrect-answer passage |
| **HotFlip Attack** | White-box gradient-based token substitution on Contriever embeddings |
| **RAGDefender** | Pair-based poisoning detection that estimates and removes adversarial documents before LLM prompting |
| **Evaluation** | ASR (Attack Success Rate) computation with and without defense across NQ, HotpotQA, MS MARCO |

---

## Experimental Results

All experiments use **Contriever** as the dense retriever with **Top-5** retrieved passages and the **LM-Targeted** (black-box) attack method. Results below are from GPT-4 (gpt-4-0613) unless noted.

### Attack Success Rate (ASR) — GPT-4

| Dataset | # Queries | ASR (No Defense) | ASR (With RAGDefender) | Config |
|---------|-----------|-----------------|------------------------|--------|
| NQ | 100 | — | **77.0%** | M10×10 |
| NQ | 10 | **100.0%** | **100.0%** | M10×1 |
| NQ | 10 | **90.0%** | **80.0%** | M10×1 v2 |
| HotpotQA | 100 | — | **76.0%** | M10×10 |
| HotpotQA | 10 | **80.0%** | **90.0%** | M10×1 |
| HotpotQA | 10 | **80.0%** | **90.0%** | M10×1 v2 |
| MS MARCO | 10 | **100.0%** | **100.0%** | M10×1 |
| MS MARCO | 10 | **100.0%** | **80.0%** | M10×1 v2 |

### Attack Success Rate (ASR) — GPT-3.5

| Dataset | # Queries | ASR (No Defense) | ASR (With RAGDefender) | Config |
|---------|-----------|-----------------|------------------------|--------|
| NQ | 2 | **100.0%** | **100.0%** | M2×1 |

### Aggregate Summary

| Metric | Mean | Std Dev |
|--------|------|---------|
| ASR (No Defense) | **92.86%** | ±8.81% |
| ASR (With RAGDefender) | **86.00%** | ±9.79% |

> **Key Finding:** The LM-Targeted black-box attack achieves >90% ASR on average across all tested datasets and models. The RAGDefender defense reduces ASR in some configurations but fails to fully neutralize the attack, highlighting the need for more robust defenses.

---

## Repository Structure

```
PoisonedRAG/
├── main.py                          # Main experiment driver
├── run.py                           # Experiment launcher (hyperparameter config)
├── gen_adv.py                       # Adversarial passage generation
├── evaluate_beir.py                 # Dense retrieval evaluation (BEIR)
├── prepare_dataset.py               # Dataset downloader
├── eval_asr.py                      # ASR metric computation
├── run_eval_3datasets.py            # Multi-dataset evaluation runner
├── run_with_and_without_defense.py  # Comparative defense evaluation
│
├── src/
│   ├── attack.py                    # LM_targeted + HotFlip attack implementations
│   ├── utils.py                     # Shared utilities
│   ├── prompts.py                   # LLM prompt templates
│   ├── models/
│   │   ├── GPT.py                   # OpenAI GPT-3.5 / GPT-4
│   │   ├── Llama.py                 # LLaMA-2 (7B / 13B)
│   │   ├── Vicuna.py                # Vicuna (7B / 13B / 33B)
│   │   └── PaLM2.py                 # Google PaLM 2
│   └── contriever_src/              # Facebook Contriever integration
│
├── defense/
│   └── defense_runner.py            # RAGDefender: pair-based poisoning detection
│
├── model_configs/
│   ├── *.template.json              # Config templates (copy and fill in API keys)
│   └── *.json                       # Your local configs (git-ignored, never committed)
│
├── scripts/
│   └── compute_asr_from_results.py  # CLI wrapper for eval_asr.py
│
├── results/
│   ├── query_results/main/          # Per-query experiment outputs (JSON)
│   └── (beir_results/, adv_targeted_results/ — git-ignored, large files)
│
├── RAGDefender/                     # RAGDefender submodule/package
├── requirements.txt                 # Python dependencies
└── asr_test.csv                     # Quick ASR summary
```

---

## Setup

### 1. Clone and Create Environment

```bash
git clone https://github.com/aquibraza/poisoned-rag-defense-framework.git
cd poisoned-rag-defense-framework

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For **GPU support** (CUDA 11.7), replace the torch lines with:

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

### 3. Configure API Keys

Copy the template config files and add your own API keys — **never commit the filled-in files**:

```bash
# OpenAI (GPT-3.5 / GPT-4)
cp model_configs/gpt4_config.template.json model_configs/gpt4_config.json
# Edit model_configs/gpt4_config.json and replace YOUR_OPENAI_API_KEY_HERE

# Google PaLM 2
cp model_configs/palm2_config.template.json model_configs/palm2_config.json

# LLaMA-2 (HuggingFace access token required)
cp model_configs/llama7b_config.template.json model_configs/llama7b_config.json
```

> **Security:** `model_configs/*.json` is listed in `.gitignore`. Only `*.template.json` files are tracked.

### 4. Download Datasets (optional — auto-downloaded on first run)

```bash
python prepare_dataset.py
```

Datasets are saved under `datasets/` (git-ignored): **NQ**, **HotpotQA**, **MS MARCO** (BEIR format).

---

## Running Experiments

### Quick Start

Configure hyperparameters in `run.py`:

```python
test_params = {
    'eval_model_code': "contriever",
    'eval_dataset': "nq",           # nq | hotpotqa | msmarco
    'model_name': 'gpt4',           # gpt3.5 | gpt4 | palm2 | llama(7b|13b) | vicuna(7b|13b|33b)
    'top_k': 5,
    'attack_method': 'LM_targeted', # LM_targeted (black-box) | hotflip (white-box)
    'adv_per_query': 5,
    'score_function': 'dot',
    'repeat_times': 10,
    'M': 10,
}
```

```bash
python run.py
```

### With Defense

```bash
python run_with_and_without_defense.py
```

### Evaluate ASR from Saved Results

```bash
python eval_asr.py --dir results/query_results/main --output_csv asr_summary.csv
```

### Multi-Dataset Evaluation

```bash
python run_eval_3datasets.py
```

---

## Attack Methods

### LM-Targeted (Black-Box)

Constructs adversarial passages by prepending the target question to an LLM-generated passage that supports an incorrect answer. No access to retriever internals required.

```
adversarial_text = f"{question}. {incorrect_supporting_passage}"
```

### HotFlip (White-Box)

Gradient-based token substitution on Contriever embeddings. Iteratively replaces tokens to maximize cosine similarity between the adversarial passage and the target query.

---

## Defense: RAGDefender

Located in `defense/defense_runner.py`, RAGDefender detects and removes poisoned documents before LLM prompting:

1. **Estimate** the number of adversarial documents using:
   - *HotpotQA*: embedding similarity heuristic
   - *NQ / MS MARCO*: agglomerative clustering + TF-IDF scoring
2. **Score** documents by similarity of top-k pairs
3. **Remove** the highest-scoring (most suspicious) documents
4. **Prompt** the LLM with the cleaned context

The default experiments above use `k=5` retrieved passages with `N=5`
injected adversarial passages — i.e. a **100%-poisoned retrieved context**,
which is a deliberate stress test rather than RAGDefender's assumed
operating point. See [`docs/RAGDEFENDER_DIAGNOSTIC_PLAN.md`](docs/RAGDEFENDER_DIAGNOSTIC_PLAN.md)
for a diagnostic evaluation (poison-labeled passage diagnostics, a `k`-sweep
past `N`, and oracle/random diagnostic controls) that investigates whether
RAGDefender's inconsistent results come from this saturation, an
implementation issue, or an evaluation issue.

---

## Defense: FilterRAG

A second, independent defense baseline (`defense/filterrag.py`), based on
Edemacu et al. (2025): a per-passage statistical filter that scores each
retrieved passage by keyword "Freq-Density" (query + small-model-generated
answer keywords vs. the passage's own text) and drops passages above a
threshold. Unlike RAGDefender's cross-passage clustering, this doesn't rely
on a clean-cluster anchor, so it's evaluated as a candidate that may be more
robust to the 100%-poisoned-context case. See
[`docs/FILTERRAG_BASELINE.md`](docs/FILTERRAG_BASELINE.md) for the
implementation, its deviations from the published method (a small local
model substitutes for the paper's LLaMA-2/3 SLM), and how to run its
diagnostics (`--defense filterrag` / `filterrag_query_only`,
`--quick_filterrag_hotpotqa`).

---

## Supported Models

| Model | Provider | Config Template |
|-------|----------|----------------|
| GPT-3.5-turbo | OpenAI | `gpt3.5_config.template.json` |
| GPT-4 (gpt-4-0613) | OpenAI | `gpt4_config.template.json` |
| PaLM 2 (text-bison-001) | Google | `palm2_config.template.json` |
| LLaMA-2-7B-Chat | HuggingFace | `llama7b_config.template.json` |
| LLaMA-2-13B-Chat | HuggingFace | `llama13b_config.template.json` |
| Vicuna-7B / 13B / 33B | HuggingFace | `vicuna*_config.template.json` |

---

## Citation

If you use this code or build on this research, please cite the original paper:

```bibtex
@article{zou2024poisonedrag,
  title={PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models},
  author={Zou, Wei and Geng, Runpeng and Wang, Binghui and Jia, Jinyuan},
  journal={arXiv preprint arXiv:2402.07867},
  year={2024}
}
```

---

## Acknowledgements

- [corpus-poisoning](https://github.com/princeton-nlp/corpus-poisoning) — base attack implementation
- [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection) — model abstraction layer
- [BEIR](https://github.com/beir-cellar/beir) — retrieval benchmark
- [Contriever](https://github.com/facebookresearch/contriever) — dense retriever
- [RAGDefender](https://github.com/leileigan/RAGDefender) — defense implementation

---

## Security Notice

**Never commit API keys.** The `model_configs/*.json` files (with real keys) are excluded via `.gitignore`. Only `*.template.json` skeleton files are tracked. If you accidentally expose a key, revoke it immediately in your provider's dashboard.
