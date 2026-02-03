# 🤖 Fine-Tuning Pipeline — Mistral + Gemma (QLoRA / SFT)

> **End-to-end supervised fine-tuning of Mistral using LoRA on the Databricks Dolly 15k dataset.  
> Designed to run on a single GPU with minimal VRAM via 4-bit quantisation (QLoRA).**

---

![Python](https://img.shields.io/badge/Python-3.10%2B-3572A5?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD580?style=flat-square&logo=huggingface)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

---

## Table of Contents

1. [Project Overview](#1-project-overview)  
2. [Architecture](#2-architecture)  
3. [Repository Structure](#3-repository-structure)  
4. [Environment Setup](#4-environment-setup)  
5. [Step-by-Step Execution](#5-step-by-step-execution)  
6. [Configuration Reference](#6-configuration-reference)  
7. [GPU & VRAM Guide](#7-gpu--vram-guide)  
8. [Example Outputs](#8-example-outputs)  
9. [Monitoring with TensorBoard](#9-monitoring-with-tensorboard)  
10. [Troubleshooting](#10-troubleshooting)  
11. [Pipeline ↔ Workflow Mapping](#11-pipeline--workflow-mapping)  
12. [References](#12-references)  

---

## 1. Project Overview

This project implements a complete **Supervised Fine-Tuning (SFT)** pipeline for a large language model.  The goal is to take a pre-trained base model (Mistral) and adapt it to follow instructions more effectively, using the Alpaca-formatted Dolly 15k dataset.

**Key techniques used:**

| Technique | Purpose |
|---|---|
| **QLoRA** (4-bit quantisation + LoRA) | Train a 12-24 B model on a single consumer GPU |
| **LoRA** (Low-Rank Adaptation) | Update < 0.2 % of parameters — preserves pre-trained knowledge |
| **SFT** (Supervised Fine-Tuning) | Train on curated instruction → response pairs |
| **Alpaca prompt format** | Standardised instruction/input/output template |

---

## 2. Architecture

```
╔═══════════════════════════════════════════════════════════════════╗
║                        PIPELINE OVERVIEW                         ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║   ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    ║
║   │ INPUT LAYER │───▶│ PRE-PROCESS  │───▶│ CONTROL & SAFETY │    ║
║   │  (data in)  │    │  (validate,  │    │  (token caps,    │    ║
║   │             │    │   format)    │    │   guardrails)    │    ║
║   └─────────────┘    └──────────────┘    └────────┬─────────┘    ║
║                                                   ▼               ║
║   ┌─────────────────────────────────────────────────────────┐    ║
║   │                   ORCHESTRATOR                           │    ║
║   │          Route tasks → split → rule routing              │    ║
║   └────────┬───────────────────────┬────────────────────────┘    ║
║            ▼                       ▼                              ║
║   ┌────────────────┐    ┌─────────────────────┐                  ║
║   │     CACHE      │    │        RAG          │                  ║
║   │ (prompt hash,  │    │ (embeddings,        │                  ║
║   │  embed cache)  │    │  vector search)     │                  ║
║   └────────┬───────┘    └──────────┬──────────┘                  ║
║            ▼                       ▼                              ║
║   ┌─────────────────────────────────────────────────────────┐    ║
║   │               MISTRAL 12B  (Base Model)                  │    ║
║   │         Core reasoning & analysis  │  Low temp           │    ║
║   └────────────────────────┬──────────────────────────────┘    ║
║                            ▼                                     ║
║   ┌──────────────────┐     │     ┌────────────────────┐         ║
║   │  FINE-TUNING     │◀────┘     │   AGENT MODELS     │         ║
║   │  SFT + LoRA      │           │   (Gemma)          │         ║
║   │  This project ✦  │           │  Execute tasks     │         ║
║   └──────────────────┘           └────────┬───────────┘         ║
║                                           ▼                      ║
║                              ┌─────────────────────────┐        ║
║                              │   POST-PROCESS & VALIDATE│        ║
║                              │   Rules, schema checks   │        ║
║                              └────────────┬────────────┘        ║
║                                           ▼                      ║
║                              ┌─────────────────────────┐        ║
║                              │       FINAL OUTPUT      │        ║
║                              │  Deliver response & log │        ║
║                              └─────────────────────────┘        ║
║                                                                   ║
║   Side systems:                                                   ║
║   ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐     ║
║   │  MONITORING    │  │ VECTOR DB    │  │  FEEDBACK LOOP  │     ║
║   │  Cost/Latency  │  │ Embeddings   │  │  Ratings, Error │     ║
║   └────────────────┘  └──────────────┘  └─────────────────┘     ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## 3. Repository Structure

```
fine-tuning-project/
│
├── config.py                   # ← ALL hyperparameters (single source of truth)
├── prepare_data.py             # Stage 1: download → validate → format → split
├── train.py                    # Stage 2: QLoRA SFT training loop
├── evaluate.py                 # Stage 3: loss, perplexity, sample generations
├── inference.py                # Stage 4: interactive / single / batch inference
│
├── sample_data/
│   ├── dolly_sample_10.json    # 10-row self-contained sample dataset
│   ├── batch_prompts.txt       # Example prompts for batch inference
│   ├── train.json              # ← generated by prepare_data.py  (gitignored)
│   └── val.json                # ← generated by prepare_data.py  (gitignored)
│
├── example_outputs/
│   └── results_placeholder.json   # Template showing expected output format
│
├── output/                     # ← generated during training  (gitignored)
│   ├── checkpoint-200/
│   ├── checkpoint-400/
│   └── final_model/
│
├── logs/                       # ← TensorBoard logs  (gitignored)
│
├── requirements.txt            # Python dependencies
├── .gitignore
├── .vscode/
│   └── launch.json             # F5 debug configs for every script
└── README.md                   # This file
```

---

## 4. Environment Setup

### 4.1 Clone the repo

```bash
git clone https://github.com/nandinij015/fine-tuning-project.git
cd fine-tuning-project
```

### 4.2 Create a virtual environment

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4.3 Install dependencies

```bash
pip install -r requirements.txt
```

> **PyTorch install note:** If you have a specific CUDA version, install PyTorch first via the [official selector](https://pytorch.org/get-started/locally/) before running the requirements install.

### 4.4 Verify the setup

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## 5. Step-by-Step Execution

Run the four stages **in order**.  Each stage produces the files the next one needs.

---

### Step 1 — Prepare Data

```bash
# Option A: Use the bundled 10-row sample (no internet needed)
python prepare_data.py

# Option B: Download the full 15 k dataset from HuggingFace
python prepare_data.py --full
```

**What happens:**

| Action | Detail |
|--------|--------|
| Load | Reads `sample_data/dolly_sample_10.json` (or downloads 15 k) |
| Validate | Drops any row where `instruction` or `output` is empty |
| Format | Wraps each row in the Alpaca template (`### Instruction / Input / Response`) |
| Split | 90 % → `train.json`, 10 % → `val.json` (seeded shuffle) |
| Save | Writes both files to `sample_data/` |

**Expected output:**
```
12:00:00  [INFO   ]  Loaded 10 rows from local sample
12:00:00  [INFO   ]  Validated  →  kept: 10  |  skipped: 0
12:00:00  [INFO   ]  Split  →  train: 9  |  val: 1
12:00:00  [INFO   ]  Saved  →  sample_data/train.json  (9 rows)
12:00:00  [INFO   ]  Saved  →  sample_data/val.json  (1 rows)
```

---

### Step 2 — Train

```bash
# Fresh training run
python train.py

# Resume from a checkpoint (if a previous run was interrupted)
python train.py --resume output/checkpoint-200
```

**What happens:**

| Phase | Detail |
|-------|--------|
| Load base model | Downloads Mistral in 4-bit (NF4 quantisation) |
| Attach LoRA | Adds trainable rank-16 matrices to q/k/v/o attention layers |
| Tokenise | Converts text → token IDs, pads to 512 tokens |
| Train | Runs SFT via HuggingFace `SFTTrainer`; saves checkpoints every 200 steps |
| Save | Writes the final LoRA adapter to `output/final_model/` |

**Expected console output (placeholder):**
```
12:01:00  [INFO   ]  Loading base model (4-bit)  →  mistralai/Mistral-Small-…
12:01:45  [INFO   ]  Attaching LoRA  →  r=16  alpha=32  modules=['q_proj', …]
12:01:45  [INFO   ]  trainable params: 24.11M || all params: 12.47B || trainable: 0.19%
12:02:00  [INFO   ]  ═══ Training starting … ═══
          {'loss': 2.8413, 'learning_rate': 0.00018, 'epoch': 0.03}
          …
          {'loss': 1.0893, 'learning_rate': 0.00003, 'epoch': 3.0}
12:04:05  [INFO   ]  ═══ Training complete  →  model saved to output/final_model ═══
```

---

### Step 3 — Evaluate

```bash
python evaluate.py                       # default: 5 sample generations
python evaluate.py --num_samples 10      # show 10 examples
```

**What happens:**

| Phase | Detail |
|-------|--------|
| Load | Base model + LoRA adapter in eval mode |
| Loss | Forward-pass each val sample; average the cross-entropy loss |
| Perplexity | `exp(avg_loss)` — lower is better |
| Generations | For each sample: run inference and print side-by-side with ground truth |
| Report | Writes `example_outputs/eval_report.json` |

**Expected output (placeholder):**
```
12:05:00  [INFO   ]  ═══ Metrics ═══
                       Avg Eval Loss : [PLACEHOLDER — e.g. 1.1247]
                       Perplexity    : [PLACEHOLDER — e.g. 3.08]
                       Samples used  : 1
```

---

### Step 4 — Inference

```bash
# 🗨️  Interactive chat loop
python inference.py

# 📝  Single prompt
python inference.py --prompt "Explain what LoRA is in simple terms"

# 📂  Batch mode (reads sample_data/batch_prompts.txt)
python inference.py --batch sample_data/batch_prompts.txt
```

**Expected output (placeholder):**
```
You (instruction): Explain what LoRA is in simple terms
You (input — or press Enter to skip): 

🤖 Response:
[PLACEHOLDER — paste your model's actual response here after running]
```

---

## 6. Configuration Reference

All tuneable values live in **`config.py`**.  Here is the full reference:

### Dataset

| Key | Default | Description |
|-----|---------|-------------|
| `hf_id` | `databricks/dolly-15k-…` | HuggingFace dataset identifier |
| `use_sample` | `True` | `True` = 10-row sample; `False` = full 15 k |
| `train_split` | `0.90` | Fraction of data used for training |
| `seed` | `42` | Random seed (ensures reproducible splits) |

### Model & Quantisation

| Key | Default | Description |
|-----|---------|-------------|
| `base_model_name` | `Mistral-Small-24B-…` | HuggingFace model ID |
| `load_in_4bit` | `True` | Enable 4-bit QLoRA quantisation |
| `quant_type` | `nf4` | Normal-Float 4-bit (best quality at 4-bit) |
| `compute_dtype` | `bfloat16` | Dtype for compute kernels |

### LoRA

| Key | Default | Description |
|-----|---------|-------------|
| `r` | `16` | Low-rank dimension (↑ = more expressiveness, more VRAM) |
| `alpha` | `32` | Scaling factor (rule: `2 × r`) |
| `dropout` | `0.05` | Dropout on LoRA layers to prevent overfitting |
| `target_modules` | `[q, k, v, o]` | Attention projections that receive adapters |

### Training

| Key | Default | Description |
|-----|---------|-------------|
| `num_train_epochs` | `3` | Full passes over the training set |
| `per_device_train_batch` | `2` | Batch size per GPU |
| `gradient_accumulation` | `8` | Effective batch = `2 × 8 = 16` |
| `learning_rate` | `2e-4` | AdamW learning rate |
| `lr_scheduler` | `cosine` | LR decay schedule |
| `max_seq_length` | `512` | Maximum tokens per sample |
| `bf16` | `True` | Use bfloat16 mixed precision |

---

## 7. GPU & VRAM Guide

| Model Size | Min VRAM (4-bit) | Recommended GPU |
|---|---|---|
| 7 B | ~8 GB | NVIDIA T4, RTX 3090 |
| 12 B | ~14 GB | RTX 4090, A10 |
| 24 B | ~26 GB | A100 40/80 GB |

> 💡 **No local GPU?**  Run on a free [Google Colab](https://colab.research.google.com) T4 runtime.  Clone the repo into Colab, install requirements, and execute the same commands.  For the 24 B model, upgrade to an A100 Colab runtime.

---

## 8. Example Outputs

A template showing the expected structure of training logs, eval metrics, and sample generations is provided in:

```
example_outputs/results_placeholder.json
```

After you run the pipeline, replace the `[PLACEHOLDER]` fields with your actual numbers.  The file documents:

- Training loss curve (step → loss)
- Final trainable parameter count and percentage
- Eval loss and perplexity
- Side-by-side ground-truth vs model-generated responses

---

## 9. Monitoring with TensorBoard

Training logs are written to `logs/`.  Start TensorBoard in a **second terminal**:

```bash
tensorboard --logdir logs/
```

Open [http://localhost:6006](http://localhost:6006) in your browser.  You will see:

- **Loss curve** — training and eval loss over steps
- **Learning rate** — the cosine decay schedule
- **GPU utilisation** — if reported by the driver

---

## 10. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch size too large for your GPU | Lower `per_device_train_batch` to `1`; raise `gradient_accumulation` to `16` |
| `train.json not found` | `prepare_data.py` was not run | Run `python prepare_data.py` first |
| `final_model not found` | Training did not complete | Check `output/` for the latest checkpoint; resume with `--resume` |
| Slow download | First run fetches the base model (~12 GB) | Ensure a stable internet connection; the model caches locally after the first download |
| `bitsandbytes` import error | CUDA toolkit mismatch | Install the CUDA-matched version of `bitsandbytes` from PyPI |

---

## 11. Pipeline ↔ Workflow Mapping

Each script in this repo maps directly to a stage in the original pipeline diagram:

| Script | Pipeline Stage(s) |
|--------|-------------------|
| `prepare_data.py` | INPUT LAYER → PRE-PROCESSING |
| `train.py` | FINE-TUNING (SFT + LoRA) → MISTRAL 12B |
| `evaluate.py` | POST-PROCESS & VALIDATE + MONITORING |
| `inference.py` | AGENT MODELS → FINAL OUTPUT |
| `config.py` | CONTROL & SAFETY (token caps, guardrails) |

Side systems (CACHE, RAG, VECTOR DATABASE, FEEDBACK LOOP) are outside the scope of this fine-tuning project but are referenced in the architecture diagram above for completeness.

---

## 12. References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/dolly-15k-instruction-alpaca-format)
- [HuggingFace PEFT](https://huggingface.co/docs/peft)
- [TRL — Transformer Reinforcement Learning](https://huggingface.co/docs/trl)

---

*Built by [nandinij015](https://github.com/nandinij015)*
