# Fine-Tuning Pipeline — Mistral + Gemma (QLoRA / SFT)

End-to-end supervised fine-tuning of Mistral using LoRA on the Databricks Dolly 15k dataset.  
Designed to run on a single GPU with minimal VRAM via 4-bit quantisation (QLoRA).


## 1. Project Overview

This project implements a complete **Supervised Fine-Tuning (SFT)** pipeline for a large language model.  The goal is to take a pre-trained base model (Mistral) and adapt it to follow instructions more effectively, using the Alpaca-formatted Dolly 15k dataset.

**Key techniques used:**

| Technique | Purpose |
|---|---|
| **QLoRA** (4-bit quantisation + LoRA) | Train a 7 B model on a single consumer GPU |
| **LoRA** (Low-Rank Adaptation) | Update < 0.2 % of parameters — preserves pre-trained knowledge |
| **SFT** (Supervised Fine-Tuning) | Train on curated instruction → response pairs |
| **Alpaca prompt format** | Standardised instruction/input/output template |

---
## Reference
https://colab.research.google.com/github/brevdev/notebooks/blob/main/mistral-finetune.ipynb#scrollTo=1hFsEFp5yRgg
