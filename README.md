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
## References
1. https://colab.research.google.com/github/brevdev/notebooks/blob/main/mistral-finetune.ipynb#scrollTo=1hFsEFp5yRgg
2. https://medium.com/@codersama/fine-tuning-mistral-7b-in-google-colab-with-qlora-complete-guide-60e12d437cca
3. https://huggingface.co/mistralai/Mistral-7B-v0.1/discussions/133
4. https://github.com/krishnaik06/Finetuning-LLM/tree/main
5. https://www.youtube.com/watch?v=qcjrduz_YS8
6. https://www.youtube.com/watch?v=UWo9r6flDjk
