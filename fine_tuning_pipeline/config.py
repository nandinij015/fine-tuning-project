"""
config.py
═══════════════════════════════════════════════════════════════
 Central Configuration — Single Source of Truth
═══════════════════════════════════════════════════════════════

 Every tuneable value in the pipeline lives here.
 No other script defines hyperparameters; they all import from
 this module.  Changing a value here propagates everywhere.

 ┌─ How to use ─────────────────────────────────────────────┐
 │  • Edit the values in this file as needed.                │
 │  • Run the pipeline:                                      │
 │      python prepare_data.py   →  downloads & formats data │
 │      python train.py          →  QLoRA fine-tuning        │
 │      python evaluate.py       →  metrics + samples        │
 │      python inference.py      →  interactive chat         │
 └───────────────────────────────────────────────────────────┘

 Last updated : 2026-02-03
 Author       : nandinij015
═══════════════════════════════════════════════════════════════
"""

import os

# ─────────────────────────────────────────────────────────────
# 1.  DIRECTORY PATHS
# ─────────────────────────────────────────────────────────────
# All paths are relative to this file's location so the project
# is portable — move the whole folder and nothing breaks.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "data_dir"        : os.path.join(BASE_DIR, "sample_data"),       # processed JSON lives here
    "output_dir"      : os.path.join(BASE_DIR, "output"),            # checkpoints
    "final_model_dir" : os.path.join(BASE_DIR, "output", "final_model"),
    "logs_dir"        : os.path.join(BASE_DIR, "logs"),              # TensorBoard logs
    "example_outputs" : os.path.join(BASE_DIR, "example_outputs"),
}

# ─────────────────────────────────────────────────────────────
# 2.  DATASET
# ─────────────────────────────────────────────────────────────
# HuggingFace dataset ID used by prepare_data.py when pulling
# the full 15 k dataset.  The 10-row sample in sample_data/ is
# used for dry runs when USE_SAMPLE = True.

DATASET = {
    "hf_id"          : "databricks/dolly-15k-instruction-alpaca-format",
    "sample_file"    : os.path.join(BASE_DIR, "sample_data", "dolly_sample_10.json"),
    "train_split"    : 0.90,   # 90 % train
    "val_split"      : 0.10,   # 10 % validation
    "seed"           : 42,
    "use_sample"     : True,   # ← set False to download the full 15 k dataset
}

# ─────────────────────────────────────────────────────────────
# 3.  BASE MODEL
# ─────────────────────────────────────────────────────────────
# Your pipeline diagram specifies Mistral 12B as the base.
# Mistral-Small-24B-Instruct is the closest publicly available
# checkpoint in that family.  Swap the ID below if you have a
# local or fine-tuned variant.
#
# ⚡ VRAM guide (4-bit QLoRA)
#    7 B  →  ~8 GB   (T4 / RTX 3090)
#   12 B  → ~14 GB   (RTX 4090 / A10)
#   24 B  → ~26 GB   (A100 40 GB)

MODEL = {
    "base_model_name" : "mistralai/Mistral-Small-24B-Instruct-2501",
    # Lighter swap if your GPU has < 16 GB:
    # "base_model_name" : "mistralai/Mistral-7B-Instruct-v0.3",
}

# ─────────────────────────────────────────────────────────────
# 4.  QUANTISATION  (QLoRA / bitsandbytes)
# ─────────────────────────────────────────────────────────────

QUANTIZATION = {
    "load_in_4bit"       : True,
    "quant_type"         : "nf4",       # Normal-Float 4-bit — best quality at 4-bit
    "compute_dtype"      : "bfloat16",  # "bfloat16" | "float16"
    "use_double_quant"   : True,        # Saves ~0.1 bit/param; negligible quality loss
}

# ─────────────────────────────────────────────────────────────
# 5.  LoRA  (Parameter-Efficient Fine-Tuning)
# ─────────────────────────────────────────────────────────────
# rank (r)     — number of low-rank dimensions.  Higher = more
#                trainable params, better expressiveness, more VRAM.
# alpha        — scaling factor.  Rule of thumb: alpha = 2 × r.
# target_modules — which projection layers receive LoRA adapters.
#                  Mistral attention layers: q, k, v, o projections.

LORA = {
    "r"              : 16,
    "alpha"          : 32,
    "dropout"        : 0.05,
    "target_modules" : ["q_proj", "v_proj", "k_proj", "o_proj"],
    "bias"           : "none",
    "task_type"      : "CAUSAL_LM",
}

# ─────────────────────────────────────────────────────────────
# 6.  TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
# Effective batch size = per_device_batch × gradient_accumulation
#                      = 2 × 8 = 16  (default)

TRAINING = {
    "num_train_epochs"          : 3,
    "per_device_train_batch"    : 2,       # lower if OOM
    "gradient_accumulation"     : 8,       # raise if you lower batch
    "learning_rate"             : 2e-4,
    "weight_decay"              : 0.01,
    "warmup_steps"              : 100,
    "lr_scheduler"              : "cosine",  # "cosine" | "linear" | "constant"
    "max_seq_length"            : 512,       # tokens per sample
    "bf16"                      : True,      # set False + fp16=True if bf16 unsupported
}

# ─────────────────────────────────────────────────────────────
# 7.  CHECKPOINTING & LOGGING
# ─────────────────────────────────────────────────────────────

CHECKPOINTING = {
    "logging_steps"        : 50,
    "save_strategy"        : "steps",   # "steps" | "epoch"
    "save_steps"           : 200,
    "save_total_limit"     : 3,         # keep only the last N checkpoints
    "eval_steps"           : 200,
    "eval_strategy"        : "steps",
    "report_to"            : "none",    # "tensorboard" | "wandb" | "none"
}

# ─────────────────────────────────────────────────────────────
# 8.  INFERENCE
# ─────────────────────────────────────────────────────────────

INFERENCE = {
    "max_new_tokens"     : 256,
    "temperature"        : 0.7,
    "top_p"              : 0.9,
    "repetition_penalty" : 1.2,
}
