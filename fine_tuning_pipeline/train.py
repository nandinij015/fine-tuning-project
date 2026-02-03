"""
train.py
═══════════════════════════════════════════════════════════════
 Stage 2 / 4 — QLoRA Supervised Fine-Tuning (SFT)
═══════════════════════════════════════════════════════════════

 What this script does
 ─────────────────────
 1. Loads the base Mistral model in 4-bit (QLoRA).
 2. Attaches LoRA adapter layers to the attention projections.
 3. Tokenises the prepared dataset (output of prepare_data.py).
 4. Runs the SFT training loop via HuggingFace's SFTTrainer.
 5. Saves checkpoints during training and the final model at the end.

 Pipeline position
 ─────────────────
 Your workflow:  … → FINE-TUNING (SFT + LoRA) → Mistral 12B → …

 Prerequisites
 ─────────────
     pip install -r requirements.txt
     python prepare_data.py          # generates train.json & val.json

 Usage
 ─────
     python train.py                                        # fresh run
     python train.py --resume output/checkpoint-200         # resume

 GPU requirement
 ───────────────
     ≥ 24 GB VRAM for the 24 B model  (A100 / RTX 4090)
     ≥  8 GB VRAM for the  7 B model  (T4 / RTX 3090)
═══════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import os
import sys

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl  import SFTTrainer

# ── project imports ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL, QUANTIZATION, LORA, TRAINING, CHECKPOINTING, PATHS

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)-7s]  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════
# 1.  TOKENISER
# ═════════════════════════════════════════════════════════════


def load_tokenizer() -> AutoTokenizer:
    """
    Load the tokeniser that matches the base model.

    Mistral does not ship a pad token, so we reuse the EOS token.
    padding_side = "right" is required for causal (left-to-right) LMs.
    """
    log.info("Loading tokenizer  →  %s", MODEL["base_model_name"])
    tok = AutoTokenizer.from_pretrained(MODEL["base_model_name"], use_fast=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token          # standard Mistral workaround

    tok.padding_side = "right"
    log.info("Tokenizer ready  |  vocab size: %d", tok.vocab_size)
    return tok


# ═════════════════════════════════════════════════════════════
# 2.  BASE MODEL  (4-bit quantised)
# ═════════════════════════════════════════════════════════════


def load_base_model():
    """
    Download and load the base model with bitsandbytes 4-bit quantisation.

    Key settings explained
    ──────────────────────
    • nf4           – Normal-Float 4-bit: best accuracy at this bit-width.
    • double_quant  – Quantises the quantisation constants again → saves ~0.1 bit/param.
    • device_map    – "auto" spreads layers across all available GPUs.

    After loading we call prepare_model_for_kbit_training() which:
      – casts LayerNorm layers to fp32 (required for stable gradients)
      – enables gradient checkpointing
    """
    dtype_map     = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    compute_dtype = dtype_map.get(QUANTIZATION["compute_dtype"], torch.bfloat16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit          = QUANTIZATION["load_in_4bit"],
        bnb_4bit_quant_type   = QUANTIZATION["quant_type"],
        bnb_4bit_compute_dtype= compute_dtype,
        bnb_4bit_use_double_quant = QUANTIZATION["use_double_quant"],
    )

    log.info("Loading base model (4-bit)  →  %s", MODEL["base_model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL["base_model_name"],
        quantization_config = bnb_config,
        device_map          = "auto",
        torch_dtype         = compute_dtype,
    )

    # Required before attaching LoRA to a kbit model
    model = prepare_model_for_kbit_training(model)
    log.info("Base model loaded  |  device map: %s", model.hf_device_map)
    return model


# ═════════════════════════════════════════════════════════════
# 3.  LoRA ADAPTER
# ═════════════════════════════════════════════════════════════


def attach_lora(model):
    """
    Wrap the base model with LoRA adapters.

    Only the LoRA matrices are trainable (~0.2 % of total params).
    Everything else stays frozen, preserving the pre-trained knowledge
    while letting the model learn task-specific behaviour.
    """
    log.info("Attaching LoRA  →  r=%d  alpha=%d  modules=%s",
             LORA["r"], LORA["alpha"], LORA["target_modules"])

    lora_cfg = LoraConfig(
        r              = LORA["r"],
        lora_alpha     = LORA["alpha"],
        lora_dropout   = LORA["dropout"],
        target_modules = LORA["target_modules"],
        bias           = LORA["bias"],
        task_type      = LORA["task_type"],
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()   # logs trainable / total / %
    return model


# ═════════════════════════════════════════════════════════════
# 4.  DATASET
# ═════════════════════════════════════════════════════════════


def load_data() -> tuple[Dataset, Dataset]:
    """
    Read the JSON files produced by prepare_data.py and return
    HuggingFace Dataset objects.
    """
    train_path = os.path.join(PATHS["data_dir"], "train.json")
    val_path   = os.path.join(PATHS["data_dir"], "val.json")

    if not os.path.exists(train_path):
        log.error("train.json not found at %s  →  run prepare_data.py first", train_path)
        sys.exit(1)

    with open(train_path) as f:
        train_rows = json.load(f)
    with open(val_path) as f:
        val_rows   = json.load(f)

    log.info("Data loaded  →  train: %d  |  val: %d", len(train_rows), len(val_rows))
    return Dataset.from_list(train_rows), Dataset.from_list(val_rows)


# ═════════════════════════════════════════════════════════════
# 5.  TOKENISATION
# ═════════════════════════════════════════════════════════════


def tokenize(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """
    Convert the 'text' column (pre-formatted Alpaca prompt) into
    token IDs.  Truncate to max_seq_length; pad to that length.
    """
    max_len = TRAINING["max_seq_length"]

    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation = True,
            max_length = max_len,
            padding    = "max_length",
        )

    tokenised = dataset.map(_tok, batched=True, remove_columns=dataset.column_names)
    tokenised.set_format("torch")
    log.info("Tokenised  →  %d samples  |  max_length=%d", len(tokenised), max_len)
    return tokenised


# ═════════════════════════════════════════════════════════════
# 6.  TRAINING ARGUMENTS
# ═════════════════════════════════════════════════════════════


def build_training_args() -> TrainingArguments:
    """
    Construct the HuggingFace TrainingArguments from config.py values.

    gradient_checkpointing = True  trades some compute for a big VRAM
    saving — essential for models > 7 B on a single GPU.
    """
    os.makedirs(PATHS["logs_dir"], exist_ok=True)

    return TrainingArguments(
        output_dir                  = PATHS["output_dir"],
        num_train_epochs            = TRAINING["num_train_epochs"],
        per_device_train_batch_size = TRAINING["per_device_train_batch"],
        per_device_eval_batch_size  = TRAINING["per_device_train_batch"],
        gradient_accumulation_steps = TRAINING["gradient_accumulation"],
        learning_rate               = TRAINING["learning_rate"],
        weight_decay                = TRAINING["weight_decay"],
        warmup_steps                = TRAINING["warmup_steps"],
        lr_scheduler_type           = TRAINING["lr_scheduler"],
        bf16                        = TRAINING["bf16"],
        fp16                        = not TRAINING["bf16"],
        logging_dir                 = PATHS["logs_dir"],
        logging_steps               = CHECKPOINTING["logging_steps"],
        save_strategy               = CHECKPOINTING["save_strategy"],
        save_steps                  = CHECKPOINTING["save_steps"],
        save_total_limit            = CHECKPOINTING["save_total_limit"],
        evaluation_strategy         = CHECKPOINTING["eval_strategy"],
        eval_steps                  = CHECKPOINTING["eval_steps"],
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        report_to                   = CHECKPOINTING["report_to"],
        dataloader_num_workers      = 2,
        gradient_checkpointing     = True,
    )


# ═════════════════════════════════════════════════════════════
# 7.  MAIN TRAINING LOOP
# ═════════════════════════════════════════════════════════════


def train(resume_checkpoint: str | None = None) -> None:
    """
    End-to-end training orchestrator.

    Parameters
    ----------
    resume_checkpoint : str | None
        Path to a previous checkpoint directory.  If provided the
        trainer skips already-completed steps and continues.
    """
    # ── load everything ──
    tokenizer       = load_tokenizer()
    model           = load_base_model()
    model           = attach_lora(model)
    train_ds, val_ds = load_data()
    train_ds        = tokenize(train_ds, tokenizer)
    val_ds          = tokenize(val_ds,   tokenizer)

    # ── collator pads each batch to its own longest sequence
    #    (more memory-efficient than always padding to max_seq_length)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer = tokenizer,
        model     = model,
        padding   = True,
    )

    # ── trainer ──
    trainer = SFTTrainer(
        model         = model,
        args          = build_training_args(),
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = data_collator,
        tokenizer     = tokenizer,
    )

    # ── run ──
    log.info("═══ Training starting … ═══")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ── persist ──
    os.makedirs(PATHS["final_model_dir"], exist_ok=True)
    trainer.save_model(PATHS["final_model_dir"])
    tokenizer.save_pretrained(PATHS["final_model_dir"])

    log.info("═══ Training complete  →  model saved to %s ═══", PATHS["final_model_dir"])


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA SFT training for Mistral.")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Checkpoint directory to resume from  (e.g. output/checkpoint-200)",
    )
    args = parser.parse_args()
    train(resume_checkpoint=args.resume)
