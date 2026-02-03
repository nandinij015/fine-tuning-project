"""
evaluate.py
═══════════════════════════════════════════════════════════════
 Stage 3 / 4 — Evaluation & Validation
═══════════════════════════════════════════════════════════════

 What this script does
 ─────────────────────
 1. Loads the fine-tuned LoRA model from output/final_model.
 2. Computes average cross-entropy loss over the validation set.
 3. Derives perplexity from the loss  (exp(loss)).
 4. Generates sample responses and prints them side-by-side with
    the ground-truth outputs so you can visually judge quality.
 5. Writes a summary report to example_outputs/eval_report.json.

 Pipeline position
 ─────────────────
 Your workflow:  … → POST-PROCESS & VALIDATE + MONITORING → …

 Prerequisites
 ─────────────
     python prepare_data.py
     python train.py              # must have completed at least one epoch

 Usage
 ─────
     python evaluate.py                       # default: 5 sample generations
     python evaluate.py --num_samples 20      # show 20 examples

 Last updated : 2026-02-03
 Author       : nandinij015
═══════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import math
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft         import PeftModel

# ── project imports ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config    import MODEL, QUANTIZATION, TRAINING, PATHS
from inference import generate, build_prompt          # reuse inference utilities

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
# 1.  MODEL LOADING
# ═════════════════════════════════════════════════════════════


def load_model_and_tokenizer():
    """
    Load base model + LoRA adapter in eval mode.
    Mirrors the loading pattern from train.py but skips the
    training-specific prepare_model_for_kbit_training() call.
    """
    dtype_map     = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    compute_dtype = dtype_map.get(QUANTIZATION["compute_dtype"], torch.bfloat16)

    # ── tokeniser ──
    tokenizer = AutoTokenizer.from_pretrained(MODEL["base_model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── base model (4-bit) ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = QUANTIZATION["load_in_4bit"],
        bnb_4bit_quant_type       = QUANTIZATION["quant_type"],
        bnb_4bit_compute_dtype    = compute_dtype,
        bnb_4bit_use_double_quant = QUANTIZATION["use_double_quant"],
    )

    log.info("Loading base model …")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL["base_model_name"],
        quantization_config = bnb_config,
        device_map          = "auto",
        torch_dtype         = compute_dtype,
    )

    # ── LoRA adapter ──
    log.info("Merging LoRA adapter from %s …", PATHS["final_model_dir"])
    model = PeftModel.from_pretrained(base, PATHS["final_model_dir"])
    model.eval()

    log.info("Model ready for evaluation.")
    return model, tokenizer


# ═════════════════════════════════════════════════════════════
# 2.  LOAD VALIDATION DATA
# ═════════════════════════════════════════════════════════════


def load_val_data() -> list[dict]:
    """Read the validation split produced by prepare_data.py."""
    path = os.path.join(PATHS["data_dir"], "val.json")
    if not os.path.exists(path):
        log.error("val.json not found at %s  →  run prepare_data.py first", path)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)
    log.info("Loaded %d validation samples", len(data))
    return data


# ═════════════════════════════════════════════════════════════
# 3.  LOSS  (per-sample, averaged)
# ═════════════════════════════════════════════════════════════


def compute_eval_loss(model, tokenizer, val_data: list[dict], max_samples: int = 100) -> float:
    """
    Forward-pass each validation sample through the model with
    labels = input_ids.  The model returns the cross-entropy loss
    automatically.  We average over all samples.

    We cap at max_samples to keep eval time reasonable when the
    validation set is large.

    Returns
    ───────
    float – average cross-entropy loss
    """
    total_loss, count = 0.0, 0
    max_len = TRAINING["max_seq_length"]

    for sample in val_data[:max_samples]:
        inputs    = tokenizer(sample["text"], return_tensors="pt",
                              truncation=True, max_length=max_len, padding=False)
        input_ids = inputs["input_ids"].to(model.device)
        labels    = input_ids.clone()                  # next-token prediction target

        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=labels).loss
            total_loss += loss.item()
            count      += 1

    avg_loss = total_loss / count if count else 0.0
    return avg_loss


# ═════════════════════════════════════════════════════════════
# 4.  SAMPLE GENERATIONS  (side-by-side)
# ═════════════════════════════════════════════════════════════


def run_generations(model, tokenizer, val_data: list[dict], n: int = 5) -> list[dict]:
    """
    For each of the first *n* validation samples:
      – build the Alpaca prompt (instruction + input, NO output)
      – call model.generate()
      – record ground_truth vs model_output

    Returns a list of dicts for the eval report.
    """
    log.info("Generating %d sample responses …", n)
    results = []

    for i, sample in enumerate(val_data[:n]):
        prompt   = build_prompt(sample["instruction"], sample.get("input", ""))
        response = generate(model, tokenizer, prompt, max_new_tokens=200)

        entry = {
            "id"           : i + 1,
            "instruction"  : sample["instruction"],
            "input"        : sample.get("input", ""),
            "ground_truth" : sample["output"],
            "model_output" : response,
        }
        results.append(entry)

        # Pretty-print to console
        log.info(
            "── Sample %d ──\n"
            "  Instruction  : %s\n"
            "  Ground Truth : %s\n"
            "  Model Output : %s",
            i + 1,
            sample["instruction"][:100],
            sample["output"][:150],
            response[:150],
        )

    return results


# ═════════════════════════════════════════════════════════════
# 5.  REPORT
# ═════════════════════════════════════════════════════════════


def save_report(metrics: dict, generations: list[dict]) -> str:
    """Write the full eval report to example_outputs/."""
    os.makedirs(PATHS["example_outputs"], exist_ok=True)

    report = {
        "metrics"    : metrics,
        "generations": generations,
    }

    path = os.path.join(PATHS["example_outputs"], "eval_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("Eval report saved  →  %s", path)
    return path


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════


def evaluate(num_samples: int = 5) -> None:
    """Full evaluation pipeline."""
    log.info("═══ Evaluation starting ═══")

    model, tokenizer = load_model_and_tokenizer()
    val_data         = load_val_data()

    # ── metrics ──
    log.info("Computing eval loss …")
    avg_loss   = compute_eval_loss(model, tokenizer, val_data)
    perplexity = math.exp(avg_loss)

    metrics = {
        "avg_eval_loss"        : round(avg_loss,   4),
        "perplexity"           : round(perplexity, 2),
        "num_eval_samples_used": min(100, len(val_data)),
    }

    log.info(
        "═══ Metrics ═══\n"
        "  Avg Eval Loss : %.4f\n"
        "  Perplexity    : %.2f\n"
        "  Samples used  : %d",
        metrics["avg_eval_loss"],
        metrics["perplexity"],
        metrics["num_eval_samples_used"],
    )

    # ── sample generations ──
    generations = run_generations(model, tokenizer, val_data, n=num_samples)

    # ── persist ──
    save_report(metrics, generations)
    log.info("═══ Evaluation complete ═══")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned model.")
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of sample generations to show (default 5).",
    )
    args = parser.parse_args()
    evaluate(num_samples=args.num_samples)
