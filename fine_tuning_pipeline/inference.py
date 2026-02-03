"""
inference.py
═══════════════════════════════════════════════════════════════
 Stage 4 / 4 — Inference & Response Generation
═══════════════════════════════════════════════════════════════

 What this script does
 ─────────────────────
 Loads the fine-tuned model and generates responses.  Three modes:

   interactive  – REPL chat loop (default when no flags are given)
   single       – one-shot prompt via --prompt "…"
   batch        – read prompts from a text file via --batch file.txt

 Pipeline position
 ─────────────────
 Your workflow:  … → AGENT MODELS → FINAL OUTPUT

 Prerequisites
 ─────────────
     python train.py   # output/final_model/ must exist

 Usage
 ─────
     python inference.py                          # interactive
     python inference.py --prompt "Explain LoRA"  # single prompt
     python inference.py --batch prompts.txt      # batch mode

 Last updated : 2026-02-03
 Author       : nandinij015
═══════════════════════════════════════════════════════════════
"""

import argparse
import logging
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft         import PeftModel

# ── project imports ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL, QUANTIZATION, TRAINING, INFERENCE, PATHS

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
# PROMPT TEMPLATE
# ═════════════════════════════════════════════════════════════
# Must be identical to the template used during training
# (defined in prepare_data.py).  The "### Response:\n" line
# is left without an answer — the model generates it.

ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def build_prompt(instruction: str, inp: str = "") -> str:
    """Format an instruction (+ optional input) into the Alpaca template."""
    return ALPACA_TEMPLATE.format(instruction=instruction, input=inp)


# ═════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════


def load_model_and_tokenizer():
    """
    Reconstruct the 4-bit base model and overlay the saved LoRA
    adapter.  The result is the same model that was checkpointed
    at the end of train.py.
    """
    dtype_map     = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    compute_dtype = dtype_map.get(QUANTIZATION["compute_dtype"], torch.bfloat16)

    # ── tokeniser ──
    log.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL["base_model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── base (4-bit) ──
    log.info("Loading base model …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = QUANTIZATION["load_in_4bit"],
        bnb_4bit_quant_type       = QUANTIZATION["quant_type"],
        bnb_4bit_compute_dtype    = compute_dtype,
        bnb_4bit_use_double_quant = QUANTIZATION["use_double_quant"],
    )

    base = AutoModelForCausalLM.from_pretrained(
        MODEL["base_model_name"],
        quantization_config = bnb_config,
        device_map          = "auto",
        torch_dtype         = compute_dtype,
    )

    # ── LoRA overlay ──
    log.info("Loading LoRA adapter from %s …", PATHS["final_model_dir"])
    model = PeftModel.from_pretrained(base, PATHS["final_model_dir"])
    model.eval()

    log.info("Model ready.")
    return model, tokenizer


# ═════════════════════════════════════════════════════════════
# GENERATION
# ═════════════════════════════════════════════════════════════


def generate(model, tokenizer, prompt: str, max_new_tokens: int | None = None) -> str:
    """
    Run inference on a single formatted prompt and return the
    generated text (new tokens only — the prompt is stripped).

    Decoding strategy
    ─────────────────
    • temperature      – controls randomness (0 = greedy, 1 = full random)
    • top_p            – nucleus sampling: only consider tokens in the top-p
                         probability mass
    • repetition_penalty – penalises repeated n-grams to avoid loops
    """
    if max_new_tokens is None:
        max_new_tokens = INFERENCE["max_new_tokens"]

    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=TRAINING["max_seq_length"])
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens     = max_new_tokens,
            do_sample          = True,
            temperature        = INFERENCE["temperature"],
            top_p              = INFERENCE["top_p"],
            repetition_penalty = INFERENCE["repetition_penalty"],
            pad_token_id       = tokenizer.pad_token_id,
        )

    # Decode only the tokens AFTER the input prompt
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ═════════════════════════════════════════════════════════════
# RUN MODES
# ═════════════════════════════════════════════════════════════


def interactive_mode(model, tokenizer) -> None:
    """
    REPL loop.  The user types an instruction (and optionally an
    input context).  The model responds.  Type 'quit' to exit.
    """
    print("\n" + "═" * 56)
    print("  🤖  Fine-tuned Mistral — Interactive Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("═" * 56 + "\n")

    while True:
        instruction = input("You (instruction): ").strip()
        if instruction.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        inp = input("You (input — or press Enter to skip): ").strip()

        prompt   = build_prompt(instruction, inp)
        response = generate(model, tokenizer, prompt)

        print(f"\n🤖 Response:\n{response}\n")
        print("─" * 56)


def single_prompt_mode(model, tokenizer, instruction: str) -> None:
    """Run one prompt and print the response."""
    prompt   = build_prompt(instruction)
    response = generate(model, tokenizer, prompt)
    print(f"\n🤖 Response:\n{response}\n")


def batch_mode(model, tokenizer, file_path: str) -> None:
    """
    Read prompts from a plain-text file (one instruction per line),
    run them sequentially, and print each response.
    """
    if not os.path.exists(file_path):
        log.error("Batch file not found: %s", file_path)
        sys.exit(1)

    with open(file_path) as f:
        instructions = [line.strip() for line in f if line.strip()]

    log.info("Batch mode  →  %d prompts", len(instructions))

    for i, instr in enumerate(instructions, 1):
        prompt   = build_prompt(instr)
        response = generate(model, tokenizer, prompt)
        print(f"\n[{i}/{len(instructions)}]  Instruction : {instr}")
        print(f"              Response  : {response}")
        print("─" * 56)


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned model.")
    parser.add_argument("--prompt", type=str, default=None, help="Single instruction to run.")
    parser.add_argument("--batch",  type=str, default=None, help="Path to a file with one prompt per line.")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()

    if args.prompt:
        single_prompt_mode(model, tokenizer, args.prompt)
    elif args.batch:
        batch_mode(model, tokenizer, args.batch)
    else:
        interactive_mode(model, tokenizer)
