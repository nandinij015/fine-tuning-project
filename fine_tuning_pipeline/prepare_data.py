
import argparse
import json
import logging
import os
import random
import sys

# ── project imports ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASET, PATHS

# ─────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)-7s]  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Alpaca prompt template  (must match train.py & inference.py)
# ─────────────────────────────────────────────────────────────
ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


# ═════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═════════════════════════════════════════════════════════════


def load_sample() -> list[dict]:
    """
    Load the bundled 10-row sample JSON that ships with the repo.
    No internet connection required.
    """
    path = DATASET["sample_file"]
    if not os.path.exists(path):
        log.error("Sample file not found: %s", path)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    log.info("Loaded %d rows from local sample (%s)", len(data), path)
    return data


def load_full_dataset() -> list[dict]:
    """
    Download the full dolly-15k dataset from HuggingFace Hub.
    Requires:  pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("'datasets' package not installed.  Run:  pip install datasets")
        sys.exit(1)

    hf_id = DATASET["hf_id"]
    log.info("Downloading full dataset from HuggingFace … (%s)", hf_id)
    ds = load_dataset(hf_id)

    # The dataset returns a DatasetDict; the training split has all rows
    rows = [dict(row) for row in ds["train"]]
    log.info("Downloaded %d rows", len(rows))
    return rows


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────


def validate_row(row: dict, idx: int) -> bool:
    """
    Return True if the row has the minimum required fields.
    Log a warning and return False otherwise.
    """
    instruction = (row.get("instruction") or "").strip()
    output      = (row.get("output")      or "").strip()

    if not instruction:
        log.warning("Row %d skipped — empty 'instruction'", idx)
        return False
    if not output:
        log.warning("Row %d skipped — empty 'output'", idx)
        return False
    return True


 def validate_token_length(row: dict, tokenizer, max_len: int) -> bool:
    """Return False if sample exceeds max length after tokenization."""
    tokens = tokenizer(row["text"], truncation=False)
    if len(tokens["input_ids"]) > max_len:
        log.warning(
            "Sample exceeds max_len=%d (actual=%d) - will be truncated",
            max_len, len(tokens["input_ids"])
        )
        return False  # or True if you want to keep it
    return True


# ─────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────


def format_row(row: dict) -> dict:
    """
    Produce the final dict that train.py will tokenise.

    Keys
    ────
    instruction : str   – the user's instruction
    input       : str   – optional context (empty string if absent)
    output      : str   – the expected model response
    text        : str   – the full Alpaca-formatted prompt + response
    """
    instruction = row["instruction"].strip()
    inp         = (row.get("input") or "").strip()
    output      = row["output"].strip()

    text = ALPACA_TEMPLATE.format(
        instruction=instruction,
        input=inp,
        output=output,
    )

    return {
        "instruction": instruction,
        "input"      : inp,
        "output"     : output,
        "text"       : text,
    }


# ─────────────────────────────────────────────────────────────
# Split
# ─────────────────────────────────────────────────────────────


def split_data(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Deterministic shuffle + split into train and validation sets.
    Ratios are controlled by DATASET["train_split"] in config.py.
    """
    random.seed(DATASET["seed"])
    random.shuffle(rows)

    cut   = int(len(rows) * DATASET["train_split"])
    train = rows[:cut]
    val   = rows[cut:]

    log.info("Split  →  train: %d  |  val: %d", len(train), len(val))
    return train, val


# ─────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────


def print_stats(rows: list[dict], label: str) -> None:
    """Print quick descriptive statistics for a set of samples."""
    lengths = [len(r["text"].split()) for r in rows]
    avg     = sum(lengths) / len(lengths) if lengths else 0

    log.info(
        "%s stats  →  count: %d  |  avg words: %.0f  |  min: %d  |  max: %d",
        label, len(rows), avg, min(lengths), max(lengths),
    )
    # Show the first sample so you can eyeball the format
    log.info("First %s sample:\n%s", label.lower(), rows[0]["text"][:320])


# ─────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────


def save(train: list[dict], val: list[dict]) -> None:
    """Write train.json and val.json to the data directory."""
    data_dir = PATHS["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    for name, rows in [("train", train), ("val", val)]:
        path = os.path.join(data_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
        log.info("Saved  →  %s  (%d rows)", path, len(rows))


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════


def main(use_full: bool = False) -> None:
    """
    Orchestrate the entire data-prep pipeline.

    Parameters
    ----------
    use_full : bool
        If True, download the full 15 k dataset from HuggingFace.
        If False (default), use the bundled 10-row sample.
    """
    log.info("═══ Data Preparation  (full=%s) ═══", use_full)

    # 1. Load
    raw = load_full_dataset() if use_full else load_sample()

    # 2. Validate + format
    formatted, skipped = [], 0
    for i, row in enumerate(raw):
        if not validate_row(row, i):
            skipped += 1
            continue
        formatted.append(format_row(row))

    log.info("Validated  →  kept: %d  |  skipped: %d", len(formatted), skipped)

    # 3. Split
    train, val = split_data(formatted)

    # 4. Stats
    print_stats(train, "TRAIN")
    print_stats(val,   "VAL")

    # 5. Save
    save(train, val)

    log.info("═══ Data preparation complete — ready for train.py ═══")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for fine-tuning.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download the full 15 k dataset instead of using the 10-row sample.",
    )
    args = parser.parse_args()
    main(use_full=args.full)
