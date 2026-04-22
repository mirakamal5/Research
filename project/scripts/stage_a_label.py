"""
Stage A — Part 2: LLM Inference, Scoring, and Labeling  (Person 2)
===================================================================
Reads:   data/interim/stage_a_base.csv          (Person 1 output)
Writes:  data/processed/stage_a_final.csv       (Person 2 output)

Designed to run on Google Colab free-tier T4 GPU.
Do NOT run locally — the MX450 has insufficient VRAM for Mistral-7B.

Setup (run once in Colab):
    !pip install transformers accelerate bitsandbytes pandas tqdm

Usage:
    python scripts/stage_a_label.py
    python scripts/stage_a_label.py --input  data/interim/stage_a_base.csv \
                                     --output data/processed/stage_a_final.csv
"""

import argparse
import logging
import os
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME        = "mistralai/Mistral-7B-Instruct-v0.2"
FAILURE_THRESHOLD = 0.20   # delta_p > 0.20  -> failure_label = 1
SEED              = 42

DEFAULT_INPUT  = "data/interim/stage_a_base.csv"
DEFAULT_OUTPUT = "data/processed/stage_a_final.csv"

# Columns that must already exist in stage_a_base.csv (Person 1 output)
REQUIRED_COLUMNS = [
    "dataset", "sample_id", "clean_text", "noisy_text", "label",
    "noise_type", "severity", "word_count", "token_count",
    "sigma_prime", "alpha", "oov_rate",
]

# Final column order matching the shared schema, plus debug columns
FINAL_COLUMN_ORDER = REQUIRED_COLUMNS + [
    "clean_raw", "clean_pred",
    "noisy_raw", "noisy_pred",
    "clean_score", "noisy_score",
    "delta_p", "failure_label",
]

# ---------------------------------------------------------------------------
# Task 2 — Load Mistral-7B with 4-bit NF4 quantization
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """
    Load Mistral-7B-Instruct v0.2 in 4-bit NF4 quantization.

    Why 4-bit NF4?
    - The model has 7 billion parameters. In full fp32 that is ~28 GB.
    - 4-bit NF4 reduces this to ~4-5 GB, fitting in a Colab T4 (16 GB VRAM).
    - double_quant applies a second quantization pass to the quantization
      constants themselves, saving an extra ~0.4 bits per parameter.
    - compute_dtype=float16 keeps matrix multiplications in fp16 for speed.

    device_map="auto" lets HuggingFace distribute layers across GPU / CPU
    automatically. On a T4 Colab the whole model fits on GPU.
    """
    log.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Mistral has no dedicated pad token; reuse eos_token to avoid warnings.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Configuring 4-bit NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    log.info("Loading model (this takes ~2-3 minutes on first run)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()  # disable dropout for deterministic outputs
    log.info("Model ready.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Task 3 — SST-2 zero-shot prompt
# ---------------------------------------------------------------------------

def build_prompt(text: str) -> str:
    """
    Force binary SST-2 output as a single digit:
    0 = negative
    1 = positive

    This is stricter than asking for 'positive' or 'negative' and greatly
    reduces free-form outputs like 'Neutral' or explanations.
    """
    instruction = (
        "You are doing binary sentiment classification for SST-2.\n"
        "Label the sentence with exactly one digit only.\n"
        "Return 0 for negative or 1 for positive.\n"
        "Do not explain.\n"
        "Do not write any word.\n"
        "Output only one character: 0 or 1.\n\n"
        f"Sentence: {text}"
    )
    return f"[INST] {instruction} [/INST]"


# ---------------------------------------------------------------------------
# Tasks 4 & 5 — Single inference call
# ---------------------------------------------------------------------------

def run_inference(text: str, model, tokenizer, max_new_tokens: int = 3) -> str:
    """
    Run one deterministic forward pass and return the raw generated string.

    max_new_tokens=3 is enough for a one-character label ('0' or '1') while
    leaving minimal room for extra explanation text.
    """
    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt tokens — decode only what the model generated.
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Prediction normalization
# ---------------------------------------------------------------------------

def extract_sentiment(raw: str) -> str:
    """
    Normalize raw model output to exactly 'positive' or 'negative'.

    Preferred format:
        0 -> negative
        1 -> positive

    Fallbacks:
        positive / negative words

    Returns 'unknown' if neither format is found. Unknown predictions are
    treated as incorrect during scoring (score = 0).
    """
    text = raw.strip().lower()

    # Preferred strict path: standalone binary digit
    digit_match = re.search(r"\b([01])\b", text)
    if digit_match:
        return "positive" if digit_match.group(1) == "1" else "negative"

    # Fallback for older or verbose model behavior
    if re.search(r"\bpositive\b", text):
        return "positive"
    if re.search(r"\bnegative\b", text):
        return "negative"

    log.warning(f"Could not parse sentiment from: {repr(raw)}")
    return "unknown"


# ---------------------------------------------------------------------------
# Task 6 — SST-2 scoring
# ---------------------------------------------------------------------------

def score_prediction(pred: str, label) -> float:
    """
    Compare a normalized prediction against the ground-truth label.

    SST-2 integer labels:  0 = negative,  1 = positive
    Returns 1.0 if correct, 0.0 otherwise.
    """
    label_map = {0: "negative", 1: "positive", "0": "negative", "1": "positive"}
    expected = label_map.get(label, str(label).strip().lower())
    return 1.0 if pred == expected else 0.0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_path: str, output_path: str):
    """
    Full Person 2 pipeline — Tasks 1 through 10.
    """

    # ------------------------------------------------------------------
    # Task 1 — Load stage_a_base.csv
    # ------------------------------------------------------------------
    log.info(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"stage_a_base.csv is missing columns: {missing}\n"
            "Make sure you are using Person 1's validated output."
        )
    log.info(f"Loaded {len(df)} rows.")

    # ------------------------------------------------------------------
    # Task 2 — Load model
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer()

    # ------------------------------------------------------------------
    # Tasks 3–6 — Inference
    #
    # Optimization: each clean sentence appears in 18 rows (6 noise types
    # × 3 severities). We run LLM inference on each unique clean sentence
    # only ONCE, then broadcast the result to all 18 rows.
    # This reduces clean-text inference calls by 18×.
    # ------------------------------------------------------------------

    # --- Clean inference (unique sentences only) ---
    unique_clean = (
        df[["sample_id", "clean_text", "label"]]
        .drop_duplicates(subset="sample_id")
        .reset_index(drop=True)
    )
    log.info(
        f"Running inference on {len(unique_clean)} unique clean sentences "
        f"(covers all {len(df)} rows)..."
    )

    clean_cache = {}   # sample_id -> (raw, pred, score)
    for _, row in tqdm(unique_clean.iterrows(), total=len(unique_clean), desc="Clean inference"):
        raw = run_inference(row["clean_text"], model, tokenizer)
        pred = extract_sentiment(raw)
        score = score_prediction(pred, row["label"])
        clean_cache[row["sample_id"]] = (raw, pred, score)

    # --- Noisy inference (every row) ---
    log.info(f"Running inference on {len(df)} noisy texts...")
    noisy_raws, noisy_preds, noisy_scores = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Noisy inference"):
        raw = run_inference(row["noisy_text"], model, tokenizer)
        pred = extract_sentiment(raw)
        score = score_prediction(pred, row["label"])
        noisy_raws.append(raw)
        noisy_preds.append(pred)
        noisy_scores.append(score)

    # --- Attach results ---
    df["clean_raw"]   = df["sample_id"].map(lambda sid: clean_cache[sid][0])
    df["clean_pred"]  = df["sample_id"].map(lambda sid: clean_cache[sid][1])
    df["clean_score"] = df["sample_id"].map(lambda sid: clean_cache[sid][2])

    df["noisy_raw"]   = noisy_raws
    df["noisy_pred"]  = noisy_preds
    df["noisy_score"] = noisy_scores

    # ------------------------------------------------------------------
    # Task 7 — delta_p  (paper Equation 1)
    # delta_p = P_clean - P_noisy
    # For SST-2: both scores are binary, so delta_p ∈ {-1, 0, 1}
    # ------------------------------------------------------------------
    df["delta_p"] = df["clean_score"] - df["noisy_score"]

    # ------------------------------------------------------------------
    # Task 8 — failure_label  (paper §V.E)
    # failure_label = 1 if delta_p > 0.20 else 0
    # For SST-2 binary scores this fires when:
    #   clean correct (1) and noisy wrong (0)  ->  delta_p = 1.0  -> FAILURE
    # ------------------------------------------------------------------
    df["failure_label"] = (df["delta_p"] > FAILURE_THRESHOLD).astype(int)

    # ------------------------------------------------------------------
    # Task 9 — Validate
    # ------------------------------------------------------------------
    _validate(df)

    # ------------------------------------------------------------------
    # Task 10 — Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[FINAL_COLUMN_ORDER].to_csv(output_path, index=False)
    log.info(f"Saved: {output_path}  ({len(df)} rows)")

    # Summary
    n_fail = int(df["failure_label"].sum())
    n_rob  = len(df) - n_fail
    pct    = 100 * n_fail / len(df)
    log.info(f"FAILURE: {n_fail} ({pct:.1f}%)   ROBUST: {n_rob} ({100-pct:.1f}%)")

    n_unk_c = int((df["clean_pred"] == "unknown").sum())
    n_unk_n = int((df["noisy_pred"] == "unknown").sum())
    if n_unk_c or n_unk_n:
        log.warning(f"Unknown predictions — clean: {n_unk_c}, noisy: {n_unk_n}")

    return df


# ---------------------------------------------------------------------------
# Validation (Task 9)
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame):
    log.info("Validating outputs...")
    checks = [
        (df["clean_raw"].isnull().any(),             "clean_raw has nulls"),
        (df["noisy_raw"].isnull().any(),             "noisy_raw has nulls"),
        (df["clean_pred"].isnull().any(),            "clean_pred has nulls"),
        (df["noisy_pred"].isnull().any(),            "noisy_pred has nulls"),
        (~df["clean_score"].isin([0.0, 1.0]).all(), "clean_score outside {0,1}"),
        (~df["noisy_score"].isin([0.0, 1.0]).all(), "noisy_score outside {0,1}"),
        (~pd.api.types.is_numeric_dtype(df["delta_p"]), "delta_p not numeric"),
        (~df["failure_label"].isin([0, 1]).all(),   "failure_label outside {0,1}"),
    ]
    for condition, message in checks:
        if condition:
            log.warning(f"Validation warning: {message}")
    log.info("Validation done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Person 2 — Stage A: LLM inference + labeling for SST-2"
    )
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()