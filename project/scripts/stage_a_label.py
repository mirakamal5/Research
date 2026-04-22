"""
Stage A — Part 2: LLM Inference, Scoring, and Labeling  (Person 2)
===================================================================
Reads:   data/interim/stage_a_base.csv          (Person 1 output)
Writes:  data/processed/stage_a_final.csv       (Person 2 output)

Designed to run on Google Colab free-tier T4 GPU.
Do NOT run locally — requires ~5 GB VRAM for 4-bit Mistral-7B.

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
FAILURE_THRESHOLD = 0.20
SEED              = 42

DEFAULT_INPUT  = "data/interim/stage_a_base.csv"
DEFAULT_OUTPUT = "data/processed/stage_a_final.csv"

DEBUG_PRINT_N = 5   # print raw model output for first N clean sentences

REQUIRED_COLUMNS = [
    "dataset", "sample_id", "clean_text", "noisy_text", "label",
    "noise_type", "severity", "word_count", "token_count",
    "sigma_prime", "alpha", "oov_rate",
]

FINAL_COLUMN_ORDER = REQUIRED_COLUMNS + [
    "clean_pred", "noisy_pred",
    "clean_score", "noisy_score",
    "delta_p", "failure_label",
]


# ---------------------------------------------------------------------------
# Task 2 — Load Mistral-7B with 4-bit NF4 quantization
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    log.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model in 4-bit NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    log.info("Model ready.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Task 3 — SST-2 zero-shot prompt
# ---------------------------------------------------------------------------

def build_prompt(text: str, tokenizer: AutoTokenizer) -> str:
    """
    Use tokenizer.apply_chat_template() instead of manually writing
    [INST]...[/INST].

    WHY: Mistral-7B-Instruct requires a BOS token <s> at the very start
    of the sequence. Manually writing "[INST]...[/INST]" skips this token,
    so the model does not recognize it as a chat instruction and outputs
    random/irrelevant text. apply_chat_template() adds <s> automatically.
    """
    messages = [
        {
            "role": "user",
            "content": (
                "Classify the sentiment of the following sentence.\n"
                "Reply with exactly one word: positive or negative.\n\n"
                f"Sentence: {text}"
            )
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Tasks 4 & 5 — Single inference call
# ---------------------------------------------------------------------------

def run_inference(text: str, model, tokenizer, max_new_tokens: int = 16) -> str:
    prompt = build_prompt(text, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Prediction normalization
# ---------------------------------------------------------------------------

def extract_sentiment(raw: str) -> str:
    """
    Normalize raw output to 'positive', 'negative', or 'unknown'.
    Handles: "Positive." / "POSITIVE" / "The sentiment is positive."
    """
    text = raw.lower()
    if re.search(r"\bpositive\b", text):
        return "positive"
    if re.search(r"\bnegative\b", text):
        return "negative"
    return "unknown"


# ---------------------------------------------------------------------------
# Task 6 — SST-2 scoring
# ---------------------------------------------------------------------------

def score_prediction(pred: str, label) -> float:
    """SST-2: 0 = negative, 1 = positive. Returns 1.0 if correct, else 0.0."""
    label_map = {0: "negative", 1: "positive", "0": "negative", "1": "positive"}
    expected = label_map.get(label, str(label).strip().lower())
    return 1.0 if pred == expected else 0.0


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def _debug_print(texts: list, model, tokenizer, n: int = DEBUG_PRINT_N):
    """
    Print raw model output for the first n sentences before the main loop.
    Check that PARSED shows 'positive' or 'negative'.
    If it shows 'unknown', the model is not responding correctly — stop.
    """
    log.info(f"=== DEBUG: raw model output for first {n} clean sentences ===")
    for i, text in enumerate(texts[:n]):
        raw  = run_inference(text, model, tokenizer)
        pred = extract_sentiment(raw)
        print(f"  [{i}] INPUT : {text[:80]}")
        print(f"  [{i}] RAW   : {repr(raw)}")
        print(f"  [{i}] PARSED: {pred}")
        print()
    log.info("=== If PARSED shows positive/negative above, model is working correctly ===")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_path: str, output_path: str):

    # Task 1 — Load
    log.info(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns from Person 1: {missing}")
    log.info(f"Loaded {len(df)} rows, {df['sample_id'].nunique()} unique samples.")

    # Task 2 — Load model
    model, tokenizer = load_model_and_tokenizer()

    # Unique clean sentences (deduplicated — no need to run LLM 18x per sample)
    unique_clean = (
        df[["sample_id", "clean_text", "label"]]
        .drop_duplicates(subset="sample_id")
        .reset_index(drop=True)
    )

    # Debug check before running full inference
    _debug_print(unique_clean["clean_text"].tolist(), model, tokenizer)

    # Clean inference
    log.info(f"Running clean inference on {len(unique_clean)} unique sentences...")
    clean_cache = {}
    for _, row in tqdm(unique_clean.iterrows(), total=len(unique_clean), desc="Clean inference"):
        raw   = run_inference(row["clean_text"], model, tokenizer)
        pred  = extract_sentiment(raw)
        score = score_prediction(pred, row["label"])
        clean_cache[row["sample_id"]] = (pred, score)

    clean_acc = sum(v[1] for v in clean_cache.values()) / len(clean_cache)
    log.info(f"Clean accuracy: {clean_acc:.3f}  (expected ~0.90+ for Mistral on SST-2)")
    if clean_acc < 0.70:
        log.warning(
            f"Clean accuracy is only {clean_acc:.3f} — something may be wrong. "
            "Check the DEBUG output above."
        )

    # Noisy inference
    log.info(f"Running noisy inference on {len(df)} rows...")
    noisy_preds, noisy_scores = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Noisy inference"):
        raw   = run_inference(row["noisy_text"], model, tokenizer)
        pred  = extract_sentiment(raw)
        score = score_prediction(pred, row["label"])
        noisy_preds.append(pred)
        noisy_scores.append(score)

    # Attach results
    df["clean_pred"]  = df["sample_id"].map(lambda sid: clean_cache[sid][0])
    df["clean_score"] = df["sample_id"].map(lambda sid: clean_cache[sid][1])
    df["noisy_pred"]  = noisy_preds
    df["noisy_score"] = noisy_scores

    # Task 7 — delta_p
    df["delta_p"] = df["clean_score"] - df["noisy_score"]

    # Task 8 — failure_label
    df["failure_label"] = (df["delta_p"] > FAILURE_THRESHOLD).astype(int)

    # Validate
    _validate(df)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[FINAL_COLUMN_ORDER].to_csv(output_path, index=False)
    log.info(f"Saved: {output_path}  ({len(df)} rows x {len(FINAL_COLUMN_ORDER)} cols)")

    n_fail  = int(df["failure_label"].sum())
    n_rob   = len(df) - n_fail
    n_unk_c = int((df["clean_pred"]  == "unknown").sum())
    n_unk_n = int((df["noisy_pred"] == "unknown").sum())
    log.info(f"FAILURE: {n_fail} ({100*n_fail/len(df):.1f}%)   ROBUST: {n_rob} ({100*n_rob/len(df):.1f}%)")
    log.info(f"Unknown predictions — clean: {n_unk_c}, noisy: {n_unk_n}")

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame):
    log.info("Validating outputs...")
    checks = [
        (df["clean_pred"].isnull().any(),           "clean_pred has nulls"),
        (df["noisy_pred"].isnull().any(),           "noisy_pred has nulls"),
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