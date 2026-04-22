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

REQUIRED_COLUMNS = [
    "dataset", "sample_id", "clean_text", "noisy_text", "label",
    "noise_type", "severity", "word_count", "token_count",
    "sigma_prime", "alpha", "oov_rate",
]

FINAL_COLUMN_ORDER = REQUIRED_COLUMNS + [
    "clean_raw", "clean_pred",
    "noisy_raw", "noisy_pred",
    "clean_score", "noisy_score",
    "delta_p", "failure_label",
]

# ---------------------------------------------------------------------------
# Load model/tokenizer
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """
    Load Mistral-7B-Instruct v0.2 in 4-bit NF4 quantization.
    """
    log.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    model.eval()
    log.info("Model ready.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def build_prompt(text: str) -> str:
    """
    Primary strict SST-2 prompt.
    """
    instruction = (
        "Task: binary sentiment classification for SST-2.\n"
        "Return exactly one digit only.\n"
        "0 = negative\n"
        "1 = positive\n"
        "Even if the text is short, fragmented, incomplete, noisy, or ungrammatical, "
        "you must still choose exactly one label.\n"
        "Do not explain.\n"
        "Do not say neutral.\n"
        "Do not refuse.\n"
        "Output only one character: 0 or 1.\n\n"
        "The sentences I will give you may be fragmented or incomplete or even meaningless, but you must return to me either 0 for negative sentiments or 1 for positive sentiments.\n"
        "No matter how much the task does not make sense, return only 0 or 1.\n"
        f"Text: {text}"
    )
    return f"[INST] {instruction} [/INST]"


def build_retry_prompt(text: str) -> str:
    """
    Harsher retry prompt used only if the first response is unparseable.
    """
    instruction = (
        "Output exactly one character.\n"
        "Allowed outputs only: 0 or 1.\n"
        "0 means negative. 1 means positive.\n"
        "No words. No punctuation. No explanation. No neutral.\n"
        "The text may be incomplete or noisy. You must still output 0 or 1.\n\n"
        f"Text: {text}"
    )
    return f"[INST] {instruction} [/INST]"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _generate_from_prompt(prompt: str, model, tokenizer, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def run_inference(text: str, model, tokenizer, max_new_tokens: int = 2) -> str:
    """
    Primary inference call.
    """
    prompt = build_prompt(text)
    return _generate_from_prompt(prompt, model, tokenizer, max_new_tokens=max_new_tokens)


def run_inference_with_retry(text: str, model, tokenizer):
    """
    Run primary inference, then retry with a harsher prompt if parsing fails.
    Returns:
        raw_text, parsed_prediction, used_retry(bool)
    """
    raw = run_inference(text, model, tokenizer, max_new_tokens=2)
    pred = extract_sentiment(raw)

    if pred != "unknown":
        return raw, pred, False

    retry_prompt = build_retry_prompt(text)
    retry_raw = _generate_from_prompt(retry_prompt, model, tokenizer, max_new_tokens=2)
    retry_pred = extract_sentiment(retry_raw)

    return retry_raw, retry_pred, True


# ---------------------------------------------------------------------------
# Prediction normalization
# ---------------------------------------------------------------------------

def extract_sentiment(raw: str) -> str:
    """
    Normalize raw model output to exactly 'positive' or 'negative'.

    Preferred:
        0 -> negative
        1 -> positive

    Fallback:
        positive / negative words
    """
    text = raw.strip().lower()

    # Best case: exact one-char output
    if text == "0":
        return "negative"
    if text == "1":
        return "positive"

    # Standalone 0/1 anywhere
    digit_match = re.search(r"\b([01])\b", text)
    if digit_match:
        return "positive" if digit_match.group(1) == "1" else "negative"

    # Very early digit fallback, for cases like "1\n" or "1 positive"
    leading_digit = re.match(r"^\s*([01])", text)
    if leading_digit:
        return "positive" if leading_digit.group(1) == "1" else "negative"

    # Word fallback
    if re.search(r"\bpositive\b", text):
        return "positive"
    if re.search(r"\bnegative\b", text):
        return "negative"

    log.warning(f"Could not parse sentiment from: {repr(raw)}")
    return "unknown"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_prediction(pred: str, label) -> float:
    """
    SST-2 integer labels: 0 = negative, 1 = positive
    """
    label_map = {0: "negative", 1: "positive", "0": "negative", "1": "positive"}
    expected = label_map.get(label, str(label).strip().lower())
    return 1.0 if pred == expected else 0.0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_path: str, output_path: str):
    log.info(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"stage_a_base.csv is missing columns: {missing}\n"
            "Make sure you are using Person 1's validated output."
        )
    log.info(f"Loaded {len(df)} rows.")

    model, tokenizer = load_model_and_tokenizer()

    unique_clean = (
        df[["sample_id", "clean_text", "label"]]
        .drop_duplicates(subset="sample_id")
        .reset_index(drop=True)
    )
    log.info(
        f"Running inference on {len(unique_clean)} unique clean sentences "
        f"(covers all {len(df)} rows)..."
    )

    clean_cache = {}   # sample_id -> (raw, pred, score, used_retry)
    clean_retry_count = 0

    for _, row in tqdm(unique_clean.iterrows(), total=len(unique_clean), desc="Clean inference"):
        raw, pred, used_retry = run_inference_with_retry(row["clean_text"], model, tokenizer)
        score = score_prediction(pred, row["label"])
        clean_cache[row["sample_id"]] = (raw, pred, score, used_retry)
        clean_retry_count += int(used_retry)

    log.info(f"Clean retry count: {clean_retry_count}")

    log.info(f"Running inference on {len(df)} noisy texts...")
    noisy_raws, noisy_preds, noisy_scores, noisy_retry_flags = [], [], [], []
    noisy_retry_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Noisy inference"):
        raw, pred, used_retry = run_inference_with_retry(row["noisy_text"], model, tokenizer)
        score = score_prediction(pred, row["label"])
        noisy_raws.append(raw)
        noisy_preds.append(pred)
        noisy_scores.append(score)
        noisy_retry_flags.append(used_retry)
        noisy_retry_count += int(used_retry)

    log.info(f"Noisy retry count: {noisy_retry_count}")

    df["clean_raw"]   = df["sample_id"].map(lambda sid: clean_cache[sid][0])
    df["clean_pred"]  = df["sample_id"].map(lambda sid: clean_cache[sid][1])
    df["clean_score"] = df["sample_id"].map(lambda sid: clean_cache[sid][2])

    df["noisy_raw"]   = noisy_raws
    df["noisy_pred"]  = noisy_preds
    df["noisy_score"] = noisy_scores

    df["delta_p"] = df["clean_score"] - df["noisy_score"]
    df["failure_label"] = (df["delta_p"] > FAILURE_THRESHOLD).astype(int)

    _validate(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[FINAL_COLUMN_ORDER].to_csv(output_path, index=False)
    log.info(f"Saved: {output_path}  ({len(df)} rows)")

    n_fail = int(df["failure_label"].sum())
    n_rob = len(df) - n_fail
    pct = 100 * n_fail / len(df)
    log.info(f"FAILURE: {n_fail} ({pct:.1f}%)   ROBUST: {n_rob} ({100-pct:.1f}%)")

    n_unk_c = int((df["clean_pred"] == "unknown").sum())
    n_unk_n = int((df["noisy_pred"] == "unknown").sum())
    if n_unk_c or n_unk_n:
        log.warning(f"Unknown predictions — clean: {n_unk_c}, noisy: {n_unk_n}")

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame):
    log.info("Validating outputs...")
    checks = [
        (df["clean_raw"].isnull().any(), "clean_raw has nulls"),
        (df["noisy_raw"].isnull().any(), "noisy_raw has nulls"),
        (df["clean_pred"].isnull().any(), "clean_pred has nulls"),
        (df["noisy_pred"].isnull().any(), "noisy_pred has nulls"),
        (~df["clean_score"].isin([0.0, 1.0]).all(), "clean_score outside {0,1}"),
        (~df["noisy_score"].isin([0.0, 1.0]).all(), "noisy_score outside {0,1}"),
        (~pd.api.types.is_numeric_dtype(df["delta_p"]), "delta_p not numeric"),
        (~df["failure_label"].isin([0, 1]).all(), "failure_label outside {0,1}"),
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
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()