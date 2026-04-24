"""
GSM8K task adapter for Stage A.

Strategy:
  - Chain-of-thought prompt; model writes reasoning then "The answer is: N".
  - Primary parser: regex "The answer is: <number>" on any line.
  - Fallback: last numeric token in the output.
  - Score: exact numeric match after normalization (strip commas, $, %, trailing .0).
"""

import re
import logging

import pandas as pd
from datasets import load_dataset

log = logging.getLogger(__name__)

MAX_NEW_TOKENS = 256
HAS_CONTEXT    = False


def load_clean(sample_size: int = 500, seed: int = 42) -> pd.DataFrame:
    log.info(f"Loading GSM8K train split (sample_size={sample_size}, seed={seed})...")
    raw = load_dataset("gsm8k", "main", split="train", trust_remote_code=True)
    raw = raw.shuffle(seed=seed).select(range(sample_size))
    labels = [item["answer"].split("####")[-1].strip() for item in raw]
    return pd.DataFrame({
        "dataset":    "gsm8k",
        "sample_id":  list(range(sample_size)),
        "clean_text": raw["question"],
        "label":      labels,
    })


def build_prompt(text: str, tokenizer, context: str = None) -> str:
    messages = [{
        "role": "user",
        "content": (
            "Solve the following math problem step by step.\n"
            "On the last line of your response, write exactly: "
            "\"The answer is: [number]\"\n\n"
            f"Problem: {text}"
        ),
    }]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_output(raw: str) -> str:
    # Primary: "The answer is: <number>" (case-insensitive, optional colon)
    m = re.search(r"[Tt]he answer is:?\s*(-?[\d,]+(?:\.\d+)?)", raw)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last integer/decimal in output
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", raw)
    if nums:
        return nums[-1].replace(",", "")
    return "unknown"


def _normalize(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(round(val, 4))
    except (ValueError, OverflowError):
        return s


def score(pred: str, label) -> float:
    if pred == "unknown":
        return 0.0
    return 1.0 if _normalize(pred) == _normalize(str(label)) else 0.0
