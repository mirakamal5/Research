"""
GSM8K task adapter for Stage A.

Strategy:
  - Chain-of-thought prompt; model writes reasoning then "The answer is: N".
  - Parser (4 layers): "the [final] answer is N" → "Answer: N" line →
    "#### N" marker → bare-number last line → "unknown".
  - Score: exact numeric match after normalization (strip commas, $, %, trailing .0).
"""

import re
import logging

import pandas as pd
from datasets import load_dataset

log = logging.getLogger(__name__)

MAX_NEW_TOKENS = 512
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
            "Solve this math problem step by step.\n"
            "After your reasoning, write your final answer on its own line "
            "in EXACTLY this format:\n"
            "The answer is: <number>\n\n"
            "Do not write anything after that line.\n\n"
            f"Problem: {text}"
        ),
    }]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_output(raw: str) -> str:
    # P1: "The answer is: N" or "The final answer is: N" (case-insensitive, colon optional)
    m = re.search(r"[Tt]he (?:final )?answer is:?\s*(-?[\d,]+(?:\.\d+)?)", raw)
    if m:
        return m.group(1).replace(",", "")

    # P2: "Answer: N" at the start of a line
    m = re.search(r"(?m)^[Aa]nswer:?\s*(-?[\d,]+(?:\.\d+)?)\s*$", raw)
    if m:
        return m.group(1).replace(",", "")

    # P3: GSM8K ground-truth marker "#### N" (model occasionally echoes it)
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", raw)
    if m:
        return m.group(1).replace(",", "")

    # P4: last line of output is a bare number (and nothing else)
    last_line = raw.strip().rsplit("\n", 1)[-1].strip()
    if re.fullmatch(r"-?[\d,]+(?:\.\d+)?", last_line):
        return last_line.replace(",", "")

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
