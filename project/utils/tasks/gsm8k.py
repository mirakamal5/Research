"""
GSM8K task adapter for Stage A.

Strategy:
  - Chain-of-thought prompt; model writes reasoning then "#### N".
  - Parser (4 layers): "#### N" → "the [final] answer is N" (last hit) →
    "Answer: N" line → bare-number last line → "unknown".
  - Score: exact numeric match after normalization (strip commas, $, %, trailing .0).
"""

import re
import logging

import pandas as pd
from datasets import load_dataset

log = logging.getLogger(__name__)

MAX_NEW_TOKENS = 128
HAS_CONTEXT    = False


def load_clean(sample_size: int = 500, seed: int = 42, offset: int = 0) -> pd.DataFrame:
    log.info(f"Loading GSM8K train split (sample_size={sample_size}, offset={offset}, seed={seed})...")
    raw = load_dataset("gsm8k", "main", split="train", trust_remote_code=True)
    raw = raw.shuffle(seed=seed).select(range(offset, offset + sample_size))
    labels = [item["answer"].split("####")[-1].strip() for item in raw]
    return pd.DataFrame({
        "dataset":    "gsm8k",
        "sample_id":  list(range(offset, offset + sample_size)),
        "clean_text": raw["question"],
        "label":      labels,
    })


def build_prompt(text: str, tokenizer, context: str = None) -> str:
    messages = [{
        "role": "user",
        "content": (
            "Solve this math problem. Use at most 3 short lines of working.\n"
            "Write the final answer on its own line as: #### [number]\n\n"
            f"Problem: {text}"
        ),
    }]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_output(raw: str) -> str:
    # P1: "#### N" — GSM8K-native format, last occurrence wins
    hits = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", raw)
    if hits:
        return hits[-1].replace(",", "")

    # P2: "the [final] answer is N" — last occurrence wins (guards against front-loaded guesses)
    hits = re.findall(r"[Tt]he (?:final )?answer is:?\s*(-?[\d,]+(?:\.\d+)?)", raw)
    if hits:
        return hits[-1].replace(",", "")

    # P3: "Answer: N" alone on a line
    m = re.search(r"(?m)^[Aa]nswer:?\s*(-?[\d,]+(?:\.\d+)?)\s*$", raw)
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
