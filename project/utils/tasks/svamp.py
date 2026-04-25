"""
SVAMP task adapter for Stage A.

Dataset: ChilleD/SVAMP (700 train samples, simple arithmetic word problems).
Full problem text comes from the pre-combined `question_concat` field.
Answer is a numeric string (integer or decimal).

Strategy matches GSM8K: chain-of-thought prompt, #### answer line.
Parser (4 layers): #### N → the [final] answer is N → Answer: N line →
bare-number last line → "unknown".
Score: exact numeric match after normalization.
"""

import re
import logging

import pandas as pd
from datasets import load_dataset

log = logging.getLogger(__name__)

MAX_NEW_TOKENS = 128
HAS_CONTEXT    = False


def load_clean(sample_size: int = 500, seed: int = 42, offset: int = 0) -> pd.DataFrame:
    log.info(f"Loading SVAMP train split (sample_size={sample_size}, offset={offset}, seed={seed})...")
    raw = load_dataset("ChilleD/SVAMP", split="train")
    if offset + sample_size > len(raw):
        log.warning(
            f"offset+sample_size ({offset + sample_size}) exceeds available "
            f"SVAMP examples ({len(raw)}). Capping."
        )
        sample_size = max(0, len(raw) - offset)
    raw = raw.shuffle(seed=seed).select(range(offset, offset + sample_size))
    return pd.DataFrame({
        "dataset":    "svamp",
        "sample_id":  list(range(offset, offset + sample_size)),
        "clean_text": raw["question_concat"],
        "label":      raw["Answer"],
    })


def build_prompt(text: str, tokenizer, context: str = None) -> str:
    messages = [{
        "role": "user",
        "content": (
            "Solve this math problem. Use concise reasoning if needed.\n"
            "End your response with the final numeric answer in this exact format:\n"
            "#### [number]\n"
            "Do not write anything after the number.\n\n"
            f"Problem: {text}"
        ),
    }]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_output(raw: str) -> str:
    # P1: "#### N" — last occurrence wins
    hits = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", raw)
    if hits:
        return hits[-1].replace(",", "")

    # P2: "the [final] answer is N" — last occurrence wins
    hits = re.findall(r"[Tt]he (?:final )?answer is:?\s*(-?[\d,]+(?:\.\d+)?)", raw)
    if hits:
        return hits[-1].replace(",", "")

    # P3: "Answer: N" alone on a line
    m = re.search(r"(?m)^[Aa]nswer:?\s*(-?[\d,]+(?:\.\d+)?)\s*$", raw)
    if m:
        return m.group(1).replace(",", "")

    # P4: last line of output is a bare number
    last_line = raw.strip().rsplit("\n", 1)[-1].strip()
    if re.fullmatch(r"-?[\d,]+(?:\.\d+)?", last_line):
        return last_line.replace(",", "")

    # P5: last number anywhere in the output (ultimate fallback)
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
