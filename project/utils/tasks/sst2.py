"""
SST-2 task adapter for Stage A.

Prompt:  commitment-forcing with "Sentiment:" completion cue.
Parser:  last occurrence of \bpositive\b or \bnegative\b; else "unknown".
Score:   binary exact match (1.0 correct / 0.0 wrong or unknown).
"""

import re
import logging

import pandas as pd
from datasets import load_dataset

log = logging.getLogger(__name__)

MAX_NEW_TOKENS = 16
HAS_CONTEXT    = False


def load_clean(sample_size: int = 500, seed: int = 42) -> pd.DataFrame:
    log.info(f"Loading SST-2 train split (sample_size={sample_size}, seed={seed})...")
    raw = load_dataset("glue", "sst2", split="train", trust_remote_code=True)
    raw = raw.shuffle(seed=seed).select(range(sample_size))
    return pd.DataFrame({
        "dataset":    "sst2",
        "sample_id":  list(range(sample_size)),
        "clean_text": raw["sentence"],
        "label":      raw["label"],
    })


def build_prompt(text: str, tokenizer, context: str = None) -> str:
    messages = [{
        "role": "user",
        "content": (
            "Classify the sentiment of the following sentence.\n"
            "Reply with exactly one word: positive or negative.\n\n"
            f"Sentence: {text}\n\n"
            "Sentiment:"
        ),
    }]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_output(raw: str) -> str:
    text = raw.lower()
    pos_hits = [(m.start(), "positive") for m in re.finditer(r"\bpositive\b", text)]
    neg_hits = [(m.start(), "negative") for m in re.finditer(r"\bnegative\b", text)]
    all_hits = pos_hits + neg_hits
    if not all_hits:
        return "unknown"
    return max(all_hits, key=lambda x: x[0])[1]


def score(pred: str, label) -> float:
    if pred == "unknown":
        return 0.0
    label_map = {0: "negative", 1: "positive"}
    if isinstance(label, str):
        try:
            label = int(label)
        except ValueError:
            pass
    expected = label_map.get(label, str(label).strip().lower())
    return 1.0 if pred == expected else 0.0
