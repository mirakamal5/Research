"""
SQuAD v2 task adapter for Stage A.

Strategy:
  - Noise the question only; context (passage) is carried unchanged.
  - 500 answerable-only questions from the validation split.
  - Prompt asks for a short phrase from the passage ("Answer:" cue).
  - Parser: strip whitespace; flag >150 chars as "unknown".
  - Score: normalized token-level F1, max over all gold answers.
  - 'context' column lives in the base CSV only; it is dropped before
    the final CSV is saved (not part of the shared schema).
"""

import json
import re
import logging
from collections import Counter

import pandas as pd
from datasets import load_dataset

log = logging.getLogger(__name__)

MAX_NEW_TOKENS = 64
HAS_CONTEXT    = True


def load_clean(sample_size: int = 500, seed: int = 42) -> pd.DataFrame:
    log.info(f"Loading SQuAD v2 validation split (sample_size={sample_size}, seed={seed})...")
    raw = load_dataset("rajpurkar/squad_v2", split="validation", trust_remote_code=True)
    answerable = raw.filter(lambda x: len(x["answers"]["text"]) > 0)
    if len(answerable) < sample_size:
        log.warning(
            f"Only {len(answerable)} answerable SQuAD examples available; "
            f"requested {sample_size}. Using all available."
        )
        sample_size = len(answerable)
    answerable = answerable.shuffle(seed=seed).select(range(sample_size))
    return pd.DataFrame({
        "dataset":    "squad",
        "sample_id":  list(range(sample_size)),
        "clean_text": answerable["question"],
        "label":      [json.dumps(item["text"]) for item in answerable["answers"]],
        "context":    answerable["context"],
    })


def build_prompt(text: str, tokenizer, context: str = None) -> str:
    messages = [{
        "role": "user",
        "content": (
            "Read the passage below and answer the question with a short phrase "
            "taken directly from the passage.\n\n"
            f"Passage: {context}\n\n"
            f"Question: {text}\n\n"
            "Answer:"
        ),
    }]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_output(raw: str) -> str:
    answer = raw.strip()
    if not answer or len(answer) > 150:
        return "unknown"
    return answer


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def _token_f1(pred_tokens: list[str], gt_tokens: list[str]) -> float:
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def score(pred: str, label) -> float:
    if pred == "unknown":
        return 0.0
    if isinstance(label, str):
        try:
            gold_answers = json.loads(label)
        except (json.JSONDecodeError, ValueError):
            gold_answers = [label]
    else:
        gold_answers = list(label)
    if not gold_answers:
        return 0.0
    pred_tokens = _normalize(pred)
    if not pred_tokens:
        return 0.0
    return max(_token_f1(pred_tokens, _normalize(g)) for g in gold_answers)
