"""
SQuAD v2 task adapter for Stage A.

Strategy:
  - Noise the question only; context (passage) is carried unchanged.
  - 500 answerable-only questions from the validation split.
  - Prompt explicitly instructs the model to output ONLY the answer phrase.
  - Parser: take first line only, strip "Answer:" echo, extract from
    "The answer is X" format, strip trailing parentheticals, then apply
    150-char and refusal checks.
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


def load_clean(sample_size: int = 500, seed: int = 42, offset: int = 0) -> pd.DataFrame:
    log.info(f"Loading SQuAD v2 validation split (sample_size={sample_size}, offset={offset}, seed={seed})...")
    raw = load_dataset("rajpurkar/squad_v2", split="validation", trust_remote_code=True)
    answerable = raw.filter(lambda x: len(x["answers"]["text"]) > 0)
    if offset + sample_size > len(answerable):
        log.warning(
            f"offset+sample_size ({offset + sample_size}) exceeds available "
            f"SQuAD examples ({len(answerable)}). Capping."
        )
        sample_size = max(0, len(answerable) - offset)
    answerable = answerable.shuffle(seed=seed).select(range(offset, offset + sample_size))
    return pd.DataFrame({
        "dataset":    "squad",
        "sample_id":  list(range(offset, offset + sample_size)),
        "clean_text": answerable["question"],
        "label":      [json.dumps(item["text"]) for item in answerable["answers"]],
        "context":    answerable["context"],
    })


def build_prompt(text: str, tokenizer, context: str = None) -> str:
    messages = [{
        "role": "user",
        "content": (
            "Read the passage and answer the question with a short phrase "
            "taken directly from the passage.\n"
            "Output ONLY the answer phrase. Do not explain or add context.\n\n"
            f"Passage: {context}\n\n"
            f"Question: {text}\n\n"
            "Answer:"
        ),
    }]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


_ANSWER_IS_RE = re.compile(r"the\s+answer\s+is[:\s]+(.+)", re.IGNORECASE)

_META_RE = re.compile(
    r"^("
    r"i cannot|i can'?t|i don'?t|i am not|i'm not|i do not"
    r"|the passage does|the text does|the passage (does not|doesn't|says nothing)"
    r"|based on the (passage|text|context|information)"
    r"|according to the (passage|text)"
    r"|there is no|there are no"
    r"|this passage|unfortunately|it is not (mentioned|stated|clear|specified)"
    r"|no (information|mention|answer|details?)"
    r")",
    re.IGNORECASE,
)


def parse_output(raw: str) -> str:
    # Take first non-empty line — model sometimes appends explanation after newline
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    answer = lines[0] if lines else ""

    # Strip "Answer:" echo (model sometimes repeats the cue word)
    answer = re.sub(r"^[Aa]nswer\s*:\s*", "", answer).strip()

    if not answer:
        return "unknown"

    # Extract from "The answer is X" format instead of rejecting it
    m = _ANSWER_IS_RE.match(answer)
    if m:
        answer = m.group(1).strip()

    # Strip trailing parenthetical (e.g. "Paris (capital of France)")
    answer = re.sub(r"\s*\([^)]*\)\s*$", "", answer).strip()

    # Strip trailing punctuation artifacts
    answer = answer.rstrip(".,;:")

    if not answer:
        return "unknown"

    if _META_RE.match(answer):
        return "unknown"

    if len(answer) > 150:
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
