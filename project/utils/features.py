# -*- coding: utf-8 -*-
"""
Tokenization feature extraction shared across all datasets.
Extracted from project/scripts/stage_a_data.py.
"""

import logging

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

TOKENIZER_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
FEATURE_COLS   = ["sigma_prime", "oov_rate", "severity", "word_count", "alpha"]
COLLINEARITY_THRESHOLD = 0.85


def load_tokenizer(name: str = TOKENIZER_NAME) -> AutoTokenizer:
    log.info(f"Loading tokenizer: {name}")
    tok = AutoTokenizer.from_pretrained(name)
    log.info("Tokenizer loaded.")
    return tok


def build_v_clean(clean_texts: pd.Series, tokenizer: AutoTokenizer) -> set[str]:
    """
    Build reference vocabulary V_clean by tokenizing all unique clean sentences.
    Used to compute oov_rate for noisy inputs.
    """
    log.info(f"Building V_clean from {len(clean_texts)} unique clean sentences...")
    v_clean: set[str] = set()
    for text in tqdm(clean_texts, desc="Building V_clean"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        tokens    = tokenizer.convert_ids_to_tokens(token_ids)
        v_clean.update(tokens)
    log.info(f"V_clean size: {len(v_clean)} unique tokens.")
    return v_clean


def _stripped_len(token_str: str) -> int:
    return len(token_str.replace("▁", ""))


def _compute_features(noisy_text: str, tokenizer: AutoTokenizer, v_clean: set[str]) -> dict:
    """
    Compute phi = [word_count, token_count, sigma_prime, alpha, oov_rate]
    from noisy_text only (paper §V.C).
    """
    token_ids = tokenizer.encode(noisy_text, add_special_tokens=False)
    tokens    = tokenizer.convert_ids_to_tokens(token_ids)
    Tn = len(tokens)
    L  = len(noisy_text.split())
    if Tn == 0 or L == 0:
        return {"word_count": L, "token_count": Tn,
                "sigma_prime": 0.0, "alpha": 0.0, "oov_rate": 0.0}
    sigma_prime = Tn / L
    alpha       = sum(1 for t in tokens if _stripped_len(t) == 1) / Tn
    oov_rate    = sum(1 for t in tokens if t not in v_clean)       / Tn
    return {
        "word_count":  L,
        "token_count": Tn,
        "sigma_prime": sigma_prime,
        "alpha":       alpha,
        "oov_rate":    oov_rate,
    }


def extract_features(
    noisy_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    v_clean: set[str],
) -> pd.DataFrame:
    """Add word_count, token_count, sigma_prime, alpha, oov_rate to noisy_df."""
    log.info(f"Extracting tokenization features for {len(noisy_df)} rows...")
    feature_rows = []
    for _, row in tqdm(noisy_df.iterrows(), total=len(noisy_df), desc="Feature extraction"):
        feature_rows.append(_compute_features(row["noisy_text"], tokenizer, v_clean))
    features_df = pd.DataFrame(feature_rows)
    return pd.concat([noisy_df.reset_index(drop=True), features_df], axis=1)


def check_feature_independence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation. Log pairs where |r| > threshold.
    Reporting only — does not remove features.
    """
    present = [c for c in FEATURE_COLS if c in df.columns]
    corr = df[present].corr(method="pearson")
    checked = set()
    flagged = []
    for i, f1 in enumerate(present):
        for f2 in present[i + 1:]:
            pair = (f1, f2)
            if pair in checked:
                continue
            checked.add(pair)
            r = corr.loc[f1, f2]
            if abs(r) > COLLINEARITY_THRESHOLD:
                flagged.append((f1, f2, r))
                log.warning(
                    f"Collinearity flag: |r({f1}, {f2})| = {abs(r):.4f} > {COLLINEARITY_THRESHOLD}"
                )
    if not flagged:
        log.info(f"No collinear pairs found (threshold |r| > {COLLINEARITY_THRESHOLD}).")
    return corr
