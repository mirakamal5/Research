"""
Stage A — Parts 1, 2, 3: Data Loading, Noise Generation, Tokenizer Feature Extraction
Loads SST-2, samples 200 examples, expands to noisy variants across 6 noise families ×
3 severity levels, then extracts the 5-dimensional tokenization feature vector
phi = [sigma_prime, oov_rate, severity, word_count, alpha] from noisy_text.
"""

import logging
import random
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATASET_NAME   = "sst2"
SAMPLE_SIZE    = 200
SEED           = 42
TOKENIZER_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

NOISE_TYPES = [
    "keyboard_proximity",
    "homoglyph",
    "char_repetition",
    "char_deletion",
    "intra_word_whitespace",
    "random_case_flip",
]
SEVERITIES = [0.05, 0.15, 0.30]

# Pairwise Pearson |r| threshold for collinearity flag (paper §V.C)
COLLINEARITY_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Noise families
# ---------------------------------------------------------------------------

_KEYBOARD_ADJACENCY: dict[str, str] = {
    "q": "was",    "w": "qeasd",  "e": "wrsd",   "r": "etdf",   "t": "ryfg",
    "y": "tugh",   "u": "yihj",   "i": "uojk",   "o": "ipkl",   "p": "ol",
    "a": "qwsz",   "s": "awedxz", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv",
    "h": "gyujnb", "j": "huikmn", "k": "jiolm",  "l": "kop",
    "z": "asx",    "x": "zsdc",   "c": "xdfv",   "v": "cfgb",   "b": "vghn",
    "n": "bhjm",   "m": "njk",
}

_HOMOGLYPHS: dict[str, str] = {
    "a": "@", "e": "3", "i": "1", "o": "0",
    "s": "5", "t": "7", "b": "6", "g": "9",
    "l": "1", "z": "2",
}


def _pick_positions(indices: list[int], n: int, rng: random.Random) -> list[int]:
    n = min(n, len(indices))
    return rng.sample(indices, n) if n > 0 else []


def _n_perturb(text: str, severity: float) -> int:
    return max(1, round(severity * len(text)))


def _keyboard_proximity(text: str, severity: float, rng: random.Random) -> str:
    alpha_idx = [i for i, c in enumerate(text) if c.lower() in _KEYBOARD_ADJACENCY]
    positions = _pick_positions(alpha_idx, _n_perturb(text, severity), rng)
    chars = list(text)
    for pos in positions:
        orig = chars[pos].lower()
        neighbors = _KEYBOARD_ADJACENCY.get(orig, "")
        if neighbors:
            sub = rng.choice(neighbors)
            chars[pos] = sub.upper() if chars[pos].isupper() else sub
    return "".join(chars)


def _homoglyph(text: str, severity: float, rng: random.Random) -> str:
    eligible = [i for i, c in enumerate(text) if c.lower() in _HOMOGLYPHS]
    positions = _pick_positions(eligible, _n_perturb(text, severity), rng)
    chars = list(text)
    for pos in positions:
        chars[pos] = _HOMOGLYPHS[chars[pos].lower()]
    return "".join(chars)


def _char_repetition(text: str, severity: float, rng: random.Random) -> str:
    alpha_idx = [i for i, c in enumerate(text) if c.isalpha()]
    positions = sorted(_pick_positions(alpha_idx, _n_perturb(text, severity), rng))
    chars = list(text)
    for pos in reversed(positions):
        chars.insert(pos + 1, chars[pos])
    return "".join(chars)


def _char_deletion(text: str, severity: float, rng: random.Random) -> str:
    alpha_idx = [i for i, c in enumerate(text) if c.isalpha()]
    positions = set(_pick_positions(alpha_idx, _n_perturb(text, severity), rng))
    return "".join(c for i, c in enumerate(text) if i not in positions)


def _intra_word_whitespace(text: str, severity: float, rng: random.Random) -> str:
    words = text.split(" ")
    eligible_idx = [i for i, w in enumerate(words) if len(w) >= 2]
    n = max(1, round(severity * len(text) / 5))
    targets = _pick_positions(eligible_idx, n, rng)
    for i in targets:
        w = words[i]
        split_at = rng.randint(1, len(w) - 1)
        words[i] = w[:split_at] + " " + w[split_at:]
    return " ".join(words)


def _random_case_flip(text: str, severity: float, rng: random.Random) -> str:
    alpha_idx = [i for i, c in enumerate(text) if c.isalpha()]
    positions = _pick_positions(alpha_idx, _n_perturb(text, severity), rng)
    chars = list(text)
    for pos in positions:
        chars[pos] = chars[pos].lower() if chars[pos].isupper() else chars[pos].upper()
    return "".join(chars)


_NOISE_FN = {
    "keyboard_proximity":    _keyboard_proximity,
    "homoglyph":             _homoglyph,
    "char_repetition":       _char_repetition,
    "char_deletion":         _char_deletion,
    "intra_word_whitespace": _intra_word_whitespace,
    "random_case_flip":      _random_case_flip,
}


def apply_noise(text: str, noise_type: str, severity: float, seed: int = SEED) -> str:
    """Apply a single noise family at the given severity. Deterministic given seed."""
    rng = random.Random(seed)
    return _NOISE_FN[noise_type](text, severity, rng)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sst2(sample_size: int = SAMPLE_SIZE, seed: int = SEED) -> pd.DataFrame:
    log.info("Loading SST-2 train split from HuggingFace datasets...")
    raw = load_dataset("glue", "sst2", split="train")
    log.info(f"Sampling {sample_size} examples with seed={seed}...")
    sampled = raw.shuffle(seed=seed).select(range(sample_size))
    return pd.DataFrame({
        "dataset":    DATASET_NAME,
        "sample_id":  range(sample_size),
        "clean_text": sampled["sentence"],
        "label":      sampled["label"],
    })


def generate_noisy_variants(clean_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Expand each clean sample into 18 rows: one per (noise_type, severity)."""
    total = len(clean_df) * len(NOISE_TYPES) * len(SEVERITIES)
    log.info(
        f"Generating noisy variants: {len(clean_df)} samples × "
        f"{len(NOISE_TYPES)} noise types × {len(SEVERITIES)} severities = {total} rows..."
    )
    rows = []
    for _, row in tqdm(clean_df.iterrows(), total=len(clean_df), desc="Noise generation"):
        for noise_type in NOISE_TYPES:
            for severity in SEVERITIES:
                rows.append({
                    "dataset":    row["dataset"],
                    "sample_id":  row["sample_id"],
                    "clean_text": row["clean_text"],
                    "noisy_text": apply_noise(row["clean_text"], noise_type, severity, seed),
                    "label":      row["label"],
                    "noise_type": noise_type,
                    "severity":   severity,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tokenizer and feature extraction (paper §V.C)
# ---------------------------------------------------------------------------

def load_tokenizer(name: str = TOKENIZER_NAME) -> AutoTokenizer:
    log.info(f"Loading tokenizer: {name}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    log.info("Tokenizer loaded.")
    return tokenizer


def build_v_clean(clean_texts: pd.Series, tokenizer: AutoTokenizer) -> set[str]:
    """
    Build reference vocabulary V_clean by tokenizing ALL clean sentences.
    V_clean is the set of unique token strings produced from the clean corpus.
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
    """
    Token length after removing SentencePiece leading-space marker (▁).
    Needed so that '▁a' (a word-initial single character) counts as length 1,
    capturing BPE collapse correctly as defined in the paper for alpha.
    """
    return len(token_str.replace("▁", ""))


def _compute_features(noisy_text: str, tokenizer: AutoTokenizer, v_clean: set[str]) -> dict:
    """
    Compute phi = [word_count, token_count, sigma_prime, alpha, oov_rate]
    EXACTLY as defined in paper §V.C. All features derived from noisy_text only.

    L  = word count of noisy_text
    Tn = token count from BPE tokenizer (no special tokens)
    sigma_prime = Tn / L
    alpha       = |{t in T(x~) : stripped_len(t) == 1}| / Tn
    oov_rate    = |{t in T(x~) : t not in V_clean}|    / Tn
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
    """
    Add word_count, token_count, sigma_prime, alpha, oov_rate columns to noisy_df.
    Features are computed exclusively from noisy_text.
    """
    log.info(f"Extracting tokenization features for {len(noisy_df)} rows...")
    feature_rows = []
    for _, row in tqdm(noisy_df.iterrows(), total=len(noisy_df), desc="Feature extraction"):
        feature_rows.append(_compute_features(row["noisy_text"], tokenizer, v_clean))

    features_df = pd.DataFrame(feature_rows)
    return pd.concat([noisy_df.reset_index(drop=True), features_df], axis=1)


# ---------------------------------------------------------------------------
# Feature independence analysis (paper §V.C)
# ---------------------------------------------------------------------------

FEATURE_COLS = ["sigma_prime", "oov_rate", "severity", "word_count", "alpha"]


def check_feature_independence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation matrix over the 5 feature columns.
    Log any pair where |r| > COLLINEARITY_THRESHOLD (0.85).
    Does NOT remove any feature — reporting only, as required before training.
    """
    corr = df[FEATURE_COLS].corr(method="pearson")

    log.info("Pairwise Pearson correlation matrix computed.")
    flagged = []
    checked = set()
    for i, f1 in enumerate(FEATURE_COLS):
        for f2 in FEATURE_COLS[i + 1:]:
            pair = (f1, f2)
            if pair in checked:
                continue
            checked.add(pair)
            r = corr.loc[f1, f2]
            if abs(r) > COLLINEARITY_THRESHOLD:
                flagged.append((f1, f2, r))
                log.warning(
                    f"Collinearity flag: |r({f1}, {f2})| = {abs(r):.4f} > {COLLINEARITY_THRESHOLD}. "
                    f"Review feature importance before training to determine which to drop."
                )

    if not flagged:
        log.info(f"No collinear pairs found (threshold |r| > {COLLINEARITY_THRESHOLD}).")

    return corr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Stage A Part 1 — load data
    clean_df = load_sst2()

    # Stage A Part 2 — noise generation
    noisy_df = generate_noisy_variants(clean_df)

    # Stage A Part 3 — tokenizer features
    tokenizer = load_tokenizer()
    # V_clean is built from unique clean sentences (deduplicated to avoid redundant tokenizations)
    v_clean   = build_v_clean(clean_df["clean_text"].drop_duplicates(), tokenizer)
    full_df   = extract_features(noisy_df, tokenizer, v_clean)

    # Feature independence check
    print("\n--- Pairwise Pearson Correlation Matrix ---")
    corr = check_feature_independence(full_df)
    print(corr.round(4).to_string())

    # Sample output
    print("\n--- 5 Sample Rows (all feature columns) ---")
    display_cols = [
        "sample_id", "noise_type", "severity",
        "word_count", "token_count", "sigma_prime", "alpha", "oov_rate",
        "clean_text", "noisy_text",
    ]
    print(full_df[display_cols].head(5).to_string(index=False))

    print(f"\nTotal rows: {len(full_df)}")
    print(f"Columns:    {list(full_df.columns)}")


if __name__ == "__main__":
    main()
