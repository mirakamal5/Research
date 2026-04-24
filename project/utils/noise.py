"""
Noise injection utilities shared across all datasets.
Extracted from project/scripts/stage_a_data.py so run_stage_a.py can
import them without importing the SST-2-specific main().
"""

import random
import logging

import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

NOISE_TYPES = [
    "keyboard_proximity",
    "homoglyph",
    "char_repetition",
    "char_deletion",
    "intra_word_whitespace",
    "random_case_flip",
]
SEVERITIES = [0.05, 0.15, 0.30]
SEED = 42

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

_EXTRA_COLS_SKIP = {"dataset", "sample_id", "clean_text", "label"}


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
    """Apply one noise family at a given severity. Deterministic given seed."""
    rng = random.Random(seed)
    return _NOISE_FN[noise_type](text, severity, rng)


def generate_noisy_variants(clean_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """
    Expand each clean sample into 18 rows: one per (noise_type, severity).
    Extra columns beyond the fixed set (e.g., 'context' for SQuAD) are
    carried forward unchanged.
    """
    total = len(clean_df) * len(NOISE_TYPES) * len(SEVERITIES)
    log.info(
        f"Generating noisy variants: {len(clean_df)} samples × "
        f"{len(NOISE_TYPES)} noise types × {len(SEVERITIES)} severities = {total} rows..."
    )
    extra_cols = [c for c in clean_df.columns if c not in _EXTRA_COLS_SKIP]
    rows = []
    for _, row in tqdm(clean_df.iterrows(), total=len(clean_df), desc="Noise generation"):
        for noise_type in NOISE_TYPES:
            for severity in SEVERITIES:
                entry = {
                    "dataset":    row["dataset"],
                    "sample_id":  row["sample_id"],
                    "clean_text": row["clean_text"],
                    "noisy_text": apply_noise(row["clean_text"], noise_type, severity, seed),
                    "label":      row["label"],
                    "noise_type": noise_type,
                    "severity":   severity,
                }
                for col in extra_cols:
                    entry[col] = row[col]
                rows.append(entry)
    return pd.DataFrame(rows)
