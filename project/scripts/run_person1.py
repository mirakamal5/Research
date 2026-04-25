"""
run_person1.py  —  Person 1: data loading, noise generation, tokenization features.

No GPU required. Run this locally (or in Colab before loading the model).
Produces a single base CSV for a given dataset, then upload it to Colab for Person 2.

Usage:
    python project/scripts/run_person1.py --dataset svamp --sample-size 500
    python project/scripts/run_person1.py --dataset gsm8k --sample-size 500
    python project/scripts/run_person1.py --dataset squad --sample-size 500
    python project/scripts/run_person1.py --dataset sst2  --sample-size 500

Output:
    data/interim/stage_a_{dataset}_base.csv
    (contains all sample_size × 18 noisy variants with tokenization features)
"""

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.noise    import generate_noisy_variants
from utils.features import (
    load_tokenizer, build_v_clean, extract_features, check_feature_independence
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.2"
SEED         = 42

TASK_MODULES = {
    "sst2":  "utils.tasks.sst2",
    "squad": "utils.tasks.squad",
    "gsm8k": "utils.tasks.gsm8k",
    "svamp": "utils.tasks.svamp",
}


def run(dataset: str, sample_size: int, seed: int, base_dir: str):
    task_mod  = importlib.import_module(TASK_MODULES[dataset])
    base_path = os.path.join(base_dir, f"stage_a_{dataset}_base.csv")

    log.info("=" * 60)
    log.info(f"PERSON 1  |  dataset={dataset}  sample_size={sample_size}  seed={seed}")
    log.info("=" * 60)

    # Load clean data
    clean_df = task_mod.load_clean(sample_size=sample_size, seed=seed, offset=0)
    log.info(f"Loaded {len(clean_df)} clean sentences.")

    # Generate 18 noisy variants per sample
    noisy_df = generate_noisy_variants(clean_df, seed=seed)

    # Tokenization features
    feat_tok = load_tokenizer(MODEL_NAME)
    v_clean  = build_v_clean(clean_df["clean_text"].drop_duplicates(), feat_tok)
    base_df  = extract_features(noisy_df, feat_tok, v_clean)

    check_feature_independence(base_df)

    # Save
    os.makedirs(base_dir, exist_ok=True)
    base_df.to_csv(base_path, index=False)
    log.info(f"Base CSV saved: {base_path}  ({len(base_df)} rows × {len(base_df.columns)} cols)")
    log.info(f"sample_ids: 0 – {sample_size - 1}  |  noise rows per sample: 18")


def main():
    parser = argparse.ArgumentParser(
        description="Person 1: generate base CSV (no GPU needed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",     required=True, choices=list(TASK_MODULES.keys()))
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed",        type=int, default=SEED)
    parser.add_argument("--base-dir",    default="data/interim")
    args = parser.parse_args()

    run(args.dataset, args.sample_size, args.seed, args.base_dir)


if __name__ == "__main__":
    main()
