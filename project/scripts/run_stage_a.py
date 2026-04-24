"""
run_stage_a.py  —  Stage A pipeline (Person 1 + Person 2) for SST-2, SQuAD v2, or GSM8K.

This single script replaces running stage_a_data.py + stage_a_label.py separately.
It works for all three datasets without any code changes.

Usage (run from the Research/ root):
    python project/scripts/run_stage_a.py --dataset sst2  --sample-size 500
    python project/scripts/run_stage_a.py --dataset squad --sample-size 500
    python project/scripts/run_stage_a.py --dataset gsm8k --sample-size 500

    # Person 1 only (no GPU needed — useful for local testing):
    python project/scripts/run_stage_a.py --dataset sst2 --sample-size 50 --skip-inference

Output files:
    data/interim/stage_a_{dataset}_base.csv    (Person 1 output, kept for debugging)
    data/processed/stage_a_{dataset}_final.csv (Person 2 output, Stage B input)

Run on Google Colab T4 for the full pipeline (requires ~5 GB VRAM for 4-bit Mistral-7B).
"""

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Allow `from utils.X import Y` regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.noise    import generate_noisy_variants
from utils.features import load_tokenizer, build_v_clean, extract_features, check_feature_independence

# torch/transformers are imported lazily inside person2() so that
# --skip-inference works locally without a GPU or torch installation.

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME        = "mistralai/Mistral-7B-Instruct-v0.2"
FAILURE_THRESHOLD = 0.20
SEED              = 42

TASK_MODULES = {
    "sst2":  "utils.tasks.sst2",
    "squad": "utils.tasks.squad",
    "gsm8k": "utils.tasks.gsm8k",
}

# Canonical 18-column schema + 2 debug columns
FINAL_COLUMNS = [
    "dataset", "sample_id", "clean_text", "noisy_text", "label",
    "noise_type", "severity", "word_count", "token_count",
    "sigma_prime", "alpha", "oov_rate",
    "clean_pred", "noisy_pred",
    "clean_score", "noisy_score",
    "delta_p", "failure_label",
    "clean_raw_output", "noisy_raw_output",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_task(dataset: str):
    return importlib.import_module(TASK_MODULES[dataset])


def _makedirs(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _load_mistral(model_name: str = MODEL_NAME):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model in 4-bit NF4 quantization...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    log.info("Model ready.")
    return model, tokenizer


def _infer(text: str, model, tokenizer, task_mod, context=None) -> str:
    import torch

    prompt = task_mod.build_prompt(text, tokenizer, context=context)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=task_mod.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _debug_check(rows_df: pd.DataFrame, model, tokenizer, task_mod, n: int = 5):
    has_ctx = getattr(task_mod, "HAS_CONTEXT", False)
    log.info(f"=== DEBUG: first {n} clean sentences ===")
    for i, (_, row) in enumerate(rows_df.head(n).iterrows()):
        ctx = row.get("context") if has_ctx else None
        raw  = _infer(row["clean_text"], model, tokenizer, task_mod, context=ctx)
        pred = task_mod.parse_output(raw)
        print(f"  [{i}] INPUT : {str(row['clean_text'])[:80]}")
        print(f"  [{i}] RAW   : {repr(raw[:120])}")
        print(f"  [{i}] PARSED: {pred}")
        print()
    log.info("=== PARSED should not be mostly 'unknown' ===")


def _validate(df: pd.DataFrame):
    log.info("Validating outputs...")
    checks = [
        (df["clean_pred"].isnull().any(),
         "clean_pred has nulls"),
        (df["noisy_pred"].isnull().any(),
         "noisy_pred has nulls"),
        (not ((df["clean_score"] >= 0.0) & (df["clean_score"] <= 1.0)).all(),
         "clean_score out of [0, 1]"),
        (not ((df["noisy_score"] >= 0.0) & (df["noisy_score"] <= 1.0)).all(),
         "noisy_score out of [0, 1]"),
        (not pd.api.types.is_numeric_dtype(df["delta_p"]),
         "delta_p not numeric"),
        (not df["failure_label"].isin([0, 1]).all(),
         "failure_label outside {0, 1}"),
    ]
    for condition, message in checks:
        if condition:
            log.warning(f"Validation warning: {message}")
    log.info("Validation done.")


# ---------------------------------------------------------------------------
# Person 1: data + noise + features
# ---------------------------------------------------------------------------

def person1(task_mod, sample_size: int, seed: int, base_path: str) -> pd.DataFrame:
    log.info("=" * 60)
    log.info(f"PERSON 1  |  dataset={task_mod.__name__.split('.')[-1]}  sample_size={sample_size}")
    log.info("=" * 60)

    clean_df = task_mod.load_clean(sample_size=sample_size, seed=seed)
    log.info(f"Loaded {len(clean_df)} clean sentences.")

    noisy_df = generate_noisy_variants(clean_df, seed=seed)

    feat_tok = load_tokenizer(MODEL_NAME)
    v_clean  = build_v_clean(clean_df["clean_text"].drop_duplicates(), feat_tok)
    base_df  = extract_features(noisy_df, feat_tok, v_clean)

    check_feature_independence(base_df)

    _makedirs(base_path)
    base_df.to_csv(base_path, index=False)
    log.info(f"Base CSV saved: {base_path}  ({len(base_df)} rows)")
    return base_df


# ---------------------------------------------------------------------------
# Person 2: Mistral inference + labeling
# ---------------------------------------------------------------------------

def person2(base_df: pd.DataFrame, task_mod, final_path: str) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("PERSON 2  |  LLM inference + labeling")
    log.info("=" * 60)

    model, inf_tok = _load_mistral()
    has_ctx        = getattr(task_mod, "HAS_CONTEXT", False)

    # Deduplicated clean sentences (run each clean sentence once)
    ctx_col  = ["context"] if has_ctx and "context" in base_df.columns else []
    uniq_cols = ["sample_id", "clean_text", "label"] + ctx_col
    unique_clean = (
        base_df[uniq_cols]
        .drop_duplicates(subset="sample_id")
        .reset_index(drop=True)
    )

    _debug_check(unique_clean, model, inf_tok, task_mod)

    # --- clean inference ---
    log.info(f"Running clean inference on {len(unique_clean)} unique sentences...")
    clean_cache: dict[int, tuple[str, float, str]] = {}
    for i, (_, row) in enumerate(
        tqdm(unique_clean.iterrows(), total=len(unique_clean), desc="Clean inference")
    ):
        ctx  = row["context"] if has_ctx and "context" in row.index else None
        raw  = _infer(row["clean_text"], model, inf_tok, task_mod, context=ctx)
        pred = task_mod.parse_output(raw)
        sc   = task_mod.score(pred, row["label"])
        clean_cache[int(row["sample_id"])] = (pred, sc, raw)

    clean_acc = sum(v[1] for v in clean_cache.values()) / len(clean_cache)
    log.info(f"Clean accuracy: {clean_acc:.3f}")
    if clean_acc < 0.50:
        log.warning(
            f"Clean accuracy is {clean_acc:.3f} — check DEBUG output above."
        )

    # --- noisy inference ---
    log.info(f"Running noisy inference on {len(base_df)} rows...")
    noisy_preds, noisy_scores, noisy_raws = [], [], []
    for _, row in tqdm(base_df.iterrows(), total=len(base_df), desc="Noisy inference"):
        ctx  = row["context"] if has_ctx and "context" in base_df.columns else None
        raw  = _infer(row["noisy_text"], model, inf_tok, task_mod, context=ctx)
        pred = task_mod.parse_output(raw)
        sc   = task_mod.score(pred, row["label"])
        noisy_preds.append(pred)
        noisy_scores.append(sc)
        noisy_raws.append(raw)

    # --- attach results ---
    df = base_df.copy()
    df["clean_pred"]       = df["sample_id"].map(lambda sid: clean_cache[int(sid)][0])
    df["clean_score"]      = df["sample_id"].map(lambda sid: clean_cache[int(sid)][1])
    df["clean_raw_output"] = df["sample_id"].map(lambda sid: clean_cache[int(sid)][2])
    df["noisy_pred"]       = noisy_preds
    df["noisy_score"]      = noisy_scores
    df["noisy_raw_output"] = noisy_raws

    df["delta_p"]       = df["clean_score"] - df["noisy_score"]
    df["failure_label"] = (df["delta_p"] > FAILURE_THRESHOLD).astype(int)

    _validate(df)

    # Keep only columns in FINAL_COLUMNS (drops 'context' for SQuAD)
    out_cols = [c for c in FINAL_COLUMNS if c in df.columns]
    _makedirs(final_path)
    df[out_cols].to_csv(final_path, index=False)
    log.info(f"Final CSV saved: {final_path}  ({len(df)} rows × {len(out_cols)} cols)")

    n_fail  = int(df["failure_label"].sum())
    n_rob   = len(df) - n_fail
    n_unk_c = int((df["clean_pred"]  == "unknown").sum())
    n_unk_n = int((df["noisy_pred"] == "unknown").sum())
    log.info(
        f"FAILURE: {n_fail} ({100*n_fail/len(df):.1f}%)   "
        f"ROBUST: {n_rob} ({100*n_rob/len(df):.1f}%)"
    )
    log.info(f"Unknown predictions — clean: {n_unk_c},  noisy: {n_unk_n}")

    return df[out_cols]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage A pipeline (Person 1 + Person 2) for SST-2, SQuAD v2, or GSM8K.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True, choices=list(TASK_MODULES.keys()),
        help="Which dataset to process.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=500,
        help="Number of clean sentences to sample.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed for sampling and noise generation.",
    )
    parser.add_argument(
        "--base-dir", default="data/interim",
        help="Directory for the intermediate base CSV (Person 1 output).",
    )
    parser.add_argument(
        "--final-dir", default="data/processed",
        help="Directory for the final labeled CSV (Person 2 output).",
    )
    parser.add_argument(
        "--skip-inference", action="store_true",
        help="Run Person 1 only (no LLM). Useful for local testing without a GPU.",
    )
    args = parser.parse_args()

    task_mod   = _load_task(args.dataset)
    base_path  = os.path.join(args.base_dir,  f"stage_a_{args.dataset}_base.csv")
    final_path = os.path.join(args.final_dir, f"stage_a_{args.dataset}_final.csv")

    base_df = person1(task_mod, args.sample_size, args.seed, base_path)

    if args.skip_inference:
        log.info("--skip-inference: stopping after Person 1.")
        return

    person2(base_df, task_mod, final_path)
    log.info(f"Stage A complete for --dataset {args.dataset}.")


if __name__ == "__main__":
    main()
