"""
run_person2.py  —  Person 2: Mistral inference + labeling on a chunk of the base CSV.

Run on Google Colab T4 GPU. Reads the base CSV produced by run_person1.py.
Process the dataset in chunks (e.g., 100 samples at a time) to survive Colab
session limits. Merge all chunks into a single final CSV when done.

─────────────────────────────────────────────────────────────
WORKFLOW
─────────────────────────────────────────────────────────────
Step 1 — Mount Drive and set paths (in Colab):
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR  = '/content/drive/MyDrive/stage_a/interim'
    FINAL_DIR = '/content/drive/MyDrive/stage_a/processed'

Step 2 — Run Person 2 in 100-sample chunks:
    python project/scripts/run_person2.py --dataset svamp --offset   0 --sample-size 100 --base-dir BASE_DIR --final-dir FINAL_DIR
    python project/scripts/run_person2.py --dataset svamp --offset 100 --sample-size 100 --base-dir BASE_DIR --final-dir FINAL_DIR
    python project/scripts/run_person2.py --dataset svamp --offset 200 --sample-size 100 ...
    python project/scripts/run_person2.py --dataset svamp --offset 300 --sample-size 100 ...
    python project/scripts/run_person2.py --dataset svamp --offset 400 --sample-size 100 ...

Step 3 — Merge all chunks:
    python project/scripts/run_person2.py --dataset svamp --merge --final-dir FINAL_DIR

─────────────────────────────────────────────────────────────
OUTPUT FILES
─────────────────────────────────────────────────────────────
  Chunks:  data/processed/stage_a_{dataset}_{offset}_{offset+size}_final.csv
  Merged:  data/processed/stage_a_{dataset}_final.csv
"""

import argparse
import glob
import importlib
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME        = "mistralai/Mistral-7B-Instruct-v0.2"
FAILURE_THRESHOLD = 0.20
SEED              = 42

TASK_MODULES = {
    "sst2":  "utils.tasks.sst2",
    "squad": "utils.tasks.squad",
    "gsm8k": "utils.tasks.gsm8k",
    "svamp": "utils.tasks.svamp",
}

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
# Inference helpers (GPU-dependent — imported lazily)
# ---------------------------------------------------------------------------

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


def _debug_check(rows_df: pd.DataFrame, model, tokenizer, task_mod, n: int = 3):
    has_ctx = getattr(task_mod, "HAS_CONTEXT", False)
    log.info(f"=== DEBUG: first {n} clean sentences ===")
    for i, (_, row) in enumerate(rows_df.head(n).iterrows()):
        ctx  = row.get("context") if has_ctx else None
        raw  = _infer(row["clean_text"], model, tokenizer, task_mod, context=ctx)
        pred = task_mod.parse_output(raw)
        print(f"  [{i}] INPUT : {str(row['clean_text'])[:80]}")
        print(f"  [{i}] RAW   : {repr(raw[:120])}")
        print(f"  [{i}] PARSED: {pred}")
        print()
    log.info("=== PARSED should not be mostly 'unknown' ===")


def _validate(df: pd.DataFrame):
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
            log.warning(f"Validation: {message}")


# ---------------------------------------------------------------------------
# Core inference for one chunk
# ---------------------------------------------------------------------------

def run_chunk(
    dataset:              str,
    offset:               int,
    sample_size:          int,
    base_path:            str,
    final_dir:            str,
    filter_clean_correct: bool  = False,
    min_clean_score:      float = 1.0,
):
    end = offset + sample_size
    chunk_path = os.path.join(final_dir, f"stage_a_{dataset}_{offset}_{end}_final.csv")

    if os.path.exists(chunk_path):
        log.info(f"Chunk already exists, skipping: {chunk_path}")
        return

    # Load base CSV and filter for this chunk's sample_ids
    log.info(f"Loading base CSV: {base_path}")
    base_df = pd.read_csv(base_path)
    chunk_df = base_df[base_df["sample_id"].between(offset, end - 1)].copy().reset_index(drop=True)

    if chunk_df.empty:
        log.warning(f"No rows found for sample_ids {offset}–{end - 1}. Check base CSV.")
        return

    log.info(
        f"Chunk: sample_ids {offset}–{end - 1}  "
        f"({chunk_df['sample_id'].nunique()} samples, {len(chunk_df)} rows)"
    )

    task_mod = importlib.import_module(TASK_MODULES[dataset])
    has_ctx  = getattr(task_mod, "HAS_CONTEXT", False)

    model, inf_tok = _load_mistral()

    # Deduplicated clean sentences within this chunk
    ctx_col     = ["context"] if has_ctx and "context" in chunk_df.columns else []
    uniq_cols   = ["sample_id", "clean_text", "label"] + ctx_col
    unique_clean = (
        chunk_df[uniq_cols]
        .drop_duplicates(subset="sample_id")
        .reset_index(drop=True)
    )

    _debug_check(unique_clean, model, inf_tok, task_mod)

    # Clean inference
    log.info(f"Clean inference: {len(unique_clean)} sentences...")
    from tqdm import tqdm
    clean_cache: dict[int, tuple[str, float, str]] = {}
    for i, (_, row) in enumerate(
        tqdm(unique_clean.iterrows(), total=len(unique_clean), desc="Clean")
    ):
        ctx  = row["context"] if has_ctx and "context" in row.index else None
        raw  = _infer(row["clean_text"], model, inf_tok, task_mod, context=ctx)
        pred = task_mod.parse_output(raw)
        sc   = task_mod.score(pred, row["label"])
        clean_cache[int(row["sample_id"])] = (pred, sc, raw)

    clean_acc = sum(v[1] for v in clean_cache.values()) / len(clean_cache)
    log.info(f"Clean accuracy: {clean_acc:.3f}")
    if clean_acc < 0.25:
        log.warning(f"Clean accuracy {clean_acc:.3f} is low — check DEBUG output above.")

    # Optional: drop sample_ids whose clean baseline was wrong
    if filter_clean_correct:
        retained_ids = {
            sid for sid, (_, sc, _) in clean_cache.items() if sc >= min_clean_score
        }
        n_evaluated    = len(clean_cache)
        n_retained     = len(retained_ids)
        n_removed      = n_evaluated - n_retained
        n_noisy_before = len(chunk_df)
        chunk_df = chunk_df[chunk_df["sample_id"].isin(retained_ids)].reset_index(drop=True)
        n_noisy_after  = len(chunk_df)
        log.info(
            f"[filter-clean-correct]  min_clean_score={min_clean_score}\n"
            f"  clean evaluated  : {n_evaluated}\n"
            f"  clean retained   : {n_retained}\n"
            f"  clean removed    : {n_removed}\n"
            f"  noisy rows before: {n_noisy_before}\n"
            f"  noisy rows after : {n_noisy_after}\n"
            f"  noisy rows skipped: {n_noisy_before - n_noisy_after}"
        )
        if chunk_df.empty:
            log.warning("No clean-correct samples in this chunk — skipping noisy inference.")
            return

    # Noisy inference
    log.info(f"Noisy inference: {len(chunk_df)} rows...")
    noisy_preds, noisy_scores, noisy_raws = [], [], []
    for _, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc="Noisy"):
        ctx  = row["context"] if has_ctx and "context" in chunk_df.columns else None
        raw  = _infer(row["noisy_text"], model, inf_tok, task_mod, context=ctx)
        pred = task_mod.parse_output(raw)
        sc   = task_mod.score(pred, row["label"])
        noisy_preds.append(pred)
        noisy_scores.append(sc)
        noisy_raws.append(raw)

    # Attach results
    df = chunk_df.copy()
    df["clean_pred"]       = df["sample_id"].map(lambda sid: clean_cache[int(sid)][0])
    df["clean_score"]      = df["sample_id"].map(lambda sid: clean_cache[int(sid)][1])
    df["clean_raw_output"] = df["sample_id"].map(lambda sid: clean_cache[int(sid)][2])
    df["noisy_pred"]       = noisy_preds
    df["noisy_score"]      = noisy_scores
    df["noisy_raw_output"] = noisy_raws

    df["delta_p"]       = df["clean_score"] - df["noisy_score"]
    df["failure_label"] = (df["delta_p"] > FAILURE_THRESHOLD).astype(int)

    _validate(df)

    out_cols = [c for c in FINAL_COLUMNS if c in df.columns]
    os.makedirs(final_dir, exist_ok=True)
    df[out_cols].to_csv(chunk_path, index=False)
    log.info(f"Chunk saved: {chunk_path}  ({len(df)} rows)")

    n_fail  = int(df["failure_label"].sum())
    n_unk_c = int((df["clean_pred"] == "unknown").sum())
    n_unk_n = int((df["noisy_pred"] == "unknown").sum())
    log.info(
        f"failure_label=1: {n_fail}/{len(df)}  |  "
        f"unknown clean: {n_unk_c}  unknown noisy: {n_unk_n}"
    )


# ---------------------------------------------------------------------------
# Merge all chunks into one final CSV
# ---------------------------------------------------------------------------

def merge_chunks(dataset: str, final_dir: str):
    pattern = os.path.join(final_dir, f"stage_a_{dataset}_[0-9]*_[0-9]*_final.csv")
    chunk_files = sorted(
        glob.glob(pattern),
        key=lambda f: int(re.search(r"_(\d+)_\d+_final\.csv$", f).group(1)),
    )

    if not chunk_files:
        log.error(f"No chunk files found matching: {pattern}")
        return

    log.info(f"Merging {len(chunk_files)} chunk files:")
    for f in chunk_files:
        log.info(f"  {f}")

    merged = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
    merged = merged.sort_values("sample_id").reset_index(drop=True)

    merged_path = os.path.join(final_dir, f"stage_a_{dataset}_final.csv")
    merged.to_csv(merged_path, index=False)
    log.info(f"Merged CSV saved: {merged_path}  ({len(merged)} rows)")

    n_fail  = int(merged["failure_label"].sum()) if "failure_label" in merged.columns else -1
    n_sids  = merged["sample_id"].nunique()
    log.info(f"Unique sample_ids: {n_sids}  |  failure_label=1: {n_fail}/{len(merged)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Person 2: Mistral inference on a chunk of the base CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",     choices=list(TASK_MODULES.keys()),
                        help="Dataset name. Required unless --merge.")
    parser.add_argument("--offset",      type=int, default=0,
                        help="First sample_id in this chunk.")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Number of sample_ids to process in this chunk.")
    parser.add_argument("--base-dir",    default="data/interim",
                        help="Directory containing the base CSV from Person 1.")
    parser.add_argument("--base-path",   default=None,
                        help="Explicit path to base CSV (overrides --base-dir).")
    parser.add_argument("--final-dir",   default="data/processed",
                        help="Directory where chunk and merged CSVs are written.")
    parser.add_argument("--merge",       action="store_true",
                        help="Merge all existing chunk files into stage_a_{dataset}_final.csv.")
    parser.add_argument("--filter-clean-correct", action="store_true",
                        help="Skip noisy inference for sample_ids where clean_score < --min-clean-score.")
    parser.add_argument("--min-clean-score", type=float, default=1.0,
                        help="Minimum clean_score to retain a sample_id (with --filter-clean-correct). "
                             "Use 1.0 for exact-match tasks (SST2/GSM8K/SVAMP), lower for SQuAD F1.")
    args = parser.parse_args()

    if not args.dataset:
        parser.error("--dataset is required.")

    if args.merge:
        merge_chunks(args.dataset, args.final_dir)
        return

    base_path = args.base_path or os.path.join(
        args.base_dir, f"stage_a_{args.dataset}_base.csv"
    )

    run_chunk(
        dataset=args.dataset,
        offset=args.offset,
        sample_size=args.sample_size,
        base_path=base_path,
        final_dir=args.final_dir,
        filter_clean_correct=args.filter_clean_correct,
        min_clean_score=args.min_clean_score,
    )


if __name__ == "__main__":
    main()
