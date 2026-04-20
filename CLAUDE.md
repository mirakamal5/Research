# CLAUDE.md — Shared Project Context

This file defines the full project context for all teammates and for Claude in every future session.
Read this before writing any code. Do not skip sections.

---

## Project Overview

**Title:** Predicting LLM Robustness Collapse from Tokenization Fragmentation under Typographical Noise
**Authors:** Yassine El Zeort, Mira Kamal, Kareen Terkawi — Lebanese American University
**Paper:** `docs/Research (9).pdf`

**Core idea:** Large language models fail unpredictably on typographically noisy input, but failure is only observable after the full (expensive) forward pass completes. This project builds a *pre-inference* failure prediction system that uses tokenization fragmentation — computed from the tokenizer alone in under 5 ms on CPU — to intercept high-risk inputs *before* any GPU resource is consumed.

The system operates in two temporally disjoint stages: **Stage A** (offline dataset construction) and **Stage B** (deployment-time failure gating classifier).

---

## Datasets

Three English benchmarks are used, each representing a distinct reasoning regime:

| Dataset    | Task                        | Metric(s)              | Notes                                              |
|------------|-----------------------------|------------------------|----------------------------------------------------|
| SST-2      | Binary sentiment classification | Accuracy           | Start development here. Simpler outputs, easier to validate end-to-end. Fragmentation expected to be weaker predictor (negative contrast). |
| SQuAD v2.0 | Reading comprehension (QA)  | F1, Exact Match        | Span extraction via regex from model output.       |
| GSM8K      | Multi-step math reasoning   | Final answer accuracy  | Strongest expected fragmentation–failure correlation. |

**Development order:** Begin with SST-2. Validate the full pipeline on SST-2 before extending to SQuAD and GSM8K.

**Scale:** 1,000 clean sentences sampled per dataset (3,000 total). After noise injection: ~54,000 (clean, noisy) labeled pairs.

---

## Stage A — Robustness Dataset Construction (Offline)

Stage A runs once, offline. Its purpose is to empirically characterize the relationship between tokenization fragmentation and LLM performance degradation, and to produce a labeled training dataset for Stage B.

### A.1 Data Loading

Load clean sentences from SST-2, SQuAD v2.0, and GSM8K via HuggingFace `datasets`. Sample 1,000 sentences per dataset. Track `dataset` name and `sample_id` for each row.

### A.2 Noise Injection

For each clean sentence `x`, generate perturbed variants `x̃` by applying one of **6 noise families** at **3 severity levels**: `s ∈ {0.05, 0.15, 0.30}`.

This produces up to **18 noisy variants per sentence** (6 families × 3 severities).

Severity is a **character-level perturbation rate** — the fraction of characters that are corrupted. Perturbation positions are sampled uniformly with **fixed seed `r=42`**.

**The 6 noise families:**

| Noise Type              | Example                    | Simulates                        |
|-------------------------|----------------------------|----------------------------------|
| `keyboard_proximity`    | `network → netqork`        | Adjacent-key slip                |
| `homoglyph`             | `role → r0le`              | Unicode lookalikes               |
| `char_repetition`       | `address → adddress`       | Key-hold errors                  |
| `char_deletion`         | `capital → caital`         | Missed keystroke                 |
| `intra_word_whitespace` | `context → con text`       | BPE boundary exploit             |
| `random_case_flip`      | `Mistral → mIsTrAl`        | Case-sensitive vocab disruption  |

Store the noise type and severity as columns `noise_type` and `severity` in the dataset.

### A.3 Tokenizer Usage

Use **Mistral-7B-Instruct v0.2**'s BPE tokenizer via HuggingFace `AutoTokenizer`. Apply it independently to both the clean text and the noisy text for each pair.

Build the **reference vocabulary `V_clean`** by tokenizing all clean training sentences before computing features. This is used for OOV rate calculation.

### A.4 Feature Extraction

Each input `x̃` is represented by a **5-dimensional feature vector** `φ = [σ', ρ, s, L, α]`, computed exclusively from the tokenizer output and corpus-level constants — no LLM needed.

Let `Tn = |T(x̃)|` (token count) and `L` = word count of `x̃`.

| Column name   | Symbol | Formula                                      | Description                                                                 |
|---------------|--------|----------------------------------------------|-----------------------------------------------------------------------------|
| `word_count`  | L      | number of words in `x̃`                       | Sentence length; decouples input-size confound from fragmentation signal.   |
| `token_count` | Tn     | `len(tokenizer(x̃).input_ids)`               | Raw token count from BPE tokenizer.                                         |
| `sigma_prime` | σ'     | `Tn / L`                                     | Tokens-per-word ratio. Primary fragmentation signal. Clean English ≈ 1.2–1.4; heavy noise > 2.0. |
| `alpha`       | α      | `count(t : len(t)==1) / Tn`                  | Single-character token rate. Captures BPE collapse under severe noise.      |
| `oov_rate`    | ρ      | `count(t ∉ V_clean) / Tn`                   | OOV token rate. Captures fragment novelty (orthogonal to σ').               |

Verify feature independence via pairwise Pearson correlation matrix before training. Remove any feature with `|r| > 0.85` against another (keep the higher-importance one).

**Note on `severity` at deployment time (Stage B):** During training, `s` is the exact perturbation rate. At deployment, it is approximated as the fraction of characters absent from the clean training character inventory.

### A.5 LLM Inference

Run **Mistral-7B-Instruct v0.2** in **4-bit NF4 quantization** on Google Colab (free-tier T4 GPU). Use HuggingFace `transformers`.

Run inference on both the clean text and the noisy text for each pair under **zero-shot prompts** with fixed templates:

- **SST-2:** Prompt for sentiment classification; extract exact string `"positive"` or `"negative"` from output. Score = accuracy (1 if correct, 0 if not).
- **SQuAD v2.0:** Provide passage + question; extract answer span via regex. Score = F1 and Exact Match.
- **GSM8K:** Extract the final number after `"The answer is:"`. Score = final answer accuracy.

Store model outputs as:
- `clean_pred`: model's prediction on clean text
- `noisy_pred`: model's prediction on noisy text
- `clean_score`: task metric score on clean text
- `noisy_score`: task metric score on noisy text

### A.6 Delta-P Computation

Compute the performance drop:

```
delta_p = clean_score - noisy_score   ∈ [0, 1]
```

Store as column `delta_p`.

### A.7 Failure Label Definition

Assign a binary failure label based on a threshold `θ = 0.20`:

```
failure_label = 1  if delta_p > 0.20   (FAILURE)
failure_label = 0  if delta_p ≤ 0.20   (ROBUST)
```

Store as column `failure_label`. This is the primary classification target for Stage B.

`delta_p` (equivalently, `noisy_score`) is retained as a secondary **regression target** for severity estimation.

**Dataset split:** 80% train / 20% test, stratified by `failure_label` to preserve class balance. Fixed seed `r=42`.

---

## Stage B — Pre-Inference Failure Classifier (Deployment)

Stage B trains lightweight classifiers on the labeled dataset produced by Stage A. At deployment time, these classifiers screen new inputs before any LLM inference is performed.

### B.1 Input Features

The 5 features extracted in Stage A.4: `sigma_prime`, `oov_rate`, `severity`, `word_count`, `alpha`.

No LLM is involved in Stage B inference — all features come from the tokenizer alone.

### B.2 Target Variable

`failure_label` (binary: 0 = ROBUST, 1 = FAILURE) — primary classification target.
`noisy_score` — secondary regression target for severity estimation.

### B.3 Models

**Classification (primary):**
- Logistic Regression (ℓ2 regularization — baseline)
- Random Forest
- XGBoost (with early stopping: patience=50, criterion=AUC-ROC)

All models output `predict_proba()` for calibrated failure probability `p̂ ∈ [0, 1]`.
Hyperparameters selected via 5-fold cross-validated grid search. All splits stratified.

**Regression (secondary):**
- Random Forest Regressor
- XGBoost Regressor
Predict `noisy_score` as a continuous severity estimate.

### B.4 Evaluation Metrics

| Task           | Target Metric   | Goal       |
|----------------|-----------------|------------|
| Classification | Macro F1        | > 0.75     |
| Classification | Precision       | Report     |
| Classification | Recall          | Report     |
| Classification | ROC-AUC         | Report     |
| Regression     | R²              | > 0.70     |
| Regression     | MAE             | Report     |

Also report: calibration curves, false positive rate.

### B.5 Deployment Logic

A routing threshold `λ` is selected on the validation set by maximizing F1:

```
if p̂ > λ  → FAIL  (intercept input, trigger safeguard, no LLM called)
if p̂ ≤ λ  → ROBUST (forward input to LLM)
```

---

## Experiments

| Experiment | Description |
|------------|-------------|
| 1 — Correlation Analysis | Pearson r and Spearman ρ between each feature and `delta_p`, stratified by task and noise family. |
| 2 — Classification Benchmark | All classifiers on held-out test set. Macro F1, precision, recall, AUC-ROC, calibration, FPR. |
| 3 — Regression Benchmark | Severity regressors. R², MAE, rank-ordering across severity levels. |
| 4 — Cross-Condition Generalization | Train on one task/noise family, test on the others. γ = F1_cross / F1_in. |
| 5 — Feature Ablation | Leave-one-out ablation reporting ΔF1 per removed feature. If exact vs. estimated `s` gap > 0.05 F1, exclude `s` and retrain. |

---

## Shared Dataset Schema

Every row in the output dataset must contain exactly these columns, in this order. **Do not rename any column.**

| Column         | Type    | Description                                              |
|----------------|---------|----------------------------------------------------------|
| `dataset`      | str     | Source dataset name (`sst2`, `squad`, `gsm8k`)           |
| `sample_id`    | int     | Index of the clean sentence within its source dataset    |
| `clean_text`   | str     | Original clean sentence                                  |
| `noisy_text`   | str     | Perturbed version                                        |
| `label`        | any     | Ground-truth label from the source dataset               |
| `noise_type`   | str     | One of the 6 noise family names                          |
| `severity`     | float   | Perturbation rate: 0.05, 0.15, or 0.30                   |
| `word_count`   | int     | Number of words in `noisy_text`                          |
| `token_count`  | int     | Number of BPE tokens in `noisy_text`                     |
| `sigma_prime`  | float   | `token_count / word_count`                               |
| `alpha`        | float   | Single-character token rate                              |
| `oov_rate`     | float   | Fraction of tokens not in `V_clean`                      |
| `clean_pred`   | any     | Model's prediction on `clean_text`                       |
| `noisy_pred`   | any     | Model's prediction on `noisy_text`                       |
| `clean_score`  | float   | Task metric score on `clean_text`                        |
| `noisy_score`  | float   | Task metric score on `noisy_text`                        |
| `delta_p`      | float   | `clean_score - noisy_score`                              |
| `failure_label`| int     | 1 if `delta_p > 0.20`, else 0                            |

---

## Repository Conventions

- **Language:** Python 3.10
- **Key libraries:** `datasets`, `transformers`, `AutoTokenizer`, `scikit-learn`, `xgboost`, `tqdm`, `logging`
- **Fixed seed:** `r=42` for all random operations (noise sampling, train/test split, model fitting)
- **Logging:** Use Python `logging` module with INFO level for all pipeline steps
- **Progress bars:** Use `tqdm` for all loops over data
- **Code structure:** Keep code modular. Each logical unit (data loading, noise injection, feature extraction, LLM inference, labeling, training, evaluation) should be in its own function or module
- **Schema:** Never rename shared columns. Column names in the schema above are fixed and shared across all modules and teammates
- **Folder structure:** Respect the project's folder layout. Do not reorganize without team agreement
- **Collaboration:** Assume multiple teammates are working in parallel. Write code that is self-contained per module and does not silently depend on global state
- **No implementation until prompt is given:** This file defines context only. Do not write production code based solely on this file without a specific task prompt

---

## LLM and Infrastructure

- **Target LLM:** Mistral-7B-Instruct v0.2 (Apache 2.0 license)
- **Tokenizer:** Mistral BPE tokenizer via `AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")`
- **Quantization:** 4-bit NF4 (`BitsAndBytesConfig`) for Colab T4 GPU (free tier)
- **Prompting:** Zero-shot only, fixed templates per task

---

## Full Pipeline Flow

```
Stage A (offline, run once):
  Load clean data (SST-2 → SQuAD → GSM8K)
      ↓
  Noise injection (6 types × 3 severities → ~18 variants/sentence)
      ↓
  Tokenizer → compute φ = [sigma_prime, oov_rate, severity, word_count, alpha]
      ↓
  Mistral-7B inference (clean + noisy) → clean_score, noisy_score
      ↓
  Compute delta_p = clean_score − noisy_score
      ↓
  Assign failure_label = 1(delta_p > 0.20)
      ↓
  Save labeled dataset (~54K rows, schema above)

Stage B (train once, deploy continuously):
  Load labeled dataset from Stage A
      ↓
  Train LR / RF / XGBoost classifiers on φ → failure_label
      ↓
  Evaluate: macro F1, precision, recall, ROC-AUC (target F1 > 0.75)
      ↓
  At deployment: new input x̃ → tokenizer → φ → classifier → p̂
      ↓
  if p̂ > λ: intercept (no LLM) | if p̂ ≤ λ: forward to LLM
```
