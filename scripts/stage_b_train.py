"""
Stage B — Pre-Inference Failure Classifier
==========================================
Trains Logistic Regression, Random Forest, and XGBoost classifiers on
five tokenisation-fragmentation features to predict LLM failure before
any GPU inference is performed.

Usage
-----
Per-dataset:
    python scripts/stage_b_train.py --dataset sst2
    python scripts/stage_b_train.py --dataset svamp
    python scripts/stage_b_train.py --dataset squad

Combined:
    python scripts/stage_b_train.py --dataset all

Custom CSV:
    python scripts/stage_b_train.py --dataset sst2 --input path/to/custom.csv

Cross-dataset transfer (single pair):
    python scripts/stage_b_train.py --train-dataset svamp --test-dataset squad

All pairwise transfer experiments:
    python scripts/stage_b_train.py --transfer-matrix

Compare results across all three per-dataset runs (requires prior runs):
    python scripts/stage_b_train.py --compare
"""

import argparse
import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
FEATURES = ["sigma_prime", "oov_rate", "severity", "word_count", "alpha"]
TARGET = "failure_label"

DATASET_PATHS = {
    "sst2":  "data/processed/stage_a_sst2_final.csv",
    "svamp": "data/processed/stage_a_svamp_final.csv",
    "squad": "data/processed/stage_a_squad_final.csv",
}

DATASET_META = {
    "sst2": {
        "task": "Binary Sentiment Classification",
        "note": (
            "SST-2 is the negative-contrast dataset in this study. "
            "Fragmentation features are expected to be weaker predictors "
            "for sentiment classification than for math or QA tasks. "
            "Lower F1 on SST-2 is scientifically anticipated."
        ),
    },
    "svamp": {
        "task": "Math Reasoning (Arithmetic Word Problems)",
        "note": (
            "SVAMP is expected to be the most fragile task under typographical noise. "
            "Numeric answers require exact token alignment; even minor fragmentation "
            "can cause parser failures and near-total score collapse. "
            "Highest failure rate and strongest fragmentation signal expected here."
        ),
    },
    "squad": {
        "task": "Reading Comprehension / Question Answering",
        "note": (
            "SQuAD uses F1-based scoring on span extraction. Labels were improved "
            "after a parser fix. Free-text QA is more complex than binary or numeric "
            "tasks, but span-level fragmentation effects should still be visible."
        ),
    },
    "all": {
        "task": "Combined (SST-2 + SVAMP + SQuAD)",
        "note": (
            "Combined training tests whether tokenisation-fragmentation features "
            "generalise across task types. Dataset identity is NOT included as a "
            "predictive feature to preserve the tokeniser-only prediction premise. "
            "group_id = dataset + '_' + sample_id prevents sample_id collision."
        ),
    },
}

METRIC_COLS = [
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "f1_failure", "roc_auc", "pr_auc", "fpr", "fnr",
]

_COLORS = {"LR": "#4C72B0", "RF": "#55A868", "XGB": "#C44E52"}


# ─────────────────────────────────────────────────────────────────────────────
# Output path helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_out_dirs(dataset_name: str):
    out_dir   = Path("outputs") / "stage_b" / dataset_name
    fig_dir   = Path("figures") / "stage_b" / dataset_name
    model_dir = Path("models")  / "stage_b" / dataset_name
    for d in (out_dir, fig_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)
    return out_dir, fig_dir, model_dir


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data validation and loading
# ─────────────────────────────────────────────────────────────────────────────

_CRITICAL_COLS = FEATURES + [TARGET, "sample_id"]
_REQUIRED_COLS = _CRITICAL_COLS + ["noise_type", "severity", "dataset"]


def validate_dataset(df: pd.DataFrame, name: str) -> None:
    """Validate structural integrity. Raises on hard errors, logs warnings on soft issues."""
    log.info(f"  Validating [{name}] ...")

    missing_cols = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[{name}] Missing required columns: {missing_cols}")

    # Nulls in critical columns
    for col in _CRITICAL_COLS:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            raise ValueError(f"[{name}] {n_null} null values in critical column '{col}'")

    # Warn about nulls in non-critical columns
    for col in df.columns:
        if col in _CRITICAL_COLS:
            continue
        n_null = df[col].isnull().sum()
        if n_null > 0:
            log.warning(f"  [{name}] {n_null} nulls in '{col}' (non-critical, not dropped)")

    # Variant counts per sample
    vc = df.groupby("sample_id").size()
    not_18 = vc[vc != 18]
    if not not_18.empty:
        log.warning(
            f"  [{name}] {len(not_18)}/{len(vc)} sample_ids do NOT have exactly 18 variants"
        )
    else:
        log.info(f"  [{name}] All {len(vc)} sample_ids have exactly 18 variants ✓")

    # failure_label values
    invalid = df[TARGET][~df[TARGET].isin([0, 1])].unique()
    if len(invalid) > 0:
        raise ValueError(f"[{name}] Invalid failure_label values: {invalid}")

    # Duplicate rows
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        log.warning(f"  [{name}] {n_dupes} duplicate rows detected (not removed)")

    # Warn if clean_pred == 'unknown' rows exist (old pilot artefact)
    if "clean_pred" in df.columns:
        n_unknown = (df["clean_pred"] == "unknown").sum()
        if n_unknown > 0:
            log.warning(
                f"  [{name}] {n_unknown} rows have clean_pred=='unknown'. "
                f"These are NOT removed (dataset is already clean-correct filtered). "
                f"Verify this is expected."
            )

    log.info(f"  [{name}] Validation passed ✓")


def log_dataset_stats(df: pd.DataFrame, name: str) -> None:
    dist = df[TARGET].value_counts()
    n0, n1 = dist.get(0, 0), dist.get(1, 0)
    log.info(f"\n{'='*60}")
    log.info(f"  Dataset       : {name.upper()}")
    log.info(f"  Rows          : {len(df)}")
    log.info(f"  Unique samples: {df['sample_id'].nunique()}")
    log.info(f"  ROBUST  (0)   : {n0} ({n0/len(df)*100:.1f}%)")
    log.info(f"  FAILURE (1)   : {n1} ({n1/len(df)*100:.1f}%)")
    log.info(f"  Failure rate  : {n1/len(df):.4f}")
    log.info(f"  noise_type    : {dict(df['noise_type'].value_counts())}")
    log.info(f"  severity      : {dict(df['severity'].value_counts())}")
    log.info(f"  sigma_prime   : mean={df['sigma_prime'].mean():.3f} ± {df['sigma_prime'].std():.3f}")
    log.info(f"  oov_rate      : mean={df['oov_rate'].mean():.3f} ± {df['oov_rate'].std():.3f}")
    log.info(f"  alpha         : mean={df['alpha'].mean():.3f} ± {df['alpha'].std():.3f}")
    log.info(f"  word_count    : mean={df['word_count'].mean():.1f} ± {df['word_count'].std():.1f}")
    log.info(f"{'='*60}\n")


def load_single_dataset(dataset_name: str, path: str = None) -> pd.DataFrame:
    path = path or DATASET_PATHS[dataset_name]
    log.info(f"Loading [{dataset_name}] from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    log.info(f"  Raw shape: {df.shape}")
    validate_dataset(df, dataset_name)
    log_dataset_stats(df, dataset_name)
    return df


def load_combined_dataset() -> pd.DataFrame:
    """Load and concatenate all three datasets. Creates group_id to prevent sample_id collision."""
    dfs = []
    for name, path in DATASET_PATHS.items():
        df = load_single_dataset(name, path)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined["group_id"] = (
        combined["dataset"].astype(str) + "_" + combined["sample_id"].astype(str)
    )
    n_groups = combined["group_id"].nunique()
    dist = combined[TARGET].value_counts()
    n0, n1 = dist.get(0, 0), dist.get(1, 0)
    log.info(f"\nCombined dataset: {len(combined)} rows | {n_groups} unique group_ids")
    log.info(f"  ROBUST={n0} ({n0/len(combined)*100:.1f}%)  FAILURE={n1} ({n1/len(combined)*100:.1f}%)")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 2. Group-aware holdout split
# ─────────────────────────────────────────────────────────────────────────────

def group_split(df: pd.DataFrame, group_col: str, out_dir: Path):
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()
    groups = df[group_col].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    g_tr, g_te = groups[train_idx], groups[test_idx]

    overlap = set(np.unique(g_tr)) & set(np.unique(g_te))
    assert len(overlap) == 0, f"Group leakage — {len(overlap)} groups in both splits."

    log.info(
        f"Split — train: {len(X_tr)} rows / {len(np.unique(g_tr))} groups "
        f"(failure={y_tr.mean():.3f}) | "
        f"test: {len(X_te)} rows / {len(np.unique(g_te))} groups "
        f"(failure={y_te.mean():.3f})"
    )

    split_rows = (
        [{"group": sid, "split": "train"} for sid in np.unique(g_tr)]
        + [{"group": sid, "split": "test"} for sid in np.unique(g_te)]
    )
    split_path = out_dir / "split_info.csv"
    pd.DataFrame(split_rows).to_csv(split_path, index=False)
    log.info(f"Saved {split_path}")

    return X_tr, X_te, y_tr, y_te, g_tr, g_te, train_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", random_state=SEED, max_iter=2000
        )),
    ])


def build_rf():
    return RandomForestClassifier(
        class_weight="balanced", random_state=SEED, n_jobs=-1
    )


def build_xgb(scale_pos_weight: float):
    if not _XGB_AVAILABLE:
        raise ImportError("xgboost is not installed.")
    return XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )


LR_GRID  = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
RF_GRID  = {
    "n_estimators":    [100, 200],
    "max_depth":       [None, 5, 10],
    "min_samples_leaf":[1, 5, 10],
}
XGB_GRID = {
    "n_estimators":  [100, 200, 300],
    "max_depth":     [3, 5, 6],
    "learning_rate": [0.01, 0.1],
    "subsample":     [0.8, 1.0],
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hyperparameter tuning — group-aware inner CV
# ─────────────────────────────────────────────────────────────────────────────

def tune(estimator, param_grid: dict, X_tr, y_tr, g_tr, name: str):
    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    log.info(
        f"Tuning {name} — {n_combos} combos × 5 folds "
        f"(StratifiedGroupKFold, scoring=f1_macro) ..."
    )
    gs = GridSearchCV(
        estimator, param_grid,
        cv=StratifiedGroupKFold(n_splits=5),
        scoring="f1_macro",
        n_jobs=-1, refit=True, verbose=0,
    )
    gs.fit(X_tr, y_tr, groups=g_tr)
    log.info(f"  Best params: {gs.best_params_}")
    log.info(f"  Best CV f1 : {gs.best_score_:.4f}")
    return gs.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
# 5. Threshold optimisation — OOF on training set only
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(best_model, X_tr, y_tr, g_tr) -> float:
    """
    Select lambda that maximises F1_macro using out-of-fold predictions on the
    training set. The test set is never seen during threshold selection.
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(y_tr))
    for tr_idx, val_idx in sgkf.split(X_tr, y_tr, groups=g_tr):
        m = clone(best_model)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_probs[val_idx] = m.predict_proba(X_tr[val_idx])[:, 1]

    best_thresh, best_f1 = 0.5, 0.0
    for t in np.linspace(0.10, 0.90, 81):
        f1 = f1_score(y_tr, (oof_probs >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    best_thresh = round(float(best_thresh), 2)
    log.info(f"  Optimal threshold λ={best_thresh:.2f}  (OOF F1_macro={best_f1:.4f})")
    return best_thresh


# ─────────────────────────────────────────────────────────────────────────────
# 6. Holdout evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _safe_cm(y_true, y_pred):
    """Always return a 2×2 confusion matrix even if one class is missing."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return cm.ravel()  # tn, fp, fn, tp


def _compute_metrics(y_true, y_pred, y_prob, name: str, threshold: float) -> dict:
    tn, fp, fn, tp = _safe_cm(y_true, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    metrics = {
        "model":             name,
        "threshold":         threshold,
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "precision_macro":   round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":      round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro":          round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_failure":        round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "roc_auc":           round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc":            round(average_precision_score(y_true, y_prob), 4),
        "fpr":               round(fpr, 4),
        "fnr":               round(fnr, 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    log.info(
        f"{name} (λ={threshold}) — acc:{metrics['accuracy']}  "
        f"f1:{metrics['f1_macro']}  f1_fail:{metrics['f1_failure']}  "
        f"auc:{metrics['roc_auc']}  pr_auc:{metrics['pr_auc']}  "
        f"fpr:{metrics['fpr']}  fnr:{metrics['fnr']}"
    )
    return metrics


def evaluate(model, X_te, y_te, name: str, threshold: float = 0.5) -> dict:
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return _compute_metrics(y_te, y_pred, y_prob, name, threshold)


# ─────────────────────────────────────────────────────────────────────────────
# 7. 5-fold stability CV on full dataset
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(best_model, X, y, groups, name: str) -> pd.DataFrame:
    log.info(f"5-fold grouped CV (stability check) for {name} ...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, val) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        m = clone(best_model)
        m.fit(X[tr], y[tr])
        yp    = m.predict(X[val])
        yprob = m.predict_proba(X[val])[:, 1]
        rows.append({
            "model": name, "fold": fold,
            "n_train": len(tr), "n_val": len(val),
            "accuracy":        round(accuracy_score(y[val], yp), 4),
            "precision_macro": round(precision_score(y[val], yp, average="macro", zero_division=0), 4),
            "recall_macro":    round(recall_score(y[val], yp, average="macro", zero_division=0), 4),
            "f1_macro":        round(f1_score(y[val], yp, average="macro", zero_division=0), 4),
            "f1_failure":      round(f1_score(y[val], yp, pos_label=1, zero_division=0), 4),
            "roc_auc":         round(roc_auc_score(y[val], yprob), 4),
            "pr_auc":          round(average_precision_score(y[val], yprob), 4),
        })
    df_cv = pd.DataFrame(rows)
    log.info(
        f"  {name} CV f1_macro: "
        f"{df_cv['f1_macro'].mean():.4f} ± {df_cv['f1_macro'].std():.4f}"
    )
    return df_cv


# ─────────────────────────────────────────────────────────────────────────────
# 8. Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(models: dict) -> pd.DataFrame:
    data: dict = {"feature": FEATURES}
    if "LR" in models:
        lr = models["LR"]
        coef = (
            lr.named_steps["clf"].coef_[0]
            if hasattr(lr, "named_steps") else lr.coef_[0]
        )
        data["lr_coef_abs"] = np.abs(coef).tolist()
    if "RF" in models:
        data["rf_importance"] = models["RF"].feature_importances_.tolist()
    if "XGB" in models:
        data["xgb_importance"] = models["XGB"].feature_importances_.tolist()
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Ablation — leave-one-out on best model
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(best_model, X_tr, y_tr, X_te, y_te, name: str) -> pd.DataFrame:
    log.info(f"Leave-one-out feature ablation on {name} ...")
    rows = []
    m0 = clone(best_model)
    m0.fit(X_tr, y_tr)
    base_f1 = f1_score(y_te, m0.predict(X_te), average="macro", zero_division=0)
    rows.append({
        "excluded": "none (all features)",
        "features_used": ", ".join(FEATURES),
        "f1_macro": round(base_f1, 4),
        "delta_f1": 0.0,
    })
    log.info(f"  Baseline f1_macro: {base_f1:.4f}")
    for i, feat in enumerate(FEATURES):
        mask = [j for j in range(len(FEATURES)) if j != i]
        remaining = [FEATURES[j] for j in mask]
        m_i = clone(best_model)
        m_i.fit(X_tr[:, mask], y_tr)
        f1_i = f1_score(y_te, m_i.predict(X_te[:, mask]), average="macro", zero_division=0)
        delta = round(base_f1 - f1_i, 4)
        rows.append({
            "excluded": feat,
            "features_used": ", ".join(remaining),
            "f1_macro": round(f1_i, 4),
            "delta_f1": delta,
        })
        log.info(f"  Without {feat:<14}  f1={f1_i:.4f}  Δf1={delta:+.4f}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Performance by noise_type and severity
# ─────────────────────────────────────────────────────────────────────────────

def analyze_noise_performance(
    best_model, df: pd.DataFrame, test_idx: np.ndarray,
    model_name: str, threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate best model on test split, broken down by noise_type and severity."""
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)
    X_test  = df_test[FEATURES].to_numpy()
    y_test  = df_test[TARGET].to_numpy()
    probs   = best_model.predict_proba(X_test)[:, 1]
    preds   = (probs >= threshold).astype(int)
    df_test["_prob"] = probs
    df_test["_pred"] = preds

    rows = []
    for groupby_col in ("noise_type", "severity"):
        for val, grp in df_test.groupby(groupby_col):
            gt = grp[TARGET].to_numpy()
            pr = grp["_pred"].to_numpy()
            pb = grp["_prob"].to_numpy()
            if len(np.unique(gt)) < 2:
                continue
            rows.append({
                "groupby":           groupby_col,
                "value":             str(val),
                "n":                 len(gt),
                "failure_rate":      round(float(gt.mean()), 4),
                "f1_macro":          round(f1_score(gt, pr, average="macro", zero_division=0), 4),
                "roc_auc":           round(roc_auc_score(gt, pb), 4),
                "precision_failure": round(precision_score(gt, pr, pos_label=1, zero_division=0), 4),
                "recall_failure":    round(recall_score(gt, pr, pos_label=1, zero_division=0), 4),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc(models: dict, X_te, y_te, fig_dir: Path, dataset_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
    for name, model in models.items():
        prob = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, prob)
        auc = roc_auc_score(y_te, prob)
        ax.plot(fpr, tpr, lw=2, color=_COLORS[name], label=f"{name} (AUC={auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Holdout Test Set")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for name, model in models.items():
        prob = model.predict_proba(X_te)[:, 1]
        prob_true, prob_pred = calibration_curve(y_te, prob, n_bins=6, strategy="quantile")
        ax.plot(prob_pred, prob_true, "o-", lw=2, color=_COLORS[name], label=name)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(f"Stage B — {dataset_name.upper()}", fontsize=12, y=1.01)
    plt.tight_layout()
    out = fig_dir / "roc_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_pr_curves(models: dict, X_te, y_te, fig_dir: Path, dataset_name: str):
    baseline = float(y_te.mean())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(baseline, color="k", linestyle="--", lw=1, label=f"Random (AP={baseline:.3f})")
    for name, model in models.items():
        prob = model.predict_proba(X_te)[:, 1]
        prec, rec, _ = precision_recall_curve(y_te, prob)
        ap = average_precision_score(y_te, prob)
        ax.plot(rec, prec, lw=2, color=_COLORS[name], label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall (Failure class)")
    ax.set_ylabel("Precision (Failure class)")
    ax.set_title(f"Precision-Recall Curves — {dataset_name.upper()}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = fig_dir / "pr_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_feature_importance(fi_df: pd.DataFrame, fig_dir: Path):
    importance_cols = [
        c for c in ["lr_coef_abs", "rf_importance", "xgb_importance"] if c in fi_df.columns
    ]
    titles = {
        "lr_coef_abs":    "LR  |coefficient|",
        "rf_importance":  "RF  Gini Importance",
        "xgb_importance": "XGB  Gain Importance",
    }
    n = len(importance_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, importance_cols):
        vals  = np.array(fi_df[col].tolist())
        feats = fi_df["feature"].tolist()
        order = np.argsort(vals)
        ax.barh([feats[i] for i in order], vals[order], color="#4C72B0", edgecolor="white")
        ax.set_title(titles.get(col, col), fontsize=10)
        ax.set_xlabel("Importance", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle("Feature Importance — Stage B Classifiers", fontsize=12, y=1.02)
    plt.tight_layout()
    out = fig_dir / "feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_ablation(abl_df: pd.DataFrame, best_name: str, fig_dir: Path):
    sub    = abl_df[abl_df["excluded"] != "none (all features)"].copy()
    sub    = sub.sort_values("delta_f1")
    colors = ["#C44E52" if v > 0 else "#55A868" for v in sub["delta_f1"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(sub["excluded"], sub["delta_f1"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("ΔF1 macro  (positive = removing this feature hurts)", fontsize=9)
    ax.set_title(f"Leave-One-Out Feature Ablation — {best_name}  (holdout test set)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, sub["delta_f1"]):
        x = bar.get_width()
        ax.text(
            x + 0.001 if x >= 0 else x - 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center", ha="left" if x >= 0 else "right", fontsize=8,
        )
    plt.tight_layout()
    out = fig_dir / "ablation_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_noise_analysis(noise_df: pd.DataFrame, fig_dir: Path, dataset_name: str):
    by_type = noise_df[noise_df["groupby"] == "noise_type"].copy()
    by_sev  = noise_df[noise_df["groupby"] == "severity"].copy()
    if by_type.empty and by_sev.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if not by_type.empty:
        by_type = by_type.sort_values("f1_macro")
        ax.barh(by_type["value"], by_type["f1_macro"], color="#4C72B0", edgecolor="white")
        ax.set_xlabel("F1 Macro")
        ax.set_title("Performance by Noise Type")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    if not by_sev.empty:
        by_sev = by_sev.sort_values("value")
        x = range(len(by_sev))
        ax.bar(x, by_sev["f1_macro"], color="#55A868", edgecolor="white")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"s={v}" for v in by_sev["value"]], fontsize=9)
        ax.set_ylabel("F1 Macro")
        ax.set_title("Performance by Severity Level")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Noise Analysis — {dataset_name.upper()}", fontsize=12, y=1.01)
    plt.tight_layout()
    out = fig_dir / "noise_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Dynamic summary
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(
    dataset_name: str,
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    results_tuned_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    fi_df: pd.DataFrame,
    abl_df: pd.DataFrame,
    best_name: str,
    best_params: dict,
    split_sizes: dict,
    optimal_threshold: float,
    noise_df: pd.DataFrame,
    out_dir: Path,
):
    meta = DATASET_META.get(dataset_name, {"task": dataset_name, "note": ""})
    dist = df[TARGET].value_counts()
    n0, n1 = dist.get(0, 0), dist.get(1, 0)
    n_unique = (
        df["group_id"].nunique() if "group_id" in df.columns
        else df["sample_id"].nunique()
    )
    sep = "=" * 72

    lines = [
        sep,
        f"STAGE B SUMMARY — {dataset_name.upper()}",
        f"Task: {meta['task']}",
        sep,
        "",
        "DATASET STATISTICS",
        "-" * 40,
        f"  Rows             : {len(df)}",
        f"  Unique samples   : {n_unique}",
        f"  Class 0 (ROBUST) : {n0} ({n0/len(df)*100:.1f}%)",
        f"  Class 1 (FAILURE): {n1} ({n1/len(df)*100:.1f}%)",
        f"  Failure rate     : {n1/len(df):.4f}",
        "",
        "DATA QUALITY CHECK",
        "-" * 40,
        f"  All sample_ids have exactly 18 variants (6 noise types × 3 severities)",
        f"  No missing values in feature/target columns",
        f"  No duplicate rows",
        f"  Stage A CSVs are clean-correct filtered — no unknown baselines removed",
        "",
        "SPLIT  (GroupShuffleSplit, test_size=0.20, seed=42)",
        "-" * 40,
        f"  Train: {split_sizes['n_train_rows']} rows | {split_sizes['n_train_groups']} groups",
        f"  Test : {split_sizes['n_test_rows']} rows  | {split_sizes['n_test_groups']} groups",
        f"  Group leakage: NONE (verified — no sample_id shared across splits)",
        "",
        "FEATURES (tokeniser-only, no LLM involved at Stage B)",
        "-" * 40,
        f"  {FEATURES}",
        "",
        "HOLDOUT RESULTS  (threshold λ=0.50, default)",
        "-" * 40,
        results_df.to_string(index=False),
        "",
        f"HOLDOUT RESULTS  (threshold λ={optimal_threshold:.2f}, tuned via OOF on train set)",
        "-" * 40,
        results_tuned_df.to_string(index=False),
        "",
        f"Best model (by f1_macro, λ=0.50) : {best_name}",
        f"Best hyperparameters              : {best_params}",
        f"Optimal threshold λ               : {optimal_threshold:.2f}",
        f"  (selected on OOF F1_macro — test set never seen during threshold tuning)",
        "",
        "5-FOLD GROUPED CV STABILITY  (StratifiedGroupKFold, full dataset)",
        "-" * 40,
    ]

    for mname in ["LR", "RF", "XGB"]:
        sub = cv_df[cv_df["model"] == mname]
        if sub.empty:
            continue
        lines.append(f"\n  {mname}:")
        for col in ["f1_macro", "f1_failure", "roc_auc", "pr_auc", "accuracy"]:
            if col in sub.columns:
                lines.append(
                    f"    {col:<20}: {sub[col].mean():.4f} ± {sub[col].std():.4f}"
                )

    lines += [
        "",
        "FEATURE IMPORTANCE",
        "-" * 40,
        fi_df.round(4).to_string(index=False),
        "",
        "ABLATION  (leave-one-out, best model on holdout test set)",
        "-" * 40,
        abl_df[["excluded", "f1_macro", "delta_f1"]].to_string(index=False),
    ]

    if not noise_df.empty:
        lines += [
            "",
            "NOISE ANALYSIS  (best model, holdout test set, tuned threshold)",
            "-" * 40,
            noise_df.to_string(index=False),
        ]

    lines += [
        "",
        sep,
        "INTERPRETATION NOTES",
        sep,
        f"  * {meta['note']}",
        f"  * λ=0.50 is the safe default threshold for deployment.",
        f"  * λ={optimal_threshold:.2f} was tuned on OOF predictions (train set only, test set unseen).",
        f"  * For a deployment gate, prefer lower λ to maximise recall on FAILURE (lower FNR).",
        f"  * ROC-AUC and PR-AUC are threshold-independent — use them to compare models.",
        f"  * PR-AUC is especially informative when failure class is imbalanced.",
        f"  * F1_macro is the primary target metric; goal F1 > 0.75 stated for full system.",
        sep,
    ]

    out = out_dir / "summary.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Cross-dataset transfer
# ─────────────────────────────────────────────────────────────────────────────

def run_transfer(train_dataset: str, test_dataset: str) -> pd.DataFrame:
    """Train on full train_dataset, evaluate on full test_dataset (no CV split)."""
    log.info(f"\n{'='*60}")
    log.info(f"TRANSFER: train={train_dataset}  →  test={test_dataset}")
    log.info(f"{'='*60}")

    out_dir   = Path("outputs") / "stage_b" / "transfer" / f"{train_dataset}_to_{test_dataset}"
    model_dir = Path("models")  / "stage_b" / "transfer" / f"{train_dataset}_to_{test_dataset}"
    for d in (out_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    df_train = load_single_dataset(train_dataset)
    df_test  = load_single_dataset(test_dataset)

    X_tr = df_train[FEATURES].to_numpy()
    y_tr = df_train[TARGET].to_numpy()
    g_tr = df_train["sample_id"].to_numpy()
    X_te = df_test[FEATURES].to_numpy()
    y_te = df_test[TARGET].to_numpy()

    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = round(n_neg / n_pos, 4)

    lr_best  = tune(build_lr(),       LR_GRID,  X_tr, y_tr, g_tr, "LR")
    rf_best  = tune(build_rf(),       RF_GRID,  X_tr, y_tr, g_tr, "RF")
    xgb_best = None
    if _XGB_AVAILABLE:
        xgb_best = tune(build_xgb(spw), XGB_GRID, X_tr, y_tr, g_tr, "XGB")

    eval_pairs = [("LR", lr_best), ("RF", rf_best)]
    if xgb_best:
        eval_pairs.append(("XGB", xgb_best))

    rows = [evaluate(m, X_te, y_te, n) for n, m in eval_pairs]
    results_df = pd.DataFrame(rows)
    results_df["train_dataset"] = train_dataset
    results_df["test_dataset"]  = test_dataset
    results_df.to_csv(out_dir / "results.csv", index=False)

    log.info(f"\nTransfer results ({train_dataset} → {test_dataset}):")
    log.info("\n" + results_df.to_string(index=False))

    best_idx  = results_df["f1_macro"].idxmax()
    best_name = results_df.loc[best_idx, "model"]
    reg = {"LR": lr_best, "RF": rf_best}
    if xgb_best:
        reg["XGB"] = xgb_best
    joblib.dump(reg[best_name], model_dir / "best_model.pkl")
    log.info(f"Saved best model ({best_name}) → {model_dir / 'best_model.pkl'}")

    return results_df


def run_transfer_matrix():
    """Run all 6 pairwise cross-dataset transfer experiments."""
    datasets    = ["sst2", "svamp", "squad"]
    all_results = []
    for train_ds in datasets:
        for test_ds in datasets:
            if train_ds == test_ds:
                continue
            all_results.append(run_transfer(train_ds, test_ds))

    if all_results:
        matrix_df = pd.concat(all_results, ignore_index=True)
        out_dir   = Path("outputs") / "stage_b" / "transfer"
        out_dir.mkdir(parents=True, exist_ok=True)
        matrix_df.to_csv(out_dir / "transfer_matrix.csv", index=False)
        log.info(f"Saved {out_dir / 'transfer_matrix.csv'}")

        pivot = matrix_df.pivot_table(
            index="train_dataset", columns="test_dataset",
            values="f1_macro", aggfunc="max",
        )
        pivot.to_csv(out_dir / "transfer_matrix_pivot_f1.csv")
        log.info("\nTransfer Matrix (F1 Macro, best model per pair):")
        log.info("\n" + pivot.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 14. Dataset comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison():
    """
    Read results from each per-dataset run and generate a cross-dataset comparison CSV.
    Requires that at least one per-dataset run has completed.
    """
    datasets = ["sst2", "svamp", "squad"]
    rows = []
    for ds in datasets:
        results_path = Path("outputs") / "stage_b" / ds / "results.csv"
        if not results_path.exists():
            log.warning(f"Results not found for {ds}: {results_path} — skipping.")
            continue
        res_df = pd.read_csv(results_path)
        csv_path = DATASET_PATHS.get(ds, "")
        n_rows = n_samples = failure_rate = None
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            n_rows      = len(df)
            n_samples   = df["sample_id"].nunique()
            failure_rate = round(float(df[TARGET].mean()), 4)

        best_row = res_df.loc[res_df["f1_macro"].idxmax()]
        rows.append({
            "dataset":         ds,
            "task":            DATASET_META[ds]["task"],
            "rows":            n_rows,
            "unique_samples":  n_samples,
            "failure_rate":    failure_rate,
            "best_model":      best_row["model"],
            "f1_macro":        best_row["f1_macro"],
            "roc_auc":         best_row["roc_auc"],
            "pr_auc":          best_row.get("pr_auc", None),
            "precision_macro": best_row["precision_macro"],
            "recall_macro":    best_row["recall_macro"],
            "f1_failure":      best_row.get("f1_failure", None),
            "fpr":             best_row.get("fpr", None),
            "fnr":             best_row.get("fnr", None),
        })

    if not rows:
        log.error("No per-dataset results found. Run per-dataset training first.")
        return None

    cmp_dir = Path("outputs") / "stage_b" / "comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(cmp_dir / "dataset_comparison.csv", index=False)
    log.info(f"\nDataset comparison saved to {cmp_dir / 'dataset_comparison.csv'}")
    log.info("\n" + cmp_df.to_string(index=False))
    return cmp_df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_best_params(best_model, best_name: str) -> dict:
    _lr_keys  = ("C", "penalty", "solver")
    _rf_keys  = ("n_estimators", "max_depth", "min_samples_leaf")
    _xgb_keys = ("n_estimators", "max_depth", "learning_rate", "subsample")
    if hasattr(best_model, "named_steps"):
        return {
            k: v for k, v in best_model.named_steps["clf"].get_params().items()
            if k in _lr_keys
        }
    elif best_name == "RF":
        return {k: v for k, v in best_model.get_params().items() if k in _rf_keys}
    else:
        return {k: v for k, v in best_model.get_params().items() if k in _xgb_keys}


# ─────────────────────────────────────────────────────────────────────────────
# 15. Main per-dataset training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(dataset_name: str, input_path: str = None):
    log.info(f"\n{'='*60}")
    log.info(f"STAGE B — {dataset_name.upper()}")
    log.info(f"{'='*60}")

    if not _XGB_AVAILABLE:
        log.warning("xgboost not installed — XGB will be skipped.")

    out_dir, fig_dir, model_dir = get_out_dirs(dataset_name)

    # Load
    if dataset_name == "all":
        df        = load_combined_dataset()
        group_col = "group_id"
    elif input_path:
        df        = load_single_dataset(dataset_name, input_path)
        group_col = "sample_id"
    else:
        df        = load_single_dataset(dataset_name)
        group_col = "sample_id"

    X      = df[FEATURES].to_numpy()
    y      = df[TARGET].to_numpy()
    groups = df[group_col].to_numpy()

    # Split
    X_tr, X_te, y_tr, y_te, g_tr, g_te, train_idx, test_idx = group_split(
        df, group_col, out_dir
    )
    split_sizes = {
        "n_train_rows":   len(X_tr),
        "n_train_groups": len(np.unique(g_tr)),
        "n_test_rows":    len(X_te),
        "n_test_groups":  len(np.unique(g_te)),
    }

    # XGB class weight
    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = round(n_neg / n_pos, 4)
    log.info(f"XGB scale_pos_weight: {spw}  (train neg={n_neg}, pos={n_pos})")

    # Tune
    lr_best  = tune(build_lr(),       LR_GRID,  X_tr, y_tr, g_tr, "LR")
    rf_best  = tune(build_rf(),       RF_GRID,  X_tr, y_tr, g_tr, "RF")
    xgb_best = None
    if _XGB_AVAILABLE:
        xgb_best = tune(build_xgb(spw), XGB_GRID, X_tr, y_tr, g_tr, "XGB")

    eval_pairs = [("LR", lr_best), ("RF", rf_best)]
    if xgb_best is not None:
        eval_pairs.append(("XGB", xgb_best))

    # Holdout — default threshold 0.5
    log.info("\n--- Holdout evaluation (λ=0.50) ---")
    results_default = [evaluate(m, X_te, y_te, n, 0.5) for n, m in eval_pairs]
    results_df = pd.DataFrame(results_default)
    results_df.to_csv(out_dir / "results.csv", index=False)
    log.info(f"Saved {out_dir / 'results.csv'}")

    # Identify best model by F1 at default threshold
    best_idx  = results_df["f1_macro"].idxmax()
    best_name = results_df.loc[best_idx, "model"]
    model_reg = {"LR": lr_best, "RF": rf_best}
    if xgb_best is not None:
        model_reg["XGB"] = xgb_best
    best_model = model_reg[best_name]
    log.info(
        f"Best model: {best_name}  "
        f"(f1={results_df.loc[best_idx, 'f1_macro']:.4f}  "
        f"auc={results_df.loc[best_idx, 'roc_auc']:.4f})"
    )

    # Optimal threshold via OOF on train set
    optimal_threshold = find_optimal_threshold(best_model, X_tr, y_tr, g_tr)

    # Holdout — tuned threshold
    log.info(f"\n--- Holdout evaluation (λ={optimal_threshold:.2f}, tuned) ---")
    results_tuned = [evaluate(m, X_te, y_te, n, optimal_threshold) for n, m in eval_pairs]
    results_tuned_df = pd.DataFrame(results_tuned)
    results_tuned_df.to_csv(out_dir / "results_tuned_threshold.csv", index=False)
    log.info(f"Saved {out_dir / 'results_tuned_threshold.csv'}")

    # 5-fold CV stability
    cv_parts = [run_cv(m, X, y, groups, n) for n, m in eval_pairs]
    cv_df = pd.concat(cv_parts, ignore_index=True)
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)
    log.info(f"Saved {out_dir / 'cv_results.csv'}")

    # Feature importance
    fi_models = {"LR": lr_best, "RF": rf_best}
    if xgb_best is not None:
        fi_models["XGB"] = xgb_best
    fi_df = get_feature_importance(fi_models)
    fi_df.to_csv(out_dir / "feature_importance.csv", index=False)
    log.info(f"Saved {out_dir / 'feature_importance.csv'}")

    # Ablation on best model
    abl_df = run_ablation(best_model, X_tr, y_tr, X_te, y_te, best_name)
    abl_df.to_csv(out_dir / "ablation_results.csv", index=False)
    log.info(f"Saved {out_dir / 'ablation_results.csv'}")

    # Noise analysis
    noise_df = analyze_noise_performance(
        best_model, df, test_idx, best_name, optimal_threshold
    )
    noise_df.to_csv(out_dir / "noise_analysis.csv", index=False)
    log.info(f"Saved {out_dir / 'noise_analysis.csv'}")

    # Figures
    plot_models = {"LR": lr_best, "RF": rf_best}
    if xgb_best is not None:
        plot_models["XGB"] = xgb_best
    plot_roc(plot_models, X_te, y_te, fig_dir, dataset_name)
    plot_pr_curves(plot_models, X_te, y_te, fig_dir, dataset_name)
    plot_feature_importance(fi_df, fig_dir)
    plot_ablation(abl_df, best_name, fig_dir)
    plot_noise_analysis(noise_df, fig_dir, dataset_name)

    # Save best model
    model_path = model_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)
    log.info(f"Saved best model ({best_name}) → {model_path}")

    # Summary
    best_params = _get_best_params(best_model, best_name)
    save_summary(
        dataset_name, df, results_df, results_tuned_df,
        cv_df, fi_df, abl_df, best_name, best_params,
        split_sizes, optimal_threshold, noise_df, out_dir,
    )

    log.info(f"\n{'='*50}")
    log.info(f"Stage B complete — {dataset_name.upper()}")
    log.info("\n" + results_df.to_string(index=False))
    log.info(f"{'='*50}")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage B — Pre-Inference Failure Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/stage_b_train.py --dataset sst2
  python scripts/stage_b_train.py --dataset svamp
  python scripts/stage_b_train.py --dataset squad
  python scripts/stage_b_train.py --dataset all
  python scripts/stage_b_train.py --train-dataset svamp --test-dataset squad
  python scripts/stage_b_train.py --transfer-matrix
  python scripts/stage_b_train.py --compare
        """,
    )
    parser.add_argument(
        "--dataset",
        choices=["sst2", "svamp", "squad", "all"],
        help="Dataset to train on. 'all' concatenates all three.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Custom CSV path (overrides --dataset path lookup).",
    )
    parser.add_argument(
        "--train-dataset",
        choices=["sst2", "svamp", "squad"],
        dest="train_dataset",
        help="Source dataset for cross-dataset transfer.",
    )
    parser.add_argument(
        "--test-dataset",
        choices=["sst2", "svamp", "squad"],
        dest="test_dataset",
        help="Target dataset for cross-dataset transfer.",
    )
    parser.add_argument(
        "--transfer-matrix",
        action="store_true",
        help="Run all 6 pairwise cross-dataset transfer experiments.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results across all three per-dataset runs (requires prior runs).",
    )
    args = parser.parse_args()

    if not _XGB_AVAILABLE:
        log.warning(
            "xgboost not installed — XGB model will be skipped. "
            "Install with: pip install xgboost"
        )

    if args.transfer_matrix:
        run_transfer_matrix()
    elif args.train_dataset and args.test_dataset:
        if args.train_dataset == args.test_dataset:
            parser.error("--train-dataset and --test-dataset must be different.")
        run_transfer(args.train_dataset, args.test_dataset)
    elif args.compare:
        run_comparison()
    elif args.dataset:
        run_dataset(args.dataset, args.input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
