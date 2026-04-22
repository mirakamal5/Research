"""
Stage B — Pre-Inference Failure Classifier
Person 3

Trains Logistic Regression, Random Forest, and XGBoost classifiers on
tokenization-fragmentation features to predict LLM failure before inference.

Input:   data/stage_a_final.csv
Outputs:
    outputs/results.csv              holdout evaluation per model
    outputs/cv_results.csv           5-fold grouped CV stability metrics
    outputs/feature_importance.csv   feature importances (LR, RF, XGB)
    outputs/ablation_results.csv     leave-one-out ablation on best model
    outputs/split_info.csv           which sample_ids went to train vs test
    outputs/pilot_summary.txt        human-readable summary
    figures/roc_curve.png            ROC + calibration curves
    figures/feature_importance.png   feature importance bar charts
    figures/ablation_results.png     ablation delta-F1 chart
    models/best_model.pkl            best model serialized with joblib

Run from repo root:
    python scripts/stage_b_train.py
"""

import argparse
import logging
import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
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
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log.warning(
        "xgboost not installed — XGB model will be skipped. "
        "Run: pip install xgboost"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_PATH = "data/stage_a_final.csv"
SEED = 42
FEATURES = ["sigma_prime", "oov_rate", "severity", "word_count", "alpha"]
TARGET = "failure_label"
GROUP_COL = "sample_id"

for _d in ("outputs", "figures", "models"):
    os.makedirs(_d, exist_ok=True)

METRIC_COLS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and preprocess
# ─────────────────────────────────────────────────────────────────────────────


def load_and_preprocess(path: str) -> pd.DataFrame:
    log.info(f"Loading {path}")
    df = pd.read_csv(path)
    log.info(f"Raw shape: {df.shape}")

    before = len(df)
    df = df[df["clean_pred"] != "unknown"].reset_index(drop=True)
    n_removed = before - len(df)
    n_groups = df[GROUP_COL].nunique()
    dist = df[TARGET].value_counts()
    log.info(
        f"Removed {n_removed} rows where clean_pred=='unknown' "
        f"(broken baselines — 38 sample_ids × 18 variants)."
    )
    log.info(
        f"Working dataset: {len(df)} rows | {n_groups} sample_ids | "
        f"ROBUST={dist.get(0, 0)} ({dist.get(0, 0)/len(df)*100:.1f}%) | "
        f"FAILURE={dist.get(1, 0)} ({dist.get(1, 0)/len(df)*100:.1f}%)"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Group-aware holdout split
# ─────────────────────────────────────────────────────────────────────────────


def group_split(df: pd.DataFrame):
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()
    groups = df[GROUP_COL].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    g_tr, g_te = groups[train_idx], groups[test_idx]

    overlap = set(np.unique(g_tr)) & set(np.unique(g_te))
    assert len(overlap) == 0, f"Group leakage — {len(overlap)} sample_ids in both splits."
    log.info(
        f"Split — train: {len(X_tr)} rows / {len(np.unique(g_tr))} groups "
        f"(failure={y_tr.mean():.3f}) | "
        f"test: {len(X_te)} rows / {len(np.unique(g_te))} groups "
        f"(failure={y_te.mean():.3f})"
    )

    # Save split membership for reproducibility
    split_rows = []
    for sid in np.unique(g_tr):
        split_rows.append({"sample_id": sid, "split": "train"})
    for sid in np.unique(g_te):
        split_rows.append({"sample_id": sid, "split": "test"})
    pd.DataFrame(split_rows).to_csv("outputs/split_info.csv", index=False)
    log.info("Saved outputs/split_info.csv")

    return X_tr, X_te, y_tr, y_te, g_tr, g_te


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model builders
# ─────────────────────────────────────────────────────────────────────────────


def build_lr():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=SEED,
                    max_iter=2000,
                ),
            ),
        ]
    )


def build_rf():
    return RandomForestClassifier(
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
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


LR_GRID = {
    "clf__C": [0.01, 0.1, 1.0, 10.0],
}

RF_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 5, 10],
}

XGB_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 6],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hyperparameter tuning — group-aware inner CV
# ─────────────────────────────────────────────────────────────────────────────


def tune(estimator, param_grid: dict, X_tr, y_tr, g_tr, name: str):
    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    log.info(
        f"Tuning {name} — {n_combos} param combos × 5 folds "
        f"(StratifiedGroupKFold, scoring=f1_macro) ..."
    )
    gs = GridSearchCV(
        estimator,
        param_grid,
        cv=StratifiedGroupKFold(n_splits=5),
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_tr, y_tr, groups=g_tr)
    log.info(f"  Best params : {gs.best_params_}")
    log.info(f"  Best CV f1  : {gs.best_score_:.4f}")
    return gs.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
# 5. Holdout evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(model, X_te, y_te, name: str) -> dict:
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    cm = confusion_matrix(y_te, y_pred)

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "precision_macro": round(
            precision_score(y_te, y_pred, average="macro", zero_division=0), 4
        ),
        "recall_macro": round(
            recall_score(y_te, y_pred, average="macro", zero_division=0), 4
        ),
        "f1_macro": round(f1_score(y_te, y_pred, average="macro", zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_te, y_prob), 4),
    }

    log.info(
        f"{name} holdout — acc:{metrics['accuracy']}  "
        f"prec:{metrics['precision_macro']}  rec:{metrics['recall_macro']}  "
        f"f1:{metrics['f1_macro']}  auc:{metrics['roc_auc']}"
    )
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    log.info(f"  Confusion matrix — TN:{tn} FP:{fp} FN:{fn} TP:{tp}  FPR:{fpr:.3f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6. 5-fold stability CV on full filtered dataset
# ─────────────────────────────────────────────────────────────────────────────


def run_cv(
    best_model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    name: str,
) -> pd.DataFrame:
    log.info(f"5-fold grouped CV (stability check) for {name} ...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, val) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        m = clone(best_model)
        m.fit(X[tr], y[tr])
        yp = m.predict(X[val])
        yprob = m.predict_proba(X[val])[:, 1]
        rows.append(
            {
                "model": name,
                "fold": fold,
                "n_train": len(tr),
                "n_val": len(val),
                "accuracy": round(accuracy_score(y[val], yp), 4),
                "precision_macro": round(
                    precision_score(y[val], yp, average="macro", zero_division=0), 4
                ),
                "recall_macro": round(
                    recall_score(y[val], yp, average="macro", zero_division=0), 4
                ),
                "f1_macro": round(
                    f1_score(y[val], yp, average="macro", zero_division=0), 4
                ),
                "roc_auc": round(roc_auc_score(y[val], yprob), 4),
            }
        )
    df_cv = pd.DataFrame(rows)
    log.info(
        f"  {name} CV f1_macro: "
        f"{df_cv['f1_macro'].mean():.4f} ± {df_cv['f1_macro'].std():.4f}"
    )
    return df_cv


# ─────────────────────────────────────────────────────────────────────────────
# 7. Feature importance
# ─────────────────────────────────────────────────────────────────────────────


def get_feature_importance(models: dict) -> pd.DataFrame:
    data: dict = {"feature": FEATURES}

    if "LR" in models:
        lr = models["LR"]
        coef = (
            lr.named_steps["clf"].coef_[0]
            if hasattr(lr, "named_steps")
            else lr.coef_[0]
        )
        data["lr_coef_abs"] = np.abs(coef).tolist()

    if "RF" in models:
        data["rf_importance"] = models["RF"].feature_importances_.tolist()

    if "XGB" in models:
        data["xgb_importance"] = models["XGB"].feature_importances_.tolist()

    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Ablation — leave-one-out on best model, holdout test set
# ─────────────────────────────────────────────────────────────────────────────


def run_ablation(best_model, X_tr, y_tr, X_te, y_te, name: str) -> pd.DataFrame:
    log.info(f"Leave-one-out feature ablation on {name} ...")
    rows = []

    # Baseline: all five features
    m0 = clone(best_model)
    m0.fit(X_tr, y_tr)
    base_f1 = f1_score(y_te, m0.predict(X_te), average="macro", zero_division=0)
    rows.append(
        {
            "excluded": "none (all features)",
            "features_used": ", ".join(FEATURES),
            "f1_macro": round(base_f1, 4),
            "delta_f1": 0.0,
        }
    )
    log.info(f"  Baseline f1_macro: {base_f1:.4f}")

    for i, feat in enumerate(FEATURES):
        mask = [j for j in range(len(FEATURES)) if j != i]
        remaining = [FEATURES[j] for j in mask]
        m_i = clone(best_model)
        m_i.fit(X_tr[:, mask], y_tr)
        f1_i = f1_score(
            y_te, m_i.predict(X_te[:, mask]), average="macro", zero_division=0
        )
        delta = round(base_f1 - f1_i, 4)
        rows.append(
            {
                "excluded": feat,
                "features_used": ", ".join(remaining),
                "f1_macro": round(f1_i, 4),
                "delta_f1": delta,
            }
        )
        log.info(f"  Without {feat:<14}  f1={f1_i:.4f}  Δf1={delta:+.4f}")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Figures
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = {"LR": "#4C72B0", "RF": "#55A868", "XGB": "#C44E52"}


def plot_roc(models: dict, X_te: np.ndarray, y_te: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: ROC curves
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

    # Right: Calibration curves
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for name, model in models.items():
        prob = model.predict_proba(X_te)[:, 1]
        prob_true, prob_pred = calibration_curve(
            y_te, prob, n_bins=6, strategy="quantile"
        )
        ax.plot(prob_pred, prob_true, "o-", lw=2, color=_COLORS[name], label=name)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle("Stage B — SST-2 Pilot", fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join("figures", "roc_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_feature_importance(fi_df: pd.DataFrame):
    importance_cols = [
        c for c in ["lr_coef_abs", "rf_importance", "xgb_importance"] if c in fi_df.columns
    ]
    titles = {
        "lr_coef_abs": "LR  |coefficient|",
        "rf_importance": "RF  Gini Importance",
        "xgb_importance": "XGB  Gain Importance",
    }

    n = len(importance_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, importance_cols):
        vals = np.array(fi_df[col].tolist())
        feats = fi_df["feature"].tolist()
        order = np.argsort(vals)
        ax.barh(
            [feats[i] for i in order],
            vals[order],
            color="#4C72B0",
            edgecolor="white",
        )
        ax.set_title(titles.get(col, col), fontsize=10)
        ax.set_xlabel("Importance", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Feature Importance — Stage B Classifiers", fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join("figures", "feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_ablation(abl_df: pd.DataFrame, best_name: str):
    sub = abl_df[abl_df["excluded"] != "none (all features)"].copy()
    sub = sub.sort_values("delta_f1")

    colors = ["#C44E52" if v > 0 else "#55A868" for v in sub["delta_f1"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(sub["excluded"], sub["delta_f1"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("ΔF1 macro  (positive = removing this feature hurts)", fontsize=9)
    ax.set_title(
        f"Leave-One-Out Feature Ablation — {best_name}  (holdout test set)",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, sub["delta_f1"]):
        x = bar.get_width()
        ax.text(
            x + 0.001 if x >= 0 else x - 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8,
        )

    plt.tight_layout()
    out = os.path.join("figures", "ablation_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Pilot summary
# ─────────────────────────────────────────────────────────────────────────────


def save_pilot_summary(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    fi_df: pd.DataFrame,
    abl_df: pd.DataFrame,
    best_name: str,
    best_params: dict,
    split_sizes: dict,
):
    sep = "=" * 72
    lines = [
        sep,
        "STAGE B PILOT SUMMARY",
        "Dataset: SST-2 (200 clean samples)  |  Pilot run",
        sep,
        "",
        "DATASET STATISTICS",
        "-" * 40,
        f"  Raw rows          : 3600  (200 samples × 18 variants)",
        f"  Removed           : 684 rows where clean_pred=='unknown'",
        f"  Reason            : 38 sample_ids with broken clean baselines",
        f"  Working dataset   : {len(df)} rows | {df[GROUP_COL].nunique()} sample_ids",
    ]
    dist = df[TARGET].value_counts()
    lines += [
        f"  Class 0 (ROBUST)  : {dist.get(0, 0)} rows ({dist.get(0, 0)/len(df)*100:.1f}%)",
        f"  Class 1 (FAILURE) : {dist.get(1, 0)} rows ({dist.get(1, 0)/len(df)*100:.1f}%)",
        "",
        "SPLIT  (GroupShuffleSplit, test_size=0.20, seed=42)",
        "-" * 40,
        f"  Train : {split_sizes['n_train_rows']} rows | {split_sizes['n_train_groups']} sample_ids",
        f"  Test  : {split_sizes['n_test_rows']} rows  | {split_sizes['n_test_groups']} sample_ids",
        f"  Group leakage : NONE (verified)",
        "",
        "FEATURES",
        "-" * 40,
        f"  {FEATURES}",
        "",
        "HOLDOUT RESULTS",
        "-" * 40,
        results_df.to_string(index=False),
        "",
        f"Best model (by f1_macro) : {best_name}",
        f"Best params              : {best_params}",
        "",
        "5-FOLD GROUPED CV STABILITY  (StratifiedGroupKFold, full 2916-row dataset)",
        "-" * 40,
    ]

    for name in ["LR", "RF", "XGB"]:
        sub = cv_df[cv_df["model"] == name]
        if sub.empty:
            continue
        lines.append(f"\n  {name}:")
        for col in METRIC_COLS:
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
        "",
        sep,
        "INTERPRETATION NOTES",
        sep,
        "  * SST-2 is the negative-contrast dataset in this study.",
        "  * Fragmentation features are expected to be weaker predictors",
        "    for sentiment classification than for reasoning tasks (GSM8K).",
        "  * F1 targets (>0.75) are stated for the full multi-dataset system;",
        "    lower performance on SST-2 alone is scientifically anticipated.",
        "  * The 38 removed sample_ids are SST-2 parse-tree fragments where",
        "    Mistral cannot produce a valid zero-shot classification even on",
        "    clean input. Their exclusion is methodologically justified.",
        sep,
    ]

    out = os.path.join("outputs", "pilot_summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Stage B — Pre-Inference Failure Classifier"
    )
    parser.add_argument(
        "--input",
        default=DATA_PATH,
        help=f"Path to stage_a_final.csv (default: {DATA_PATH})",
    )
    args = parser.parse_args()

    # 1. Load and preprocess
    df = load_and_preprocess(args.input)
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()
    groups = df[GROUP_COL].to_numpy()

    # 2. Group-aware holdout split
    X_tr, X_te, y_tr, y_te, g_tr, g_te = group_split(df)
    split_sizes = {
        "n_train_rows": len(X_tr),
        "n_train_groups": len(np.unique(g_tr)),
        "n_test_rows": len(X_te),
        "n_test_groups": len(np.unique(g_te)),
    }

    # 3. XGBoost class weight from training labels
    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = round(n_neg / n_pos, 4)
    log.info(
        f"XGBoost scale_pos_weight: {spw}  "
        f"(train neg={n_neg}, pos={n_pos})"
    )

    # 4. Tune all available models on training set with group-aware CV
    lr_best = tune(build_lr(), LR_GRID, X_tr, y_tr, g_tr, "LR")
    rf_best = tune(build_rf(), RF_GRID, X_tr, y_tr, g_tr, "RF")
    xgb_best = None
    if _XGB_AVAILABLE:
        xgb_best = tune(build_xgb(spw), XGB_GRID, X_tr, y_tr, g_tr, "XGB")
    else:
        log.warning("Skipping XGB — xgboost not installed.")

    # 5. Holdout evaluation
    eval_pairs = [("LR", lr_best), ("RF", rf_best)]
    if xgb_best is not None:
        eval_pairs.append(("XGB", xgb_best))

    results_rows = [evaluate(m, X_te, y_te, n) for n, m in eval_pairs]
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv("outputs/results.csv", index=False)
    log.info("Saved outputs/results.csv")

    # 6. 5-fold stability CV on full filtered dataset
    cv_parts = []
    for name, model in eval_pairs:
        cv_parts.append(run_cv(model, X, y, groups, name))
    cv_df = pd.concat(cv_parts, ignore_index=True)
    cv_df.to_csv("outputs/cv_results.csv", index=False)
    log.info("Saved outputs/cv_results.csv")

    # 7. Feature importance
    fi_models = {"LR": lr_best, "RF": rf_best}
    if xgb_best is not None:
        fi_models["XGB"] = xgb_best
    fi_df = get_feature_importance(fi_models)
    fi_df.to_csv("outputs/feature_importance.csv", index=False)
    log.info("Saved outputs/feature_importance.csv")

    # 8. Identify best model by f1_macro on holdout
    best_idx = results_df["f1_macro"].idxmax()
    best_name = results_df.loc[best_idx, "model"]
    _model_registry = {"LR": lr_best, "RF": rf_best}
    if xgb_best is not None:
        _model_registry["XGB"] = xgb_best
    best_model = _model_registry[best_name]
    log.info(
        f"Best model: {best_name}  "
        f"(f1_macro={results_df.loc[best_idx, 'f1_macro']:.4f}  "
        f"roc_auc={results_df.loc[best_idx, 'roc_auc']:.4f})"
    )

    # Retrieve best params for summary
    _lr_keys  = ("C", "penalty", "solver")
    _rf_keys  = ("n_estimators", "max_depth", "min_samples_leaf")
    _xgb_keys = ("n_estimators", "max_depth", "learning_rate", "subsample")
    if hasattr(best_model, "named_steps"):  # LR Pipeline
        best_params = {
            k: v for k, v in best_model.named_steps["clf"].get_params().items()
            if k in _lr_keys
        }
    elif best_name == "RF":
        best_params = {
            k: v for k, v in best_model.get_params().items()
            if k in _rf_keys
        }
    else:
        best_params = {
            k: v for k, v in best_model.get_params().items()
            if k in _xgb_keys
        }

    # 9. Ablation on best model
    abl_df = run_ablation(best_model, X_tr, y_tr, X_te, y_te, best_name)
    abl_df.to_csv("outputs/ablation_results.csv", index=False)
    log.info("Saved outputs/ablation_results.csv")

    # 10. Figures
    plot_models = {"LR": lr_best, "RF": rf_best}
    if xgb_best is not None:
        plot_models["XGB"] = xgb_best
    plot_roc(plot_models, X_te, y_te)
    plot_feature_importance(fi_df)
    plot_ablation(abl_df, best_name)

    # 11. Save best model
    model_path = "models/best_model.pkl"
    joblib.dump(best_model, model_path)
    log.info(f"Saved best model ({best_name}) → {model_path}")

    # 12. Pilot summary
    save_pilot_summary(
        df, results_df, cv_df, fi_df, abl_df,
        best_name, best_params, split_sizes,
    )

    # Final console summary
    log.info("\n" + "=" * 50)
    log.info("Stage B complete. Results:")
    log.info("\n" + results_df.to_string(index=False))
    log.info("=" * 50)


if __name__ == "__main__":
    main()
