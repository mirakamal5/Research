"""
Correlation Analysis for Stage B
================================

This script computes Pearson and Spearman correlations between the
tokenizer-only Stage B features and the observed failure/degradation targets.

Purpose:
    This supports the final paper by showing whether the engineered
    tokenization-fragmentation features have a direct statistical relationship
    with LLM failure before training classifiers.

Inputs:
    data/processed/stage_a_sst2_final.csv
    data/processed/stage_a_svamp_final.csv
    data/processed/stage_a_squad_final.csv

Outputs:
    outputs/stage_b/correlation/feature_target_correlations.csv
    outputs/stage_b/correlation/correlation_summary.txt

Run from repo root:
    python project/scripts/correlation_analysis.py

Optional custom paths:
    python project/scripts/correlation_analysis.py --data-dir data/processed
    python project/scripts/correlation_analysis.py --out-dir outputs/stage_b/correlation
"""

import argparse
from pathlib import Path

import pandas as pd


FEATURES = ["sigma_prime", "oov_rate", "severity", "word_count", "alpha"]
TARGETS = ["failure_label", "delta_p", "noisy_score"]

DEFAULT_FILES = {
    "sst2": "stage_a_sst2_final.csv",
    "svamp": "stage_a_svamp_final.csv",
    "squad": "stage_a_squad_final.csv",
}


def validate_dataset(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Basic validation before correlation analysis.
    Raises an error if critical columns are missing or invalid.
    """

    required = ["dataset", "sample_id"] + FEATURES + TARGETS
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"[{dataset_name}] Missing required columns: {missing}"
        )

    critical = FEATURES + TARGETS + ["sample_id"]

    for col in critical:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            raise ValueError(
                f"[{dataset_name}] Critical column '{col}' has {n_missing} missing values."
            )

    invalid_labels = sorted(set(df["failure_label"].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(
            f"[{dataset_name}] Invalid failure_label values: {invalid_labels}"
        )


def compute_correlations(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations between each feature and target.
    """

    rows = []

    for feature in FEATURES:
        for target in TARGETS:
            pearson = df[feature].corr(df[target], method="pearson")
            spearman = df[feature].corr(df[target], method="spearman")

            rows.append(
                {
                    "dataset": dataset_name,
                    "feature": feature,
                    "target": target,
                    "pearson": round(float(pearson), 4),
                    "spearman": round(float(spearman), 4),
                    "abs_spearman": round(abs(float(spearman)), 4),
                }
            )

    return pd.DataFrame(rows)


def summarize_dataset(df: pd.DataFrame, dataset_name: str) -> dict:
    """
    Return simple dataset statistics for the summary file.
    """

    return {
        "dataset": dataset_name,
        "rows": len(df),
        "unique_samples": df["sample_id"].nunique(),
        "failure_rate": round(float(df["failure_label"].mean()), 4),
        "mean_delta_p": round(float(df["delta_p"].mean()), 4),
        "mean_noisy_score": round(float(df["noisy_score"].mean()), 4),
    }


def write_summary(
    stats_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Write a human-readable text summary.
    """

    lines = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("STAGE B CORRELATION ANALYSIS")
    lines.append(sep)
    lines.append("")
    lines.append("Purpose:")
    lines.append(
        "This analysis checks whether tokenizer-only fragmentation features "
        "are statistically associated with LLM degradation and failure labels."
    )
    lines.append("")
    lines.append("Features:")
    lines.append(", ".join(FEATURES))
    lines.append("")
    lines.append("Targets:")
    lines.append(", ".join(TARGETS))
    lines.append("")

    lines.append(sep)
    lines.append("DATASET STATISTICS")
    lines.append(sep)
    lines.append(stats_df.to_string(index=False))
    lines.append("")

    for target in TARGETS:
        lines.append(sep)
        lines.append(f"TOP CORRELATIONS WITH {target}")
        lines.append(sep)

        sub = corr_df[corr_df["target"] == target].copy()
        sub = sub.sort_values(
            ["dataset", "abs_spearman"],
            ascending=[True, False],
        )

        for dataset in sorted(sub["dataset"].unique()):
            lines.append("")
            lines.append(f"[{dataset.upper()}]")
            ds_sub = sub[sub["dataset"] == dataset].copy()
            ds_sub = ds_sub[
                ["feature", "pearson", "spearman", "abs_spearman"]
            ]
            lines.append(ds_sub.to_string(index=False))

        lines.append("")

    lines.append(sep)
    lines.append("INTERPRETATION GUIDE")
    lines.append(sep)
    lines.append(
        "Positive correlation with failure_label or delta_p means that the "
        "feature increases when failure/degradation increases."
    )
    lines.append(
        "Negative correlation with noisy_score means that the feature increases "
        "when model performance decreases."
    )
    lines.append(
        "Spearman correlation is especially useful here because relationships "
        "may be monotonic but not perfectly linear."
    )
    lines.append("")

    lines.append("Suggested paper wording:")
    lines.append(
        "Correlation analysis showed that tokenizer-derived fragmentation "
        "features are associated with degradation and failure labels. In "
        "particular, severity and fragmentation-related variables such as "
        "alpha, sigma_prime, and oov_rate generally show positive association "
        "with failure_label and delta_p, and negative association with "
        "noisy_score. This supports the use of tokenizer-only features as "
        "pre-inference predictors rather than arbitrary engineered variables."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute feature-target correlations for Stage B."
    )

    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing final Stage A processed CSV files.",
    )

    parser.add_argument(
        "--out-dir",
        default="outputs/stage_b/correlation",
        help="Directory where correlation outputs will be saved.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_corr = []
    stats = []

    print("=== Stage B Correlation Analysis ===")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {out_dir}")

    for dataset_name, filename in DEFAULT_FILES.items():
        path = data_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file for {dataset_name}: {path}"
            )

        print(f"\nLoading {dataset_name}: {path}")
        df = pd.read_csv(path)

        validate_dataset(df, dataset_name)

        print(f"Rows: {len(df)}")
        print(f"Unique sample_id: {df['sample_id'].nunique()}")
        print(f"Failure rate: {df['failure_label'].mean():.4f}")

        corr = compute_correlations(df, dataset_name)
        all_corr.append(corr)

        stats.append(summarize_dataset(df, dataset_name))

    corr_df = pd.concat(all_corr, ignore_index=True)
    stats_df = pd.DataFrame(stats)

    corr_path = out_dir / "feature_target_correlations.csv"
    stats_path = out_dir / "dataset_stats.csv"
    summary_path = out_dir / "correlation_summary.txt"

    corr_df.to_csv(corr_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    write_summary(stats_df, corr_df, summary_path)

    print("\n=== Saved Outputs ===")
    print(corr_path)
    print(stats_path)
    print(summary_path)

    print("\n=== Top Spearman correlations with failure_label ===")
    top_failure = (
        corr_df[corr_df["target"] == "failure_label"]
        .sort_values(["dataset", "abs_spearman"], ascending=[True, False])
    )
    print(
        top_failure[
            ["dataset", "feature", "pearson", "spearman"]
        ].to_string(index=False)
    )

    print("\n=== Top Spearman correlations with delta_p ===")
    top_delta = (
        corr_df[corr_df["target"] == "delta_p"]
        .sort_values(["dataset", "abs_spearman"], ascending=[True, False])
    )
    print(
        top_delta[
            ["dataset", "feature", "pearson", "spearman"]
        ].to_string(index=False)
    )

    print("\n=== Top Spearman correlations with noisy_score ===")
    top_noisy = (
        corr_df[corr_df["target"] == "noisy_score"]
        .sort_values(["dataset", "abs_spearman"], ascending=[True, False])
    )
    print(
        top_noisy[
            ["dataset", "feature", "pearson", "spearman"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()