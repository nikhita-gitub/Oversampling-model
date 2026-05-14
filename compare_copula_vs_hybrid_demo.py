from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gaussian_copula_oversampling import (
    compare_normality_metrics,
    oversample_class_with_copula,
    oversample_class_with_hybrid_normal_deficits,
)


DEFAULT_CSV_PATH = Path("loan_approved_csv.csv")
DEFAULT_TARGET_COLUMN = "Loan_Status (Approved)"
DEFAULT_MINORITY_LABEL = "N"
DEFAULT_FEATURE_COLUMNS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]


def plot_distributions(original_df, copula_df, hybrid_df, columns, output_dir):
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(len(columns), 3, figsize=(15, 4 * len(columns)))

    if len(columns) == 1:
        axes = np.array([axes])

    for row_index, column in enumerate(columns):
        datasets = [
            ("Original minority", original_df[column], axes[row_index, 0]),
            ("Copula synthetic", copula_df[column], axes[row_index, 1]),
            ("Hybrid synthetic", hybrid_df[column], axes[row_index, 2]),
        ]

        for title, values, axis in datasets:
            sns.histplot(values, kde=True, stat="density", bins=20, ax=axis, color="#2b6cb0")
            axis.set_title(f"{title}: {column}")
            axis.set_xlabel(column)

    figure.tight_layout()
    figure.savefig(output_dir / "normality_comparison.png", dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--minority-label", default=DEFAULT_MINORITY_LABEL)
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=DEFAULT_FEATURE_COLUMNS,
    )
    parser.add_argument("--output-dir", default="comparison_outputs_real")
    return parser.parse_args()


def load_dataset(csv_path, target_column, feature_columns):
    df = pd.read_csv(csv_path)
    required_columns = feature_columns + [target_column]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    working_df = df[required_columns].copy()
    working_df[feature_columns] = working_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    working_df = working_df.dropna(subset=required_columns).reset_index(drop=True)
    return working_df


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    df = load_dataset(
        csv_path=args.csv_path,
        target_column=args.target_column,
        feature_columns=args.feature_columns,
    )
    target_column = args.target_column
    target_value = args.minority_label
    feature_columns = args.feature_columns

    original_minority = df[df[target_column] == target_value].copy()
    majority_count = df[df[target_column] != target_value].shape[0]
    minority_count = len(original_minority)
    n_new = majority_count - minority_count

    if n_new <= 0:
        raise ValueError("Minority class is not smaller than the rest of the dataset.")

    copula_synthetic, copula_augmented = oversample_class_with_copula(
        df=df,
        target_column=target_column,
        target_value=target_value,
        n_new=n_new,
        feature_columns=feature_columns,
        random_state=42,
    )

    hybrid_synthetic, hybrid_augmented = oversample_class_with_hybrid_normal_deficits(
        df=df,
        target_column=target_column,
        target_value=target_value,
        n_new=n_new,
        feature_columns=feature_columns,
        random_state=42,
    )

    copula_metrics = compare_normality_metrics(
        original_df=original_minority[feature_columns],
        synthetic_df=copula_synthetic[feature_columns],
        augmented_df=copula_augmented[copula_augmented[target_column] == target_value][feature_columns],
        columns=feature_columns,
    )
    copula_metrics.insert(0, "method", "copula")

    hybrid_metrics = compare_normality_metrics(
        original_df=original_minority[feature_columns],
        synthetic_df=hybrid_synthetic[feature_columns],
        augmented_df=hybrid_augmented[hybrid_augmented[target_column] == target_value][feature_columns],
        columns=feature_columns,
    )
    hybrid_metrics.insert(0, "method", "hybrid")

    metrics = pd.concat([copula_metrics, hybrid_metrics], ignore_index=True)
    metrics.to_csv(output_dir / "normality_metrics.csv", index=False)

    summary = (
        metrics.groupby("method", as_index=False)["augmented_normality_score"]
        .mean()
        .sort_values("augmented_normality_score")
    )
    summary["winner"] = summary["augmented_normality_score"] == summary["augmented_normality_score"].min()
    summary.to_csv(output_dir / "normality_summary.csv", index=False)

    plot_distributions(
        original_df=original_minority,
        copula_df=copula_synthetic,
        hybrid_df=hybrid_synthetic,
        columns=feature_columns,
        output_dir=output_dir,
    )

    print(f"Dataset rows after dropping missing values in selected columns: {len(df)}")
    print(f"Minority label: {target_value}")
    print(f"Minority rows used for fitting: {minority_count}")
    print(f"Synthetic rows generated per method: {n_new}")
    print()
    print("Average augmented normality score by method:")
    print(summary.to_string(index=False))
    print()
    print(f"Saved metrics to: {output_dir / 'normality_metrics.csv'}")
    print(f"Saved summary to: {output_dir / 'normality_summary.csv'}")
    print(f"Saved plot to: {output_dir / 'normality_comparison.png'}")


if __name__ == "__main__":
    main()
