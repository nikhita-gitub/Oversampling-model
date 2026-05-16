from pathlib import Path
import argparse

from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from clustered_hybrid_oversampling import clustered_hybrid_oversample
from gaussian_copula_oversampling import generate_mixed_type_synthetic_rows


CSV_PATH = Path("loan_approved_csv.csv")
TARGET_COLUMN = "Loan_Status (Approved)"
ID_COLUMN = "Loan_ID"
MINORITY_LABEL = "N"
OUTPUT_DIR = Path("model_comparison_outputs")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default=str(CSV_PATH))
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--minority-label", default=str(MINORITY_LABEL))
    parser.add_argument("--id-column", default=ID_COLUMN)
    parser.add_argument("--feature-columns", nargs="*", default=None)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def parse_label(value):
    if value in {"0", "1"}:
        return int(value)
    return value


def load_dataset(csv_path):
    return pd.read_csv(csv_path)


def infer_numeric_feature_columns(df, target_column, id_column, requested_feature_columns):
    if requested_feature_columns:
        return requested_feature_columns

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    return [column for column in numeric_columns if column not in [target_column, id_column]]


def get_model_feature_columns(df, target_column, id_column):
    return [column for column in df.columns if column not in [target_column, id_column]]


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def evaluate_dataset(train_df, test_df, dataset_name, target_column, id_column, minority_label):
    feature_columns = get_model_feature_columns(train_df, target_column, id_column)
    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    results = []
    for model_name, estimator in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
        if model_name == "xgboost":
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            pipeline.fit(X_train, y_train_encoded)
            predictions = label_encoder.inverse_transform(pipeline.predict(X_test))
        else:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

        results.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                "precision_minority": precision_score(y_test, predictions, pos_label=minority_label, zero_division=0),
                "recall_minority": recall_score(y_test, predictions, pos_label=minority_label, zero_division=0),
                "f1_minority": f1_score(y_test, predictions, pos_label=minority_label, zero_division=0),
            }
        )

    return pd.DataFrame(results)


def _prepare_resampler_features(df, feature_columns):
    numeric_features = [column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])]
    categorical_features = [column for column in feature_columns if column not in numeric_features]

    if numeric_features:
        numeric_imputer = SimpleImputer(strategy="median")
        numeric_matrix = numeric_imputer.fit_transform(df[numeric_features])
    else:
        numeric_imputer = None
        numeric_matrix = np.empty((len(df), 0))

    if categorical_features:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        categorical_filled = categorical_imputer.fit_transform(df[categorical_features])
        categorical_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        categorical_matrix = categorical_encoder.fit_transform(categorical_filled)
    else:
        categorical_imputer = None
        categorical_encoder = None
        categorical_matrix = np.empty((len(df), 0))

    matrix = np.column_stack([numeric_matrix, categorical_matrix])
    categorical_indices = list(range(len(numeric_features), len(numeric_features) + len(categorical_features)))
    return {
        "matrix": matrix,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categorical_indices": categorical_indices,
        "numeric_imputer": numeric_imputer,
        "categorical_imputer": categorical_imputer,
        "categorical_encoder": categorical_encoder,
    }


def _decode_resampled_features(resampled_matrix, feature_columns, metadata):
    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]
    numeric_width = len(numeric_features)
    categorical_width = len(categorical_features)

    restored = {}
    if numeric_features:
        numeric_values = resampled_matrix[:, :numeric_width]
        for offset, column in enumerate(numeric_features):
            restored[column] = numeric_values[:, offset]

    if categorical_features:
        categorical_values = resampled_matrix[:, numeric_width:numeric_width + categorical_width]
        clipped_columns = []
        for offset, categories in enumerate(metadata["categorical_encoder"].categories_):
            codes = np.rint(categorical_values[:, offset]).astype(int)
            codes = np.clip(codes, 0, len(categories) - 1)
            clipped_columns.append(codes)

        categorical_codes = np.column_stack(clipped_columns)
        decoded = metadata["categorical_encoder"].inverse_transform(categorical_codes)
        for offset, column in enumerate(categorical_features):
            restored[column] = decoded[:, offset]

    restored_df = pd.DataFrame(restored)
    return restored_df[feature_columns]


def generate_baseline_synthetic_rows(df, target_column, target_value, id_column=None, method="smote", random_state=None):
    feature_columns = get_model_feature_columns(df, target_column, id_column)
    class_counts = df[target_column].value_counts()
    minority_count = int(class_counts.get(target_value, 0))
    majority_count = int(len(df) - minority_count)
    n_new = majority_count - minority_count

    if n_new <= 0:
        empty = pd.DataFrame(columns=df.columns)
        return empty, df.copy()

    metadata = _prepare_resampler_features(df, feature_columns)
    X = metadata["matrix"]
    y = df[target_column].to_numpy()

    if method == "random_oversample":
        sampler = RandomOverSampler(random_state=random_state)
    elif method == "smote":
        if minority_count < 2:
            raise ValueError("SMOTE requires at least two minority rows.")

        k_neighbors = min(5, minority_count - 1)
        if metadata["categorical_features"]:
            sampler = SMOTENC(
                categorical_features=metadata["categorical_indices"],
                k_neighbors=k_neighbors,
                random_state=random_state,
            )
        else:
            sampler = SMOTE(
                k_neighbors=k_neighbors,
                random_state=random_state,
            )
    else:
        raise ValueError("method must be either 'random_oversample' or 'smote'.")

    X_resampled, y_resampled = sampler.fit_resample(X, y)
    synthetic_count = len(X_resampled) - len(X)
    synthetic_matrix = X_resampled[-synthetic_count:]
    synthetic_labels = y_resampled[-synthetic_count:]

    synthetic_df = _decode_resampled_features(synthetic_matrix, feature_columns, metadata)
    synthetic_df[target_column] = synthetic_labels

    if id_column is not None and id_column in df.columns:
        prefix = "ROS" if method == "random_oversample" else "SMOTE"
        synthetic_df[id_column] = [f"SYN_{prefix}_{index + 1:05d}" for index in range(synthetic_count)]

    synthetic_df = synthetic_df[df.columns]
    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
    return synthetic_df, augmented_df


def summarize_results(results):
    summary = (
        results.groupby("dataset", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision_minority=("precision_minority", "mean"),
            mean_recall_minority=("recall_minority", "mean"),
            mean_f1_minority=("f1_minority", "mean"),
        )
        .sort_values(["mean_f1_minority", "mean_balanced_accuracy"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary


def save_metric_plot(results, output_path, metric, title):
    pivot = results.pivot(index="model", columns="dataset", values=metric)
    dataset_scores = results.groupby("dataset")[metric].mean().sort_values(ascending=False)
    pivot = pivot.reindex(columns=dataset_scores.index)

    axis = pivot.plot(kind="bar", figsize=(12, 6))
    axis.set_title(title)
    axis.set_xlabel("Model")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    target_column = args.target_column
    minority_label = parse_label(args.minority_label)
    id_column = args.id_column if args.id_column else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    df = load_dataset(csv_path)
    numeric_feature_columns = infer_numeric_feature_columns(
        df=df,
        target_column=target_column,
        id_column=id_column,
        requested_feature_columns=args.feature_columns,
    )
    passthrough_columns = [
        column for column in df.columns if column not in numeric_feature_columns + [target_column]
    ]

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[target_column],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    majority_count = int((train_df[target_column] != minority_label).sum())
    minority_count = int((train_df[target_column] == minority_label).sum())
    n_new = int(majority_count - minority_count)

    synthetic_random_oversample, train_random_oversample = generate_baseline_synthetic_rows(
        df=train_df,
        target_column=target_column,
        target_value=minority_label,
        id_column=id_column,
        method="random_oversample",
        random_state=RANDOM_STATE,
    )
    synthetic_smote, train_smote = generate_baseline_synthetic_rows(
        df=train_df,
        target_column=target_column,
        target_value=minority_label,
        id_column=id_column,
        method="smote",
        random_state=RANDOM_STATE,
    )
    synthetic_copula = generate_mixed_type_synthetic_rows(
        df=train_df,
        target_column=target_column,
        target_value=minority_label,
        n_new=n_new,
        numeric_columns=numeric_feature_columns,
        passthrough_columns=passthrough_columns,
        id_column=id_column,
        method="copula",
        random_state=RANDOM_STATE,
    )
    synthetic_hybrid = generate_mixed_type_synthetic_rows(
        df=train_df,
        target_column=target_column,
        target_value=minority_label,
        n_new=n_new,
        numeric_columns=numeric_feature_columns,
        passthrough_columns=passthrough_columns,
        id_column=id_column,
        method="hybrid",
        random_state=RANDOM_STATE,
    )
    synthetic_clustered_hybrid, train_clustered_hybrid, clustered_summary = clustered_hybrid_oversample(
        df=train_df,
        target_column=target_column,
        target_value=minority_label,
        n_new=n_new,
        numeric_columns=numeric_feature_columns,
        passthrough_columns=passthrough_columns,
        id_column=id_column,
        random_state=RANDOM_STATE,
    )

    train_copula = pd.concat([train_df, synthetic_copula], ignore_index=True)
    train_hybrid = pd.concat([train_df, synthetic_hybrid], ignore_index=True)

    train_df.to_csv(output_dir / "train_original.csv", index=False)
    test_df.to_csv(output_dir / "test_holdout.csv", index=False)
    synthetic_random_oversample.to_csv(output_dir / "synthetic_random_oversample.csv", index=False)
    synthetic_smote.to_csv(output_dir / "synthetic_smote.csv", index=False)
    synthetic_copula.to_csv(output_dir / "synthetic_copula.csv", index=False)
    synthetic_hybrid.to_csv(output_dir / "synthetic_hybrid.csv", index=False)
    synthetic_clustered_hybrid.to_csv(output_dir / "synthetic_clustered_hybrid.csv", index=False)
    train_random_oversample.to_csv(output_dir / "train_random_oversample_augmented.csv", index=False)
    train_smote.to_csv(output_dir / "train_smote_augmented.csv", index=False)
    train_copula.to_csv(output_dir / "train_copula_augmented.csv", index=False)
    train_hybrid.to_csv(output_dir / "train_hybrid_augmented.csv", index=False)
    train_clustered_hybrid.to_csv(output_dir / "train_clustered_hybrid_augmented.csv", index=False)
    clustered_summary.to_csv(output_dir / "clustered_hybrid_summary.csv", index=False)

    results = pd.concat(
        [
            evaluate_dataset(train_df, test_df, "original", target_column, id_column, minority_label),
            evaluate_dataset(train_random_oversample, test_df, "random_oversample", target_column, id_column, minority_label),
            evaluate_dataset(train_smote, test_df, "smote", target_column, id_column, minority_label),
            evaluate_dataset(train_copula, test_df, "copula", target_column, id_column, minority_label),
            evaluate_dataset(train_hybrid, test_df, "hybrid", target_column, id_column, minority_label),
            evaluate_dataset(train_clustered_hybrid, test_df, "clustered_hybrid", target_column, id_column, minority_label),
        ],
        ignore_index=True,
    )
    results = results.sort_values(["model", "f1_minority", "balanced_accuracy"], ascending=[True, False, False]).reset_index(drop=True)
    results.to_csv(output_dir / "model_metrics.csv", index=False)

    dataset_summary = summarize_results(results)
    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)

    best_by_model_accuracy = results.loc[results.groupby("model")["accuracy"].idxmax()].reset_index(drop=True)
    best_by_model_accuracy.to_csv(output_dir / "best_by_model.csv", index=False)
    best_by_model_accuracy.to_csv(output_dir / "best_by_model_accuracy.csv", index=False)

    best_by_model_f1 = results.loc[results.groupby("model")["f1_minority"].idxmax()].reset_index(drop=True)
    best_by_model_f1.to_csv(output_dir / "best_by_model_f1.csv", index=False)

    save_metric_plot(
        results=results,
        output_path=output_dir / "minority_f1_comparison.png",
        metric="f1_minority",
        title=f"Minority F1 Comparison ({csv_path.name})",
    )
    save_metric_plot(
        results=results,
        output_path=output_dir / "balanced_accuracy_comparison.png",
        metric="balanced_accuracy",
        title=f"Balanced Accuracy Comparison ({csv_path.name})",
    )

    print(f"Dataset: {csv_path.name}")
    print(f"Target column: {target_column}")
    print(f"Minority label: {minority_label}")
    print(f"Numeric features used for generative methods: {len(numeric_feature_columns)}")
    print(f"Training rows before oversampling: {len(train_df)}")
    print(f"Holdout test rows: {len(test_df)}")
    print(f"Minority rows in train: {minority_count}")
    print(f"Synthetic rows generated per method: {n_new}")
    print()
    print(results.to_string(index=False))
    print()
    print("Mean metric summary by dataset:")
    print(dataset_summary.to_string(index=False))
    print()
    print("Best dataset by model (minority F1):")
    print(best_by_model_f1.to_string(index=False))
    print()
    print(f"Saved outputs in: {output_dir}")


if __name__ == "__main__":
    main()
