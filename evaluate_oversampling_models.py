from pathlib import Path
import argparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from clustered_hybrid_oversampling import clustered_hybrid_oversample
from gaussian_copula_oversampling import generate_mixed_type_synthetic_rows


CSV_PATH = Path("loan_approved_csv.csv")
TARGET_COLUMN = "Loan_Status (Approved)"
ID_COLUMN = "Loan_ID"
MINORITY_LABEL = "N"
CONTINUOUS_COLUMNS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
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
    df = pd.read_csv(csv_path)
    return df


def infer_numeric_feature_columns(df, target_column, id_column, requested_feature_columns):
    if requested_feature_columns:
        return requested_feature_columns

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    return [
        column for column in numeric_columns
        if column not in [target_column, id_column]
    ]


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
    feature_columns = [column for column in train_df.columns if column not in [target_column, id_column]]
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
                "precision_minority": precision_score(y_test, predictions, pos_label=minority_label),
                "recall_minority": recall_score(y_test, predictions, pos_label=minority_label),
                "f1_minority": f1_score(y_test, predictions, pos_label=minority_label),
            }
        )

    return pd.DataFrame(results)


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

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[target_column],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    majority_count = (train_df[target_column] != minority_label).sum()
    minority_count = (train_df[target_column] == minority_label).sum()
    n_new = int(majority_count - minority_count)

    synthetic_copula = generate_mixed_type_synthetic_rows(
        df=train_df,
        target_column=target_column,
        target_value=minority_label,
        n_new=n_new,
        numeric_columns=numeric_feature_columns,
        passthrough_columns=[
            column for column in train_df.columns
            if column not in numeric_feature_columns + [target_column]
        ],
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
        passthrough_columns=[
            column for column in train_df.columns
            if column not in numeric_feature_columns + [target_column]
        ],
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
        passthrough_columns=[
            column for column in train_df.columns
            if column not in numeric_feature_columns + [target_column]
        ],
        id_column=id_column,
        random_state=RANDOM_STATE,
    )

    train_copula = pd.concat([train_df, synthetic_copula], ignore_index=True)
    train_hybrid = pd.concat([train_df, synthetic_hybrid], ignore_index=True)

    train_df.to_csv(output_dir / "train_original.csv", index=False)
    test_df.to_csv(output_dir / "test_holdout.csv", index=False)
    synthetic_copula.to_csv(output_dir / "synthetic_copula.csv", index=False)
    synthetic_hybrid.to_csv(output_dir / "synthetic_hybrid.csv", index=False)
    synthetic_clustered_hybrid.to_csv(output_dir / "synthetic_clustered_hybrid.csv", index=False)
    train_copula.to_csv(output_dir / "train_copula_augmented.csv", index=False)
    train_hybrid.to_csv(output_dir / "train_hybrid_augmented.csv", index=False)
    train_clustered_hybrid.to_csv(output_dir / "train_clustered_hybrid_augmented.csv", index=False)
    clustered_summary.to_csv(output_dir / "clustered_hybrid_summary.csv", index=False)

    results = pd.concat(
        [
            evaluate_dataset(train_df, test_df, "original", target_column, id_column, minority_label),
            evaluate_dataset(train_copula, test_df, "copula", target_column, id_column, minority_label),
            evaluate_dataset(train_hybrid, test_df, "hybrid", target_column, id_column, minority_label),
            evaluate_dataset(train_clustered_hybrid, test_df, "clustered_hybrid", target_column, id_column, minority_label),
        ],
        ignore_index=True,
    )
    results = results.sort_values(["model", "accuracy"], ascending=[True, False]).reset_index(drop=True)
    results.to_csv(output_dir / "model_metrics.csv", index=False)

    best_by_model = results.loc[results.groupby("model")["accuracy"].idxmax()].reset_index(drop=True)
    best_by_model.to_csv(output_dir / "best_by_model.csv", index=False)

    print(f"Dataset: {csv_path.name}")
    print(f"Target column: {target_column}")
    print(f"Minority label: {minority_label}")
    print(f"Numeric features used: {len(numeric_feature_columns)}")
    print(f"Training rows before oversampling: {len(train_df)}")
    print(f"Holdout test rows: {len(test_df)}")
    print(f"Minority rows in train: {minority_count}")
    print(f"Synthetic rows generated per method: {n_new}")
    print()
    print(results.to_string(index=False))
    print()
    print("Best dataset by model:")
    print(best_by_model.to_string(index=False))
    print()
    print(f"Saved outputs in: {output_dir}")


if __name__ == "__main__":
    main()
