import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from gaussian_copula_oversampling import generate_mixed_type_synthetic_rows


def clustered_hybrid_oversample(
    df,
    target_column,
    target_value,
    n_new,
    numeric_columns=None,
    passthrough_columns=None,
    id_column=None,
    n_clusters=None,
    min_cluster_size=25,
    max_clusters=8,
    random_state=None,
):
    """
    Cluster the minority class first, then run hybrid oversampling inside each cluster.

    This is useful when the minority class is multi-modal and one global distribution
    is too crude. Each cluster gets its own local hybrid generator, which makes the
    synthetic samples more boundary-aware and locally consistent.
    """

    minority_df = df.loc[df[target_column] == target_value].copy()
    if minority_df.empty:
        raise ValueError(f"No rows found for {target_column} == {target_value!r}.")

    if numeric_columns is None:
        numeric_columns = [
            column
            for column in minority_df.select_dtypes(include=[np.number]).columns
            if column != target_column
        ]

    if not numeric_columns:
        raise ValueError("No numeric columns available for clustered hybrid oversampling.")

    if passthrough_columns is None:
        passthrough_columns = [
            column for column in df.columns
            if column not in set(numeric_columns + [target_column])
        ]

    working_df = minority_df[numeric_columns + passthrough_columns + [target_column]].copy()
    working_df[numeric_columns] = working_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    working_df = working_df.dropna(subset=numeric_columns).reset_index(drop=True)
    if working_df.empty:
        raise ValueError("No minority rows remain after dropping missing numeric values.")

    resolved_clusters = _resolve_cluster_count(
        minority_count=len(working_df),
        requested_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        max_clusters=max_clusters,
    )

    if resolved_clusters == 1:
        synthetic_df = generate_mixed_type_synthetic_rows(
            df=working_df,
            target_column=target_column,
            target_value=target_value,
            n_new=n_new,
            numeric_columns=numeric_columns,
            passthrough_columns=passthrough_columns,
            id_column=id_column,
            method="hybrid",
            random_state=random_state,
        )
        augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
        summary_df = pd.DataFrame(
            [
                {
                    "cluster": 0,
                    "real_rows": len(working_df),
                    "synthetic_rows": n_new,
                }
            ]
        )
        return synthetic_df, augmented_df, summary_df

    cluster_labels = _fit_minority_clusters(
        df=working_df,
        numeric_columns=numeric_columns,
        n_clusters=resolved_clusters,
        random_state=random_state,
    )
    working_df["_cluster"] = cluster_labels

    cluster_sizes = working_df["_cluster"].value_counts().sort_index()
    synthetic_counts = _allocate_synthetic_counts(cluster_sizes, n_new)

    synthetic_parts = []
    summary_rows = []

    for cluster_id, synth_count in synthetic_counts.items():
        cluster_df = working_df.loc[working_df["_cluster"] == cluster_id].drop(columns="_cluster").reset_index(drop=True)
        real_count = len(cluster_df)

        if synth_count > 0:
            cluster_synthetic = generate_mixed_type_synthetic_rows(
                df=cluster_df,
                target_column=target_column,
                target_value=target_value,
                n_new=int(synth_count),
                numeric_columns=numeric_columns,
                passthrough_columns=passthrough_columns,
                id_column=id_column,
                method="hybrid",
                random_state=None if random_state is None else random_state + int(cluster_id),
            )
            cluster_synthetic["_source_cluster"] = cluster_id
            synthetic_parts.append(cluster_synthetic)

        summary_rows.append(
            {
                "cluster": int(cluster_id),
                "real_rows": int(real_count),
                "synthetic_rows": int(synth_count),
            }
        )

    if synthetic_parts:
        synthetic_df = pd.concat(synthetic_parts, ignore_index=True)
    else:
        synthetic_df = pd.DataFrame(columns=df.columns.tolist() + ["_source_cluster"])

    augmented_df = pd.concat([df, synthetic_df.drop(columns="_source_cluster", errors="ignore")], ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    return synthetic_df, augmented_df, summary_df


def _resolve_cluster_count(minority_count, requested_clusters, min_cluster_size, max_clusters):
    if minority_count <= min_cluster_size:
        return 1

    max_by_size = max(1, minority_count // max(1, min_cluster_size))
    allowed_max = min(max_clusters, max_by_size, minority_count)

    if requested_clusters is None:
        return max(1, allowed_max)

    requested_clusters = int(requested_clusters)
    return max(1, min(requested_clusters, allowed_max))


def _fit_minority_clusters(df, numeric_columns, n_clusters, random_state):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_columns])

    model = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_state,
    )
    return model.fit_predict(scaled)


def _allocate_synthetic_counts(cluster_sizes, total_new):
    proportions = cluster_sizes / cluster_sizes.sum()
    raw_counts = proportions * total_new
    counts = np.floor(raw_counts).astype(int)

    remainder = int(total_new - counts.sum())
    if remainder > 0:
        order = np.argsort((raw_counts - counts).to_numpy())[::-1]
        index_values = cluster_sizes.index.to_numpy()
        for position in order[:remainder]:
            counts.loc[index_values[position]] += 1

    return counts.sort_index()
