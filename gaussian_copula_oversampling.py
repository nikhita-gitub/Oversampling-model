import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, normaltest


class GaussianCopulaSampler:
    """Generate synthetic rows while preserving dependence between numeric columns."""

    def __init__(self, random_state=None):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.columns_ = None
        self.sorted_values_ = {}
        self.correlation_ = None

    def fit(self, df, columns=None):
        numeric_df = df.copy()
        if columns is None:
            columns = numeric_df.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            raise ValueError("No numeric columns were provided for Gaussian copula fitting.")

        numeric_df = numeric_df[columns].apply(pd.to_numeric, errors="coerce").dropna()
        if numeric_df.empty:
            raise ValueError("No valid numeric rows remain after coercion and dropping NaNs.")

        self.columns_ = columns
        gaussian_scores = []

        for column in self.columns_:
            values = numeric_df[column].to_numpy(dtype=float)
            self.sorted_values_[column] = np.sort(values)

            ranks = pd.Series(values).rank(method="average", pct=True).to_numpy()
            ranks = np.clip(ranks, 1e-6, 1 - 1e-6)
            gaussian_scores.append(norm.ppf(ranks))

        gaussian_matrix = np.column_stack(gaussian_scores)

        if gaussian_matrix.shape[1] == 1:
            self.correlation_ = np.array([[1.0]])
        else:
            correlation = np.corrcoef(gaussian_matrix, rowvar=False)
            correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(correlation, 1.0)
            self.correlation_ = self._make_positive_semidefinite(correlation)

        return self

    def sample(self, n_samples):
        if self.columns_ is None or self.correlation_ is None:
            raise ValueError("Call fit() before sample().")

        gaussian_draws = self.sample_latent(n_samples)

        uniform_draws = norm.cdf(gaussian_draws)
        synthetic = {}

        for index, column in enumerate(self.columns_):
            sorted_values = self.sorted_values_[column]
            n_observed = len(sorted_values)
            empirical_grid = (np.arange(1, n_observed + 1) - 0.5) / n_observed

            synthetic[column] = np.interp(
                uniform_draws[:, index],
                empirical_grid,
                sorted_values,
                left=sorted_values[0],
                right=sorted_values[-1],
            )

        return pd.DataFrame(synthetic)

    def sample_latent(self, n_samples):
        return self.rng.multivariate_normal(
            mean=np.zeros(len(self.columns_)),
            cov=self.correlation_,
            size=n_samples,
        )

    @staticmethod
    def _make_positive_semidefinite(matrix, min_eigenvalue=1e-8):
        symmetric = (matrix + matrix.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        eigenvalues = np.clip(eigenvalues, min_eigenvalue, None)
        repaired = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        scale = np.sqrt(np.diag(repaired))
        repaired = repaired / np.outer(scale, scale)
        return repaired


def oversample_class_with_copula(
    df,
    target_column,
    target_value,
    n_new,
    feature_columns=None,
    random_state=None,
):
    """
    Fit the copula on one class only, sample new feature rows, then attach the class label.

    This is the standard classification setup:
    1. Filter one class.
    2. Model the joint feature distribution for that class.
    3. Generate synthetic rows for that class.
    4. Concatenate them back to the original dataset.
    """

    class_df = df.loc[df[target_column] == target_value].copy()
    if class_df.empty:
        raise ValueError(f"No rows found for {target_column} == {target_value!r}.")

    if feature_columns is None:
        feature_columns = [
            column
            for column in class_df.select_dtypes(include=[np.number]).columns
            if column != target_column
        ]

    sampler = GaussianCopulaSampler(random_state=random_state).fit(class_df, columns=feature_columns)
    synthetic_features = sampler.sample(n_new)
    synthetic_features[target_column] = target_value

    augmented_df = pd.concat([df, synthetic_features], ignore_index=True)
    return synthetic_features, augmented_df


def oversample_class_with_hybrid_normal_deficits(
    df,
    target_column,
    target_value,
    n_new,
    feature_columns=None,
    random_state=None,
):
    """
    Hybrid method:
    1. Fit a Gaussian copula on one class to preserve cross-feature dependence.
    2. For each feature, compute deficit counts against the target normal-bin percentages.
    3. Generate marginal values to fill those deficits.
    4. Assign those marginal values to rows by copula rank order.
    """

    class_df = df.loc[df[target_column] == target_value].copy()
    if class_df.empty:
        raise ValueError(f"No rows found for {target_column} == {target_value!r}.")

    if feature_columns is None:
        feature_columns = [
            column
            for column in class_df.select_dtypes(include=[np.number]).columns
            if column != target_column
        ]

    sampler = GaussianCopulaSampler(random_state=random_state).fit(class_df, columns=feature_columns)
    latent = sampler.sample_latent(n_new)
    if latent.ndim == 1:
        latent = latent.reshape(-1, 1)

    synthetic_features = {}
    for index, column in enumerate(feature_columns):
        marginal_values = _generate_feature_values_from_normal_deficits(
            series=class_df[column],
            n_new=n_new,
            random_state=None if random_state is None else random_state + index,
        )

        order = np.argsort(latent[:, index], kind="mergesort")
        ranked_values = np.sort(marginal_values)
        column_values = np.empty(n_new, dtype=float)
        column_values[order] = ranked_values
        synthetic_features[column] = column_values

    synthetic_features = pd.DataFrame(synthetic_features)
    synthetic_features[target_column] = target_value
    augmented_df = pd.concat([df, synthetic_features], ignore_index=True)
    return synthetic_features, augmented_df


def generate_mixed_type_synthetic_rows(
    df,
    target_column,
    target_value,
    n_new,
    numeric_columns,
    passthrough_columns=None,
    id_column=None,
    method="copula",
    random_state=None,
):
    """
    Generate full minority-class rows for mixed tabular data.

    Numeric columns are synthesized with the selected method.
    Remaining passthrough columns are copied from the nearest minority donor row
    in numeric feature space so categorical/discrete relationships are retained.
    """

    class_df = df.loc[df[target_column] == target_value].copy()
    if class_df.empty:
        raise ValueError(f"No rows found for {target_column} == {target_value!r}.")

    if passthrough_columns is None:
        passthrough_columns = [
            column for column in df.columns
            if column not in set(numeric_columns + [target_column])
        ]

    fit_df = class_df[numeric_columns + passthrough_columns + [target_column]].copy()
    fit_df[numeric_columns] = fit_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    fit_df = fit_df.dropna(subset=numeric_columns).reset_index(drop=True)
    if fit_df.empty:
        raise ValueError("No minority rows remain after dropping missing numeric values.")

    if method == "copula":
        synthetic_numeric, _ = oversample_class_with_copula(
            df=fit_df[numeric_columns + [target_column]],
            target_column=target_column,
            target_value=target_value,
            n_new=n_new,
            feature_columns=numeric_columns,
            random_state=random_state,
        )
    elif method == "hybrid":
        synthetic_numeric, _ = oversample_class_with_hybrid_normal_deficits(
            df=fit_df[numeric_columns + [target_column]],
            target_column=target_column,
            target_value=target_value,
            n_new=n_new,
            feature_columns=numeric_columns,
            random_state=random_state,
        )
    else:
        raise ValueError("method must be either 'copula' or 'hybrid'.")

    if not passthrough_columns and (id_column is None or id_column not in df.columns):
        synthetic_full = synthetic_numeric[numeric_columns].copy()
        synthetic_full[target_column] = target_value
        column_order = [column for column in df.columns if column in synthetic_full.columns]
        return synthetic_full[column_order]

    donor_indices = _nearest_donor_indices(
        donor_values=fit_df[numeric_columns].to_numpy(dtype=float),
        synthetic_values=synthetic_numeric[numeric_columns].to_numpy(dtype=float),
    )

    donor_rows = fit_df.iloc[donor_indices].reset_index(drop=True)
    synthetic_full = donor_rows[passthrough_columns].copy()
    for column in numeric_columns:
        synthetic_full[column] = synthetic_numeric[column].to_numpy()

    synthetic_full[target_column] = target_value

    if id_column is not None and id_column in df.columns:
        synthetic_full[id_column] = [
            f"SYN_{method.upper()}_{index + 1:05d}" for index in range(n_new)
        ]

    column_order = [column for column in df.columns if column in synthetic_full.columns]
    synthetic_full = synthetic_full[column_order]
    return synthetic_full


def compare_skewness(original_df, synthetic_df, augmented_df, columns=None):
    """
    Return skewness before and after oversampling.

    `augmented_skewness` is the skewness after combining original and synthetic data.
    """

    if columns is None:
        columns = [
            column
            for column in original_df.select_dtypes(include=[np.number]).columns
            if column in synthetic_df.columns and column in augmented_df.columns
        ]

    records = []
    for column in columns:
        original_skew = pd.to_numeric(original_df[column], errors="coerce").dropna().skew()
        synthetic_skew = pd.to_numeric(synthetic_df[column], errors="coerce").dropna().skew()
        augmented_skew = pd.to_numeric(augmented_df[column], errors="coerce").dropna().skew()

        records.append(
            {
                "column": column,
                "original_skewness": original_skew,
                "synthetic_skewness": synthetic_skew,
                "augmented_skewness": augmented_skew,
                "delta_original_to_augmented": augmented_skew - original_skew,
            }
        )

    return pd.DataFrame(records)


def compare_normality_metrics(original_df, synthetic_df, augmented_df, columns=None):
    if columns is None:
        columns = [
            column
            for column in original_df.select_dtypes(include=[np.number]).columns
            if column in synthetic_df.columns and column in augmented_df.columns
        ]

    records = []
    for column in columns:
        original = pd.to_numeric(original_df[column], errors="coerce").dropna()
        synthetic = pd.to_numeric(synthetic_df[column], errors="coerce").dropna()
        augmented = pd.to_numeric(augmented_df[column], errors="coerce").dropna()

        records.append(
            {
                "column": column,
                "original_skewness": original.skew(),
                "synthetic_skewness": synthetic.skew(),
                "augmented_skewness": augmented.skew(),
                "original_kurtosis": kurtosis(original, fisher=True, bias=False),
                "synthetic_kurtosis": kurtosis(synthetic, fisher=True, bias=False),
                "augmented_kurtosis": kurtosis(augmented, fisher=True, bias=False),
                "original_normaltest_pvalue": _safe_normaltest_pvalue(original),
                "synthetic_normaltest_pvalue": _safe_normaltest_pvalue(synthetic),
                "augmented_normaltest_pvalue": _safe_normaltest_pvalue(augmented),
            }
        )

    metrics = pd.DataFrame(records)
    metrics["augmented_normality_score"] = (
        metrics["augmented_skewness"].abs() + metrics["augmented_kurtosis"].abs()
    )
    return metrics


def joint_sample_numeric_table(df, n_new, columns=None, random_state=None):
    """
    Sample a full numeric table jointly.

    Use this when the target is continuous and should be generated jointly with features.
    Pass all numeric feature columns plus the numeric target column in `columns`.
    """

    sampler = GaussianCopulaSampler(random_state=random_state).fit(df, columns=columns)
    return sampler.sample(n_new)


def _safe_normaltest_pvalue(values):
    if len(values) < 8:
        return np.nan
    try:
        return normaltest(values).pvalue
    except Exception:
        return np.nan


def _generate_feature_values_from_normal_deficits(series, n_new, random_state=None):
    rng = np.random.default_rng(random_state)
    numeric = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        raise ValueError("Feature has no valid numeric values.")

    feature_min = numeric.min()
    feature_max = numeric.max()
    if np.isclose(feature_min, feature_max):
        return np.full(n_new, feature_min, dtype=float)

    normalized = 2 * (numeric - feature_min) / (feature_max - feature_min) - 1
    boundaries = _compute_normal_bin_boundaries(normalized)
    expected_percentages = np.array(
        [0.13, 0.49, 1.66, 4.40, 9.19, 14.98, 19.15, 19.15, 14.98, 9.19, 4.40, 1.66, 0.49, 0.13]
    ) / 100

    current_counts = np.histogram(normalized, bins=boundaries)[0]
    desired_total = len(normalized) + n_new
    desired_counts = _counts_from_percentages(expected_percentages, desired_total)
    deficits = np.maximum(desired_counts - current_counts, 0)
    synthetic_counts = _adjust_counts_to_total(deficits, expected_percentages, n_new)

    normalized_samples = []
    for index, count in enumerate(synthetic_counts):
        if count <= 0:
            continue

        start = max(boundaries[index], -1.0)
        end = min(boundaries[index + 1], 1.0)

        if np.isclose(start, end) or start > end:
            normalized_samples.append(np.full(count, normalized.mean(), dtype=float))
        else:
            normalized_samples.append(rng.uniform(start, end, size=count))

    if normalized_samples:
        normalized_values = np.concatenate(normalized_samples)
    else:
        normalized_values = np.full(n_new, normalized.mean(), dtype=float)

    if len(normalized_values) < n_new:
        padding = np.full(n_new - len(normalized_values), normalized.mean(), dtype=float)
        normalized_values = np.concatenate([normalized_values, padding])
    elif len(normalized_values) > n_new:
        normalized_values = normalized_values[:n_new]

    rng.shuffle(normalized_values)
    return (normalized_values + 1) / 2 * (feature_max - feature_min) + feature_min


def _nearest_donor_indices(donor_values, synthetic_values):
    donor_values = np.asarray(donor_values, dtype=float)
    synthetic_values = np.asarray(synthetic_values, dtype=float)
    if donor_values.ndim == 1:
        donor_values = donor_values.reshape(-1, 1)
    if synthetic_values.ndim == 1:
        synthetic_values = synthetic_values.reshape(-1, 1)

    scales = donor_values.std(axis=0, ddof=1)
    scales = np.where((scales == 0) | np.isnan(scales), 1.0, scales)

    donor_scaled = donor_values / scales
    synthetic_scaled = synthetic_values / scales

    distances = np.sum(
        (synthetic_scaled[:, None, :] - donor_scaled[None, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(distances, axis=1)


def _compute_normal_bin_boundaries(normalized_values):
    median = np.median(normalized_values)
    spread = np.std(normalized_values, ddof=1)

    if np.isclose(spread, 0):
        return np.linspace(-1, 1, 15)

    pseudo_std = (np.max(np.abs(normalized_values)) - median) / 3.5
    if pseudo_std <= 0 or np.isnan(pseudo_std):
        pseudo_std = spread

    quantiles = np.array([0.0013, 0.0062, 0.0228, 0.0668, 0.1587, 0.3085, 0.5, 0.6915, 0.8413, 0.9332, 0.9772, 0.9938, 0.9987])
    middle = norm.ppf(quantiles, loc=median, scale=pseudo_std)
    extreme_low = middle[0] - pseudo_std / 2
    extreme_high = middle[-1] + pseudo_std / 2
    boundaries = np.concatenate(([extreme_low], middle, [extreme_high]))
    boundaries = np.sort(boundaries)

    for index in range(1, len(boundaries)):
        if boundaries[index] <= boundaries[index - 1]:
            boundaries[index] = boundaries[index - 1] + 1e-6

    return boundaries


def _counts_from_percentages(percentages, total):
    raw = percentages * total
    counts = np.floor(raw).astype(int)
    remainder = int(total - counts.sum())
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remainder]] += 1
    return counts


def _adjust_counts_to_total(counts, weights, total):
    adjusted = counts.astype(int).copy()
    current_total = int(adjusted.sum())

    if current_total < total:
        order = np.argsort(weights)[::-1]
        for index in order:
            if current_total == total:
                break
            adjusted[index] += 1
            current_total += 1
            if index == order[-1] and current_total < total:
                order = np.roll(order, -1)

        while current_total < total:
            for index in order:
                if current_total == total:
                    break
                adjusted[index] += 1
                current_total += 1

    elif current_total > total:
        order = np.argsort(adjusted)[::-1]
        for index in order:
            while adjusted[index] > 0 and current_total > total:
                adjusted[index] -= 1
                current_total -= 1

    return adjusted
