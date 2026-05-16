import unittest
import warnings

import numpy as np
import pandas as pd

from clustered_hybrid_oversampling import clustered_hybrid_oversample
from evaluate_oversampling_models import generate_baseline_synthetic_rows
from gaussian_copula_oversampling import (
    GaussianCopulaSampler,
    generate_mixed_type_synthetic_rows,
    oversample_class_with_copula,
    oversample_class_with_hybrid_normal_deficits,
)


class GaussianCopulaSamplerTests(unittest.TestCase):
    def setUp(self):
        self.numeric_df = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "feature_b": [10.0, 11.0, 13.0, 15.0, 18.0, 20.0],
                "target": [0, 0, 0, 1, 1, 1],
            }
        )

    def test_fit_and_sample_preserve_shape_and_bounds(self):
        sampler = GaussianCopulaSampler(random_state=7).fit(self.numeric_df, columns=["feature_a", "feature_b"])
        synthetic = sampler.sample(8)

        self.assertEqual(list(synthetic.columns), ["feature_a", "feature_b"])
        self.assertEqual(synthetic.shape, (8, 2))
        self.assertFalse(synthetic.isna().any().any())

        for column in synthetic.columns:
            self.assertGreaterEqual(synthetic[column].min(), self.numeric_df[column].min())
            self.assertLessEqual(synthetic[column].max(), self.numeric_df[column].max())

    def test_oversample_class_with_copula_adds_requested_rows(self):
        synthetic, augmented = oversample_class_with_copula(
            df=self.numeric_df,
            target_column="target",
            target_value=1,
            n_new=4,
            feature_columns=["feature_a", "feature_b"],
            random_state=11,
        )

        self.assertEqual(len(synthetic), 4)
        self.assertEqual(len(augmented), len(self.numeric_df) + 4)
        self.assertTrue((synthetic["target"] == 1).all())

    def test_hybrid_handles_constant_feature(self):
        df = pd.DataFrame(
            {
                "feature_a": [5.0, 5.0, 5.0, 5.0, 8.0, 9.0, 10.0, 11.0],
                "feature_b": [1.0, 1.2, 1.4, 1.6, 3.0, 3.2, 3.4, 3.6],
                "target": [1, 1, 1, 1, 0, 0, 0, 0],
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            synthetic, _ = oversample_class_with_hybrid_normal_deficits(
                df=df,
                target_column="target",
                target_value=1,
                n_new=4,
                feature_columns=["feature_a", "feature_b"],
                random_state=5,
            )

        self.assertTrue(np.allclose(synthetic["feature_a"].to_numpy(), 5.0))

    def test_fit_raises_when_no_numeric_columns_are_available(self):
        df = pd.DataFrame({"category": ["a", "b", "c"]})
        with self.assertRaises(ValueError):
            GaussianCopulaSampler().fit(df)


class MixedTypeGenerationTests(unittest.TestCase):
    def setUp(self):
        self.mixed_df = pd.DataFrame(
            {
                "loan_id": ["A1", "A2", "A3", "A4", "A5", "A6"],
                "region": ["north", "north", "south", "south", "east", "east"],
                "segment": ["retail", "retail", "retail", "sme", "sme", "sme"],
                "income": [30.0, 35.0, 40.0, 60.0, 65.0, 70.0],
                "balance": [3.0, 4.0, 5.0, 8.0, 9.0, 10.0],
                "approved": ["N", "N", "N", "Y", "Y", "Y"],
            }
        )

    def test_generate_mixed_type_rows_preserves_schema_and_assigns_ids(self):
        synthetic = generate_mixed_type_synthetic_rows(
            df=self.mixed_df,
            target_column="approved",
            target_value="N",
            n_new=3,
            numeric_columns=["income", "balance"],
            passthrough_columns=["loan_id", "region", "segment"],
            id_column="loan_id",
            method="copula",
            random_state=17,
        )

        self.assertEqual(list(synthetic.columns), list(self.mixed_df.columns))
        self.assertEqual(len(synthetic), 3)
        self.assertTrue((synthetic["approved"] == "N").all())
        self.assertTrue(synthetic["loan_id"].str.startswith("SYN_COPULA_").all())
        self.assertTrue(set(synthetic["region"]).issubset(set(self.mixed_df["region"])))
        self.assertTrue(set(synthetic["segment"]).issubset(set(self.mixed_df["segment"])))

    def test_clustered_hybrid_returns_expected_counts(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            synthetic, augmented, summary = clustered_hybrid_oversample(
                df=self.mixed_df,
                target_column="approved",
                target_value="N",
                n_new=4,
                numeric_columns=["income", "balance"],
                passthrough_columns=["loan_id", "region", "segment"],
                id_column="loan_id",
                n_clusters=2,
                min_cluster_size=1,
                random_state=23,
            )

        self.assertEqual(len(synthetic), 4)
        self.assertEqual(len(augmented), len(self.mixed_df) + 4)
        self.assertEqual(int(summary["synthetic_rows"].sum()), 4)


class BaselineResamplerTests(unittest.TestCase):
    def setUp(self):
        self.loan_like_df = pd.DataFrame(
            {
                "Loan_ID": [f"L{i:03d}" for i in range(10)],
                "Gender": ["M", "F", "M", "F", "M", "M", "F", "M", "F", "M"],
                "Education": ["Grad", "Grad", "Grad", "Not Grad", "Grad", "Grad", "Not Grad", "Grad", "Not Grad", "Grad"],
                "ApplicantIncome": [50, 55, 53, 48, 52, 75, 78, 82, 88, 91],
                "LoanAmount": [100, 105, 103, 99, 110, 130, 132, 135, 140, 145],
                "Loan_Status (Approved)": ["N", "N", "N", "N", "Y", "Y", "Y", "Y", "Y", "Y"],
            }
        )

    def test_random_oversample_generates_balanced_rows_with_ids(self):
        synthetic, augmented = generate_baseline_synthetic_rows(
            df=self.loan_like_df,
            target_column="Loan_Status (Approved)",
            target_value="N",
            id_column="Loan_ID",
            method="random_oversample",
            random_state=19,
        )

        self.assertEqual(len(synthetic), 2)
        self.assertEqual(len(augmented), len(self.loan_like_df) + 2)
        self.assertTrue(synthetic["Loan_ID"].str.startswith("SYN_ROS_").all())
        self.assertTrue((synthetic["Loan_Status (Approved)"] == "N").all())

    def test_smote_generates_balanced_rows_for_mixed_type_data(self):
        synthetic, augmented = generate_baseline_synthetic_rows(
            df=self.loan_like_df,
            target_column="Loan_Status (Approved)",
            target_value="N",
            id_column="Loan_ID",
            method="smote",
            random_state=29,
        )

        self.assertEqual(len(synthetic), 2)
        self.assertEqual(len(augmented), len(self.loan_like_df) + 2)
        self.assertTrue(synthetic["Loan_ID"].str.startswith("SYN_SMOTE_").all())
        self.assertTrue(set(synthetic["Education"]).issubset(set(self.loan_like_df["Education"])))
        self.assertTrue((synthetic["Loan_Status (Approved)"] == "N").all())


if __name__ == "__main__":
    unittest.main()
