# Oversampling for Imbalanced Tabular Data

This repository explores synthetic oversampling methods for imbalanced classification problems on tabular data. The project focuses on generating minority-class samples that preserve useful structure in the original data, then comparing those augmented datasets against standard baselines.

The repo contains:

- Gaussian copula based oversampling
- A hybrid oversampler that reshapes marginals toward a target normal-bin profile
- A clustered hybrid oversampler for multi-modal minority classes
- Baseline comparisons with random oversampling and SMOTE
- Model evaluation pipelines for loan approval and credit card fraud style datasets
- Tests for sampler behavior and edge cases

## Methods

### Gaussian Copula

Implemented in [`gaussian_copula_oversampling.py`](./gaussian_copula_oversampling.py).

This method fits a Gaussian copula on minority-class numeric features, samples in latent Gaussian space, and maps the draws back to empirical feature distributions.

### Hybrid Normal-Deficit Oversampling

Also implemented in [`gaussian_copula_oversampling.py`](./gaussian_copula_oversampling.py).

This variant keeps the copula-based dependency structure but replaces marginal generation with a deficit-filling scheme across target normal-like bins.

### Clustered Hybrid

Implemented in [`clustered_hybrid_oversampling.py`](./clustered_hybrid_oversampling.py).

This method first clusters the minority class, then applies the hybrid generator inside each cluster to better handle multi-modal data.

### Baselines

Implemented in [`evaluate_oversampling_models.py`](./evaluate_oversampling_models.py).

The evaluation pipeline now compares:

- `original`
- `random_oversample`
- `smote`
- `copula`
- `hybrid`
- `clustered_hybrid`

For mixed-type tables such as the loan dataset, the SMOTE baseline uses `SMOTENC`.

## Main Files

- [`evaluate_oversampling_models.py`](./evaluate_oversampling_models.py): generates augmented training sets, trains models, saves metrics, and writes comparison plots
- [`compare_copula_vs_hybrid_demo.py`](./compare_copula_vs_hybrid_demo.py): compares copula vs hybrid marginal behavior and saves distribution plots
- [`gaussian_copula_oversampling.py`](./gaussian_copula_oversampling.py): core samplers and mixed-type row synthesis
- [`clustered_hybrid_oversampling.py`](./clustered_hybrid_oversampling.py): clustered hybrid oversampling
- [`tests/test_oversampling.py`](./tests/test_oversampling.py): unit tests for correctness and edge cases
- [`Oversampling.ipynb`](./Oversampling.ipynb), [`Final.ipynb`](./Final.ipynb): exploratory notebooks

## Datasets

The repository does not include the dataset CSV files. To run the experiments, place your datasets locally and pass their paths with `--csv-path`.

Examples used during development:

- `loan_approved_csv.csv`
- `creditcard.csv`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main evaluation pipeline:

```bash
python evaluate_oversampling_models.py
```

Run it with a separate output folder:

```bash
python evaluate_oversampling_models.py --output-dir model_comparison_outputs_full
```

Run it on a local dataset file:

```bash
python evaluate_oversampling_models.py --csv-path path/to/your_dataset.csv
```

Compare copula vs hybrid distribution behavior:

```bash
python compare_copula_vs_hybrid_demo.py
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Outputs

The evaluation script saves:

- Original and augmented train sets
- Synthetic samples for each method
- `model_metrics.csv`
- `dataset_summary.csv`
- `best_by_model_accuracy.csv`
- `best_by_model_f1.csv`
- `minority_f1_comparison.png`
- `balanced_accuracy_comparison.png`

## Results Snapshot

Example local run on the loan dataset with all six methods:

| Dataset | Mean Accuracy | Mean Balanced Accuracy | Mean Minority F1 |
| --- | ---: | ---: | ---: |
| original | 0.8347 | 0.7688 | 0.6915 |
| copula | 0.7995 | 0.7773 | 0.6892 |
| hybrid | 0.8266 | 0.7678 | 0.6865 |
| random_oversample | 0.7940 | 0.7564 | 0.6638 |
| clustered_hybrid | 0.7995 | 0.7555 | 0.6636 |
| smote | 0.8022 | 0.7526 | 0.6608 |

Per-model winners by minority F1 from the same local run:

| Model | Best Dataset | Minority F1 |
| --- | --- | ---: |
| Logistic Regression | copula | 0.7250 |
| Random Forest | original | 0.6866 |
| XGBoost | hybrid | 0.6944 |

In that local run, the copula dataset produced the best average minority F1 across models, while the original dataset retained the best average accuracy.

The higher accuracy for `original` should not be treated as proof that the imbalanced dataset is better. In imbalanced classification, accuracy can stay high even when the model performs poorly on the minority class. For example, if a dataset has 90 samples from class A and 10 from class B, a model that predicts every case as class A can still achieve about 90% accuracy while completely failing to identify class B. Because of that, `balanced_accuracy`, minority-class `recall`, and minority-class `f1` are more informative than raw accuracy for this project.

Generated CSV summaries and PNG plots are intentionally excluded from the repository by `.gitignore`, so you should expect those artifacts to appear only after running the scripts locally.

## Tests

The test suite covers:

- Copula sampler output shape, bounds, and label handling
- Hybrid generation with constant-feature edge cases
- Mixed-type row synthesis with synthetic ID generation
- Clustered hybrid allocation counts
- Random oversampling and SMOTE baselines on mixed-type data

## Repository Structure

```text
.
|-- gaussian_copula_oversampling.py
|-- clustered_hybrid_oversampling.py
|-- evaluate_oversampling_models.py
|-- compare_copula_vs_hybrid_demo.py
|-- requirements.txt
|-- tests/
|-- Oversampling.ipynb
|-- Final.ipynb
`-- .gitignore
```

## Next Steps

- Add a dedicated script for credit card experiments using the same baseline set
- Add statistical validation beyond skewness, kurtosis, and normality-score summaries
- Add reproducible experiment configs for multiple datasets and output folders
