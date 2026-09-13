# Optimized, Explainable, Reliable Ensemble Framework for Heart Disease Prediction

This repository implements the research framework for heart disease prediction based on the primary dataset migration from Cleveland-only (303 patients) to the **COMBINED 4-site UCI Heart Disease dataset** (Cleveland, Hungarian, Switzerland, VA Long Beach; **920 patients total**), along with cross-dataset external validation on the Framingham Heart Study cohort.

## Framework Architecture & Progression

### 1. Data Ingestion & Preprocessing (`src/preprocessing.py`)
- **Multi-Site Concatenation**: Ingests Cleveland (N=303), Hungarian (N=294), Switzerland (N=123), and VA Long Beach (N=200). Total N = 920.
- **Source Site Tracking**: Adds metadata column `source_site` (`cleveland`, `hungarian`, `switzerland`, `va_long_beach`) to track provenance without feeding it as a predictive feature.
- **Data Quality Fix — Zero-as-Missing Conversion**: In non-Cleveland UCI files (Hungarian, Switzerland, VA Long Beach), unrecorded continuous measurements (`chol`, `trestbps`, `thalach`) were recorded as literal `0`s instead of `'?'`. `load_single_site()` explicitly converts physiologically implausible zero values in continuous variables to `NaN` for non-Cleveland sites (e.g., converting 123 cholesterol zeros in Switzerland and 49 in VA Long Beach), preventing fake zero values from distorting median imputation and scaling statistics.
- **Missingness Reporting**: Documented missingness report saved to `results/combined_missingness_report.csv`. Notes high missingness in `ca` (66.4% combined), `thal` (52.8% combined), and corrected `chol` (21.96% combined).
- **Preprocessing Pipeline**: Handles missing values (`?` and 0s in continuous non-Cleveland attributes), binarizes target `num` (0: no disease, 1: disease present), performs median/mode imputation, scaling, and one-hot encoding.

### 2. Base Classifiers & Optuna Tuning (`src/models.py`, `src/optimization.py`, `src/ensemble.py`)
- **Classifiers**: Random Forest, XGBoost, and AdaBoost.
- **Hyperparameter Optimization**: Bayesian optimization using Optuna (TPE) with inner cross-validation.
- **Ensemble**: Soft-voting ensemble combining the three tuned classifiers.

### 3. Leak-Free Nested Cross-Validation & Calibration (`src/evaluation.py`, `src/calibration.py`)
- **Nested CV**: 5-fold outer / 3-fold inner nested CV ensuring preprocessing and hyperparameter tuning are performed strictly on training folds.
- **Out-of-Sample Calibration**: To prevent in-sample overfitting, outer training folds are split into 75% model-fit and 25% calibration-fit subsets, fitting Platt Scaling and Isotonic Regression calibrators strictly on out-of-sample predictions before refitting models on full outer training folds.

### 4. Subgroup Fairness & Per-Patient Model Agreement (`src/fairness.py`, `src/agreement.py`)
- **Subgroup Fairness**: Disaggregates model performance across `sex`, `age_band`, and **`source_site`**. For per-site analysis, reports **Accuracy, Recall, Precision, Specificity (TNR), F1-Score, and ROC-AUC** alongside disease prevalence to guard against positive-class prediction bias in high-prevalence cohorts (e.g. Switzerland at 92.7% disease prevalence).
- **Per-Patient Agreement**: Measures consensus across base classifiers per patient using probability standard deviation, categorizing patients into `high agreement` (std < 0.05), `moderate agreement` (0.05 <= std < 0.15), and `low agreement` (std >= 0.15), generating plain-language warning flags.

### 5. Cross-Dataset External Validation (`src/data_harmonization.py` & `src/external_validation.py`)
- **Schema Harmonization**: Harmonizes Cleveland/Combined datasets and the Framingham Heart Study dataset (4,240 patients) onto a 5-feature common schema (`age`, `sex`, `sysBP`, `totChol`, `diabetes`, `target`).
- **External Evaluation**: Trains tuned classifiers and soft-voting ensemble on full harmonized Cleveland data and evaluates directly on the unseen Framingham cohort.

---

## Directory Structure
```
.
├── config.yaml                                 # Global project and hyperparameter configuration
├── SRS.md                                      # Living Software Requirements Specification
├── data/
│   ├── raw_cleveland.csv                       # UCI Cleveland dataset (N=303)
│   ├── raw_hungarian.csv                       # UCI Hungarian dataset (N=294)
│   ├── raw_switzerland.csv                     # UCI Switzerland dataset (N=123)
│   ├── raw_va_long_beach.csv                   # UCI VA Long Beach dataset (N=200)
│   └── raw_framingham.csv                      # Framingham Heart Study dataset (N=4,240)
├── notebooks/
│   ├── 01_baseline_pipeline.ipynb              # Original Cleveland-only Baseline Notebook
│   ├── 01b_combined_dataset_baseline.ipynb     # Combined 4-Site Baseline Notebook (N=920)
│   ├── 02_reliability_layer.ipynb              # Original Cleveland Reliability Notebook
│   ├── 02b_combined_reliability_and_agreement.ipynb # Combined 4-Site Reliability & Agreement Notebook
│   ├── 03_model_agreement.ipynb                # Per-Patient Model Agreement Notebook
│   └── 04_cross_dataset_validation.ipynb       # Cross-Dataset External Validation
├── results/                                    # Output plots and visualizations
│   ├── combined_adaboost_calibration_curve.png
│   ├── combined_ensemble_calibration_curve.png
│   ├── combined_missingness_report.csv         # Documented missingness % across 4 sites (with 0-as-missing fix)
│   ├── combined_model_agreement_distribution.png
│   ├── combined_nested_cv_confusion_matrices.png
│   ├── combined_random_forest_calibration_curve.png
│   ├── combined_xgboost_calibration_curve.png
│   ├── combined_xgboost_shap_bar.png
│   ├── combined_xgboost_shap_summary.png
│   ├── framingham_external_confusion_matrices.png
│   └── nested_cv_confusion_matrices.png
├── src/
│   ├── preprocessing.py                        # Multi-source data loading, concatenation, 0-as-missing fix, scaling
│   ├── models.py                               # Baseline model instantiators (RF, XGBoost, AdaBoost)
│   ├── optimization.py                         # Optuna Bayesian hyperparameter optimization
│   ├── ensemble.py                             # Soft-voting ensemble implementation
│   ├── evaluation.py                           # Classification metrics & nested cross-validation
│   ├── explainability.py                       # SHAP explanation generation and visualization
│   ├── calibration.py                          # Brier score, ECE, Platt scaling, Isotonic regression
│   ├── fairness.py                             # Demographic & per-site performance breakdown (with Precision & Specificity)
│   ├── agreement.py                            # Per-patient model agreement scoring & flag generation
│   ├── data_harmonization.py                   # Cleveland & Framingham dataset schema harmonization
│   └── external_validation.py                  # Cross-dataset model training and external evaluation
├── tests/
│   ├── test_pipeline.py                        # Unit tests for baseline pipeline & evaluation
│   ├── test_calibration_fairness.py           # Unit tests for calibration and fairness metrics
│   ├── test_agreement.py                      # Unit tests for model agreement scoring & flags
│   ├── test_combined_dataset.py               # Unit tests for 4-site multi-source loading & zero-as-missing fix
│   └── test_data_harmonization.py             # Unit tests for dataset harmonization & external validation
├── requirements.txt                            # Pinned dependency requirements
└── README.md
```

---

## Setup Instructions

### 1. Environment Requirements
- Python 3.10+

### 2. Installation
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Usage

### Running Unit Tests
Execute all tests using `pytest`:
```bash
PYTHONPATH=. pytest -v
```

### Running Executable Notebooks
Run any pipeline notebook using `jupyter nbconvert`:
```bash
# Combined 4-Site Baseline Pipeline (N=920)
jupyter nbconvert --to notebook --execute notebooks/01b_combined_dataset_baseline.ipynb --output notebooks/01b_combined_dataset_baseline.ipynb

# Combined 4-Site Reliability Layer & Model Agreement Pipeline
jupyter nbconvert --to notebook --execute notebooks/02b_combined_reliability_and_agreement.ipynb --output notebooks/02b_combined_reliability_and_agreement.ipynb
```

---

## Pipeline Summary Findings

### 1. Cleveland-Only vs Corrected Combined 4-Site Cohort Baseline Comparison (5-Fold Outer Nested CV)

| Metric / Cohort | Model | Cleveland-Only (N=303) | Corrected Combined 4-Site Cohort (N=920) |
|---|---|---|---|
| **Accuracy** | Random Forest | 82.8% ± 0.9% | **81.5% ± 3.4%** |
| | XGBoost | 82.2% ± 0.4% | **82.1% ± 3.8%** |
| | AdaBoost | 82.2% ± 1.7% | **81.0% ± 3.1%** |
| | Soft Ensemble | 83.2% ± 1.4% | **81.8% ± 3.2%** |
| **ROC-AUC** | Random Forest | 90.7% ± 0.8% | **89.5% ± 2.6%** |
| | XGBoost | 90.8% ± 1.7% | **88.6% ± 3.3%** |
| | AdaBoost | 88.8% ± 0.5% | **88.2% ± 2.5%** |
| | Soft Ensemble | 91.3% ± 1.6% | **89.7% ± 2.8%** |

### 2. Expanded Per-Site Subgroup Performance Breakdown (Soft Ensemble on Combined Cohort)

| Source Site | Sample Size (N) | Disease Prevalence | Accuracy | Recall | Precision | Specificity (TNR) | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| **Cleveland** | 303 | 45.9% | 84.8% | 81.3% | 84.8% | 87.8% | 83.0% | 91.8% |
| **Hungarian** | 294 | 36.1% | 83.3% | 71.7% | 80.2% | 89.9% | 75.7% | 89.2% |
| **Switzerland** | 123 | 92.7% | 91.1% | 97.4% | 93.3% | 11.1% | 95.3% | 83.6% |
| **VA Long Beach** | 200 | 74.5% | 71.5% | 78.5% | 82.5% | 51.0% | 80.5% | 76.5% |

*Expanded Per-Site Fairness Analysis & Class-Imbalance Caveat:*
- High Accuracy (91.1%) and Recall (97.4%) in **Switzerland** reflect extreme positive-class prevalence (92.7% disease rate) where the model heavily predicts the majority positive class. Contextualizing with **Specificity (11.1%)** reveals that negative cases are rarely identified correctly.
- In **VA Long Beach** (74.5% prevalence), Specificity is 51.0% with an F1-score of 80.5%.
- Contextualizing Accuracy and Recall alongside Precision and Specificity guards against misinterpreting class-imbalance effects as superior site-specific performance.
