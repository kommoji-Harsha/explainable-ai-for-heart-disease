# Optimized, Explainable, Reliable Ensemble Framework for Heart Disease Prediction

This repository implements Month 1 — Baseline of the research framework for heart disease prediction based on the UCI Cleveland dataset.

## Framework Overview & Architecture
The Month 1 baseline phase establishes a leak-free Machine Learning pipeline combining ensemble learning, Optuna Bayesian hyperparameter optimization, nested cross-validation, and SHAP-based explainability.

### Features
- **Data Acquisition & Preprocessing**: Loads the UCI Cleveland dataset (`processed.cleveland.data`), converts missing values (`?`) into proper representation, imputes missing values (median for numeric, mode for categorical), one-hot encodes categorical variables, and scales numerical features. Converts target `num` to binary classification (0: no disease, 1: disease present).
- **Classifiers**: Random Forest, XGBoost, and AdaBoost classifiers with modular configuration.
- **Optuna Hyperparameter Optimization**: Bayesian optimization using Tree-structured Parzen Estimators (TPE) with inner cross-validation.
- **Ensemble Learning**: Soft-voting ensemble combining predictions from tuned Random Forest, XGBoost, and AdaBoost models.
- **Leak-Free Nested Cross-Validation**: 5 outer folds / 3 inner folds nested cross-validation pipeline ensuring preprocessing and hyperparameter tuning are conducted strictly inside training folds to prevent optimistic evaluation bias.
- **Comprehensive Metrics**: Reports Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix across outer cross-validation folds.
- **Explainability**: SHAP (SHapley Additive exPlanations) summary beeswarm plots, feature importance bar charts, and individual prediction waterfall plots saved into `results/`.

---

## Directory Structure
```
.
├── config.yaml                   # Global project and hyperparameter configuration
├── data/
│   └── raw_cleveland.csv         # UCI Cleveland raw/processed dataset
├── notebooks/
│   └── 01_baseline_pipeline.ipynb# End-to-end executable Jupyter notebook
├── results/                      # SHAP plots and confusion matrices
│   ├── nested_cv_confusion_matrices.png
│   ├── xgboost_shap_bar.png
│   ├── xgboost_shap_sample_0.png
│   ├── xgboost_shap_sample_1.png
│   └── xgboost_shap_summary.png
├── src/
│   ├── preprocessing.py          # Data loading, cleaning, imputation, encoding, scaling
│   ├── models.py                 # Baseline model instantiators (RF, XGBoost, AdaBoost)
│   ├── optimization.py           # Optuna Bayesian hyperparameter optimization
│   ├── ensemble.py               # Soft-voting ensemble implementation
│   ├── evaluation.py             # Classification metrics & nested cross-validation
│   └── explainability.py         # SHAP explanation generation and visualization
├── tests/
│   └── test_pipeline.py          # Pytest unit tests for preprocessing and evaluation
├── requirements.txt              # Pinned dependency requirements
└── README.md
```

---

## Setup Instructions

### 1. Environment Requirements
- Python 3.10+

### 2. Installation
Clone the repository and install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Usage

### Running Tests
Execute the unit test suite with `pytest`:
```bash
PYTHONPATH=. pytest -v
```

### Running the End-to-End Notebook
Launch Jupyter Notebook or run via `jupyter nbconvert`:
```bash
jupyter nbconvert --to notebook --execute notebooks/01_baseline_pipeline.ipynb --output notebooks/01_baseline_pipeline.ipynb
```

---

## Baseline Pipeline Results Summary

When evaluated using 5-fold outer / 3-fold inner nested cross-validation on the Cleveland dataset (303 instances), representative metrics are:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | ~82.8% | ~84.2% | ~77.0% | ~80.3% | ~90.7% |
| **XGBoost** | ~82.2% | ~81.0% | ~79.9% | ~80.3% | ~90.8% |
| **AdaBoost** | ~82.2% | ~81.7% | ~78.4% | ~79.8% | ~88.8% |
| **Soft Ensemble** | ~83.2% | ~83.5% | ~79.2% | ~81.2% | ~91.3% |

*Note: Results are evaluated strictly with nested CV to prevent data leakage and hyperparameter overfitting.*
