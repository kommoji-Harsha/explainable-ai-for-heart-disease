# Optimized, Explainable, Reliable Ensemble Framework for Heart Disease Prediction

This repository implements the research framework for heart disease prediction based on the UCI Cleveland dataset.

## Framework Architecture & Progression

### Month 1 — Baseline Pipeline
- **Preprocessing**: Handles missing values (`?`), binarizes target `num` (0: no disease, 1: disease present), performs median/mode imputation, scaling, and one-hot encoding.
- **Classifiers**: Random Forest, XGBoost, and AdaBoost.
- **Hyperparameter Optimization**: Bayesian optimization using Optuna (TPE) with inner cross-validation.
- **Ensemble**: Soft-voting ensemble combining the three tuned classifiers.
- **Nested Cross-Validation**: Leak-free 5-fold outer / 3-fold inner nested CV ensuring preprocessing and tuning are performed strictly on training folds.
- **Explainability**: SHAP summary beeswarm, feature importance bar, and individual prediction waterfall plots saved to `results/`.
- **Notebook**: `notebooks/01_baseline_pipeline.ipynb`

### Month 2 — Reliability Layer
- **Probability Calibration (`src/calibration.py`)**: Computes Brier Score Loss and Expected Calibration Error (ECE) for uncalibrated probabilities, Platt Scaling (logistic sigmoid), and Isotonic Regression. Calibrators are fitted strictly on training fold predictions during nested CV to prevent data leakage. Calibration curves saved to `results/`.
- **Subgroup Fairness Analysis (`src/fairness.py`)**: Disaggregates model performance (Accuracy, Recall, ROC-AUC) across demographic subgroups: `sex` (Male/Female) and `age_band` (`< 50`, `50-60`, `> 60`). Includes explicit methodological caveats regarding small subgroup sample sizes (e.g. N < 100).
- **Notebook**: `notebooks/02_reliability_layer.ipynb`

---

## Directory Structure
```
.
├── config.yaml                       # Global project and hyperparameter configuration
├── data/
│   └── raw_cleveland.csv             # UCI Cleveland dataset
├── notebooks/
│   ├── 01_baseline_pipeline.ipynb    # Month 1 Baseline Notebook
│   └── 02_reliability_layer.ipynb    # Month 2 Reliability Layer Notebook
├── results/                          # Output plots and visualizations
│   ├── adaboost_calibration_curve.png
│   ├── ensemble_calibration_curve.png
│   ├── nested_cv_confusion_matrices.png
│   ├── random_forest_calibration_curve.png
│   ├── xgboost_calibration_curve.png
│   ├── xgboost_shap_bar.png
│   ├── xgboost_shap_sample_0.png
│   └── xgboost_shap_summary.png
├── src/
│   ├── preprocessing.py              # Data loading, cleaning, imputation, encoding, scaling
│   ├── models.py                     # Baseline model instantiators (RF, XGBoost, AdaBoost)
│   ├── optimization.py               # Optuna Bayesian hyperparameter optimization
│   ├── ensemble.py                   # Soft-voting ensemble implementation
│   ├── evaluation.py                 # Classification metrics & nested cross-validation
│   ├── explainability.py             # SHAP explanation generation and visualization
│   ├── calibration.py                # Brier score, ECE, Platt scaling, Isotonic regression
│   └── fairness.py                   # Demographic subgroup performance breakdown & caveats
├── tests/
│   ├── test_pipeline.py              # Unit tests for baseline pipeline & evaluation
│   └── test_calibration_fairness.py # Unit tests for calibration and fairness metrics
├── requirements.txt                  # Pinned dependency requirements
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
Run either pipeline notebook using `jupyter nbconvert`:
```bash
# Month 1 Baseline Pipeline
jupyter nbconvert --to notebook --execute notebooks/01_baseline_pipeline.ipynb --output notebooks/01_baseline_pipeline.ipynb

# Month 2 Reliability Layer Pipeline
jupyter nbconvert --to notebook --execute notebooks/02_reliability_layer.ipynb --output notebooks/02_reliability_layer.ipynb
```

---

## Pipeline Summary Findings

### 1. Nested Cross-Validation Metrics (5-Fold Outer / 3-Fold Inner)
- **Random Forest**: Accuracy ~82.8%, F1 ~80.3%, ROC-AUC ~90.7%
- **XGBoost**: Accuracy ~82.2%, F1 ~80.3%, ROC-AUC ~90.8%
- **AdaBoost**: Accuracy ~82.2%, F1 ~79.8%, ROC-AUC ~88.8%
- **Soft Ensemble**: Accuracy ~83.2%, F1 ~81.2%, ROC-AUC ~91.3%

### 2. Reliability & Calibration Metrics (Out-Of-Fold Nested CV)
- **Uncalibrated Ensemble**: Brier Score ~0.1397, ECE ~0.1302
- **Platt Calibrated Ensemble**: Brier Score ~0.1288, ECE ~0.0862
- **Isotonic Calibrated Ensemble**: Brier Score ~0.1312, ECE ~0.0915

*Platt scaling effectively reduces Expected Calibration Error (ECE) and improves probability reliability across predictions.*

### 3. Subgroup Fairness Breakdown (Soft Ensemble)
- **Sex Breakdown**:
  - Female (N=97): Accuracy ~89.7%, Recall ~72.0%, ROC-AUC ~93.8%
  - Male (N=206): Accuracy ~80.6%, Recall ~80.7%, ROC-AUC ~89.8%
- **Age Band Breakdown**:
  - `< 50` (N=88): Accuracy ~85.2%, Recall ~63.0%, ROC-AUC ~90.0%
  - `50-60` (N=127): Accuracy ~80.3%, Recall ~78.0%, ROC-AUC ~89.3%
  - `> 60` (N=88): Accuracy ~85.2%, Recall ~88.0%, ROC-AUC ~93.2%

*Caveat: Subgroup sample sizes are small (N < 100 per subgroup in female and age categories). Performance differences should be viewed as exploratory rather than statistically definitive.*
