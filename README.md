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
- **Probability Calibration (`src/calibration.py`)**: Computes Brier Score Loss and Expected Calibration Error (ECE) for uncalibrated probabilities, Platt Scaling (logistic sigmoid), and Isotonic Regression. To prevent in-sample overfitting, outer training folds are split into 75% model-fit and 25% calibration-fit subsets, fitting calibrators strictly on out-of-sample predictions before refitting models on full outer training folds. Calibration curves are saved to `results/`.
- **Subgroup Fairness Analysis (`src/fairness.py`)**: Disaggregates model performance (Accuracy, Recall, ROC-AUC) across demographic subgroups: `sex` (Male/Female) and `age_band` (`< 50`, `50-60`, `> 60`). Includes explicit methodological caveats regarding small subgroup sample sizes (e.g. N < 100).
- **Notebook**: `notebooks/02_reliability_layer.ipynb`

### Per-Patient Model Agreement Scoring (`src/agreement.py`)
- **Rationale**: An ensemble's averaged probability output can appear unremarkable (e.g., 53% probability) even when individual base classifiers strongly disagree (e.g., Random Forest predicts 20%, XGBoost predicts 85%, and AdaBoost predicts 54%). Model agreement scoring measures internal model consensus per patient.
- **Key Difference from Calibration & Fairness**:
  - *Calibration* checks overall probability honesty against true outcomes across populations.
  - *Fairness* checks model equity across demographic subgroups.
  - *Model Agreement* checks internal model consistency for an individual patient at prediction time.
- **Implementation**: Computes standard deviation across the three base classifier probabilities (RF, XGBoost, AdaBoost) and categorizes patients into `high agreement` (std < 0.05), `moderate agreement` (0.05 <= std < 0.15), or `low agreement` (std >= 0.15). Generates plain-language warning flags (e.g., *"Model agreement: low — RF: 31%, XGBoost: 75%, AdaBoost: 58%. Interpret this prediction with extra caution."*).
- **Notebook**: `notebooks/03_model_agreement.ipynb`

---

## Directory Structure
```
.
├── config.yaml                       # Global project and hyperparameter configuration
├── data/
│   └── raw_cleveland.csv             # UCI Cleveland dataset
├── notebooks/
│   ├── 01_baseline_pipeline.ipynb    # Month 1 Baseline Notebook
│   ├── 02_reliability_layer.ipynb    # Month 2 Reliability Layer Notebook
│   └── 03_model_agreement.ipynb      # Per-Patient Model Agreement Notebook
├── results/                          # Output plots and visualizations
│   ├── adaboost_calibration_curve.png
│   ├── ensemble_calibration_curve.png
│   ├── model_agreement_distribution.png
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
│   ├── fairness.py                   # Demographic subgroup performance breakdown & caveats
│   └── agreement.py                  # Per-patient model agreement scoring & flag generation
├── tests/
│   ├── test_pipeline.py              # Unit tests for baseline pipeline & evaluation
│   ├── test_calibration_fairness.py # Unit tests for calibration and fairness metrics
│   └── test_agreement.py            # Unit tests for model agreement scoring & flags
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
Run any pipeline notebook using `jupyter nbconvert`:
```bash
# Month 1 Baseline Pipeline
jupyter nbconvert --to notebook --execute notebooks/01_baseline_pipeline.ipynb --output notebooks/01_baseline_pipeline.ipynb

# Month 2 Reliability Layer Pipeline
jupyter nbconvert --to notebook --execute notebooks/02_reliability_layer.ipynb --output notebooks/02_reliability_layer.ipynb

# Per-Patient Model Agreement Analysis
jupyter nbconvert --to notebook --execute notebooks/03_model_agreement.ipynb --output notebooks/03_model_agreement.ipynb
```

---

## Pipeline Summary Findings

### 1. Nested Cross-Validation Metrics (5-Fold Outer / 3-Fold Inner)
- **Random Forest**: Accuracy ~82.8%, F1 ~80.3%, ROC-AUC ~90.7%
- **XGBoost**: Accuracy ~82.2%, F1 ~80.3%, ROC-AUC ~90.8%
- **AdaBoost**: Accuracy ~82.2%, F1 ~79.8%, ROC-AUC ~88.8%
- **Soft Ensemble**: Accuracy ~83.2%, F1 ~81.2%, ROC-AUC ~91.3%

### 2. Reliability & Calibration Metrics (Out-Of-Fold Nested CV with Held-Out Calibration Fits)

| Model | Calibration State | Brier Score (Lower is better) | ECE (Lower is better) |
|---|---|---|---|
| **Random Forest** | Uncalibrated | 0.1244 | 0.0877 |
| | Platt Scaling | **0.1274** | **0.0611** |
| | Isotonic Regression | 0.1401 | 0.0811 |
| **XGBoost** | Uncalibrated | 0.1186 | 0.0373 |
| | Platt Scaling | 0.1262 | 0.0423 |
| | Isotonic Regression | 0.1336 | 0.0854 |
| **AdaBoost** | Uncalibrated | 0.1510 | 0.1810 |
| | Platt Scaling | **0.1266** | **0.0499** |
| | Isotonic Regression | 0.1430 | 0.0890 |
| **Ensemble** | Uncalibrated | 0.1248 | 0.1022 |
| | Platt Scaling | **0.1237** | **0.0691** |
| | Isotonic Regression | 0.1341 | 0.0731 |

### 3. Model Agreement Breakdown (Out-Of-Fold Test Predictions)

| Agreement Level | Std Dev Threshold | Patient Count (N) | % of Cohort | Accuracy |
|---|---|---|---|---|
| **High Agreement** | std < 0.05 | 224 | 73.9% | **89.3%** |
| **Moderate Agreement** | 0.05 <= std < 0.15 | 74 | 24.4% | 67.6% |
| **Low Agreement** | std >= 0.15 | 5 | 1.7% | 60.0% |

*Key finding: Prediction accuracy is significantly higher when base models agree (89.3% accuracy for high agreement patients) compared to when models disagree (67.6% for moderate agreement, 60.0% for low agreement). Flagging disagreement provides actionable clinical risk signaling.*
