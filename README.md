# Optimized, Explainable, Reliable Ensemble Framework for Heart Disease Prediction

This repository implements the research framework for heart disease prediction based on the UCI Cleveland dataset and cross-dataset external validation on the Framingham Heart Study cohort.

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
- **Per-Patient Model Agreement Scoring (`src/agreement.py`)**: Measures consensus across base classifiers per patient using probability standard deviation, categorizing patients into `high agreement` (std < 0.05), `moderate agreement` (0.05 <= std < 0.15), and `low agreement` (std >= 0.15), generating plain-language warning flags.
- **Notebooks**: `notebooks/02_reliability_layer.ipynb` and `notebooks/03_model_agreement.ipynb`

### Month 3 — Cross-Dataset External Validation (`src/data_harmonization.py` & `src/external_validation.py`)
- **Data Harmonization**: Maps the UCI Cleveland dataset (303 rows) and Framingham Heart Study dataset (4,240 rows) onto a unified common 5-feature schema: `age` (years), `sex` (0=Female, 1=Male), `sysBP` (systolic BP in mmHg), `totChol` (total cholesterol in mg/dL), `diabetes` (0/1 binary).
- **Documentation of Feature Loss**:
  - *Dropped from Cleveland*: `cp`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.
  - *Dropped from Framingham*: `education`, `currentSmoker`, `cigsPerDay`, `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diaBP`, `BMI`, `heartRate`, `glucose`.
- **External Evaluation**: Trains tuned classifiers and soft-voting ensemble on the full Cleveland dataset (harmonized schema) and evaluates them directly on the unseen Framingham dataset.
- **Notebook**: `notebooks/04_cross_dataset_validation.ipynb`

---

## Directory Structure
```
.
├── config.yaml                       # Global project and hyperparameter configuration
├── data/
│   ├── raw_cleveland.csv             # UCI Cleveland dataset
│   └── raw_framingham.csv            # Framingham Heart Study dataset
├── notebooks/
│   ├── 01_baseline_pipeline.ipynb    # Month 1 Baseline Notebook
│   ├── 02_reliability_layer.ipynb    # Month 2 Reliability Layer Notebook
│   ├── 03_model_agreement.ipynb      # Per-Patient Model Agreement Notebook
│   └── 04_cross_dataset_validation.ipynb # Month 3 Cross-Dataset External Validation
├── results/                          # Output plots and visualizations
│   ├── adaboost_calibration_curve.png
│   ├── ensemble_calibration_curve.png
│   ├── framingham_external_confusion_matrices.png
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
│   ├── agreement.py                  # Per-patient model agreement scoring & flag generation
│   ├── data_harmonization.py         # Cleveland & Framingham dataset schema harmonization
│   └── external_validation.py        # Cross-dataset model training and external evaluation
├── tests/
│   ├── test_pipeline.py              # Unit tests for baseline pipeline & evaluation
│   ├── test_calibration_fairness.py # Unit tests for calibration and fairness metrics
│   ├── test_agreement.py            # Unit tests for model agreement scoring & flags
│   └── test_data_harmonization.py   # Unit tests for dataset harmonization & external validation
├── requirements.txt                  # Pinned dependency requirements
├── SRS.md                            # Software Requirements Specification
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

# Month 3 Cross-Dataset External Validation
jupyter nbconvert --to notebook --execute notebooks/04_cross_dataset_validation.ipynb --output notebooks/04_cross_dataset_validation.ipynb
```

---

## Pipeline Summary Findings

### 1. Full Cleveland Feature Set Nested CV (Month 1 Baseline)
- **Random Forest**: Accuracy ~82.8%, F1 ~80.3%, ROC-AUC ~90.7%
- **XGBoost**: Accuracy ~82.2%, F1 ~80.3%, ROC-AUC ~90.8%
- **AdaBoost**: Accuracy ~82.2%, F1 ~79.8%, ROC-AUC ~88.8%
- **Soft Ensemble**: Accuracy ~83.2%, F1 ~81.2%, ROC-AUC ~91.3%

### 2. Cross-Dataset External Validation Metrics (Month 3 Harmonized Schema)

| Model | Cleveland CV Accuracy (5 Features) | Framingham Ext Accuracy (5 Features) | Cleveland CV ROC-AUC | Framingham Ext ROC-AUC |
|---|---|---|---|---|
| **Random Forest** | 0.6801 | **0.7302** | 0.7304 | **0.6559** |
| **XGBoost** | 0.6700 | **0.7771** | 0.7271 | **0.6764** |
| **AdaBoost** | 0.6336 | **0.7863** | 0.6973 | **0.6747** |
| **Soft Ensemble** | 0.6799 | **0.7545** | 0.7295 | **0.6715** |

*Key finding: Moving from full Cleveland features (13 attributes) to a 5-feature harmonized schema reduces internal CV performance, but models trained on Cleveland generalize to the external Framingham cohort (4,240 patients) with moderate ROC-AUC (~0.6715 for Ensemble, ~0.6764 for XGBoost).*
