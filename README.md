# Optimized, Explainable, Reliable Ensemble Framework for Heart Disease Prediction

This repository implements the research framework for heart disease prediction based on the primary dataset migration from Cleveland-only (303 patients) to the **COMBINED 4-site UCI Heart Disease dataset** (Cleveland, Hungarian, Switzerland, VA Long Beach; **920 patients total**), along with cross-dataset external validation on the Framingham Heart Study cohort (4,240 patients), explanation depth modules (LIME, counterfactuals, explanation stability), and final model artifact deployment.

## Quick Demo

You can train the final model on the full 920-patient dataset and run live predictions in just two commands:

```bash
# 1. Fit preprocessor, base models, and ensemble on full dataset and save to models/
PYTHONPATH=. python3 src/train_final_model.py

# 2. Run single-patient risk assessment CLI
PYTHONPATH=. python3 src/predict.py --age 60 --sex 1 --cp 4 --trestbps 140 --chol 260 --thalach 130 --exang 1 --oldpeak 2.0
```

To see raw JSON output:
```bash
PYTHONPATH=. python3 src/predict.py --json
```

To run the interactive presentation notebook:
```bash
jupyter nbconvert --to notebook --execute notebooks/06_live_demo.ipynb --output 06_live_demo.ipynb
```

---

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

### 5. Explanation Depth (`src/lime_explainability.py`, `src/counterfactuals.py`, `src/explanation_stability.py`)
- **SHAP vs. LIME Local Explainability**: Compares top-5 features identified by SHAP (TreeExplainer) vs. LIME (`LimeTabularExplainer`) on individual patient predictions, computing Overlap@5 agreement ratios (saved to `results/shap_vs_lime_comparison.csv`).
- **Counterfactual What-If Reasoning**: Uses a grid search over actionable features (`chol`, `trestbps`, `thalach`, `oldpeak`) to identify minimal realistic modifications that flip high-risk patient predictions below 0.50 risk. Statements explicitly include non-medical-advice disclaimers (saved to `results/counterfactual_scenarios.txt`).
- **Explanation Stability Across CV Folds**: Measures fold-to-fold ranking stability of top SHAP features across 5 outer cross-validation folds using Overlap@5 and Kendall's tau correlation.
- **Notebook**: `notebooks/05_explanation_depth.ipynb`

### 6. Cross-Dataset External Validation (`src/data_harmonization.py` & `src/external_validation.py`)
- **Combined 4-Site Training Cohort**: The framework trains tuned models and the soft-voting ensemble on the full harmonized combined 4-site dataset (920 patients across 4 international sites).
- **Framingham External Cohort**: Evaluates trained models directly on the unseen Framingham Heart Study dataset (4,240 patients).
- **Schema Harmonization**: Harmonizes both datasets onto a 5-feature common schema (`age`, `sex`, `sysBP`, `totChol`, `diabetes`, `target`).
- **Extended Evaluation**: Computes Precision, Recall, F1, Specificity, ROC-AUC, PR-AUC, and checks for degenerate constant predictions to account for Framingham's lower disease prevalence (15.2%).
- **Notebook**: `notebooks/04_cross_dataset_validation.ipynb`

### 7. Final Model Training & Inference Pipeline (`src/train_final_model.py` & `src/predict.py`)
- **Final Full Training**: Fits preprocessor and Optuna-tuned base models + soft ensemble on the full 920-patient cohort.
- **Persisted Artifacts**: Saves `preprocessor.joblib`, `random_forest.joblib`, `xgboost.joblib`, `adaboost.joblib`, `ensemble.joblib`, and `metadata.json` to `models/`.
- **Inference & CLI**: `src/predict.py` takes raw feature inputs and outputs probability, risk label, model agreement flag, top SHAP features, and explicit disclaimer: *"This is a machine learning estimate, not a medical diagnosis."*
- **Presentation Notebook**: `notebooks/06_live_demo.ipynb`

---

## Directory Structure

.
├── config.yaml # Global project and hyperparameter configuration
├── SRS.md # Living Software Requirements Specification
├── data/
│ ├── raw_cleveland.csv # UCI Cleveland dataset (N=303)
│ ├── raw_hungarian.csv # UCI Hungarian dataset (N=294)
│ ├── raw_switzerland.csv # UCI Switzerland dataset (N=123)
│ ├── raw_va_long_beach.csv # UCI VA Long Beach dataset (N=200)
│ └── raw_framingham.csv # Framingham Heart Study dataset (N=4,240)
├── models/ # Saved model artifacts and metadata
│ ├── adaboost.joblib
│ ├── ensemble.joblib
│ ├── metadata.json
│ ├── preprocessor.joblib
│ ├── random_forest.joblib
│ └── xgboost.joblib
├── notebooks/
│ ├── 01_baseline_pipeline.ipynb # Original Cleveland-only Baseline Notebook
│ ├── 01b_combined_dataset_baseline.ipynb # Combined 4-Site Baseline Notebook (N=920)
│ ├── 02_reliability_layer.ipynb # Original Cleveland Reliability Notebook
│ ├── 02b_combined_reliability_and_agreement.ipynb # Combined 4-Site Reliability & Agreement Notebook
│ ├── 03_model_agreement.ipynb # Per-Patient Model Agreement Notebook
│ ├── 04_cross_dataset_validation.ipynb # 3-Way Cross-Dataset External Validation
│ ├── 05_explanation_depth.ipynb # Explanation Depth (LIME, Counterfactuals, Stability)
│ └── 06_live_demo.ipynb # Presentation-ready Live Demonstration Notebook
├── results/ # Output plots and visualizations
│ ├── combined_adaboost_calibration_curve.png
│ ├── combined_ensemble_calibration_curve.png
│ ├── combined_missingness_report.csv # Documented missingness % across 4 sites
│ ├── combined_model_agreement_distribution.png
│ ├── combined_nested_cv_confusion_matrices.png
│ ├── combined_random_forest_calibration_curve.png
│ ├── combined_xgboost_calibration_curve.png
│ ├── combined_xgboost_shap_bar.png
│ ├── combined_xgboost_shap_summary.png
│ ├── counterfactual_scenarios.txt # Counterfactual what-if statements with disclaimers
│ ├── framingham_external_confusion_matrices.png
│ ├── nested_cv_confusion_matrices.png
│ └── shap_vs_lime_comparison.csv # Top feature comparison & Overlap@5 ratios
├── src/
│ ├── preprocessing.py # Multi-source data loading, concatenation, 0-as-missing fix, scaling
│ ├── models.py # Baseline model instantiators (RF, XGBoost, AdaBoost)
│ ├── optimization.py # Optuna Bayesian hyperparameter optimization
│ ├── ensemble.py # Soft-voting ensemble implementation
│ ├── evaluation.py # Classification metrics & nested cross-validation
│ ├── explainability.py # SHAP explanation generation and visualization
│ ├── calibration.py # Brier score, ECE, Platt scaling, Isotonic regression
│ ├── fairness.py # Demographic & per-site performance breakdown
│ ├── agreement.py # Per-patient model agreement scoring & flag generation
│ ├── lime_explainability.py # LIME tabular explainer & SHAP vs LIME comparison
│ ├── counterfactuals.py # Counterfactual search & what-if scenario statements
│ ├── explanation_stability.py # SHAP ranking stability across CV folds (Overlap@K & Kendall's tau)
│ ├── data_harmonization.py # Combined UCI & Framingham dataset schema harmonization
│ ├── external_validation.py # Cross-dataset model training and 3-way external evaluation
│ ├── train_final_model.py # Trains final preprocessor & models on full dataset
│ └── predict.py # Patient prediction CLI, risk labeling, & SHAP explanations
├── tests/
│ ├── test_pipeline.py # Unit tests for baseline pipeline & evaluation
│ ├── test_calibration_fairness.py # Unit tests for calibration and fairness metrics
│ ├── test_agreement.py # Unit tests for model agreement scoring & flags
│ ├── test_combined_dataset.py # Unit tests for 4-site multi-source loading & zero-as-missing fix
│ ├── test_explanation_depth.py # Unit tests for Overlap@K, Kendall's tau, and counterfactuals
│ ├── test_data_harmonization.py # Unit tests for dataset harmonization & external validation
│ └── test_predict.py # Unit tests for final model loading and prediction format
├── requirements.txt # Pinned dependency requirements
└── README.md


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

# Explanation Depth (LIME, Counterfactuals, SHAP Stability)
jupyter nbconvert --to notebook --execute notebooks/05_explanation_depth.ipynb --output notebooks/05_explanation_depth.ipynb

# Month 3 Cross-Dataset External Validation Pipeline
jupyter nbconvert --to notebook --execute notebooks/04_cross_dataset_validation.ipynb --output notebooks/04_cross_dataset_validation.ipynb

# Live Presentation Demo
jupyter nbconvert --to notebook --execute notebooks/06_live_demo.ipynb --output notebooks/06_live_demo.ipynb
```

---

## Pipeline Summary Findings

### 1. Explanation Depth Summary (LIME Comparison, Counterfactuals, Stability)
- **SHAP vs. LIME Agreement**: Evaluated top-5 feature agreement across sample patients. Average **Overlap@5 is 80.0%** (sharing key top features such as `cp`, `oldpeak`, `thalach`, `exang`, `thal`), indicating strong local explanation consensus between TreeExplainer and LimeTabularExplainer.
- **Counterfactual What-If Reasoning**: Successfully generated minimal realistic feature modifications for high-risk patients (e.g., reducing `oldpeak` or increasing `thalach`) that flip model predictions below 0.50 risk. All statements explicitly include non-medical-advice disclaimers.
- **SHAP Explanation Stability Across CV Folds**: Evaluated SHAP ranking consistency across 5 outer cross-validation folds on the 920-patient combined dataset:
  - **Mean Overlap@5**: **80.0%**
  - **Mean Kendall's Tau Correlation**: **0.780**
  - *Interpretation*: High explanation stability. The model consistently identifies `cp` (chest pain type), `oldpeak`, `thalach`, `ca`, and `exang` as top risk drivers regardless of fold data split variations.

### 2. 3-Way Performance Transferability Comparison (Harmonized 5-Feature Schema)

| Model | Cleveland-Only CV Acc (5 Feat) | Combined 4-Site CV Acc (5 Feat) | Framingham External Acc (5 Feat) | Cleveland CV ROC-AUC | Combined 4-Site CV ROC-AUC | Framingham External ROC-AUC | Framingham External F1 | Framingham External PR-AUC |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** | 68.01% | 69.13% | **70.14%** | 73.04% | **74.99%** | **67.84%** | 0.3386 | 0.2580 |
| **XGBoost** | 67.00% | 68.59% | **71.67%** | 72.71% | **74.89%** | **69.30%** | 0.3546 | 0.2705 |
| **AdaBoost** | 63.36% | 68.59% | **73.56%** | 69.73% | **74.73%** | **68.53%** | 0.3339 | 0.2721 |
| **Soft Ensemble** | 67.99% | 68.91% | **71.93%** | 72.95% | **75.29%** | **68.82%** | 0.3454 | 0.2690 |
