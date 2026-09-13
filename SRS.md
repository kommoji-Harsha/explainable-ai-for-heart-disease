# Software Requirements Specification (SRS)
## Optimized, Explainable, Reliable Ensemble Framework for Heart Disease Prediction

### 1. Purpose and Scope
This document specifies the software requirements for an academic and clinical research machine learning framework designed for heart disease prediction. The framework integrates ensemble learning (Random Forest, XGBoost, AdaBoost), Bayesian hyperparameter optimization (Optuna), nested cross-validation, model probability calibration, subgroup fairness evaluation, per-patient uncertainty/disagreement flagging, and cross-dataset external validation.

---

### 2. Functional Requirements

#### 2.1 Month 1 — Baseline Pipeline
- **FR-1.1**: The system shall load the UCI Cleveland heart disease dataset (`processed.cleveland.data`), replacing missing indicator values (`?`) with standard null values (`NaN`).
- **FR-1.2**: The system shall convert target labels into a binary classification target (0: no disease, 1: heart disease present where original target > 0).
- **FR-1.3**: The system shall preprocess input data using a scikit-learn `ColumnTransformer` pipeline that applies median imputation and standard scaling to numerical features, and mode imputation with one-hot encoding to categorical features.
- **FR-1.4**: The system shall construct individual Random Forest, XGBoost, and AdaBoost classifiers.
- **FR-1.5**: The system shall perform Bayesian hyperparameter optimization using Optuna (TPE Sampler) on inner cross-validation folds for each base classifier.
- **FR-1.6**: The system shall combine the tuned base classifiers into a soft-voting ensemble (`VotingClassifier`).
- **FR-1.7**: The system shall evaluate model performance using leak-free 5-fold outer / 3-fold inner nested cross-validation, reporting Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
- **FR-1.8**: The system shall generate SHAP feature importance summary, bar, and individual prediction waterfall plots saved into `results/`.
- **FR-1.9**: The system shall provide an executable Jupyter notebook (`notebooks/01_baseline_pipeline.ipynb`) demonstrating the baseline pipeline end-to-end.

#### 2.2 Month 2 — Reliability Layer
- **FR-2.1**: The system shall compute probability calibration metrics, specifically Brier Score Loss and Expected Calibration Error (ECE), across out-of-fold predictions.
- **FR-2.2**: The system shall implement Platt Scaling (logistic sigmoid) and Isotonic Regression calibration methods.
- **FR-2.3**: To prevent in-sample probability overfitting, the system shall fit calibrators on out-of-sample predictions generated from a 75/25 model-fit/calibration-fit split within outer training folds, before refitting final models on full outer training folds.
- **FR-2.4**: The system shall generate calibration curve plots (uncalibrated vs. Platt vs. Isotonic) saved into `results/`.
- **FR-2.5**: The system shall compute subgroup fairness breakdowns across `sex` (Male/Female) and `age_band` (`< 50`, `50-60`, `> 60`), reporting sample sizes, Accuracy, Recall, and ROC-AUC.
- **FR-2.6**: The system shall include explicit methodological caveats highlighting that small subgroup sample sizes (N < 100) mean subgroup metrics are exploratory/indicative.
- **FR-2.7**: The system shall provide an executable Jupyter notebook (`notebooks/02_reliability_layer.ipynb`) demonstrating calibration and subgroup fairness analysis end-to-end.

#### 2.3 Per-Patient Model Agreement Scoring
- **FR-2.8**: The system shall compute a per-patient model agreement score defined as the standard deviation across individual base classifier probabilities (RF, XGBoost, AdaBoost).
- **FR-2.9**: The system shall categorize agreement scores into three discrete levels: `high agreement` (std < 0.05), `moderate agreement` (0.05 <= std < 0.15), and `low agreement` (std >= 0.15).
- **FR-2.10**: The system shall generate plain-language patient warning flags for prediction time (e.g., *"Model agreement: low — RF: 31%, XGBoost: 75%, AdaBoost: 58%. Interpret this prediction with extra caution."*).
- **FR-2.11**: The system shall summarize performance and accuracy disaggregated by agreement level across out-of-fold predictions.
- **FR-2.12**: The system shall provide an executable Jupyter notebook (`notebooks/03_model_agreement.ipynb`) demonstrating model agreement scoring and uncertainty analysis.

#### 2.4 Month 3 — Cross-Dataset External Validation (Added 2026-09-12)
- **FR-3.1**: The system shall load the Framingham Heart Study dataset (`framingham.csv`) alongside the UCI Cleveland dataset.
- **FR-3.2**: The system shall harmonize both datasets onto a unified common feature schema (`age`, `sex`, `sysBP`, `totChol`, `diabetes`, `target`), resolving differing column names, units, and label definitions.
- **FR-3.3**: The system shall document all features that cannot be aligned between Cleveland and Framingham datasets, along with any resulting data simplification or feature loss.
- **FR-3.4**: The system shall train the full tuned ensemble framework on the full harmonized Cleveland dataset and evaluate it directly on the unseen harmonized Framingham dataset.
- **FR-3.5**: The system shall report and compare external validation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix) against Cleveland-only nested CV baseline results to quantify cross-dataset performance transferability.
- **FR-3.6**: The system shall provide an executable Jupyter notebook (`notebooks/04_cross_dataset_validation.ipynb`) running cross-dataset harmonization and external validation end-to-end.

#### 2.5 Dataset Migration to Combined 4-Site Cohort (Added 2026-09-12)
- **FR-4.1**: The system shall load and concatenate all four UCI Heart Disease source datasets: Cleveland (N=303), Hungarian (N=294), Switzerland (N=123), and VA Long Beach (N=200), expanding the primary cohort to 920 patients total.
- **FR-4.2**: The system shall add and track a metadata column `source_site` (`cleveland`, `hungarian`, `switzerland`, `va_long_beach`) while ensuring `source_site` is excluded from model feature matrices.
- **FR-4.3**: The system shall generate and document a missingness percentage report table per feature column per site, saved to `results/combined_missingness_report.csv`.
- **FR-4.4**: The system shall document data quality limitations associated with high missingness in `ca` (66.4% combined) and `thal` (52.8% combined) across the 4 sites while maintaining full schema consistency via median/mode imputation.
- **FR-4.5**: The system shall compute subgroup fairness and performance breakdowns disaggregated by `source_site` in `src/fairness.py`, reporting sample sizes, Accuracy, Recall, and ROC-AUC per site.
- **FR-4.6**: The system shall provide executable Jupyter notebooks (`notebooks/01b_combined_dataset_baseline.ipynb` and `notebooks/02b_combined_reliability_and_agreement.ipynb`) demonstrating baseline, calibration, fairness, and model agreement pipelines on the combined cohort.

#### 2.6 Zero-as-Missing Data Quality Fix & Fairness Metric Expansion (Added 2026-09-12)
- **FR-5.1**: The system shall explicitly convert zero values (`0`) in physiologically implausible continuous attributes (`chol`, `trestbps`, `thalach`) to `NaN` specifically for non-Cleveland source datasets (Hungarian, Switzerland, VA Long Beach) in `src/preprocessing.py`, preventing unrecorded cholesterol/blood pressure measurements from contaminating median imputation and scaling statistics.
- **FR-5.2**: The system shall preserve legitimate zero values in Cleveland (and in legitimate zero-count/categorical fields like `ca` or `oldpeak`) without applying zero-as-missing conversion.
- **FR-5.3**: The system shall update `results/combined_missingness_report.csv` to reflect the corrected cholesterol missingness (increasing overall cholesterol missingness from 3.26% to 21.96%, specifically 100% missing in Switzerland and 28% in VA Long Beach).
- **FR-5.4**: The system shall expand per-subgroup and per-site fairness evaluation in `src/fairness.py` to calculate Precision, Specificity (True Negative Rate), and F1-score alongside Accuracy, Recall, and ROC-AUC, providing explicit caveats regarding positive-class prediction bias in high-prevalence cohorts (e.g., Switzerland at 92.7% disease prevalence).

#### 2.7 Cross-Dataset External Validation using Combined 4-Site Training Cohort (Added 2026-09-12)
- **FR-6.1**: The system shall update `src/data_harmonization.py` to accept the combined 920-patient 4-site UCI dataset as training-side input, applying zero-as-missing awareness for continuous variables (`sysBP`, `totChol`) before schema alignment.
- **FR-6.2**: The system shall update `src/external_validation.py` to train Optuna-tuned classifiers and the soft-voting ensemble on the full harmonized combined 4-site dataset (920 patients) and evaluate them directly on the unseen full Framingham cohort (4,240 patients).
- **FR-6.3**: The system shall report extended metrics (Precision, Recall, F1, Specificity, ROC-AUC, PR-AUC) and check for degenerate constant-label predictions during external validation.
- **FR-6.4**: The system shall generate a 3-way comparison table comparing: 1) Cleveland-only nested CV (5 features), 2) Combined 4-site nested CV (5 features), and 3) Framingham External Validation (5 features).
- **FR-6.5**: The system shall update `notebooks/04_cross_dataset_validation.ipynb` to execute the updated cross-dataset validation pipeline end-to-end and display 3-way comparison tables and confusion matrices.

---

### 3. Non-Functional Requirements
- **NFR-1 (Performance & Runtime)**: The pipeline execution shall run efficiently within standard free-tier CPU compute environments (e.g. Kaggle / Colab) in under 3 minutes.
- **NFR-2 (No-Cost Constraint)**: The system shall rely exclusively on open-source Python libraries (`scikit-learn`, `xgboost`, `optuna`, `shap`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `pytest`) without requiring paid APIs or commercial services.
- **NFR-3 (Reproducibility)**: The system shall enforce reproducible stochastic behavior by configuring fixed random seeds (`random_state=42`) across data splits, model initializations, and Optuna samplers.
- **NFR-4 (Modularity & Maintainability)**: Code shall be organized into modular, decoupled Python scripts under `src/`, with unit tests maintained under `tests/`.

---

### 4. Data Requirements
- **Combined 4-Site UCI Heart Disease Dataset** (Updated 2026-09-12):
  - Sources: UCI Machine Learning Repository (`processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data`, `processed.va.data`).
  - Total Size: 920 rows (Cleveland: 303, Hungarian: 294, Switzerland: 123, VA Long Beach: 200).
  - Attributes: 14 feature columns (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`) + 1 metadata column (`source_site`).
  - Missingness Limitations (Corrected): High missingness in `ca` (66.4% overall), `thal` (52.8% overall), and `chol` (21.96% overall after zero-as-missing fix).
  - Binary Target: 0 = no heart disease, 1 = heart disease present (`target > 0`).
- **Framingham Heart Study Dataset**:
  - Source: Framingham Heart Study public dataset (`framingham.csv`).
  - Size: 4,240 rows, 16 raw attributes.
- **Harmonized Dataset Schema**:
  - Features: `age` (years), `sex` (0=Female, 1=Male), `sysBP` (mmHg), `totChol` (mg/dL), `diabetes` (0/1 binary).
  - Target: `target` (0/1 binary).

---

### 5. Dependencies
- Pinned Python packages listed in [`requirements.txt`](./requirements.txt):
  - `numpy==2.5.3`
  - `pandas==3.0.5`
  - `scikit-learn==1.9.1`
  - `xgboost==3.4.1`
  - `optuna==5.0.0`
  - `shap==0.52.0`
  - `matplotlib==3.11.2`
  - `seaborn==0.13.2`
  - `pyyaml==6.0.3`
  - `pytest==9.1.1`
  - `jupyter==1.1.1`

---

### 6. Constraints and Assumptions
- **C-1**: Local storage is assumed for dataset caching (`data/raw_*.csv`).
- **C-2**: Datasets use binary classification target definitions.
- **C-3**: Subgroup sample sizes and per-site cohorts vary in size and disease prevalence; fairness and per-site metrics must be interpreted with Precision and Specificity to guard against prevalence bias.
