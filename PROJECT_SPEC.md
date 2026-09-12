# Project: Optimized, Explainable, Reliable Ensemble Framework for Heart Disease Prediction

## Reference paper
Mienye & Jere (2024), "Optimized Ensemble Learning Approach with Explainable AI
for Improved Heart Disease Prediction" — uses AdaBoost, Random Forest, XGBoost,
Bayesian hyperparameter optimization, and SHAP, evaluated on Cleveland and
Framingham datasets.

## What this project inherits from the reference
- Ensemble of AdaBoost + Random Forest + XGBoost
- Bayesian/Optuna hyperparameter optimization
- SHAP-based explainability
- Cleveland and Framingham datasets

## What this project ADDS beyond the reference (the actual contribution)
1. Nested cross-validation (prevent optimistic bias from tuning on same folds as evaluation)
2. Probability calibration (Brier score, calibration curve, ECE)
3. Subgroup/fairness analysis across sex, age bands, and other available demographics,
   reported with explicit caveats about small sample sizes
4. Cross-dataset external validation: train/tune on Cleveland, externally
   validate on Framingham after feature harmonization
5. LIME alongside SHAP, plus a handful of counterfactual "what-if" examples
6. Explanation-stability scoring across CV folds (e.g. Kendall's tau, Overlap@k)
7. Model disagreement / uncertainty flagging at prediction time

## Constraints
- No paid services or APIs. Free tools only (scikit-learn, xgboost, shap, lime,
  optuna, matplotlib/seaborn, pandas).
- Must run on free-tier compute (Google Colab / Kaggle Notebooks / local CPU).
- Code must be modular, tested, and documented — this is an academic project
  that will be reviewed and defended, not just a notebook.

## Repo structure
data/
notebooks/
src/
  preprocessing.py
  feature_engineering.py
  models.py
  optimization.py
  ensemble.py
  evaluation.py
  explainability.py
tests/
results/
README.md
requirements.txt
config.yaml

## Current phase: Month 1 — baseline
Goal: a working end-to-end pipeline on the Cleveland dataset only.
- Load and clean Cleveland dataset
- Preprocessing (missing values, encoding, scaling)
- Train RF, XGBoost, AdaBoost individually
- Bayesian/Optuna tuning for each
- Simple voting ensemble combining all three
- Nested cross-validation for honest evaluation
- SHAP explanations for a handful of sample predictions
- Evaluate with accuracy, precision, recall, F1, ROC-AUC, confusion matrix — not just accuracy

Later phases (calibration, fairness, cross-dataset validation, LIME/counterfactuals,
explanation stability, model disagreement) are NOT part of this phase — do not
implement them yet unless explicitly instructed.
