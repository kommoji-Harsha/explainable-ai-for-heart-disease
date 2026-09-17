import pandas as pd
import numpy as np
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.preprocessing import load_data
from src.models import get_random_forest, get_xgboost, get_adaboost
from src.optimization import optimize_all_models
from src.ensemble import build_voting_ensemble
from src.evaluation import calculate_metrics, evaluate_nested_cv


def build_harmonized_preprocessor():
    """
    Builds a preprocessor pipeline tailored for the 5 harmonized input features:
      ['age', 'sex', 'sysBP', 'totChol', 'diabetes']
    All 5 features are numeric/binary, so median imputation and standard scaling are applied.
    """
    feature_names = ["age", "sex", "sysBP", "totChol", "diabetes"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, feature_names)
    ])

    return preprocessor, feature_names


def calculate_extended_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Computes classification metrics including Accuracy, Precision, Recall, F1, Specificity,
    ROC-AUC, PR-AUC, and degenerate model flags (e.g. predicting all 0s or all 1s).
    """
    base_m = calculate_metrics(y_true, y_pred, y_prob)

    # Specificity (TNR = TN / (TN + FP))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan

    # PR-AUC
    if y_prob is not None and len(np.unique(y_true)) > 1:
        p_prec, p_rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(p_rec, p_prec))
    else:
        pr_auc = np.nan

    # Degenerate model check (constant prediction)
    is_degenerate = bool(len(np.unique(y_pred)) == 1)
    if is_degenerate:
        warnings.warn(f"Degenerate prediction detected: model predicted constant label {y_pred[0]} across all instances.")

    base_m.update({
        "specificity": spec,
        "pr_auc": pr_auc,
        "is_degenerate": is_degenerate
    })
    return base_m


def run_cross_dataset_validation(
    df_combined_harm: pd.DataFrame,
    df_framingham_harm: pd.DataFrame,
    target_col: str = "target",
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_trials: int = 15,
    random_state: int = 42
) -> dict:
    """
    Trains the full tuned ensemble framework on the FULL harmonized combined 4-site dataset (N=920),
    and evaluates it directly on the unseen FULL harmonized Framingham dataset (N=4,240).

    Also runs baseline Cleveland-only nested CV and Combined 4-site nested CV on the harmonized schema
    to generate a 3-way performance comparison table.

    Returns:
      dict containing:
        - 'cleveland_nested_cv': Cleveland-only CV performance (5 features)
        - 'combined_nested_cv': Combined 4-site CV performance (5 features)
        - 'framingham_external': External validation performance on Framingham
        - 'comparison_table': 3-way summary DataFrame comparing metrics
    """
    # 1. Cleveland-Only Nested CV on Harmonized Schema
    print("Running Cleveland-only nested CV on harmonized schema...")
    df_cleveland_raw = load_data("data/raw_cleveland.csv")
    clev_rename = {"trestbps": "sysBP", "chol": "totChol", "fbs": "diabetes"}
    df_cleveland_harm = df_cleveland_raw.rename(columns=clev_rename)[["age", "sex", "sysBP", "totChol", "diabetes", "target"]].copy()

    cleveland_nested_results = evaluate_nested_cv(
        df_cleveland_harm,
        target_col=target_col,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        n_trials=n_trials,
        random_state=random_state
    )

    # 2. Combined 4-Site Nested CV on Harmonized Schema
    print("Running Combined 4-site nested CV on harmonized schema...")
    combined_nested_results = evaluate_nested_cv(
        df_combined_harm,
        target_col=target_col,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        n_trials=n_trials,
        random_state=random_state
    )

    # 3. Train Full Tuned Models on Full Combined 4-Site Dataset
    print("Fitting preprocessor and tuning hyper-parameters on full combined 4-site dataset...")
    X_comb_raw = df_combined_harm.drop(columns=[target_col])
    y_comb = df_combined_harm[target_col].values

    X_fram_raw = df_framingham_harm.drop(columns=[target_col])
    y_fram = df_framingham_harm[target_col].values

    preprocessor, feature_names = build_harmonized_preprocessor()
    X_comb_proc = preprocessor.fit_transform(X_comb_raw)
    X_fram_proc = preprocessor.transform(X_fram_raw)

    # Optuna tuning on Combined dataset
    tuned_models, best_params, best_scores = optimize_all_models(
        X_comb_proc, y_comb, cv=inner_splits, n_trials=n_trials, random_state=random_state
    )

    # Fit tuned base classifiers on full Combined dataset
    for name, model in tuned_models.items():
        model.fit(X_comb_proc, y_comb)

    # Build and fit soft-voting ensemble
    ensemble_model = build_voting_ensemble(tuned_models, voting="soft")
    ensemble_model.fit(X_comb_proc, y_comb)
    tuned_models["ensemble"] = ensemble_model

    # 4. Evaluate Tuned Pipeline on Full Unseen Framingham Dataset
    print("Evaluating trained pipeline on unseen Framingham dataset...")
    framingham_metrics = {}

    for name, model in tuned_models.items():
        y_pred_fram = model.predict(X_fram_proc)
        y_prob_fram = model.predict_proba(X_fram_proc)[:, 1]

        metrics = calculate_extended_metrics(y_fram, y_pred_fram, y_prob_fram)
        framingham_metrics[name] = metrics

    # 5. Construct 3-Way Comparison Summary Table
    comparison_rows = []
    model_names = ["random_forest", "xgboost", "adaboost", "ensemble"]

    for m_name in model_names:
        clev_res = cleveland_nested_results[m_name]
        comb_res = combined_nested_results[m_name]
        fram_res = framingham_metrics[m_name]

        comparison_rows.append({
            "Model": m_name.upper(),
            "Cleveland CV Acc": f"{clev_res['accuracy_mean']:.4f}",
            "Combined 4-Site CV Acc": f"{comb_res['accuracy_mean']:.4f}",
            "Framingham Ext Acc": f"{fram_res['accuracy']:.4f}",
            "Cleveland CV ROC-AUC": f"{clev_res['roc_auc_mean']:.4f}",
            "Combined 4-Site CV ROC-AUC": f"{comb_res['roc_auc_mean']:.4f}",
            "Framingham Ext ROC-AUC": f"{fram_res['roc_auc']:.4f}",
            "Framingham Ext F1": f"{fram_res['f1']:.4f}",
            "Framingham Ext PR-AUC": f"{fram_res['pr_auc']:.4f}"
        })

    comparison_df = pd.DataFrame(comparison_rows)

    return {
        "cleveland_nested_cv": cleveland_nested_results,
        "combined_nested_cv": combined_nested_results,
        "framingham_external": framingham_metrics,
        "comparison_table": comparison_df,
        "best_params": best_params
    }
