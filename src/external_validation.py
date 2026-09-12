import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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


def run_cross_dataset_validation(
    df_cleveland_harm: pd.DataFrame,
    df_framingham_harm: pd.DataFrame,
    target_col: str = "target",
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_trials: int = 15,
    random_state: int = 42
) -> dict:
    """
    Trains the full tuned ensemble framework on the FULL harmonized Cleveland dataset,
    and evaluates it directly on the unseen FULL harmonized Framingham dataset.

    Also runs baseline Cleveland-only nested cross-validation on the harmonized feature set
    for direct side-by-side performance comparison.

    Returns:
      dict containing:
        - 'cleveland_nested_cv': Nested CV performance on Cleveland (harmonized)
        - 'framingham_external': External validation performance on Framingham
        - 'comparison_table': Summary DataFrame comparing metrics
    """
    # 1. Cleveland-Only Nested CV on Harmonized Schema
    print("Running Cleveland-only nested CV on harmonized schema...")
    cleveland_nested_results = evaluate_nested_cv(
        df_cleveland_harm,
        target_col=target_col,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        n_trials=n_trials,
        random_state=random_state
    )

    # 2. Train Full Tuned Models on Full Cleveland Dataset
    print("Fitting preprocessor and tuning hyper-parameters on full Cleveland dataset...")
    X_clev_raw = df_cleveland_harm.drop(columns=[target_col])
    y_clev = df_cleveland_harm[target_col].values

    X_fram_raw = df_framingham_harm.drop(columns=[target_col])
    y_fram = df_framingham_harm[target_col].values

    preprocessor, feature_names = build_harmonized_preprocessor()
    X_clev_proc = preprocessor.fit_transform(X_clev_raw)
    X_fram_proc = preprocessor.transform(X_fram_raw)

    # Optuna tuning on Cleveland
    tuned_models, best_params, best_scores = optimize_all_models(
        X_clev_proc, y_clev, cv=inner_splits, n_trials=n_trials, random_state=random_state
    )

    # Fit tuned base classifiers on full Cleveland dataset
    for name, model in tuned_models.items():
        model.fit(X_clev_proc, y_clev)

    # Build and fit soft-voting ensemble
    ensemble_model = build_voting_ensemble(tuned_models, voting="soft")
    ensemble_model.fit(X_clev_proc, y_clev)
    tuned_models["ensemble"] = ensemble_model

    # 3. Evaluate Tuned Pipeline on Full Unseen Framingham Dataset
    print("Evaluating trained pipeline on unseen Framingham dataset...")
    framingham_metrics = {}

    for name, model in tuned_models.items():
        y_pred_fram = model.predict(X_fram_proc)
        y_prob_fram = model.predict_proba(X_fram_proc)[:, 1]

        metrics = calculate_metrics(y_fram, y_pred_fram, y_prob_fram)
        framingham_metrics[name] = metrics

    # 4. Construct Comparison Summary Table
    comparison_rows = []
    model_names = ["random_forest", "xgboost", "adaboost", "ensemble"]

    for m_name in model_names:
        clev_res = cleveland_nested_results[m_name]
        fram_res = framingham_metrics[m_name]

        comparison_rows.append({
            "Model": m_name.upper(),
            "Cleveland CV Accuracy": f"{clev_res['accuracy_mean']:.4f}",
            "Framingham Ext Accuracy": f"{fram_res['accuracy']:.4f}",
            "Cleveland CV F1": f"{clev_res['f1_mean']:.4f}",
            "Framingham Ext F1": f"{fram_res['f1']:.4f}",
            "Cleveland CV ROC-AUC": f"{clev_res['roc_auc_mean']:.4f}",
            "Framingham Ext ROC-AUC": f"{fram_res['roc_auc']:.4f}"
        })

    comparison_df = pd.DataFrame(comparison_rows)

    return {
        "cleveland_nested_cv": cleveland_nested_results,
        "framingham_external": framingham_metrics,
        "comparison_table": comparison_df,
        "best_params": best_params
    }
