import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from src.preprocessing import build_preprocessor, get_feature_lists
from src.models import get_random_forest, get_xgboost, get_adaboost
from src.optimization import optimize_random_forest, optimize_xgboost, optimize_adaboost
from src.ensemble import build_voting_ensemble


def calculate_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Computes classification evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc) if auc is not None else None,
        "confusion_matrix": cm.tolist()
    }


def evaluate_nested_cv(df, target_col="target", outer_splits=5, inner_splits=3, n_trials=15, random_state=42):
    """
    Runs nested cross-validation on the dataset `df`.
    Preprocessing (imputation, scaling, encoding) is fitted inside each outer training fold
    to prevent data leakage. Hyperparameters are tuned in the inner CV loop.

    Evaluates:
      - Random Forest
      - XGBoost
      - AdaBoost
      - Soft-Voting Ensemble

    Returns:
      nested_results: dict containing fold-level and aggregated metrics for each model.
    """
    X_raw = df.drop(columns=[target_col])
    y = df[target_col].values

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    model_names = ["random_forest", "xgboost", "adaboost", "ensemble"]
    fold_metrics = {name: [] for name in model_names}

    outer_predictions = {name: {"y_true": [], "y_pred": [], "y_prob": []} for name in model_names}

    numeric_features, categorical_features = get_feature_lists()

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_raw, y)):
        X_train_raw, X_test_raw = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit preprocessor strictly on outer training data to prevent leakage
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        # Optimize hyperparameters on outer training data using inner CV
        rf_params, _, _ = optimize_random_forest(
            X_train, y_train, cv=inner_splits, n_trials=n_trials, random_state=random_state + fold
        )
        xgb_params, _, _ = optimize_xgboost(
            X_train, y_train, cv=inner_splits, n_trials=n_trials, random_state=random_state + fold
        )
        ada_params, _, _ = optimize_adaboost(
            X_train, y_train, cv=inner_splits, n_trials=n_trials, random_state=random_state + fold
        )

        rf_model = get_random_forest(random_state=random_state, **rf_params)
        xgb_model = get_xgboost(random_state=random_state, **xgb_params)
        ada_model = get_adaboost(random_state=random_state, **ada_params)

        rf_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)
        ada_model.fit(X_train, y_train)

        models = {
            "random_forest": rf_model,
            "xgboost": xgb_model,
            "adaboost": ada_model
        }

        ensemble = build_voting_ensemble(models, voting="soft")
        ensemble.fit(X_train, y_train)
        models["ensemble"] = ensemble

        # Evaluate models on test fold
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = calculate_metrics(y_test, y_pred, y_prob)
            fold_metrics[name].append(metrics)

            outer_predictions[name]["y_true"].extend(y_test.tolist())
            outer_predictions[name]["y_pred"].extend(y_pred.tolist())
            outer_predictions[name]["y_prob"].extend(y_prob.tolist())

    # Aggregate metrics across outer folds
    aggregated_results = {}
    for name in model_names:
        accs = [m["accuracy"] for m in fold_metrics[name]]
        precs = [m["precision"] for m in fold_metrics[name]]
        recs = [m["recall"] for m in fold_metrics[name]]
        f1s = [m["f1"] for m in fold_metrics[name]]
        aucs = [m["roc_auc"] for m in fold_metrics[name]]

        # Overall confusion matrix on all concatenated outer fold test predictions
        overall_cm = confusion_matrix(
            outer_predictions[name]["y_true"],
            outer_predictions[name]["y_pred"]
        ).tolist()

        aggregated_results[name] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "precision_mean": float(np.mean(precs)),
            "precision_std": float(np.std(precs)),
            "recall_mean": float(np.mean(recs)),
            "recall_std": float(np.std(recs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "roc_auc_mean": float(np.mean(aucs)),
            "roc_auc_std": float(np.std(aucs)),
            "overall_confusion_matrix": overall_cm,
            "fold_metrics": fold_metrics[name]
        }

    return aggregated_results
