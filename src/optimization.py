import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.models import get_random_forest, get_xgboost, get_adaboost

# Suppress Optuna verbose logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_random_forest(X, y, cv=3, n_trials=20, random_state=42):
    """
    Tune Random Forest hyperparameters using Optuna.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 200),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        }
        model = get_random_forest(random_state=random_state, **params)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value, study


def optimize_xgboost(X, y, cv=3, n_trials=20, random_state=42):
    """
    Tune XGBoost hyperparameters using Optuna.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 200),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0)
        }
        model = get_xgboost(random_state=random_state, **params)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value, study


def optimize_adaboost(X, y, cv=3, n_trials=20, random_state=42):
    """
    Tune AdaBoost hyperparameters using Optuna.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True)
        }
        model = get_adaboost(random_state=random_state, **params)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value, study


def optimize_all_models(X, y, cv=3, n_trials=20, random_state=42):
    """
    Optimizes Random Forest, XGBoost, and AdaBoost models.
    Returns a dict with best_params, best_models, and best_scores.
    """
    rf_params, rf_score, rf_study = optimize_random_forest(X, y, cv=cv, n_trials=n_trials, random_state=random_state)
    xgb_params, xgb_score, xgb_study = optimize_xgboost(X, y, cv=cv, n_trials=n_trials, random_state=random_state)
    ada_params, ada_score, ada_study = optimize_adaboost(X, y, cv=cv, n_trials=n_trials, random_state=random_state)

    tuned_models = {
        "random_forest": get_random_forest(random_state=random_state, **rf_params),
        "xgboost": get_xgboost(random_state=random_state, **xgb_params),
        "adaboost": get_adaboost(random_state=random_state, **ada_params)
    }

    best_params = {
        "random_forest": rf_params,
        "xgboost": xgb_params,
        "adaboost": ada_params
    }

    best_scores = {
        "random_forest": rf_score,
        "xgboost": xgb_score,
        "adaboost": ada_score
    }

    return tuned_models, best_params, best_scores
