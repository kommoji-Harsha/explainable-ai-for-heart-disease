import os
import json
import datetime
import joblib
import yaml
import numpy as np
import pandas as pd

from src.preprocessing import load_combined_dataset, preprocess_data, COLUMN_NAMES
from src.optimization import optimize_all_models
from src.ensemble import build_voting_ensemble


def load_config(config_path: str = "config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {
        "random_state": 42,
        "cv": {"inner_folds": 3, "optuna_n_trials": 20},
        "results_dir": "results"
    }


def train_and_save_final_models(models_dir: str = "models", config_path: str = "config.yaml"):
    """
    Fits the preprocessor and three base classifiers (RF, XGBoost, AdaBoost) on the FULL
    combined 920-patient dataset using Optuna-found best hyperparameters.
    Builds soft-voting ensemble, saves models/preprocessor via joblib, and writes metadata.json.
    """
    config = load_config(config_path)
    random_state = config.get("random_state", 42)
    inner_folds = config.get("cv", {}).get("inner_folds", 3)
    n_trials = config.get("cv", {}).get("optuna_n_trials", 20)

    os.makedirs(models_dir, exist_ok=True)

    print("Loading full combined 920-patient UCI dataset...")
    df = load_combined_dataset()
    dataset_size = len(df)

    print("Fitting preprocessing pipeline on full dataset...")
    X_processed, y, preprocessor, feature_names = preprocess_data(df)

    print(f"Finding / loading best hyperparameters using Optuna (n_trials={n_trials}, cv={inner_folds})...")
    tuned_models, best_params, best_scores = optimize_all_models(
        X_processed, y, cv=inner_folds, n_trials=n_trials, random_state=random_state
    )

    print("Fitting individual base classifiers on full preprocessed dataset...")
    for model_name, model in tuned_models.items():
        print(f"  Fitting {model_name}...")
        model.fit(X_processed, y)

    print("Building and fitting soft-voting ensemble...")
    ensemble = build_voting_ensemble(tuned_models, voting="soft")
    ensemble.fit(X_processed, y)

    # Save preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessor to {preprocessor_path}")

    # Save individual models
    model_paths = {}
    for name, model in tuned_models.items():
        p = os.path.join(models_dir, f"{name}.joblib")
        joblib.dump(model, p)
        model_paths[name] = p
        print(f"Saved {name} model to {p}")

    # Save ensemble model
    ensemble_path = os.path.join(models_dir, "ensemble.joblib")
    joblib.dump(ensemble, ensemble_path)
    print(f"Saved ensemble model to {ensemble_path}")

    # Feature schema details
    raw_feature_list = [c for c in COLUMN_NAMES if c != "target"]
    
    # Save metadata
    metadata = {
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_name": "Combined 4-Site UCI Heart Disease Dataset",
        "dataset_size": dataset_size,
        "processed_feature_count": len(feature_names),
        "raw_features": raw_feature_list,
        "processed_features": feature_names,
        "random_state": random_state,
        "best_hyperparameters": best_params,
        "optuna_validation_roc_auc_scores": best_scores,
        "saved_artifacts": {
            "preprocessor": preprocessor_path,
            "random_forest": model_paths["random_forest"],
            "xgboost": model_paths["xgboost"],
            "adaboost": model_paths["adaboost"],
            "ensemble": ensemble_path
        }
    }

    metadata_path = os.path.join(models_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print("Final model training and saving complete successfully.")
    return preprocessor, tuned_models, ensemble, metadata


if __name__ == "__main__":
    train_and_save_final_models()
