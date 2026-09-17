import os
import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd
import shap

from src.agreement import generate_agreement_flag, calculate_agreement_score, categorize_agreement
from src.preprocessing import COLUMN_NAMES, get_feature_lists


DEFAULT_MODELS_DIR = "models"
DISCLAIMER_TEXT = "This is a machine learning estimate, not a medical diagnosis."


def load_artifacts(models_dir: str = DEFAULT_MODELS_DIR):
    """
    Loads saved preprocessor, base models, and soft-voting ensemble from models_dir.
    """
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    rf_path = os.path.join(models_dir, "random_forest.joblib")
    xgb_path = os.path.join(models_dir, "xgboost.joblib")
    ada_path = os.path.join(models_dir, "adaboost.joblib")
    ensemble_path = os.path.join(models_dir, "ensemble.joblib")

    if not all(os.path.exists(p) for p in [preprocessor_path, rf_path, xgb_path, ada_path, ensemble_path]):
        raise FileNotFoundError(
            f"Model artifacts not found in directory '{models_dir}'. "
            "Please run 'python3 src/train_final_model.py' first."
        )

    preprocessor = joblib.load(preprocessor_path)
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)
    ada_model = joblib.load(ada_path)
    ensemble = joblib.load(ensemble_path)

    return {
        "preprocessor": preprocessor,
        "random_forest": rf_model,
        "xgboost": xgb_model,
        "adaboost": ada_model,
        "ensemble": ensemble
    }


def categorize_risk_level(probability: float) -> str:
    """
    Categorizes predicted probability into risk levels:
      - Low Risk: prob < 0.35
      - Moderate Risk: 0.35 <= prob < 0.65
      - High Risk: prob >= 0.65
    """
    if probability < 0.35:
        return "Low Risk"
    elif probability < 0.65:
        return "Moderate Risk"
    else:
        return "High Risk"


def generate_shap_top_features(ensemble, X_processed: np.ndarray, feature_names: list, top_k: int = 3) -> list:
    """
    Generates SHAP feature impact explanations for a single patient sample using ensemble or XGBoost base estimator.
    Returns list of dicts with feature name, patient feature value (if available), SHAP value, and impact direction.
    """
    # Use ensemble's xgboost estimator or ensemble itself for SHAP
    model_to_explain = None
    if hasattr(ensemble, "named_estimators_") and "xgboost" in ensemble.named_estimators_:
        model_to_explain = ensemble.named_estimators_["xgboost"]
    else:
        model_to_explain = ensemble

    try:
        explainer = shap.TreeExplainer(model_to_explain)
        shap_vals = explainer(X_processed)
    except Exception:
        try:
            explainer = shap.Explainer(model_to_explain, X_processed)
            shap_vals = explainer(X_processed)
        except Exception:
            return []

    if len(shap_vals.shape) == 3 and shap_vals.shape[2] == 2:
        vals = shap_vals[0, :, 1].values
    elif hasattr(shap_vals[0], "values"):
        vals = shap_vals[0].values
    else:
        vals = np.array(shap_vals[0])

    # Rank features by absolute SHAP impact
    top_indices = np.argsort(np.abs(vals))[::-1][:top_k]

    explanations = []
    for idx in top_indices:
        f_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        shap_v = float(vals[idx])
        direction = "increases risk" if shap_v > 0 else "decreases risk"
        explanations.append({
            "feature": f_name,
            "shap_value": round(shap_v, 4),
            "impact": direction,
            "summary": f"{f_name} ({direction}, SHAP impact: {shap_v:+.3f})"
        })

    return explanations


def predict_patient_risk(patient_dict: dict, models_dir: str = DEFAULT_MODELS_DIR) -> dict:
    """
    Accepts raw patient feature dictionary, runs through preprocessor and models,
    returns prediction dictionary with probability, risk level, agreement flag, top features, and disclaimer.
    """
    artifacts = load_artifacts(models_dir=models_dir)
    preprocessor = artifacts["preprocessor"]
    rf_model = artifacts["random_forest"]
    xgb_model = artifacts["xgboost"]
    ada_model = artifacts["adaboost"]
    ensemble = artifacts["ensemble"]

    # Convert patient_dict to DataFrame
    raw_feature_cols = [c for c in COLUMN_NAMES if c != "target"]
    df_single = pd.DataFrame([patient_dict])

    # Ensure all required features exist in df_single (fill missing with NaN if omitted)
    for col in raw_feature_cols:
        if col not in df_single.columns:
            df_single[col] = np.nan

    # Order columns
    df_single = df_single[raw_feature_cols]

    # Preprocess
    X_processed = preprocessor.transform(df_single)

    # Feature names
    num_feats, cat_feats = get_feature_lists(df_single)
    if "cat" in preprocessor.named_transformers_:
        cat_transformer = preprocessor.named_transformers_["cat"]
        onehot = cat_transformer.named_steps["onehot"]
        cat_feature_names = list(onehot.get_feature_names_out(cat_feats))
        feature_names = list(num_feats) + cat_feature_names
    else:
        feature_names = list(raw_feature_cols)

    # Get predictions
    rf_prob = float(rf_model.predict_proba(X_processed)[0, 1])
    xgb_prob = float(xgb_model.predict_proba(X_processed)[0, 1])
    ada_prob = float(ada_model.predict_proba(X_processed)[0, 1])
    ensemble_prob = float(ensemble.predict_proba(X_processed)[0, 1])

    risk_label = categorize_risk_level(ensemble_prob)
    agreement_flag = generate_agreement_flag(rf_prob, xgb_prob, ada_prob)

    top_explanations = generate_shap_top_features(ensemble, X_processed, feature_names, top_k=3)

    result = {
        "predicted_probability": round(ensemble_prob, 4),
        "predicted_probability_percent": f"{ensemble_prob * 100:.1f}%",
        "risk_label": risk_label,
        "agreement_flag": agreement_flag,
        "base_model_probabilities": {
            "Random Forest": round(rf_prob, 4),
            "XGBoost": round(xgb_prob, 4),
            "AdaBoost": round(ada_prob, 4)
        },
        "top_contributing_features": top_explanations,
        "disclaimer": DISCLAIMER_TEXT
    }

    return result


def format_summary_output(patient_dict: dict, result: dict) -> str:
    """
    Formats the prediction dictionary into a clean, human-readable summary.
    """
    lines = []
    lines.append("==================================================")
    lines.append("     HEART DISEASE RISK ASSESSMENT SUMMARY       ")
    lines.append("==================================================")
    lines.append(f"Predicted Probability : {result['predicted_probability_percent']} ({result['predicted_probability']:.4f})")
    lines.append(f"Risk Category         : {result['risk_label'].upper()}")
    lines.append("--------------------------------------------------")
    lines.append("MODEL CONSENSUS & UNCERTAINTY:")
    lines.append(f"  {result['agreement_flag']}")
    lines.append("  Base Classifiers:")
    for m_name, prob in result["base_model_probabilities"].items():
        lines.append(f"    - {m_name:15s}: {prob * 100:.1f}%")
    lines.append("--------------------------------------------------")
    lines.append("TOP CONTRIBUTING FEATURES (SHAP Explanation):")
    if result["top_contributing_features"]:
        for i, item in enumerate(result["top_contributing_features"], 1):
            lines.append(f"  {i}. {item['summary']}")
    else:
        lines.append("  N/A")
    lines.append("--------------------------------------------------")
    lines.append(result["disclaimer"])
    lines.append("==================================================")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict heart disease risk for a patient.")
    parser.add_argument("--age", type=float, default=58.0, help="Age in years")
    parser.add_argument("--sex", type=float, default=1.0, help="Sex (1=male, 0=female)")
    parser.add_argument("--cp", type=float, default=4.0, help="Chest pain type (1-4)")
    parser.add_argument("--trestbps", type=float, default=140.0, help="Resting blood pressure (mmHg)")
    parser.add_argument("--chol", type=float, default=240.0, help="Serum cholesterol (mg/dL)")
    parser.add_argument("--fbs", type=float, default=0.0, help="Fasting blood sugar > 120 mg/dL (1=true, 0=false)")
    parser.add_argument("--restecg", type=float, default=0.0, help="Resting ECG results (0-2)")
    parser.add_argument("--thalach", type=float, default=140.0, help="Maximum heart rate achieved")
    parser.add_argument("--exang", type=float, default=1.0, help="Exercise induced angina (1=yes, 0=no)")
    parser.add_argument("--oldpeak", type=float, default=1.8, help="ST depression induced by exercise")
    parser.add_argument("--slope", type=float, default=2.0, help="Slope of peak exercise ST segment")
    parser.add_argument("--ca", type=float, default=1.0, help="Number of major vessels (0-3) colored by flourosopy")
    parser.add_argument("--thal", type=float, default=7.0, help="Thalassemia (3=normal, 6=fixed defect, 7=reversable defect)")
    parser.add_argument("--models-dir", type=str, default=DEFAULT_MODELS_DIR, help="Directory with saved joblib models")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of human-readable summary")
    return parser.parse_args()


def main():
    args = parse_args()
    patient_input = {
        "age": args.age,
        "sex": args.sex,
        "cp": args.cp,
        "trestbps": args.trestbps,
        "chol": args.chol,
        "fbs": args.fbs,
        "restecg": args.restecg,
        "thalach": args.thalach,
        "exang": args.exang,
        "oldpeak": args.oldpeak,
        "slope": args.slope,
        "ca": args.ca,
        "thal": args.thal
    }

    result = predict_patient_risk(patient_input, models_dir=args.models_dir)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        summary = format_summary_output(patient_input, result)
        print(summary)


if __name__ == "__main__":
    main()
