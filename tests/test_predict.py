import os
import pytest
import numpy as np
import pandas as pd

from src.train_final_model import train_and_save_final_models
from src.predict import (
    load_artifacts,
    predict_patient_risk,
    categorize_risk_level,
    DISCLAIMER_TEXT
)


@pytest.fixture(scope="module")
def models_directory(tmp_path_factory):
    """
    Creates a temporary directory and trains final models into it.
    """
    tmp_dir = str(tmp_path_factory.mktemp("models"))
    train_and_save_final_models(models_dir=tmp_dir)
    return tmp_dir


def test_models_loading(models_directory):
    """
    Verifies that all required model artifacts load correctly from joblib.
    """
    artifacts = load_artifacts(models_dir=models_directory)
    assert "preprocessor" in artifacts
    assert "random_forest" in artifacts
    assert "xgboost" in artifacts
    assert "adaboost" in artifacts
    assert "ensemble" in artifacts


def test_predict_patient_risk_format_and_range(models_directory):
    """
    Verifies that predict_patient_risk produces valid predictions and summary formats for a known input.
    """
    sample_patient = {
        "age": 60.0,
        "sex": 1.0,
        "cp": 4.0,
        "trestbps": 140.0,
        "chol": 260.0,
        "fbs": 0.0,
        "restecg": 1.0,
        "thalach": 130.0,
        "exang": 1.0,
        "oldpeak": 2.0,
        "slope": 2.0,
        "ca": 2.0,
        "thal": 7.0
    }

    result = predict_patient_risk(sample_patient, models_dir=models_directory)

    # Check structure
    assert "predicted_probability" in result
    assert "predicted_probability_percent" in result
    assert "risk_label" in result
    assert "agreement_flag" in result
    assert "base_model_probabilities" in result
    assert "top_contributing_features" in result
    assert "disclaimer" in result

    # Check value ranges
    prob = result["predicted_probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

    # Check risk label validity
    assert result["risk_label"] in ["Low Risk", "Moderate Risk", "High Risk"]

    # Check base model probabilities
    base_probs = result["base_model_probabilities"]
    for m_name in ["Random Forest", "XGBoost", "AdaBoost"]:
        assert m_name in base_probs
        assert 0.0 <= base_probs[m_name] <= 1.0

    # Check top features
    top_feats = result["top_contributing_features"]
    assert len(top_feats) > 0
    assert "feature" in top_feats[0]
    assert "shap_value" in top_feats[0]

    # Check disclaimer line
    assert result["disclaimer"] == DISCLAIMER_TEXT


def test_categorize_risk_level():
    assert categorize_risk_level(0.20) == "Low Risk"
    assert categorize_risk_level(0.50) == "Moderate Risk"
    assert categorize_risk_level(0.80) == "High Risk"
