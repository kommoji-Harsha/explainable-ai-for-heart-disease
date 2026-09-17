import pytest
from src.predict import predict_patient_risk, DISCLAIMER_TEXT


def test_app_prediction_wrapper():
    """
    Smoke test verifying that patient prediction logic reused in app.py
    executes cleanly and returns expected output dictionary structure.
    """
    patient_input = {
        "age": 58.0,
        "sex": 1.0,
        "cp": 4.0,
        "trestbps": 135.0,
        "chol": 240.0,
        "fbs": 0.0,
        "restecg": 0.0,
        "thalach": 145.0,
        "exang": 1.0,
        "oldpeak": 1.5,
        "slope": 2.0,
        "ca": 0.0,
        "thal": 7.0
    }

    res = predict_patient_risk(patient_input, models_dir="models")

    assert "predicted_probability" in res
    assert 0.0 <= res["predicted_probability"] <= 1.0
    assert "risk_label" in res
    assert res["risk_label"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert "agreement_flag" in res
    assert "base_model_probabilities" in res
    assert "top_contributing_features" in res
    assert res["disclaimer"] == DISCLAIMER_TEXT
