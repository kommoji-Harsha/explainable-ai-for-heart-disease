import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.calibration import calculate_brier_score, calculate_ece, fit_platt_scaler, fit_isotonic_calibrator
from src.fairness import compute_fairness_breakdown, evaluate_subgroup_performance, create_age_bands
from src.evaluation import evaluate_nested_cv


def test_calculate_brier_score_known_values():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2])
    # MSE = ( (0.1-0)^2 + (0.9-1)^2 + (0.8-1)^2 + (0.2-0)^2 ) / 4
    #     = ( 0.01 + 0.01 + 0.04 + 0.04 ) / 4 = 0.10 / 4 = 0.025
    brier = calculate_brier_score(y_true, y_prob)
    assert pytest.approx(brier, 1e-5) == 0.025


def test_calculate_ece_known_values():
    # Case 1: Perfect confidence match (ECE = 0.0)
    y_true_perfect = np.array([1, 1, 0, 0])
    y_prob_perfect = np.array([1.0, 1.0, 0.0, 0.0])
    ece_perfect = calculate_ece(y_true_perfect, y_prob_perfect, n_bins=5)
    assert pytest.approx(ece_perfect, 1e-5) == 0.0

    # Case 2: Known confidence mismatch
    # Bin [0.0, 0.2): avg_acc=0.0, avg_conf=0.1 -> diff=0.1, weight=0.5
    # Bin [0.8, 1.0]: avg_acc=1.0, avg_conf=0.9 -> diff=0.1, weight=0.5
    # Expected ECE = 0.5*0.1 + 0.5*0.1 = 0.1
    y_prob_mismatch = np.array([0.9, 0.9, 0.1, 0.1])
    ece_mismatch = calculate_ece(y_true_perfect, y_prob_mismatch, n_bins=5)
    assert pytest.approx(ece_mismatch, 1e-5) == 0.10


def test_platt_and_isotonic_calibrators():
    y_train = np.array([0, 0, 0, 1, 1, 1])
    y_prob_train = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])

    platt = fit_platt_scaler(y_prob_train, y_train)
    iso = fit_isotonic_calibrator(y_prob_train, y_train)

    test_probs = np.array([0.25, 0.75])
    cal_platt = platt.predict_proba(test_probs.reshape(-1, 1))[:, 1]
    cal_iso = iso.predict(test_probs)

    assert len(cal_platt) == 2
    assert len(cal_iso) == 2
    assert (cal_platt >= 0.0).all() and (cal_platt <= 1.0).all()
    assert (cal_iso >= 0.0).all() and (cal_iso <= 1.0).all()


def test_fairness_subgroup_sizes_sum_to_total():
    N = 20
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(30, 80, size=N),
        "sex": np.random.choice([0.0, 1.0], size=N),
        "target": np.random.choice([0, 1], size=N)
    })
    y_true = df["target"].values
    y_pred = np.random.choice([0, 1], size=N)
    y_prob = np.random.uniform(0.0, 1.0, size=N)

    fairness_res = compute_fairness_breakdown(df, y_true, y_pred, y_prob)

    sex_df = fairness_res["sex_breakdown"]
    age_df = fairness_res["age_breakdown"]

    assert sex_df["sample_size"].sum() == N
    assert age_df["sample_size"].sum() == N
    assert "caveat" in fairness_res
    assert "small" in fairness_res["caveat"].lower()


def test_create_age_bands():
    df = pd.DataFrame({"age": [45, 50, 60, 65]})
    bands = create_age_bands(df)
    assert bands.tolist() == ["< 50", "50-60", "50-60", "> 60"]


def test_calibrator_fit_uses_out_of_sample_predictions(monkeypatch):
    """
    Verifies that calibrator fitting function fit_platt_scaler is never called with
    predictions made on the exact same dataset used to fit the underlying classifier (in-sample predictions).
    """
    from src import evaluation

    fit_call_args = []
    original_fit_platt = evaluation.fit_platt_scaler

    def mock_fit_platt(y_prob, y_true):
        # Record tuple of length of predictions passed into fit_platt_scaler
        fit_call_args.append((len(y_prob), len(y_true)))
        return original_fit_platt(y_prob, y_true)

    monkeypatch.setattr(evaluation, "fit_platt_scaler", mock_fit_platt)

    # Synthetic dataset of 20 samples
    N = 20
    df = pd.DataFrame({
        "age": np.linspace(30, 70, N),
        "sex": [0.0, 1.0] * (N // 2),
        "cp": [1.0, 2.0, 3.0, 4.0] * (N // 4),
        "trestbps": [120.0] * N,
        "chol": [200.0] * N,
        "fbs": [0.0] * N,
        "restecg": [0.0] * N,
        "thalach": [150.0] * N,
        "exang": [0.0] * N,
        "oldpeak": [1.0] * N,
        "slope": [1.0] * N,
        "ca": [0.0] * N,
        "thal": [3.0] * N,
        "target": [0, 1] * (N // 2)
    })

    # Execute 2-fold outer CV
    # Outer train size = 10 samples.
    # Model-fit subset = 7 samples (75%).
    # Calibration-fit subset = 3 samples (25%).
    evaluation.evaluate_nested_cv(df, outer_splits=2, inner_splits=2, n_trials=1, random_state=42)

    assert len(fit_call_args) > 0
    for prob_len, true_len in fit_call_args:
        # In-sample size would be 10 (the full outer train fold).
        # Out-of-sample calibration-fit size should be 3 (25% of 10 for test split), NOT 10!
        assert prob_len < 10, f"Calibrator was fit on full in-sample train set of size {prob_len}!"
        assert prob_len == 3 or prob_len == 2
