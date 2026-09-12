import pytest
import numpy as np
import pandas as pd
from src.calibration import calculate_brier_score, calculate_ece, fit_platt_scaler, fit_isotonic_calibrator
from src.fairness import compute_fairness_breakdown, evaluate_subgroup_performance, create_age_bands


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
