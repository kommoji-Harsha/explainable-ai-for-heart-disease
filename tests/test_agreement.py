import pytest
import numpy as np
import pandas as pd
from src.agreement import (
    calculate_agreement_score,
    categorize_agreement,
    generate_agreement_flag,
    analyze_patient_agreements,
    summarize_agreement_performance
)


def test_calculate_agreement_score_identical_probabilities():
    # Identical probabilities should have std_dev == 0.0
    rf_p, xgb_p, ada_p = 0.85, 0.85, 0.85
    std = calculate_agreement_score(rf_p, xgb_p, ada_p)
    assert std == 0.0
    assert categorize_agreement(std) == "high agreement"


def test_calculate_agreement_score_divergent_probabilities():
    # High disagreement (e.g. 0.10, 0.90, 0.50)
    rf_p, xgb_p, ada_p = 0.10, 0.90, 0.50
    std = calculate_agreement_score(rf_p, xgb_p, ada_p)
    # Mean = 0.50. Variances: (0.1-0.5)^2=0.16, (0.9-0.5)^2=0.16, (0.5-0.5)^2=0
    # Mean var = 0.32 / 3 = 0.10666... std = sqrt(0.10666...) ~ 0.3266
    assert std > 0.15
    assert categorize_agreement(std) == "low agreement"


def test_generate_agreement_flag_formatting():
    rf_p, xgb_p, ada_p = 0.314, 0.751, 0.582
    flag = generate_agreement_flag(rf_p, xgb_p, ada_p)

    assert "Model agreement:" in flag
    assert "RF: 31%" in flag
    assert "XGBoost: 75%" in flag
    assert "AdaBoost: 58%" in flag
    assert "Interpret this prediction with extra caution." in flag


def test_analyze_patient_agreements_summary():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])

    # Patient 0: high agreement (all 0.10)
    # Patient 1: high agreement (all 0.90)
    # Patient 2: moderate agreement (0.70, 0.80, 0.85) -> std = 0.0623
    # Patient 3: low agreement (0.10, 0.80, 0.50) -> std = 0.2867
    rf_probs = np.array([0.10, 0.90, 0.70, 0.10])
    xgb_probs = np.array([0.10, 0.90, 0.80, 0.80])
    ada_probs = np.array([0.10, 0.90, 0.85, 0.50])

    df_patients = analyze_patient_agreements(
        y_true=y_true,
        y_pred_ensemble=y_pred,
        rf_probs=rf_probs,
        xgb_probs=xgb_probs,
        ada_probs=ada_probs
    )

    assert len(df_patients) == 4
    assert df_patients.loc[0, "agreement_level"] == "high agreement"
    assert df_patients.loc[1, "agreement_level"] == "high agreement"
    assert df_patients.loc[2, "agreement_level"] == "moderate agreement"
    assert df_patients.loc[3, "agreement_level"] == "low agreement"

    summary = summarize_agreement_performance(df_patients)
    assert len(summary) == 3
    assert set(summary["Agreement Level"].unique()) == {"high agreement", "moderate agreement", "low agreement"}
    assert summary.loc[summary["Agreement Level"] == "high agreement", "Sample Size (N)"].values[0] == 2
