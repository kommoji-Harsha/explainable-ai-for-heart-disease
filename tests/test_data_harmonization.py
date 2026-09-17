import pytest
import numpy as np
import pandas as pd
from src.data_harmonization import harmonize_datasets, COMMON_COLUMNS
from src.external_validation import build_harmonized_preprocessor, run_cross_dataset_validation, calculate_extended_metrics


@pytest.fixture
def sample_raw_datasets():
    N = 20
    np.random.seed(42)

    df_combined_raw = pd.DataFrame({
        "age": np.random.randint(30, 80, N),
        "sex": np.array([0.0, 1.0] * 10),
        "cp": np.random.choice([1.0, 2.0, 3.0, 4.0], N),
        "trestbps": np.random.randint(100, 180, N),
        "chol": np.random.randint(150, 350, N),
        "fbs": np.random.choice([0.0, 1.0], N),
        "restecg": np.random.choice([0.0, 1.0, 2.0], N),
        "thalach": np.random.randint(90, 200, N),
        "exang": np.random.choice([0.0, 1.0], N),
        "oldpeak": np.random.uniform(0.0, 4.0, N),
        "slope": np.random.choice([1.0, 2.0, 3.0], N),
        "ca": np.random.choice([0.0, 1.0, 2.0, 3.0], N),
        "thal": np.random.choice([3.0, 6.0, 7.0], N),
        "target": np.array([0, 1] * 10),
        "source_site": ["cleveland", "hungarian", "switzerland", "va_long_beach"] * 5
    })

    df_framingham_raw = pd.DataFrame({
        "male": np.array([0, 1] * 10),
        "age": np.random.randint(30, 80, N),
        "education": np.random.choice([1.0, 2.0, 3.0, 4.0], N),
        "currentSmoker": np.random.choice([0, 1], N),
        "cigsPerDay": np.random.randint(0, 40, N),
        "BPMeds": np.random.choice([0.0, 1.0], N),
        "prevalentStroke": np.random.choice([0, 1], N),
        "prevalentHyp": np.random.choice([0, 1], N),
        "diabetes": np.random.choice([0, 1], N),
        "totChol": np.random.randint(150, 350, N),
        "sysBP": np.random.randint(100, 180, N),
        "diaBP": np.random.randint(60, 110, N),
        "BMI": np.random.uniform(18.0, 40.0, N),
        "heartRate": np.random.randint(50, 110, N),
        "glucose": np.random.randint(70, 200, N),
        "TenYearCHD": np.array([0, 1] * 10)
    })

    return df_combined_raw, df_framingham_raw


def test_harmonize_datasets_schema_alignment(sample_raw_datasets):
    df_comb, df_fram = sample_raw_datasets
    c_harm, f_harm = harmonize_datasets(df_comb, df_fram)

    assert list(c_harm.columns) == COMMON_COLUMNS
    assert list(f_harm.columns) == COMMON_COLUMNS

    assert len(c_harm) == len(df_comb)
    assert len(f_harm) == len(df_fram)

    assert set(c_harm["target"].unique()).issubset({0, 1})
    assert set(f_harm["target"].unique()).issubset({0, 1})


def test_harmonized_preprocessor_shape(sample_raw_datasets):
    df_comb, df_fram = sample_raw_datasets
    c_harm, f_harm = harmonize_datasets(df_comb, df_fram)

    preprocessor, feature_names = build_harmonized_preprocessor()
    X_comb = c_harm.drop(columns=["target"])

    X_proc = preprocessor.fit_transform(X_comb)
    assert X_proc.shape == (20, 5)
    assert feature_names == ["age", "sex", "sysBP", "totChol", "diabetes"]


def test_calculate_extended_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8])

    m = calculate_extended_metrics(y_true, y_pred, y_prob)

    assert "specificity" in m
    assert "pr_auc" in m
    assert "is_degenerate" in m
    assert m["is_degenerate"] is False
    assert m["specificity"] == 1.0


def test_cross_dataset_validation_execution_3way(sample_raw_datasets):
    df_comb, df_fram = sample_raw_datasets
    c_harm, f_harm = harmonize_datasets(df_comb, df_fram)

    res = run_cross_dataset_validation(
        c_harm, f_harm, outer_splits=2, inner_splits=2, n_trials=1, random_state=42
    )

    assert "cleveland_nested_cv" in res
    assert "combined_nested_cv" in res
    assert "framingham_external" in res
    assert "comparison_table" in res

    comp_df = res["comparison_table"]
    assert len(comp_df) == 4
    assert "Cleveland CV Acc" in comp_df.columns
    assert "Combined 4-Site CV Acc" in comp_df.columns
    assert "Framingham Ext Acc" in comp_df.columns
