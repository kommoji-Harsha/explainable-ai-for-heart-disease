import os
import pytest
import numpy as np
import pandas as pd
from src.preprocessing import load_data, preprocess_data, build_preprocessor, get_feature_lists
from src.evaluation import calculate_metrics, evaluate_nested_cv


@pytest.fixture
def sample_raw_data():
    """
    Creates a small synthetic dataset resembling the raw UCI Cleveland dataset structure.
    Target values match raw dataset (0, 1, 2, 3, 4).
    """
    data = {
        "age": [63.0, 67.0, 67.0, 37.0, 41.0, 56.0, 62.0, 57.0, 63.0, 53.0],
        "sex": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        "cp": [1.0, 4.0, 4.0, 3.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0],
        "trestbps": [145.0, 160.0, 120.0, 130.0, 130.0, 120.0, 140.0, 120.0, 130.0, 140.0],
        "chol": [233.0, 286.0, 229.0, 250.0, 204.0, 236.0, 268.0, 354.0, 254.0, 203.0],
        "fbs": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "restecg": [2.0, 2.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 2.0],
        "thalach": [150.0, 108.0, 129.0, 187.0, 172.0, 178.0, 160.0, 163.0, 147.0, 155.0],
        "exang": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        "oldpeak": [2.3, 1.5, 2.6, 3.5, 1.4, 0.8, 3.6, 0.6, 1.4, 3.1],
        "slope": [3.0, 2.0, 2.0, 3.0, 1.0, 1.0, 3.0, 1.0, 2.0, 3.0],
        "ca": [0.0, 3.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0],
        "thal": [6.0, 3.0, 7.0, 3.0, 3.0, 3.0, 3.0, 3.0, 7.0, 7.0],
        "target": [0, 2, 1, 0, 0, 0, 3, 0, 2, 1]
    }
    df = pd.DataFrame(data)
    # Target binarization as performed in load_data
    df["target"] = (df["target"] > 0).astype(int)
    return df


def test_load_data_and_target_binarization(tmp_path):
    csv_file = tmp_path / "test_cleveland.csv"
    data = {
        "age": [63.0, 67.0, 37.0],
        "sex": [1.0, 1.0, 1.0],
        "cp": [1.0, 4.0, 3.0],
        "trestbps": [145.0, 160.0, 130.0],
        "chol": [233.0, 286.0, 250.0],
        "fbs": [1.0, 0.0, 0.0],
        "restecg": [2.0, 2.0, 0.0],
        "thalach": [150.0, 108.0, 187.0],
        "exang": [0.0, 1.0, 0.0],
        "oldpeak": [2.3, 1.5, 3.5],
        "slope": [3.0, 2.0, 3.0],
        "ca": ["0.0", "?", "0.0"],
        "thal": ["6.0", "3.0", "?"],
        "target": [0, 2, 1]
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    df = load_data(str(csv_file))

    # Assert binary target conversion (0 remains 0, >0 becomes 1)
    assert set(df["target"].unique()) == {0, 1}
    assert df["target"].tolist() == [0, 1, 1]

    # Assert '?' was converted to NaN / numeric float
    assert pd.isna(df.loc[1, "ca"])
    assert pd.isna(df.loc[2, "thal"])


def test_preprocessing_pipeline(sample_raw_data):
    X_processed, y, preprocessor, feature_names = preprocess_data(sample_raw_data)

    assert X_processed.ndim == 2
    assert len(y) == len(sample_raw_data)
    assert set(y) == {0, 1}
    assert not np.isnan(X_processed).any()
    assert len(feature_names) == X_processed.shape[1]


def test_calculate_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8])

    metrics = calculate_metrics(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics

    assert metrics["accuracy"] == 0.8
    assert metrics["roc_auc"] > 0.5
    assert len(metrics["confusion_matrix"]) == 2


def test_evaluate_nested_cv(sample_raw_data):
    # Test nested CV on synthetic dataset with minimal outer/inner splits and trials
    results = evaluate_nested_cv(
        sample_raw_data,
        target_col="target",
        outer_splits=2,
        inner_splits=2,
        n_trials=2,
        random_state=42
    )

    for model_name in ["random_forest", "xgboost", "adaboost", "ensemble"]:
        assert model_name in results
        m = results[model_name]
        assert 0.0 <= m["accuracy_mean"] <= 1.0
        assert 0.0 <= m["f1_mean"] <= 1.0
        assert 0.0 <= m["roc_auc_mean"] <= 1.0
        assert len(m["overall_confusion_matrix"]) == 2
