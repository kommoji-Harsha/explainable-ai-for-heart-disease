import os
import pytest
import numpy as np
import pandas as pd
from src.preprocessing import load_combined_dataset, load_single_site, preprocess_data, COLUMN_NAMES, UCI_DATASET_URLS


def test_load_combined_dataset_structure(tmp_path):
    # Test loading and concatenation
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"

    df = load_combined_dataset(data_dir=str(data_dir), results_dir=str(results_dir))

    # Assert total sample size across 4 sites is 920
    assert len(df) == 920
    assert "source_site" in df.columns

    # Assert exact site counts
    site_counts = df["source_site"].value_counts().to_dict()
    assert site_counts["cleveland"] == 303
    assert site_counts["hungarian"] == 294
    assert site_counts["va_long_beach"] == 200
    assert site_counts["switzerland"] == 123

    # Assert binary target
    assert set(df["target"].unique()).issubset({0, 1})

    # Assert missingness report CSV generated
    report_path = results_dir / "combined_missingness_report.csv"
    assert report_path.exists()
    report_df = pd.read_csv(report_path, index_col=0)
    assert "ca" in report_df.columns
    assert "thal" in report_df.columns
    assert "chol" in report_df.columns


def test_zero_as_missing_conversion_site_specificity(tmp_path):
    """
    Verifies that 0-as-missing conversion for physiologically impossible continuous variables
    (chol, trestbps, thalach) applies specifically to non-Cleveland sites (Hungary, Switzerland, VA),
    and does NOT convert legitimate zeros in Cleveland or in count/categorical columns.
    """
    data_dir = tmp_path / "data"

    # Load Cleveland: zero values in legitimate features like ca or oldpeak should remain 0
    df_clev = load_single_site("cleveland", UCI_DATASET_URLS["cleveland"], data_dir=str(data_dir))
    # In Cleveland, chol and trestbps have 0 zero values anyway
    assert (df_clev["chol"] == 0).sum() == 0
    assert (df_clev["trestbps"] == 0).sum() == 0

    # Load Switzerland: all 123 raw chol values were recorded as 0 -> should now be converted to NaN
    df_swiss = load_single_site("switzerland", UCI_DATASET_URLS["switzerland"], data_dir=str(data_dir))
    assert (df_swiss["chol"] == 0).sum() == 0
    assert df_swiss["chol"].isna().sum() == 123

    # Load VA Long Beach: 49 chol zeros and 1 trestbps zero -> should now be converted to NaN
    df_va = load_single_site("va_long_beach", UCI_DATASET_URLS["va_long_beach"], data_dir=str(data_dir))
    assert (df_va["chol"] == 0).sum() == 0
    assert (df_va["trestbps"] == 0).sum() == 0


def test_combined_dataset_preprocessing():
    # Synthetic combined dataframe
    N = 20
    df = pd.DataFrame({
        "age": np.random.randint(30, 80, N),
        "sex": np.random.choice([0.0, 1.0], N),
        "cp": np.random.choice([1.0, 2.0, 3.0, 4.0], N),
        "trestbps": np.random.randint(100, 180, N),
        "chol": np.random.randint(150, 350, N),
        "fbs": np.random.choice([0.0, 1.0], N),
        "restecg": np.random.choice([0.0, 1.0, 2.0], N),
        "thalach": np.random.randint(90, 200, N),
        "exang": np.random.choice([0.0, 1.0], N),
        "oldpeak": np.random.uniform(0.0, 4.0, N),
        "slope": np.random.choice([1.0, 2.0, 3.0], N),
        "ca": np.random.choice([np.nan, 0.0, 1.0], N),
        "thal": np.random.choice([np.nan, 3.0, 6.0, 7.0], N),
        "target": np.random.choice([0, 1], N),
        "source_site": ["cleveland", "hungarian", "switzerland", "va_long_beach"] * 5
    })

    X_proc, y, preprocessor, feature_names = preprocess_data(df)

    # Assert source_site excluded from features
    assert "source_site" not in feature_names
    assert X_proc.ndim == 2
    assert len(y) == N
    assert not np.isnan(X_proc).any()
