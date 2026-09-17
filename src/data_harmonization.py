import os
import urllib.request
import pandas as pd
import numpy as np


FRAMINGHAM_URL = "https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv"

# Standard Common Feature Schema for Cross-Dataset Alignment
COMMON_COLUMNS = ["age", "sex", "sysBP", "totChol", "diabetes", "target"]


def load_framingham_raw(data_path: str = "data/raw_framingham.csv", url: str = FRAMINGHAM_URL) -> pd.DataFrame:
    """
    Loads raw Framingham dataset from local path or downloads from URL.
    """
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        urllib.request.urlretrieve(url, data_path)
        df = pd.read_csv(data_path)

    return df


def harmonize_datasets(
    df_combined: pd.DataFrame,
    df_framingham: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Harmonizes the combined 4-site UCI dataset (N=920) and Framingham dataset (N=4,240) onto a unified schema:
      - age: Patient age in years (continuous)
      - sex: Binary sex encoding (0 = Female, 1 = Male)
      - sysBP: Systolic blood pressure in mmHg (continuous)
               (mapped from UCI 'trestbps' and Framingham 'sysBP')
      - totChol: Total cholesterol in mg/dL (continuous)
                 (mapped from UCI 'chol' and Framingham 'totChol')
      - diabetes: Fasting blood sugar / diabetes indicator (binary: 0 or 1)
                  (mapped from UCI 'fbs' where fbs > 120 mg/dl -> 1, and Framingham 'diabetes')
      - target: Heart disease presence (binary: 0 = No Disease, 1 = Disease Present)
                (mapped from UCI 'target > 0' and Framingham 'TenYearCHD')

    ZERO-AS-MISSING CONVERSION AWARENESS:
    Applies the zero-as-missing fix to continuous variables ('totChol', 'sysBP') for non-Cleveland UCI sites.

    Documentation of Feature Loss & Simplification:
      1. Features dropped from combined UCI dataset (missing in Framingham):
         - 'cp', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
      2. Features dropped from Framingham (missing in UCI dataset):
         - 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diaBP', 'BMI', 'heartRate', 'glucose'

    Returns:
      (df_combined_harmonized, df_framingham_harmonized)
    """
    # 1. Harmonize Combined UCI Dataset
    df_c = df_combined.copy()

    # Ensure binary target
    if "target" in df_c.columns:
        df_c["target"] = (df_c["target"] > 0).astype(int)

    uci_rename = {
        "trestbps": "sysBP",
        "chol": "totChol",
        "fbs": "diabetes"
    }

    df_c = df_c.rename(columns=uci_rename)

    # Standardize '?' to NaN
    df_c = df_c.replace("?", np.nan)

    # Convert numeric columns
    for col in COMMON_COLUMNS:
        if col in df_c.columns:
            df_c[col] = pd.to_numeric(df_c[col], errors="coerce")

    # Zero-as-missing fix for non-Cleveland sites on continuous columns
    if "source_site" in df_c.columns:
        non_clev_mask = df_c["source_site"] != "cleveland"
        for col in ["sysBP", "totChol"]:
            if col in df_c.columns:
                df_c.loc[non_clev_mask & (df_c[col] == 0), col] = np.nan

    df_c_harmonized = df_c[COMMON_COLUMNS].copy()

    # 2. Harmonize Framingham Dataset
    df_f = df_framingham.copy()

    framingham_rename = {
        "male": "sex",
        "TenYearCHD": "target"
    }

    df_f = df_f.rename(columns=framingham_rename)

    df_f = df_f.replace("?", np.nan)
    for col in COMMON_COLUMNS:
        if col in df_f.columns:
            df_f[col] = pd.to_numeric(df_f[col], errors="coerce")

    df_f["diabetes"] = df_f["diabetes"].fillna(0).astype(int)
    df_f["target"] = df_f["target"].astype(int)

    df_f_harmonized = df_f[COMMON_COLUMNS].copy()

    return df_c_harmonized, df_f_harmonized
