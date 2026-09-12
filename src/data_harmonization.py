import os
import urllib.request
import pandas as pd
import numpy as np


FRAMINGHAM_URL = "https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv"
CLEVELAND_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

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
    df_cleveland: pd.DataFrame,
    df_framingham: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Harmonizes Cleveland and Framingham datasets onto a unified common feature schema:
      - age: Patient age in years (continuous)
      - sex: Binary sex encoding (0 = Female, 1 = Male)
      - sysBP: Systolic blood pressure in mmHg (continuous)
               (mapped from Cleveland 'trestbps' and Framingham 'sysBP')
      - totChol: Total cholesterol in mg/dL (continuous)
                 (mapped from Cleveland 'chol' and Framingham 'totChol')
      - diabetes: Fasting blood sugar / diabetes indicator (binary: 0 or 1)
                  (mapped from Cleveland 'fbs' where fbs > 120 mg/dl -> 1, and Framingham 'diabetes')
      - target: Heart disease presence (binary: 0 = No Disease, 1 = Disease Present)
                (mapped from Cleveland 'target > 0' and Framingham 'TenYearCHD')

    Documentation of Feature Loss & Simplification:
      1. Features dropped from Cleveland (missing in Framingham):
         - 'cp' (chest pain type)
         - 'restecg' (resting electrocardiographic results)
         - 'thalach' (maximum heart rate achieved)
         - 'exang' (exercise induced angina)
         - 'oldpeak' (ST depression induced by exercise)
         - 'slope' (slope of peak exercise ST segment)
         - 'ca' (number of major vessels colored by flourosopy)
         - 'thal' (thalassemia type)
      2. Features dropped from Framingham (missing in Cleveland):
         - 'education' (education level)
         - 'currentSmoker' & 'cigsPerDay' (smoking habits)
         - 'BPMeds' (anti-hypertensive medication)
         - 'prevalentStroke' (history of stroke)
         - 'prevalentHyp' (hypertension status)
         - 'diaBP' (diastolic blood pressure)
         - 'BMI' (body mass index)
         - 'heartRate' (resting heart rate)
         - 'glucose' (fasting blood glucose level)

    Returns:
      (df_cleveland_harmonized, df_framingham_harmonized)
    """
    # 1. Harmonize Cleveland Dataset
    df_c = df_cleveland.copy()

    # Ensure binary target
    if "target" in df_c.columns:
        df_c["target"] = (df_c["target"] > 0).astype(int)

    cleveland_rename = {
        "trestbps": "sysBP",
        "chol": "totChol",
        "fbs": "diabetes"
    }

    df_c = df_c.rename(columns=cleveland_rename)

    # Clean missing values '?' if any remain
    df_c = df_c.replace("?", np.nan)
    for col in COMMON_COLUMNS:
        if col in df_c.columns:
            df_c[col] = pd.to_numeric(df_c[col], errors="coerce")

    df_c_harmonized = df_c[COMMON_COLUMNS].copy()

    # 2. Harmonize Framingham Dataset
    df_f = df_framingham.copy()

    framingham_rename = {
        "male": "sex",
        "TenYearCHD": "target"
    }

    df_f = df_f.rename(columns=framingham_rename)

    # Clean missing values
    df_f = df_f.replace("?", np.nan)
    for col in COMMON_COLUMNS:
        if col in df_f.columns:
            df_f[col] = pd.to_numeric(df_f[col], errors="coerce")

    # Map fbs/diabetes to binary int
    df_f["diabetes"] = df_f["diabetes"].fillna(0).astype(int)
    df_f["target"] = df_f["target"].astype(int)

    df_f_harmonized = df_f[COMMON_COLUMNS].copy()

    return df_c_harmonized, df_f_harmonized
