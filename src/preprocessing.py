import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

DEFAULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def load_data(data_path: str = "data/raw_cleveland.csv", url: str = DEFAULT_URL) -> pd.DataFrame:
    """
    Load Cleveland dataset. If raw_path exists, loads from it; otherwise fetches from URL,
    saves to data_path, and returns the DataFrame.
    Replaces missing values '?' with NaN and converts target > 0 to 1 (binary classification).
    """
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, header=0)
    else:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        urllib.request.urlretrieve(url, data_path)
        df = pd.read_csv(data_path, header=None, names=COLUMN_NAMES, na_values="?")
        # Save clean raw file with headers
        df.to_csv(data_path, index=False)

    # Ensure missing value standard representation
    df = df.replace("?", np.nan)

    # Ensure numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binary classification: target = 1 if disease present (1, 2, 3, 4), 0 otherwise
    if "target" in df.columns:
        df["target"] = (df["target"] > 0).astype(int)

    return df


def get_feature_lists(df: pd.DataFrame = None):
    """
    Returns tuple of (numeric_features, categorical_features) based on DataFrame columns if provided,
    otherwise returns default Cleveland full feature set.
    """
    if df is not None:
        cols = [c for c in df.columns if c != "target"]
        default_num = ["age", "trestbps", "chol", "thalach", "oldpeak", "sysBP", "totChol"]
        default_cat = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "diabetes"]

        numeric_features = [c for c in cols if c in default_num or pd.api.types.is_float_dtype(df[c])]
        categorical_features = [c for c in cols if c not in numeric_features]
        return numeric_features, categorical_features

    # Default continuous numeric features for full Cleveland
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    # Default categorical / discrete features
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features=None, categorical_features=None):
    """
    Builds a scikit-learn ColumnTransformer that imputes and scales numeric features,
    and imputes and one-hot encodes categorical features.
    """
    if numeric_features is None or categorical_features is None:
        numeric_features, categorical_features = get_feature_lists()

    transformers = []

    if len(numeric_features) > 0:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, numeric_features))

    if len(categorical_features) > 0:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers)
    return preprocessor


def preprocess_data(df: pd.DataFrame, target_col: str = "target"):
    """
    Splits DataFrame into X and y, fits preprocessor, and returns (X_processed, y, preprocessor, feature_names).
    """
    X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
    y = df[target_col].values if target_col in df.columns else None

    numeric_features, categorical_features = get_feature_lists(X)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_processed = preprocessor.fit_transform(X)
    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)

    return X_processed, y, preprocessor, feature_names


def get_feature_names(preprocessor, numeric_features, categorical_features):
    """
    Extract feature names after OneHotEncoder transformation.
    """
    feature_names = list(numeric_features)
    if "cat" in preprocessor.named_transformers_:
        cat_transformer = preprocessor.named_transformers_["cat"]
        onehot = cat_transformer.named_steps["onehot"]
        cat_feature_names = list(onehot.get_feature_names_out(categorical_features))
        feature_names += cat_feature_names
    return feature_names
