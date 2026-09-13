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

UCI_DATASET_URLS = {
    "cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "va_long_beach": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
}

DEFAULT_RAW_DIR = "data"


def load_single_site(site_name: str, url: str, data_dir: str = DEFAULT_RAW_DIR, file_path: str = None) -> pd.DataFrame:
    """
    Loads raw dataset for a single site. Downloads and caches locally if not present.
    """
    if file_path is None:
        file_path = os.path.join(data_dir, f"raw_{site_name}.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, header=0)
    else:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, file_path)
        df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES, na_values="?")
        df.to_csv(file_path, index=False)

    df["source_site"] = site_name

    # Standardize '?' to NaN
    df = df.replace("?", np.nan)

    # Convert numeric columns
    for col in COLUMN_NAMES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Target binarization (num > 0 -> 1)
    if "target" in df.columns:
        df["target"] = (df["target"] > 0).astype(int)

    return df


def load_combined_dataset(data_dir: str = DEFAULT_RAW_DIR, results_dir: str = "results") -> pd.DataFrame:
    """
    Loads and concatenates all four UCI Heart Disease datasets:
    - Cleveland (N=303)
    - Hungarian (N=294)
    - Switzerland (N=123)
    - VA Long Beach (N=200)
    Total N = 920.

    Adds metadata column `source_site`.
    Handles missing value indicators ('?'), binarizes target (num > 0 -> 1).

    DOCUMENTED DATA QUALITY LIMITATION:
    - Combined missingness in `ca` (number of major vessels) is 66.4% across the 4 sites
      (Cleveland: 1.3%, Hungarian: 99.0%, Switzerland: 95.9%, VA: 99.0%).
    - Combined missingness in `thal` (thalassemia) is 52.8% across the 4 sites
      (Cleveland: 0.7%, Hungarian: 90.5%, Switzerland: 42.3%, VA: 83.0%).
    To preserve full schema consistency across sites, median/mode imputation is applied,
    but metrics should be interpreted with awareness of these high-missingness attributes.
    """
    site_dfs = []
    for site_name, url in UCI_DATASET_URLS.items():
        df_site = load_single_site(site_name, url, data_dir=data_dir)
        site_dfs.append(df_site)

    combined_df = pd.concat(site_dfs, ignore_index=True)

    # Generate and save missingness report table
    generate_missingness_report(combined_df, output_dir=results_dir)

    return combined_df


def generate_missingness_report(df: pd.DataFrame, output_dir: str = "results") -> pd.DataFrame:
    """
    Calculates missingness percentage per feature column per site and overall combined missingness.
    Saves CSV report to results/combined_missingness_report.csv.
    """
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = [c for c in COLUMN_NAMES if c != "target"]

    site_missing = df.groupby("source_site")[feature_cols].apply(lambda g: g.isna().mean() * 100)
    overall_missing = df[feature_cols].isna().mean() * 100
    overall_missing.name = "combined_overall"

    report_df = pd.concat([site_missing, overall_missing.to_frame().T])
    report_df = report_df.round(2)

    report_path = os.path.join(output_dir, "combined_missingness_report.csv")
    report_df.to_csv(report_path)
    return report_df


def load_data(data_path: str = "data/raw_cleveland.csv", url: str = UCI_DATASET_URLS["cleveland"]) -> pd.DataFrame:
    """
    Backward-compatible loader for Cleveland dataset.
    """
    return load_single_site("cleveland", url, file_path=data_path)


def get_feature_lists(df: pd.DataFrame = None):
    """
    Returns tuple of (numeric_features, categorical_features) based on DataFrame columns if provided,
    excluding metadata columns like 'source_site' and target column 'target'.
    """
    if df is not None:
        cols = [c for c in df.columns if c not in ["target", "source_site"]]
        default_num = ["age", "trestbps", "chol", "thalach", "oldpeak", "sysBP", "totChol"]
        default_cat = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "diabetes"]

        numeric_features = [c for c in cols if c in default_num or pd.api.types.is_float_dtype(df[c])]
        categorical_features = [c for c in cols if c not in numeric_features]
        return numeric_features, categorical_features

    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
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
    Splits DataFrame into X and y (excluding 'source_site' metadata), fits preprocessor,
    and returns (X_processed, y, preprocessor, feature_names).
    """
    drop_cols = [c for c in [target_col, "source_site"] if c in df.columns]
    X = df.drop(columns=drop_cols)
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
