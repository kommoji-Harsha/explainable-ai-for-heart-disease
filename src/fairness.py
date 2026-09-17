import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix


CAVEAT_MESSAGE = (
    "NOTE / CAVEAT: Subgroup and per-site sample sizes vary in size (e.g. small female or per-site cohorts) "
    "and disease prevalence (e.g. Switzerland disease prevalence is 92.7% while Cleveland is 45.9%). "
    "High Accuracy or Recall in high-prevalence cohorts can reflect positive-class prediction bias unless "
    "contextualized with Precision, Specificity, and F1-score. Subgroup breakdowns should be read as exploratory/indicative."
)


def create_age_bands(df: pd.DataFrame, age_col: str = "age") -> pd.Series:
    """
    Splits age into 3 sensible age bands: '< 50', '50-60', '> 60'.
    """
    ages = df[age_col]
    bands = pd.cut(
        ages,
        bins=[-np.inf, 49.99, 60.0, np.inf],
        labels=["< 50", "50-60", "> 60"]
    )
    return bands


def evaluate_subgroup_performance(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, group_col: str) -> pd.DataFrame:
    """
    Computes Accuracy, Recall, Precision, Specificity (True Negative Rate), F1, ROC-AUC,
    and sample size for each subgroup in group_col.

    Args:
        df: DataFrame containing the group column (same length/order as y_true).
        y_true: True binary targets.
        y_pred: Binary predictions.
        y_prob: Predicted positive class probabilities.
        group_col: Column name in df to group by (e.g. 'sex', 'age_band', 'source_site').

    Returns:
        DataFrame with comprehensive subgroup performance metrics and sample sizes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    groups = df[group_col].values

    unique_groups = sorted(pd.Series(groups).unique())
    rows = []

    for g in unique_groups:
        mask = (groups == g)
        sub_true = y_true[mask]
        sub_pred = y_pred[mask]
        sub_prob = y_prob[mask]
        n_sub = int(np.sum(mask))

        acc = float(accuracy_score(sub_true, sub_pred)) if n_sub > 0 else np.nan
        rec = float(recall_score(sub_true, sub_pred, zero_division=0)) if n_sub > 0 else np.nan
        prec = float(precision_score(sub_true, sub_pred, zero_division=0)) if n_sub > 0 else np.nan
        f1 = float(f1_score(sub_true, sub_pred, zero_division=0)) if n_sub > 0 else np.nan

        # Specificity (True Negative Rate = TN / (TN + FP))
        cm = confusion_matrix(sub_true, sub_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan

        # ROC-AUC requires both classes present in subgroup
        if len(np.unique(sub_true)) > 1:
            auc = float(roc_auc_score(sub_true, sub_prob))
        else:
            auc = np.nan

        disease_prev = float(np.mean(sub_true)) * 100 if n_sub > 0 else np.nan

        rows.append({
            "subgroup": str(g),
            "sample_size": n_sub,
            "disease_prevalence": f"{disease_prev:.1f}%",
            "accuracy": acc,
            "recall": rec,
            "precision": prec,
            "specificity": spec,
            "f1_score": f1,
            "roc_auc": auc
        })

    results_df = pd.DataFrame(rows)
    return results_df


def compute_fairness_breakdown(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, target_col: str = "target") -> dict:
    """
    Computes subgroup fairness breakdown across sex, age bands, and source_site (if present).

    Caveat Notice:
        Subgroup sample sizes and per-site cohorts vary in size and disease prevalence.
        Precision and Specificity contextualize Accuracy/Recall to guard against prevalence bias.

    Args:
        df: Original DataFrame (containing 'sex', 'age', and optionally 'source_site').
        y_true: True targets.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.

    Returns:
        dict containing DataFrames for sex, age_band, and site_breakdown performance breakdowns,
        along with the caveat message.
    """
    df_meta = df.copy()

    # Map sex values for readability if numeric
    if "sex" in df_meta.columns:
        df_meta["sex_label"] = df_meta["sex"].map({0.0: "Female (0)", 1.0: "Male (1)", 0: "Female (0)", 1: "Male (1)"}).fillna(df_meta["sex"])
    else:
        df_meta["sex_label"] = "Unknown"

    # Create age bands
    df_meta["age_band"] = create_age_bands(df_meta, age_col="age")

    sex_breakdown = evaluate_subgroup_performance(df_meta, y_true, y_pred, y_prob, group_col="sex_label")
    age_breakdown = evaluate_subgroup_performance(df_meta, y_true, y_pred, y_prob, group_col="age_band")

    res = {
        "sex_breakdown": sex_breakdown,
        "age_breakdown": age_breakdown,
        "caveat": CAVEAT_MESSAGE
    }

    if "source_site" in df_meta.columns:
        site_breakdown = evaluate_subgroup_performance(df_meta, y_true, y_pred, y_prob, group_col="source_site")
        res["site_breakdown"] = site_breakdown

    return res
