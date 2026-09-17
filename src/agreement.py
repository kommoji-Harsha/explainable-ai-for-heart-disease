import numpy as np
import pandas as pd


# Default thresholds for model probability standard deviation.
# Threshold reasoning:
# std < 0.05: High agreement (all models predict probabilities within ~5-10% of each other).
# 0.05 <= std < 0.15: Moderate agreement (minor differences in model confidence, e.g., 60% vs 75%).
# std >= 0.15: Low agreement (strong model disagreement, e.g., RF predicts 30% while XGBoost predicts 80%).
DEFAULT_HIGH_THRESHOLD = 0.05
DEFAULT_LOW_THRESHOLD = 0.15


def calculate_agreement_score(rf_prob: float, xgb_prob: float, ada_prob: float) -> float:
    """
    Computes standard deviation across the three base models' predicted positive class probabilities.
    Lower standard deviation indicates higher model agreement.
    """
    probs = [rf_prob, xgb_prob, ada_prob]
    return float(np.std(probs, ddof=0))


def categorize_agreement(
    std_val: float,
    high_thresh: float = DEFAULT_HIGH_THRESHOLD,
    low_thresh: float = DEFAULT_LOW_THRESHOLD
) -> str:
    """
    Categorizes the standard deviation score into a three-level agreement label.

    - "high agreement": std < high_thresh (default < 0.05)
    - "moderate agreement": high_thresh <= std < low_thresh (default 0.05 <= std < 0.15)
    - "low agreement": std >= low_thresh (default >= 0.15)
    """
    if std_val < high_thresh:
        return "high agreement"
    elif std_val < low_thresh:
        return "moderate agreement"
    else:
        return "low agreement"


def generate_agreement_flag(
    rf_prob: float,
    xgb_prob: float,
    ada_prob: float,
    high_thresh: float = DEFAULT_HIGH_THRESHOLD,
    low_thresh: float = DEFAULT_LOW_THRESHOLD
) -> str:
    """
    Generates a short, plain-language flag string summarizing model agreement for a patient.

    Example output:
    "Model agreement: low — RF: 31%, XGBoost: 75%, AdaBoost: 58%. Interpret this prediction with extra caution."
    """
    std_val = calculate_agreement_score(rf_prob, xgb_prob, ada_prob)
    label = categorize_agreement(std_val, high_thresh, low_thresh)

    rf_pct = int(round(rf_prob * 100))
    xgb_pct = int(round(xgb_prob * 100))
    ada_pct = int(round(ada_prob * 100))

    if label == "low agreement":
        caution_str = " Interpret this prediction with extra caution."
    elif label == "moderate agreement":
        caution_str = " Moderate variance among base classifiers."
    else:
        caution_str = " High consensus among base classifiers."

    flag = f"Model agreement: {label} — RF: {rf_pct}%, XGBoost: {xgb_pct}%, AdaBoost: {ada_pct}%.{caution_str}"
    return flag


def analyze_patient_agreements(
    y_true: np.ndarray,
    y_pred_ensemble: np.ndarray,
    rf_probs: np.ndarray,
    xgb_probs: np.ndarray,
    ada_probs: np.ndarray,
    high_thresh: float = DEFAULT_HIGH_THRESHOLD,
    low_thresh: float = DEFAULT_LOW_THRESHOLD
) -> pd.DataFrame:
    """
    Analyzes per-patient model agreement across a set of out-of-fold predictions.

    Returns a DataFrame with columns:
    [y_true, y_pred, rf_prob, xgb_prob, ada_prob, std_dev, agreement_level, is_correct, flag_string]
    """
    y_true = np.asarray(y_true)
    y_pred_ensemble = np.asarray(y_pred_ensemble)
    rf_probs = np.asarray(rf_probs)
    xgb_probs = np.asarray(xgb_probs)
    ada_probs = np.asarray(ada_probs)

    records = []
    for i in range(len(y_true)):
        rf_p = float(rf_probs[i])
        xgb_p = float(xgb_probs[i])
        ada_p = float(ada_probs[i])

        std_v = calculate_agreement_score(rf_p, xgb_p, ada_p)
        label = categorize_agreement(std_v, high_thresh, low_thresh)
        flag = generate_agreement_flag(rf_p, xgb_p, ada_p, high_thresh, low_thresh)

        records.append({
            "patient_index": i,
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred_ensemble[i]),
            "rf_prob": rf_p,
            "xgb_prob": xgb_p,
            "ada_prob": ada_p,
            "std_dev": std_v,
            "agreement_level": label,
            "is_correct": int(y_true[i] == y_pred_ensemble[i]),
            "flag_string": flag
        })

    df_res = pd.DataFrame(records)
    return df_res


def summarize_agreement_performance(df_agreement: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes accuracy, sample size, and percentage of total for each agreement level.
    """
    levels = ["high agreement", "moderate agreement", "low agreement"]
    summary_rows = []

    total_n = len(df_agreement)

    for level in levels:
        sub = df_agreement[df_agreement["agreement_level"] == level]
        n_sub = len(sub)
        pct_total = (n_sub / total_n * 100) if total_n > 0 else 0.0

        if n_sub > 0:
            acc = float(sub["is_correct"].mean())
        else:
            acc = np.nan

        summary_rows.append({
            "Agreement Level": level,
            "Sample Size (N)": n_sub,
            "% of Total Patients": f"{pct_total:.1f}%",
            "Accuracy": acc
        })

    return pd.DataFrame(summary_rows)
