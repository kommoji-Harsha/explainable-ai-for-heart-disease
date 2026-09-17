import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def calculate_brier_score(y_true, y_prob) -> float:
    """
    Computes the Brier score loss.
    Lower values indicate better calibration (0 is perfect).
    """
    return float(brier_score_loss(y_true, y_prob))


def calculate_ece(y_true, y_prob, n_bins=10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples in bin
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = np.sum(in_bin)

        if bin_size > 0:
            avg_acc = np.mean(y_true[in_bin])
            avg_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / n_samples) * np.abs(avg_acc - avg_conf)

    return float(ece)


def fit_platt_scaler(y_train_prob, y_train):
    """
    Fits Platt scaling (logistic regression on predicted log-odds / probabilities).
    """
    # Transform probabilities to log-odds (logit) or fit directly on probabilities
    lr = LogisticRegression(C=1e5, solver="lbfgs")
    # Reshape
    X_tr = np.asarray(y_train_prob).reshape(-1, 1)
    lr.fit(X_tr, y_train)
    return lr


def fit_isotonic_calibrator(y_train_prob, y_train):
    """
    Fits isotonic regression calibrator on probabilities.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_train_prob, y_train)
    return iso


def plot_calibration_curves(
    calibration_data: dict,
    model_name: str,
    output_dir: str = "results",
    n_bins: int = 10
):
    """
    Plots calibration curves (Uncalibrated vs Platt vs Isotonic) along with a perfectly calibrated baseline.

    Args:
        calibration_data: dict containing keys like 'uncalibrated', 'platt', 'isotonic',
                          each mapping to a dict with 'y_true', 'y_prob', 'brier', 'ece'.
        model_name: Name of model (e.g. 'xgboost', 'ensemble').
        output_dir: Output path for plot.
        n_bins: Number of bins for calibration curve.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    # Reference line
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    colors = {
        "uncalibrated": "red",
        "platt": "blue",
        "isotonic": "green"
    }

    labels = {
        "uncalibrated": "Uncalibrated",
        "platt": "Platt Scaling (Sigmoid)",
        "isotonic": "Isotonic Regression"
    }

    for method, data in calibration_data.items():
        if method in colors:
            prob_true, prob_pred = calibration_curve(
                data["y_true"], data["y_prob"], n_bins=n_bins, strategy="uniform"
            )
            brier = data["brier"]
            ece = data["ece"]
            label_str = f"{labels[method]} (Brier: {brier:.4f}, ECE: {ece:.4f})"
            plt.plot(prob_pred, prob_true, "s-", color=colors[method], label=label_str)

    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives", fontsize=12)
    plt.title(f"Calibration Curves - {model_name.upper()}", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    file_path = os.path.join(output_dir, f"{model_name}_calibration_curve.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    return file_path
