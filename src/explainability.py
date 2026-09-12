import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
import numpy as np
import shap


def generate_shap_explanations(
    model,
    X_processed: np.ndarray,
    feature_names: list,
    output_dir: str = "results",
    prefix: str = "model",
    max_display: int = 15,
    sample_indices: list = None
):
    """
    Generates SHAP explanations for a fitted model.
    Saves summary plot, bar plot, and waterfall plots for specified sample predictions.

    Args:
        model: Fitted model object (e.g., TreeEnsemble / XGBoost / Random Forest).
        X_processed: Processed input features matrix (2D numpy array).
        feature_names: List of feature names corresponding to columns of X_processed.
        output_dir: Folder to save plots.
        prefix: Model name prefix for file naming.
        max_display: Max features to display in summary plot.
        sample_indices: List of sample row indices to create individual explanation plots.

    Returns:
        explainer, shap_values
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use TreeExplainer if tree model, otherwise Explainer
    explainer = None
    shap_values = None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_processed)
    except Exception:
        explainer = shap.Explainer(model, X_processed)
        shap_values = explainer(X_processed)

    # Handle multi-class / probability output dimensions if present
    # SHAP output for binary classification with TreeExplainer might have shape (N, D, 2) or (N, D)
    if len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
        # Extract values for class 1 (positive class - heart disease)
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values

    # Set feature names on Explanation object
    shap_values_class1.feature_names = feature_names

    # 1. SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values_class1, max_display=max_display, show=False)
    plt.title(f"SHAP Summary Plot - {prefix}", fontsize=14, pad=15)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, f"{prefix}_shap_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    # 2. SHAP Feature Importance Bar Plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_class1, max_display=max_display, show=False)
    plt.title(f"SHAP Feature Importance - {prefix}", fontsize=14, pad=15)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, f"{prefix}_shap_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    # 3. Individual Sample Waterfall Plots
    if sample_indices is None:
        sample_indices = [0, 1]

    for idx in sample_indices:
        if idx < len(X_processed):
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values_class1[idx], max_display=max_display, show=False)
            plt.title(f"SHAP Waterfall Plot - {prefix} (Sample {idx})", fontsize=14, pad=15)
            plt.tight_layout()
            sample_path = os.path.join(output_dir, f"{prefix}_shap_sample_{idx}.png")
            plt.savefig(sample_path, dpi=300, bbox_inches="tight")
            plt.close("all")

    return explainer, shap_values_class1
