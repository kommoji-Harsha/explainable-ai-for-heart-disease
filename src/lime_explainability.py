import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer


def build_lime_explainer(
    X_train_processed: np.ndarray,
    feature_names: list,
    mode: str = "classification"
) -> LimeTabularExplainer:
    """
    Builds a LIME Tabular Explainer fitted on the training feature matrix.
    """
    explainer = LimeTabularExplainer(
        training_data=X_train_processed,
        feature_names=feature_names,
        class_names=["No Disease", "Disease"],
        mode=mode,
        discretize_continuous=True,
        random_state=42
    )
    return explainer


def generate_lime_explanation(
    explainer: LimeTabularExplainer,
    model,
    instance_processed: np.ndarray,
    num_features: int = 10
):
    """
    Generates LIME explanation for a single processed patient instance.
    Returns the LIME explanation object.
    """
    exp = explainer.explain_instance(
        data_row=instance_processed,
        predict_fn=model.predict_proba,
        num_features=num_features
    )
    return exp


def compare_shap_and_lime(
    model,
    X_processed: np.ndarray,
    feature_names: list,
    sample_indices: list = None,
    output_dir: str = "results",
    top_k: int = 5
) -> pd.DataFrame:
    """
    Generates side-by-side comparison of top K contributing features between SHAP and LIME
    for specific sample patients using the XGBoost model.

    Computes Overlap@K (fraction of top K features shared between SHAP and LIME) and documents
    whether SHAP and LIME agree or diverge for each patient.

    Returns:
      DataFrame summarizing SHAP vs LIME top features, Overlap@K, and agreement status.
    """
    os.makedirs(output_dir, exist_ok=True)

    if sample_indices is None:
        sample_indices = [0, 1]

    # Compute SHAP values
    tree_explainer = shap.TreeExplainer(model)
    shap_vals = tree_explainer(X_processed)

    if len(shap_vals.shape) == 3 and shap_vals.shape[2] == 2:
        shap_vals_class1 = shap_vals.values[:, :, 1]
    else:
        shap_vals_class1 = shap_vals.values

    # Build LIME explainer
    lime_explainer = build_lime_explainer(X_processed, feature_names)

    comparison_records = []

    for idx in sample_indices:
        if idx >= len(X_processed):
            continue

        instance = X_processed[idx]

        # 1. Top K features from SHAP (by absolute SHAP magnitude)
        instance_shap = shap_vals_class1[idx]
        top_shap_idx = np.argsort(np.abs(instance_shap))[::-1][:top_k]
        top_shap_features = [feature_names[i] for i in top_shap_idx]

        # 2. Top K features from LIME
        lime_exp = generate_lime_explanation(lime_explainer, model, instance, num_features=top_k)
        # LIME list of tuples: (feature_rule_string, weight)
        lime_tuples = lime_exp.as_list()

        # Extract base feature names from LIME feature rule strings
        top_lime_features = []
        for rule_str, weight in lime_tuples:
            matched_feat = None
            for fname in feature_names:
                if fname in rule_str:
                    matched_feat = fname
                    break
            if matched_feat is None:
                matched_feat = rule_str
            top_lime_features.append(matched_feat)

        # Compute Overlap@K
        set_shap = set(top_shap_features)
        set_lime = set(top_lime_features)
        overlap_count = len(set_shap.intersection(set_lime))
        overlap_at_k = overlap_count / float(top_k)

        status = "Strong Agreement" if overlap_at_k >= 0.6 else "Partial/Divergent Agreement"

        comparison_records.append({
            "patient_index": idx,
            "top_k": top_k,
            "shap_top_features": ", ".join(top_shap_features),
            "lime_top_features": ", ".join(top_lime_features),
            "overlap_count": overlap_count,
            "overlap_ratio": f"{overlap_at_k * 100:.0f}%",
            "agreement_status": status
        })

    comp_df = pd.DataFrame(comparison_records)
    comp_df.to_csv(os.path.join(output_dir, "shap_vs_lime_comparison.csv"), index=False)

    return comp_df
