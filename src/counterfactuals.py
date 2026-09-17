import os
import numpy as np
import pandas as pd


MEDICAL_DISCLAIMER = (
    "DISCLAIMER: These counterfactual statements are purely model-based 'what-if' "
    "sensitivity analyses computed for explainability purposes. They represent mathematical "
    "feature perturbations and do NOT constitute medical advice or clinical guidance."
)


def find_counterfactual_for_patient(
    model,
    patient_df_row: pd.Series,
    preprocessor,
    actionable_features: list = None,
    decision_threshold: float = 0.50,
    grid_steps: int = 50
) -> dict:
    """
    Finds minimal realistic changes in 1-3 actionable features (e.g. cholesterol 'chol',
    systolic blood pressure 'trestbps', max heart rate 'thalach', or ST depression 'oldpeak')
    that flip a patient's predicted risk below the decision threshold (0.50).

    Uses a systematic local grid search over plausible physiological feature ranges.

    Args:
        model: Fitted estimator/ensemble expecting preprocessed feature inputs.
        patient_df_row: Unprocessed DataFrame row (Series) for a single patient predicted as high-risk.
        preprocessor: Fitted ColumnTransformer preprocessor.
        actionable_features: List of column names allowed to be modified.
        decision_threshold: Target probability threshold (default: 0.50).
        grid_steps: Number of grid search evaluation points.

    Returns:
        dict containing initial probability, modified probability, feature changes,
        counterfactual statement, and disclaimer.
    """
    if actionable_features is None:
        actionable_features = ["chol", "trestbps", "thalach", "oldpeak"]

    # Preprocess original row to get initial probability
    df_single = pd.DataFrame([patient_df_row])
    X_proc_orig = preprocessor.transform(df_single.drop(columns=["target", "source_site"], errors="ignore"))
    init_prob = float(model.predict_proba(X_proc_orig)[0, 1])

    if init_prob <= decision_threshold:
        return {
            "initial_prob": init_prob,
            "flipped": False,
            "statement": f"Patient is already predicted as low-risk (predicted risk: {init_prob:.1%}).",
            "disclaimer": MEDICAL_DISCLAIMER
        }

    best_cf = None
    min_pct_change = float("inf")

    # Range limits for realistic reductions or increases
    feature_ranges = {
        "chol": (max(100.0, float(patient_df_row.get("chol", 250.0)) * 0.4), float(patient_df_row.get("chol", 250.0))),
        "trestbps": (max(90.0, float(patient_df_row.get("trestbps", 140.0)) * 0.6), float(patient_df_row.get("trestbps", 140.0))),
        "thalach": (float(patient_df_row.get("thalach", 120.0)), min(210.0, float(patient_df_row.get("thalach", 120.0)) * 1.5)),
        "oldpeak": (0.0, float(patient_df_row.get("oldpeak", 1.0)))
    }

    # 1. Search Single Feature Modifications
    for feat in actionable_features:
        if feat not in patient_df_row or pd.isna(patient_df_row[feat]):
            continue

        orig_val = float(patient_df_row[feat])
        min_v, max_v = feature_ranges.get(feat, (orig_val * 0.5, orig_val))

        test_vals = np.linspace(orig_val, min_v if feat != "thalach" else max_v, grid_steps)

        for val in test_vals:
            df_mod = df_single.copy()
            df_mod[feat] = val
            X_proc_mod = preprocessor.transform(df_mod.drop(columns=["target", "source_site"], errors="ignore"))
            mod_prob = float(model.predict_proba(X_proc_mod)[0, 1])

            if mod_prob < decision_threshold:
                pct_change = abs(val - orig_val) / (orig_val if orig_val != 0 else 1.0)
                if pct_change < min_pct_change:
                    min_pct_change = pct_change
                    best_cf = {
                        "type": "single",
                        "feature": feat,
                        "orig_val": orig_val,
                        "new_val": val,
                        "mod_prob": mod_prob,
                        "pct_change": pct_change
                    }
                break

    # 2. If single feature modification fails, search Dual Feature Modifications
    if best_cf is None and len(actionable_features) >= 2:
        for i in range(len(actionable_features)):
            for j in range(i + 1, len(actionable_features)):
                f1, f2 = actionable_features[i], actionable_features[j]
                if f1 in patient_df_row and f2 in patient_df_row and not pd.isna(patient_df_row[f1]) and not pd.isna(patient_df_row[f2]):
                    orig_v1, orig_v2 = float(patient_df_row[f1]), float(patient_df_row[f2])
                    min_v1, max_v1 = feature_ranges.get(f1, (orig_v1 * 0.5, orig_v1))
                    min_v2, max_v2 = feature_ranges.get(f2, (orig_v2 * 0.5, orig_v2))

                    v1_grid = np.linspace(orig_v1, min_v1 if f1 != "thalach" else max_v1, 15)
                    v2_grid = np.linspace(orig_v2, min_v2 if f2 != "thalach" else max_v2, 15)

                    for v1 in v1_grid:
                        for v2 in v2_grid:
                            df_mod = df_single.copy()
                            df_mod[f1] = v1
                            df_mod[f2] = v2
                            X_proc_mod = preprocessor.transform(df_mod.drop(columns=["target", "source_site"], errors="ignore"))
                            mod_prob = float(model.predict_proba(X_proc_mod)[0, 1])

                            if mod_prob < decision_threshold:
                                pct_change = (abs(v1 - orig_v1) / (orig_v1 if orig_v1 != 0 else 1.0)) + (abs(v2 - orig_v2) / (orig_v2 if orig_v2 != 0 else 1.0))
                                if pct_change < min_pct_change:
                                    min_pct_change = pct_change
                                    best_cf = {
                                        "type": "dual",
                                        "f1": f1, "orig_v1": orig_v1, "new_v1": v1,
                                        "f2": f2, "orig_v2": orig_v2, "new_v2": v2,
                                        "mod_prob": mod_prob,
                                        "pct_change": pct_change
                                    }
                                break

    if best_cf is not None:
        if best_cf["type"] == "single":
            stmt = (
                f"Model-based what-if scenario: If this patient's {best_cf['feature']} were modified "
                f"from {best_cf['orig_val']:.1f} to {best_cf['new_val']:.1f}, with other factors unchanged, "
                f"the model's predicted risk would drop from {init_prob:.1%} to {best_cf['mod_prob']:.1%} "
                f"(below the {decision_threshold:.0%} threshold)."
            )
        else:
            stmt = (
                f"Model-based what-if scenario: If this patient's {best_cf['f1']} were modified from "
                f"{best_cf['orig_v1']:.1f} to {best_cf['new_v1']:.1f} and {best_cf['f2']} were modified from "
                f"{best_cf['orig_v2']:.1f} to {best_cf['new_v2']:.1f}, with other factors unchanged, "
                f"the model's predicted risk would drop from {init_prob:.1%} to {best_cf['mod_prob']:.1%} "
                f"(below the {decision_threshold:.0%} threshold)."
            )

        return {
            "initial_prob": init_prob,
            "flipped": True,
            "counterfactual_details": best_cf,
            "statement": stmt,
            "disclaimer": MEDICAL_DISCLAIMER
        }
    else:
        return {
            "initial_prob": init_prob,
            "flipped": False,
            "statement": f"No simple counterfactual modification in {actionable_features} was sufficient to flip prediction below {decision_threshold:.0%}.",
            "disclaimer": MEDICAL_DISCLAIMER
        }


def generate_counterfactual_scenarios(
    model,
    df_raw: pd.DataFrame,
    preprocessor,
    sample_indices: list = None,
    actionable_features: list = None,
    output_dir: str = "results"
) -> list:
    """
    Generates counterfactual what-if scenarios for sample high-risk patients.
    Saves statements to results/counterfactual_scenarios.txt.
    """
    os.makedirs(output_dir, exist_ok=True)

    if actionable_features is None:
        actionable_features = ["chol", "trestbps", "thalach", "oldpeak"]

    if sample_indices is None:
        # Automatically find 3-4 high-risk patients (probabilities between 0.51 and 0.85)
        drop_cols = [c for c in ["target", "source_site"] if c in df_raw.columns]
        X_all_proc = preprocessor.transform(df_raw.drop(columns=drop_cols))
        probs = model.predict_proba(X_all_proc)[:, 1]
        candidate_indices = [i for i, p in enumerate(probs) if 0.51 <= p <= 0.85]
        sample_indices = candidate_indices[:4] if candidate_indices else [0, 1, 2]

    scenarios = []
    out_lines = [MEDICAL_DISCLAIMER, "\n" + "="*80 + "\n"]

    for idx in sample_indices:
        if idx >= len(df_raw):
            continue
        row = df_raw.iloc[idx]
        cf_res = find_counterfactual_for_patient(model, row, preprocessor, actionable_features=actionable_features)
        scenarios.append(cf_res)

        out_lines.append(f"PATIENT INDEX: {idx}")
        out_lines.append(cf_res["statement"])
        out_lines.append("-" * 80 + "\n")

    with open(os.path.join(output_dir, "counterfactual_scenarios.txt"), "w") as f:
        f.write("\n".join(out_lines))

    return scenarios
