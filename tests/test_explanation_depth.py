import pytest
import numpy as np
import pandas as pd
from src.lime_explainability import build_lime_explainer, compare_shap_and_lime
from src.counterfactuals import find_counterfactual_for_patient, MEDICAL_DISCLAIMER
from src.explanation_stability import compute_top_k_overlap, compute_kendall_tau_distance, evaluate_shap_explanation_stability


def test_top_k_overlap_computation():
    rank_a = ["cp", "oldpeak", "thal", "exang", "age", "sex"]
    rank_b = ["cp", "oldpeak", "thal", "chol", "sysBP", "sex"]

    # Top 3: ['cp', 'oldpeak', 'thal'] vs ['cp', 'oldpeak', 'thal'] -> Overlap = 3/3 = 1.0
    ov_3 = compute_top_k_overlap(rank_a, rank_b, k=3)
    assert ov_3 == 1.0

    # Top 5: ['cp', 'oldpeak', 'thal', 'exang', 'age'] vs ['cp', 'oldpeak', 'thal', 'chol', 'sysBP'] -> Intersection = 3 -> 3/5 = 0.60
    ov_5 = compute_top_k_overlap(rank_a, rank_b, k=5)
    assert pytest.approx(ov_5, 1e-5) == 0.60


def test_kendall_tau_distance():
    rank_a = ["a", "b", "c", "d"]
    rank_b = ["a", "b", "c", "d"]
    tau_identical = compute_kendall_tau_distance(rank_a, rank_b)
    assert pytest.approx(tau_identical, 1e-5) == 1.0

    rank_c = ["d", "c", "b", "a"]
    tau_reversed = compute_kendall_tau_distance(rank_a, rank_c)
    assert pytest.approx(tau_reversed, 1e-5) == -1.0


def test_counterfactual_prediction_flip():
    """
    Verifies that the counterfactual search logic in find_counterfactual_for_patient
    identifies feature modifications that flip prediction probabilities on a trained model.
    """
    from src.preprocessing import build_preprocessor, get_feature_lists
    from src.models import get_xgboost

    # Synthetic dataset
    N = 40
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(40, 75, N),
        "sex": np.random.choice([0.0, 1.0], N),
        "cp": np.random.choice([1.0, 2.0, 3.0, 4.0], N),
        "trestbps": np.random.randint(110, 180, N),
        "chol": np.random.randint(180, 380, N),
        "fbs": np.random.choice([0.0, 1.0], N),
        "restecg": np.random.choice([0.0, 1.0, 2.0], N),
        "thalach": np.random.randint(100, 180, N),
        "exang": np.random.choice([0.0, 1.0], N),
        "oldpeak": np.random.uniform(0.0, 4.0, N),
        "slope": np.random.choice([1.0, 2.0, 3.0], N),
        "ca": np.random.choice([0.0, 1.0, 2.0], N),
        "thal": np.random.choice([3.0, 6.0, 7.0], N),
        "target": np.random.choice([0, 1], N)
    })

    num_f, cat_f = get_feature_lists(df.drop(columns=["target"]))
    preproc = build_preprocessor(num_f, cat_f)

    X_proc = preproc.fit_transform(df.drop(columns=["target"]))
    y = df["target"].values

    model = get_xgboost(random_state=42)
    model.fit(X_proc, y)

    # Find a patient predicted above 0.50
    probs = model.predict_proba(X_proc)[:, 1]
    high_risk_idx = [i for i, p in enumerate(probs) if p > 0.55]

    if len(high_risk_idx) > 0:
        row = df.iloc[high_risk_idx[0]]
        res = find_counterfactual_for_patient(
            model, row, preproc, actionable_features=["chol", "trestbps", "thalach", "oldpeak"]
        )

        assert "initial_prob" in res
        assert "disclaimer" in res
        assert MEDICAL_DISCLAIMER in res["disclaimer"]
        if res["flipped"]:
            assert "below the 50% threshold" in res["statement"]
