import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.model_selection import StratifiedKFold
import shap

from src.preprocessing import build_preprocessor, get_feature_lists
from src.models import get_xgboost, get_random_forest, get_adaboost
from src.optimization import optimize_xgboost


def compute_top_k_overlap(ranking_a: list, ranking_b: list, k: int = 5) -> float:
    """
    Computes Overlap@K between two feature ranking lists.
    Overlap@K = |TopK(A) intersection TopK(B)| / K
    """
    top_a = set(ranking_a[:k])
    top_b = set(ranking_b[:k])
    intersection = len(top_a.intersection(top_b))
    return float(intersection / k)


def compute_kendall_tau_distance(ranking_a: list, ranking_b: list) -> float:
    """
    Computes Kendall's tau correlation coefficient between two ordered feature rankings.
    Converts list of features into rank indices across shared features.
    """
    shared_features = [f for f in ranking_a if f in ranking_b]
    rank_map_a = {feat: i for i, feat in enumerate(ranking_a)}
    rank_map_b = {feat: i for i, feat in enumerate(ranking_b)}

    ranks_a = [rank_map_a[f] for f in shared_features]
    ranks_b = [rank_map_b[f] for f in shared_features]

    tau, _ = kendalltau(ranks_a, ranks_b)
    return float(tau) if not np.isnan(tau) else 0.0


def evaluate_shap_explanation_stability(
    df: pd.DataFrame,
    target_col: str = "target",
    outer_splits: int = 5,
    top_k: int = 5,
    random_state: int = 42
) -> dict:
    """
    Evaluates SHAP explanation stability across nested CV outer folds.
    For each fold:
      1. Fits preprocessor on outer training fold.
      2. Trains model on outer training fold.
      3. Computes mean absolute SHAP value for each feature across fold dataset.
      4. Ranks features from most important to least important.

    Computes pairwise Overlap@K and Kendall's tau correlation across fold rankings.

    Returns:
      dict containing fold rankings, mean Overlap@K, mean Kendall's tau, and plain-language interpretation.
    """
    X_raw = df.drop(columns=[c for c in [target_col, "source_site"] if c in df.columns])
    y = df[target_col].values

    numeric_features, categorical_features = get_feature_lists(X_raw)

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    fold_rankings = []
    fold_top_k_features = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_raw, y)):
        X_train_raw = X_raw.iloc[train_idx]
        y_train = y[train_idx]

        preprocessor = build_preprocessor(numeric_features, categorical_features)
        X_train = preprocessor.fit_transform(X_train_raw)

        from src.preprocessing import get_feature_names
        feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)

        # Train XGBoost model for fold
        xgb_params, _, _ = optimize_xgboost(X_train, y_train, cv=3, n_trials=5, random_state=random_state + fold)
        model = get_xgboost(random_state=random_state, **xgb_params)
        model.fit(X_train, y_train)

        # Compute SHAP values for fold training data
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(X_train)

        if len(shap_vals.shape) == 3 and shap_vals.shape[2] == 2:
            shap_vals_class1 = shap_vals.values[:, :, 1]
        else:
            shap_vals_class1 = shap_vals.values

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(shap_vals_class1), axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]

        sorted_features = [feature_names[i] for i in sorted_indices]
        fold_rankings.append(sorted_features)
        fold_top_k_features.append(sorted_features[:top_k])

    # Compute Pairwise Fold Stability Metrics
    overlap_scores = []
    kendall_taus = []

    num_folds = len(fold_rankings)
    for i in range(num_folds):
        for j in range(i + 1, num_folds):
            ov = compute_top_k_overlap(fold_rankings[i], fold_rankings[j], k=top_k)
            kt = compute_kendall_tau_distance(fold_rankings[i], fold_rankings[j])

            overlap_scores.append(ov)
            kendall_taus.append(kt)

    mean_overlap = float(np.mean(overlap_scores)) if len(overlap_scores) > 0 else 1.0
    mean_kendall = float(np.mean(kendall_taus)) if len(kendall_taus) > 0 else 1.0

    # Plain-language interpretation
    if mean_overlap >= 0.8:
        interpretation = (
            f"High Explanation Stability: Across the {outer_splits} cross-validation folds, "
            f"the Top-{top_k} SHAP features exhibit an average Overlap@{top_k} of {mean_overlap:.1%} "
            f"(Kendall's tau: {mean_kendall:.3f}). The model consistently identifies the same key risk drivers "
            f"regardless of train-test split variations."
        )
    elif mean_overlap >= 0.6:
        interpretation = (
            f"Moderate Explanation Stability: Top-{top_k} SHAP features exhibit an average Overlap@{top_k} "
            f"of {mean_overlap:.1%} across folds (Kendall's tau: {mean_kendall:.3f}). Core risk drivers remain "
            f"largely stable, with minor re-ordering in secondary features across folds."
        )
    else:
        interpretation = (
            f"Low Explanation Stability: Top-{top_k} SHAP features exhibit an average Overlap@{top_k} "
            f"of {mean_overlap:.1%} across folds (Kendall's tau: {mean_kendall:.3f}). Feature importance rankings "
            f"shift substantially across training fold variations."
        )

    return {
        "mean_overlap_at_k": mean_overlap,
        "mean_kendall_tau": mean_kendall,
        "fold_top_k_features": fold_top_k_features,
        "interpretation": interpretation
    }
