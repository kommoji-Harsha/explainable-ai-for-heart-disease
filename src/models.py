from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier


def get_random_forest(random_state: int = 42, **kwargs) -> RandomForestClassifier:
    """
    Returns a Random Forest classifier instance.
    Default hyperparameters can be overridden via kwargs.
    """
    params = {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": random_state,
        "n_jobs": -1
    }
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_xgboost(random_state: int = 42, **kwargs) -> XGBClassifier:
    """
    Returns an XGBoost classifier instance.
    Default hyperparameters can be overridden via kwargs.
    """
    params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "random_state": random_state,
        "eval_metric": "logloss",
        "n_jobs": -1
    }
    params.update(kwargs)
    return XGBClassifier(**params)


def get_adaboost(random_state: int = 42, **kwargs) -> AdaBoostClassifier:
    """
    Returns an AdaBoost classifier instance.
    Default hyperparameters can be overridden via kwargs.
    """
    params = {
        "n_estimators": 50,
        "learning_rate": 1.0,
        "random_state": random_state
    }
    params.update(kwargs)
    return AdaBoostClassifier(**params)


def get_baseline_models(random_state: int = 42) -> dict:
    """
    Returns a dictionary of default/baseline model instances.
    """
    return {
        "random_forest": get_random_forest(random_state=random_state),
        "xgboost": get_xgboost(random_state=random_state),
        "adaboost": get_adaboost(random_state=random_state)
    }
