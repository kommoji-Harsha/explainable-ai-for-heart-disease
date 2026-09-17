from sklearn.ensemble import VotingClassifier


def build_voting_ensemble(models: dict, voting: str = "soft") -> VotingClassifier:
    """
    Builds a VotingClassifier (soft or hard) using a dictionary of name -> estimator.

    Args:
        models (dict): Dictionary mapping model name to fitted/unfitted sklearn-compatible estimator.
        voting (str): Voting scheme, default 'soft'.

    Returns:
        VotingClassifier: Scikit-learn soft-voting ensemble model.
    """
    estimators = [(name, model) for name, model in models.items()]
    ensemble = VotingClassifier(estimators=estimators, voting=voting)
    return ensemble
