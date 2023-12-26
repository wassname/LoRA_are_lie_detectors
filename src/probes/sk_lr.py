from einops import rearrange
from src.eval.interventions import preproc, postproc
from sklearn.linear_model import LogisticRegression


def check_lr_intervention_predictive(hs, y, verbose=False, scale=True):
    """
    We want the hidden states resulting from interventions to have predictive power
    Lets compare normal hidden states to intervened hidden states
    """
    X = rearrange(hs, 'b l hs -> b (l hs)')
    X_train, X_val, y_train, y_val = preproc(X, y)

    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced',).fit(X_train, y_train)
    y_val_prob = clf.predict_proba(X_val)[: ,1]
    
    return postproc(y_val_prob, y_val, verbose=verbose)
