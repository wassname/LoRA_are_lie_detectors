# https://botorch.org/tutorials/fit_model_with_torch_optimizer
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from einops import rearrange
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import SGD
from sklearn.preprocessing import StandardScaler
from src.helpers.pandas_classification_report import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def _get_and_fit_model(train_X, train_Y, **kwargs):
    # https://botorch.org/tutorials/fit_model_with_torch_optimizer
    model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
    model.train()

    optimizer = SGD([{"params": model.parameters()}], lr=kwargs.get("lr"))
    for epoch in range(kwargs.get("epochs")):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -mll(output, model.train_targets)
        loss.backward()
        optimizer.step()

    return model


def check_intervention_bayesian(hs, y, verbose=False):
    """
    We want the hidden states resulting from interventions to have predictive power
    Lets compare normal hidden states to intervened hidden states
    """
    X = rearrange(hs, 'b l hs -> b (l hs)')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = _get_and_fit_model(X_train, y_train)

    model.eval()
    y_val_pred = model.posterior(X_val).mean
    y_val_pred = model.posterior(X_val).mean

    # y_val_prob = model.predict_proba(X_val)[: ,1]
    score = roc_auc_score(y_val, y_val_prob)

    if verbose:
        target_names = [0, 1]
        cm = confusion_matrix(y_val, y_val_pred, target_names=target_names, normalize='true')
        cr = classification_report(y_val, y_val_pred, target_names=target_names)
        print(cm)
        print(cr)
    
    return score
