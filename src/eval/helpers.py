import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from einops import rearrange
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd


def check_intervention_predictive(hs, y):
    """
    We want the hidden states resulting from interventions to have predictive power
    Lets compare normal hidden states to intervened hidden states
    """
    X = rearrange(hs, 'b l hs -> b (l hs)')
    N = len(X)//2
    X_train, X_val = X[:N], X[N:]
    y_train, y_val = y[:N], y[N:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced',).fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    score = roc_auc_score(y_val, y_val_pred)
    return score


def test_intervention_quality2(ds_out, label_fn, tokenizer, thresh=0.03, take_diff=False):
    """
    Check interventions are ordered and different and valid

    TODO better metrics
    - primary metric: **predictive** or a linear classifier on top of intervention hidden states can predict my labels
    - debug metric: **significant** it's not just a small change
    - debug metric: **coherent** 
        - it's not just outputting nonsense, 
        - or just "yes", the choices keep coverage, 
        - it's not over confident
    """

    # collect labels    
    label = label_fn(ds_out)

    # collect hidden states
    hs_normal = ds_out['end_residual_stream_base']
    hs_intervene = ds_out['end_residual_stream_adapt']
    if take_diff:
        print("taking diff")
        hs_normal = hs_normal.diff(1)
        hs_intervene = hs_intervene.diff(1)

    print("primary metric: predictive power (of logistic regression on top of intervened hidden states)")
    s1_baseline = check_intervention_predictive(hs_normal, label)
    s1_interven = check_intervention_predictive(hs_intervene, label)
    predictive = s1_interven - s1_baseline > thresh
    print(f"predictive power? {predictive} [i] = baseline: {s1_baseline:.3f} > {s1_interven:.3f} roc_auc")
    s1_interven = check_intervention_predictive(hs_intervene-hs_normal, label)
    predictive = s1_interven - s1_baseline > thresh
    print(f"predictive power? {predictive} [i-b] = baseline: {s1_baseline:.3f} > {s1_interven:.3f} roc_auc")

    s1_baseline = check_intervention_predictive(hs_normal.diff(1), label)
    s1_interven = check_intervention_predictive(hs_intervene.diff(1), label)
    predictive = s1_interven - s1_baseline > thresh
    print(f"predictive power? {predictive} [diff]  = baseline: {s1_baseline:.3f} > {s1_interven:.3f} roc_auc")
    s1_interven = check_intervention_predictive((hs_intervene-hs_normal).diff(1), label)
    predictive = s1_interven - s1_baseline > thresh
    print(f"predictive power? {predictive} [diff(i-b)] = baseline: {s1_baseline:.3f} > {s1_interven:.3f} roc_auc")

    # also check coverage
    # also check reasonable probs (e.g choices not too high, others not too low)
    # also check the probs actually makes a differen't to ans
    # We would hope that an unrelated tokens would have it's probability mostly uneffected

    # id_unrelated = tokenizer.encode('\n')[0]
    # unrelated_probs_a = torch.softmax(ds_out['end_logits_adapt'], 0)[:, id_unrelated].mean(0).item()
    # unrelated_probs_b = torch.softmax(ds_out['end_logits_base'], 0)[:, id_unrelated].mean(0).item()
    df_metrics = pd.DataFrame({
        'coverage': [
            ds_out['choice_probs_base'].mean(0).sum(0).item(),
            ds_out['choice_probs_adapt'].mean(0).sum(0).item(),
        ],
        'ans': [
            ds_out['binary_ans_base'].mean(0).item(),
            ds_out['binary_ans_adapt'].mean(0).item()
            ],
        # 'unrelated_probs': [unrelated_probs_a, unrelated_probs_b],
    }, index=['baseline', 'intervene']).T
    print(df_metrics)

