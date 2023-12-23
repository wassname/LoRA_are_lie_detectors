import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from einops import rearrange
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
from src.helpers.pandas_classification_report import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def get_classification_report(y_test, y_pred):
    '''Source: https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format'''
    from sklearn import metrics
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report

def check_intervention_predictive(hs, y, verbose=False):
    """
    We want the hidden states resulting from interventions to have predictive power
    Lets compare normal hidden states to intervened hidden states
    """
    X = rearrange(hs, 'b l hs -> b (l hs)')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced',).fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_val_prob = clf.predict_proba(X_val)[: ,1]
    score = roc_auc_score(y_val, y_val_prob)

    if verbose:
        target_names = [0, 1]
        cm = confusion_matrix(y_val, y_val_pred, target_names=target_names, normalize='true')
        cr = classification_report(y_val, y_val_pred, target_names=target_names)
        print(cm)
        print(cr)
    
    return score

def check_intervention_predictive_nn(hs, y):
    """
    We want the hidden states resulting from interventions to have predictive power
    Lets compare normal hidden states to intervened hidden states
    """
    # TODO use a linear layer...
    X = rearrange(hs, 'b l hs -> b (l hs)')
    N = len(X)//2
    X_train, X_val = X[:N], X[N:]
    y_train, y_val = y[:N], y[N:]

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced',).fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    score = roc_auc_score(y_val, y_val_pred)
    return score


def test_intervention_quality2(ds_out, label_fn, tokenizer, thresh=0.03, take_diff=False, verbose=False):
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
    res = {}

    # collect labels    
    label = label_fn(ds_out)

    # collect hidden states
    hs_normal = ds_out['end_residual_stream_base']
    hs_intervene = ds_out['end_residual_stream_adapt']

    # print(f"## primary metric: predictive power (of logistic regression on top of intervened hidden states to predict base model Y) [N={len(label)//2}]")
    s1_baseline = check_intervention_predictive(hs_normal, label)
    s1_interven = check_intervention_predictive(hs_intervene, label)
    predictive = s1_interven - s1_baseline > thresh
    if verbose: print(f"  - predictive power? {predictive} [i]    = baseline: {s1_baseline:.3f} > {s1_interven:.3f} roc_auc [N={len(label)//2}]")
    res['predictive'] = dict(roc_auc_baseline=s1_baseline, roc_auc_interven=s1_interven, predictive=predictive)

    s1_interven = check_intervention_predictive(hs_intervene-hs_normal, label)
    predictive = s1_interven - s1_baseline > thresh
    
    if verbose: print(f"  - predictive power? {predictive} [i-b]  = baseline: {s1_baseline:.3f} > {s1_interven:.3f} roc_auc")
    res['predictive_diff'] = dict(roc_auc_baseline=s1_baseline, roc_auc_interven=s1_interven, predictive=predictive)

    # hs = torch.concat([hs_intervene, hs_normal], 1)
    # s1_interven = check_intervention_predictive(hs, label)
    # predictive = s1_interven - s1_baseline > thresh
    # print(f"- predictive power? {predictive} [i, b] = baseline: {s1_baseline:.3f} > {s1_interven:.3f} roc_auc")

    # s1_baseline = check_intervention_predictive(hs_normal.diff(1), label)
    # s1_interven = check_intervention_predictive(hs_intervene.diff(1), label)
    # predictive = s1_interven - s1_baseline > thresh
    # print(f"predictive power? {predictive} [diff]  = baseline: {s1_baseline:.3f} > {s1_interven:.3f} roc_auc")
    # s1_interven = check_intervention_predictive((hs_intervene-hs_normal).diff(1), label)
    # predictive = s1_interven - s1_baseline > thresh
    # print(f"predictive power? {predictive} [diff(i-b)] = baseline: {s1_baseline:.3f} > {s1_interven:.3f} roc_auc")

    # also check coverage
    # also check reasonable probs (e.g choices not too high, others not too low)
    # also check the probs actually makes a differen't to ans
    # We would hope that an unrelated tokens would have it's probability mostly uneffected



    df_res = pd.DataFrame(res).T
    return df_res

