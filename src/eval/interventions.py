import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn import metrics
from einops import rearrange
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import pandas as pd
from src.helpers.pandas_classification_report import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from src.probes.utils import get_classification_report, preproc, postproc, make_dfres_pretty
from src.probes.sk_lr import check_lr_intervention_predictive


def test_intervention_quality2(ds_out, label_fn, thresh=0.03, take_diff=False, verbose=False, title="Intervention predictive power", skip=0, stride=1, model_kwargs={}):
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
    hs_normal = ds_out['end_residual_stream_base'][:, skip::stride]
    hs_intervene = ds_out['end_residual_stream_adapt'][:, skip::stride]

    # print(f"## primary metric: predictive power (of logistic regression on top of intervened hidden states to predict base model Y) [N={len(label)//2}]")
    s1_baseline = check_lr_intervention_predictive(hs_normal, label, **model_kwargs)
    s1_interven = check_lr_intervention_predictive(hs_intervene, label, **model_kwargs)
    predictive = s1_interven['score'] - s1_baseline['score']# > thresh
    # if verbose: print(f"  - predictive power? {predictive} [i]    = baseline: {s1_baseline:.3f} > {s1_interven:.3f} roc_auc [N={len(label)//2}]")
    res['residual_{base}'] = dict(roc_auc=s1_baseline['score'], diff=0)
    res['residual_{adapter}'] = dict(roc_auc=s1_interven['score'], diff=predictive)

    s1_interven2 = check_lr_intervention_predictive(hs_normal-hs_intervene, label, **model_kwargs)
    predictive = s1_interven2['score'] - s1_baseline['score']# > thresh
    
    # if verbose: print(f"  - predictive power? {predictive} [i-b]  = baseline: {s1_baseline:.3f} > {s1_interven:.3f} roc_auc")
    res['residual_{base-adapter}'] = dict(roc_auc=s1_interven2['score'], diff=predictive)

    df_res = pd.DataFrame(res).T
    df_res['pass'] = df_res['diff'] > thresh
    df_styled = df_res.style.pipe(make_dfres_pretty, title)
    return df_styled

