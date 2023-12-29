import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn import metrics
from einops import rearrange
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import pandas as pd
from src.helpers.pandas_classification_report import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


def get_classification_report(y_test, y_pred):
    """Source: https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format"""

    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(
        by=["f1-score"], ascending=False
    )
    return df_classification_report


# def preproc(X, y, with_scaling=True, with_centering=False):
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=False)

#     scaler = RobustScaler(with_centering=with_centering, with_scaling=with_scaling)
#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.transform(X_val)

#     if torch.is_tensor(X):
#         X_train = torch.from_numpy(X_train).float()
#         X_val = torch.from_numpy(X_val).float()

#     return X_train, X_val, y_train, y_val


def postproc(y_val_prob, y_val, verbose=True):
    score = roc_auc_score(y_val, y_val_prob)
    y_val_pred = y_val_prob > 0.5
    target_names = [0, 1]
    cm = confusion_matrix(
        y_val, y_val_pred, target_names=target_names, normalize="true"
    )
    cr = classification_report(y_val, y_val_pred, target_names=target_names)
    print(f"roc_auc_score: {score:.3f}")
    if verbose:
        print(cm)
        print(cr)

    metrics = pd.Series(dict(score=score,
        class_balance=(y_val*1.0).mean(),
        support=len(y_val)))
    return dict(
        metrics=metrics,

        # dfs
        cm=cm,
        cr=cr,

        # tensors
        y_val_pred=y_val_pred,
        y_val_prob=y_val_prob,
        y_val=y_val,
    )


def make_dfres_pretty(styler, title):
    styler.set_caption(title)
    styler.background_gradient(
        axis="index", vmin=0, vmax=1, cmap="RdYlGn", subset=["roc_auc", "pass"]
    )
    styler.background_gradient(
        axis="index", vmin=-0.05, vmax=0.05, cmap="RdYlGn", subset=["diff"]
    )
    return styler

