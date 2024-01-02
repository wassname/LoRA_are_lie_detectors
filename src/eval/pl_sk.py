
from src.eval.ds import ds2df
from src.probes.lr import TorchRobustScaler, TorchLogisticRegression, TorchDummyClassifier
from src.probes.utils import postproc
from datasets import Dataset
import pandas as pd
from src.helpers.ds import train_test_split_ds
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from IPython.display import display

import dataclasses
from typing import Callable
from src.probes.utils import postproc
from src.probes.lr import TorchRobustScaler, TorchLogisticRegression, TorchDummyClassifier
from src.helpers.ds import train_test_split_ds
from sklearn.metrics import roc_auc_score, accuracy_score
from IPython.display import display
from typing import Tuple
from torch import Tensor


# FIXME, I should subclass instead of passing around functions...


def analyze_dfres(df_res, insample_datasets, trainval_datasets):
    data = {}
    for n, g in df_res.groupby('ds_string_base'):
        roc_auc = roc_auc_score(g['y'], g['y_prob'])
        acc = accuracy_score(g['y'], g['y_prob']>0.5)
        in_adapter_distribution = n in insample_datasets
        in_train = n in trainval_datasets


        roc_auc_adapter = roc_auc_score(g['label_true_adapt2'], g['binary_ans_adapt2'])
        roc_auc_base = roc_auc_score(g['label_true_base2'], g['binary_ans_base2'])


        s= pd.Series(dict(
            roc_auc=roc_auc,
            improvement=roc_auc-max(roc_auc_adapter, roc_auc_base),
            acc=acc,
            n=len(g),
            in_dist_adapter=in_adapter_distribution,
            in_dist_probe=in_train,
            balance=g['y'].mean(),
            balance_proxy=g['y_test_proxy'].mean(),
            roc_auc_adapter=roc_auc_adapter,
            roc_auc_base=roc_auc_base,

            # baseline=g['y_proxy_prob'].mean(),
        ))
        # print(s)
        data[n]=s

    df = pd.DataFrame(data).T.sort_values('improvement', ascending=False)
    # df['better'] = df['roc_auc']-df[['roc_auc_adapter', 'roc_auc_base']].values.max(1)
    display(df)
    return df



@dataclasses.dataclass(kw_only=True)
class SKEvaluator:

    """
    a dataset to use for training the probe.
    """
    ds_trainval: Dataset

    """
    a dataset to use for testing the probe.
    """
    ds_test: Dataset


    def ds2xy(self, ds: Dataset) -> Tuple[Tensor, Tensor]:
        """function which will transform a dataset into x and y."""
        raise NotImplementedError

    
    def ds2proxy(self, ds: Dataset):
        """
        extract the proxy label from the dataset, can be the label is proxy2label is not set
        """
        raise NotImplementedError
    
    def proxy2label(self, proxy, ds: Dataset):
        """
        will transform a proxy label back into the ground truth label, can be noop
        """
        raise NotImplementedError


    def ds2dfres(self, model, scaler, ds_test, verbose=True):
        """dataset to dataframe with predictions"""
        X_test, y_test_proxy = self.ds2xy(ds_test)
        X_test = scaler.transform(X_test)
        y_test_proxy_prob = model.predict_proba(X_test)
        y_test = self.proxy2label(y_test_proxy, ds_test)
        y_test_prob = self.proxy2label(y_test_proxy_prob, ds_test)
        df_test = ds2df(ds_test)
        df_test['y_prob'] = y_test_prob
        df_test['y'] = y_test > 0.5
        df_test['y_proxy_prob'] = y_test_prob
        df_test['y_test_proxy'] = y_test_proxy
        if verbose:
            postproc(y_test_prob, y_test, verbose=verbose)
        return df_test


    def eval(self, model):
        """Evaluate a scikit learn style model, with a ranking proxy label."""

        # split
        ds_train, ds_val = train_test_split_ds(self.ds_trainval, test_size=0.3)
        X_val, y_val_proxy = self.ds2xy(ds_val)
        X_train, y_train_proxy = self.ds2xy(ds_train)

        # scale
        scaler = TorchRobustScaler(with_centering=False, with_scaling=True)
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model.fit(X_train=X_train, y_train=y_train_proxy, X_val=X_val, y_val=y_val_proxy)

        # must return a single torch float probability
        y_val_proxy_prob = model.predict_proba(X_val)
        assert y_val_proxy_prob.max()<=1
        assert y_val_proxy_prob.min()>=0
        assert y_val_proxy_prob.ndim == 1, "model must return a single probability"
        assert y_val_proxy_prob.shape[0] == X_val.shape[0], "model must return a single probability for each sample"
        assert isinstance(y_val_proxy_prob, torch.Tensor), "model must return a torch tensor"
        assert torch.is_floating_point(y_val_proxy_prob), "model must return a torch tensor"

        df_res1 = self.ds2dfres(model, scaler, ds_val, False)
        df_res2 = self.ds2dfres(model, scaler, self.ds_test, False)
        df_res = pd.concat([df_res1, df_res2])
        return df_res
    


@dataclasses.dataclass(kw_only=True)
class PlainTruthEval(SKEvaluator):

    importance_matrix: Tensor

    def ds2xy(self, ds: Dataset) -> Tuple[Tensor, Tensor]:
        X1 = torch.stack([ds['end_residual_fc1_base'], ds['end_residual_fc1_adapt']], dim=-1)
        X2 = torch.stack([ds['end_residual_Wqkv_base'], ds['end_residual_Wqkv_adapt']], dim=-1)
        x = torch.concat([X1, X2], dim=2)
        x = x * self.importance_matrix[None, :, :, None]**2
        y = self.ds2proxy(ds)
        return x, y

    def ds2proxy(self, ds: Dataset):
        """label: whether the model told the truth"""
        ans = ds["binary_ans_base"] > 0.5
        labels_true_ans = ds["label_true_base"] == ans
        return labels_true_ans

    def proxy2label(self, proxy, ds: Dataset):
        return proxy

    
