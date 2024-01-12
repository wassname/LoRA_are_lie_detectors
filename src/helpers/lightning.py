from lightning.pytorch.loggers.csv_logs import CSVLogger
from pathlib import Path
import pandas as pd
import re
from matplotlib import pyplot as plt
from typing import List, Dict


def read_metrics_csv(metrics_file_path):
    df_hist = pd.read_csv(metrics_file_path)
    df_hist["epoch"] = df_hist["epoch"].ffill()
    df_histe = df_hist.set_index("epoch").groupby("epoch").last().ffill().bfill()

    # note the "step" columns is the step of either val or train, not total steps
    df_hist = pd.read_csv(metrics_file_path)
    df_hist_step = df_hist.copy().dropna(axis=1, thresh=len(df_hist)//10)#.dropna(axis=0)
    if 'epoch' in df_hist_step.columns:
        df_hist_step = df_hist_step.drop(columns=['epoch'])
    df_hist_step.index.name = 'total_step'

    return df_histe, df_hist_step
        

def _transform_dl_k(k: str) -> str:
    """
    >>> _transform_dl_k('test/loss_epoch/dataloader_idx_0') -> "val"
    """
    p = re.match(r"test\/(.+)\/dataloader_idx_\d", k)
    return p.group(1) if p else k


def rename_pl_test_results(rs: List[Dict[str, float]], ks=["train", "val", "test"], verbose=True):
    """
    pytorch lighting test outputs `List of dictionaries with metrics logged during the test phase` where the dataloaders are named `test/val/dataloader_idx_0` etc. This renames them to `val` etc.

    usage:
        rs = trainer3.test(net, dataloaders=[dl_train, dl_val, dl_test, dl_ood])
        df_rs = rename_pl_test_results(rs, ["train", "val", "test", "ood"])
    """
    rs = {
        ks[i]: {_transform_dl_k
        (k): v for k, v in rs[i].items()} for i in range(len(ks))
    }
    if verbose:
        print(pd.DataFrame(rs).round(3).to_markdown())
    return pd.DataFrame(rs)

def plot_hist(df_hist, allowlist=None, logy=False):
    """
    assuming lightning logs metrics as train/loss etc, lets plot groups of suffixes together and ignore indexes like "step"
    """
    suffixes = list(set([c.split('/')[-1] for c in df_hist.columns if '/' in c]))
    for suffix in suffixes:
        if allowlist and suffix not in allowlist:
            continue
        plt.figure(figsize=(5, 2))
        df_hist[[c for c in df_hist.columns if c.endswith(suffix) and '/' in c]].plot(title=suffix, style='.', logy=logy, ax=plt.gca())
        plt.title(suffix)   
        plt.show()
