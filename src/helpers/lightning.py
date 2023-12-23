from lightning.pytorch.loggers.csv_logs import CSVLogger
from pathlib import Path
import pandas as pd
import re

def read_metrics_csv(metrics_file_path):
    df_hist = pd.read_csv(metrics_file_path)
    df_hist["epoch"] = df_hist["epoch"].ffill()
    df_histe = df_hist.set_index("epoch").groupby("epoch").last().ffill().bfill()

    # FIXME, it turns out steps doesn't make sense. That is step of train and val are not comparable
    df_hist = pd.read_csv(metrics_file_path)
    df_hist_step = df_hist.copy().dropna(axis=1, thresh=len(df_hist)//10)#.dropna(axis=0)
    if 'epoch' in df_hist_step.columns:
        df_hist_step = df_hist_step.drop(columns=['epoch'])

    return df_histe, df_hist_step
        

def _transform_dl_k(k: str) -> str:
    """
    >>> _transform_dl_k('test/loss_epoch/dataloader_idx_0') -> "val"
    """
    p = re.match(r"test\/(.+)\/dataloader_idx_\d", k)
    return p.group(1) if p else k


def rename_pl_test_results(rs, ks=["train", "val", "test"]):
    """
    pytorch lighting test outputs `List of dictionaries with metrics logged during the test phase` where the dataloaders are named `test/val/dataloader_idx_0` etc. This renames them to `val` etc.
    """
    rs = {
        ks[i]: {_transform_dl_k
        (k): v for k, v in rs[i].items()} for i in range(len(ks))
    }
    return rs
