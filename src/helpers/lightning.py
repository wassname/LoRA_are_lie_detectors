from lightning.pytorch.loggers.csv_logs import CSVLogger
from pathlib import Path
import pandas as pd

def read_metrics_csv(metrics_file_path):
    df_hist = pd.read_csv(metrics_file_path)
    df_hist["epoch"] = df_hist["epoch"].ffill()
    df_histe = df_hist.set_index("epoch").groupby("epoch").mean().ffill().bfill()

    df_hist = pd.read_csv(metrics_file_path)
    df_hist_step = df_hist.copy().set_index("step").dropna(axis=1, thresh=len(df_hist)//2).dropna(axis=0)

    return df_histe, df_hist_step
        