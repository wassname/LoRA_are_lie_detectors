from lightning.pytorch.loggers.csv_logs import CSVLogger
from pathlib import Path
import pandas as pd

def read_metrics_csv(metrics_file_path):
    df_hist = pd.read_csv(metrics_file_path)
    df_hist["epoch"] = df_hist["epoch"].ffill()
    df_histe = df_hist.set_index("epoch").groupby("epoch").last().ffill().bfill()

    df_hist = pd.read_csv(metrics_file_path)
    df_hist_step = df_hist.copy().groupby('step').first().dropna(axis=1, thresh=len(df_hist)//10)#.dropna(axis=0)
    if 'epoch' in df_hist_step.columns:
        df_hist_step = df_hist_step.drop(columns=['epoch'])

    return df_histe, df_hist_step
        