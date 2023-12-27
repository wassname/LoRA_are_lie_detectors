"""
A lightning template for a sklearn-like interface (with  fit, predict_proba)
"""


import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torchmetrics.functional import accuracy, auroc
import pandas as pd
import numpy as np

class PLSKBase(pl.LightningModule):
    """
    Base pytorch lightning module, subclass to add model
    """
    def __init__(self, epoch_steps: int, max_epochs: int, lr=4e-3, weight_decay=1e-9):
        super().__init__()
        self.model = None # subclasses must add this
        self.total_steps = epoch_steps * max_epochs
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
        
    def _step(self, batch, batch_idx, stage='train'):
        x0, y = batch

        logits = self(x0)
        y_probs = F.sigmoid(logits)
        y_cls = y_probs > 0.5
        
        if stage=='pred':
            return y_probs
        
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        self.log(f"{stage}/acc", accuracy(y_cls, y, "binary"), on_epoch=True, on_step=False)
        self.log(f"{stage}/auroc", auroc(y_probs, y, "binary"), on_epoch=True, on_step=False)
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f"{stage}/n", len(y), on_epoch=True, on_step=False, reduce_fx=torch.sum)
        return loss
    
    def training_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx=0):
        return self._step(batch, batch_idx, stage='val')
    
    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx, stage='pred').cpu().detach()
    
    def test_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx, stage='test')
    
    def configure_optimizers(self):
        """a one cycle adam optimizer if very general and robust"""
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, self.hparams.lr, total_steps=self.hparams.epoch_steps * self.hparams.max_epochs,
        )
        return [optimizer], [lr_scheduler]
    

def read_metrics_csv(metrics_file_path):
    df_hist = pd.read_csv(metrics_file_path)
    df_hist["epoch"] = df_hist["epoch"].ffill()
    df_histe = df_hist.set_index("epoch").groupby("epoch").last().ffill().bfill()
    return df_histe

class PLSKWrapper:

    def __init__(self, pl_model, max_epochs=20, batch_size=32, verbose=False):
        self.max_epochs = max_epochs

        self.pl_model = pl_model
        self.verbose=verbose
        self.batch_size = batch_size

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):

        dl_train = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        dl_val = None
        if X_val is not None:
            dl_val = DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)
        self.trainer = pl.Trainer(
            # gradient_clip_val=20,
            # accelerator="auto",
            max_epochs=self.max_epochs,
            log_every_n_steps=1,
            enable_progress_bar=self.verbose, enable_model_summary=self.verbose,
            **kwargs
        )
        self.trainer.fit(model=self.pl_model, train_dataloaders=dl_train, val_dataloaders=dl_val)
        

        self.df_hist = read_metrics_csv(self.trainer.logger.experiment.metrics_file_path)
        if self.verbose:
            suffixes = sorted(set([c.split('/')[0] for c in self.df_hist.columns]))
            for s in suffixes:
                self.df_hist[[c for c in self.df_hist.columns if c.endswith(s)]].plot(title=s)
        return self

    def predict_proba(self, X):
        y = torch.zeros(len(X))
        dl_val = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=False)
        r = self.trainer.predict(self.pl_model, dataloaders=dl_val)
        y_pred_prob = torch.cat(r).flatten()
        return y_pred_prob

    def predict(self, X):
        return self.pl_model.predict_proba(X) > 0.5
    
    def score(self, X, y):
        return self.pl_model.score(X, y)
