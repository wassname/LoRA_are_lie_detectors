
"""from https://github.com/saprmarks/geometry-of-truth/blob/91b223224699754efe83bbd3cae04d434dda0760/probes.py#L75

FIXME
"""
import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from src.helpers.sklearn_lightning import PLSKWrapper, PLSKBase
from torchmetrics.functional import accuracy, auroc

def ccs_loss(p_neg, p_pos):
    consistency_losses = (p_pos - (1 - p_neg)) ** 2
    confidence_losses = torch.min(torch.stack((p_pos, p_neg), dim=-1), dim=-1).values ** 2
    return torch.mean(consistency_losses + confidence_losses)

class PLSKCCS(PLSKBase):
    def __init__(self, c_in, dropout=0, hs=32, depth=2, **kwargs):
        super().__init__(**kwargs)
        self.model = nn.Sequential(
            nn.Linear(np.prod(c_in), 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).squeeze(1)

    def _step(self, batch, batch_idx, stage='train'):
        x, y = batch
        x = rearrange(x, 'b l h v -> b (l h v)').float()
        yb = y>0.5
        ypred = self(x)
        if stage=='pred':
            return ypred

        p_neg = ypred[~yb].mean()
        p_pos = ypred[yb].mean()
        loss = ccs_loss(p_neg, p_pos)
        assert torch.isfinite(loss)

        y_cls = ypred>0.5
        self.log(f"{stage}/acc", accuracy(y_cls, y>0.5, "binary"), on_epoch=True, on_step=False)
        self.log(f"{stage}/auroc", auroc(ypred, y>0.5, "binary"), on_epoch=True, on_step=False)

        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f"{stage}/n", len(y), on_epoch=True, on_step=False, reduce_fx=torch.sum)
        return loss
