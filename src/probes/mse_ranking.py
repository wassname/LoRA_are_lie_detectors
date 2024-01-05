
from src.probes.pl_ranking_probe import InceptionBlock, LinBnDrop
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.probes.pl_sk_wrapper import PLSKWrapper, PLSKBase
from torchmetrics.functional import accuracy, auroc


class HSModel2d(nn.Module):
    def __init__(self, c_in, hs, dropout=0, depth=2):
        super().__init__()

        layers = [nn.BatchNorm1d(c_in[1], affine=False)]
        for i in range(depth+1):
            if (i>0) and (i<depth):
                layers.append(InceptionBlock(hs*4, hs, conv_dropout=dropout))
            elif i==0: # first layer
                if depth==0: 
                    layers.append(InceptionBlock(c_in[1], 1))
                else:
                    layers.append(InceptionBlock(c_in[1], hs, conv_dropout=dropout))
            else: # last layer
                layers.append(nn.Conv1d(hs*4, 1, 1))
        self.conv = nn.Sequential(*layers)
        
        n = c_in[0]
        self.head = nn.Sequential(
            LinBnDrop(n, n, p=dropout),
            LinBnDrop(n, n, p=dropout),
            nn.Linear(n, 1),  
        )

    def forward(self, x):
        x = x.float()
        x = rearrange(x, 'b l h -> b h l')
        x = self.conv(x)
        x = rearrange(x, 'b l h -> b (l h)')
        return self.head(x).squeeze(1)


class PLSKMSErank(PLSKBase):
    def __init__(self, c_in, dropout=0, hs=32, depth=2, **kwargs):
        super().__init__(**kwargs)

        self.model = HSModel2d(c_in, hs, dropout)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx, stage='train'):
        x, y = batch
        x0 = x[..., 0]
        x1 = x[..., 1]
        ypred0 = self(x0)
        ypred1 = self(x1)

        y_probs = F.sigmoid(ypred0-ypred1+0.5)
        y_cls = y_probs > 0.5
        
        if stage=='pred':
            return y_probs
        
        loss = F.smooth_l1_loss(ypred0-ypred1, y)

        self.log(f"{stage}/acc", accuracy(y_cls, y>0.5, "binary"), on_epoch=True, on_step=False)
        self.log(f"{stage}/auroc", auroc(y_probs, y>0.5, "binary"), on_epoch=True, on_step=False)
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f"{stage}/n", len(y), on_epoch=True, on_step=False, reduce_fx=torch.sum)
        return loss


class PLSKBoolRank(PLSKMSErank):

    def _step(self, batch, batch_idx, stage='train'):
        x, y = batch
        x0 = x[..., 0]
        x1 = x[..., 1]
        ypred0 = self(x0)
        ypred1 = self(x1)

        y_probs = F.sigmoid(ypred1-ypred0+0.5)
        y_cls = y_probs > 0.5
        
        if stage=='pred':
            return y_probs

        ranking_y = (y>0)*2-1 # from 0,1 to -1,1
        loss = F.margin_ranking_loss(ypred0, ypred1, ranking_y, margin=1)
        
        y_cls = ypred1>ypred0 # switch2bool(ypred1-ypred0)
        self.log(f"{stage}/acc", accuracy(y_cls, y>0.5, "binary"), on_epoch=True, on_step=False)
        self.log(f"{stage}/auroc", auroc(y_probs, y>0.5, "binary"), on_epoch=True, on_step=False)
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f"{stage}/n", len(y), on_epoch=True, on_step=False, reduce_fx=torch.sum)
        return loss
