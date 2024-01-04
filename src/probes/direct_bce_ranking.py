
from src.probes.pl_ranking_probe import InceptionBlock, LinBnDrop
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

class HSModel3d(nn.Module):
    def __init__(self, c_in, hs, dropout=0, depth=2):
        super().__init__()
        self.preconv = nn.Sequential(
            nn.Conv2d(c_in[1], hs*4, (1, 2)),
        )

        layers = [nn.BatchNorm1d(hs*4, affine=False)]
        for i in range(depth+1):
            if (i>0) and (i<depth):
                layers.append(InceptionBlock(hs*4, hs, conv_dropout=dropout))
            elif i==0: # first layer
                if depth==0: 
                    layers.append(InceptionBlock(hs*4, 1))
                else:
                    layers.append(InceptionBlock(hs*4, hs, conv_dropout=dropout))
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
        if x.ndim==4:
            x = x.squeeze(3)
        x = rearrange(x, 'b l h v -> b h l v')
        x = self.preconv(x)
        x = rearrange(x, 'b h l v -> b h (l v)')
        x = self.conv(x)
        x = rearrange(x, 'b l h -> b (l h)')
        return self.head(x).squeeze(1)

from src.probes.pl_sk_wrapper import PLSKWrapper, PLSKBase
from torchmetrics.functional import accuracy

class PLSK_BCE(PLSKBase):
    """Using binary cross entropy."""
    def __init__(self, c_in, dropout=0, hs=32, depth=2, **kwargs):
        super().__init__(**kwargs)

        self.model = HSModel3d(c_in, hs, dropout)

    def forward(self, x):
        return self.model(x)

