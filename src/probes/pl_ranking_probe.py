from pytorch_optimizer import Ranger21
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics.functional import accuracy
from einops import rearrange
from random import random as rand

from src.probes.pl_base import PLBase


class PLRankingBase(PLBase):
    """
    
    This uses SmoothL1Loss to tackle a ranking objective and does better in terms of performance and overfitting compared to MarginRanking loss, as well as setting it up to classify the direction, or estimate the distance between the pair direction with MSE.
    """
        
    def _step(self, batch, batch_idx, stage='train'):
        x0, x1, y = batch

        # # Swap sometimes, not really needed?
        # if rand()>0.5:
        #     x0, x1 = x1, x0
        #     assert y.dtype==torch.float32, f"y.dtype={y.dtype}"
        #     y = -y
        
        ypred0 = self(x0)
        ypred1 = self(x1)
        
        if stage=='pred':
            return (ypred1-ypred0).float()
        
        loss = F.smooth_l1_loss(ypred1-ypred0, y)
        
        y_cls = ypred1>ypred0 # switch2bool(ypred1-ypred0)
        self.log(f"{stage}/acc", accuracy(y_cls, y>0, "binary"), on_epoch=True, on_step=False)
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f"{stage}/n", len(y), on_epoch=True, on_step=False, reduce_fx=torch.sum)
        return loss



noop = lambda x: x

class AddCoords1d(nn.Module):
    """Add coordinates to ease position identification without modifying mean and std"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        bs, _, seq_len = x.shape
        cc = torch.linspace(-1,1,x.shape[-1], device=x.device).repeat(bs, 1, 1)
        cc = (cc - cc.mean()) / cc.std()
        x = torch.cat([x, cc], dim=1)
        return x

class ConvBlock(nn.Sequential):
    "Create a sequence of conv1d (`ni` to `nf`), activation (if `act_cls`) and `norm_type` layers."
    def __init__(self, ni, nf, kernel_size=None, ks=3, stride=1, padding='same', bias=None, 
                 act=nn.ReLU, act_kwargs={},  dropout=0., coord=False, bn=True, **kwargs):
        kernel_size = kernel_size or ks
        layers = [AddCoords1d()] if coord else []
        if bias is None: bias = not bn
        else: conv = nn.Conv1d(ni + coord, nf, kernel_size, bias=bias, stride=stride, padding=padding, **kwargs)
        act = None if act is None else act(**act_kwargs)
        layers += [conv]
        act_bn = []
        if act is not None: act_bn.append(act)
        if bn: act_bn.append(nn.BatchNorm1d(nf))
        if dropout: layers += [nn.Dropout(dropout)]
        layers += act_bn
        super().__init__(*layers)    
    
class InceptionBlock(nn.Module):
    """inspired by InceptionModulePlus
    
    https://github.com/timeseriesAI/tsai/blob/f20027e236ff06ed8fa3f5d30da5ebdcc67fe5aa/tsai/models/InceptionTimePlus.py#L24
    https://github.com/baidut/fastiqa/blob/ca287e71b7a555191b59a9e9c953be57bbe677cb/fastiqa/patchvq/model/inceptiontime.py#L21
    """
    def __init__(self, ni, nf, df=32, kernel_size=1, conv_dropout=0, stride=1, dilation=1, padding='same', coord=True, ks=[12, 7, 3], bottleneck=True, **kwargs):
        super().__init__()
        
        bottleneck = False if ni == nf else bottleneck
        self.bottleneck = ConvBlock(ni, nf, 1, coord=coord, bias=False) if bottleneck else noop # 
        self.convs = nn.ModuleList()
        for i in range(len(ks)): self.convs.append(ConvBlock(nf if bottleneck else ni, nf, ks[i], padding=padding, coord=coord,                                                          dilation=dilation**i, stride=stride, bias=False))
        self.mp_conv = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), ConvBlock(ni, nf, 1, coord=coord, bias=False)])
        # self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size, **kwargs)
        self.bn = nn.BatchNorm1d(nf*4)
        self.conv_dropout = nn.Dropout(conv_dropout)
        self.act = nn.ReLU()
        
    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(x)
        x = torch.cat([l(x) for l in self.convs] + [self.mp_conv(input_tensor)], dim=1)
        x = self.bn(x)
        x = self.conv_dropout(x)
        x = self.act(x)
        return x
    
    
class LinBnDrop(nn.Module):
    "A linear layer with `nn.BatchNorm1d` and `nn.Dropout`."
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, **kwargs):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        self.bn = nn.BatchNorm1d(n_out) if bn else noop
        self.act = nn.ReLU() if act is None else act()
        self.do = nn.Dropout(p) if p else noop

    def forward(self, x):
        x = self.bn(self.act(self.lin(x)))
        return self.do(x)
    

class PLConvProbeLinear(PLRankingBase):
    def __init__(self, c_in, epoch_steps, max_epochs, depth=0, lr=4e-3, weight_decay=1e-9, hs=64, dropout=0, **kwargs):
        super().__init__(epoch_steps=epoch_steps, max_epochs=max_epochs, lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters()
        
        
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
            # nn.Tanh(), 
        )
        
    def forward(self, x):
        if x.ndim==4:
            x = x.squeeze(3)
        x = rearrange(x, 'b l h -> b h l')
        x = self.conv(x)
        x = rearrange(x, 'b l h -> b (l h)')
        return self.head(x).squeeze(1)
    