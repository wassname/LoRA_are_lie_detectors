import einops
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange
from src.probes.pl_ranking_probe import InceptionBlock, LinBnDrop
from src.probes.pl_base import PLBase
from torchmetrics.functional import accuracy, auroc

class InceptionEncoder(nn.Module):
    def __init__(self, n_layers, n_channels, hs, c_out, ks=[7, 5, 3], dropout=0):
        super().__init__()
        self.n_layers = n_layers

        self.conv = nn.Sequential(
            nn.BatchNorm1d(n_channels, affine=False),
            InceptionBlock(n_channels, hs, ks=ks, coord=True, conv_dropout=dropout),
            InceptionBlock(hs * 4, hs, ks=ks, coord=True, conv_dropout=dropout),
            InceptionBlock(hs * 4, hs, ks=ks, conv_dropout=dropout),
            InceptionBlock(hs * 4, hs, ks=ks),
            InceptionBlock(hs * 4, hs, ks=ks),
        )

        self.fc = nn.Sequential(
            LinBnDrop(hs * 4 * n_layers, c_out * n_layers, dropout=dropout),
            nn.Linear(c_out * n_layers, c_out * n_layers),
            nn.BatchNorm1d(
                c_out * n_layers, affine=False
            ),  # center it, regularize it
        )

    def forward(self, x):
        x = self.conv(x)
        x = rearrange(x, "b c l -> b (c l)")
        x = self.fc(x)
        x = rearrange(x, "b (c l) -> b c l", l=self.n_layers)
        return x


class InceptiopnDecoder(nn.Module):
    def __init__(self, n_latent, n_layers, hs, c_out=1, ks=[7, 5, 3], dropout=0):
        super().__init__()
        hs = hs * 2
        self.layers = n_layers

        self.fc = nn.Sequential(
            # nn.BatchNorm1d(
            #     n_latent * n_layers, affine=False
            # ),  # center it, regularize it
            LinBnDrop(n_latent * n_layers, hs * n_layers, dropout=dropout),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            InceptionBlock(hs, hs, ks=ks, coord=True, conv_dropout=dropout),
            InceptionBlock(hs * 4, hs, ks=ks, conv_dropout=dropout),
            InceptionBlock(hs * 4, hs, ks=ks, coord=True),
            InceptionBlock(hs * 4, hs, ks=ks),
            nn.Conv1d(hs * 4, c_out, 1),

        )

    def forward(self, x):
        x = rearrange(x, "b l c -> b (l c)")
        x = self.fc(x)
        x = rearrange(x, "b (c l) -> b c l", l=self.layers)
        x = self.conv(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(
        self, c_in, depth=3, n_hidden=32, n_latent=32, l1_coeff: float = 1.0, dropout=0, importance_matrix=None
    ):
        super().__init__()
        self.l1_coeff = l1_coeff
        self.importance_matrix = importance_matrix
        n_layers, n_channels = c_in
        self.enc = InceptionEncoder(n_layers, n_channels, n_hidden, n_latent, dropout=dropout)
        self.dec = InceptiopnDecoder(
            n_latent, n_layers, n_hidden // 4, c_out=n_channels, dropout=dropout
        )
    #     self.apply_weight_norm(self.dec)
    #     self.apply_weight_norm(self.enc)

    # def apply_weight_norm(self, net):
    #     for m in net.modules():
    #         if isinstance(m, nn.Conv1d):
    #             # I think it's 1. In the example they use 2, but their weights are transposed before use
    #             torch.nn.utils.parametrizations.weight_norm(m, dim=1)

    def forward(self, h: Float[Tensor, "batch_size n_hidden n_channels"]):
        device = next(self.parameters()).device
        h = h.to(device)
        latent = self.enc(h)
        h_rec = self.dec(latent)

        if self.importance_matrix is None:
            importance_matrix = torch.ones_like(h_rec)
        else:
            importance_matrix = self.importance_matrix.to(device).T[None]

        # Compute loss, return values
        l2_loss = (
            ((h_rec - h) * importance_matrix).pow(2).mean(-1).mean(1)
        )  # shape [batch_size sum(neurons) mean(layers)] - punish the model for not reconstructing the input
        l1_loss = (
            latent.abs().sum(-1).sum(1)
        )  # shape [batch_size sum(latent) sum(layers)] - punish the model for large latent values
        loss = (self.l1_coeff * l1_loss + l2_loss).mean(0)  # scalar
        loss = l2_loss.mean(0)  # scalar

        return l1_loss, l2_loss, loss, latent, h_rec



def recursive_requires_grad(model, mode: bool = False):
    # print(f"requires_grad: {mode}")
    for param in model.parameters():
        param.requires_grad = mode


class PLAE(PLBase):
    def __init__(
        self,
        c_in,
        steps_per_epoch,
        max_epochs,
        depth=0,
        lr=4e-3,
        weight_decay=1e-9,
        hs=64,
        n_latent=32,
        l1_coeff=1,
        dropout=0,
        importance_matrix=None,
        **kwargs,
    ):
        super().__init__(steps_per_epoch=steps_per_epoch, max_epochs=max_epochs, lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters()

        self.ae = AutoEncoder(
            c_in,
            n_hidden=hs,
            n_latent=n_latent,
            depth=depth,
            l1_coeff=l1_coeff,
            dropout=dropout,
            importance_matrix=importance_matrix
        )
        n_layers, n_channels = c_in
        n = n_latent * n_layers
        self.head = nn.Sequential(
            LinBnDrop(n, n, bn=False),
            LinBnDrop(n, n // 4, dropout=dropout, bn=False),
            LinBnDrop(n // 4, n // 12, bn=False),
            nn.Linear(n // 12, 1),
            # nn.Tanh(),
        )
        self._ae_mode = True

    def ae_mode(self, mode=0):
        """
        mode 0, train the ae
        mode 1, train only the prob
        mode 2, train both
        """
        if mode==0:
            print('training ae')
        elif mode==1:
            print('training probe')
        elif mode==2:
            print('training both ae and probe')
        self._ae_mode = mode
        recursive_requires_grad(self.ae, mode in [0, 2])

    def forward(self, x):
        if x.ndim == 4:
            x = x.squeeze(3)
        x = rearrange(x, "b l h -> b h l")
        # if not self._ae_mode:
        #     with torch.no_grad():
        #         l1_loss, l2_loss, loss, latent, h_rec = self.ae(x)
        # else:
        l1_loss, l2_loss, loss, latent, h_rec = self.ae(x)

        latent2 = rearrange(latent, "b l h -> b (l h)")
        pred = self.head(latent2).squeeze(1)
        return dict(
            pred=pred,
            l1_loss=l1_loss,
            l2_loss=l2_loss,
            loss=loss,
            latent=latent,
            h_rec=h_rec,
        )

    def _step(self, batch, batch_idx, stage="train"):
        # if stage=='train':
        #     # Normalize the decoder weights before each optimization step (from https://colab.research.google.com/drive/1rPy82rL3iZzy2_Rd3F82RwFhlVnnroIh?usp=sharing#scrollTo=q1JctT2Pvw-r)
        #     # Presumably this is a way to implement weight norm to regularize the decoder
        #     self.normalize_decoder()
        device = next(self.parameters()).device
        x, y = batch # batch['X'], batch['y']
        x = x.to(device)
        y = y.to(device)
        x0 = x[..., 0]
        # x1 = x[..., 1]
        info0 = self(x0)
        # info1 = self(x1)
        # ypred1 = info1["pred"]
        logits = info0["pred"]
        y_probs = F.sigmoid(logits)
        y_cls = y_probs > 0.5

        if stage == "pred":
            return (y_probs).float()
        
        pred_loss = F.binary_cross_entropy_with_logits(logits, (y>0.).float())

        # pred_loss = F.smooth_l1_loss(ypred0, y)
        rec_loss = info0["loss"] 
        l1_loss = info0["l1_loss"].mean()
        l2_loss = info0["l2_loss"].mean()

        self.log(
            f"{stage}/auroc",
            auroc(y_probs, y > 0, "binary"),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            f"{stage}/acc",
            accuracy(y_cls, y > 0, "binary"),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            f"{stage}/loss_pred",
            float(pred_loss),
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        self.log(
            f"{stage}/loss_rec",
            float(rec_loss),
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        self.log(f"{stage}/l1_loss", l1_loss, on_epoch=True, on_step=False)
        self.log(f"{stage}/l2_loss", l2_loss, on_epoch=True, on_step=False)
        self.log(
            f"{stage}/n",
            float(len(y)),
            on_epoch=True,
            on_step=False,
            reduce_fx=torch.sum,
        )
        if self._ae_mode == 0:
            assert torch.isfinite(rec_loss), "rec_loss is not finite"
            return rec_loss
        elif self._ae_mode == 1:
            assert torch.isfinite(pred_loss), "pred_loss is not finite"
            return pred_loss
        elif self._ae_mode == 2:
            # , train/loss_pred_epoch=0.0195, train/loss_rec_epoch=169.0
            assert torch.isfinite(pred_loss), "pred_loss is not finite"
            assert torch.isfinite(rec_loss), "rec_loss is not finite"
            return pred_loss * 50000 + rec_loss
