from einops.layers.torch import Rearrange, Reduce
from einops import reduce
from src.iris.tokenizer import Tokenizer
from src.iris.nets import EncoderDecoderConfig, Encoder, Decoder
from src.vae.conv_inception import (
    LinBnDrop,
    PLBase,
    recursive_requires_grad,
)
from torchmetrics.functional import accuracy, auroc
from src.vae.sae2 import AutoEncoderConfig, Encoder, Decoder, Affines
from src.iris.tokenizer import TokenizedAutoEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC

class PL_TAE(PLBase):
    def __init__(
        self,
        c_in,
        steps_per_epoch,
        max_epochs,
        lr=4e-3,
        weight_decay=0,
        embed_dim=512,
        vocab_size=2048,
        encoder_sizes=[32, ],
        importance_matrix=None,
        tokens_per_layer=6,
        **kwargs,
    ):
        super().__init__(
            steps_per_epoch=steps_per_epoch,
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.save_hyperparameters()
        
        self.importance_matrix = importance_matrix

        self.tae = TokenizedAutoEncoder(
            c_in,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            tokens_per_layer=tokens_per_layer,
            encoder_sizes=encoder_sizes,
            importance_matrix=importance_matrix,
        )

    def forward(self, x):
        o = self.tae.tokenizer.encode(x)
        z, z_q, tokens = o.z, o.z_quantized, o.tokens
        h_rec = self.tae.tokenizer.decode(z_q)
        losses = self.tae.tokenizer.compute_loss(x)
        loss = losses.loss_total
        
        return dict(
            loss=loss,
            z=z, 
            z_q=z_q,
            tokens=tokens,
            h_rec=h_rec,
            **losses.intermediate_losses
        )


    def _step(self, batch, batch_idx, stage="train"):
        device = next(self.parameters()).device
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        x0 = x  # [..., 0]
        info0 = self(x0)

        rec_loss = info0["loss"]

        self.log(
            f"{stage}/loss_rec",
            float(rec_loss),
            on_epoch=True,
            on_step=True,
        )
        self.log(f"{stage}/commitment_loss", info0['commitment_loss'], on_epoch=True, on_step=False)
        self.log(f"{stage}/reconstruction_loss", info0['reconstruction_loss'], on_epoch=True, on_step=False)
        self.log(
            f"{stage}/n",
            float(len(y)),
            on_epoch=True,
            on_step=False,
            reduce_fx=torch.sum,
        )
        
        assert torch.isfinite(rec_loss), "rec_loss is not finite"
        return rec_loss



class PL_TAEProbeBase(PLBase):
    def __init__(
        self,
        tae: TokenizedAutoEncoder,
        steps_per_epoch,
        max_epochs,
        lr=4e-3,
        weight_decay=0,
        **kwargs,
    ):
        super().__init__(
            steps_per_epoch=steps_per_epoch,
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.save_hyperparameters(ignore=["tae"])
        self.tae = tae
        
        # extract config from tokenized autoencoder
        # tokens_per_layer = tae.tokenizer.tokens_per_layer
        # vocab_size = tae.tokenizer.vocab_size
        # n_layers, n_channels = c_in
        # n_feats = n_layers * tokens_per_layer * vocab_size
        
        self.head = None
        
        recursive_requires_grad(tae, False)
        
        self.auroc_fn = {k:AUROC(task="binary") for k in ["train", "val", "test", "ood"]}
        
        

    def forward(self, x):
        with torch.no_grad():
            o = self.tae.tokenizer.encode(x)
        z, z_q, tokens = o.z, o.z_quantized, o.tokens
        logits = self.head(z_q)
        y_probs = F.sigmoid(logits)
        return y_probs


    def _step(self, batch, batch_idx, stage="train"):
        y_thresh = 0.5
        device = next(self.parameters()).device
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        x0 = x  # [..., 0]
        
        y_probs = self(x0)
        
        assert torch.isfinite(y_probs).all(), 'no nans'
        assert (y_probs >= 0).all(), "y_probs should be positive"
        assert (y_probs <= 1).all(), "y_probs should be less than 1"

        if stage == "pred":
            return y_probs

        # pred_loss = F.binary_cross_entropy_with_logits(y_probs, (y > 0.5).float())
        # OR
        pred_loss = F.smooth_l1_loss(y_probs, y)
        
        m = self.auroc_fn[stage](y_probs, y > y_thresh)
        self.log('train_acc_step', m)
        
        self.log(
            f"{stage}/auroc",
            auroc(y_probs, y > y_thresh, "binary"),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            f"{stage}/loss_pred",
            float(pred_loss),
            on_epoch=True,
            on_step=True,
        )
        self.log(
            f"{stage}/n",
            float(len(y)),
            on_epoch=True,
            on_step=False,
            reduce_fx=torch.sum,
        )
        assert torch.isfinite(pred_loss), "pred_loss is not finite"
        return pred_loss
