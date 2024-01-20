from einops.layers.torch import Rearrange, Reduce
from src.iris.tokenizer import Tokenizer
from src.iris.nets import EncoderDecoderConfig, Encoder, Decoder
from src.vae.conv_inception import (
    LinBnDrop,
    PLBase,
    recursive_requires_grad,
    accuracy,
    auroc,
)
from src.vae.sae2 import AutoEncoderConfig, Encoder, Decoder, Affines
from einops import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

class PL_TAE(PLBase):
    def __init__(
        self,
        c_in,
        steps_per_epoch,
        max_epochs,
        lr=4e-3,
        weight_decay=0,
        n_latent=512,
        dropout=0,
        encoder_sizes=[32, ],
        importance_matrix=None,
        tokens_per_layer=6,
        probe_embedd_dim=3,
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

        vocab_size = n_latent
        embed_dim = n_latent


        n_layers, n_channels = c_in
        self.norm = Affines(n_layers, n_channels)
        self.ae_cfg = AutoEncoderConfig(
            n_instances=n_layers,
            n_input_ae=n_channels,
            n_hidden_ae=vocab_size * tokens_per_layer, # output size of decoder (tokens)
            encoder_sizes=encoder_sizes,
        )
        encoder = Encoder(self.ae_cfg)

        self.ae_cfg = AutoEncoderConfig(
            n_instances=n_layers,
            n_input_ae=n_channels,
            n_hidden_ae=embed_dim * tokens_per_layer, # input channel to decoder (embedded tokens)
            encoder_sizes=encoder_sizes,
        )
        decoder = Decoder(self.ae_cfg)
        

        self.ae = Tokenizer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            tokens_per_layer=tokens_per_layer,
            encoder=encoder,
            decoder=decoder,
        )

        # self.probe_embed = nn.Embedding(vocab_size, probe_embedd_dim, 
        #                                 max_norm=1.0
        #                                 )
        n = n_layers * tokens_per_layer * probe_embedd_dim
        # self.probe_embed.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

        self.only_emb = True
        
        if self.only_emb:
            self.head = nn.Sequential(
                nn.Embedding(vocab_size, probe_embedd_dim, 
                                            max_norm=1.0
                                            ),
                Reduce("b l h v -> b", "max"),
            )
        else:
            
            self.head = nn.Sequential(
                nn.Embedding(vocab_size, probe_embedd_dim, 
                                            max_norm=1.0
                                            ),
                Rearrange("b l h v -> b (l h v)"),                
                # LinBnDrop(n*embed_dim2, n, bn=True, dropout=dropout),
                # LinBnDrop(n, n, bn=True, dropout=dropout),
                # LinBnDrop(n, n, dropout=dropout, bn=False),
                # LinBnDrop(n // 4, n // 12, bn=False),
                nn.Linear(n, 1),
            )
        self._ae_mode = True

    def ae_mode(self, mode=0):
        """
        mode 0, train the ae
        mode 1, train only the prob
        mode 2, train both
        """
        if mode == 0:
            print("training ae")
        elif mode == 1:
            print("training probe")
        elif mode == 2:
            print("training both ae and probe")
        self._ae_mode = mode
        recursive_requires_grad(self.ae, mode in [0, 2])

    def forward(self, x):
        x = self.norm(x)
        o = self.ae.encode(x)
        z, z_q, tokens = o.z, o.z_quantized, o.tokens
        h_rec = self.ae.decode(z_q)
        h_rec = self.norm.inv(h_rec)
        losses = self.ae.compute_loss(x)
        loss = losses.loss_total

        # we want a probe tokenizer. reusing the decoder hurts performance
        # latent_decoder = self.ae.decode(z_q)
        latent_probe = self.head[0](tokens)
        pred = self.head(tokens)#.squeeze(1)
        return dict(
            pred=pred,
            loss=loss,
            latent_probe=latent_probe,
            z=z, 
            z_q=z_q,
            tokens=tokens,
            h_rec=h_rec,
            **losses.intermediate_losses
        )


    def _step(self, batch, batch_idx, stage="train"):
        y_thresh = 0.5
        device = next(self.parameters()).device
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        x0 = x  # [..., 0]
        info0 = self(x0)
        logits = info0["pred"]
        
        
        # if using head
        if self.only_emb:
            y_probs = logits
        else:
            y_probs = F.sigmoid(logits)
        
        assert (y_probs >= 0).all(), "y_probs should be positive"
        assert (y_probs <= 1).all(), "y_probs should be less than 1"
        
        y_cls = y_probs > 0.5

        if stage == "pred":
            return (y_probs).float()

        pred_loss = F.binary_cross_entropy_with_logits(y_probs, (y > y_thresh).float())

        pred_loss = F.smooth_l1_loss(y_probs, y)
        rec_loss = info0["loss"]

        if self._ae_mode in [1, 2]:
            self.log(
                f"{stage}/auroc",
                auroc(y_probs, y > y_thresh, "binary"),
                on_epoch=True,
                on_step=False,
            )
            self.log(
                f"{stage}/acc",
                accuracy(y_cls, y > y_thresh, "binary"),
                on_epoch=True,
                on_step=False,
            )
            self.log(
                f"{stage}/loss_pred",
                float(pred_loss),
                on_epoch=True,
                on_step=True,
            )
        if self._ae_mode in [0, 2]:
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
