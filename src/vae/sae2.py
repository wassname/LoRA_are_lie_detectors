"""
This Sparse AutoEncoder was modified from
- [Callum McDougall's work](https://github.com/callummcdougall/sae-exercises-mats/blob/116ecf3f8f7ffcd66cf628518009f81989e57bac/solutions.py#L692)
- [ai-safety-foundation](https://github.com/ai-safety-foundation/sparse_autoencoder)

Aims:
- try to reuse pytorch layers where possible. 
- I also used a batch norm type setup instead of a tied bias and geometric mean, this reduces the hyperparameters and setup.
- Enable multiple layers... because why not
"""
import torch as t
from torch import nn, Tensor
from torch.nn import functional as F
from dataclasses import dataclass
import einops
import numpy as np

from jaxtyping import Float
from typing import Optional, Union, Callable, Tuple


@dataclass
class AutoEncoderConfig:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as
    # kinda like a batch dimension, but one which is built into our training setup.
    n_instances: int
    # this is the hidden states, and the latent size
    n_input_ae: int
    # output size
    n_hidden_ae: int
    l1_coeff: float = 0.5

    # list of layer sizes, placed in between the input and output
    encoder_sizes: Optional[Tuple] = tuple()


class NormedLinear(nn.Linear):
    def __init__(self, *args, norm_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_dim = norm_dim
        self.weight_norm()

    @t.no_grad
    def weight_norm(self) -> None:
        # Note in callums they norm by input (dim=0), and in ai-saftey they norm by output. I'm not sure which is correct but Callum's seemed to perform better.
        if self.norm_dim is not None:
            F.normalize(self.weight, dim=self.norm_dim, out=self.weight)


class NormedLinears(nn.Module):
    def __init__(
        self,
        n_instances: int,
        n_input_ae: int,
        n_output: int,
        weight_norm_dim=None,
        act: Callable = nn.ReLU(),
        bias=True,
    ):
        super().__init__()
        self.linears = nn.ModuleList(
            NormedLinear(n_input_ae, n_output, norm_dim=weight_norm_dim, bias=bias)
            for _ in range(n_instances)
        )
        self.act = act

    def weight_norm(self) -> None:
        for m in self.linears:
            m.weight_norm()

    def forward(self, x: Tensor) -> Tensor:
        x = t.stack([m(x[:, i]) for i, m in enumerate(self.linears)], dim=1)
        if self.act is not None:
            x = self.act(x)
        return x
    
    def __rep__(self):
        return f"NormedLinears({self.linears[0]})"


class AffineInstanceNorm1d(nn.BatchNorm1d):
    """
    A Batch norm which can be inverted
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, affine=False, track_running_stats=True, **kwargs)

    def inv(self, x: Float[Tensor, "batch_size n_hidden"]):
        return x * self.running_var + self.running_mean


class Affines(nn.Module):
    """
    Multiple batch norms.

    We use this to track and adjust for data for the mean and std of the training dataset
    """

    def __init__(self, n_instances: int, n_input_ae: int):
        super().__init__()
        self.affines = nn.ModuleList(
            AffineInstanceNorm1d(n_input_ae) for _ in range(n_instances)
        )

    def forward(self, x: Float[Tensor, "batch_size n_instances n_hidden"]) -> Tensor:
        return t.stack([m(x[:, i]) for i, m in enumerate(self.affines)], dim=1)

    def inv(self, x: Float[Tensor, "batch_size n_instances n_hidden"]) -> Tensor:
        return t.stack([m.inv(x[:, i]) for i, m in enumerate(self.affines)], dim=1)
    
from einops.layers.torch import Rearrange, Reduce

class Encoder(nn.Module):
    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()
        self.cfg = cfg

        encoder_sizes = [cfg.n_input_ae] + cfg.encoder_sizes + [cfg.n_hidden_ae]
        depth = len(encoder_sizes)

        self.encoder = []
        for i in range(depth - 2):
            self.encoder.append(
                NormedLinears(cfg.n_instances, encoder_sizes[i], encoder_sizes[i + 1], 
                              bias=(i>1),
                              )
            )
        
        # mixer: mix the layers
        self.encoder += [
            Rearrange('b l f -> b (l f)'),
            nn.Linear(
                encoder_sizes[-2]*cfg.n_instances,
                encoder_sizes[-2]*cfg.n_instances,
            ),
            Rearrange('b (l f) -> b l f', l=cfg.n_instances),
        ]
            

        # final layer
        self.encoder.append(
            NormedLinears(
                cfg.n_instances,
                encoder_sizes[-2],
                cfg.n_hidden_ae,
                bias=True,
                act=None,
            )
        )
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x):
        layer_outputs = []
        for i, m in enumerate(self.encoder):
            x = m(x)
            layer_outputs.append(x)
        return x, layer_outputs
    

class Decoder(nn.Module):
    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()
        self.cfg = cfg
        encoder_sizes = [cfg.n_input_ae] + cfg.encoder_sizes + [cfg.n_hidden_ae]
        decoder_sizes = encoder_sizes[::-1]
        depth = len(decoder_sizes)

        self.decoder = []
        for i in range(depth - 2):
            self.decoder.append(
                NormedLinears(
                    cfg.n_instances,
                    decoder_sizes[i],
                    decoder_sizes[i + 1],
                    weight_norm_dim=0,
                )
            )

        # mixer: mix the layers
        self.decoder += [
            Rearrange('b l f -> b (l f)'),
            nn.Linear(
                decoder_sizes[-2]*cfg.n_instances,
                decoder_sizes[-2]*cfg.n_instances,
            ),
            Rearrange('b (l f) -> b l f', l=cfg.n_instances),
        ]

        # final layer
        self.decoder.append(
            NormedLinears(
                cfg.n_instances,
                decoder_sizes[-2],
                cfg.n_input_ae,
                bias=True,
                act=None,
            )
        )

        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        layer_outputs = []
        for i, m in enumerate(self.decoder):
            x = m(x)
            layer_outputs.append(x)
        return x, layer_outputs


class AutoEncoder(nn.Module):
    def __init__(
        self,
        cfg: AutoEncoderConfig,
        importance_matrix: Float[Tensor, "n_instances n_input_ae"] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.importance_matrix = importance_matrix

        # instead of a tied bias, we use a batch norm type module to track and adjust for the training mean and std. We also have an inverse function to undo the normalization.
        self.norm = Affines(cfg.n_instances, cfg.n_input_ae)

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)


    def forward(self, h: Float[Tensor, "batch_size n_instances n_hidden"]):
        # Compute activations
        depth = len(self.cfg.encoder_sizes)
        x = self.norm(h)

        # We're trying gradual sparsity. Where the middle layer has the full sparsity, but the outer layers have less. This should encourage the model to learn increasing sparse representations.
        l1_losses = []
        l1_raw = []
        for i, m in enumerate(self.encoder):
            x = m(x)
            l1 = calc_hoyer(x, pow=1.5)
            l1_raw.append(l1)
            l1_losses.append (
                l1 / (depth - i + 1) ** 2
            )
        latent = x

        for i, m in enumerate(self.decoder):
            x = m(x)
            if i<depth-1:
                l1 = calc_hoyer(x, pow=1.5)
                l1_raw.append(l1)
                l1_losses.append(l1 / (i + 2) ** 2)
        h_reconstructed = self.norm.inv(x)

        # Mean over layers
        l1_loss = t.stack(l1_losses, 1).mean(1)

        # h_reconstructed = self.norm.inv(self.decoder(acts))

        # Compute loss, return values
        h_err = h_reconstructed - h
        if self.importance_matrix is not None:
            importance_matrix = self.importance_matrix[None, :].to(h_err.device)
            h_err = h_err * importance_matrix
        l2_loss = calc_hoyer(h_err, pow=2)  # shape [batch_size n_instances features]
        # l1_loss = acts.abs().mean(2).sum(1) # shape [batch_size n_instances n_latent]
        loss = (self.cfg.l1_coeff * l1_loss + l2_loss)
        # print(l1_loss, l2_loss)
        loss = einops.reduce(loss, 'b -> ', 'mean')

        l1_losses = t.tensor([l1.mean().item() for l1 in l1_losses])
        l1_raw = t.tensor( [l1.mean().item() for l1 in l1_raw])
        return (
            loss,
            latent,
            h_reconstructed,
            dict(l1_losses=l1_losses, l1_loss=l1_loss, l2_loss=l2_loss, l1_raw=l1_raw),
        )

    @t.no_grad()
    def normalize_decoder(self) -> None:
        """
        Normalizes the decoder weights to have unit norm. If using tied weights, we we assume W_enc is used for both.
        """
        for m in self.decoder:
            if isinstance(m, NormedLinears):
                m.weight_norm()


def calc_l1_loss(x: Float[Tensor, "batch_size n_instances n_latent"]):
    """
    Not we chose to sum over the latent and feature, even thought they have differen't dimensions. This is because we want the total information stored
    """
    return einops.reduce(x.abs(), 'b n l -> b n', 'sum')

def calc_l2_loss(h_err: Float[Tensor, "batch_size n_instances features"]):
    return einops.reduce(h_err.pow(2), 'b n f -> b n', 'sum')

def calc_hoyer(x, pow=2):
    """
    https://github.com/yanghr/DeepHoyer
    """
    a = x.pow(pow).sum(1).pow(1/pow).sum(1) 
    b = x.pow(pow).sum(2).pow(1/pow).sum(1)
    d = x.pow(pow).sum(2).sum(1).pow(1/pow)
    return (a+b)/d
