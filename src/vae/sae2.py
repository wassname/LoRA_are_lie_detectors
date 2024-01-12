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

from jaxtyping import Float
from typing import Optional, Union, Callable

@dataclass
class AutoEncoderConfig:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as
    # kinda like a batch dimension, but one which is built into our training setup.
    n_instances: int
    # this is the hidden states, and the latent size
    n_input_ae: int
    n_hidden_ae: int
    l1_coeff: float = 0.5
    tied_weights: bool = False
    depth: int = 1


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
    def __init__(self, n_instances: int, n_input_ae: int, n_output: int, weight_norm_dim=None):
        super().__init__()
        self.linears = nn.ModuleList(NormedLinear(n_input_ae, n_output, norm_dim=weight_norm_dim) for _ in range(n_instances))

    def weight_norm(self) -> None:
        for m in self.linears:
            m.weight_norm()

    def forward(self, x: Tensor) -> Tensor:
        return t.stack([m(x[:, i]) for i, m in enumerate(self.linears)], dim=1)
    

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
        self.affines = nn.ModuleList(AffineInstanceNorm1d(n_input_ae) for _ in range(n_instances))

    def forward(self, x: Float[Tensor, "batch_size n_instances n_hidden"]) -> Tensor:
        return t.stack([m(x[:, i]) for i, m in enumerate(self.affines)], dim=1)

    def inv(self, x: Float[Tensor, "batch_size n_instances n_hidden"]) -> Tensor:
        return t.stack([m.inv(x[:, i]) for i, m in enumerate(self.affines)], dim=1)


class AutoEncoder(nn.Module):

    def __init__(self, cfg: AutoEncoderConfig, importance_matrix: Float[Tensor, "n_instances n_input_ae"] = None):
        super().__init__()
        self.cfg = cfg
        self.importance_matrix = importance_matrix

        # instead of a tied bias, we use a batch norm type module to track and adjust for the training mean and std. We also have an inverse function to undo the normalization.
        self.norm = Affines(cfg.n_instances, cfg.n_input_ae)

        self.encoder = []
        for i in range(cfg.depth):
            self.encoder.append(NormedLinears(cfg.n_instances, cfg.n_input_ae, cfg.n_hidden_ae))
            self.encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = []
        for i in range(cfg.depth):
            self.decoder.append(NormedLinears(cfg.n_instances, cfg.n_hidden_ae, cfg.n_input_ae, weight_norm_dim=0))
            if i<cfg.depth-1:
                self.decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, h: Float[Tensor, "batch_size n_instances n_hidden"]):

        # Compute activations
        h_cent = self.norm(h)
        acts = self.encoder(h_cent)
        h_reconstructed = self.norm.inv(self.decoder(acts))

        # Compute loss, return values
        h_err = h_reconstructed - h
        if self.importance_matrix is not None:
            importance_matrix = self.importance_matrix[None, : ].to(h_err.device)
            h_err = h_err * importance_matrix
        l2_loss = h_err.pow(2).sum(2).sum(1) # shape [batch_size n_instances features]
        l1_loss = acts.abs().sum(2).sum(1) # shape [batch_size n_instances n_latent]
        loss = (self.cfg.l1_coeff * l1_loss + l2_loss).mean(0) # scalar

        return l1_loss, l2_loss, loss, acts, h_reconstructed

    @t.no_grad()
    def normalize_decoder(self) -> None:
        '''
        Normalizes the decoder weights to have unit norm. If using tied weights, we we assume W_enc is used for both.
        '''
        for m in self.decoder:
            if isinstance(m, NormedLinears):
                m.weight_norm()
