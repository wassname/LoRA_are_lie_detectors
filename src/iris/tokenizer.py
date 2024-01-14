"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from jaxtyping import Float, Int
from einops import rearrange
from torch import Tensor
import torch
import torch.nn as nn

# from .lpips import LPIPS
from .nets import Encoder, Decoder

Batch = Dict[str, torch.Tensor]

class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor

Input = Float[Tensor, 'batch layers act']


class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        # self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        outputs = self.encode(x)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input)
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, x: Input, **kwargs: Any) -> LossWithIntermediateLosses:
        # observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, x_rec = self(x)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()

        reconstruction_loss = torch.abs(x - x_rec).mean()

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss)

    def encode(self, x: Input) -> TokenizerEncoderOutput:
        z, _ = self.encoder(x)
        # z = self.pre_quant_conv(z) just 1x1 z_channels -> embed_dim
        b, l, e = z.shape
        z_flattened = rearrange(z, 'b l a -> (b l) a')
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = rearrange(self.embedding(tokens), '(b l) e -> b l e', b=b, l=l).contiguous()
        tokens = rearrange(tokens, '(b l) -> b l', b=b, l=l).contiguous() # TODO add extra dim for tokens per layer

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: Float[Tensor, 'batch layers embedding_dim']) -> Input:
        rec, _ = self.decoder(z_q)
        return rec

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        z_q = self.encode(x).z_quantized
        return self.decode(z_q)

