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
from src.vae.sae2 import AutoEncoderConfig, Encoder, Decoder, Affines


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
    def __init__(self, vocab_size: int, embed_dim: int, tokens_per_layer:int , encoder: Encoder, decoder: Decoder, norm: Affines, importance_matrix=None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.tokens_per_layer = tokens_per_layer
        self.importance_matrix = importance_matrix

        self.encoder = encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder = decoder
        self.norm = norm
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.norm(x)
        outputs = self.encode(x)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input)
        reconstructions = self.norm.inv(reconstructions)
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, x: Input, **kwargs: Any) -> LossWithIntermediateLosses:
        z, z_quantized, x_rec = self(x)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()

        reconstruction_loss = torch.abs(x - x_rec)
        if self.importance_matrix is not None:
            reconstruction_loss = reconstruction_loss * self.importance_matrix
        reconstruction_loss = reconstruction_loss.mean()

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss)

    def encode(self, x: Input) -> TokenizerEncoderOutput:
        z, _ = self.encoder(x)

        z = rearrange(z, 'b l (a v) -> b l a v', v=self.tokens_per_layer)
        b, l, e, v = z.shape
        z_flattened = rearrange(z, 'b l a v -> (b l v) a')
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = rearrange(self.embedding(tokens), '(b l v) e -> b l e v', b=b, l=l).contiguous()
        tokens = rearrange(tokens, '(b l v) -> b l v', b=b, l=l, v=v).contiguous() # TODO add extra dim for tokens per layer

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: Float[Tensor, 'batch layers embedding_dim']) -> Input:
        z_q = rearrange(z_q, 'b l e v -> b l (e v)', v=self.tokens_per_layer)
        rec, _ = self.decoder(z_q)
        return rec

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        z_q = self.encode(x).z_quantized
        return self.decode(z_q)



class TokenizedAutoEncoder(nn.Module):
    def __init__(self, c_in, vocab_size:int=512, embed_dim:int=512, tokens_per_layer:int=4, encoder_sizes=[32,32], importance_matrix=None,) -> None:
        super().__init__()
        n_instances, n_input_ae = c_in
        self.ae_cfg = AutoEncoderConfig(
            n_instances=n_instances,
            n_input_ae=n_input_ae,
            n_hidden_ae=vocab_size * tokens_per_layer, # output size of decoder (tokens)
            encoder_sizes=encoder_sizes,
        )
        encoder = Encoder(self.ae_cfg)

        self.ae_cfg = AutoEncoderConfig(
            n_instances=n_instances,
            n_input_ae=n_input_ae,
            n_hidden_ae=embed_dim * tokens_per_layer, # input channel to decoder (embedded tokens)
            encoder_sizes=encoder_sizes,
        )
        decoder = Decoder(self.ae_cfg)
        norm = Affines(n_instances, n_input_ae)

        self.tokenizer = Tokenizer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            tokens_per_layer=tokens_per_layer,
            encoder=encoder,
            decoder=decoder,
            norm=norm,
            importance_matrix=importance_matrix,
        )
        
    def forward(self, x):
        o = self.tokenizer.encode(x)
        z, z_q, tokens = o.z, o.z_quantized, o.tokens
        h_rec = self.tokenizer.decode(z_q)
        losses = self.tokenizer.compute_loss(x)
        loss = losses.loss_total
        
        return dict(
            loss=loss,
            z=z, 
            z_q=z_q,
            tokens=tokens,
            h_rec=h_rec,
            **losses.intermediate_losses
        )
