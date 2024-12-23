from typing import Optional, Dict

import torch

from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.functional import dropout
from cs336_basics.transformer.transformer_block import TransformerBlock


class TransformerLM(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float,
        residual_pdrop: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        ep_norm: float = 1e-5,
        weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.residual_pdrop = residual_pdrop

        self.token_embeddings = torch.nn.Parameter(
            torch.randn(vocab_size, d_model)
            if weights is None
            else weights["token_embeddings.weight"]
        )
        self.position_embeddings = torch.nn.Parameter(
            torch.randn(context_length, d_model)
            if weights is None
            else weights["position_embeddings.weight"]
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    attn_pdrop=attn_pdrop,
                    residual_pdrop=residual_pdrop,
                    ep_norm=ep_norm,
                    weights=(
                        weights
                        if weights is None
                        else {
                            k.replace(f"layers.{i}.", ""): v for k, v in weights.items()
                        }
                    ),
                )
                for i in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(
            d_model=d_model,
            epsilon=ep_norm,
            weights=(
                weights if weights is None else {"weight": weights["ln_final.weight"]}
            ),
        )
        self.lm_head = torch.nn.Parameter(
            torch.randn(d_model, vocab_size)
            if weights is None
            else weights["lm_head.weight"].T
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        tok_embs = self.token_embeddings[x]
        x_pos = torch.arange(seq_len).repeat(batch_size, 1)
        pos_embs = self.position_embeddings[x_pos]
        emb_sum = tok_embs + pos_embs
        xx = dropout(emb_sum, self.residual_pdrop)
        for tblock in self.transformer_blocks:
            xx = tblock(xx)
        xx = self.ln_final(xx)
        logits = torch.matmul(xx, self.lm_head)
        return logits
