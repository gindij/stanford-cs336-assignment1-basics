from typing import Optional, Dict

import torch

from cs336_basics.transformer.causal_multihead_self_attention import (
    CausalMultiheadSelfAttention,
)
from cs336_basics.transformer.ffn import FeedForwardNetwork
from cs336_basics.transformer.functional import dropout
from cs336_basics.transformer.rmsnorm import RMSNorm


class TransformerBlock(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float,
        residual_pdrop: float,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        ep_norm: float = 1e-5,
    ):

        super().__init__()

        if weights is None:
            weights = {}

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.ep_norm = ep_norm

        self.attn = self._init_attn(d_model, num_heads, attn_pdrop, weights)
        self.ffn = self._init_ffn(d_model, d_ff, weights)
        self.ln1 = self._init_rmsnorm(d_model, ep_norm, weights, "ln1")
        self.ln2 = self._init_rmsnorm(d_model, ep_norm, weights, "ln2")

    def _init_rmsnorm(
        self,
        d_model: int,
        ep: float,
        weights: Optional[Dict[str, torch.Tensor]],
        prefix: str,
    ) -> RMSNorm:
        if weights is None:
            return RMSNorm(d_model, ep)
        rmsnorm_weights = {"weight": weights[f"{prefix}.weight"]}
        return RMSNorm(d_model=d_model, epsilon=ep, weights=rmsnorm_weights)

    def _init_ffn(
        self, d_model: int, d_ff: int, weights: Optional[Dict[str, torch.Tensor]]
    ) -> FeedForwardNetwork:
        if weights is None:
            return FeedForwardNetwork(d_model, d_ff)
        ffn_weights = {
            "w1.weight": weights["ffn.w1.weight"],
            "w2.weight": weights["ffn.w2.weight"],
        }
        return FeedForwardNetwork(d_model=d_model, d_ff=d_ff, weights=ffn_weights)

    def _init_attn(
        self,
        d_model: int,
        num_heads: int,
        attn_pdrop: float,
        weights: Dict[str, torch.Tensor],
    ) -> CausalMultiheadSelfAttention:
        if weights is None:
            return CausalMultiheadSelfAttention(
                num_heads=num_heads, d_model=d_model, attn_pdrop=attn_pdrop
            )
        attn_weights = {
            "q_proj.weight": weights["attn.q_proj.weight"],
            "k_proj.weight": weights["attn.k_proj.weight"],
            "v_proj.weight": weights["attn.v_proj.weight"],
            "output_proj.weight": weights["attn.output_proj.weight"],
        }
        return CausalMultiheadSelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            attn_pdrop=attn_pdrop,
            weights=attn_weights,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.ln1(x)
        y1 = self.attn(y1)
        y1 = dropout(y1, self.residual_pdrop, training=self.training)
        y1 = x + y1
        y2 = self.ln2(y1)
        y2 = self.ffn(y2)
        y2 = dropout(y2, self.residual_pdrop, training=self.training)
        return y2 + y1
