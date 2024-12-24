from typing import Dict, Optional

import torch

from cs336_basics.transformer.functional import attention


def generate_causal_attn_mask(seq_len: int) -> torch.Tensor:
    return torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()


class CausalMultiheadSelfAttention(torch.nn.Module):

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        attn_pdrop: Optional[float] = 0.0,
        weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.dv = d_model // num_heads
        self.attn_pdrop = attn_pdrop

        if weights is not None:
            # handle the different ways the weights are passed in different tests
            if "q_heads.0.weight" in weights:
                # Combine weights per head into single matrices
                q_matrices = [
                    weights[f"q_heads.{i}.weight"].T for i in range(num_heads)
                ]
                k_matrices = [
                    weights[f"k_heads.{i}.weight"].T for i in range(num_heads)
                ]
                v_matrices = [
                    weights[f"v_heads.{i}.weight"].T for i in range(num_heads)
                ]

                self.q_proj = torch.nn.Parameter(torch.cat(q_matrices, dim=1))
                self.k_proj = torch.nn.Parameter(torch.cat(k_matrices, dim=1))
                self.v_proj = torch.nn.Parameter(torch.cat(v_matrices, dim=1))
            else:
                self.q_proj = torch.nn.Parameter(weights["q_proj.weight"].T)
                self.k_proj = torch.nn.Parameter(weights["k_proj.weight"].T)
                self.v_proj = torch.nn.Parameter(weights["v_proj.weight"].T)
        else:
            # Initialize as single matrices
            self.q_proj = torch.nn.Parameter(torch.randn(self.d_model, self.d_model))
            self.k_proj = torch.nn.Parameter(torch.randn(self.d_model, self.d_model))
            self.v_proj = torch.nn.Parameter(torch.randn(self.d_model, self.d_model))

        self.output_proj = torch.nn.Parameter(
            weights["output_proj.weight"].T
            if weights is not None
            else torch.randn(self.d_model, self.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = (
            torch.matmul(x, self.q_proj)
            .view(batch_size, seq_len, self.num_heads, self.dk)
            .transpose(1, 2)
        )
        k = (
            torch.matmul(x, self.k_proj)
            .view(batch_size, seq_len, self.num_heads, self.dk)
            .transpose(1, 2)
        )
        v = (
            torch.matmul(x, self.v_proj)
            .view(batch_size, seq_len, self.num_heads, self.dv)
            .transpose(1, 2)
        )
        mask = generate_causal_attn_mask(seq_len)
        attn_output = attention(
            q, k, v, mask=mask, pdrop=self.attn_pdrop, training=self.training
        )
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        return torch.matmul(attn_output, self.output_proj)
