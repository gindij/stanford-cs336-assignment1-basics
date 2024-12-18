from typing import Dict, Optional

import torch

from cs336_basics.transformer.functional import attention


def generate_causal_attn_mask(seq_len: int) -> torch.Tensor:
    return torch.triu(torch.ones((seq_len, seq_len)).bool(), diagonal=1)


# class CausalMultiheadSelfAttention(torch.nn.Module):

#     def __init__(
#         self,
#         num_heads: int,
#         d_model: int,
#         attn_pdrop: Optional[float] = 0.0,
#         weights: Optional[Dict[str, torch.Tensor]] = None,
#     ):
#         super().__init__()

#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.dk = d_model // num_heads
#         self.dv = d_model // num_heads
#         self.attn_pdrop = attn_pdrop

#         self.wq = self._init_weights(weights, "q_heads")
#         self.wk = self._init_weights(weights, "k_heads")
#         self.wv = self._init_weights(weights, "v_heads")
#         self.wo = self._init_weights(weights, "output_proj")

#     def _init_weights(
#         self, weights: Optional[Dict[str, torch.Tensor]], prefix: str
#     ) -> torch.nn.Parameter:
#         if weights is None:
#             return torch.nn.Parameter(
#                 torch.randn((self.d_model, self.d_model))
#             )  # type: ignore
#         if prefix == "output_proj":
#             return torch.nn.Parameter(weights["output_proj.weight"].T)  # type: ignore
#         matching_keys = [key for key in weights if key.startswith(prefix)]
#         weight_matrices = [weights[key].T for key in sorted(matching_keys)]
#         return torch.nn.Parameter(torch.concat(weight_matrices, dim=1))  # type: ignore

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size, seq_len, _ = x.shape
#         q = (
#             torch.matmul(x, self.wq)
#             .view(batch_size, seq_len, self.num_heads, self.dk)
#             .transpose(1, 2)
#         )
#         k = (
#             torch.matmul(x, self.wk)
#             .view(batch_size, seq_len, self.num_heads, self.dk)
#             .transpose(1, 2)
#         )
#         v = (
#             torch.matmul(x, self.wv)
#             .view(batch_size, seq_len, self.num_heads, self.dv)
#             .transpose(1, 2)
#         )
#         mask = generate_causal_attn_mask(seq_len)
#         attn_output = attention(q, k, v, mask=mask, pdrop=self.attn_pdrop)
#         attn_output = (
#             attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
#         )
#         return torch.matmul(attn_output, self.wo)


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
            # Combine weights per head into single matrices
            q_matrices = [weights[f"q_heads.{i}.weight"].T for i in range(num_heads)]
            k_matrices = [weights[f"k_heads.{i}.weight"].T for i in range(num_heads)]
            v_matrices = [weights[f"v_heads.{i}.weight"].T for i in range(num_heads)]

            self.wq = torch.nn.Parameter(torch.cat(q_matrices, dim=1))
            self.wk = torch.nn.Parameter(torch.cat(k_matrices, dim=1))
            self.wv = torch.nn.Parameter(torch.cat(v_matrices, dim=1))
        else:
            # Initialize as single matrices
            self.wq = torch.nn.Parameter(torch.randn(self.d_model, self.d_model))
            self.wk = torch.nn.Parameter(torch.randn(self.d_model, self.d_model))
            self.wv = torch.nn.Parameter(torch.randn(self.d_model, self.d_model))

        self.wo = torch.nn.Parameter(
            weights["output_proj.weight"].T
            if weights is not None
            else torch.randn(self.d_model, self.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = (
            torch.matmul(x, self.wq)
            .view(batch_size, seq_len, self.num_heads, self.dk)
            .transpose(1, 2)
        )
        k = (
            torch.matmul(x, self.wk)
            .view(batch_size, seq_len, self.num_heads, self.dk)
            .transpose(1, 2)
        )
        v = (
            torch.matmul(x, self.wv)
            .view(batch_size, seq_len, self.num_heads, self.dv)
            .transpose(1, 2)
        )
        mask = generate_causal_attn_mask(seq_len)
        attn_output = attention(q, k, v, mask=mask, pdrop=self.attn_pdrop)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        return torch.matmul(attn_output, self.wo)
