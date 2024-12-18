from typing import Optional

import numpy as np
import torch


def gelu(x: torch.FloatTensor) -> torch.FloatTensor:
    return 0.5 * x * (1.0 + torch.erf(x / np.sqrt(2)))  # type: ignore


def softmax(x: torch.FloatTensor, dim: int) -> torch.FloatTensor:
    xmax = torch.max(x, dim, keepdim=True).values
    xadj = x - xmax
    xexp = torch.exp(xadj)
    return xexp / torch.sum(xexp, dim=dim, keepdim=True)  # type: ignore


def attention(
    q: torch.FloatTensor,
    k: torch.FloatTensor,
    v: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    seq_len, dk = k.shape[-2], k.shape[-1]
    a = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(dk)
    if mask is None:
        mask = torch.zeros((seq_len, seq_len)).bool()
    if pdrop is not None and pdrop > 0.0:
        dropout_mask = torch.bernoulli(torch.ones_like(mask) * pdrop).bool()
        mask = torch.logical_or(mask, dropout_mask)
    a[..., mask] = -torch.inf
    return torch.matmul(softmax(a, dim=-1), v)  # type: ignore
