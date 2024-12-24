from typing import Optional

import numpy as np
import torch


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / np.sqrt(2)))  #


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    xmax = torch.max(x, dim, keepdim=True).values
    xadj = x - xmax
    xexp = torch.exp(xadj)
    return xexp / torch.sum(xexp, dim=dim, keepdim=True)  #


def dropout(x: torch.Tensor, p: float, training: bool = True):
    if not training:
        return x
    return x * torch.bernoulli(torch.ones_like(x) * (1 - p))


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    pdrop: Optional[float] = None,
    training: Optional[bool] = True,
) -> torch.Tensor:
    seq_len, dk = k.shape[-2], k.shape[-1]
    a = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(dk)
    if mask is None:
        mask = torch.zeros((seq_len, seq_len)).bool()
    if training and (pdrop is not None and pdrop > 0.0):
        mask = dropout(mask, pdrop)
    a[..., mask.bool()] = -torch.inf
    return torch.matmul(softmax(a, dim=-1), v)


def cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, reduce: str = "mean"
) -> torch.Tensor:
    # logits is [batch_size * seq_len, vocab_size]
    # targets is [batch_size * seq_len]
    Dm, _ = logits.shape
    logits -= torch.max(logits, dim=1, keepdim=True).values
    true_logit = logits[torch.arange(Dm), targets]
    log_sum_exp = torch.logsumexp(logits, dim=1, keepdim=True)
    cross_entropies = -true_logit + log_sum_exp
    if reduce == "mean":
        return torch.mean(-true_logit + log_sum_exp)
    if reduce == "none":
        return cross_entropies
    raise ValueError(f"unknown reduction {reduce}")


def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    cross_entropies = cross_entropy(logits, targets, reduce="none")
    return torch.exp(torch.mean(cross_entropies))
