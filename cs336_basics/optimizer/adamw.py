from collections.abc import Iterable, Callable
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[Dict[str, Any]],
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.95,
        device: str = "cpu",
    ):
        defaults = {"lr": lr}
        super().__init__(params, defaults)

        self.beta1, self.beta2 = betas
        self.ep = eps
        self.weight_decay = weight_decay
        self.device = device

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                t = state.get("t", 1)
                m, v = state.get(
                    "m", torch.zeros(grad.shape).to(self.device)
                ), state.get("v", torch.zeros(grad.shape).to(self.device))

                m = self.beta1 * m + (1 - self.beta1) * grad
                mhat = m / (1 - self.beta1**t)
                v = self.beta2 * v + (1 - self.beta2) * grad**2
                vhat = v / (1 - self.beta2**t)

                p.data = p.data * (1 - lr * self.weight_decay)
                p.data = p.data - lr * mhat / (torch.sqrt(vhat) + self.ep)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
                self.state[p] = state
        return loss
