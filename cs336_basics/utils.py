from collections.abc import Iterable

import torch


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter], max_l2: float, epsilon: float = 1e-6
):
    norms = torch.Tensor(
        [torch.norm(p.grad, 2) for p in parameters if p.grad is not None]
    )
    l2 = torch.norm(norms)
    if l2 >= max_l2:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad = p.grad * max_l2 / (l2 + epsilon)
