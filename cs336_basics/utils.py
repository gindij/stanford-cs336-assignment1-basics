from collections.abc import Iterable
import os
from typing import BinaryIO, IO, Union

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


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    torch.save(
        obj={
            "iteration": iteration,
            "model": model_state,
            "optimizer": optimizer_state,
        },
        f=out,
    )


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iteration"]
