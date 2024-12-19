import numpy as np


def get_cosine_annealing_lr(
    t: int, lr_min: float, lr_max: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    if t < warmup_iters:
        return lr_max * t / warmup_iters
    if t > cosine_cycle_iters:
        return lr_min
    return lr_min + 0.5 * (
        1 + np.cos(np.pi * (t - warmup_iters) / (cosine_cycle_iters - warmup_iters))
    ) * (lr_max - lr_min)
