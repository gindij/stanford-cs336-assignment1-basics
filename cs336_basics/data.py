import random
from typing import Tuple

import numpy as np
import torch


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    rng: random.Random = random.Random(),
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(x)
    start = torch.randint(0, n - context_length - 1, (batch_size, 1))
    offset = torch.arange(context_length)
    all_indices = start + offset  # broadcasting
    x_tensor = torch.from_numpy(x).to(device)
    x_batch = x_tensor[all_indices]  # .to(device)
    y_batch = x_tensor[all_indices + 1]  # .to(device)
    return x_batch, y_batch
