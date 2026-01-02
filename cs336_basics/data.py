from typing import Tuple

import numpy as np
import torch


def get_batch(
    x: np.ndarray,  # Keep as np.ndarray for as long as possible
    batch_size: int,
    context_length: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(x)
    ix = np.random.randint(0, n - context_length, (batch_size,))
    data_stack = np.stack([x[i : i + context_length + 1] for i in ix])
    data_tensor = torch.from_numpy(data_stack).to(device)
    x_batch = data_tensor[:, :-1]
    y_batch = data_tensor[:, 1:]
    return x_batch, y_batch
