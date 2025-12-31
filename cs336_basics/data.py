import random
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
    # 1. Generate indices on CPU using torch
    ix = torch.randint(0, n - context_length - 1, (batch_size,))

    # 2. Slice the NumPy array directly.
    # Slicing a numpy mmap is efficient; it only reads the needed bytes from disk.
    data_stack = np.stack([x[i : i + context_length + 1] for i in ix])
    data_tensor = torch.from_numpy(data_stack).to(device)
    # y_stack = np.stack([x[i + 1 : i + context_length + 1] for i in ix])

    # 3. Convert the small batch to torch and move to device
    # This creates a fresh, writable copy of just the batch, resolving the warning.
    x_batch = data_tensor[:, :-1]
    y_batch = data_tensor[:, 1:]

    return x_batch, y_batch
