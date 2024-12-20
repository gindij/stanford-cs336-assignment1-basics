import random
from typing import Tuple

import numpy as np
import torch


def get_batch(
    x: np.ndarray, batch_size: int, context_length: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(x)
    data_batch = np.zeros((batch_size, context_length))
    label_batch = np.zeros((batch_size, context_length))
    for ix_batch in range(batch_size):
        ix_start = random.randint(0, n - context_length - 1)
        data_batch[ix_batch, :] = x[ix_start : ix_start + context_length]
        label_batch[ix_batch, :] = x[ix_start + 1 : ix_start + context_length + 1]
    return (torch.Tensor(data_batch).to(device), torch.Tensor(label_batch).to(device))
