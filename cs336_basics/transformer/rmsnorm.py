from typing import Dict, Optional

import torch


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        epsilon: float = 1e-5,
        weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if weights is None:
            weights = {}

        super().__init__()
        self.ep = epsilon
        self.d_model = d_model
        self.weight = torch.nn.Parameter(weights.get("weight", torch.randn((d_model,))))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(a**2, dim=-1, keepdim=True) + self.ep)
        return a * self.weight.view(1, 1, -1) / rms
