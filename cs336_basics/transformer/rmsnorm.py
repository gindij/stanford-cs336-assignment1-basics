from typing import Dict, Optional

import torch


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        epsilon: float,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.ep = epsilon
        self.d_model = d_model
        if state is None:
            weights = torch.randn((d_model,))
        else:
            assert "weight" in state
            weights = state["weight"]
        self.g = torch.nn.Parameter(weights)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(a**2, dim=-1, keepdim=True) + self.ep)
        return a * self.g.view(1, 1, -1) / rms  # type: ignore
