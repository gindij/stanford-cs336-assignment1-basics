from typing import Dict, Optional
import torch

from cs336_basics.transformer.functional import gelu


class FeedForwardNetwork(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()

        if weights is None:
            weights = {}

        self.d_model = d_model
        self.d_ff = d_ff or (4 * d_model)

        norm = torch.distributions.Normal(0.0, 0.02)
        w1_params = norm.sample((self.d_ff, self.d_model))
        w2_params = norm.sample((self.d_model, self.d_ff))
        self.w1 = torch.nn.Parameter(w1_params)
        self.w2 = torch.nn.Parameter(w2_params)

        if len(weights) > 0:
            self.load_state_dict({k.replace(".weight", ""): v for k, v in weights.items()})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gelu(x @ self.w1.T) @ self.w2.T
