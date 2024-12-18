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

        print(d_model, d_ff)

        w1_weights = weights.get("w1.weight", torch.randn(self.d_ff, self.d_model)).T
        w2_weights = weights.get("w2.weight", torch.randn(self.d_model, self.d_ff)).T

        self.w1 = torch.nn.Parameter(w1_weights)
        self.w2 = torch.nn.Parameter(w2_weights)

    @classmethod
    def from_weights(cls, weights: Dict[str, torch.Tensor]) -> "FeedForwardNetwork":
        d_ff, d_model = weights["w1.weight"].shape
        return cls(d_model, d_ff, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gelu(x @ self.w1) @ self.w2
