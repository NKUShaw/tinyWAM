from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HighResolutionWristAdapter(nn.Module):
    """Same-shape residual correction from a denser wrist patch grid."""

    def __init__(self, dim: int, bottleneck_dim: int = 96) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        self.alpha = nn.Parameter(torch.ones(()))
        # Exact no-op at initialization, while `up` still receives gradients.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        base_wrist: torch.Tensor,
        high_resolution_wrist: torch.Tensor,
        base_grid: tuple[int, int],
        high_resolution_grid: tuple[int, int],
    ) -> torch.Tensor:
        batch, tokens, dim = base_wrist.shape
        if tokens != base_grid[0] * base_grid[1]:
            raise ValueError(f"base wrist token/grid mismatch: {tokens} vs {base_grid}")
        if high_resolution_wrist.shape[1] != high_resolution_grid[0] * high_resolution_grid[1]:
            raise ValueError("high-resolution wrist token/grid mismatch")
        hr = high_resolution_wrist.transpose(1, 2).reshape(
            batch, dim, high_resolution_grid[0], high_resolution_grid[1]
        )
        hr = F.interpolate(hr, size=base_grid, mode="bilinear", align_corners=False)
        hr = hr.flatten(2).transpose(1, 2)
        detail = hr - base_wrist
        delta = self.up(F.gelu(self.down(self.norm(detail))))
        return base_wrist + self.alpha.to(delta.dtype) * delta
