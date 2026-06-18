"""Clinical conditioning via the Dynamic Affine Feature-map Transform.

From a feature map inside a residual block and a vector of clinical covariates, a
small bottleneck network predicts a per-channel scale and shift. It generalizes
feature-wise linear modulation. Whether the conditioning helps is decided
empirically in E5.
"""
import torch
import torch.nn as nn


class DAFT(nn.Module):
    def __init__(self, n_channels: int, n_tabular: int, bottleneck: int = 7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Linear(n_channels + n_tabular, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, 2 * n_channels),
        )
        self.n_channels = n_channels

    def forward(self, fmap: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        g = self.pool(fmap).flatten(1)
        scale, shift = self.mlp(torch.cat([g, tabular], dim=1)).chunk(2, dim=1)
        view = (-1, self.n_channels, 1, 1, 1)
        return scale.view(*view) * fmap + shift.view(*view)
