"""Condicionamiento clinico mediante el Dynamic Affine Feature-map Transform.

A partir de un mapa de caracteristicas dentro de un bloque residual y un vector
de covariables clinicas, una pequena red de cuello de botella predice una escala
y un desplazamiento por canal. Generaliza la modulacion lineal por
caracteristicas. Si el condicionamiento ayuda se decide de forma empirica en E5.
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
