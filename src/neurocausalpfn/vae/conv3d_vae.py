"""Stage 1 3D convolutional variational autoencoder.

The encoder is a stack of stride-2 convolutions; the decoder mirrors it with
transposed convolutions. To ensure that the output has exactly the shape of the
input at any resolution (the padded MNI grid in full mode, or a reduced grid in
prototype mode) the decoder ends with a trilinear interpolation to the target
shape before the logits layer.
"""
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm3d(c_out),
        nn.SiLU(),
    )


def _deconv_block(c_in: int, c_out: int, last: bool = False) -> nn.Sequential:
    layers = [nn.ConvTranspose3d(c_in, c_out, kernel_size=4, stride=2, padding=1)]
    if not last:
        layers += [nn.BatchNorm3d(c_out), nn.SiLU()]
    return nn.Sequential(*layers)


class Encoder3D(nn.Module):
    def __init__(self, in_channels: int = 1,
                 channels: Sequence[int] = (16, 32, 64, 128, 256),
                 zdim: int = 50, in_shape: Tuple[int, int, int] = (96, 112, 96)):
        super().__init__()
        chs = (in_channels,) + tuple(channels)
        self.body = nn.Sequential(*[_conv_block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        with torch.no_grad():
            feat = self.body(torch.zeros(1, in_channels, *in_shape))
        self.feat_shape = tuple(int(s) for s in feat.shape[1:])  # (C, d, h, w)
        flat = 1
        for s in self.feat_shape:
            flat *= s
        self.flat = flat
        self.fc_mu = nn.Linear(flat, zdim)
        self.fc_logvar = nn.Linear(flat, zdim)

    def forward(self, x: torch.Tensor):
        h = self.body(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder3D(nn.Module):
    def __init__(self, out_channels: int = 1,
                 channels: Sequence[int] = (16, 32, 64, 128, 256),
                 zdim: int = 50, feat_shape: Tuple[int, int, int, int] = (256, 3, 4, 3),
                 out_shape: Tuple[int, int, int] = (96, 112, 96)):
        super().__init__()
        self.feat_shape = tuple(feat_shape)
        self.out_shape = tuple(out_shape)
        flat = 1
        for s in self.feat_shape:
            flat *= s
        self.fc = nn.Linear(zdim, flat)
        rev = tuple(reversed(channels))            # e.g. (256, 128, 64, 32, 16)
        targets = rev[1:] + (out_channels,)        # (128, 64, 32, 16, out_channels)
        blocks = []
        c_prev = self.feat_shape[0]
        for i, c_out in enumerate(targets):
            blocks.append(_deconv_block(c_prev, c_out, last=(i == len(targets) - 1)))
            c_prev = c_out
        self.body = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, *self.feat_shape)
        h = self.body(h)
        h = F.interpolate(h, size=self.out_shape, mode="trilinear", align_corners=False)
        return h  # per-voxel logits


class ConvVAE3D(nn.Module):
    def __init__(self, in_channels: int = 1,
                 channels: Sequence[int] = (16, 32, 64, 128, 256),
                 zdim: int = 50, in_shape: Tuple[int, int, int] = (96, 112, 96)):
        super().__init__()
        self.enc = Encoder3D(in_channels, channels, zdim, in_shape)
        self.dec = Decoder3D(in_channels, channels, zdim, self.enc.feat_shape, in_shape)
        self.zdim = zdim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        logits = self.dec(z)
        return logits, mu, logvar, z

    @torch.no_grad()
    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic code (the posterior mean), used when exporting."""
        mu, _ = self.enc(x)
        return mu
