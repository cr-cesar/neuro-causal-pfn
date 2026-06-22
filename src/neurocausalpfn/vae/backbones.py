"""Selectable encoder backbones (E7).

The encoder body maps a volume [B, C_in, D, H, W] to a feature map
[B, C_last, d, h, w]; the rest of the VAE (DAFT, the latent heads and the
decoder, which is built from the resulting feature shape) is agnostic to which
backbone produced it. The plan compares five options:

- "cnn":     Pombo-style flat stride-2 convolutions (the default, ~2M params).
- "wide":    the same depth as "cnn" with wider channels (capacity without depth,
             the control for E7).
- "resnet":  Giles-style 3D residual blocks following the configured channels.
- "resnet18": a 3D ResNet-18 (BasicBlock, [2, 2, 2, 2], ~11M params).
- "resnet50": a 3D ResNet-50 (Bottleneck, [3, 4, 6, 3], ~25M params).
"""
from typing import Sequence

import torch.nn as nn


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm3d(c_out),
        nn.SiLU(),
    )


def _cnn_body(in_channels: int, channels: Sequence[int], width: int = 1) -> nn.Sequential:
    ch = [int(c * width) for c in channels]
    chs = (in_channels,) + tuple(ch)
    return nn.Sequential(*[_conv_block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(c_out)
        self.conv2 = nn.Conv3d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(c_out)
        self.act = nn.SiLU(inplace=True)
        self.down = None
        if stride != 1 or c_in != c_out * self.expansion:
            self.down = nn.Sequential(
                nn.Conv3d(c_in, c_out * self.expansion, 1, stride, bias=False),
                nn.BatchNorm3d(c_out * self.expansion))

    def forward(self, x):
        idn = x if self.down is None else self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + idn)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, c_in: int, c_mid: int, stride: int = 1):
        super().__init__()
        c_out = c_mid * self.expansion
        self.conv1 = nn.Conv3d(c_in, c_mid, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(c_mid)
        self.conv2 = nn.Conv3d(c_mid, c_mid, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(c_mid)
        self.conv3 = nn.Conv3d(c_mid, c_out, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(c_out)
        self.act = nn.SiLU(inplace=True)
        self.down = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(
                nn.Conv3d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm3d(c_out))

    def forward(self, x):
        idn = x if self.down is None else self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.act(out + idn)


class _ResidualBody(nn.Module):
    """Giles-style residual encoder: one residual block per configured channel,
    each downsampling by a stride of 2."""

    def __init__(self, in_channels: int, channels: Sequence[int]):
        super().__init__()
        blocks = []
        c_prev = in_channels
        for c in channels:
            blocks.append(_BasicBlock(c_prev, c, stride=2))
            c_prev = c
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        return self.body(x)


class _ResNet(nn.Module):
    """Standard 3D ResNet body (stem plus four stages), without the global pooling
    and classifier, so it returns a spatial feature map."""

    def __init__(self, in_channels: int, block, layers, base: int = 64):
        super().__init__()
        self.in_planes = base
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base), nn.SiLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1))
        self.layer1 = self._make(block, base, layers[0], stride=1)
        self.layer2 = self._make(block, base * 2, layers[1], stride=2)
        self.layer3 = self._make(block, base * 4, layers[2], stride=2)
        self.layer4 = self._make(block, base * 8, layers[3], stride=2)

    def _make(self, block, c_mid, n, stride):
        layers = [block(self.in_planes, c_mid, stride)]
        self.in_planes = c_mid * block.expansion
        for _ in range(1, n):
            layers.append(block(self.in_planes, c_mid, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def build_encoder_backbone(name: str, in_channels: int, channels: Sequence[int],
                           width: int = 2) -> nn.Module:
    name = (name or "cnn").lower()
    if name == "cnn":
        return _cnn_body(in_channels, channels, width=1)
    if name == "wide":
        return _cnn_body(in_channels, channels, width=width)
    if name == "resnet":
        return _ResidualBody(in_channels, channels)
    if name == "resnet18":
        return _ResNet(in_channels, _BasicBlock, [2, 2, 2, 2])
    if name == "resnet50":
        return _ResNet(in_channels, _Bottleneck, [3, 4, 6, 3])
    raise ValueError(f"unknown encoder backbone '{name}'")
