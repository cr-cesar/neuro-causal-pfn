"""Contrastive fusion encoder (Arm C).

Two backbones (the E7 winner) encode the lesion and the disconnectome
separately; their pooled features are fused by cross-attention over the two
modality tokens and projected to the representation Z (Tsai et al. 2024). A
projection head maps Z to the supervised-contrastive space, two per-modality
heads feed the intra-modal term, and optional decoders reconstruct each modality
from Z to keep the latent grounded in anatomy (the hybrid objective).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vae.backbones import build_encoder_backbone
from ..vae.conv3d_vae import Decoder3D


def _mlp(d_in: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_in), nn.ReLU(inplace=True), nn.Linear(d_in, d_out))


class ContrastiveFusionEncoder(nn.Module):
    def __init__(self, in_shape=(96, 112, 96), channels=(16, 32, 64, 128, 256),
                 zdim: int = 50, backbone: str = "cnn", d_model: int = 128,
                 proj_dim: int = 128, n_heads: int = 4, recon: bool = True):
        super().__init__()
        self.enc_lesion = build_encoder_backbone(backbone, 1, channels)
        self.enc_disco = build_encoder_backbone(backbone, 1, channels)
        self.enc_lesion.eval()
        self.enc_disco.eval()
        with torch.no_grad():
            fl = self.enc_lesion(torch.zeros(1, 1, *in_shape))
            fd = self.enc_disco(torch.zeros(1, 1, *in_shape))
        self.enc_lesion.train()
        self.enc_disco.train()
        self.feat_shape = tuple(int(s) for s in fl.shape[1:])
        self.proj_l = nn.Linear(int(fl.shape[1]), d_model)
        self.proj_d = nn.Linear(int(fd.shape[1]), d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.to_z = nn.Linear(2 * d_model, zdim)
        self.head = _mlp(zdim, proj_dim)          # fused SupCon head
        self.head_l = _mlp(d_model, proj_dim)     # intra-modal heads
        self.head_d = _mlp(d_model, proj_dim)
        self.recon = recon
        self.backbone = backbone
        self.zdim = zdim
        if recon:
            self.dec_lesion = Decoder3D(1, channels, zdim, self.feat_shape, in_shape)
            self.dec_disco = Decoder3D(1, channels, zdim, self.feat_shape, in_shape)

    @staticmethod
    def _pool(f: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool3d(f, 1).flatten(1)

    def encode(self, lesion: torch.Tensor, disco: torch.Tensor):
        fl = self.proj_l(self._pool(self.enc_lesion(lesion)))    # [B, d_model]
        fd = self.proj_d(self._pool(self.enc_disco(disco)))      # [B, d_model]
        tokens = torch.stack([fl, fd], dim=1)                    # [B, 2, d_model]
        fused, _ = self.attn(tokens, tokens, tokens)             # cross-attention
        z = self.to_z(fused.flatten(1))                          # [B, zdim]
        return z, fl, fd

    def forward(self, lesion: torch.Tensor, disco: torch.Tensor) -> dict:
        z, fl, fd = self.encode(lesion, disco)
        out = {"z": z,
               "p": F.normalize(self.head(z), dim=1),
               "p_lesion": F.normalize(self.head_l(fl), dim=1),
               "p_disco": F.normalize(self.head_d(fd), dim=1)}
        if self.recon:
            out["recon_lesion"] = self.dec_lesion(z)
            out["recon_disco"] = self.dec_disco(z)
        return out

    @torch.no_grad()
    def encode_z(self, lesion: torch.Tensor, disco: torch.Tensor) -> torch.Tensor:
        """Deterministic fused representation Z, for export and downstream use."""
        z, _, _ = self.encode(lesion, disco)
        return z
