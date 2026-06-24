"""E9b: DMVAE (disentangled multimodal VAE), shared plus private latents.

Following the shared-private decomposition (Lee and Pavlovic 2021), the lesion
and the disconnectome share a latent z_s that carries the cross-modal signal, and
each keeps a private latent (z_p) for its modality-specific content. The shared
posterior fuses the two modalities by a product of experts (with a prior expert
for stability); each modality is reconstructed from its [shared, private] pair. A
KL term on each private latent provides the disentangling pressure that separates
shared from private. The exported representation concatenates the three means.
"""
import torch
import torch.nn as nn

from .backbones import build_encoder_backbone
from .conv3d_vae import Decoder3D
from .losses import bce_dice_loss, kl_diag_gaussian, mse_recon_loss


def product_of_experts(mus, logvars):
    """Precision-weighted combination of Gaussian experts (lists of [B, d])."""
    var = [lv.exp() for lv in logvars]
    precision = sum(1.0 / v for v in var)
    mu = sum(m / v for m, v in zip(mus, var)) / precision
    return mu, (1.0 / precision).log()


def _reparam(mu, logvar):
    return mu + torch.randn_like(mu) * (0.5 * logvar).exp()


class DMVAE3D(nn.Module):
    def __init__(self, in_shape=(96, 112, 96), channels=(16, 32, 64, 128, 256),
                 shared_dim: int = 50, private_dim: int = 25, backbone: str = "cnn"):
        super().__init__()
        self.enc_l = build_encoder_backbone(backbone, 1, channels)
        self.enc_d = build_encoder_backbone(backbone, 1, channels)
        self.enc_l.eval()
        self.enc_d.eval()
        with torch.no_grad():
            feat = self.enc_l(torch.zeros(1, 1, *in_shape))
        self.enc_l.train()
        self.enc_d.train()
        self.feat_shape = tuple(int(s) for s in feat.shape[1:])
        c = int(feat.shape[1])
        self.shared_dim = shared_dim
        self.private_dim = private_dim
        self.zdim = shared_dim + 2 * private_dim
        self.backbone = backbone

        self.head_l_shared = nn.Linear(c, 2 * shared_dim)
        self.head_d_shared = nn.Linear(c, 2 * shared_dim)
        self.head_l_priv = nn.Linear(c, 2 * private_dim)
        self.head_d_priv = nn.Linear(c, 2 * private_dim)
        dec_in = shared_dim + private_dim
        self.dec_l = Decoder3D(1, channels, dec_in, self.feat_shape, in_shape)
        self.dec_d = Decoder3D(1, channels, dec_in, self.feat_shape, in_shape)

    @staticmethod
    def _pool(feat):
        return torch.nn.functional.adaptive_avg_pool3d(feat, 1).flatten(1)

    @staticmethod
    def _split(x, d):
        return x[:, :d], x[:, d:]

    def encode(self, lesion, disco):
        hl = self._pool(self.enc_l(lesion))
        hd = self._pool(self.enc_d(disco))
        mu_sl, lv_sl = self._split(self.head_l_shared(hl), self.shared_dim)
        mu_sd, lv_sd = self._split(self.head_d_shared(hd), self.shared_dim)
        zeros = torch.zeros_like(mu_sl)
        mu_s, lv_s = product_of_experts([zeros, mu_sl, mu_sd], [zeros, lv_sl, lv_sd])  # prior expert + both
        mu_pl, lv_pl = self._split(self.head_l_priv(hl), self.private_dim)
        mu_pd, lv_pd = self._split(self.head_d_priv(hd), self.private_dim)
        return (mu_s, lv_s), (mu_pl, lv_pl), (mu_pd, lv_pd)

    def forward(self, lesion, disco, beta: float = 1.0, lambda_priv: float = 1.0):
        (mu_s, lv_s), (mu_pl, lv_pl), (mu_pd, lv_pd) = self.encode(lesion, disco)
        z_s = _reparam(mu_s, lv_s)
        z_pl = _reparam(mu_pl, lv_pl)
        z_pd = _reparam(mu_pd, lv_pd)
        rec_l, _ = bce_dice_loss(self.dec_l(torch.cat([z_s, z_pl], dim=1)), lesion)
        rec_d = mse_recon_loss(self.dec_d(torch.cat([z_s, z_pd], dim=1)), disco)
        kl_s = kl_diag_gaussian(mu_s, lv_s)
        kl_priv = kl_diag_gaussian(mu_pl, lv_pl) + kl_diag_gaussian(mu_pd, lv_pd)
        loss = rec_l + rec_d + beta * kl_s + lambda_priv * beta * kl_priv
        parts = {"rec_l": float(rec_l.detach()), "rec_d": float(rec_d.detach()),
                 "kl_s": float(kl_s.detach()), "kl_priv": float(kl_priv.detach()),
                 "total": float(loss.detach())}
        return loss, parts

    @torch.no_grad()
    def encode_z(self, lesion, disco):
        """Concatenated means [shared, private_lesion, private_disco]."""
        (mu_s, _), (mu_pl, _), (mu_pd, _) = self.encode(lesion, disco)
        return torch.cat([mu_s, mu_pl, mu_pd], dim=1)
