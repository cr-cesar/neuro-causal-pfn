"""Deep Structural Causal Model arm (Arm E, sections 9.5 and 11.3).

A conditional hierarchical VAE in the De Sousa Ribeiro / Pawlowski-Castro-Glocker
style. Causal structure is imposed architecturally: the latent prior is
conditioned on the parents of the image pa_x (the covariates that causally
precede it), p(z | pa_x), rather than a fixed N(0, I). The latents form a small
hierarchy, each group's prior conditioned on pa_x and the coarser groups
(top-down). Because every conditional is Gaussian, counterfactuals follow Pearl's
three steps in closed form: abduction recovers the exogenous noise, the action
intervenes on pa_x, and prediction propagates the noise through the new
conditionals.
"""
import torch
import torch.nn as nn

from ..vae.backbones import build_encoder_backbone
from ..vae.conv3d_vae import Decoder3D


def kl_two_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    """KL(N(mu_q, var_q) || N(mu_p, var_p)) summed over dimensions, averaged over
    the batch."""
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    return kl.sum(dim=1).mean()


def _mlp(d_in, d_out, hidden=64):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, d_out))


class ConditionalHVAE(nn.Module):
    def __init__(self, in_shape=(96, 112, 96), channels=(16, 32, 64, 128, 256),
                 group_dims=(25, 25), pa_dim: int = 4, backbone: str = "cnn",
                 use_ard: bool = False):
        super().__init__()
        self.enc = build_encoder_backbone(backbone, 1, channels)
        self.enc.eval()
        with torch.no_grad():
            feat = self.enc(torch.zeros(1, 1, *in_shape))
        self.enc.train()
        self.feat_shape = tuple(int(s) for s in feat.shape[1:])
        c = int(feat.shape[1])
        self.groups = list(group_dims)
        self.zdim = sum(self.groups)
        self.pa_dim = pa_dim
        self.use_ard = use_ard
        self.backbone = backbone

        # bottom-up posterior heads (one per group): pooled feature -> mu, logvar
        self.q_heads = nn.ModuleList([nn.Linear(c, 2 * g) for g in self.groups])
        # top-down conditional priors: [pa_x, coarser groups] -> mu, logvar
        self.prior_mlps = nn.ModuleList()
        prev = 0
        for g in self.groups:
            self.prior_mlps.append(_mlp(pa_dim + prev, 2 * g))
            prev += g
        self.decoder = Decoder3D(1, channels, self.zdim, self.feat_shape, in_shape)
        if use_ard:
            self.ard_log_scale = nn.Parameter(torch.zeros(self.zdim))

    @staticmethod
    def _pool(feat):
        return torch.nn.functional.adaptive_avg_pool3d(feat, 1).flatten(1)

    def encode_post(self, x):
        h = self._pool(self.enc(x))
        mus, logvars = [], []
        for head, g in zip(self.q_heads, self.groups):
            out = head(h)
            mus.append(out[:, :g])
            logvars.append(out[:, g:])
        return mus, logvars

    def _ard_offset(self, k):
        if not self.use_ard:
            return 0.0
        start = sum(self.groups[:k])
        return self.ard_log_scale[start:start + self.groups[k]]

    def _prior(self, k, pa, prev_groups):
        inp = torch.cat([pa, *prev_groups], dim=1) if prev_groups else pa
        out = self.prior_mlps[k](inp)
        g = self.groups[k]
        return out[:, :g], out[:, g:] + self._ard_offset(k)

    @staticmethod
    def _reparam(mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def forward(self, x, pa):
        mus_q, logvars_q = self.encode_post(x)
        zs, kl = [], 0.0
        for k in range(len(self.groups)):
            mu_p, logvar_p = self._prior(k, pa, zs)        # prior conditioned on pa_x and coarser zs
            z_k = self._reparam(mus_q[k], logvars_q[k])
            kl = kl + kl_two_diag_gaussians(mus_q[k], logvars_q[k], mu_p, logvar_p)
            zs.append(z_k)
        z = torch.cat(zs, dim=1)
        return self.decoder(z), z, kl

    @torch.no_grad()
    def encode_z(self, x):
        """Deterministic latent for export: the concatenated posterior means."""
        mus_q, _ = self.encode_post(x)
        return torch.cat(mus_q, dim=1)

    @torch.no_grad()
    def counterfactual(self, x, pa, pa_cf):
        """Pearl's abduction-action-prediction for the conditional Gaussian SCM.
        Returns the counterfactual reconstruction logits and latent under do(pa=pa_cf)."""
        mus_q, _ = self.encode_post(x)
        factual = [m for m in mus_q]                       # abduction uses posterior means as z
        # abduction: recover exogenous noise per group at the factual parents
        us, prev = [], []
        for k in range(len(self.groups)):
            mu_p, logvar_p = self._prior(k, pa, prev)
            sigma = (0.5 * logvar_p).exp()
            us.append((factual[k] - mu_p) / (sigma + 1e-6))
            prev.append(factual[k])
        # action + prediction: propagate the same noise through the counterfactual parents
        z_cf, prev_cf = [], []
        for k in range(len(self.groups)):
            mu_p, logvar_p = self._prior(k, pa_cf, prev_cf)
            sigma = (0.5 * logvar_p).exp()
            z_cf.append(mu_p + sigma * us[k])
            prev_cf.append(z_cf[-1])
        z = torch.cat(z_cf, dim=1)
        return self.decoder(z), z
