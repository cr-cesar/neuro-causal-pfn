"""The ELBO balance: the dim-summed KL must enter the total on the per-voxel
scale of the mean-reduced reconstruction terms (losses.kl_voxel_scale).
Without the scaling, beta=1 acts like beta~1e6 and the posterior collapses —
the failure observed as T1 Dice 0.000-0.05 on the full cohort (E1/E2)."""
import torch

from neurocausalpfn.vae.losses import (kl_diag_gaussian, kl_voxel_scale,
                                       vae_loss, vae_loss_mse,
                                       vae_loss_two_channel)


def _batch(channels=1):
    torch.manual_seed(0)
    target = (torch.rand(2, channels, 16, 16, 16) > 0.95).float()
    logits = torch.randn(2, channels, 16, 16, 16)
    mu, logvar = torch.randn(2, 50), torch.randn(2, 50) * 0.1
    return logits, target, mu, logvar


def test_kl_voxel_scale_value():
    t = torch.zeros(2, 1, 16, 16, 16)
    assert kl_voxel_scale(t) == 1.0 / (16 * 16 * 16)
    t2 = torch.zeros(2, 2, 16, 16, 16)
    assert kl_voxel_scale(t2) == 1.0 / (2 * 16 * 16 * 16)


def test_vae_loss_kl_contribution_is_per_voxel():
    logits, target, mu, logvar = _batch()
    total1, parts = vae_loss(logits, target, mu, logvar, beta=1.0)
    total0, _ = vae_loss(logits, target, mu, logvar, beta=0.0)
    kl = kl_diag_gaussian(mu, logvar)
    expected = float(kl) * kl_voxel_scale(target)
    assert abs(float(total1 - total0) - expected) < 1e-6
    # the raw (unscaled) KL stays in the diagnostics
    assert abs(parts["kl"] - float(kl)) < 1e-5


def test_mse_and_two_channel_use_the_same_scale():
    logits, target, mu, logvar = _batch()
    t1, _ = vae_loss_mse(logits, torch.sigmoid(target), mu, logvar, beta=1.0)
    t0, _ = vae_loss_mse(logits, torch.sigmoid(target), mu, logvar, beta=0.0)
    kl = float(kl_diag_gaussian(mu, logvar))
    assert abs(float(t1 - t0) - kl * kl_voxel_scale(target)) < 1e-6

    logits2, target2, mu2, logvar2 = _batch(channels=2)
    kl2 = float(kl_diag_gaussian(mu2, logvar2))
    t1, _ = vae_loss_two_channel(logits2, target2, mu2, logvar2, beta=1.0)
    t0, _ = vae_loss_two_channel(logits2, target2, mu2, logvar2, beta=0.0)
    assert abs(float(t1 - t0) - kl2 * kl_voxel_scale(target2)) < 1e-6
