import torch

from neurocausalpfn.vae.losses import bce_dice_loss, kl_standard_normal, soft_dice_loss


def test_dice_zero_on_identical_masks():
    t = (torch.rand(2, 1, 8, 8, 8) > 0.5).float()
    logits = (t * 2 - 1) * 20.0  # sigmoide ~ t
    assert soft_dice_loss(logits, t).item() < 1e-2


def test_bce_dice_finite():
    t = (torch.rand(2, 1, 8, 8, 8) > 0.5).float()
    logits = torch.randn(2, 1, 8, 8, 8)
    loss, parts = bce_dice_loss(logits, t)
    assert torch.isfinite(loss)
    assert "bce" in parts and "dice" in parts


def test_kl_nonnegative():
    mu = torch.randn(4, 16)
    logvar = torch.randn(4, 16) * 0.1
    assert kl_standard_normal(mu, logvar).item() >= -1e-5
