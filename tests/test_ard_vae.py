import os
import tempfile

import numpy as np
import torch

from neurocausalpfn.train.train_vae import prototype_config as vae_proto, run_vae
from neurocausalpfn.vae.conv3d_vae import ConvVAE3D
from neurocausalpfn.vae.losses import (ard_update_prior_var, kl_diag_gaussian,
                                       kl_standard_normal, per_dim_kl)


def test_kl_diag_gaussian_matches_standard_when_prior_is_one():
    mu = torch.randn(8, 10)
    logvar = torch.randn(8, 10) * 0.1
    standard = kl_standard_normal(mu, logvar)
    with_ones = kl_diag_gaussian(mu, logvar, torch.ones(10))
    none_prior = kl_diag_gaussian(mu, logvar, None)
    assert torch.allclose(standard, with_ones, atol=1e-5)
    assert torch.allclose(standard, none_prior, atol=1e-5)


def test_per_dim_kl_and_prior_update():
    mu = torch.randn(16, 6)
    logvar = torch.zeros(16, 6)
    pdk = per_dim_kl(mu, logvar, None)
    assert pdk.shape == (6,)
    # conjugate update returns the mean second moment, floored
    sumsq = (mu.pow(2) + logvar.exp()).sum(0)
    pv = ard_update_prior_var(sumsq, 16)
    assert pv.shape == (6,)
    assert torch.all(pv >= 1e-4)
    assert torch.allclose(pv, sumsq / 16, atol=1e-4)


def test_smaller_prior_var_increases_kl():
    mu = torch.ones(4, 3)
    logvar = torch.zeros(4, 3)
    kl_unit = kl_diag_gaussian(mu, logvar, torch.ones(3))
    kl_tight = kl_diag_gaussian(mu, logvar, torch.full((3,), 0.1))
    assert kl_tight > kl_unit


def test_ard_vae_trains_and_checkpoints():
    cfg = vae_proto()
    cfg["vae"]["use_ard"] = True
    cfg["data"]["resolution"] = [24, 28, 24]
    cfg["data"]["n_synth"] = 8
    cfg["data"]["val_frac"] = 0.25
    cfg["vae"]["epochs"] = 2
    cfg["vae"]["batch_size"] = 2
    cfg["vae"]["channels"] = [8, 16, 32, 64]
    cfg["vae"]["zdim"] = 12
    with tempfile.TemporaryDirectory() as out:
        cfg["out_dir"] = out
        model, hist = run_vae(cfg)
        assert hasattr(model, "ard_prior_var") and model.ard_prior_var.shape == (12,)
        # the prior variance was updated away from its all-ones initialisation
        assert not torch.allclose(model.ard_prior_var, torch.ones(12))
        assert 0 <= hist[-1]["active_dims"] <= 12
        ckpt = torch.load(os.path.join(out, "vae_lesion.pt"), map_location="cpu")
        assert ckpt["use_ard"] is True
        assert "ard_prior_var" in ckpt["state_dict"]
