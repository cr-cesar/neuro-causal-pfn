import os
import tempfile

import numpy as np
import torch

from neurocausalpfn.dscm.model import ConditionalHVAE, kl_two_diag_gaussians
from neurocausalpfn.train.train_dscm import prototype_config, run_dscm


# ------------------------------- KL ---------------------------------------- #
def test_kl_zero_when_equal_positive_when_not():
    mu = torch.randn(4, 5)
    lv = torch.zeros(4, 5)
    assert abs(float(kl_two_diag_gaussians(mu, lv, mu, lv))) < 1e-5
    assert float(kl_two_diag_gaussians(mu, lv, mu + 1.0, lv)) > 0.0


def test_kl_nonnegative():
    torch.manual_seed(0)
    for _ in range(5):
        mq, lq = torch.randn(8, 6), torch.randn(8, 6)
        mp, lp = torch.randn(8, 6), torch.randn(8, 6)
        assert float(kl_two_diag_gaussians(mq, lq, mp, lp)) >= -1e-5


# ------------------------------ model -------------------------------------- #
def _model(pa_dim=4, use_ard=False):
    return ConditionalHVAE(in_shape=(24, 28, 24), channels=(8, 16, 32, 64),
                           group_dims=(8, 8), pa_dim=pa_dim, use_ard=use_ard)


def test_forward_shapes():
    model = _model()
    x = (torch.rand(2, 1, 24, 28, 24) > 0.85).float()
    pa = torch.randn(2, 4)
    logits, z, kl = model(x, pa)
    assert logits.shape == (2, 1, 24, 28, 24)
    assert z.shape == (2, 16)
    assert torch.isfinite(kl)


def test_encode_z_shape_and_deterministic():
    model = _model().eval()
    x = (torch.rand(2, 1, 24, 28, 24) > 0.85).float()
    z1 = model.encode_z(x)
    z2 = model.encode_z(x)
    assert z1.shape == (2, 16)
    assert torch.allclose(z1, z2)


def test_counterfactual_consistency():
    model = _model().eval()
    x = (torch.rand(2, 1, 24, 28, 24) > 0.85).float()
    pa = torch.randn(2, 4)
    # do(pa = pa) must reproduce the factual latent (SCM consistency)
    _, z_same = model.counterfactual(x, pa, pa)
    assert torch.allclose(z_same, model.encode_z(x), atol=1e-4)
    # a different intervention changes the latent
    _, z_diff = model.counterfactual(x, pa, pa + 2.0)
    assert not torch.allclose(z_diff, model.encode_z(x), atol=1e-3)


def test_ard_scale_present():
    model = _model(use_ard=True)
    assert model.ard_log_scale.shape == (16,)
    x = (torch.rand(2, 1, 24, 28, 24) > 0.85).float()
    _, _, kl = model(x, torch.randn(2, 4))
    assert torch.isfinite(kl)


# ---------------------------- training ------------------------------------- #
def _cfg(use_ard=False, multi_env=False):
    cfg = prototype_config()
    cfg["model"]["use_ard"] = use_ard
    cfg["model"]["multi_env"] = multi_env
    cfg["model"]["n_regimes"] = 5
    cfg["model"]["channels"] = [8, 16, 32, 64]
    cfg["model"]["group_dims"] = [8, 8]
    cfg["data"]["resolution"] = [24, 28, 24]
    cfg["data"]["n_synth"] = 8
    cfg["data"]["val_frac"] = 0.25
    cfg["train"]["epochs"] = 2
    cfg["train"]["batch_size"] = 4
    cfg["export"] = True
    return cfg


def test_run_dscm_e8a():
    cfg = _cfg()
    with tempfile.TemporaryDirectory() as out:
        cfg["out_dir"] = out
        _, hist = run_dscm(cfg)
        assert np.isfinite(hist[-1]["total"])
        ckpt = torch.load(os.path.join(out, "dscm.pt"), map_location="cpu")
        assert ckpt["zdim"] == 16 and ckpt["pa_dim"] == 4 and ckpt["multi_env"] is False
        assert np.load(os.path.join(out, "latents_dscm.npz"))["Z"].shape == (8, 16)


def test_run_dscm_e8b_multi_env():
    cfg = _cfg(multi_env=True)
    with tempfile.TemporaryDirectory() as out:
        cfg["out_dir"] = out
        _, hist = run_dscm(cfg)
        ckpt = torch.load(os.path.join(out, "dscm.pt"), map_location="cpu")
        assert ckpt["multi_env"] is True and ckpt["pa_dim"] == 4 + 5  # clinical + regimes


def test_run_dscm_e8c_ard():
    cfg = _cfg(use_ard=True)
    with tempfile.TemporaryDirectory() as out:
        cfg["out_dir"] = out
        _, hist = run_dscm(cfg)
        ckpt = torch.load(os.path.join(out, "dscm.pt"), map_location="cpu")
        assert ckpt["use_ard"] is True
