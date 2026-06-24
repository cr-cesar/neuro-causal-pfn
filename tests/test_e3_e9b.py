import os
import tempfile

import numpy as np
import torch

from neurocausalpfn.train.sweep_dims import e3_grid, run_dim_sweep
from neurocausalpfn.train.sweep_dims import prototype_config as e3_prototype
from neurocausalpfn.train.train_dmvae import prototype_config as dmvae_prototype
from neurocausalpfn.train.train_dmvae import run_dmvae
from neurocausalpfn.vae.dmvae import DMVAE3D, product_of_experts


# ------------------------------- E3 ---------------------------------------- #
def test_e3_grid():
    full = e3_grid()
    assert len(full) == 8
    assert len(e3_grid(asymmetric=False)) == 4
    # the asymmetric points all keep a total of 100
    for d_les, d_dis in full[4:]:
        assert d_les + d_dis == 100


def _tiny_e3_cfg():
    cfg = e3_prototype()
    cfg["data"]["resolution"] = [16, 16, 16]
    cfg["data"]["n_synth"] = 4
    cfg["data"]["val_frac"] = 0.5
    cfg["vae"]["channels"] = [8, 16]
    cfg["vae"]["epochs"] = 1
    cfg["vae"]["batch_size"] = 2
    return cfg


def test_run_dim_sweep_records_both_modalities():
    cfg = _tiny_e3_cfg()
    with tempfile.TemporaryDirectory() as out:
        cfg["out_dir"] = out
        res = run_dim_sweep(cfg, grid=[(2, 2), (3, 1)])
    assert len(res) == 2
    for r in res:
        assert r["total_dim"] == r["d_lesion"] + r["d_disco"]
        assert np.isfinite(r["val_lesion"]) and np.isfinite(r["val_disco"])


# ------------------------------ E9b ---------------------------------------- #
def test_product_of_experts_reduces_variance():
    mu = torch.zeros(2, 4)
    lv = torch.zeros(2, 4)                          # var = 1
    mu_c, lv_c = product_of_experts([mu, mu], [lv, lv])
    assert torch.allclose(mu_c, mu)
    assert float(lv_c.mean()) < 0.0                 # combined variance below 1


def _dmvae(shared=8, private=4):
    return DMVAE3D(in_shape=(16, 16, 16), channels=(8, 16), shared_dim=shared,
                   private_dim=private)


def test_dmvae_forward_and_parts():
    model = _dmvae()
    les = (torch.rand(2, 1, 16, 16, 16) > 0.85).float()
    dis = torch.rand(2, 1, 16, 16, 16)
    loss, parts = model(les, dis, beta=1.0, lambda_priv=1.0)
    assert torch.isfinite(loss)
    for key in ("rec_l", "rec_d", "kl_s", "kl_priv", "total"):
        assert key in parts
    assert model.zdim == 8 + 2 * 4


def test_dmvae_encode_z_shape_and_deterministic():
    model = _dmvae().eval()
    les = (torch.rand(2, 1, 16, 16, 16) > 0.85).float()
    dis = torch.rand(2, 1, 16, 16, 16)
    z1 = model.encode_z(les, dis)
    z2 = model.encode_z(les, dis)
    assert z1.shape == (2, 16)
    assert torch.allclose(z1, z2)


def test_run_dmvae_e9b():
    cfg = dmvae_prototype()
    cfg["data"]["resolution"] = [16, 16, 16]
    cfg["data"]["n_synth"] = 8
    cfg["data"]["val_frac"] = 0.25
    cfg["model"]["channels"] = [8, 16]
    cfg["model"]["shared_dim"] = 8
    cfg["model"]["private_dim"] = 4
    cfg["train"]["epochs"] = 2
    cfg["train"]["batch_size"] = 4
    cfg["export"] = True
    with tempfile.TemporaryDirectory() as out:
        cfg["out_dir"] = out
        _, hist = run_dmvae(cfg)
        assert np.isfinite(hist[-1]["total"]) and "kl_priv" in hist[-1]
        ckpt = torch.load(os.path.join(out, "dmvae.pt"), map_location="cpu")
        assert ckpt["shared_dim"] == 8 and ckpt["private_dim"] == 4 and ckpt["zdim"] == 16
        assert np.load(os.path.join(out, "latents_dmvae.npz"))["Z"].shape == (8, 16)
