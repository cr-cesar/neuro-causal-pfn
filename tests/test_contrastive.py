import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from neurocausalpfn.contrastive.losses import nt_xent_loss, supcon_loss
from neurocausalpfn.contrastive.model import ContrastiveFusionEncoder
from neurocausalpfn.data.augmentations import (augment_batch, augment_volume,
                                               territory_mask)
from neurocausalpfn.train.train_contrastive import (prototype_config,
                                                    run_contrastive)


# --------------------------- augmentations --------------------------------- #
def test_augmentation_keeps_mask_binary():
    rng = np.random.default_rng(0)
    mask = (rng.random((24, 28, 24)) > 0.6).astype(np.float32)
    for _ in range(5):
        out = augment_volume(mask, binary=True, rng=rng)
        assert set(np.unique(out)).issubset({0.0, 1.0})
        assert out.shape == mask.shape


def test_augmentation_keeps_continuous_in_range():
    rng = np.random.default_rng(1)
    vol = rng.random((24, 28, 24)).astype(np.float32)
    for _ in range(5):
        out = augment_volume(vol, binary=False, rng=rng)
        assert out.min() >= 0.0 and out.max() <= 1.0001
        assert out.shape == vol.shape


def test_territory_mask_removes_signal():
    vol = np.ones((24, 28, 24), dtype=np.float32)
    out = territory_mask(vol, np.random.default_rng(0), frac=0.25)
    assert out.sum() < vol.sum()


def test_augment_batch_shape():
    batch = torch.rand(3, 1, 24, 28, 24)
    out = augment_batch(batch, binary=False, seed=0)
    assert out.shape == batch.shape


# ----------------------------- losses -------------------------------------- #
def test_supcon_lower_when_grouped_by_label():
    labels = torch.tensor([0, 1, 0, 0, 1, 0])
    rand = F.normalize(torch.randn(6, 8, generator=torch.Generator().manual_seed(0)), dim=1)
    grouped = torch.zeros(6, 8)
    grouped[labels == 0, 0] = 1.0
    grouped[labels == 1, 1] = 1.0
    grouped = F.normalize(grouped, dim=1)
    assert torch.isfinite(supcon_loss(rand, labels))
    assert float(supcon_loss(grouped, labels)) < float(supcon_loss(rand, labels))


def test_nt_xent_lower_when_views_aligned():
    base = F.normalize(torch.randn(4, 8), dim=1)
    aligned = F.normalize(torch.cat([base, base], dim=0), dim=1)   # views identical
    rand = F.normalize(torch.randn(8, 8), dim=1)
    assert float(nt_xent_loss(aligned, 4)) < float(nt_xent_loss(rand, 4))


# ----------------------------- model --------------------------------------- #
def _small_model(recon=True, backbone="cnn"):
    return ContrastiveFusionEncoder(in_shape=(24, 28, 24), channels=(8, 16, 32, 64),
                                    zdim=8, backbone=backbone, d_model=32, proj_dim=16,
                                    n_heads=4, recon=recon)


def test_model_forward_shapes():
    model = _small_model(recon=True)
    les, dis = torch.rand(2, 1, 24, 28, 24), torch.rand(2, 1, 24, 28, 24)
    out = model(les, dis)
    assert out["z"].shape == (2, 8)
    assert out["p"].shape == (2, 16)
    assert out["p_lesion"].shape == (2, 16) and out["p_disco"].shape == (2, 16)
    assert out["recon_lesion"].shape == (2, 1, 24, 28, 24)
    # projections are L2-normalised
    assert torch.allclose(out["p"].norm(dim=1), torch.ones(2), atol=1e-4)


def test_model_without_recon():
    model = _small_model(recon=False)
    les, dis = torch.rand(2, 1, 24, 28, 24), torch.rand(2, 1, 24, 28, 24)
    out = model(les, dis)
    assert "recon_lesion" not in out and "recon_disco" not in out
    assert out["z"].shape == (2, 8)


def test_encode_z_shape_and_deterministic():
    model = _small_model(recon=True).eval()
    les, dis = torch.rand(2, 1, 24, 28, 24), torch.rand(2, 1, 24, 28, 24)
    z1 = model.encode_z(les, dis)
    z2 = model.encode_z(les, dis)
    assert z1.shape == (2, 8)
    assert torch.allclose(z1, z2)


def test_model_with_resnet_backbone():
    model = _small_model(recon=False, backbone="resnet")
    out = model(torch.rand(2, 1, 24, 28, 24), torch.rand(2, 1, 24, 28, 24))
    assert out["z"].shape == (2, 8)


# ---------------------------- training ------------------------------------- #
def _cfg(recon):
    cfg = prototype_config()
    cfg["model"]["recon"] = recon
    cfg["model"]["channels"] = [8, 16, 32, 64]
    cfg["model"]["zdim"] = 8
    cfg["model"]["d_model"] = 32
    cfg["model"]["proj_dim"] = 16
    cfg["data"]["resolution"] = [24, 28, 24]
    cfg["data"]["n_synth"] = 8
    cfg["data"]["val_frac"] = 0.25
    cfg["train"]["epochs"] = 2
    cfg["train"]["batch_size"] = 4
    cfg["export"] = True
    return cfg


def test_run_contrastive_e10c_with_recon():
    cfg = _cfg(recon=True)
    with tempfile.TemporaryDirectory() as out:
        cfg["out_dir"] = out
        _, hist = run_contrastive(cfg)
        assert np.isfinite(hist[-1]["total"]) and "recon" in hist[-1]
        ckpt = torch.load(os.path.join(out, "contrastive.pt"), map_location="cpu")
        assert ckpt["zdim"] == 8
        Z = np.load(os.path.join(out, "latents_contrastive.npz"))["Z"]
        assert Z.shape == (8, 8)


def test_run_contrastive_e10a_without_recon():
    cfg = _cfg(recon=False)
    with tempfile.TemporaryDirectory() as out:
        cfg["out_dir"] = out
        _, hist = run_contrastive(cfg)
        assert np.isfinite(hist[-1]["total"]) and "recon" not in hist[-1]
