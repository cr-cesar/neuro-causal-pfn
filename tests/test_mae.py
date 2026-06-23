import os
import tempfile

import numpy as np
import torch

from neurocausalpfn.mae.losses import masked_lesion_bce
from neurocausalpfn.mae.model import HiEndMAE3D, block_masking, patchify
from neurocausalpfn.train.train_mae import prototype_config, run_mae


# --------------------------- masking --------------------------------------- #
def test_block_masking_ratio_and_fixed_keep():
    grid = (6, 7, 6)
    n = grid[0] * grid[1] * grid[2]
    ids_keep, ids_restore, mask = block_masking(3, grid, (2, 2, 2), 0.75, "cpu")
    len_keep = round(n * 0.25)
    assert ids_keep.shape == (3, len_keep)
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})
    # 75% masked per sample
    assert torch.allclose(mask.sum(1), torch.full((3,), float(n - len_keep)))
    # kept patches are exactly the complement of the mask
    assert int(mask[0].sum()) == n - len_keep


def test_block_masking_keeps_are_unmasked():
    grid = (4, 4, 4)
    ids_keep, _, mask = block_masking(2, grid, (2, 2, 2), 0.75, "cpu")
    for b in range(2):
        assert torch.all(mask[b, ids_keep[b]] == 0.0)


# --------------------------- patchify -------------------------------------- #
def test_patchify_shape():
    x = torch.rand(2, 1, 24, 28, 24)
    out = patchify(x, 4)
    assert out.shape == (2, 6 * 7 * 6, 4 ** 3)


# --------------------------- model ----------------------------------------- #
def _small_model(zdim=8):
    return HiEndMAE3D(in_shape=(24, 28, 24), patch=4, embed_dim=64, depth=2, heads=4,
                      decoder_dim=32, decoder_depth=2, decoder_heads=4, zdim=zdim,
                      mask_ratio=0.75, block=(2, 2, 2))


def test_model_forward_shapes():
    model = _small_model()
    x = (torch.rand(2, 1, 24, 28, 24) > 0.85).float()
    pred, mask = model(x)
    assert pred.shape == (2, 6 * 7 * 6, 4 ** 3)
    assert mask.shape == (2, 6 * 7 * 6)
    assert model.patchify(x).shape == pred.shape


def test_encode_z_shape_and_deterministic():
    model = _small_model(zdim=8).eval()
    x = (torch.rand(2, 1, 24, 28, 24) > 0.85).float()
    z1 = model.encode_z(x)
    z2 = model.encode_z(x)
    assert z1.shape == (2, 8)
    assert torch.allclose(z1, z2)


# ----------------------------- loss ---------------------------------------- #
def test_lesion_weight_increases_loss_on_lesion_error():
    target = torch.zeros(1, 2, 8)
    target[0, 0] = 1.0                                   # patch 0 bears a lesion
    mask = torch.ones(1, 2)
    pred = torch.full((1, 2, 8), -5.0)                   # both predict ~0: patch 0 is wrong
    l1 = masked_lesion_bce(pred, target, mask, lesion_weight=1.0)
    l10 = masked_lesion_bce(pred, target, mask, lesion_weight=10.0)
    assert float(l10) > float(l1)


def test_loss_only_on_masked_patches():
    target = torch.zeros(1, 2, 8)
    mask = torch.tensor([[1.0, 0.0]])                    # patch 1 is visible
    good = torch.full((1, 2, 8), -5.0)
    bad_visible = good.clone()
    bad_visible[0, 1] = 5.0                              # corrupt the visible patch
    l_good = masked_lesion_bce(good, target, mask, 10.0)
    l_bad = masked_lesion_bce(bad_visible, target, mask, 10.0)
    assert torch.allclose(l_good, l_bad)


# ---------------------------- training ------------------------------------- #
def test_run_mae_e10b():
    cfg = prototype_config()
    cfg["data"]["resolution"] = [24, 28, 24]
    cfg["data"]["n_synth"] = 8
    cfg["data"]["val_frac"] = 0.25
    cfg["model"]["embed_dim"] = 64
    cfg["model"]["depth"] = 2
    cfg["model"]["decoder_depth"] = 2
    cfg["model"]["zdim"] = 8
    cfg["train"]["epochs"] = 2
    cfg["train"]["batch_size"] = 4
    cfg["export"] = True
    with tempfile.TemporaryDirectory() as out:
        cfg["out_dir"] = out
        _, hist = run_mae(cfg)
        assert np.isfinite(hist[-1]["train"])
        ckpt = torch.load(os.path.join(out, "mae.pt"), map_location="cpu")
        assert ckpt["zdim"] == 8
        Z = np.load(os.path.join(out, "latents_mae.npz"))["Z"]
        assert Z.shape == (8, 8)
