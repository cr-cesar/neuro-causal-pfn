import os
import tempfile

import numpy as np
import nibabel as nib
import torch

from neurocausalpfn.causal.pns import (pns_lower_bound, soft_pns_per_dim,
                                       soft_pns_value)
from neurocausalpfn.data.clinical import load_outcome_table
from neurocausalpfn.data.nifti_dataset import LesionMaskDataset
from neurocausalpfn.train.train_vae import prototype_config as vae_proto, run_vae


def _write_nifti(path, arr):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4)), path)


def test_pns_lower_bound_recovers_driver():
    rng = np.random.default_rng(0)
    n = 400
    Z = rng.normal(size=(n, 6))
    Y = (Z[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(float)   # outcome driven by dim 0
    pns = pns_lower_bound(Z, Y, k=3)
    assert pns.shape == (6,)
    assert int(np.argmax(pns)) == 0
    assert pns[0] > 0.3
    assert np.all(pns >= 0.0)


def test_pns_zero_when_outcome_constant():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(100, 5))
    Y = np.ones(100)
    assert np.allclose(pns_lower_bound(Z, Y, k=2), 0.0)


def test_soft_pns_differentiable_and_ranks_driver():
    rng = np.random.default_rng(2)
    n = 300
    Z = rng.normal(size=(n, 6)).astype(np.float32)
    Y = (Z[:, 0] > 0).astype(np.float32)
    zt = torch.tensor(Z, requires_grad=True)
    yt = torch.tensor(Y)
    val = soft_pns_value(zt, yt, k=3)
    val.backward()
    assert torch.isfinite(zt.grad).all()
    pd = soft_pns_per_dim(torch.tensor(Z), yt, k=3)
    assert int(torch.argmax(pd)) == 0


def test_outcome_table_and_dataset_target():
    shape = (24, 28, 24)
    with tempfile.TemporaryDirectory() as les_dir:
        rng = np.random.default_rng(0)
        for i in (1, 2):
            _write_nifti(os.path.join(les_dir, f"lesion{i:04d}_70_M.nii.gz"),
                         (rng.random(shape) > 0.7).astype(np.float32))
        csv = os.path.join(les_dir, "outcome.csv")
        with open(csv, "w") as f:
            f.write("id,outcome\n0001,1\n0002,0\n")
        table = load_outcome_table(csv)
        assert table["0001"] == 1.0 and table["0002"] == 0.0

        ds = LesionMaskDataset(les_dir, in_shape=shape, with_target=True, outcome_csv=csv)
        assert list(ds.target_vector()) == [1.0, 0.0]
        vol, target = ds[0]
        assert vol.shape == (1, *shape) and float(target) == 1.0


def test_target_after_clinical_when_both():
    ds = LesionMaskDataset(None, in_shape=(24, 28, 24), n_synth=4,
                           with_clinical=True, with_target=True)
    item = ds[0]
    assert len(item) == 3                       # (volume, clinical, target)
    assert item[0].shape == (1, 24, 28, 24)
    assert item[1].ndim == 1                     # clinical vector
    assert item[2].ndim == 0                     # scalar target


def _arm_b_cfg(use_daft):
    cfg = vae_proto()
    cfg["vae"]["use_pns"] = True
    cfg["vae"]["lambda_pns"] = 0.5
    cfg["vae"]["use_daft"] = use_daft
    cfg["data"]["resolution"] = [24, 28, 24]
    cfg["data"]["n_synth"] = 8
    cfg["data"]["val_frac"] = 0.25
    cfg["vae"]["epochs"] = 2
    cfg["vae"]["batch_size"] = 2
    cfg["vae"]["channels"] = [8, 16, 32, 64]
    cfg["vae"]["zdim"] = 8
    return cfg


def test_arm_b_trains_e5b():
    cfg = _arm_b_cfg(use_daft=False)
    with tempfile.TemporaryDirectory() as out:
        cfg["out_dir"] = out
        _, hist = run_vae(cfg)
        assert "pns" in hist[-1] and np.isfinite(hist[-1]["total"])
        ckpt = torch.load(os.path.join(out, "vae_lesion.pt"), map_location="cpu")
        assert ckpt["use_pns"] is True


def test_arm_b_combines_with_daft_e5c():
    cfg = _arm_b_cfg(use_daft=True)
    with tempfile.TemporaryDirectory() as out:
        cfg["out_dir"] = out
        _, hist = run_vae(cfg)
        assert "pns" in hist[-1]
        ckpt = torch.load(os.path.join(out, "vae_lesion.pt"), map_location="cpu")
        assert ckpt["use_pns"] is True and ckpt["use_daft"] is True
