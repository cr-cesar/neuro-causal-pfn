import os
import tempfile

import numpy as np
import nibabel as nib
import torch

from neurocausalpfn.data.nifti_dataset import PairedLesionDisconnectomeDataset
from neurocausalpfn.train.train_vae import prototype_config as vae_proto, run_vae
from neurocausalpfn.vae.losses import vae_loss_two_channel


def _write_nifti(path, arr):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4)), path)


def test_two_channel_loss_has_both_terms():
    logits = torch.randn(2, 2, 6, 6, 6)
    target = torch.cat([(torch.rand(2, 1, 6, 6, 6) > 0.6).float(),  # binary lesion
                        torch.rand(2, 1, 6, 6, 6)], dim=1)          # continuous disconnectome
    mu = torch.zeros(2, 4)
    logvar = torch.zeros(2, 4)
    _, parts = vae_loss_two_channel(logits, target, mu, logvar, beta=1.0)
    for key in ("bce", "dice", "mse", "rec", "kl", "total"):
        assert key in parts
    assert np.isfinite(parts["total"])


def test_paired_dataset_stacks_channels():
    shape = (24, 28, 24)
    with tempfile.TemporaryDirectory() as les_dir, tempfile.TemporaryDirectory() as dis_dir:
        rng = np.random.default_rng(0)
        for i in (1, 2, 3):
            _write_nifti(os.path.join(les_dir, f"lesion{i:04d}_70_M.nii.gz"),
                         (rng.random(shape) > 0.7).astype(np.float32))
            _write_nifti(os.path.join(dis_dir, f"lesion{i:04d}_70_M.nii.gz"),
                         rng.random(shape).astype(np.float32))
        ds = PairedLesionDisconnectomeDataset(les_dir, dis_dir, in_shape=shape, stack_channels=True)
        vol = ds[0]
        assert vol.shape == (2, *shape)                                  # two channels
        assert set(torch.unique(vol[0]).tolist()).issubset({0.0, 1.0})   # channel 0 binary
        assert torch.unique(vol[1]).numel() > 2                          # channel 1 continuous


def test_early_fusion_vae_runs():
    cfg = vae_proto()
    cfg["representation"] = "early_fusion"
    cfg["data"]["resolution"] = [24, 28, 24]
    cfg["data"]["n_synth"] = 8
    cfg["data"]["val_frac"] = 0.25
    cfg["vae"]["epochs"] = 2
    cfg["vae"]["batch_size"] = 2
    cfg["vae"]["channels"] = [8, 16, 32, 64]
    cfg["vae"]["zdim"] = 8
    with tempfile.TemporaryDirectory() as out:
        cfg["out_dir"] = out
        _, hist = run_vae(cfg)
        assert {"bce", "dice", "mse"}.issubset(hist[-1].keys())   # both reconstruction terms
        ckpt_path = os.path.join(out, "vae_early_fusion.pt")
        assert os.path.exists(ckpt_path)
        assert torch.load(ckpt_path, map_location="cpu")["in_channels"] == 2
