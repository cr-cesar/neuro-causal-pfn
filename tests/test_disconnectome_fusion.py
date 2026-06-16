import os
import tempfile

import numpy as np
import nibabel as nib
import pytest
import torch

from neurocausalpfn.data.nifti_dataset import PairedLesionDisconnectomeDataset
from neurocausalpfn.pfn.tokens import to_tensors
from neurocausalpfn.train.run_stage2_real import build_real_prior, infer_cate_real
from neurocausalpfn.train.train_pfn import build_model
from neurocausalpfn.train.train_vae import prototype_config as vae_proto, run_vae
from neurocausalpfn.vae.conv3d_vae import ConvVAE3D
from neurocausalpfn.vae.fusion import fuse_representation
from neurocausalpfn.vae.losses import vae_loss_mse


def _write_nifti(path, arr):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4)), path)


def test_mse_recon_zero_on_perfect_match():
    target = torch.rand(2, 1, 6, 6, 6) * 0.8 + 0.1          # valores en (0.1, 0.9)
    logits = torch.log(target / (1.0 - target))             # la sigmoide recupera el objetivo
    mu = torch.zeros(2, 4)
    logvar = torch.zeros(2, 4)
    _, parts = vae_loss_mse(logits, target, mu, logvar, beta=0.0)
    assert parts["mse"] < 1e-6
    assert "bce" not in parts and np.isfinite(parts["total"])


def test_paired_dataset_matches_by_id():
    shape = (24, 28, 24)
    with tempfile.TemporaryDirectory() as les_dir, tempfile.TemporaryDirectory() as dis_dir:
        rng = np.random.default_rng(0)
        for i in (1, 2, 3):
            _write_nifti(os.path.join(les_dir, f"lesion{i:04d}_70_M.nii.gz"),
                         (rng.random(shape) > 0.7).astype(np.float32))
            _write_nifti(os.path.join(dis_dir, f"lesion{i:04d}_70_M.nii.gz"),
                         rng.random(shape).astype(np.float32))
        # un id extra solo en lesiones: no debe emparejar
        _write_nifti(os.path.join(les_dir, "lesion0009_NA_NA.nii.gz"),
                     (rng.random(shape) > 0.7).astype(np.float32))

        ds = PairedLesionDisconnectomeDataset(les_dir, dis_dir, in_shape=shape)
        assert len(ds) == 3
        assert ds.ids() == ["0001", "0002", "0003"]
        les, dis = ds[0]
        assert les.shape == (1, *shape) and dis.shape == (1, *shape)
        assert set(torch.unique(les).tolist()).issubset({0.0, 1.0})  # lesion binaria
        assert torch.unique(dis).numel() > 2                          # disconnectoma continuo
        assert float(dis.max()) <= 1.0
        assert ds.clinical_matrix().shape[0] == 3


def test_disconnectome_vae_runs_with_mse():
    cfg = vae_proto()
    cfg["representation"] = "disconnectome"
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
        assert "mse" in hist[-1] and np.isfinite(hist[-1]["mse"])
        assert "bce" not in hist[-1]                       # MSE continuo, no BCE+Dice
        assert os.path.exists(os.path.join(out, "vae_disconnectome.pt"))


def test_fusion_modes_and_dims():
    z_les = np.zeros((5, 8))
    z_dis = np.ones((5, 6))
    assert fuse_representation(z_les, None, "lesion").shape == (5, 8)
    assert fuse_representation(None, z_dis, "disconnectome").shape == (5, 6)
    assert fuse_representation(z_les, z_dis, "both").shape == (5, 14)
    with pytest.raises(ValueError):
        fuse_representation(z_les, np.ones((4, 6)), "both")   # no alineados
    with pytest.raises(ValueError):
        fuse_representation(None, None, "both")               # faltan ambos


def test_stage2_real_wiring_and_inference():
    enc_shape = (24, 28, 24)
    les_vae = ConvVAE3D(zdim=4, in_shape=enc_shape, channels=(8, 16, 32, 64))
    dis_vae = ConvVAE3D(zdim=4, in_shape=enc_shape, channels=(8, 16, 32, 64))
    cfg = {
        "seed": 0, "out_dir": tempfile.mkdtemp(), "fusion_mode": "both",
        "lesion_vae_ckpt": "", "disconnectome_vae_ckpt": "", "n_synth_fallback": 24,
        "data": {"lesion_root": None, "disconnectome_root": None, "atlas_dir": None,
                 "modality": "receptor", "encode_resolution": list(enc_shape),
                 "atlas_resolution": list(enc_shape)},
        "pfn": {"d_model": 32, "n_layers": 1, "n_col_layers": 1, "n_heads": 4,
                "n_bins": 64, "sigma": 0.05, "arch": "tabicl",
                "context_min": 16, "context_max": 24, "n_query": 6, "batch_size": 2,
                "iters": 3, "lr": 3e-4, "weight_decay": 0.01, "grad_clip": 1.0,
                "unobserved_strength": 0.0},
        "device": "cpu", "log_every": 50,
    }
    prior = build_real_prior(cfg, les_vae, dis_vae)
    assert prior.d_x == 8                                   # 2 * zdim en el modo both
    model = build_model(cfg, prior.d_x)
    batch = to_tensors(prior.sample_batch(2, n_context=16))
    loss = model.head.loss(model(batch["Xc"], batch["Tc"], batch["Yc"], batch["Xq"], batch["Tq"]),
                           batch["mu_q"])
    assert torch.isfinite(loss)

    ctxZ = np.random.randn(10, 8)
    ctxT = (np.arange(10) % 2).astype(float)
    ctxY = np.random.rand(10)
    qZ = np.random.randn(3, 8)
    out = infer_cate_real(model, ctxZ, ctxT, ctxY, qZ)
    assert out["cate"].shape == (3,) and out["ci_low"].shape == (3,)
    assert np.isfinite(out["cate"]).all()
