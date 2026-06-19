import os
import tempfile

import numpy as np
import nibabel as nib
import torch

from neurocausalpfn.data.clinical import (CLINICAL_DIM_EXTENDED, build_clinical_vector_extended,
                                          load_clinical_table)
from neurocausalpfn.data.nifti_dataset import LesionMaskDataset
from neurocausalpfn.train.train_vae import prototype_config as vae_proto, run_vae
from neurocausalpfn.vae.conv3d_vae import ConvVAE3D
from neurocausalpfn.vae.losses import vae_loss


def _write_nifti(path, arr):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4)), path)


def test_extended_clinical_vector():
    full = build_clinical_vector_extended(70.0, "M", nihss=12, time_to_scan=3.0)
    assert full.shape == (CLINICAL_DIM_EXTENDED,)
    assert full[5] == 0.0 and full[7] == 0.0          # nihss and tts present -> indicator 0
    missing = build_clinical_vector_extended(70.0, "M", nihss=None, time_to_scan=None)
    assert missing[4] == 0.0 and missing[5] == 1.0    # nihss missing -> value 0, indicator 1
    assert missing[6] == 0.0 and missing[7] == 1.0    # tts missing -> value 0, indicator 1


def test_clinical_table_and_dataset_dim():
    shape = (24, 28, 24)
    with tempfile.TemporaryDirectory() as les_dir:
        rng = np.random.default_rng(0)
        for i in (1, 2):
            _write_nifti(os.path.join(les_dir, f"lesion{i:04d}_70_M.nii.gz"),
                         (rng.random(shape) > 0.7).astype(np.float32))
        csv = os.path.join(les_dir, "clinical.csv")
        with open(csv, "w") as f:
            f.write("id,nihss,time_to_scan\n0001,12,3.0\n0002,NA,5.0\n")
        table = load_clinical_table(csv)
        assert table["0001"]["nihss"] == "12"
        assert table["0002"]["nihss"] is None         # NA -> missing

        with_csv = LesionMaskDataset(les_dir, in_shape=shape, clinical_csv=csv)
        assert with_csv.clinical_dim() == CLINICAL_DIM_EXTENDED   # 8
        without_csv = LesionMaskDataset(les_dir, in_shape=shape)
        assert without_csv.clinical_dim() == 4


def test_daft_vae_forward_and_train():
    shape = (24, 28, 24)
    model = ConvVAE3D(in_channels=1, zdim=8, in_shape=shape, channels=(8, 16, 32, 64),
                      use_daft=True, n_clinical=4)
    x = torch.rand(2, 1, *shape)
    clin = torch.randn(2, 4)
    logits, mu, logvar, _ = model(x, clin)
    assert logits.shape == x.shape and mu.shape == (2, 8)

    # backward compatible: without a clinical vector the DAFT block is skipped
    logits2, _, _, _ = model(x)
    assert logits2.shape == x.shape

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # one training step
    logits, mu, logvar, _ = model(x, clin)
    loss, _ = vae_loss(logits, x, mu, logvar, beta=1.0)
    opt.zero_grad(); loss.backward(); opt.step()
    assert torch.isfinite(loss)


def test_run_vae_with_daft_and_conditioned_export():
    cfg = vae_proto()
    cfg["vae"]["use_daft"] = True
    cfg["export"] = True
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
        assert np.isfinite(hist[-1]["total"])
        ckpt = torch.load(os.path.join(out, "vae_lesion.pt"), map_location="cpu")
        assert ckpt["use_daft"] is True and ckpt["n_clinical"] == 4
        # conditioned export wrote the latents
        npz = np.load(os.path.join(out, "latents_lesion.npz"))
        assert npz["Z"].shape[1] == 8
