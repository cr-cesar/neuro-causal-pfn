"""The latent export: checkpoint round-trip, ordering, and naming."""
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")

_spec = importlib.util.spec_from_file_location(
    "export_latents",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "export_latents.py"))
ex = importlib.util.module_from_spec(_spec)
sys.modules["export_latents"] = ex
_spec.loader.exec_module(ex)

from neurocausalpfn.vae.conv3d_vae import ConvVAE3D  # noqa: E402

RES = (16, 16, 16)


def _save_ckpt(tmp_path, rep="disconnectome", zdim=4):
    cfg = {"data": {"resolution": list(RES)},
           "vae": {"zdim": zdim, "channels": [4, 8], "backbone": "cnn"}}
    model = ConvVAE3D(in_channels=1, zdim=zdim, in_shape=RES,
                      channels=(4, 8), backbone="cnn")
    path = os.path.join(tmp_path, f"vae_{rep}.pt")
    torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                "representation": rep, "in_channels": 1, "backbone": "cnn",
                "use_daft": False, "use_ard": False}, path)
    return path


def _save_images(tmp_path, n=5):
    img_dir = os.path.join(tmp_path, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    files = []
    for i in range(n):
        vol = rng.random(RES).astype(np.float32)
        p = os.path.join(img_dir, f"lesion{i:04d}_NA_NA.nii.gz")
        nib.save(nib.Nifti1Image(vol, np.eye(4)), p)
        files.append(p)
    return img_dir, files


def test_export_shape_and_determinism(tmp_path):
    ckpt = _save_ckpt(tmp_path)
    _, files = _save_images(tmp_path)
    model, rep, res = ex.load_frozen_vae(ckpt)
    assert rep == "disconnectome" and res == RES
    Z1 = ex.export_latents(model, rep, res, files, batch_size=2)
    Z2 = ex.export_latents(model, rep, res, files, batch_size=3)
    assert Z1.shape == (5, 4)
    np.testing.assert_allclose(Z1, Z2, atol=1e-5)   # batching must not matter


def test_export_rows_follow_file_order(tmp_path):
    ckpt = _save_ckpt(tmp_path)
    _, files = _save_images(tmp_path)
    model, rep, res = ex.load_frozen_vae(ckpt)
    Z = ex.export_latents(model, rep, res, files, batch_size=2)
    Zrev = ex.export_latents(model, rep, res, files[::-1], batch_size=2)
    np.testing.assert_allclose(Z, Zrev[::-1], atol=1e-5)


def test_daft_checkpoint_is_refused(tmp_path):
    path = _save_ckpt(tmp_path)
    ckpt = torch.load(path, weights_only=False)
    ckpt["use_daft"] = True
    torch.save(ckpt, path)
    with pytest.raises(SystemExit, match="DAFT"):
        ex.load_frozen_vae(path)


def test_bracketed_concrete_path_is_not_globbed(tmp_path):
    # experiment labels put [] in directory names; the shell hands the script
    # concrete paths that glob would misread as character classes
    d = os.path.join(tmp_path, "E2[w_dice=0.5]", "seed0", "disco")
    os.makedirs(d)
    ck = _save_ckpt(d)
    import subprocess
    r = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "..",
                                      "scripts", "export_latents.py"),
         "--checkpoints", ck, "--images-dir", os.path.join(tmp_path, "none"),
         "--out", str(tmp_path)],
        capture_output=True, text=True)
    # it must get past checkpoint resolution and fail on the empty images dir
    assert "no checkpoint matches" not in (r.stdout + r.stderr)
    assert "no niftis" in (r.stdout + r.stderr)


def test_default_name_from_runner_layout():
    name = ex.default_name(
        "outputs/experiments/E2/E2[w_dice=0.5]/seed0/disco/vae_disconnectome.pt")
    assert name == "E2_E2_w_dice=0.5_seed0_disco.npz"
