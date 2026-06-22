import os
import tempfile

import numpy as np
import pytest
import torch

from neurocausalpfn.train.run_stage2_real import load_vae
from neurocausalpfn.train.train_vae import prototype_config as vae_proto, run_vae
from neurocausalpfn.vae.backbones import build_encoder_backbone
from neurocausalpfn.vae.conv3d_vae import ConvVAE3D
from neurocausalpfn.vae.losses import vae_loss

SHAPE = (24, 28, 24)
BACKBONES = ["cnn", "wide", "resnet", "resnet18", "resnet50"]


def _nparams(m):
    return sum(p.numel() for p in m.parameters())


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        build_encoder_backbone("does_not_exist", 1, (16, 32, 64, 128, 256))


@pytest.mark.parametrize("backbone", BACKBONES)
def test_backbone_forward_shapes(backbone):
    model = ConvVAE3D(in_channels=1, zdim=16, in_shape=SHAPE,
                      channels=(16, 32, 64, 128, 256), backbone=backbone)
    x = torch.rand(2, 1, *SHAPE)
    logits, mu, logvar, _ = model(x)
    assert logits.shape == x.shape
    assert mu.shape == (2, 16) and logvar.shape == (2, 16)


def test_param_count_ordering():
    def total(bb):
        return _nparams(ConvVAE3D(in_channels=1, zdim=16, in_shape=SHAPE,
                                  channels=(16, 32, 64, 128, 256), backbone=bb))
    assert total("resnet50") > total("resnet18") > total("cnn")


def test_residual_backbone_trains():
    model = ConvVAE3D(in_channels=1, zdim=16, in_shape=SHAPE,
                      channels=(16, 32, 64, 128, 256), backbone="resnet")
    x = torch.rand(2, 1, *SHAPE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    logits, mu, logvar, _ = model(x)
    loss, _ = vae_loss(logits, x, mu, logvar, beta=1.0)
    opt.zero_grad(); loss.backward(); opt.step()
    assert torch.isfinite(loss)


def test_backbone_combines_with_daft():
    model = ConvVAE3D(in_channels=1, zdim=16, in_shape=SHAPE,
                      channels=(16, 32, 64, 128, 256), backbone="resnet",
                      use_daft=True, n_clinical=4)
    x = torch.rand(2, 1, *SHAPE)
    clin = torch.randn(2, 4)
    logits, mu, _, _ = model(x, clin)
    assert logits.shape == x.shape and mu.shape == (2, 16)


def test_run_vae_with_backbone():
    cfg = vae_proto()
    cfg["vae"]["backbone"] = "resnet"
    cfg["data"]["resolution"] = list(SHAPE)
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
        assert ckpt["backbone"] == "resnet"


def test_load_vae_restores_backbone():
    model = ConvVAE3D(in_channels=1, zdim=8, in_shape=SHAPE,
                      channels=(8, 16, 32, 64), backbone="resnet18")
    ckpt = {"cfg": {"vae": {"zdim": 8, "channels": [8, 16, 32, 64]},
                    "data": {"resolution": list(SHAPE)}},
            "state_dict": model.state_dict(), "in_channels": 1,
            "backbone": "resnet18", "use_daft": False, "n_clinical": 0, "use_ard": False}
    with tempfile.TemporaryDirectory() as out:
        path = os.path.join(out, "vae.pt")
        torch.save(ckpt, path)
        loaded = load_vae(path)
        assert loaded.backbone == "resnet18"
        z = loaded.encode_mean(torch.rand(2, 1, *SHAPE))
        assert z.shape == (2, 8)
