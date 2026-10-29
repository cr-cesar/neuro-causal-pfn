"""29 Oct, checks against the reference code: the disconnectome VAE can train
on the Bernoulli likelihood (E1b) instead of MSE, and the replica's logistic
regression can use the released code's sklearn default of 100 iterations."""
import os
import tempfile

import numpy as np
import pytest

from neurocausalpfn.experiments.registry import get_experiment
from neurocausalpfn.experiments.runner import build_runs


def test_e1b_pins_the_bernoulli_disconnectome_and_e1_is_unchanged():
    (e1b,) = build_runs(get_experiment("E1b"), "full", {"dims": (100, 100), "use_ard": True})
    m = e1b.meta
    assert e1b.label == "E1b" and m["disco_recon"] == "bce" and m["fusion_mode"] == "disconnectome"
    assert (m["d_lesion"], m["d_disco"]) == (50, 50) and m["use_ard"] is False and m["w_dice"] == 0.0
    (e1,) = build_runs(get_experiment("E1"), "full", {"dims": (100, 100), "use_ard": True})
    assert "disco_recon" not in e1.meta                      # earlier runs keep their metadata


def test_lr_max_iter_follows_the_environment(monkeypatch):
    from neurocausalpfn.prior.giles_replica import _classifier

    monkeypatch.delenv("GILES_LR_MAX_ITER", raising=False)
    assert _classifier("logistic_regression").max_iter == 1000
    monkeypatch.setenv("GILES_LR_MAX_ITER", "100")
    assert _classifier("logistic_regression").max_iter == 100


def test_disconnectome_vae_trains_with_bce_when_asked():
    pytest.importorskip("torch")
    from neurocausalpfn.train.train_vae import prototype_config, run_vae

    cfg = prototype_config()
    cfg["representation"] = "disconnectome"
    cfg["data"].update({"resolution": [24, 28, 24], "n_synth": 8, "val_frac": 0.25})
    cfg["vae"].update({"epochs": 2, "batch_size": 2, "channels": [8, 16, 32, 64], "zdim": 8,
                       "disco_recon": "bce"})
    with tempfile.TemporaryDirectory() as out:
        cfg["out_dir"] = out
        _, hist = run_vae(cfg)
        last = hist[-1]
        assert "bce" in last and np.isfinite(last["bce"]) and "mse" not in last
        # no Dice in the objective: the reconstruction term is the BCE alone
        assert last["rec"] == pytest.approx(last["bce"], rel=1e-5)
        assert os.path.exists(os.path.join(out, "vae_disconnectome.pt"))
