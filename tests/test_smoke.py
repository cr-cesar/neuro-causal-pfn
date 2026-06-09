"""Prueba de humo de principio a fin en modo prototipo.

Corre en CPU en segundos y confirma que ambas etapas se entrenan, que las
perdidas son finitas y que la inferencia del efecto del tratamiento produce
salidas con la forma correcta.
"""
import math

import torch

from neurocausalpfn.pfn.inference import credible_interval, predict_cate
from neurocausalpfn.train.train_pfn import prototype_config as pfn_config
from neurocausalpfn.train.train_pfn import quick_eval, run_pfn
from neurocausalpfn.train.train_vae import prototype_config as vae_config
from neurocausalpfn.train.train_vae import run_vae


def test_vae_prototype_runs_and_reconstructs():
    cfg = vae_config()
    cfg["vae"]["epochs"] = 2
    cfg["data"]["n_synth"] = 8
    cfg["out_dir"] = "outputs/test_vae"
    model, history = run_vae(cfg)
    assert math.isfinite(history[-1]["total"])
    x = torch.zeros(1, 1, *cfg["data"]["resolution"])
    logits, mu, logvar, z = model(x)
    assert logits.shape == x.shape
    assert z.shape == (1, cfg["vae"]["zdim"])


def test_pfn_prototype_runs_and_predicts():
    cfg = pfn_config()
    cfg["pfn"]["iters"] = 30
    cfg["pfn"]["batch_size"] = 4
    cfg["pfn"]["context_max"] = 128
    cfg["out_dir"] = "outputs/test_pfn"
    model, history = run_pfn(cfg)
    assert all(math.isfinite(h["loss"]) for h in history)

    metrics = quick_eval(model, cfg, n_eval=4)
    assert math.isfinite(metrics["root_pehe"])
    assert 0.0 <= metrics["prescriptive_accuracy"] <= 1.0


def test_pfn_inference_shapes():
    cfg = pfn_config()
    cfg["pfn"]["iters"] = 5
    cfg["pfn"]["batch_size"] = 2
    cfg["pfn"]["context_max"] = 64
    cfg["out_dir"] = "outputs/test_pfn2"
    model, _ = run_pfn(cfg)

    B, n_ctx, n_qry, d_x = 2, 32, 6, cfg["pfn"]["d_x"]
    Xc = torch.randn(B, n_ctx, d_x)
    Tc = (torch.rand(B, n_ctx) > 0.5).float()
    Yc = torch.rand(B, n_ctx)
    Xq = torch.randn(B, n_qry, d_x)
    out = predict_cate(model, Xc, Tc, Yc, Xq)
    assert out["cate"].shape == (B, n_qry)
    lo, hi = credible_interval(model.head, out["logits1"])
    assert lo.shape == (B, n_qry) and hi.shape == (B, n_qry)
