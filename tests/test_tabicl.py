import tempfile

import numpy as np
import torch

from neurocausalpfn.pfn.tabicl_model import NeuroCausalPFNTabICL
from neurocausalpfn.train.train_pfn import build_model, prototype_config, run_pfn


def _toy_batch(B=2, nc=12, nq=4, d_x=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    Xc = torch.randn(B, nc, d_x, generator=g)
    Tc = (torch.rand(B, nc, generator=g) > 0.5).float()
    Yc = torch.rand(B, nc, generator=g)
    Xq = torch.randn(B, nq, d_x, generator=g)
    Tq = (torch.rand(B, nq, generator=g) > 0.5).float()
    return Xc, Tc, Yc, Xq, Tq


def test_tabicl_forward_and_train():
    d_x = 5
    model = NeuroCausalPFNTabICL(d_x=d_x, d_model=32, n_row_layers=1, n_col_layers=1,
                                 n_heads=4, n_bins=64, sigma=0.05)
    Xc, Tc, Yc, Xq, Tq = _toy_batch(d_x=d_x)
    logits = model(Xc, Tc, Yc, Xq, Tq)
    assert logits.shape == (2, 4, 64)

    mu = torch.rand(2, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        loss = model.head.loss(model(Xc, Tc, Yc, Xq, Tq), mu)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)


def test_tabicl_query_independence():
    # la mascara causal aisla las consultas: perturbar una consulta no afecta a otra,
    # mientras que cambiar el contexto si cambia las predicciones.
    torch.manual_seed(0)
    d_x = 4
    model = NeuroCausalPFNTabICL(d_x=d_x, d_model=24, n_row_layers=1, n_col_layers=1,
                                 n_heads=4, n_bins=32, sigma=0.05).eval()
    Xc, Tc, Yc, Xq, Tq = _toy_batch(B=1, nc=8, nq=3, d_x=d_x, seed=1)
    with torch.no_grad():
        base = model(Xc, Tc, Yc, Xq, Tq)
        Xq2 = Xq.clone(); Xq2[:, 2, :] += 5.0
        Tq2 = Tq.clone(); Tq2[:, 2] = 1.0 - Tq2[:, 2]
        pert = model(Xc, Tc, Yc, Xq2, Tq2)
        assert torch.allclose(base[:, 0], pert[:, 0], atol=1e-5)      # consulta 0 intacta
        assert not torch.allclose(base[:, 2], pert[:, 2], atol=1e-4)  # consulta 2 cambia

        Yc2 = Yc.clone(); Yc2[:, 0] += 1.0
        ctx_changed = model(Xc, Tc, Yc2, Xq, Tq)
        assert not torch.allclose(base[:, 0], ctx_changed[:, 0], atol=1e-4)  # el contexto influye


def test_build_model_selects_arch():
    cfg = prototype_config()
    cfg["pfn"]["arch"] = "tabicl"
    assert isinstance(build_model(cfg, d_x=6), NeuroCausalPFNTabICL)
    cfg["pfn"]["arch"] = "linear"
    assert not isinstance(build_model(cfg, d_x=6), NeuroCausalPFNTabICL)


def test_run_pfn_smoke_tabicl():
    cfg = prototype_config()
    cfg["pfn"].update({"arch": "tabicl", "d_model": 32, "n_layers": 1, "n_col_layers": 1,
                       "n_heads": 4, "n_bins": 64, "context_min": 16, "context_max": 24,
                       "n_query": 4, "batch_size": 2, "iters": 3})
    cfg["log_every"] = 50
    with tempfile.TemporaryDirectory() as out:
        cfg["out_dir"] = out
        _, hist = run_pfn(cfg)
        assert np.isfinite(hist[-1]["loss"])
