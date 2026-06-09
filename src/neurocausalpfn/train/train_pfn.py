"""Entrenamiento de la Etapa 2 (el transformer causal).

Entrena con lotes muestreados del Neuro-Prior minimizando la perdida de
histograma sobre el resultado potencial esperado condicional verdadero. La
longitud de contexto sigue un curriculo de menor a mayor. Igual que en la Etapa
1, un unico punto de entrada sirve al prototipo y al modo completo.
"""
import os
from typing import Dict

import numpy as np
import torch

from ..eval.metrics import prescriptive_accuracy, root_pehe
from ..prior.cohort import NeuroPrior
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from ..pfn.inference import predict_cate
from ..pfn.model import NeuroCausalPFN
from ..pfn.tokens import to_tensors

log = get_logger()


def prototype_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/pfn_prototype",
        "pfn": {"d_x": 16, "d_model": 128, "n_layers": 2, "n_heads": 4,
                "n_bins": 256, "sigma": 0.02,
                "context_min": 64, "context_max": 256, "n_query": 16,
                "batch_size": 8, "iters": 2000,
                "lr": 3e-4, "weight_decay": 0.01, "grad_clip": 1.0},
        "device": "cpu",
        "log_every": 100,
    }


def full_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/pfn_full",
        "pfn": {"d_x": 104, "d_model": 512, "n_layers": 12, "n_heads": 8,
                "n_bins": 1024, "sigma": 0.02,
                "context_min": 1000, "context_max": 20000, "n_query": 64,
                "batch_size": 8, "iters": 162000,
                "lr": 3e-4, "weight_decay": 0.01, "grad_clip": 1.0},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_every": 200,
    }


def _context_length(cfg: Dict, it: int) -> int:
    """Curriculo lineal de longitud de contexto de menor a mayor."""
    p = cfg["pfn"]
    frac = min(1.0, (it + 1) / max(1, int(0.5 * p["iters"])))
    return int(p["context_min"] + frac * (p["context_max"] - p["context_min"]))


def run_pfn(cfg: Dict):
    set_seed(cfg["seed"])
    device = cfg.get("device", "cpu")
    p = cfg["pfn"]

    model = NeuroCausalPFN(d_x=p["d_x"], d_model=p["d_model"], n_layers=p["n_layers"],
                           n_heads=p["n_heads"], n_bins=p["n_bins"], sigma=p["sigma"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    n_params = sum(t.numel() for t in model.parameters())
    log.info("PFN: %.2fM parametros, d_x=%d, %d capas", n_params / 1e6, p["d_x"], p["n_layers"])

    history = []
    model.train()
    for it in range(p["iters"]):
        n_ctx = _context_length(cfg, it)
        prior = NeuroPrior(d_x=p["d_x"], n_context=n_ctx, n_query=p["n_query"],
                           seed=cfg["seed"] + it)
        batch = to_tensors(prior.sample_batch(p["batch_size"]), device=device)
        logits = model(batch["Xc"], batch["Tc"], batch["Yc"], batch["Xq"], batch["Tq"])
        loss = model.head.loss(logits, batch["mu_q"])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        opt.step()

        history.append({"iter": it, "loss": float(loss.detach()), "n_ctx": n_ctx})
        if (it + 1) % cfg["log_every"] == 0 or it == 0:
            log.info("iter %d/%d  n_ctx=%d  loss=%.4f", it + 1, p["iters"], n_ctx, float(loss.detach()))

    os.makedirs(cfg["out_dir"], exist_ok=True)
    ckpt = os.path.join(cfg["out_dir"], "pfn.pt")
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, ckpt)
    log.info("checkpoint guardado en %s", ckpt)
    return model, history


@torch.no_grad()
def quick_eval(model, cfg: Dict, n_eval: int = 8) -> Dict[str, float]:
    """Evaluacion rapida sobre procesos sinteticos reservados."""
    p = cfg["pfn"]
    prior = NeuroPrior(d_x=p["d_x"], n_context=p["context_max"], n_query=p["n_query"],
                       seed=10_000 + cfg["seed"])
    batch = to_tensors(prior.sample_batch(n_eval), device=cfg.get("device", "cpu"))
    out = predict_cate(model, batch["Xc"], batch["Tc"], batch["Yc"], batch["Xq"])
    cate_true = (batch["mu1"] - batch["mu0"])
    return {"root_pehe": root_pehe(out["cate"], cate_true),
            "prescriptive_accuracy": prescriptive_accuracy(out["cate"], cate_true)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prototype", choices=["prototype", "full"])
    args = ap.parse_args()
    cfg = prototype_config() if args.mode == "prototype" else full_config()
    trained, _ = run_pfn(cfg)
    if args.mode == "prototype":
        log.info("evaluacion rapida: %s", quick_eval(trained, cfg))
