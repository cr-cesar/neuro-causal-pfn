"""Entrenamiento de la Etapa 2 (el transformer causal).

Entrena con lotes muestreados del Neuro-Prior minimizando la perdida de
histograma sobre el resultado potencial esperado condicional verdadero. La
longitud de contexto sigue un curriculo de menor a mayor.

El prior se elige por configuracion (cfg["prior"]["kind"]):
- "synthetic": el generador ligero (covariables gaussianas desde cero).
- "intersynth": el mecanismo anatomico real, que cruza las lesiones con la
  parcelacion funcional. En este modo, d_x se deriva del prior (el latente del
  encoder si se proporciona, o las covariables observadas en su defecto).
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


def build_model(cfg: Dict, d_x: int):
    """Construye el transformer de la Etapa 2 segun la arquitectura pedida:
    'linear' (proyeccion por fila) o 'tabicl' (atencion por columna y luego por
    fila)."""
    p = cfg["pfn"]
    if p.get("arch", "linear") == "tabicl":
        from ..pfn.tabicl_model import NeuroCausalPFNTabICL

        return NeuroCausalPFNTabICL(
            d_x=d_x, d_model=p["d_model"], n_row_layers=p["n_layers"],
            n_col_layers=p.get("n_col_layers", 2), n_heads=p["n_heads"],
            n_bins=p["n_bins"], sigma=p["sigma"])
    return NeuroCausalPFN(
        d_x=d_x, d_model=p["d_model"], n_layers=p["n_layers"], n_heads=p["n_heads"],
        n_bins=p["n_bins"], sigma=p["sigma"])


def prototype_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/pfn_prototype",
        "prior": {"kind": "synthetic"},
        "pfn": {"d_x": 16, "d_model": 128, "n_layers": 2, "n_heads": 4,
                "n_bins": 256, "sigma": 0.02, "arch": "linear", "n_col_layers": 2,
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
        # cambiar kind a "intersynth" para usar el sustrato anatomico real;
        # apuntar atlas_dir a data/atlases y, idealmente, pasar los latentes del
        # encoder como z_pool desde un script de orquestacion.
        "prior": {"kind": "synthetic",
                  "atlas_dir": "data/atlases", "atlas_shape": [96, 112, 96],
                  "modality": "receptor",
                  "pool_size": 4000, "unobserved_strength": 0.0},
        "pfn": {"d_x": 104, "d_model": 512, "n_layers": 12, "n_heads": 8,
                "n_bins": 1024, "sigma": 0.02, "arch": "tabicl", "n_col_layers": 3,
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


def _build_prior(cfg: Dict, seed_offset: int = 0):
    """Devuelve (objeto_prior_o_None, d_x, es_intersynth). Para el prior
    sintetico el objeto es None (se reinstancia por iteracion); para InterSynth
    se construye una sola vez y se reutiliza."""
    p = cfg["pfn"]
    pr = cfg.get("prior", {"kind": "synthetic"})
    if pr.get("kind") == "intersynth":
        from ..prior.atlas import FunctionalAtlas
        from ..prior.cohort import NeuroPriorInterSynth, build_synthetic_lesion_pool

        shape = tuple(pr.get("atlas_shape", [48, 56, 48]))
        seed = cfg["seed"] + seed_offset
        modality = pr.get("modality", "receptor")
        atlas = FunctionalAtlas.from_dir(pr.get("atlas_dir"), shape=shape, seed=seed, modality=modality)
        shape = atlas.shape   # el conjunto de lesiones debe vivir en la rejilla del atlas
        pool = build_synthetic_lesion_pool(int(pr.get("pool_size", 128)), shape=shape, seed=seed)
        prior = NeuroPriorInterSynth(atlas, pool, seed=seed,
                                     n_context=p["context_max"], n_query=p["n_query"],
                                     unobserved_strength=float(pr.get("unobserved_strength", 0.0)))
        return prior, prior.d_x, True
    return None, p["d_x"], False


def run_pfn(cfg: Dict):
    set_seed(cfg["seed"])
    device = cfg.get("device", "cpu")
    p = cfg["pfn"]

    prior_obj, d_x, is_intersynth = _build_prior(cfg)
    model = build_model(cfg, d_x).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    n_params = sum(t.numel() for t in model.parameters())
    log.info("PFN: %.2fM parametros, d_x=%d, arch=%s, prior=%s",
             n_params / 1e6, d_x, p.get("arch", "linear"), "intersynth" if is_intersynth else "synthetic")

    history = []
    model.train()
    for it in range(p["iters"]):
        n_ctx = _context_length(cfg, it)
        if is_intersynth:
            batch_np = prior_obj.sample_batch(p["batch_size"], n_context=n_ctx)
        else:
            prior = NeuroPrior(d_x=d_x, n_context=n_ctx, n_query=p["n_query"], seed=cfg["seed"] + it)
            batch_np = prior.sample_batch(p["batch_size"])
        batch = to_tensors(batch_np, device=device)
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
    """Evaluacion rapida sobre procesos reservados (con el mismo tipo de prior)."""
    p = cfg["pfn"]
    device = cfg.get("device", "cpu")
    prior_obj, d_x, is_intersynth = _build_prior(cfg, seed_offset=10_000)
    if is_intersynth:
        batch_np = prior_obj.sample_batch(n_eval, n_context=p["context_max"])
    else:
        prior = NeuroPrior(d_x=d_x, n_context=p["context_max"], n_query=p["n_query"],
                           seed=10_000 + cfg["seed"])
        batch_np = prior.sample_batch(n_eval)
    batch = to_tensors(batch_np, device=device)
    out = predict_cate(model, batch["Xc"], batch["Tc"], batch["Yc"], batch["Xq"])
    cate_true = batch["mu1"] - batch["mu0"]
    return {"root_pehe": root_pehe(out["cate"], cate_true),
            "prescriptive_accuracy": prescriptive_accuracy(out["cate"], cate_true)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prototype", choices=["prototype", "full"])
    ap.add_argument("--prior", default=None, choices=["synthetic", "intersynth"],
                    help="sobrescribe cfg['prior']['kind']")
    ap.add_argument("--arch", default=None, choices=["linear", "tabicl"],
                    help="sobrescribe cfg['pfn']['arch']")
    args = ap.parse_args()
    cfg = prototype_config() if args.mode == "prototype" else full_config()
    if args.prior is not None:
        cfg.setdefault("prior", {})["kind"] = args.prior
    if args.arch is not None:
        cfg["pfn"]["arch"] = args.arch
    trained, _ = run_pfn(cfg)
    log.info("evaluacion rapida: %s", quick_eval(trained, cfg))
