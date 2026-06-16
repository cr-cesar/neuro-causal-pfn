"""Cableado de la Etapa 2 sobre datos reales.

Une las dos etapas: carga los encoders congelados de la Etapa 1, calcula el
latente de cada paciente (lesion y, opcionalmente, disconnectoma), los fusiona
segun la variante elegida, construye el Neuro-Prior anclado en anatomia pasando
esos latentes reales como covariable (z_pool) y las lesiones en su rejilla
nativa para los solapamientos con el atlas, y entrena el transformer.

Tambien expone la inferencia sobre datos reales: dada una cohorte real como
contexto y un paciente nuevo como consulta, devuelve su efecto individualizado
(CATE) con un intervalo creible.
"""
import os
from typing import Dict, Optional

import numpy as np
import torch

from ..data.nifti_dataset import (PairedLesionDisconnectomeDataset, _load_nifti)
from ..data.transforms import binarize
from ..pfn.inference import credible_interval, predict_cate
from ..pfn.tokens import to_tensors
from ..prior.atlas import FunctionalAtlas
from ..prior.cohort import NeuroPriorInterSynth, build_synthetic_lesion_pool
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from ..vae.conv3d_vae import ConvVAE3D
from ..vae.fusion import compute_latents, fuse_representation
from .train_pfn import _context_length, build_model

log = get_logger()


def stage2_real_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/pfn_real",
        "fusion_mode": "both",                 # lesion, disconnectome o both
        "lesion_vae_ckpt": "outputs/vae_full_lesion/vae_lesion.pt",
        "disconnectome_vae_ckpt": "outputs/vae_full_disconnectome/vae_disconnectome.pt",
        "data": {"lesion_root": "data/lesions", "disconnectome_root": "data/disconnectomes",
                 "atlas_dir": "data/atlases", "modality": "receptor",
                 "encode_resolution": [96, 112, 96], "atlas_resolution": [91, 109, 91]},
        "pfn": {"d_model": 512, "n_layers": 12, "n_col_layers": 3, "n_heads": 8,
                "n_bins": 1024, "sigma": 0.02, "arch": "tabicl",
                "context_min": 1000, "context_max": 20000, "n_query": 64,
                "batch_size": 8, "iters": 162000, "lr": 3e-4,
                "weight_decay": 0.01, "grad_clip": 1.0, "unobserved_strength": 0.0},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_every": 200,
    }


def load_vae(ckpt_path: str, device: str = "cpu") -> ConvVAE3D:
    ck = torch.load(ckpt_path, map_location=device)
    v = ck["cfg"]["vae"]
    res = tuple(ck["cfg"]["data"]["resolution"])
    model = ConvVAE3D(zdim=v["zdim"], in_shape=res, channels=tuple(v["channels"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def native_lesion_pool(paired: PairedLesionDisconnectomeDataset, atlas_shape, seed: int = 0) -> np.ndarray:
    """Mascaras de lesion en la rejilla del atlas (nativa), para los solapamientos."""
    if paired.synthetic:
        return build_synthetic_lesion_pool(len(paired), shape=atlas_shape, seed=seed)
    pool = [binarize(_load_nifti(lp, atlas_shape)) for lp, _ in paired.pairs]
    return np.stack(pool, axis=0)


def encode_and_fuse(lesion_vae, disconnectome_vae, paired: PairedLesionDisconnectomeDataset,
                    mode: str, device: str = "cpu", batch_size: int = 8) -> np.ndarray:
    """Latente fusionado [N, d_x] por paciente segun la variante."""
    z_les = z_dis = None
    if mode in ("lesion", "both"):
        z_les = compute_latents(lesion_vae, paired, device=device, batch_size=batch_size, item_index=0)
    if mode in ("disconnectome", "both"):
        z_dis = compute_latents(disconnectome_vae, paired, device=device, batch_size=batch_size, item_index=1)
    return fuse_representation(z_les, z_dis, mode)


def build_real_prior(cfg: Dict, lesion_vae, disconnectome_vae) -> NeuroPriorInterSynth:
    d = cfg["data"]
    enc_shape = tuple(d.get("encode_resolution", [96, 112, 96]))
    atlas_shape = tuple(d.get("atlas_resolution", [91, 109, 91]))
    paired = PairedLesionDisconnectomeDataset(
        lesion_root=d.get("lesion_root"), disconnectome_root=d.get("disconnectome_root"),
        in_shape=enc_shape, n_synth=cfg.get("n_synth_fallback", 64), seed=cfg["seed"])
    z_pool = encode_and_fuse(lesion_vae, disconnectome_vae, paired, cfg["fusion_mode"],
                             device=cfg.get("device", "cpu"), batch_size=cfg["pfn"]["batch_size"])
    pool = native_lesion_pool(paired, atlas_shape, seed=cfg["seed"])
    atlas = FunctionalAtlas.from_dir(d.get("atlas_dir"), shape=atlas_shape, seed=cfg["seed"],
                                     modality=d.get("modality", "receptor"))
    log.info("Etapa 2 real: %d pacientes, fusion=%s, d_x=%d", len(z_pool), cfg["fusion_mode"], z_pool.shape[1])
    return NeuroPriorInterSynth(atlas, pool, seed=cfg["seed"], z_pool=z_pool,
                                n_context=cfg["pfn"]["context_max"], n_query=cfg["pfn"]["n_query"],
                                unobserved_strength=cfg["pfn"].get("unobserved_strength", 0.0))


def run_stage2_real(cfg: Dict):
    set_seed(cfg["seed"])
    device = cfg.get("device", "cpu")
    p = cfg["pfn"]

    lesion_vae = load_vae(cfg["lesion_vae_ckpt"], device) if os.path.exists(cfg["lesion_vae_ckpt"]) else None
    disconnectome_vae = None
    if cfg["fusion_mode"] in ("disconnectome", "both") and os.path.exists(cfg["disconnectome_vae_ckpt"]):
        disconnectome_vae = load_vae(cfg["disconnectome_vae_ckpt"], device)

    prior = build_real_prior(cfg, lesion_vae, disconnectome_vae)
    model = build_model(cfg, prior.d_x).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    log.info("PFN real: %.2fM parametros, arch=%s", sum(t.numel() for t in model.parameters()) / 1e6,
             p.get("arch", "tabicl"))

    history = []
    model.train()
    for it in range(p["iters"]):
        n_ctx = _context_length(cfg, it)
        batch = to_tensors(prior.sample_batch(p["batch_size"], n_context=n_ctx), device=device)
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
    ckpt = os.path.join(cfg["out_dir"], "pfn_real.pt")
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "d_x": prior.d_x}, ckpt)
    log.info("checkpoint guardado en %s", ckpt)
    return model, history


@torch.no_grad()
def infer_cate_real(model, context_Z: np.ndarray, context_T: np.ndarray, context_Y: np.ndarray,
                    query_Z: np.ndarray, device: str = "cpu", lo: float = 0.05, hi: float = 0.95):
    """Inferencia sobre datos reales. context_* describen la cohorte observada
    (latentes, tratamiento y desenlace); query_Z son los latentes de los pacientes
    a evaluar. Devuelve el CATE y un intervalo creible por paciente."""
    batch = to_tensors({
        "Xc": np.asarray(context_Z)[None], "Tc": np.asarray(context_T)[None],
        "Yc": np.asarray(context_Y)[None], "Xq": np.asarray(query_Z)[None],
        "Tq": np.zeros((1, len(query_Z))), "mu_q": np.zeros((1, len(query_Z))),
        "mu0": np.zeros((1, len(query_Z))), "mu1": np.zeros((1, len(query_Z)))}, device=device)
    out = predict_cate(model, batch["Xc"], batch["Tc"], batch["Yc"], batch["Xq"])
    lo_q, hi_q = credible_interval(model.head, out["logits1"], lo, hi)
    return {"cate": out["cate"][0].cpu().numpy(), "mu0": out["mu0"][0].cpu().numpy(),
            "mu1": out["mu1"][0].cpu().numpy(),
            "ci_low": lo_q[0].cpu().numpy(), "ci_high": hi_q[0].cpu().numpy()}


if __name__ == "__main__":
    run_stage2_real(stage2_real_config())
