"""E9b training: DMVAE (shared plus private latents).

Trains the shared-private decomposition on paired lesion and disconnectome
volumes. The shared latent is fused across modalities by a product of experts;
each modality is reconstructed from its [shared, private] pair (BCE plus soft
Dice for the binary lesion, MSE for the continuous disconnectome). Beta warms up
the KL, and lambda_priv scales the disentangling pressure on the private latents.
"""
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..data.nifti_dataset import PairedLesionDisconnectomeDataset
from ..utils.logging_utils import get_logger
from ..utils.runtime import (autocast_ctx, log_runtime, make_grad_scaler,
                             make_loader, optim_step, resolve_device, use_amp)
from ..utils.seed import set_seed
from ..vae.dmvae import DMVAE3D

log = get_logger()


def prototype_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/dmvae_prototype",
        "export": False,
        "data": {"lesion_root": None, "disconnectome_root": None,
                 "resolution": [24, 28, 24], "n_synth": 16, "val_frac": 0.25},
        "model": {"shared_dim": 8, "private_dim": 4, "channels": [16, 32, 64, 128, 256],
                  "backbone": "cnn"},
        "train": {"batch_size": 4, "epochs": 3, "lr": 1e-3, "beta_max": 1.0,
                  "warmup_frac": 0.2, "lambda_priv": 1.0},
        "device": "cpu",
    }


def full_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/dmvae_full",
        "export": True,
        "data": {"lesion_root": "data/lesions", "disconnectome_root": "data/disconnectomes",
                 "resolution": [96, 112, 96], "n_synth": 0, "val_frac": 0.1},
        "model": {"shared_dim": 50, "private_dim": 25, "channels": [16, 32, 64, 128, 256],
                  "backbone": "cnn"},
        "train": {"batch_size": 8, "epochs": 100, "lr": 1e-4, "beta_max": 1.0,
                  "warmup_frac": 0.2, "lambda_priv": 1.0},
        "device": "auto",
        "amp": True,
        "num_workers": 4,
    }


def _split(n: int, val_frac: float, seed: int):
    idx = np.random.default_rng(seed).permutation(n)
    n_val = int(val_frac * n)
    if n_val < 1:
        return idx, idx
    return idx[n_val:], idx[:n_val]


def _beta(epoch, epochs, beta_max, warmup_frac):
    w = max(1, int(warmup_frac * epochs))
    return beta_max * min(1.0, (epoch + 1) / w)


def _step(model, les, dis, beta, lambda_priv, device, train, opt=None, scaler=None, amp=False) -> Dict:
    les, dis = les.to(device), dis.to(device)
    with autocast_ctx(device, amp):
        loss, parts = model(les, dis, beta=beta, lambda_priv=lambda_priv)
    if train:
        optim_step(loss, opt, scaler)
    return parts


def run_dmvae(cfg: Dict):
    set_seed(cfg["seed"])
    device = resolve_device(cfg)
    amp = use_amp(cfg, device)
    scaler = make_grad_scaler(amp)
    workers = int(cfg.get("num_workers", 0))
    in_shape = tuple(cfg["data"]["resolution"])
    m = cfg["model"]
    dataset = PairedLesionDisconnectomeDataset(
        lesion_root=cfg["data"].get("lesion_root"),
        disconnectome_root=cfg["data"].get("disconnectome_root"),
        in_shape=in_shape, n_synth=cfg["data"]["n_synth"], seed=cfg["seed"], stack_channels=False)
    tr_idx, va_idx = _split(len(dataset), cfg["data"].get("val_frac", 0.1), cfg["seed"])
    train_loader = make_loader(Subset(dataset, tr_idx), cfg["train"]["batch_size"], True, device, workers)
    val_loader = make_loader(Subset(dataset, va_idx), cfg["train"]["batch_size"], False, device, workers)
    zdim = m["shared_dim"] + 2 * m["private_dim"]
    log.info("DMVAE (E9b): %d pairs (%s), resolution %s, shared %d + private %dx2 = %d, %d train / %d val",
             len(dataset), "synthetic" if dataset.synthetic else "real", in_shape,
             m["shared_dim"], m["private_dim"], zdim, len(tr_idx), len(va_idx))

    log_runtime("DMVAE", device, amp)
    model = DMVAE3D(in_shape=in_shape, channels=tuple(m["channels"]), shared_dim=m["shared_dim"],
                    private_dim=m["private_dim"], backbone=m["backbone"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    os.makedirs(cfg["out_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["out_dir"], "dmvae.pt")
    best_val, history = float("inf"), []
    for epoch in range(cfg["train"]["epochs"]):
        beta = _beta(epoch, cfg["train"]["epochs"], cfg["train"]["beta_max"], cfg["train"]["warmup_frac"])
        lam = cfg["train"]["lambda_priv"]
        model.train()
        tr = [_step(model, les, dis, beta, lam, device, True, opt, scaler, amp) for les, dis in train_loader][-1]
        model.eval()
        with torch.no_grad():
            vt = [_step(model, les, dis, beta, lam, device, False, amp=amp)["total"] for les, dis in val_loader]
        tr["val"] = float(np.mean(vt)) if vt else tr["total"]
        history.append(tr)
        log.info("epoch %d/%d  beta=%.2f  rec_l=%.4f  rec_d=%.4f  kl_s=%.3f  kl_priv=%.3f  total=%.4f  val=%.4f",
                 epoch + 1, cfg["train"]["epochs"], beta, tr["rec_l"], tr["rec_d"],
                 tr["kl_s"], tr["kl_priv"], tr["total"], tr["val"])
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "cfg": cfg, "zdim": zdim,
                    "shared_dim": m["shared_dim"], "private_dim": m["private_dim"]}, ckpt_path)
        best_val = min(best_val, tr["val"])
    log.info("checkpoint at %s", ckpt_path)

    if cfg.get("export"):
        _export(model, dataset, cfg, device)
    return model, history


def _export(model, dataset, cfg, device):
    loader = make_loader(dataset, cfg["train"]["batch_size"], False, device, 0)
    model.eval()
    codes = []
    with torch.no_grad():
        for les, dis in loader:
            codes.append(model.encode_z(les.to(device), dis.to(device)).cpu().numpy())
    path = os.path.join(cfg["out_dir"], "latents_dmvae.npz")
    np.savez(path, Z=np.concatenate(codes, axis=0))
    log.info("representation exported to %s", path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prototype", choices=["prototype", "full"])
    ap.add_argument("--lambda-priv", type=float, default=None, help="weight on the private KL")
    args = ap.parse_args()
    cfg = prototype_config() if args.mode == "prototype" else full_config()
    if args.lambda_priv is not None:
        cfg["train"]["lambda_priv"] = args.lambda_priv
    run_dmvae(cfg)
