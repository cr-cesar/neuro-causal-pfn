"""Arm C training: supervised-contrastive hybrid.

Objective L = L_SupCon(Z, Y) + lambda * L_IntraModal + mu * L_Recon.

Each sample yields two augmented views per modality (binary-coherent
augmentations). The fused projection feeds the supervised-contrastive term with
the outcome as the label (E10a); the per-modality projections feed the
intra-modal NT-Xent term; and, when enabled, the reconstruction term keeps the
latent grounded in anatomy and prevents collapse to easy features (E10c).
"""
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..contrastive.losses import nt_xent_loss, supcon_loss
from ..contrastive.model import ContrastiveFusionEncoder
from ..data.augmentations import augment_batch
from ..data.nifti_dataset import PairedLesionDisconnectomeDataset
from ..utils.logging_utils import get_logger
from ..utils.runtime import (autocast_ctx, log_runtime, make_grad_scaler,
                             make_loader, optim_step, resolve_device, use_amp)
from ..utils.seed import set_seed
from ..vae.losses import bce_dice_loss, mse_recon_loss

log = get_logger()


def prototype_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/contrastive_prototype",
        "export": False,
        "outcome_csv": None,
        "data": {"lesion_root": None, "disconnectome_root": None,
                 "resolution": [24, 28, 24], "n_synth": 16, "val_frac": 0.25},
        "model": {"zdim": 16, "channels": [16, 32, 64, 128, 256], "backbone": "cnn",
                  "d_model": 64, "proj_dim": 64, "n_heads": 4, "recon": True},
        "train": {"batch_size": 4, "epochs": 3, "lr": 1e-3, "tau": 0.1,
                  "lambda_intra": 0.5, "mu_recon": 1.0},
        "device": "cpu",
    }


def full_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/contrastive_full",
        "export": True,
        "outcome_csv": None,            # binary outcome by id; SupCon supervision
        "data": {"lesion_root": "data/lesions", "disconnectome_root": "data/disconnectomes",
                 "resolution": [96, 112, 96], "n_synth": 0, "val_frac": 0.1},
        "model": {"zdim": 50, "channels": [16, 32, 64, 128, 256], "backbone": "cnn",
                  "d_model": 128, "proj_dim": 128, "n_heads": 4, "recon": True},
        "train": {"batch_size": 16, "epochs": 100, "lr": 1e-4, "tau": 0.1,
                  "lambda_intra": 0.5, "mu_recon": 1.0},
        "device": "auto",
        "amp": True,
        "num_workers": 4,
    }


def _split_indices(n: int, val_frac: float, seed: int):
    idx = np.random.default_rng(seed).permutation(n)
    n_val = int(val_frac * n)
    if n_val < 1:
        return idx, idx
    return idx[n_val:], idx[:n_val]


def _step(model, les, dis, y, cfg, device, train, opt=None, scaler=None, amp=False) -> Dict:
    tr = cfg["train"]
    les, dis = les.to(device), dis.to(device)
    y = y.to(device).long()
    les1, les2 = augment_batch(les, binary=True), augment_batch(les, binary=True)
    dis1, dis2 = augment_batch(dis, binary=False), augment_batch(dis, binary=False)
    with autocast_ctx(device, amp):
        out1, out2 = model(les1, dis1), model(les2, dis2)
        B = les.shape[0]

        feats = torch.cat([out1["p"], out2["p"]], dim=0)
        labels = torch.cat([y, y], dim=0)
        loss_supcon = supcon_loss(feats, labels, tau=tr["tau"])
        les_feats = torch.cat([out1["p_lesion"], out2["p_lesion"]], dim=0)
        dis_feats = torch.cat([out1["p_disco"], out2["p_disco"]], dim=0)
        loss_intra = 0.5 * (nt_xent_loss(les_feats, B, tau=tr["tau"])
                            + nt_xent_loss(dis_feats, B, tau=tr["tau"]))
        loss = loss_supcon + tr["lambda_intra"] * loss_intra
        parts = {"supcon": float(loss_supcon.detach()), "intra": float(loss_intra.detach())}
        if model.recon:
            rec_l, _ = bce_dice_loss(out1["recon_lesion"], les1)
            rec_d = mse_recon_loss(out1["recon_disco"], dis1)
            loss_recon = rec_l + rec_d
            loss = loss + tr["mu_recon"] * loss_recon
            parts["recon"] = float(loss_recon.detach())
        parts["total"] = float(loss.detach())
    if train:
        optim_step(loss, opt, scaler)
    return parts


def run_contrastive(cfg: Dict):
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
        in_shape=in_shape, n_synth=cfg["data"]["n_synth"], seed=cfg["seed"],
        with_target=True, outcome_csv=cfg.get("outcome_csv"))
    tr_idx, va_idx = _split_indices(len(dataset), cfg["data"].get("val_frac", 0.1), cfg["seed"])
    train_loader = make_loader(Subset(dataset, tr_idx), cfg["train"]["batch_size"], True, device, workers)
    val_loader = make_loader(Subset(dataset, va_idx), cfg["train"]["batch_size"], False, device, workers)
    log.info("Contrastive (Arm C): %d pairs (%s), resolution %s, backbone=%s, recon=%s, %d train / %d val",
             len(dataset), "synthetic" if dataset.synthetic else "real", in_shape,
             m["backbone"], m["recon"], len(tr_idx), len(va_idx))

    log_runtime("Contrastive", device, amp)
    model = ContrastiveFusionEncoder(in_shape=in_shape, channels=tuple(m["channels"]),
                                     zdim=m["zdim"], backbone=m["backbone"], d_model=m["d_model"],
                                     proj_dim=m["proj_dim"], n_heads=m["n_heads"], recon=m["recon"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    os.makedirs(cfg["out_dir"], exist_ok=True)
    best_path = os.path.join(cfg["out_dir"], "contrastive.pt")
    best_val, history = float("inf"), []
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        last = {}
        for les, dis, y in train_loader:
            last = _step(model, les, dis, y, cfg, device, train=True, opt=opt, scaler=scaler, amp=amp)
        model.eval()
        with torch.no_grad():
            vt = [_step(model, les, dis, y, cfg, device, train=False, amp=amp)["total"]
                  for les, dis, y in val_loader]
        last["val_total"] = float(np.mean(vt)) if vt else last.get("total", float("nan"))
        history.append(last)
        log.info("epoch %d/%d  supcon=%.4f  intra=%.4f  total=%.4f  val=%.4f",
                 epoch + 1, cfg["train"]["epochs"], last.get("supcon", float("nan")),
                 last.get("intra", float("nan")), last["total"], last["val_total"])
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "cfg": cfg,
                    "backbone": m["backbone"], "zdim": m["zdim"]}, best_path)
        best_val = min(best_val, last["val_total"])
    log.info("checkpoint at %s", best_path)

    if cfg.get("export"):
        _export(model, dataset, cfg, device)
    return model, history


def _export(model, dataset, cfg, device):
    prev = dataset.with_target
    dataset.with_target = False
    try:
        loader = make_loader(dataset, cfg["train"]["batch_size"], False, device, 0)
        model.eval()
        codes = []
        with torch.no_grad():
            for les, dis in loader:
                codes.append(model.encode_z(les.to(device), dis.to(device)).cpu().numpy())
    finally:
        dataset.with_target = prev
    path = os.path.join(cfg["out_dir"], "latents_contrastive.npz")
    np.savez(path, Z=np.concatenate(codes, axis=0))
    log.info("representation exported to %s", path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prototype", choices=["prototype", "full"])
    ap.add_argument("--backbone", default=None,
                    choices=["cnn", "wide", "resnet", "resnet18", "resnet50"])
    ap.add_argument("--no-recon", action="store_true", help="drop the reconstruction term (E10a)")
    ap.add_argument("--outcome-csv", default=None, help="binary outcome by id (SupCon labels)")
    args = ap.parse_args()
    cfg = prototype_config() if args.mode == "prototype" else full_config()
    if args.backbone is not None:
        cfg["model"]["backbone"] = args.backbone
    if args.no_recon:
        cfg["model"]["recon"] = False
    if args.outcome_csv is not None:
        cfg["outcome_csv"] = args.outcome_csv
    run_contrastive(cfg)
