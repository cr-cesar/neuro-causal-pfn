"""Stage 1 training (the autoencoders).

A single entry point serves both modalities and both execution modes; only the
configuration values change.

- representation = "lesion": binary input, reconstruction with BCE plus Dice.
- representation = "disconnectome": continuous input in [0, 1], reconstruction
  with MSE (without binarizing).

Includes a validation split to monitor reconstruction and pick the best
checkpoint, saving of the last state to resume on the cluster, and an optional
export of the frozen latents at the end.
"""
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..data.nifti_dataset import LesionMaskDataset
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from ..vae.conv3d_vae import ConvVAE3D
from ..vae.losses import vae_loss, vae_loss_mse

log = get_logger()


def prototype_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/vae_prototype",
        "representation": "lesion",
        "resume": None,
        "export": False,
        "data": {"root": None, "resolution": [48, 56, 48], "n_synth": 64, "val_frac": 0.2},
        "vae": {"zdim": 16, "channels": [16, 32, 64, 128, 256],
                "batch_size": 2, "epochs": 5, "lr": 1e-4,
                "beta_max": 1.0, "warmup_frac": 0.2},
        "device": "cpu",
    }


def full_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/vae_full",
        "representation": "lesion",   # change to "disconnectome" for the other modality
        "resume": None,               # path to a checkpoint to resume from
        "export": True,               # exports the frozen latents at the end
        "data": {"root": "data/lesions", "resolution": [96, 112, 96],
                 "n_synth": 0, "val_frac": 0.1},
        "vae": {"zdim": 50, "channels": [16, 32, 64, 128, 256],
                "batch_size": 8, "epochs": 200, "lr": 1e-4,
                "beta_max": 1.0, "warmup_frac": 0.2},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def _split_indices(n: int, val_frac: float, seed: int):
    idx = np.random.default_rng(seed).permutation(n)
    n_val = int(val_frac * n)
    if n_val < 1:
        return idx, idx   # small cohort: validate on the same set
    return idx[n_val:], idx[:n_val]


def _epoch(model, loader, loss_fn, beta, device, opt=None):
    train = opt is not None
    model.train(train)
    last = {}
    torch.set_grad_enabled(train)
    for x in loader:
        x = x.to(device)
        logits, mu, logvar, _ = model(x)
        loss, parts = loss_fn(logits, x, mu, logvar, beta=beta)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        last = parts
    torch.set_grad_enabled(True)
    return last


def run_vae(cfg: Dict):
    set_seed(cfg["seed"])
    device = cfg.get("device", "cpu")
    in_shape = tuple(cfg["data"]["resolution"])
    representation = cfg.get("representation", "lesion")
    binarize = representation == "lesion"
    loss_fn = vae_loss if binarize else vae_loss_mse

    dataset = LesionMaskDataset(root=cfg["data"]["root"], in_shape=in_shape,
                                n_synth=cfg["data"]["n_synth"], seed=cfg["seed"],
                                binarize=binarize)
    train_idx, val_idx = _split_indices(len(dataset), cfg["data"].get("val_frac", 0.1), cfg["seed"])
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=cfg["vae"]["batch_size"], shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=cfg["vae"]["batch_size"], shuffle=False)
    log.info("VAE (%s): %d volumes (%s), resolution %s, %d training / %d validation",
             representation, len(dataset), "synthetic" if dataset.synthetic else "real",
             in_shape, len(train_idx), len(val_idx))

    model = ConvVAE3D(zdim=cfg["vae"]["zdim"], in_shape=in_shape,
                      channels=tuple(cfg["vae"]["channels"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["vae"]["lr"])

    start_epoch, best_val = 0, float("inf")
    if cfg.get("resume") and os.path.exists(cfg["resume"]):
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val = ckpt.get("best_val", float("inf"))
        log.info("resuming from %s at epoch %d", cfg["resume"], start_epoch)

    epochs = cfg["vae"]["epochs"]
    warmup_epochs = max(1, int(cfg["vae"]["warmup_frac"] * epochs))
    os.makedirs(cfg["out_dir"], exist_ok=True)
    best_path = os.path.join(cfg["out_dir"], f"vae_{representation}.pt")
    last_path = os.path.join(cfg["out_dir"], f"vae_{representation}_last.pt")
    history = []
    for epoch in range(start_epoch, epochs):
        beta = min(1.0, (epoch + 1) / warmup_epochs) * cfg["vae"]["beta_max"]
        tr = _epoch(model, train_loader, loss_fn, beta, device, opt=opt)
        va = _epoch(model, val_loader, loss_fn, beta, device, opt=None)
        tr["val_total"] = va["total"]
        history.append(tr)
        log.info("epoch %d/%d  beta=%.2f  rec=%.4f  kl=%.3f  train=%.4f  val=%.4f",
                 epoch + 1, epochs, tr["beta"], tr["rec"], tr["kl"], tr["total"], va["total"])

        ckpt = {"epoch": epoch, "state_dict": model.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg, "best_val": best_val, "representation": representation}
        torch.save(ckpt, last_path)
        if va["total"] < best_val:
            best_val = va["total"]
            ckpt["best_val"] = best_val
            torch.save(ckpt, best_path)

    log.info("best checkpoint at %s (val=%.4f)", best_path, best_val)

    if cfg.get("export"):
        from ..vae.export_encoder import export_representation

        full_loader = DataLoader(dataset, batch_size=cfg["vae"]["batch_size"], shuffle=False)
        rep_path = export_representation(model, full_loader, cfg["out_dir"],
                                         clinical=dataset.clinical_matrix(), device=device)
        log.info("representation exported to %s", rep_path)

    return model, history


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prototype", choices=["prototype", "full"])
    ap.add_argument("--representation", default=None, choices=["lesion", "disconnectome"])
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = prototype_config() if args.mode == "prototype" else full_config()
    if args.representation is not None:
        cfg["representation"] = args.representation
        if args.representation == "disconnectome" and cfg["data"]["root"] == "data/lesions":
            cfg["data"]["root"] = "data/disconnectomes"
        cfg["out_dir"] = f"outputs/vae_{args.mode}_{args.representation}"
    if args.resume is not None:
        cfg["resume"] = args.resume
    run_vae(cfg)
