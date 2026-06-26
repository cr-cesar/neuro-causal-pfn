"""Arm D training: vascular-constrained Hi-End MAE pretraining.

Self-supervised masked-autoencoding on binary lesion masks: 75% of the patches
are hidden in contiguous blocks and the model reconstructs them under a
lesion-weighted BCE. After pretraining, the encoder is the frozen representation
for the downstream CausalPFN (E10b); encode_z exports the latent.
"""
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..data.nifti_dataset import LesionMaskDataset
from ..mae.losses import masked_lesion_bce
from ..mae.model import HiEndMAE3D
from ..utils.logging_utils import get_logger
from ..utils.runtime import (autocast_ctx, log_runtime, make_grad_scaler,
                             make_loader, optim_step, resolve_device, use_amp)
from ..utils.seed import set_seed

log = get_logger()


def prototype_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/mae_prototype",
        "export": False,
        "data": {"root": None, "resolution": [24, 28, 24], "n_synth": 16, "val_frac": 0.25},
        "model": {"patch": 4, "embed_dim": 64, "depth": 4, "heads": 4, "decoder_dim": 32,
                  "decoder_depth": 2, "decoder_heads": 4, "zdim": 16, "mask_ratio": 0.75,
                  "block": [2, 2, 2]},
        "train": {"batch_size": 4, "epochs": 3, "lr": 1e-3, "lesion_weight": 10.0},
        "device": "cpu",
    }


def full_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/mae_full",
        "export": True,
        "data": {"root": "data/lesions", "resolution": [96, 112, 96], "n_synth": 0, "val_frac": 0.1},
        "model": {"patch": 16, "embed_dim": 384, "depth": 12, "heads": 6, "decoder_dim": 192,
                  "decoder_depth": 8, "decoder_heads": 6, "zdim": 50, "mask_ratio": 0.75,
                  "block": [2, 2, 2]},
        "train": {"batch_size": 8, "epochs": 100, "lr": 1.5e-4, "lesion_weight": 10.0},
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


def _step(model, vol, lesion_weight, device, train, opt=None, scaler=None, amp=False) -> float:
    vol = vol.to(device)
    with autocast_ctx(device, amp):
        pred, mask = model(vol)
        target = model.patchify(vol)
        loss = masked_lesion_bce(pred, target, mask, lesion_weight)
    if train:
        optim_step(loss, opt, scaler)
    return float(loss.detach())


def run_mae(cfg: Dict):
    set_seed(cfg["seed"])
    device = resolve_device(cfg)
    amp = use_amp(cfg, device)
    scaler = make_grad_scaler(amp)
    workers = int(cfg.get("num_workers", 0))
    in_shape = tuple(cfg["data"]["resolution"])
    m = cfg["model"]
    dataset = LesionMaskDataset(root=cfg["data"].get("root"), in_shape=in_shape,
                                n_synth=cfg["data"]["n_synth"], seed=cfg["seed"], binarize=True)
    tr_idx, va_idx = _split(len(dataset), cfg["data"].get("val_frac", 0.1), cfg["seed"])
    train_loader = make_loader(Subset(dataset, tr_idx), cfg["train"]["batch_size"], True, device, workers)
    val_loader = make_loader(Subset(dataset, va_idx), cfg["train"]["batch_size"], False, device, workers)
    log.info("Hi-End MAE (Arm D): %d masks (%s), resolution %s, patch %d, grid %s, mask_ratio %.2f, %d train / %d val",
             len(dataset), "synthetic" if dataset.synthetic else "real", in_shape, m["patch"],
             _grid_str(in_shape, m["patch"]), m["mask_ratio"], len(tr_idx), len(va_idx))

    log_runtime("Hi-End MAE", device, amp)
    model = HiEndMAE3D(in_shape=in_shape, patch=m["patch"], embed_dim=m["embed_dim"],
                       depth=m["depth"], heads=m["heads"], decoder_dim=m["decoder_dim"],
                       decoder_depth=m["decoder_depth"], decoder_heads=m["decoder_heads"],
                       zdim=m["zdim"], mask_ratio=m["mask_ratio"], block=tuple(m["block"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    os.makedirs(cfg["out_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["out_dir"], "mae.pt")
    best_val, history = float("inf"), []
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        last = np.mean([_step(model, vol, cfg["train"]["lesion_weight"], device, True, opt, scaler, amp)
                        for vol in train_loader])
        model.eval()
        with torch.no_grad():
            vt = [_step(model, vol, cfg["train"]["lesion_weight"], device, False, amp=amp) for vol in val_loader]
        val = float(np.mean(vt)) if vt else float(last)
        rec = {"train": float(last), "val": val}
        history.append(rec)
        log.info("epoch %d/%d  train=%.4f  val=%.4f", epoch + 1, cfg["train"]["epochs"], rec["train"], val)
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "cfg": cfg, "zdim": m["zdim"]}, ckpt_path)
        best_val = min(best_val, val)
    log.info("checkpoint at %s", ckpt_path)

    if cfg.get("export"):
        _export(model, dataset, cfg, device)
    return model, history


def _grid_str(in_shape, patch):
    return f"{in_shape[0] // patch}x{in_shape[1] // patch}x{in_shape[2] // patch}"


def _export(model, dataset, cfg, device):
    loader = make_loader(dataset, cfg["train"]["batch_size"], False, device, 0)
    model.eval()
    codes = []
    with torch.no_grad():
        for vol in loader:
            codes.append(model.encode_z(vol.to(device)).cpu().numpy())
    path = os.path.join(cfg["out_dir"], "latents_mae.npz")
    np.savez(path, Z=np.concatenate(codes, axis=0))
    log.info("representation exported to %s", path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prototype", choices=["prototype", "full"])
    ap.add_argument("--mask-ratio", type=float, default=None, help="fraction of patches masked")
    ap.add_argument("--lesion-weight", type=float, default=None, help="up-weight for lesion-bearing patches")
    args = ap.parse_args()
    cfg = prototype_config() if args.mode == "prototype" else full_config()
    if args.mask_ratio is not None:
        cfg["model"]["mask_ratio"] = args.mask_ratio
    if args.lesion_weight is not None:
        cfg["train"]["lesion_weight"] = args.lesion_weight
    run_mae(cfg)
