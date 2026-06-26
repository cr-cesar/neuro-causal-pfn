"""Stage 1 training (the autoencoders).

A single entry point serves the modalities and execution modes; only the
configuration changes.

- representation = "lesion": binary input, reconstruction with BCE plus Dice.
- representation = "disconnectome": continuous input in [0, 1], reconstruction
  with MSE (without binarizing).
- representation = "early_fusion": a single VAE on a two-channel input, lesion in
  channel 0 and disconnectome in channel 1, with BCE plus Dice on the lesion
  channel and MSE on the disconnectome channel (E9a). This is the early-fusion
  comparison point against the separate-encoder late fusion.

Optional clinical conditioning (E5a): with use_daft the encoder is modulated by
the clinical vector through a DAFT block; an optional clinical CSV adds NIHSS and
time-to-scan to the age/sex vector.

Includes a validation split to monitor reconstruction and pick the best
checkpoint, saving of the last state to resume on the cluster, and an optional
export of the frozen latents at the end.
"""
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..causal.pns import soft_pns_value
from ..data.nifti_dataset import LesionMaskDataset, PairedLesionDisconnectomeDataset
from ..utils.logging_utils import get_logger
from ..utils.runtime import (autocast_ctx, log_runtime, make_grad_scaler,
                             make_loader, optim_step, resolve_device, use_amp)
from ..utils.seed import set_seed
from ..vae.conv3d_vae import ConvVAE3D
from ..vae.losses import (ard_update_prior_var, per_dim_kl, vae_loss,
                          vae_loss_mse, vae_loss_two_channel)

log = get_logger()


def prototype_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/vae_prototype",
        "representation": "lesion",
        "resume": None,
        "export": False,
        "clinical_csv": None,
        "outcome_csv": None,
        "data": {"root": None, "lesion_root": None, "disconnectome_root": None,
                 "resolution": [48, 56, 48], "n_synth": 64, "val_frac": 0.2},
        "vae": {"zdim": 16, "channels": [16, 32, 64, 128, 256],
                "batch_size": 2, "epochs": 5, "lr": 1e-4, "backbone": "cnn",
                "beta_max": 1.0, "warmup_frac": 0.2, "use_daft": False, "use_ard": False,
                "use_pns": False, "lambda_pns": 0.1, "pns_factors": 5},
        "device": "cpu",
        "amp": True,
        "num_workers": 0,
    }


def full_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/vae_full",
        "representation": "lesion",   # lesion | disconnectome | early_fusion
        "resume": None,               # path to a checkpoint to resume from
        "export": True,               # exports the frozen latents at the end
        "clinical_csv": None,         # optional CSV with NIHSS and time_to_scan by id
        "outcome_csv": None,          # optional CSV with a binary outcome by id (Arm B / PNS)
        "data": {"root": "data/lesions", "lesion_root": "data/lesions",
                 "disconnectome_root": "data/disconnectomes",
                 "resolution": [96, 112, 96], "n_synth": 0, "val_frac": 0.1},
        "vae": {"zdim": 50, "channels": [16, 32, 64, 128, 256],
                "batch_size": 8, "epochs": 200, "lr": 1e-4, "backbone": "cnn",
                "beta_max": 1.0, "warmup_frac": 0.2, "use_daft": False, "use_ard": False,
                "use_pns": False, "lambda_pns": 0.1, "pns_factors": 5},
        "device": "auto",             # 'auto' resolves to cuda (V100) when available
        "amp": True,                  # mixed precision on the V100 Tensor Cores
        "num_workers": 4,             # data-loading workers (GPU runs)
    }


def _split_indices(n: int, val_frac: float, seed: int):
    idx = np.random.default_rng(seed).permutation(n)
    n_val = int(val_frac * n)
    if n_val < 1:
        return idx, idx   # small cohort: validate on the same set
    return idx[n_val:], idx[:n_val]


def _build_dataset(cfg: Dict, representation: str, in_shape, use_daft: bool):
    """Returns (dataset, in_channels, loss_fn) for the chosen representation."""
    n_synth = cfg["data"]["n_synth"]
    seed = cfg["seed"]
    clinical_csv = cfg.get("clinical_csv")
    use_pns = bool(cfg["vae"].get("use_pns", False))
    outcome_csv = cfg.get("outcome_csv")
    if representation == "early_fusion":
        ds = PairedLesionDisconnectomeDataset(
            lesion_root=cfg["data"].get("lesion_root"),
            disconnectome_root=cfg["data"].get("disconnectome_root"),
            in_shape=in_shape, n_synth=n_synth, seed=seed,
            with_clinical=use_daft, stack_channels=True, clinical_csv=clinical_csv,
            with_target=use_pns, outcome_csv=outcome_csv)
        return ds, 2, vae_loss_two_channel
    binarize = representation == "lesion"
    ds = LesionMaskDataset(root=cfg["data"]["root"], in_shape=in_shape,
                           n_synth=n_synth, seed=seed, binarize=binarize,
                           with_clinical=use_daft, clinical_csv=clinical_csv,
                           with_target=use_pns, outcome_csv=outcome_csv)
    return ds, 1, (vae_loss if binarize else vae_loss_mse)


def _epoch(model, loader, loss_fn, beta, device, opt=None, use_daft=False,
           prior_var=None, ard_accum=None, use_pns=False, lambda_pns=0.0, pns_factors=5,
           scaler=None, amp=False):
    train = opt is not None
    model.train(train)
    last = {}
    torch.set_grad_enabled(train)
    for batch in loader:
        items = list(batch) if isinstance(batch, (list, tuple)) else [batch]
        x = items[0].to(device)
        clin = items[1].to(device) if use_daft else None
        target = items[-1].to(device) if use_pns else None
        with autocast_ctx(device, amp):
            logits, mu, logvar, _ = model(x, clin) if use_daft else model(x)
            loss, parts = loss_fn(logits, x, mu, logvar, beta=beta, prior_var=prior_var)
            if use_pns and target is not None:
                # Arm B: maximise the PNS surrogate via a -lambda * value term
                pns_val = soft_pns_value(mu, target, k=pns_factors)
                loss = loss - lambda_pns * pns_val
                parts["pns"] = float(pns_val.detach())
        if train:
            optim_step(loss, opt, scaler)
        if ard_accum is not None and train:
            n = mu.shape[0]
            ard_accum["sumsq"] += (mu.detach().float().pow(2) + logvar.detach().float().exp()).sum(0)
            ard_accum["kl_sum"] += per_dim_kl(mu.detach().float(), logvar.detach().float(), prior_var) * n
            ard_accum["n"] += n
        last = parts
    torch.set_grad_enabled(True)
    return last


def _export_latents(model, dataset, cfg, representation, use_daft, device):
    out_dir = cfg["out_dir"]
    prev_target = getattr(dataset, "with_target", False)
    dataset.with_target = False   # the export does not need the outcome
    try:
        loader = make_loader(dataset, cfg["vae"]["batch_size"], False, device, 0)
        if use_daft:
            # conditioned export: pass the clinical vector to the encoder
            model.eval()
            codes = []
            with torch.no_grad():
                for x, clin in loader:
                    codes.append(model.encode_mean(x.to(device), clin.to(device)).cpu().numpy())
            path = os.path.join(out_dir, f"latents_{representation}.npz")
            np.savez(path, Z=np.concatenate(codes, axis=0), clinical=dataset.clinical_matrix())
        else:
            from ..vae.export_encoder import export_representation

            path = export_representation(model, loader, out_dir,
                                         clinical=dataset.clinical_matrix(), device=device)
    finally:
        dataset.with_target = prev_target
    log.info("representation exported to %s", path)


def run_vae(cfg: Dict):
    set_seed(cfg["seed"])
    device = resolve_device(cfg)
    amp = use_amp(cfg, device)
    scaler = make_grad_scaler(amp)
    in_shape = tuple(cfg["data"]["resolution"])
    representation = cfg.get("representation", "lesion")
    use_daft = bool(cfg["vae"].get("use_daft", False))
    use_ard = bool(cfg["vae"].get("use_ard", False))
    backbone = cfg["vae"].get("backbone", "cnn")
    use_pns = bool(cfg["vae"].get("use_pns", False))
    lambda_pns = float(cfg["vae"].get("lambda_pns", 0.1))
    pns_factors = int(cfg["vae"].get("pns_factors", 5))
    zdim = cfg["vae"]["zdim"]
    workers = int(cfg.get("num_workers", 0))

    dataset, in_channels, loss_fn = _build_dataset(cfg, representation, in_shape, use_daft)
    n_clinical = dataset.clinical_dim() if use_daft else 0

    train_idx, val_idx = _split_indices(len(dataset), cfg["data"].get("val_frac", 0.1), cfg["seed"])
    train_loader = make_loader(Subset(dataset, train_idx), cfg["vae"]["batch_size"], True, device, workers)
    val_loader = make_loader(Subset(dataset, val_idx), cfg["vae"]["batch_size"], False, device, workers)
    log.info("VAE (%s): %d volumes (%s), resolution %s, %d channel(s), backbone=%s, DAFT=%s, ARD=%s, %d train / %d val",
             representation, len(dataset), "synthetic" if dataset.synthetic else "real",
             in_shape, in_channels, backbone, use_daft, use_ard, len(train_idx), len(val_idx))
    log_runtime("VAE", device, amp)

    model = ConvVAE3D(in_channels=in_channels, zdim=zdim, in_shape=in_shape,
                      channels=tuple(cfg["vae"]["channels"]), backbone=backbone,
                      use_daft=use_daft, n_clinical=n_clinical, use_ard=use_ard).to(device)
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
        prior_var = model.ard_prior_var if use_ard else None
        ard_accum = ({"sumsq": torch.zeros(zdim, device=device),
                      "kl_sum": torch.zeros(zdim, device=device), "n": 0}
                     if use_ard else None)
        tr = _epoch(model, train_loader, loss_fn, beta, device, opt=opt, use_daft=use_daft,
                    prior_var=prior_var, ard_accum=ard_accum,
                    use_pns=use_pns, lambda_pns=lambda_pns, pns_factors=pns_factors,
                    scaler=scaler, amp=amp)
        va = _epoch(model, val_loader, loss_fn, beta, device, opt=None, use_daft=use_daft,
                    prior_var=prior_var, ard_accum=None,
                    use_pns=use_pns, lambda_pns=lambda_pns, pns_factors=pns_factors,
                    scaler=scaler, amp=amp)
        tr["val_total"] = va["total"]
        if use_ard:
            # closed-form prior update from this epoch's encoded second moments
            model.ard_prior_var.copy_(ard_update_prior_var(ard_accum["sumsq"], ard_accum["n"]))
            active = int((ard_accum["kl_sum"] / max(ard_accum["n"], 1) > 0.01).sum())
            tr["active_dims"] = active
        history.append(tr)
        msg = "epoch %d/%d  beta=%.2f  rec=%.4f  kl=%.3f  train=%.4f  val=%.4f"
        args_ = [epoch + 1, epochs, tr["beta"], tr["rec"], tr["kl"], tr["total"], va["total"]]
        if use_ard:
            msg += "  active_dims=%d/%d"
            args_ += [tr["active_dims"], zdim]
        if use_pns:
            msg += "  pns=%.4f"
            args_ += [tr.get("pns", float("nan"))]
        log.info(msg, *args_)

        ckpt = {"epoch": epoch, "state_dict": model.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg, "best_val": best_val, "representation": representation,
                "in_channels": in_channels, "n_clinical": n_clinical,
                "use_daft": use_daft, "use_ard": use_ard, "backbone": backbone,
                "use_pns": use_pns, "lambda_pns": lambda_pns, "pns_factors": pns_factors}
        torch.save(ckpt, last_path)
        if va["total"] < best_val:
            best_val = va["total"]
            ckpt["best_val"] = best_val
            torch.save(ckpt, best_path)

    log.info("best checkpoint at %s (val=%.4f)", best_path, best_val)

    if cfg.get("export"):
        _export_latents(model, dataset, cfg, representation, use_daft, device)

    return model, history


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prototype", choices=["prototype", "full"])
    ap.add_argument("--representation", default=None,
                    choices=["lesion", "disconnectome", "early_fusion"])
    ap.add_argument("--use-daft", action="store_true", help="enable DAFT clinical conditioning (E5a)")
    ap.add_argument("--use-ard", action="store_true", help="enable ARD data-driven dimensionality (E4)")
    ap.add_argument("--backbone", default=None,
                    choices=["cnn", "wide", "resnet", "resnet18", "resnet50"],
                    help="encoder backbone (E7)")
    ap.add_argument("--use-pns", action="store_true", help="enable the PNS auxiliary loss (Arm B)")
    ap.add_argument("--lambda-pns", type=float, default=None, help="weight of the PNS term")
    ap.add_argument("--pns-factors", type=int, default=None, help="number of common-cause factors")
    ap.add_argument("--outcome-csv", default=None, help="CSV with a binary outcome by id (Arm B)")
    ap.add_argument("--clinical-csv", default=None, help="CSV with NIHSS and time_to_scan by id")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = prototype_config() if args.mode == "prototype" else full_config()
    if args.representation is not None:
        cfg["representation"] = args.representation
        if args.representation == "disconnectome" and cfg["data"]["root"] == "data/lesions":
            cfg["data"]["root"] = "data/disconnectomes"
        cfg["out_dir"] = f"outputs/vae_{args.mode}_{args.representation}"
    if args.use_daft:
        cfg["vae"]["use_daft"] = True
    if args.use_ard:
        cfg["vae"]["use_ard"] = True
    if args.backbone is not None:
        cfg["vae"]["backbone"] = args.backbone
    if args.use_pns:
        cfg["vae"]["use_pns"] = True
    if args.lambda_pns is not None:
        cfg["vae"]["lambda_pns"] = args.lambda_pns
    if args.pns_factors is not None:
        cfg["vae"]["pns_factors"] = args.pns_factors
    if args.outcome_csv is not None:
        cfg["outcome_csv"] = args.outcome_csv
    if args.clinical_csv is not None:
        cfg["clinical_csv"] = args.clinical_csv
    if args.resume is not None:
        cfg["resume"] = args.resume
    run_vae(cfg)
