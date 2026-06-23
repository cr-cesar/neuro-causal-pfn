"""Arm E training: conditional hierarchical VAE (DSCM).

Trains the parent-conditioned HVAE on binary lesion masks with a BCE plus soft
Dice reconstruction and the KL against the conditional prior p(z | pa_x). The
parents pa_x are the clinical covariates; E8b additionally conditions on a
one-hot environment (regime) index for iVAE-style identifiability across regimes,
and E8c adds an ARD relevance scale to the conditional prior. E8a is the plain
conditional HVAE versus the standard VAE baseline.
"""
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..data.nifti_dataset import LesionMaskDataset
from ..dscm.model import ConditionalHVAE
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from ..vae.losses import bce_dice_loss

log = get_logger()


class _EnvWrap(Dataset):
    """Appends a per-sample one-hot environment index to (vol, clinical) for E8b."""

    def __init__(self, base, env_onehot):
        self.base = base
        self.env = env_onehot

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        vol, clin = self.base[i]
        return vol, clin, torch.from_numpy(self.env[i])


def prototype_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/dscm_prototype",
        "export": False,
        "clinical_csv": None,
        "data": {"root": None, "resolution": [24, 28, 24], "n_synth": 16, "val_frac": 0.25},
        "model": {"group_dims": [8, 8], "channels": [16, 32, 64, 128, 256], "backbone": "cnn",
                  "use_ard": False, "multi_env": False, "n_regimes": 20},
        "train": {"batch_size": 4, "epochs": 3, "lr": 1e-3, "beta_max": 1.0, "warmup_frac": 0.2},
        "device": "cpu",
    }


def full_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/dscm_full",
        "export": True,
        "clinical_csv": None,
        "data": {"root": "data/lesions", "resolution": [96, 112, 96], "n_synth": 0, "val_frac": 0.1},
        "model": {"group_dims": [25, 25], "channels": [16, 32, 64, 128, 256], "backbone": "cnn",
                  "use_ard": False, "multi_env": False, "n_regimes": 20},
        "train": {"batch_size": 8, "epochs": 100, "lr": 1e-4, "beta_max": 1.0, "warmup_frac": 0.2},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
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


def _step(model, batch, multi_env, beta, device, train, opt=None) -> Dict:
    if multi_env:
        vol, clin, env = batch
        pa = torch.cat([clin.float(), env.float()], dim=1).to(device)
    else:
        vol, clin = batch
        pa = clin.float().to(device)
    vol = vol.to(device)
    logits, _, kl = model(vol, pa)
    rec, _ = bce_dice_loss(logits, vol)
    loss = rec + beta * kl
    if train:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"rec": float(rec.detach()), "kl": float(kl.detach()), "total": float(loss.detach())}


def run_dscm(cfg: Dict):
    set_seed(cfg["seed"])
    device = cfg.get("device", "cpu")
    in_shape = tuple(cfg["data"]["resolution"])
    m = cfg["model"]
    multi_env = bool(m.get("multi_env", False))
    n_reg = int(m.get("n_regimes", 20))
    base = LesionMaskDataset(root=cfg["data"].get("root"), in_shape=in_shape,
                             n_synth=cfg["data"]["n_synth"], seed=cfg["seed"], binarize=True,
                             with_clinical=True, clinical_csv=cfg.get("clinical_csv"))
    pa_dim = base.clinical_dim() + (n_reg if multi_env else 0)
    if multi_env:
        labels = np.random.default_rng(cfg["seed"] + 7).integers(0, n_reg, len(base))
        dataset = _EnvWrap(base, np.eye(n_reg, dtype=np.float32)[labels])
    else:
        dataset = base
    tr_idx, va_idx = _split(len(dataset), cfg["data"].get("val_frac", 0.1), cfg["seed"])
    train_loader = DataLoader(Subset(dataset, tr_idx), batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(Subset(dataset, va_idx), batch_size=cfg["train"]["batch_size"], shuffle=False)
    log.info("Conditional HVAE (Arm E): %d masks (%s), resolution %s, groups %s, pa_dim %d, ard=%s, multi_env=%s, %d train / %d val",
             len(base), "synthetic" if base.synthetic else "real", in_shape, m["group_dims"],
             pa_dim, m["use_ard"], multi_env, len(tr_idx), len(va_idx))

    model = ConditionalHVAE(in_shape=in_shape, channels=tuple(m["channels"]),
                            group_dims=tuple(m["group_dims"]), pa_dim=pa_dim,
                            backbone=m["backbone"], use_ard=m["use_ard"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    os.makedirs(cfg["out_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["out_dir"], "dscm.pt")
    best_val, history = float("inf"), []
    for epoch in range(cfg["train"]["epochs"]):
        beta = _beta(epoch, cfg["train"]["epochs"], cfg["train"]["beta_max"], cfg["train"]["warmup_frac"])
        model.train()
        tr = [_step(model, b, multi_env, beta, device, True, opt) for b in train_loader][-1]
        model.eval()
        with torch.no_grad():
            vt = [_step(model, b, multi_env, beta, device, False)["total"] for b in val_loader]
        tr["val"] = float(np.mean(vt)) if vt else tr["total"]
        history.append(tr)
        log.info("epoch %d/%d  beta=%.2f  rec=%.4f  kl=%.3f  total=%.4f  val=%.4f",
                 epoch + 1, cfg["train"]["epochs"], beta, tr["rec"], tr["kl"], tr["total"], tr["val"])
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "cfg": cfg,
                    "zdim": model.zdim, "pa_dim": pa_dim, "use_ard": m["use_ard"],
                    "multi_env": multi_env}, ckpt_path)
        best_val = min(best_val, tr["val"])
    log.info("checkpoint at %s", ckpt_path)

    if cfg.get("export"):
        _export(model, base, cfg, device)
    return model, history


def _export(model, base, cfg, device):
    loader = DataLoader(base, batch_size=cfg["train"]["batch_size"], shuffle=False)
    model.eval()
    codes = []
    with torch.no_grad():
        for vol, _ in loader:
            codes.append(model.encode_z(vol.to(device)).cpu().numpy())
    path = os.path.join(cfg["out_dir"], "latents_dscm.npz")
    np.savez(path, Z=np.concatenate(codes, axis=0))
    log.info("representation exported to %s", path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prototype", choices=["prototype", "full"])
    ap.add_argument("--use-ard", action="store_true", help="add the ARD relevance scale (E8c)")
    ap.add_argument("--multi-env", action="store_true", help="condition on the environment index (E8b)")
    ap.add_argument("--n-regimes", type=int, default=None, help="number of environments for E8b")
    ap.add_argument("--clinical-csv", default=None, help="CSV with NIHSS and time_to_scan by id")
    args = ap.parse_args()
    cfg = prototype_config() if args.mode == "prototype" else full_config()
    if args.use_ard:
        cfg["model"]["use_ard"] = True
    if args.multi_env:
        cfg["model"]["multi_env"] = True
    if args.n_regimes is not None:
        cfg["model"]["n_regimes"] = args.n_regimes
    if args.clinical_csv is not None:
        cfg["clinical_csv"] = args.clinical_csv
    run_dscm(cfg)
