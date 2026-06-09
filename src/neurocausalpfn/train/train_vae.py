"""Entrenamiento de la Etapa 1 (los autoencoders).

Un unico punto de entrada sirve al modo prototipo y al modo completo; solo
cambian los valores de configuracion. Las funciones prototype_config y
full_config son la fuente de verdad para la ejecucion directa; los archivos YAML
en configs/ las reflejan para una composicion posterior con Hydra en el cluster.
"""
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader

from ..data.nifti_dataset import LesionMaskDataset
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from ..vae.conv3d_vae import ConvVAE3D
from ..vae.losses import vae_loss

log = get_logger()


def prototype_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/vae_prototype",
        "data": {"root": None, "resolution": [48, 56, 48], "n_synth": 64},
        "vae": {"zdim": 16, "channels": [16, 32, 64, 128, 256],
                "batch_size": 2, "epochs": 5, "lr": 1e-4,
                "beta_max": 1.0, "warmup_frac": 0.2},
        "device": "cpu",
    }


def full_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/vae_full",
        "data": {"root": "data/lesions", "resolution": [96, 112, 96], "n_synth": 0},
        "vae": {"zdim": 50, "channels": [16, 32, 64, 128, 256],
                "batch_size": 8, "epochs": 200, "lr": 1e-4,
                "beta_max": 1.0, "warmup_frac": 0.2},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def run_vae(cfg: Dict):
    set_seed(cfg["seed"])
    device = cfg.get("device", "cpu")
    in_shape = tuple(cfg["data"]["resolution"])

    dataset = LesionMaskDataset(root=cfg["data"]["root"], in_shape=in_shape,
                                n_synth=cfg["data"]["n_synth"], seed=cfg["seed"])
    loader = DataLoader(dataset, batch_size=cfg["vae"]["batch_size"], shuffle=True)
    log.info("VAE: %d volumenes (%s), resolucion %s",
             len(dataset), "sinteticos" if dataset.synthetic else "reales", in_shape)

    model = ConvVAE3D(zdim=cfg["vae"]["zdim"], in_shape=in_shape,
                      channels=tuple(cfg["vae"]["channels"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["vae"]["lr"])

    epochs = cfg["vae"]["epochs"]
    warmup_epochs = max(1, int(cfg["vae"]["warmup_frac"] * epochs))
    history = []
    for epoch in range(epochs):
        beta = min(1.0, (epoch + 1) / warmup_epochs) * cfg["vae"]["beta_max"]
        last = {}
        for x in loader:
            x = x.to(device)
            logits, mu, logvar, _ = model(x)
            loss, parts = vae_loss(logits, x, mu, logvar, beta=beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = parts
        history.append(last)
        log.info("epoca %d/%d  beta=%.2f  rec=%.4f  dice=%.4f  kl=%.3f  total=%.4f",
                 epoch + 1, epochs, last["beta"], last["rec"], last["dice"],
                 last["kl"], last["total"])

    os.makedirs(cfg["out_dir"], exist_ok=True)
    ckpt = os.path.join(cfg["out_dir"], "vae_lesion.pt")
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, ckpt)
    log.info("checkpoint guardado en %s", ckpt)
    return model, history


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prototype", choices=["prototype", "full"])
    args = ap.parse_args()
    run_vae(prototype_config() if args.mode == "prototype" else full_config())
