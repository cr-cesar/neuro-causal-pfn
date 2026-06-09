"""Congelado y exportacion de la representacion de la Etapa 1.

Calcula el codigo de la cohorte una sola vez con el encoder congelado y lo
escribe a disco junto con las covariables clinicas. El nombre del archivo
incluye un hash de los pesos del encoder, de modo que cualquier resultado de la
Etapa 2 es trazable hasta una representacion exacta. Soporta una o dos
modalidades (lesion y disconectoma); con ambas, los codigos se concatenan.
"""
import hashlib
import os
from typing import Optional

import numpy as np
import torch


def encoder_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha1()
    for _, param in sorted(model.state_dict().items()):
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


@torch.no_grad()
def export_representation(vae_lesion, loader, out_dir: str,
                          vae_disco=None, clinical: Optional[np.ndarray] = None,
                          device: str = "cpu") -> str:
    os.makedirs(out_dir, exist_ok=True)
    vae_lesion.eval().to(device)
    if vae_disco is not None:
        vae_disco.eval().to(device)

    codes = []
    for batch in loader:
        x = batch.to(device)
        z = vae_lesion.encode_mean(x)
        if vae_disco is not None:
            z = torch.cat([z, vae_disco.encode_mean(x)], dim=-1)
        codes.append(z.cpu().numpy())
    Z = np.concatenate(codes, axis=0).astype(np.float32)

    tag = encoder_hash(vae_lesion)
    path = os.path.join(out_dir, f"representation_{tag}.npz")
    payload = {"Z": Z}
    if clinical is not None:
        payload["clinical"] = np.asarray(clinical, dtype=np.float32)[: len(Z)]
    np.savez(path, **payload)
    return path
