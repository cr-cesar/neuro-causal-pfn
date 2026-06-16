"""Fusion de las representaciones de lesion y disconnectoma para la Etapa 2.

Cada modalidad tiene su propio VAE congelado y, por tanto, su propio latente. La
covariable que ve el transformer puede ser una de tres variantes, listas para
comparar como en Giles: solo lesion, solo disconnectoma, o la fusion de ambas.
La fusion por defecto es tardia y simple: concatenar los dos latentes. El
emparejamiento lesion-disconnectoma es por paciente (mismo id), as que la fila i
de cada matriz de latentes corresponde al mismo sujeto.
"""
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

FUSION_MODES = ("lesion", "disconnectome", "both")


def fuse_representation(z_lesion: Optional[np.ndarray], z_disconnectome: Optional[np.ndarray],
                        mode: str = "both") -> np.ndarray:
    """Devuelve la matriz de covariables [N, d_x] segun la variante elegida.

    - 'lesion': solo el latente de la lesion.
    - 'disconnectome': solo el latente del disconnectoma.
    - 'both': concatenacion de ambos (fusion tardia).
    """
    if mode not in FUSION_MODES:
        raise ValueError(f"modo de fusion desconocido: {mode}")
    if mode == "lesion":
        if z_lesion is None:
            raise ValueError("se requiere z_lesion para el modo 'lesion'")
        return np.asarray(z_lesion, dtype=np.float64)
    if mode == "disconnectome":
        if z_disconnectome is None:
            raise ValueError("se requiere z_disconnectome para el modo 'disconnectome'")
        return np.asarray(z_disconnectome, dtype=np.float64)
    if z_lesion is None or z_disconnectome is None:
        raise ValueError("se requieren ambos latentes para el modo 'both'")
    if len(z_lesion) != len(z_disconnectome):
        raise ValueError("los latentes de lesion y disconnectoma no estan alineados por paciente")
    return np.concatenate([np.asarray(z_lesion, np.float64),
                           np.asarray(z_disconnectome, np.float64)], axis=1)


@torch.no_grad()
def compute_latents(vae, dataset, device: str = "cpu", batch_size: int = 8,
                    item_index: Optional[int] = None) -> np.ndarray:
    """Latentes congelados [N, zdim] de un dataset con un VAE dado.

    Si el dataset devuelve tuplas (por ejemplo el emparejado, que da lesion y
    disconnectoma), item_index selecciona cual de los volumenes codificar."""
    vae.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out = []
    for batch in loader:
        x = batch if item_index is None else batch[item_index]
        out.append(vae.encode_mean(x.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)
