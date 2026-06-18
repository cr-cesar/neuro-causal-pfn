"""Fusion of the lesion and disconnectome representations for Stage 2.

Each modality has its own frozen VAE and therefore its own latent. The covariate
seen by the transformer can be one of three variants, ready to compare as in
Giles: lesion only, disconnectome only, or the fusion of both. The default fusion
is late and simple: concatenate the two latents. The lesion-disconnectome
pairing is per patient (same id), so row i of each latent matrix corresponds to
the same subject.
"""
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

FUSION_MODES = ("lesion", "disconnectome", "both")


def fuse_representation(z_lesion: Optional[np.ndarray], z_disconnectome: Optional[np.ndarray],
                        mode: str = "both") -> np.ndarray:
    """Returns the covariate matrix [N, d_x] according to the chosen variant.

    - 'lesion': the lesion latent only.
    - 'disconnectome': the disconnectome latent only.
    - 'both': concatenation of both (late fusion).
    """
    if mode not in FUSION_MODES:
        raise ValueError(f"unknown fusion mode: {mode}")
    if mode == "lesion":
        if z_lesion is None:
            raise ValueError("z_lesion is required for mode 'lesion'")
        return np.asarray(z_lesion, dtype=np.float64)
    if mode == "disconnectome":
        if z_disconnectome is None:
            raise ValueError("z_disconnectome is required for mode 'disconnectome'")
        return np.asarray(z_disconnectome, dtype=np.float64)
    if z_lesion is None or z_disconnectome is None:
        raise ValueError("both latents are required for mode 'both'")
    if len(z_lesion) != len(z_disconnectome):
        raise ValueError("the lesion and disconnectome latents are not aligned per patient")
    return np.concatenate([np.asarray(z_lesion, np.float64),
                           np.asarray(z_disconnectome, np.float64)], axis=1)


@torch.no_grad()
def compute_latents(vae, dataset, device: str = "cpu", batch_size: int = 8,
                    item_index: Optional[int] = None) -> np.ndarray:
    """Frozen latents [N, zdim] from a dataset with a given VAE.

    If the dataset returns tuples (for example the paired one, which gives lesion
    and disconnectome), item_index selects which of the volumes to encode."""
    vae.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out = []
    for batch in loader:
        x = batch if item_index is None else batch[item_index]
        out.append(vae.encode_mean(x.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)
