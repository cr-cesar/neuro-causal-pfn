"""Dataset de mascaras de lesion.

Carga mascaras NIfTI binarias desde un directorio. Si el directorio no existe o
esta vacio, sintetiza mascaras tipo lesion (elipsoides aleatorios) para que el
modo prototipo se ejecute sin los datos reales preprocesados.
"""
import glob
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import binarize, pad_or_crop


class LesionMaskDataset(Dataset):
    def __init__(self, root: Optional[str] = None,
                 in_shape: Tuple[int, int, int] = (96, 112, 96),
                 n_synth: int = 64, seed: int = 0):
        self.in_shape = tuple(int(s) for s in in_shape)
        self.paths = []
        if root and os.path.isdir(root):
            for pat in ("*.nii", "*.nii.gz"):
                self.paths.extend(glob.glob(os.path.join(root, pat)))
            self.paths = sorted(self.paths)
        self.synthetic = len(self.paths) == 0
        self.seed = seed
        self.n = n_synth if self.synthetic else len(self.paths)

    def __len__(self) -> int:
        return self.n

    def _make_synth(self, idx: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 1000 + idx)
        vol = np.zeros(self.in_shape, dtype=np.float32)
        zz, yy, xx = np.indices(self.in_shape)
        for _ in range(int(rng.integers(1, 3))):
            center = [int(rng.integers(int(0.3 * s), int(0.7 * s) + 1)) for s in self.in_shape]
            radius = [int(rng.integers(max(3, int(0.05 * s)), int(0.18 * s) + 1)) for s in self.in_shape]
            ellipsoid = (((zz - center[0]) / radius[0]) ** 2
                         + ((yy - center[1]) / radius[1]) ** 2
                         + ((xx - center[2]) / radius[2]) ** 2)
            vol[ellipsoid <= 1.0] = 1.0
        return vol

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.synthetic:
            vol = self._make_synth(idx)
        else:
            import nibabel as nib

            vol = np.asarray(nib.load(self.paths[idx]).get_fdata(), dtype=np.float32)
            vol = pad_or_crop(vol, self.in_shape)
        vol = binarize(vol)
        return torch.from_numpy(vol).unsqueeze(0)  # [1, D, H, W]
