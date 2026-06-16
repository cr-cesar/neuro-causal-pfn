"""Dataset de mascaras de lesion.

Carga mascaras NIfTI binarias desde un directorio. Si el directorio no existe o
esta vacio, sintetiza mascaras tipo lesion (elipsoides aleatorios) para que el
modo prototipo se ejecute sin los datos reales preprocesados.

Cuando los archivos siguen el patron lesion{id}_{age}_{sex}.nii.gz, las
covariables clinicas (edad y sexo) se extraen del nombre y quedan disponibles,
alineadas con el orden de las mascaras, mediante clinical_matrix().
"""
import glob
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .clinical import (CLINICAL_DIM, build_clinical_vector, parse_lesion_filename,
                       synthesize_clinical)
from .transforms import binarize, pad_or_crop


class LesionMaskDataset(Dataset):
    def __init__(self, root: Optional[str] = None,
                 in_shape: Tuple[int, int, int] = (96, 112, 96),
                 n_synth: int = 64, seed: int = 0, with_clinical: bool = False):
        self.in_shape = tuple(int(s) for s in in_shape)
        self.with_clinical = with_clinical
        self.paths = []
        if root and os.path.isdir(root):
            for pat in ("*.nii", "*.nii.gz"):
                self.paths.extend(glob.glob(os.path.join(root, pat)))
            self.paths = sorted(self.paths)
        self.synthetic = len(self.paths) == 0
        self.seed = seed
        self.n = n_synth if self.synthetic else len(self.paths)
        self._clinical = None

    def __len__(self) -> int:
        return self.n

    def clinical_matrix(self) -> np.ndarray:
        """Matriz [N, CLINICAL_DIM] alineada con las mascaras. Para datos reales
        se parsea del nombre de archivo; en modo sintetico se genera al azar."""
        if self._clinical is None:
            if self.synthetic:
                self._clinical = synthesize_clinical(self.n, CLINICAL_DIM, self.seed)
            else:
                rows = []
                for p in self.paths:
                    meta = parse_lesion_filename(p)
                    rows.append(build_clinical_vector(meta["age"], meta["sex"]))
                self._clinical = np.stack(rows, axis=0)
        return self._clinical

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

    def _load_volume(self, idx: int) -> np.ndarray:
        if self.synthetic:
            vol = self._make_synth(idx)
        else:
            import nibabel as nib

            vol = np.asarray(nib.load(self.paths[idx]).get_fdata(), dtype=np.float32)
            vol = pad_or_crop(vol, self.in_shape)
        return binarize(vol)

    def __getitem__(self, idx: int):
        vol = torch.from_numpy(self._load_volume(idx)).unsqueeze(0)  # [1, D, H, W]
        if self.with_clinical:
            clinical = torch.from_numpy(self.clinical_matrix()[idx])
            return vol, clinical
        return vol
