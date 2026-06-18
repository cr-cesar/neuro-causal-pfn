"""3D volume datasets.

LesionMaskDataset loads NIfTI masks from a directory. With binarize=True it
thresholds to a binary mask (lesion); with binarize=False it keeps the
continuous values, which is what the disconnectome needs (a probability map in
[0, 1]). If the directory does not exist, it synthesizes volumes so that the
prototype runs: a smooth field in [0, 1] that, thresholded, gives a binary
lesion and, without thresholding, serves as a synthetic disconnectome.

PairedLesionDisconnectomeDataset pairs lesion and disconnectome by the id in the
filename (lesion{id}_{age}_{sex}.nii.gz, shared by both modalities), so that the
fusion of the two representations is per patient.
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


def _soft_field(in_shape: Tuple[int, int, int], seed: int) -> np.ndarray:
    """Smooth lesion-like field in [0, 1]. Equals 1 in the core, 0.5 at the
    ellipsoid border and decays outside, so that thresholding at 0.5 recovers the
    interior of the ellipsoid."""
    rng = np.random.default_rng(seed)
    vol = np.zeros(in_shape, dtype=np.float32)
    zz, yy, xx = np.indices(in_shape)
    for _ in range(int(rng.integers(1, 3))):
        c = [int(rng.integers(int(0.3 * s), int(0.7 * s) + 1)) for s in in_shape]
        r = [int(rng.integers(max(3, int(0.05 * s)), int(0.18 * s) + 1)) for s in in_shape]
        e = (((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2)
        vol = np.maximum(vol, np.clip(1.5 - e, 0.0, 1.0))
    return vol


def _load_nifti(path: str, in_shape: Tuple[int, int, int]) -> np.ndarray:
    import nibabel as nib

    vol = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
    return pad_or_crop(vol, in_shape)


def _list_nifti(root: Optional[str]):
    paths = []
    if root and os.path.isdir(root):
        for pat in ("*.nii", "*.nii.gz"):
            paths.extend(glob.glob(os.path.join(root, pat)))
    return sorted(paths)


class LesionMaskDataset(Dataset):
    def __init__(self, root: Optional[str] = None,
                 in_shape: Tuple[int, int, int] = (96, 112, 96),
                 n_synth: int = 64, seed: int = 0, with_clinical: bool = False,
                 binarize: bool = True):
        self.in_shape = tuple(int(s) for s in in_shape)
        self.with_clinical = with_clinical
        self.binarize = binarize
        self.paths = _list_nifti(root)
        self.synthetic = len(self.paths) == 0
        self.seed = seed
        self.n = n_synth if self.synthetic else len(self.paths)
        self._clinical = None

    def __len__(self) -> int:
        return self.n

    def clinical_matrix(self) -> np.ndarray:
        if self._clinical is None:
            if self.synthetic:
                self._clinical = synthesize_clinical(self.n, CLINICAL_DIM, self.seed)
            else:
                rows = [build_clinical_vector(*_age_sex(p)) for p in self.paths]
                self._clinical = np.stack(rows, axis=0)
        return self._clinical

    def _load_volume(self, idx: int) -> np.ndarray:
        if self.synthetic:
            vol = _soft_field(self.in_shape, self.seed + 1000 + idx)
        else:
            vol = _load_nifti(self.paths[idx], self.in_shape)
        return binarize(vol) if self.binarize else vol.astype(np.float32)

    def __getitem__(self, idx: int):
        vol = torch.from_numpy(self._load_volume(idx)).unsqueeze(0)  # [1, D, H, W]
        if self.with_clinical:
            return vol, torch.from_numpy(self.clinical_matrix()[idx])
        return vol


class PairedLesionDisconnectomeDataset(Dataset):
    """Pairs lesion (binary) and disconnectome (continuous) by patient id."""

    def __init__(self, lesion_root: Optional[str] = None,
                 disconnectome_root: Optional[str] = None,
                 in_shape: Tuple[int, int, int] = (96, 112, 96),
                 n_synth: int = 64, seed: int = 0, with_clinical: bool = False):
        self.in_shape = tuple(int(s) for s in in_shape)
        self.with_clinical = with_clinical
        self.seed = seed
        les = _list_nifti(lesion_root)
        dis = _list_nifti(disconnectome_root)
        self.synthetic = len(les) == 0 or len(dis) == 0
        if self.synthetic:
            self.n = n_synth
            self.pairs = None
            self._ids = [f"synth{i:04d}" for i in range(self.n)]
        else:
            les_by_id = {parse_lesion_filename(p)["id"]: p for p in les}
            dis_by_id = {parse_lesion_filename(p)["id"]: p for p in dis}
            common = sorted(set(les_by_id) & set(dis_by_id))
            self.pairs = [(les_by_id[i], dis_by_id[i]) for i in common]
            self._ids = common
            self.n = len(self.pairs)
        self._clinical = None

    def __len__(self) -> int:
        return self.n

    def ids(self):
        return list(self._ids)

    def clinical_matrix(self) -> np.ndarray:
        if self._clinical is None:
            if self.synthetic:
                self._clinical = synthesize_clinical(self.n, CLINICAL_DIM, self.seed)
            else:
                rows = [build_clinical_vector(*_age_sex(lp)) for lp, _ in self.pairs]
                self._clinical = np.stack(rows, axis=0)
        return self._clinical

    def _load_pair(self, idx: int):
        if self.synthetic:
            soft = _soft_field(self.in_shape, self.seed + 1000 + idx)
            return binarize(soft), soft.astype(np.float32)
        lp, dp = self.pairs[idx]
        return binarize(_load_nifti(lp, self.in_shape)), _load_nifti(dp, self.in_shape).astype(np.float32)

    def __getitem__(self, idx: int):
        les, dis = self._load_pair(idx)
        les_t = torch.from_numpy(les).unsqueeze(0)
        dis_t = torch.from_numpy(dis).unsqueeze(0)
        if self.with_clinical:
            return les_t, dis_t, torch.from_numpy(self.clinical_matrix()[idx])
        return les_t, dis_t


def _age_sex(path: str):
    meta = parse_lesion_filename(path)
    return meta["age"], meta["sex"]
