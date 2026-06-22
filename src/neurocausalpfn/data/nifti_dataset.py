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

from .clinical import (CLINICAL_DIM, CLINICAL_DIM_EXTENDED, build_clinical_vector,
                       build_clinical_vector_extended, load_clinical_table,
                       load_outcome_table, parse_lesion_filename, synthesize_clinical)
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


def _clinical_rows(paths, clinical_csv: Optional[str]) -> np.ndarray:
    """Clinical matrix from a list of lesion filenames. With a CSV it builds the
    extended vector (adding NIHSS and time-to-scan by id); without it, the
    age/sex vector parsed from the filename."""
    if clinical_csv:
        table = load_clinical_table(clinical_csv)
        rows = []
        for p in paths:
            m = parse_lesion_filename(p)
            rec = table.get(str(m["id"]).strip(), {})
            rows.append(build_clinical_vector_extended(
                m["age"], m["sex"], rec.get("nihss"), rec.get("time_to_scan")))
        return np.stack(rows, axis=0)
    rows = [build_clinical_vector(*_age_sex(p)) for p in paths]
    return np.stack(rows, axis=0)


def _target_rows(ids, outcome_csv: Optional[str], n: int, synthetic: bool, seed: int) -> np.ndarray:
    """Binary outcome vector for the PNS auxiliary loss (Arm B). With an outcome
    CSV and real data it is read by id; otherwise a seeded synthetic binary label
    is used so the path runs in the prototype."""
    if outcome_csv and not synthetic and ids is not None:
        table = load_outcome_table(outcome_csv)
        return np.array([float(table.get(str(i).strip(), 0.0)) for i in ids], dtype=np.float32)
    rng = np.random.default_rng(seed + 999)
    return rng.integers(0, 2, n).astype(np.float32)


class LesionMaskDataset(Dataset):
    def __init__(self, root: Optional[str] = None,
                 in_shape: Tuple[int, int, int] = (96, 112, 96),
                 n_synth: int = 64, seed: int = 0, with_clinical: bool = False,
                 binarize: bool = True, clinical_csv: Optional[str] = None,
                 with_target: bool = False, outcome_csv: Optional[str] = None):
        self.in_shape = tuple(int(s) for s in in_shape)
        self.with_clinical = with_clinical
        self.binarize = binarize
        self.clinical_csv = clinical_csv
        self.with_target = with_target
        self.outcome_csv = outcome_csv
        self.paths = _list_nifti(root)
        self.synthetic = len(self.paths) == 0
        self.seed = seed
        self.n = n_synth if self.synthetic else len(self.paths)
        self._clinical = None
        self._target = None

    def __len__(self) -> int:
        return self.n

    def clinical_matrix(self) -> np.ndarray:
        if self._clinical is None:
            if self.synthetic:
                d = CLINICAL_DIM_EXTENDED if self.clinical_csv else CLINICAL_DIM
                self._clinical = synthesize_clinical(self.n, d, self.seed)
            else:
                self._clinical = _clinical_rows(self.paths, self.clinical_csv)
        return self._clinical

    def clinical_dim(self) -> int:
        return int(self.clinical_matrix().shape[1])

    def target_vector(self) -> np.ndarray:
        if self._target is None:
            self._target = _target_rows(
                [parse_lesion_filename(p)["id"] for p in self.paths] if not self.synthetic else None,
                self.outcome_csv, self.n, self.synthetic, self.seed)
        return self._target

    def _load_volume(self, idx: int) -> np.ndarray:
        if self.synthetic:
            vol = _soft_field(self.in_shape, self.seed + 1000 + idx)
        else:
            vol = _load_nifti(self.paths[idx], self.in_shape)
        return binarize(vol) if self.binarize else vol.astype(np.float32)

    def __getitem__(self, idx: int):
        vol = torch.from_numpy(self._load_volume(idx)).unsqueeze(0)  # [1, D, H, W]
        extras = []
        if self.with_clinical:
            extras.append(torch.from_numpy(self.clinical_matrix()[idx]))
        if self.with_target:
            extras.append(torch.tensor(self.target_vector()[idx]))
        return (vol, *extras) if extras else vol


class PairedLesionDisconnectomeDataset(Dataset):
    """Pairs lesion (binary) and disconnectome (continuous) by patient id."""

    def __init__(self, lesion_root: Optional[str] = None,
                 disconnectome_root: Optional[str] = None,
                 in_shape: Tuple[int, int, int] = (96, 112, 96),
                 n_synth: int = 64, seed: int = 0, with_clinical: bool = False,
                 stack_channels: bool = False, clinical_csv: Optional[str] = None,
                 with_target: bool = False, outcome_csv: Optional[str] = None):
        self.in_shape = tuple(int(s) for s in in_shape)
        self.with_clinical = with_clinical
        self.stack_channels = stack_channels
        self.clinical_csv = clinical_csv
        self.with_target = with_target
        self.outcome_csv = outcome_csv
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
        self._target = None

    def __len__(self) -> int:
        return self.n

    def ids(self):
        return list(self._ids)

    def clinical_matrix(self) -> np.ndarray:
        if self._clinical is None:
            if self.synthetic:
                d = CLINICAL_DIM_EXTENDED if self.clinical_csv else CLINICAL_DIM
                self._clinical = synthesize_clinical(self.n, d, self.seed)
            else:
                self._clinical = _clinical_rows([lp for lp, _ in self.pairs], self.clinical_csv)
        return self._clinical

    def clinical_dim(self) -> int:
        return int(self.clinical_matrix().shape[1])

    def target_vector(self) -> np.ndarray:
        if self._target is None:
            self._target = _target_rows(self._ids if not self.synthetic else None,
                                        self.outcome_csv, self.n, self.synthetic, self.seed)
        return self._target

    def _load_pair(self, idx: int):
        if self.synthetic:
            soft = _soft_field(self.in_shape, self.seed + 1000 + idx)
            return binarize(soft), soft.astype(np.float32)
        lp, dp = self.pairs[idx]
        return binarize(_load_nifti(lp, self.in_shape)), _load_nifti(dp, self.in_shape).astype(np.float32)

    def __getitem__(self, idx: int):
        les, dis = self._load_pair(idx)
        clin = torch.from_numpy(self.clinical_matrix()[idx]) if self.with_clinical else None
        target = torch.tensor(self.target_vector()[idx]) if self.with_target else None
        if self.stack_channels:
            vol = torch.from_numpy(np.stack([les, dis], axis=0))  # [2, D, H, W]
            extras = [t for t in (clin, target) if t is not None]
            return (vol, *extras) if extras else vol
        les_t = torch.from_numpy(les).unsqueeze(0)
        dis_t = torch.from_numpy(dis).unsqueeze(0)
        extras = [t for t in (clin, target) if t is not None]
        return (les_t, dis_t, *extras) if extras else (les_t, dis_t)


def _age_sex(path: str):
    meta = parse_lesion_filename(path)
    return meta["age"], meta["sex"]
