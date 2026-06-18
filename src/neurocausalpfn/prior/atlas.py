"""Functional atlas and its subdivisions for InterSynth.

The parcellation (the 16 functional networks derived from NeuroQuery) and its
subdivisions by transcriptome (Allen) or receptome (Hansen) are reference
templates in MNI space. Here they are loaded from a NIfTI directory and the
utilities required by the InterSynth mechanism are exposed: the fraction of
each subnetwork covered by a lesion, and the subnetwork centroids to model
treatment assignment with observed confounding.

If the directory does not exist, a toy atlas is synthesized (regions labeled
in a small volume) so that the module is runnable and verifiable without the
real atlases. The mechanism logic is identical in both cases; only the source
of the labels changes.
"""
import os
from typing import Dict, Optional, Tuple

import numpy as np


class FunctionalAtlas:
    def __init__(self, network_labels: np.ndarray,
                 subnetworks: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        # network_labels: integer volume [D, H, W], 0 = background, 1..K = networks.
        # subnetworks[k] = (subnetwork_A_mask, subnetwork_B_mask) of network k.
        self.network_labels = network_labels
        self.subnetworks = subnetworks
        self.shape = tuple(network_labels.shape)
        self.networks = sorted(subnetworks.keys())
        self._centroids = {}
        for k, (a, b) in subnetworks.items():
            self._centroids[(k, 0)] = _centroid(a)
            self._centroids[(k, 1)] = _centroid(b)

    @property
    def n_networks(self) -> int:
        return len(self.networks)

    def subnetwork_overlap(self, lesion: np.ndarray, network: int) -> Tuple[float, float]:
        """Fraction of each subnetwork (A, B) of a network covered by the lesion."""
        a, b = self.subnetworks[network]
        return _overlap_fraction(lesion, a), _overlap_fraction(lesion, b)

    def centroid(self, network: int, sub: int) -> np.ndarray:
        return self._centroids[(network, sub)]

    @classmethod
    def from_dir(cls, atlas_dir: Optional[str], shape=(48, 56, 48), seed: int = 0,
                 modality: str = "receptor"):
        if atlas_dir and os.path.isdir(atlas_dir):
            return cls._load(atlas_dir, modality=modality)
        return cls.synthetic(shape=shape, seed=seed)

    @classmethod
    def _load(cls, atlas_dir: str, modality: str = "receptor"):
        """Loads the real Giles atlas.

        Expected structure:
        - functional_parcellation_2mm.nii.gz: volume with the networks labeled
          1..K (0 = background).
        - 2mm_parcellations/{modality}/: one file per network, with the two
          subnetworks encoded as labels 1 (subnetwork A) and 2 (subnetwork B);
          the number at the start of the filename is the network index.
        modality is 'receptor' (receptomic subdivision, Hansen) or 'genetics'
        (transcriptomic subdivision, Allen)."""
        import glob
        import nibabel as nib

        net_path = None
        for ext in (".nii.gz", ".nii"):
            cand = os.path.join(atlas_dir, "functional_parcellation_2mm" + ext)
            if os.path.exists(cand):
                net_path = cand
                break
        if net_path is None:
            raise FileNotFoundError(
                "functional_parcellation_2mm not found in " + atlas_dir)
        network_labels = np.asarray(nib.load(net_path).get_fdata()).astype(np.int64)

        sub_dir = os.path.join(atlas_dir, "2mm_parcellations", modality)
        files = sorted(glob.glob(os.path.join(sub_dir, "*.nii"))
                       + glob.glob(os.path.join(sub_dir, "*.nii.gz")))
        if not files:
            raise FileNotFoundError("no subnetworks found in " + sub_dir)
        subnetworks = {}
        for f in files:
            stem = os.path.basename(f)
            for ext in (".nii.gz", ".nii"):
                if stem.lower().endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            try:
                k = int(stem.split("_")[0])
            except ValueError:
                continue
            vol = np.asarray(nib.load(f).get_fdata())
            subnetworks[k] = ((vol == 1).astype(np.float32), (vol == 2).astype(np.float32))
        return cls(network_labels, subnetworks)

    @classmethod
    def synthetic(cls, shape=(48, 56, 48), n_networks: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        network_labels = np.zeros(shape, dtype=np.int64)
        subnetworks = {}
        zz, yy, xx = np.indices(shape)
        for k in range(1, n_networks + 1):
            center = np.array([int(rng.integers(int(0.25 * s), int(0.75 * s) + 1)) for s in shape])
            radius = np.array([int(rng.integers(int(0.12 * s), int(0.22 * s) + 1)) for s in shape])
            ell = (((zz - center[0]) / radius[0]) ** 2
                   + ((yy - center[1]) / radius[1]) ** 2
                   + ((xx - center[2]) / radius[2]) ** 2) <= 1.0
            network_labels[ell] = k
            axis = int(rng.integers(0, 3))            # subdivide by a plane passing through the centroid
            coord = (zz, yy, xx)[axis]
            half = coord <= center[axis]
            subnetworks[k] = ((ell & half).astype(np.float32), (ell & (~half)).astype(np.float32))
        return cls(network_labels, subnetworks)


def _overlap_fraction(lesion: np.ndarray, region: np.ndarray) -> float:
    region_size = float(region.sum())
    if region_size < 1.0:
        return 0.0
    return float((lesion * region).sum()) / region_size


def _centroid(mask: np.ndarray) -> np.ndarray:
    idx = np.argwhere(mask > 0.5)
    if len(idx) == 0:
        return np.array(mask.shape, dtype=np.float32) / 2.0
    return idx.mean(axis=0).astype(np.float32)
