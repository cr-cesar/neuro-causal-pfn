"""Random anatomical partitions for the virtual-trial family (route C, 27 Oct).

The virtual-trial family of the Neuro-Prior gives each lesion an all-or-none
effect by whether it covers, above a random threshold, one half or the other
of an atlas network. The halves of the cache's atlas are the benchmark's own
subnetwork pairs, so a prior built on them alone trains on the partitions it
is tested on (guardrail 2 of the 26 Oct plan). This module builds, per
process, a set of *virtual networks* whose two sides are random unions of the
anatomical atoms the cache holds (the halves of the receptor and genetics
splits and the whole networks), never the two halves of one benchmark
network. Overlap of a union is approximated by the largest overlap among its
atoms (a lesion hits the side when it hits any atom).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Atom = Tuple[str, int, int]     # (channel, network index, half); half = -1 for a whole network


def atom_table(overlaps: np.ndarray, extra: Optional[Dict[str, np.ndarray]] = None
               ) -> Tuple[List[Atom], np.ndarray]:
    """Every anatomical atom of a cache as a column: returns the atom labels
    and a [m, n_atoms] array of overlap fractions. ``overlaps`` is the primary
    [m, K, 2] array; ``extra`` the optional channels of the cache."""
    cols, labels = [], []
    arrays = {"primary": np.asarray(overlaps, dtype=np.float32)}
    for key, arr in (extra or {}).items():
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 2:
            arrays[key.replace("overlaps_", "")] = arr
        elif arr.ndim == 2:
            arrays[key.replace("overlaps_", "")] = arr[:, :, None]
    for name, arr in arrays.items():
        for k in range(arr.shape[1]):
            for h in range(arr.shape[2]):
                cols.append(arr[:, k, h])
                labels.append((name, k, h if arr.shape[2] == 2 else -1))
    return labels, np.stack(cols, axis=1)


def is_benchmark_pair(side_a: Sequence[int], side_b: Sequence[int], labels: Sequence[Atom]) -> bool:
    """True when the two sides are exactly the two halves of one network of
    one split: the benchmark's own partition."""
    if len(side_a) != 1 or len(side_b) != 1:
        return False
    ca, ka, ha = labels[side_a[0]]
    cb, kb, hb = labels[side_b[0]]
    return ca == cb and ka == kb and {ha, hb} == {0, 1}


def draw_virtual_networks(rng: np.random.Generator, labels: Sequence[Atom], table: np.ndarray,
                          n_networks: int, atoms_per_side=(1, 2), max_tries: int = 100
                          ) -> Tuple[np.ndarray, List[Dict]]:
    """``n_networks`` virtual networks: each side a union of 1 to 2 atoms,
    sides disjoint, never a benchmark pair. Returns the virtual overlaps
    [m, n_networks, 2] and a description per network."""
    lo, hi = int(atoms_per_side[0]), int(atoms_per_side[1])
    n_atoms = table.shape[1]
    out = np.zeros((table.shape[0], n_networks, 2), dtype=np.float32)
    desc = []
    for p in range(n_networks):
        for _ in range(max_tries):
            na, nb = int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1))
            pick = rng.choice(n_atoms, size=min(na + nb, n_atoms), replace=False)
            a, b = list(pick[:na]), list(pick[na:])
            if not b or is_benchmark_pair(a, b, labels):
                continue
            out[:, p, 0] = table[:, a].max(axis=1)
            out[:, p, 1] = table[:, b].max(axis=1)
            desc.append({"A": [labels[i] for i in a], "B": [labels[i] for i in b]})
            break
        else:
            raise RuntimeError("could not draw a virtual network that is not a benchmark pair")
    return out, desc


class VirtualAtlas:
    """The atlas interface the virtual-trial DGP needs (``networks``,
    ``n_networks``, ``centroid``) for a set of virtual networks. Centroids are
    overlap-weighted means of the pool lesions' centroids, so the 'axis'
    allocation mechanism keeps a geometric meaning."""

    def __init__(self, virtual_overlaps: np.ndarray, centroids: np.ndarray):
        self.virtual_overlaps = np.asarray(virtual_overlaps, dtype=np.float32)
        self.networks = list(range(self.virtual_overlaps.shape[1]))
        self.n_networks = len(self.networks)
        self._centroids = np.asarray(centroids, dtype=np.float64)
        self.subnetworks = {}

    def centroid(self, network: int, sub: int) -> np.ndarray:
        w = self.virtual_overlaps[:, network, sub].astype(np.float64)
        if w.sum() <= 0:
            return self._centroids.mean(axis=0)
        return (self._centroids * w[:, None]).sum(axis=0) / w.sum()
