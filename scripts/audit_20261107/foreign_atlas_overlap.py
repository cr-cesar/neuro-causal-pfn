"""Brief 8 audit (7 Nov), question 7: does a proposed 'foreign' atlas
(Brainnetome-246, Schaefer-400 with the Tian subcortex, ...) overlap the 16
benchmark networks too much to count as held out? CPU, seconds.

For every benchmark atom (16 whole networks, 32 receptor halves, 32 genetics
halves; the atoms of the virtual-trial family) it reports
  - best_single: the best Dice of one foreign parcel (or foreign network);
  - best_union2: the best Dice of a union of at most two foreign parcels,
    the atoms-per-side rule of the virtual-trial family (partitions.py);
  - best_greedy: the best Dice of a greedy union of any number of parcels,
    each added while it raises the Dice (an upper bound on what a prior
    family drawing unions of foreign parcels could reproduce).
Voxel Dice is necessary, not sufficient: the stricter test is the proximity
of the induced effect vectors (the audit's proximity check), because a 5 %
coverage rule on a small parcel can mimic a half that its voxels do not.

Both atlases must be in the same 2 mm MNI grid (resample the foreign one
with nearest-neighbour interpolation first). Aggregates only.

    python scripts/audit_20261107/foreign_atlas_overlap.py --atlas-dir ~/export_bcb/atlases \\
        --foreign BN_Atlas_246_2mm.nii.gz [--foreign-networks parcel_to_network.csv] --out bn246_overlap.csv
"""
import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple

import numpy as np


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else 0.0


def best_overlaps(atom: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """atom: boolean mask; labels: integer foreign parcellation (0 = none)."""
    ids, counts = np.unique(labels[atom], return_counts=True)
    keep = ids != 0
    ids, counts = ids[keep], counts[keep]
    if not len(ids):
        return {"best_single": 0.0, "best_union2": 0.0, "best_greedy": 0.0, "n_touching": 0}
    sizes = {int(i): int((labels == i).sum()) for i in ids}
    inter = {int(i): int(c) for i, c in zip(ids, counts)}
    na = int(atom.sum())
    d1 = {i: 2.0 * inter[i] / (na + sizes[i]) for i in inter}
    order = sorted(d1, key=d1.get, reverse=True)
    top = order[:30]                                       # pairs among the 30 best parcels
    best2 = max(d1.values())
    for x in range(len(top)):
        for y in range(x + 1, len(top)):
            i, j = top[x], top[y]
            best2 = max(best2, 2.0 * (inter[i] + inter[j]) / (na + sizes[i] + sizes[j]))
    cur_i, cur_s, best_g = 0, 0, 0.0
    for i in sorted(inter, key=lambda k: inter[k] / sizes[k], reverse=True):
        cand = 2.0 * (cur_i + inter[i]) / (na + cur_s + sizes[i])
        if cand > best_g:
            best_g, cur_i, cur_s = cand, cur_i + inter[i], cur_s + sizes[i]
    return {"best_single": max(d1.values()), "best_union2": best2, "best_greedy": best_g,
            "n_touching": len(inter)}


def benchmark_atoms(atlas_dir: str) -> List[Tuple[str, int, int, np.ndarray]]:
    import nibabel as nib

    atoms = []
    func = np.asarray(nib.load(os.path.join(atlas_dir, "functional_parcellation_2mm.nii.gz")).dataobj)
    for k in range(1, 17):
        atoms.append(("network", k, -1, func == k))
    for split in ("receptor", "genetics"):
        for f in sorted(glob.glob(os.path.join(atlas_dir, "2mm_parcellations", split, "*.nii*"))):
            k = int(os.path.basename(f).split("_")[0])
            arr = np.asarray(nib.load(f).dataobj)
            for h, lab in ((0, 1), (1, 2)):
                atoms.append((split, k, h, arr == lab))
    return atoms


def main():
    import nibabel as nib

    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--foreign", required=True, help="integer label image in the benchmark's grid")
    ap.add_argument("--foreign-networks", default=None,
                    help="optional CSV parcel,network: also score the foreign networks as units")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    labels = np.asarray(nib.load(a.foreign).dataobj).astype(int)
    atoms = benchmark_atoms(a.atlas_dir)
    if labels.shape != atoms[0][3].shape:
        raise SystemExit(f"grid mismatch: foreign {labels.shape} vs benchmark {atoms[0][3].shape}; resample first")
    net_labels = None
    if a.foreign_networks:
        m = {}
        with open(a.foreign_networks) as f:
            for r in csv.DictReader(f):
                m[int(r["parcel"])] = int(r["network"])
        lut = np.zeros(labels.max() + 1, dtype=int)
        for p, n in m.items():
            if p <= labels.max():
                lut[p] = n
        net_labels = lut[labels]
    rows = []
    for ch, k, h, mask in atoms:
        r = {"channel": ch, "network": k, "half": h, "voxels": int(mask.sum()), **best_overlaps(mask, labels)}
        if net_labels is not None:
            r["best_single_foreign_network"] = best_overlaps(mask, net_labels)["best_single"]
        rows.append(r)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for key in ("best_single", "best_union2", "best_greedy"):
        v = np.array([r[key] for r in rows])
        print(f"{key}: median {np.median(v):.2f}, max {v.max():.2f}, atoms >= 0.5: {(v >= 0.5).sum()} of {len(v)}")


if __name__ == "__main__":
    main()
