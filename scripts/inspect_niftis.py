"""Summarise the geometry of a folder of NIfTI files before feeding a cohort
to the pipeline (which expects binary masks on the MNI152 2 mm grid,
91x109x91, one shared affine).

    python scripts/inspect_niftis.py <folder> [--ref <one lesion of the training cohort>]
        [--sample 300]

Prints, grouped by (shape, voxel size, orientation, dtype): how many files,
how many are binary, the value range, and whether the affine equals the
reference's. Filenames are only shown as patterns (digits replaced by N), so
the output can be shared.
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import re

import numpy as np
import nibabel as nib


def pattern(name: str) -> str:
    return re.sub(r"\d", "N", name)


def describe(path: str, ref_affine=None):
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    finite = data[np.isfinite(data)]
    vals = np.unique(finite) if finite.size and finite.size < 5_000_000 else np.array([])
    binary = bool(vals.size <= 2 and set(np.round(vals, 6).tolist()) <= {0.0, 1.0}) if vals.size else False
    same_affine = (np.allclose(img.affine, ref_affine, atol=1e-3) if ref_affine is not None else None)
    return {
        "shape": tuple(int(s) for s in img.shape[:3]),
        "zooms": tuple(round(float(z), 2) for z in img.header.get_zooms()[:3]),
        "orient": "".join(nib.aff2axcodes(img.affine)),
        "dtype": str(data.dtype),
        "binary": binary,
        "vmin": float(finite.min()) if finite.size else float("nan"),
        "vmax": float(finite.max()) if finite.size else float("nan"),
        "n_pos": int((data > 0).sum()),
        "same_affine": same_affine,
        "ndim": data.ndim,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--ref", default=None, help="a NIfTI whose grid is the target (e.g. one training lesion)")
    ap.add_argument("--sample", type=int, default=300, help="files to open per name pattern")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "**", "*.nii.gz"), recursive=True)
                   + glob.glob(os.path.join(args.folder, "**", "*.nii"), recursive=True))
    if not files:
        raise SystemExit(f"no NIfTI files under {args.folder}")
    ref_affine = nib.load(args.ref).affine if args.ref else None

    by_pattern = collections.defaultdict(list)
    for f in files:
        by_pattern[pattern(os.path.basename(f))].append(f)
    print(f"{len(files)} files | {len(by_pattern)} name patterns"
          + (f" | reference grid {nib.load(args.ref).shape} from {pattern(os.path.basename(args.ref))}"
             if args.ref else ""))
    for pat, paths in sorted(by_pattern.items(), key=lambda kv: -len(kv[1])):
        groups = collections.Counter()
        n_bin = n_same = n_pos_zero = 0
        vmins, vmaxs, npos = [], [], []
        for p in paths[: args.sample]:
            d = describe(p, ref_affine)
            groups[(d["shape"], d["zooms"], d["orient"], d["dtype"], d["ndim"])] += 1
            n_bin += d["binary"]
            n_same += bool(d["same_affine"])
            n_pos_zero += d["n_pos"] == 0
            vmins.append(d["vmin"]); vmaxs.append(d["vmax"]); npos.append(d["n_pos"])
        n = min(len(paths), args.sample)
        print(f"\n== {pat}   ({len(paths)} files, {n} inspected)")
        for (shape, zooms, orient, dtype, ndim), c in groups.most_common():
            print(f"   {c:5d}  shape {shape}  voxel mm {zooms}  orient {orient}  dtype {dtype}"
                  + (f"  ndim {ndim}" if ndim != 3 else ""))
        print(f"   binary {n_bin}/{n} | values [{np.nanmin(vmins):.3g}, {np.nanmax(vmaxs):.3g}] | "
              f"empty {n_pos_zero} | positive voxels median {int(np.median(npos))}"
              + (f" | same affine as ref {n_same}/{n}" if ref_affine is not None else ""))


if __name__ == "__main__":
    main()
