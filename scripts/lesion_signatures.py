"""Content signatures for lesion masks, to identify overlapping files between
two copies of a dataset whose filenames differ.

Copies of the same mask can carry different filenames and headers while the
voxels are identical. This computes a per-file signature that ignores filename
and header:

- sha1 of the canonicalised binary mask (definitive when both copies share the
  same voxel grid);
- lesion volume in mm^3 and centroid in world coordinates (mm), comparable even
  if the two copies are on different voxel grids.

Run the SAME script over each copy (one CSV each), then cross the two small CSVs
with match_lesion_datasets.py -- only the signatures travel, never the images.

    python scripts/lesion_signatures.py <dir_of_nii_gz> -o signatures.csv

Requires numpy + nibabel.
"""
import argparse
import csv
import glob
import hashlib
import os

import numpy as np
import nibabel as nib


def signature(path: str) -> dict:
    img = nib.as_closest_canonical(nib.load(path))       # stable orientation, header-independent
    data = np.asanyarray(img.dataobj)
    mask = data > 0
    n = int(mask.sum())
    zooms = img.header.get_zooms()[:3]
    vol_mm3 = n * float(np.prod(zooms))
    if n > 0:
        world = nib.affines.apply_affine(img.affine, np.argwhere(mask).mean(axis=0))
        cx, cy, cz = (float(v) for v in world)
    else:
        cx = cy = cz = float("nan")
    sha1 = hashlib.sha1(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()
    return {"shape": "x".join(str(s) for s in mask.shape), "n_voxels": n,
            "vol_mm3": round(vol_mm3, 3), "cx_mm": round(cx, 3),
            "cy_mm": round(cy, 3), "cz_mm": round(cz, 3), "sha1": sha1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("-o", "--out", default="signatures.csv")
    ap.add_argument("--pattern", default="*.nii.gz")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.directory, "**", args.pattern), recursive=True))
    if not paths:
        raise SystemExit(f"no {args.pattern} found under {args.directory}")
    cols = ["filename", "shape", "n_voxels", "vol_mm3", "cx_mm", "cy_mm", "cz_mm", "sha1"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, p in enumerate(paths):
            try:
                row = signature(p)
                row["filename"] = os.path.basename(p)
                w.writerow(row)
            except Exception as e:                        # keep going, report the file
                print(f"skip {p}: {e}")
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(paths)}")
    print(f"wrote {len(paths)} signatures to {args.out}")


if __name__ == "__main__":
    main()
