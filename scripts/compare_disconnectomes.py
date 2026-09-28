"""Compare two sets of disconnectome maps of the same lesions, file by file
(same basename in both folders): calibration of a freshly built disconnectome
pipeline against the reference maps of the training cohort.

    python scripts/compare_disconnectomes.py <new maps dir> <reference maps dir> [-o compare.csv]

Per lesion: Pearson correlation over the union of non-zero voxels, Dice of the
supports (> 0) and of the cores (> 0.5), the two map sums, their ratio and the
maximum absolute difference. Prints medians and 5-95 % ranges plus the value
ranges of both sets, so a mismatch in scale (e.g. percent vs fraction) or in
support (thresholded vs raw maps) is visible at once. Only basenames are
printed.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import nibabel as nib


def compare_pair(a: np.ndarray, b: np.ndarray) -> dict:
    """Agreement statistics between two non-negative maps on one grid."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sa, sb = a > 0, b > 0
    union = sa | sb
    if union.sum() >= 2 and a[union].std() > 0 and b[union].std() > 0:
        r = float(np.corrcoef(a[union], b[union])[0, 1])
    else:
        r = float("nan")
    dice_sup = 2.0 * (sa & sb).sum() / max(sa.sum() + sb.sum(), 1)
    ca, cb = a > 0.5, b > 0.5
    dice_core = 2.0 * (ca & cb).sum() / max(ca.sum() + cb.sum(), 1)
    sum_a, sum_b = float(a.sum()), float(b.sum())
    return {"pearson": round(r, 4), "dice_support": round(float(dice_sup), 4),
            "dice_core": round(float(dice_core), 4),
            "sum_new": round(sum_a, 1), "sum_ref": round(sum_b, 1),
            "sum_ratio": round(sum_a / sum_b, 4) if sum_b > 0 else float("nan"),
            "max_new": round(float(a.max()), 4), "max_ref": round(float(b.max()), 4),
            "max_abs_diff": round(float(np.abs(a - b).max()), 4)}


def _stem(path: str) -> str:
    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new_dir")
    ap.add_argument("ref_dir")
    ap.add_argument("-o", "--out", default="compare_disconnectomes.csv")
    args = ap.parse_args()

    new = {_stem(p): p for p in glob.glob(os.path.join(args.new_dir, "*.nii*"))}
    ref = {_stem(p): p for p in glob.glob(os.path.join(args.ref_dir, "*.nii*"))}
    common = sorted(set(new) & set(ref))
    if not common:
        raise SystemExit(f"no common basenames between {args.new_dir} and {args.ref_dir}")
    print(f"new {len(new)} | reference {len(ref)} | common {len(common)}")

    rows = []
    for k, stem in enumerate(common, 1):
        ia, ib = nib.load(new[stem]), nib.load(ref[stem])
        if ia.shape != ib.shape:
            print(f"  {stem}: shape {ia.shape} vs {ib.shape}, skipped")
            continue
        if not np.allclose(ia.affine, ib.affine, atol=1e-3):
            print(f"  {stem}: affines differ (values compared voxel-wise anyway)")
        stats = compare_pair(np.asanyarray(ia.dataobj), np.asanyarray(ib.dataobj))
        rows.append({"file": stem, **stats})
        if k % 10 == 0:
            print(f"  {k}/{len(common)}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def _q(col):
        v = np.array([r[col] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        return (f"median {np.median(v):.3f}, 5-95 % [{np.percentile(v, 5):.3f}, "
                f"{np.percentile(v, 95):.3f}]" if v.size else "n/a")

    print(f"pairs compared {len(rows)}")
    for col in ("pearson", "dice_support", "dice_core", "sum_ratio", "max_abs_diff"):
        print(f"  {col:13s} {_q(col)}")
    print(f"  max value     new {_q('max_new')} | ref {_q('max_ref')}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
