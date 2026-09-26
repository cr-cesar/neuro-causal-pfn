"""Resample the ISLES24 follow-up DWI and lesion mask of every case to an
isotropic grid (default 2 mm) before uploading them to the cluster.

    python scripts/isles24_downsample.py <release root> <out root> [--mm 2.0]

The native CT-space grids (~0.4 x 0.4 x 2 mm, 20-30 M voxels, float64) are
far finer than needed to drive an affine registration to a 2 mm template and
weigh ~60 MB per case compressed. Resampling DWI (trilinear) and mask (nearest
neighbour) onto the same coarse grid keeps the world coordinates, shrinks the
upload by ~30x and leaves the registration script unchanged: the output
mirrors ``derivatives/<sub>/ses-02/`` with the original filenames.

Requires nibabel + scipy.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="release root (contains derivatives/)")
    ap.add_argument("out", help="output root (derivatives/<sub>/ses-02/ is created inside)")
    ap.add_argument("--mm", type=float, default=2.0)
    args = ap.parse_args()

    subs = sorted(glob.glob(os.path.join(args.root, "derivatives", "sub-*")))
    done = skipped = 0
    for sd in subs:
        sub = os.path.basename(sd)
        ses = os.path.join(sd, "ses-02")
        dwi = glob.glob(os.path.join(ses, "*_space-ncct_dwi.nii.gz"))
        msk = glob.glob(os.path.join(ses, "*_space-ncct_lesion-msk.nii.gz"))
        if not dwi or not msk:
            skipped += 1
            continue
        od = os.path.join(args.out, "derivatives", sub, "ses-02")
        os.makedirs(od, exist_ok=True)
        d_img = nib.load(dwi[0])
        d_out = resample_to_output(d_img, voxel_sizes=args.mm, order=1)
        d_out.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(np.asanyarray(d_out.dataobj).astype(np.float32), d_out.affine),
                 os.path.join(od, os.path.basename(dwi[0])))
        m_img = nib.load(msk[0])
        m_bin = nib.Nifti1Image((np.asanyarray(m_img.dataobj) > 0).astype(np.uint8), m_img.affine)
        m_out = resample_to_output(m_bin, voxel_sizes=args.mm, order=0)
        nib.save(nib.Nifti1Image(np.asanyarray(m_out.dataobj).astype(np.uint8), m_out.affine),
                 os.path.join(od, os.path.basename(msk[0])))
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(subs)}", flush=True)
    print(f"cases written {done} | skipped (no dwi or mask) {skipped} | grid {args.mm} mm -> {args.out}")


if __name__ == "__main__":
    main()
