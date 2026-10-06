#!/usr/bin/env python3
"""Precompute the anatomy the Neuro-Prior needs from a lesion directory.

For every lesion mask: overlap fractions with the two subnetworks of each
atlas network [m, K, 2], centroid [m, 3] and volume [m]; optionally the
encoder latents of the same files (an npz with Z and files, as written by
export_latents.py) aligned by basename. The result is one npz that
``NeuroPriorCohort.from_cache`` loads, so the transformer trains without the
masks in memory and the same cache serves every latent pool of that cohort.

    python scripts/build_prior_cache.py --lesions-dir "data/Full data/lesions" \
        --atlas-dir data/atlases --latents outputs/latents_E1/E1_seed0_disco.npz \
        --out outputs/prior_cache/uclh_E1_seed0.npz
"""
import argparse
import glob
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lesions-dir", required=True)
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--modality", default="receptor", choices=["receptor", "genetics"])
    ap.add_argument("--latents", default=None, help="npz with Z [m, d] and files (optional)")
    ap.add_argument("--threshold", type=float, default=0.5, help="binarisation of continuous masks")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import nibabel as nib

    from neurocausalpfn.prior.atlas import FunctionalAtlas, _centroid
    from neurocausalpfn.prior.intersynth_atlas import compute_overlaps

    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.lesions_dir), "*.nii*")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"no niftis in {args.lesions_dir}")
    atlas = FunctionalAtlas.from_dir(args.atlas_dir, modality=args.modality)
    overlaps, centroids, volumes = [], [], []
    for i, f in enumerate(files):
        m = (np.asarray(nib.load(f).get_fdata()) > args.threshold).astype(np.float32)
        if m.shape != atlas.shape:
            sys.exit(f"{os.path.basename(f)}: shape {m.shape} differs from the atlas {atlas.shape}")
        overlaps.append(compute_overlaps(atlas, m)); centroids.append(_centroid(m)); volumes.append(float(m.sum()))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)
    out = {"overlaps": np.stack(overlaps), "centroids": np.stack(centroids),
           "volumes": np.asarray(volumes), "files": np.asarray([os.path.basename(f) for f in files]),
           "networks": np.asarray(atlas.networks)}
    if args.latents:
        with np.load(args.latents, allow_pickle=True) as z:
            Z, names = np.asarray(z["Z"], dtype=np.float32), [os.path.basename(str(n)) for n in z["files"]]
        pos = {n: k for k, n in enumerate(names)}
        missing = [n for n in out["files"] if n not in pos]
        if missing:
            sys.exit(f"{len(missing)} lesions without a latent, e.g. {missing[0]}")
        out["Z"] = Z[[pos[n] for n in out["files"]]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    hit = (out["overlaps"].max(axis=(1, 2)) > 0.05).mean()
    print(f"wrote {args.out}: {len(files)} lesions, {atlas.n_networks} networks, "
          f"{100 * hit:.0f} % hit at least one subnetwork above 5 %"
          + (f", Z {out['Z'].shape}" if "Z" in out else ""))


if __name__ == "__main__":
    main()
