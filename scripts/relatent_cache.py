#!/usr/bin/env python3
"""Copies of a prior cache with the latents of other encoders (multi-encoder pools).

The anatomy of a cache (overlaps, centroids, volumes, the files it lists) does
not depend on the encoder; only ``Z`` does. Given a cache built once with
scripts/build_prior_cache.py and the latents of the same images under other
encoders (scripts/export_latents.py, one npz per encoder), this writes one
cache per latents file with ``Z`` replaced, rows aligned by file name. Every
latents file must list exactly the cache's images, have finite values and the
cache's latent size; nothing is written otherwise.

    python scripts/relatent_cache.py --cache caches/kch_bcb.npz \\
        --latents "latents/kch/seed*/fold*/*.npz" --out-dir caches/kch_multi

Output names: ``<cache stem>__<seedS>_<foldK>.npz`` when the latents path holds
seed and fold folders, else the latents file's stem. Only counts are printed.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np


def _tag(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    keep = [p for p in parts[:-1] if re.fullmatch(r"(seed|fold)\d+", p)]
    return "_".join(keep) if keep else os.path.splitext(os.path.basename(path))[0]


def relatent(cache: dict, latents: dict, source: str) -> dict:
    """``cache`` with ``Z`` from ``latents`` (rows matched by basename)."""
    names = [os.path.basename(str(f)) for f in cache["files"]]
    lat_names = [os.path.basename(str(f)) for f in latents["files"]]
    if sorted(names) != sorted(lat_names) or len(set(lat_names)) != len(lat_names):
        raise ValueError(f"{source}: the latents do not list exactly the cache's images")
    Z = np.asarray(latents["Z"], dtype=np.float32)
    if not np.isfinite(Z).all():
        raise ValueError(f"{source}: non-finite latents")
    if "Z" in cache and np.asarray(cache["Z"]).shape[1] != Z.shape[1]:
        raise ValueError(f"{source}: latent size {Z.shape[1]}, the cache has {np.asarray(cache['Z']).shape[1]}")
    pos = {n: i for i, n in enumerate(lat_names)}
    out = dict(cache)
    out["Z"] = Z[[pos[n] for n in names]]
    out["latents_source"] = np.array(source)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="a cache from scripts/build_prior_cache.py")
    ap.add_argument("--latents", required=True, nargs="+", help="latents npz files or globs (Z and files)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    paths = sorted({p for g in a.latents for p in glob.glob(g)})
    if not paths:
        sys.exit("no latents files")
    with np.load(a.cache, allow_pickle=True) as z:
        cache = {k: z[k] for k in z.files}
    stem = os.path.splitext(os.path.basename(a.cache))[0]
    tags = [_tag(p) for p in paths]
    if len(set(tags)) != len(tags):
        sys.exit("two latents files map to the same output name")
    built = []
    for p, t in zip(paths, tags):
        with np.load(p, allow_pickle=True) as z:
            built.append((t, relatent(cache, {k: z[k] for k in z.files}, os.path.abspath(p))))
    os.makedirs(a.out_dir, exist_ok=True)
    for t, out in built:
        np.savez_compressed(os.path.join(a.out_dir, f"{stem}__{t}.npz"), **out)
    print(f"{len(built)} caches of {len(cache['files'])} images, latent size {built[0][1]['Z'].shape[1]} "
          f"-> {a.out_dir}")


if __name__ == "__main__":
    main()
