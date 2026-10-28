#!/usr/bin/env python3
"""Which image directory a latents file was encoded from (F33 provenance).

Latents exported before the export recorded images_dir (13 Oct) carry only
Z and the file names. When two directories hold maps under the same names
(e.g. one cohort's disconnectomes from two generators), this re-encodes the
first --limit listed images from each candidate with the checkpoint that
produced the latents and compares the codes with the stored Z: the directory
the latents came from reproduces them up to floating-point noise, any other
does not.

    python scripts/which_image_dir.py --latents outputs/latents_isles_disco/E1_seed0_disco.npz \\
        --checkpoint <the vae_disconnectome.pt of E1 seed 0> \\
        --candidates ~/Scratch/isles_qdisco ~/Scratch/isles_disco --limit 16

Prints one line per candidate (its folder name, max |dZ|, median row
correlation); no file name is printed.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np


def _export_module():
    spec = importlib.util.spec_from_file_location(
        "export_latents", os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_latents.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("export_latents", mod)
    spec.loader.exec_module(mod)
    return mod


def compare_codes(z_ref: np.ndarray, z_new: np.ndarray) -> dict:
    """Max absolute difference and the median per-row Pearson correlation."""
    z_ref, z_new = np.asarray(z_ref, np.float64), np.asarray(z_new, np.float64)
    corr = [np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else float("nan")
            for a, b in zip(z_ref, z_new)]
    return {"max_abs_diff": float(np.abs(z_ref - z_new).max()), "median_row_corr": float(np.nanmedian(corr))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--latents", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    ex = _export_module()
    with np.load(args.latents, allow_pickle=True) as z:
        Z, names = np.asarray(z["Z"]), [os.path.basename(str(n)) for n in z["files"]]
    k = min(args.limit, len(names)) if args.limit else len(names)
    model, rep, res = ex.load_frozen_vae(args.checkpoint, args.device)
    out = {}
    for d in args.candidates:
        d = os.path.expanduser(d)
        files = [os.path.join(d, n) for n in names[:k]]
        missing = sum(not os.path.exists(f) for f in files)
        if missing:
            print(f"{os.path.basename(os.path.normpath(d))}: {missing} of {k} listed images missing")
            continue
        s = compare_codes(Z[:k], ex.export_latents(model, rep, res, files, device=args.device))
        out[d] = s
        print(f"{os.path.basename(os.path.normpath(d))}: {k} images | max |dZ| {s['max_abs_diff']:.2e} | "
              f"median row corr {s['median_row_corr']:.4f}")
    return out


if __name__ == "__main__":
    main()
