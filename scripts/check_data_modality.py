#!/usr/bin/env python3
"""Check whether the lesion and disconnectome volumes are binary or continuous,
and whether that matches what the pipeline assumes.

Design contract (see the Theory/Implementation specs):
  - lesions        -> BINARY masks {0, 1}      -> loaded with binarize=True,
                      trained with BCE + soft Dice.
  - disconnectomes -> CONTINUOUS maps in [0, 1] -> loaded with binarize=False,
                      trained with mean squared error (MSE).

The script samples files from each folder, reads the raw voxel values with
nibabel, classifies each volume, aggregates per modality, and prints a verdict.
It also runs the sanity checks a reviewer will ask for: out-of-range values,
NaN/Inf, empty (all-zero) volumes, shape/dtype consistency, and whether the two
modalities are paired one-to-one by the id in the filename
(lesion{id}_{age}_{sex}.nii.gz).

Exit code is 0 when both modalities match the contract, 1 otherwise, so it can
gate CI or a preprocessing step.

Usage:
    python scripts/check_data_modality.py                      # defaults: the 'full' tier (data/Full data)
    python scripts/check_data_modality.py --tier trial         # the pilot tier (data/Trial data)
    python scripts/check_data_modality.py --sample 0           # scan ALL files (slower)
    python scripts/check_data_modality.py \\
        --lesions "data/Full data/lesions" \\
        --disconnectomes "data/Full data/disconnectomes" --sample 300
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

try:
    import numpy as np
    import nibabel as nib
except ImportError as exc:  # pragma: no cover
    sys.exit(f"missing dependency: {exc}. Install with: pip install nibabel numpy")

TOL = 1e-6
ID_RE = re.compile(r"(lesion\d+)", re.IGNORECASE)


def list_volumes(root):
    paths = []
    for pattern in ("*.nii", "*.nii.gz"):
        paths.extend(glob.glob(os.path.join(root, pattern)))
    return sorted(paths)


def file_id(path):
    m = ID_RE.search(os.path.basename(path))
    return m.group(1).lower() if m else os.path.basename(path)


def classify(path):
    """Return a dict of stats + a class label for one volume."""
    img = nib.load(path)
    stored_dtype = str(img.get_data_dtype())
    v = np.asarray(img.get_fdata(), dtype=np.float64)
    n_nan = int(np.isnan(v).sum())
    n_inf = int(np.isinf(v).sum())
    finite = v[np.isfinite(v)]
    vmin = float(finite.min()) if finite.size else float("nan")
    vmax = float(finite.max()) if finite.size else float("nan")
    uniq = np.unique(finite)
    n_unique = int(uniq.size)
    fg = float((finite > TOL).mean()) if finite.size else 0.0

    is_binary = n_unique <= 2 and set(np.round(uniq, 6)).issubset({0.0, 1.0})
    in_unit = (vmin >= -TOL) and (vmax <= 1.0 + TOL)
    if n_unique <= 1:
        label = "empty/constant"
    elif is_binary:
        label = "binary"
    elif in_unit:
        label = "continuous[0,1]"
    else:
        label = "continuous(other-range)"

    return dict(dtype=stored_dtype, shape=tuple(int(s) for s in v.shape),
                vmin=vmin, vmax=vmax, n_unique=n_unique, fg=fg,
                n_nan=n_nan, n_inf=n_inf, label=label)


def scan(root, sample):
    paths = list_volumes(root)
    total = len(paths)
    if sample and sample > 0 and sample < total:
        # evenly spaced sample so we don't just look at the first ids
        idx = np.linspace(0, total - 1, sample).round().astype(int)
        paths = [paths[i] for i in sorted(set(idx.tolist()))]
    stats = [dict(path=p, **classify(p)) for p in paths]
    return total, stats


def summarize(name, total, stats):
    from collections import Counter
    labels = Counter(s["label"] for s in stats)
    shapes = Counter(s["shape"] for s in stats)
    dtypes = Counter(s["dtype"] for s in stats)
    vmin = min(s["vmin"] for s in stats)
    vmax = max(s["vmax"] for s in stats)
    uniq_lo = min(s["n_unique"] for s in stats)
    uniq_hi = max(s["n_unique"] for s in stats)
    fg = np.array([s["fg"] for s in stats]) * 100
    n_nan = sum(s["n_nan"] for s in stats)
    n_inf = sum(s["n_inf"] for s in stats)

    print(f"\n=== {name} ===")
    print(f"  files: {total} total, {len(stats)} inspected")
    print(f"  classes: " + ", ".join(f"{k}={v}" for k, v in labels.most_common()))
    print(f"  value range across files: [{vmin:.4f}, {vmax:.4f}]")
    print(f"  unique values per file: {uniq_lo} to {uniq_hi}")
    print(f"  foreground (>0): {fg.min():.2f}% to {fg.max():.2f}% (median {np.median(fg):.2f}%)")
    print(f"  dtypes: " + ", ".join(f"{k}({v})" for k, v in dtypes.items()))
    print(f"  shapes: " + ", ".join(f"{k}({v})" for k, v in shapes.items()))
    if n_nan or n_inf:
        print(f"  !! NaN voxels: {n_nan}, Inf voxels: {n_inf}")
    return labels, shapes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="full", choices=["trial", "full"],
                    help="cohort tier used for the default roots")
    ap.add_argument("--lesions", default=None,
                    help="explicit lesion root (overrides --tier)")
    ap.add_argument("--disconnectomes", default=None,
                    help="explicit disconnectome root (overrides --tier)")
    ap.add_argument("--sample", type=int, default=100,
                    help="files per modality to inspect (0 = all). Default 100.")
    args = ap.parse_args()
    if args.lesions is None or args.disconnectomes is None:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from neurocausalpfn.data.paths import disconnectome_root, lesion_root
        if args.lesions is None:
            args.lesions = lesion_root(args.tier)
        if args.disconnectomes is None:
            args.disconnectomes = disconnectome_root(args.tier)

    problems = []

    for root in (args.lesions, args.disconnectomes):
        if not os.path.isdir(root):
            sys.exit(f"directory not found: {root}")

    les_total, les = scan(args.lesions, args.sample)
    dis_total, dis = scan(args.disconnectomes, args.sample)
    if not les or not dis:
        sys.exit("no .nii/.nii.gz files found in one of the directories")

    les_labels, les_shapes = summarize("LESIONS", les_total, les)
    dis_labels, dis_shapes = summarize("DISCONNECTOMES", dis_total, dis)

    # ---- verdict against the pipeline contract ----
    print("\n=== VERDICT (against the pipeline contract) ===")

    non_binary_les = [s for s in les if s["label"] not in ("binary", "empty/constant")]
    if non_binary_les:
        problems.append("some LESION files are not binary")
        print(f"  LESIONS: MISMATCH — {len(non_binary_les)} of {len(les)} are not binary "
              f"(e.g. {os.path.basename(non_binary_les[0]['path'])}: "
              f"{non_binary_les[0]['n_unique']} unique values). BCE+Dice assumes {{0,1}}.")
    else:
        print("  LESIONS: OK — binary {0,1}. Correct for binarize=True + BCE + soft Dice.")

    out_of_range = [s for s in dis if s["vmax"] > 1.0 + TOL or s["vmin"] < -TOL]
    binary_dis = [s for s in dis if s["label"] == "binary"]
    if out_of_range:
        problems.append("some DISCONNECTOME files fall outside [0,1]")
        s = out_of_range[0]
        print(f"  DISCONNECTOMES: MISMATCH — {len(out_of_range)} of {len(dis)} fall outside [0,1] "
              f"(e.g. {os.path.basename(s['path'])}: [{s['vmin']:.3f}, {s['vmax']:.3f}]). "
              f"MSE on the probability assumes [0,1] — rescale, or revisit the loss.")
    elif binary_dis:
        problems.append("some DISCONNECTOME files look binary")
        print(f"  DISCONNECTOMES: WARNING — {len(binary_dis)} of {len(dis)} look binary, not graded. "
              f"MSE on a binary map is degenerate; check these are the continuous ChaCo-style maps.")
    else:
        print("  DISCONNECTOMES: OK — continuous in [0,1]. Correct for binarize=False + MSE.")

    # ---- pairing + shape consistency ----
    les_ids = {file_id(s["path"]) for s in scan(args.lesions, 0)[1]}
    dis_ids = {file_id(s["path"]) for s in scan(args.disconnectomes, 0)[1]}
    only_les, only_dis = les_ids - dis_ids, dis_ids - les_ids
    print("\n=== PAIRING (by id in filename) ===")
    print(f"  lesion ids: {len(les_ids)}, disconnectome ids: {len(dis_ids)}, paired: {len(les_ids & dis_ids)}")
    if only_les or only_dis:
        problems.append("lesion/disconnectome ids are not a 1:1 match")
        print(f"  !! only in lesions: {len(only_les)}; only in disconnectomes: {len(only_dis)} "
              f"(fusion mode 'both' needs a per-patient pair)")
    else:
        print("  ids match 1:1 — ready for per-patient fusion.")

    all_shapes = set(les_shapes) | set(dis_shapes)
    if len(all_shapes) > 1:
        print(f"  note: mixed volume shapes {sorted(all_shapes)} — the loader pads/crops to a common grid.")

    print("\n" + ("ALL CHECKS PASSED — data matches the pipeline contract."
                  if not problems else
                  "ISSUES FOUND: " + "; ".join(problems)))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
