"""Lesion sizes of prior pools (28 Oct): are the pool's lesions smaller than
the trial's? Reads the ``volumes`` (voxels of each lesion mask) of one or more
prior caches and prints, per cache, the count, the 10/25/50/75/90 % quantiles
and the mean. Aggregates only; no names.

A cache built from a fold's UCLH training side (scripts/build_fold_pool_cache.py)
has the size distribution of that fold's test patients, since the folds are
random over the same cohort; a KCH cache is the KCH cohort.

    python scripts/audit_20261028/lesion_sizes.py POOL.npz [POOL2.npz ...] [--out sizes.csv]
"""
import argparse
import csv
import os

import numpy as np

QS = (10, 25, 50, 75, 90)


def size_row(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        if "volumes" not in z.files:
            raise SystemExit(f"{os.path.basename(path)}: no volumes in this cache")
        v = np.asarray(z["volumes"], dtype=float)
    q = np.percentile(v, QS)
    return {"cache": os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path), "n": int(len(v)),
            **{f"q{p}": float(x) for p, x in zip(QS, q)}, "mean": float(v.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [size_row(p) for p in a.caches]
    for r in rows:
        print(f"{r['cache']}: n {r['n']}, voxels " + ", ".join(f"q{p} {r[f'q{p}']:.0f}" for p in QS)
              + f", mean {r['mean']:.0f}")
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
