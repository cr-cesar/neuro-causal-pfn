"""Holm correction over a family of paired tests (29 Oct).

Takes the summary CSVs written by scripts/paired_deficit_test.py (one row
each, or several), stacks them, and adds the Holm step-down adjusted p of the
paired t-test and of the Wilcoxon test over the whole family. The tables of
Table 17 and of plan B were corrected this way with unversioned code; this is
the versioned version, checked against those files.

    python scripts/holm_adjust.py --inputs b3_*_summary.csv --out b3_family_holm.csv
"""
import argparse
import glob
import os
import sys
from typing import List

import numpy as np
import pandas as pd


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p values, in the input order."""
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    order = np.argsort(q, kind="stable")
    m, running, adj = len(q), 0.0, np.empty(len(q))
    for k, i in enumerate(order):
        running = max(running, (m - k) * q[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def main(argv: List[str] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="summary CSV files or globs (per_deficit files are skipped)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    files = sorted({f for pat in a.inputs for f in glob.glob(pat) if "per_deficit" not in os.path.basename(f)})
    if not files:
        sys.exit("no input files")
    d = pd.concat([pd.read_csv(f, comment="#").assign(source=os.path.basename(f)) for f in files], ignore_index=True)
    for raw, name in (("paired_t_p", "p_holm_t"), ("wilcoxon_p", "p_holm_wilcoxon")):
        if raw in d:
            d[name] = holm(d[raw])
    d["holm_family_size"] = len(d)
    d.to_csv(a.out, index=False)
    print(f"{len(d)} tests from {len(files)} files -> {a.out}")


if __name__ == "__main__":
    main()
