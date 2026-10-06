#!/usr/bin/env python3
"""Build the cohort CSV that clinical_validation.py consumes from a participants
table and a lesion directory.

The table may be comma- or tab-separated whatever its extension. Rows are
matched to the lesion files by id prefix (``<id>_...nii.gz``), trying each of
``--id-cols`` in turn and keeping the column that matches most files. The
total of a 15-item severity scale is derived from its item columns (NaN when
any item is missing), two-level sex codes are kept as given, and the lesion
volume in ml is computed from each mask on its own grid. Only aggregate
counts are printed; no row is echoed.

    python scripts/prepare_clinical_cohort.py --table participants.tsv \
        --lesions-dir ~/Scratch/kch_lesions --id-cols participant_id id \
        --age-col S1AgeOnArrival --sex-col S1Gender --items-prefix S2NihssArrival \
        --out outputs/kch_cohort.csv
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    with open(path) as f:
        head = f.readline()
    sep = "\t" if head.count("\t") >= head.count(",") else ","
    return pd.read_csv(path, sep=sep, dtype=str)


def match_column(df: pd.DataFrame, id_cols, basenames):
    """(column, row index per file or -1) for the id column that matches most files."""
    best = (None, None, -1)
    for col in id_cols:
        if col not in df.columns:
            continue
        ids = df[col].astype(str).str.strip()
        pos = {}
        for k, i in enumerate(ids):
            if i and i.lower() != "nan":
                pos.setdefault(i, k)
        # longest id first so that "sub-10" does not capture "sub-100_..."
        keys = sorted(pos, key=len, reverse=True)
        rows = []
        for b in basenames:
            hit = next((i for i in keys if b.startswith(i)), None)
            rows.append(pos[hit] if hit is not None else -1)
        n = sum(r >= 0 for r in rows)
        if n > best[2]:
            best = (col, np.asarray(rows), n)
    return best


def item_total(df: pd.DataFrame, prefix: str, n_expected: int = 15) -> pd.Series:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if len(cols) != n_expected:
        print(f"warning: {len(cols)} item columns with prefix {prefix!r} (expected {n_expected})")
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    total = vals.sum(axis=1, min_count=len(cols))
    return total.where(vals.notna().all(axis=1))


def lesion_volume_ml(path: str) -> float:
    import nibabel as nib
    img = nib.load(path)
    vox_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    return float((np.asarray(img.dataobj) > 0.5).sum()) * vox_ml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--lesions-dir", required=True)
    ap.add_argument("--id-cols", nargs="+", default=["participant_id", "id"])
    ap.add_argument("--age-col", default=None)
    ap.add_argument("--sex-col", default=None)
    ap.add_argument("--items-prefix", default=None, help="prefix of the scale's item columns (total derived)")
    ap.add_argument("--total-name", default="nihss_total")
    ap.add_argument("--keep", nargs="*", default=[], help="further columns to carry over as is")
    ap.add_argument("--no-volume", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = read_table(args.table)
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.lesions_dir), "*.nii*")))
    if not files:
        sys.exit(f"no niftis in {args.lesions_dir}")
    base = [os.path.basename(f) for f in files]
    col, rows, n = match_column(df, args.id_cols, base)
    print(f"table: {len(df)} rows; lesions: {len(files)}; matched {n} files by column {col!r}")
    if n == 0:
        sys.exit("no file matches any id; check --id-cols or the file names")
    dup = pd.Series(rows[rows >= 0]).duplicated().sum()
    if dup:
        print(f"warning: {dup} files share a table row (several images per participant)")

    out = pd.DataFrame({"sub": [df[col].iloc[r] if r >= 0 else "" for r in rows], "file": base})
    if args.age_col:
        out["Age"] = pd.to_numeric(df[args.age_col], errors="coerce").to_numpy()[rows]
        out.loc[rows < 0, "Age"] = np.nan
    if args.sex_col:
        out["Sex"] = df[args.sex_col].to_numpy()[rows]
        out.loc[rows < 0, "Sex"] = np.nan
    if args.items_prefix:
        tot = item_total(df, args.items_prefix).to_numpy()
        out[args.total_name] = tot[rows]
        out.loc[rows < 0, args.total_name] = np.nan
    for c in args.keep:
        out[c] = df[c].to_numpy()[rows]
    if not args.no_volume:
        out["vol_mni_ml"] = [lesion_volume_ml(f) for f in files]
    out = out[rows >= 0].reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} rows")
    for c in out.columns:
        if c in ("sub", "file"):
            continue
        v = pd.to_numeric(out[c], errors="coerce")
        if v.notna().sum():
            print(f"  {c:14s} n={int(v.notna().sum()):5d}  mean={v.mean():.2f}  sd={v.std():.2f}  "
                  f"min={v.min():.1f}  max={v.max():.1f}")
        else:
            print(f"  {c:14s} levels={out[c].dropna().nunique()}  n={int(out[c].notna().sum())}")


if __name__ == "__main__":
    main()
