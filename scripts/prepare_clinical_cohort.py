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


def stem(basename: str) -> str:
    """File name without extensions: sub-K123.nii.gz -> sub-K123."""
    return basename.split(".")[0]


def match_column(df: pd.DataFrame, id_cols, basenames, strip_prefix: str = "", exact: bool = False):
    """(column, row index per file or -1) for the id column that matches most
    files. ``strip_prefix`` is removed from the file names first (``sub-``);
    ``exact`` requires the file stem to equal the id, otherwise the longest id
    that prefixes the name wins."""
    best = (None, None, -1)
    names = [b[len(strip_prefix):] if strip_prefix and b.startswith(strip_prefix) else b for b in basenames]
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
        for b in names:
            if exact:
                hit = stem(b) if stem(b) in pos else None
            else:
                hit = next((i for i in keys if b.startswith(i)), None)
            rows.append(pos[hit] if hit is not None else -1)
        n = sum(r >= 0 for r in rows)
        if n > best[2]:
            best = (col, np.asarray(rows), n)
    return best


def mask_digits(s: str) -> str:
    """Shape of a string with every digit replaced by '#', letters kept: safe to print."""
    import re
    return re.sub(r"\d", "#", s)


def diagnose(df: pd.DataFrame, id_cols, basenames, n_show: int = 5, strip_prefix: str = ""):
    """Print how the table's ids and the file names look and how they match,
    without echoing any id: digit-masked shapes, counts, ambiguities."""
    if strip_prefix:
        basenames = [b[len(strip_prefix):] if b.startswith(strip_prefix) else b for b in basenames]
    print(f"files: {len(basenames)}; name shapes (digits masked):")
    for shp, n in pd.Series([mask_digits(b) for b in basenames]).value_counts().head(n_show).items():
        print(f"   {n:5d}  {shp}")
    for col in id_cols:
        if col not in df.columns:
            print(f"column {col!r}: absent"); continue
        ids = df[col].astype(str).str.strip()
        ids = ids[(ids != "") & (ids.str.lower() != "nan")]
        print(f"column {col!r}: {len(ids)} non-empty, {ids.nunique()} unique, "
              f"{int(ids.duplicated().sum())} duplicated; id shapes:")
        for shp, n in ids.map(mask_digits).value_counts().head(n_show).items():
            print(f"   {n:5d}  {shp}")
        uniq = sorted(set(ids), key=len, reverse=True)
        n_match, n_ambig = 0, 0
        for b in basenames:
            hits = [i for i in uniq if b.startswith(i)]
            if hits:
                n_match += 1
                # ambiguous when a shorter id is a strict prefix of the longest hit's match
                if len(hits) > 1 and not all(hits[0].startswith(h) for h in hits):
                    n_ambig += 1
        rows_hit = ids[ids.isin({i for i in uniq if any(b.startswith(i) for b in basenames)})]
        print(f"   files matched: {n_match}/{len(basenames)}; ambiguous files: {n_ambig}; "
              f"table rows matched: {len(rows_hit)} (of which duplicated ids: {int(rows_hit.duplicated().sum())})")


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
    ap.add_argument("--strip-prefix", default="", help="prefix removed from file names before matching, e.g. sub-")
    ap.add_argument("--match", default="prefix", choices=["prefix", "exact"],
                    help="exact: file stem must equal the id (no ambiguity possible)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only print digit-masked shapes of ids and file names and the match counts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run")

    df = read_table(args.table)
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.lesions_dir), "*.nii*")))
    if not files:
        sys.exit(f"no niftis in {args.lesions_dir}")
    base = [os.path.basename(f) for f in files]
    if args.dry_run:
        diagnose(df, args.id_cols, base, strip_prefix=args.strip_prefix)
        return
    col, rows, n = match_column(df, args.id_cols, base, strip_prefix=args.strip_prefix,
                                exact=args.match == "exact")
    print(f"table: {len(df)} rows; lesions: {len(files)}; matched {n} files by column {col!r}")
    if n == 0:
        sys.exit("no file matches any id; check --id-cols or the file names")
    dup = pd.Series(rows[rows >= 0]).duplicated().sum()
    if dup:
        print(f"warning: {dup} files share a table row (several images per participant)")

    out = pd.DataFrame({"sub": [stem(b) for b in base], "table_id": [df[col].iloc[r] if r >= 0 else "" for r in rows],
                        "file": base})
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
        if c in ("sub", "file", "table_id"):
            continue
        v = pd.to_numeric(out[c], errors="coerce")
        if v.notna().sum():
            print(f"  {c:14s} n={int(v.notna().sum()):5d}  mean={v.mean():.2f}  sd={v.std():.2f}  "
                  f"min={v.min():.1f}  max={v.max():.1f}")
        else:
            print(f"  {c:14s} levels={out[c].dropna().nunique()}  n={int(out[c].notna().sum())}")


if __name__ == "__main__":
    main()
