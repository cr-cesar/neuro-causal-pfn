"""Manifest of an external mask cohort: one row per image with site, the
curation label of its participant, whether the mask is empty, and an include
flag (trial).

    python scripts/combined_manifest.py <signatures.csv> <participants table> \\
        --label-col manual_check --keep keep,keep_artifact,keep_false_positive,\\
keep_missing_spots,keep_under_segmented,keep_horrible_quality,Use \\
        -o combined_manifest.csv

signatures.csv comes from lesion_signatures.py (filename, n_voxels, ...). The
participant key is the filename stem without the ``sub-`` prefix. Only the
curation label travels into the manifest; no clinical column is copied.
Prints the cross-tabulation label x empty and the include counts per site.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re


def stem(name: str) -> str:
    base = os.path.basename(name)
    for ext in (".nii.gz", ".nii"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
    return base[4:] if base.startswith("sub-") else base


def site_of(key: str) -> str:
    m = re.match(r"\s*([A-Za-z]+)", key)
    return m.group(1).lower() if m else ""


def build(sig_rows, part_rows, pkey_col, label_col, keep):
    label = {str(r[pkey_col]).strip(): str(r.get(label_col, "")).strip() for r in part_rows}
    out = []
    for r in sig_rows:
        key = stem(r["filename"])
        lab = label.get(key, "")
        empty = int(int(float(r["n_voxels"])) == 0)
        out.append({"filename": os.path.basename(r["filename"]), "participant": key,
                    "site": site_of(key), "label": lab, "empty": empty,
                    "n_voxels": int(float(r["n_voxels"])),
                    "include": int(not empty and lab in keep)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("signatures")
    ap.add_argument("participants")
    ap.add_argument("--pkey-col", default="participant_id")
    ap.add_argument("--label-col", default="manual_check")
    ap.add_argument("--keep", default="keep,keep_artifact,keep_false_positive,keep_missing_spots,"
                                      "keep_under_segmented,keep_horrible_quality,Use")
    ap.add_argument("-o", "--out", default="combined_manifest.csv")
    args = ap.parse_args()

    with open(args.signatures, newline="") as f:
        sig = list(csv.DictReader(f))
    with open(args.participants, newline="") as f:
        head = f.read(8192)
        f.seek(0)
        sep = "\t" if head.count("\t") > head.count(",") else ","
        parts = list(csv.DictReader(f, delimiter=sep))
    keep = {k.strip() for k in args.keep.split(",") if k.strip()}
    rows = build(sig, parts, args.pkey_col, args.label_col, keep)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    unmatched = sum(1 for r in rows if r["label"] == "")
    print(f"images {len(rows)} | without participant row {unmatched}")
    tab = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        tab[r["label"] or "<none>"][r["empty"]] += 1
    print(f"{'label':40s} {'non-empty':>9s} {'empty':>6s}  keep?")
    for lab, (ne, e) in sorted(tab.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        print(f"{lab:40s} {ne:9d} {e:6d}  {'yes' if lab in keep else ''}")
    for s in sorted({r["site"] for r in rows}):
        sub = [r for r in rows if r["site"] == s]
        print(f"site {s}: images {len(sub)} | empty {sum(r['empty'] for r in sub)} | "
              f"include {sum(r['include'] for r in sub)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
