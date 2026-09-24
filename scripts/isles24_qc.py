"""Quality control of the ISLES'24 masks registered to MNI 2 mm, plus the
phenotype merge.

    python scripts/isles24_qc.py <isles root> <mni2mm folder> [--ref <training lesion>]
        [-o isles24_cohort.csv]

Per case: native and MNI lesion volume (ml), their ratio, the fraction of the
registered lesion inside the MNI brain mask, whether the grid equals the
reference (shape and affine), the acquisition centre read from the baseline
phenotype file, and the outcome columns (NIHSS at 24 h / discharge, mRS at
discharge / 3 months, TICI) when present. ``ok`` = registered, non-empty,
volume ratio within [0.7, 1.4], >= 95 % inside the brain, grid matches.
Writes one CSV row per case; nothing identifying is in the release, and only
aggregate counts are printed.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import nibabel as nib

OUTCOME_COLS = ["Center", "Sex", "Age", "NIHSS at admission", "NIHSS 24h", "NIHSS discharge",
                "mRS premorbid", "mRS at admission", "mRS discharge", "mRS 3 months",
                "TICI postinterventional", "Door to recanalization"]


def _vol_ml(img) -> float:
    data = np.asanyarray(img.dataobj) > 0
    return float(data.sum() * np.prod(img.header.get_zooms()[:3]) / 1000.0)


def _read_phenotype(root: str, sub: str) -> dict:
    """Merge every CSV of the case found anywhere under phenotype/ (the release
    documentation and the actual folder names differ), first row of each."""
    out = {}
    paths = sorted(glob.glob(os.path.join(root, "phenotype", "**", f"{sub}*.csv"), recursive=True))
    for p in paths:
        with open(p, newline="") as f:
            head = f.read(4096)
            f.seek(0)
            sep = "\t" if head.count("\t") > head.count(",") else ","
            rows = list(csv.DictReader(f, delimiter=sep))
        if rows:
            out.update({k.strip(): v for k, v in rows[0].items() if k})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("mni_dir")
    ap.add_argument("--ref", default=None, help="one training lesion (grid reference)")
    ap.add_argument("--brain-mask", default=None,
                    help="MNI152 2 mm brain mask; default $FSLDIR/data/standard/MNI152_T1_2mm_brain_mask")
    ap.add_argument("-o", "--out", default="isles24_cohort.csv")
    args = ap.parse_args()

    bm_path = args.brain_mask or os.path.join(os.environ.get("FSLDIR", ""), "data", "standard",
                                              "MNI152_T1_2mm_brain_mask.nii.gz")
    brain = np.asanyarray(nib.load(bm_path).dataobj) > 0 if os.path.exists(bm_path) else None
    ref = nib.load(args.ref) if args.ref else None

    subs = sorted(os.path.basename(d.rstrip("/")) for d in glob.glob(os.path.join(args.root, "derivatives", "sub-*/")))
    rows = []
    for sub in subs:
        row = {"sub": sub}
        native = glob.glob(os.path.join(args.root, "derivatives", sub, "ses-02", "*_space-ncct_lesion-msk.nii.gz"))
        mni = os.path.join(args.mni_dir, f"{sub}_lesion-msk_mni2mm.nii.gz")
        row["vol_native_ml"] = round(_vol_ml(nib.load(native[0])), 2) if native else ""
        if os.path.exists(mni):
            img = nib.load(mni)
            data = np.asanyarray(img.dataobj) > 0
            row["vol_mni_ml"] = round(_vol_ml(img), 2)
            row["ratio"] = (round(row["vol_mni_ml"] / row["vol_native_ml"], 3)
                            if row["vol_native_ml"] not in ("", 0, 0.0) else "")
            row["frac_in_brain"] = (round(float((data & brain).sum() / max(data.sum(), 1)), 3)
                                    if brain is not None else "")
            row["grid_ok"] = int(img.shape == (91, 109, 91)
                                 and (ref is None or np.allclose(img.affine, ref.affine, atol=1e-3)))
            row["empty"] = int(data.sum() == 0)
        else:
            row.update({"vol_mni_ml": "", "ratio": "", "frac_in_brain": "", "grid_ok": 0, "empty": ""})
        ph = _read_phenotype(args.root, sub)
        for c in OUTCOME_COLS:
            row[c] = ph.get(c, "")
        ok = (os.path.exists(mni) and row["empty"] == 0 and row["grid_ok"] == 1
              and row["ratio"] != "" and 0.7 <= float(row["ratio"]) <= 1.4
              and (row["frac_in_brain"] == "" or float(row["frac_in_brain"]) >= 0.95))
        row["ok"] = int(ok)
        rows.append(row)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    reg = sum(1 for r in rows if r["vol_mni_ml"] != "")
    print(f"cases {n} | registered {reg} | empty {sum(1 for r in rows if r['empty'] == 1)} | "
          f"grid ok {sum(r['grid_ok'] for r in rows)} | ok {sum(r['ok'] for r in rows)}")
    ratios = [float(r["ratio"]) for r in rows if r["ratio"] != ""]
    if ratios:
        print(f"volume ratio MNI/native: median {np.median(ratios):.3f}, "
              f"5-95 % [{np.percentile(ratios, 5):.3f}, {np.percentile(ratios, 95):.3f}]")
    for c in ("Center", "NIHSS 24h", "mRS 3 months", "TICI postinterventional"):
        have = sum(1 for r in rows if str(r[c]).strip() != "")
        print(f"{c}: present in {have}/{n}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
