"""Quality control of the ISLES'24 masks registered to MNI 2 mm, plus the
phenotype merge.

    python scripts/isles24_qc.py <isles root> <mni2mm folder> [--ref <training lesion>]
        [-o isles24_cohort.csv]

Per case: native and MNI lesion volume (ml), their ratio, the fraction of the
registered lesion inside the MNI brain mask, whether the grid equals the
reference (shape and affine), the acquisition centre read from the baseline
phenotype file, and the outcome columns (NIHSS at 24 h / discharge, mRS at
discharge / 3 months, TICI) when present. ``ok`` = registered, non-empty,
volume ratio divided by the affine's scale factor within [0.75, 1.35], >= 95 %
inside the brain, grid matches; ``fail_reason`` names the failed criteria.
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
                "mRS premorbid", "mRS at admission", "mRS 24h", "mRS discharge", "mRS 3 months",
                "TICI postinterventional", "Door to recanalization"]


def _mat_scale(path: str):
    """Volume scale factor of a FLIRT 4x4 matrix (|det| of its 3x3 part):
    an affine to MNI enlarges or shrinks every structure by this factor, so
    the MNI lesion volume is expected to be native volume x scale."""
    try:
        m = np.loadtxt(path)
        return float(abs(np.linalg.det(m[:3, :3])))
    except Exception:
        return None


def _vol_ml(img) -> float:
    data = np.asanyarray(img.dataobj) > 0
    return float(data.sum() * np.prod(img.header.get_zooms()[:3]) / 1000.0)


def _read_phenotype(root: str, sub: str) -> dict:
    """Merge every phenotype table of the case found anywhere under phenotype/
    (the release ships one .xlsx per case and session; .csv is accepted too).
    First row of each table; header cells are stripped."""
    out = {}
    paths = sorted(glob.glob(os.path.join(root, "phenotype", "**", f"{sub}*.xlsx"), recursive=True)
                   + glob.glob(os.path.join(root, "phenotype", "**", f"{sub}*.csv"), recursive=True))
    for p in paths:
        if p.lower().endswith(".xlsx"):
            import pandas as pd
            df = pd.read_excel(p)
            if len(df):
                out.update({str(k).strip(): ("" if pd.isna(v) else v) for k, v in df.iloc[0].items()})
            continue
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
            scale = _mat_scale(os.path.join(args.mni_dir, f"{sub}_dwi2mni.mat"))
            row["scale"] = round(scale, 3) if scale else ""
            row["ratio_adj"] = (round(row["ratio"] / scale, 3)
                                if scale and row["ratio"] != "" else row["ratio"])
            row["frac_in_brain"] = (round(float((data & brain).sum() / max(data.sum(), 1)), 3)
                                    if brain is not None else "")
            row["grid_ok"] = int(img.shape == (91, 109, 91)
                                 and (ref is None or np.allclose(img.affine, ref.affine, atol=1e-3)))
            row["empty"] = int(data.sum() == 0)
        else:
            row.update({"vol_mni_ml": "", "ratio": "", "scale": "", "ratio_adj": "",
                        "frac_in_brain": "", "grid_ok": 0, "empty": ""})
        ph = _read_phenotype(args.root, sub)
        for c in OUTCOME_COLS:
            row[c] = ph.get(c, "")
        reasons = []
        if not os.path.exists(mni):
            reasons.append("not_registered")
        else:
            if row["empty"] == 1:
                reasons.append("empty")
            if row["grid_ok"] != 1:
                reasons.append("grid")
            if row["ratio_adj"] == "" or not (0.75 <= float(row["ratio_adj"]) <= 1.35):
                reasons.append("volume")
            if row["frac_in_brain"] != "" and float(row["frac_in_brain"]) < 0.95:
                reasons.append("outside_brain")
        row["fail_reason"] = ";".join(reasons)
        row["ok"] = int(not reasons)
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
        print(f"volume ratio MNI/native (raw): median {np.median(ratios):.3f}, "
              f"5-95 % [{np.percentile(ratios, 5):.3f}, {np.percentile(ratios, 95):.3f}]")
    adj = [float(r["ratio_adj"]) for r in rows if r["ratio_adj"] != "" and r["scale"] != ""]
    if adj:
        print(f"volume ratio / affine scale: median {np.median(adj):.3f}, "
              f"5-95 % [{np.percentile(adj, 5):.3f}, {np.percentile(adj, 95):.3f}]")
    import collections
    fails = collections.Counter(x for r in rows for x in r["fail_reason"].split(";") if x)
    print("failures by criterion:", dict(fails) if fails else "none")
    for c in ("Center", "NIHSS 24h", "mRS 3 months", "TICI postinterventional"):
        have = sum(1 for r in rows if str(r[c]).strip() != "")
        print(f"{c}: present in {have}/{n}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
