"""Compare the receptomic and transcriptomic splits of the Giles atlas.

Each of the 16 functional networks is cut in two subnetworks (labels 1 and 2)
twice over: once by receptor maps (2mm_parcellations/receptor) and once by
gene expression (2mm_parcellations/genetics). The virtual trial makes one
trial per network per split, so the two splits give two different trials.
This reports, per network and split:

- laterality of each half: the fraction of its voxels on the left of the
  mid-sagittal plane. A split that cuts a network into left and right
  hemisphere shows one half near 1 and the other near 0.
- the volume balance of the two halves and which one maps to treatment 1
  (the larger, as in the paper's code);
- the agreement between the two splits: the Dice of receptor half 1 against
  each genetics half, taking the better of the two pairings, so 1 means the
  two splits cut the network the same way and 0.5 or less that they are
  unrelated cuts.

Aggregates over the atlas only; no patient data is read.

    python scripts/compare_atlas_splits.py --atlas-dir <atlases> --out outputs/atlas_splits.csv
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _halves(path: str):
    import nibabel as nib

    img = nib.load(path)
    vol = np.asarray(img.get_fdata())
    return (vol == 1), (vol == 2), img.affine


def _laterality(mask: np.ndarray, affine: np.ndarray) -> float:
    """Fraction of the mask's voxels with x < 0 in world coordinates (left)."""
    idx = np.argwhere(mask)
    if not len(idx):
        return float("nan")
    xs = affine[0, 0] * idx[:, 0] + affine[0, 3]
    return float((xs < 0).mean())


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / s) if s else float("nan")


def _index(sub_dir: str):
    out = {}
    for f in sorted(glob.glob(os.path.join(sub_dir, "*.nii")) + glob.glob(os.path.join(sub_dir, "*.nii.gz"))):
        stem = os.path.basename(f)
        for ext in (".nii.gz", ".nii"):
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
                break
        try:
            out[int(stem.split("_")[0])] = f
        except ValueError:
            continue
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import pandas as pd

    base = os.path.join(args.atlas_dir, "2mm_parcellations")
    rec, gen = _index(os.path.join(base, "receptor")), _index(os.path.join(base, "genetics"))
    if not rec:
        raise SystemExit(f"no receptor parcellations under {base}")
    if not gen:
        print(f"warning: no genetics parcellations under {base}; reporting the receptor split only")
    rows = []
    for k in sorted(rec):
        r1, r2, aff = _halves(rec[k])
        row = {"network": k,
               "receptor_vox_1": int(r1.sum()), "receptor_vox_2": int(r2.sum()),
               "receptor_left_frac_1": _laterality(r1, aff), "receptor_left_frac_2": _laterality(r2, aff)}
        row["receptor_treatment1_is_half"] = 1 if r1.sum() >= r2.sum() else 2
        if k in gen:
            g1, g2, affg = _halves(gen[k])
            row.update({"genetics_vox_1": int(g1.sum()), "genetics_vox_2": int(g2.sum()),
                        "genetics_left_frac_1": _laterality(g1, affg), "genetics_left_frac_2": _laterality(g2, affg),
                        "genetics_treatment1_is_half": 1 if g1.sum() >= g2.sum() else 2})
            row["dice_best_pairing"] = max((_dice(r1, g1) + _dice(r2, g2)) / 2.0,
                                           (_dice(r1, g2) + _dice(r2, g1)) / 2.0)
        rows.append(row)
    df = pd.DataFrame(rows)

    def _lat_span(a, b):
        """How hemispheric a split is: 1 when one half is all left and the other all right."""
        return (a - b).abs()

    print(f"{len(df)} networks from {base}")
    cols = ["network", "receptor_left_frac_1", "receptor_left_frac_2"]
    if "genetics_left_frac_1" in df:
        cols += ["genetics_left_frac_1", "genetics_left_frac_2", "dice_best_pairing"]
    print(df[cols].round(3).to_string(index=False))
    print("\nlaterality separation (|left_frac_1 - left_frac_2|; 1 = a left/right cut):")
    rec_sep = _lat_span(df["receptor_left_frac_1"], df["receptor_left_frac_2"])
    print(f"  receptor: mean {rec_sep.mean():.3f}, median {rec_sep.median():.3f}, "
          f"networks above 0.8: {int((rec_sep > 0.8).sum())} of {len(df)}")
    if "genetics_left_frac_1" in df:
        gen_sep = _lat_span(df["genetics_left_frac_1"], df["genetics_left_frac_2"])
        print(f"  genetics: mean {gen_sep.mean():.3f}, median {gen_sep.median():.3f}, "
              f"networks above 0.8: {int((gen_sep > 0.8).sum())} of {len(df)}")
        print(f"  agreement between splits (Dice, best pairing): mean {df['dice_best_pairing'].mean():.3f}, "
              f"min {df['dice_best_pairing'].min():.3f}, max {df['dice_best_pairing'].max():.3f}")
    if args.out:
        df.to_csv(args.out, index=False)
        print("wrote", args.out)
    return df


if __name__ == "__main__":
    main()
