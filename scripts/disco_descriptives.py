# Provided by the PI for audit finding F33 (dashboard code block); kept unchanged.
"""Aggregate descriptive statistics for a cohort of lesion or disconnectome maps.

Applies Giles et al. (2025) enrolment rule exactly as in the released code
(software/deficit_modelling.py): binarise the map at 0.5; a lesion responds to
a subnetwork when the overlapping voxel count exceeds round(0.05 * subnetwork
voxels); within each network, the larger subnetwork is treatment A (W1).

Writes ONLY aggregate outputs (no filenames, no per-patient rows):
  summary.json            n, volume quantiles and histogram, trials-per-lesion counts
  per_network.csv         per network and split: neither / A only / B only / both
  freq_map.nii.gz         number of maps covering each voxel (group sum)

Usage (CPU, a few minutes per 4,000 maps):
  python disco_descriptives.py --maps "/path/to/disconnectomes/*.nii.gz" \
      --atlas-dir /path/to/individualized_prescriptive_inference/atlases \
      --out status/descriptives_uclh_disco --threshold 0.5
"""
import argparse, glob, json, os
import numpy as np
import nibabel as nib

NAMES = ["hearing", "language", "introspection", "cognition", "mood", "memory", "aversion",
         "coordination", "interoception", "sleep", "reward", "visual recognition",
         "visual perception", "spatial reasoning", "motor", "somatosensory"]


def lookup(atlas_dir, kind, shape):
    """Voxel -> subnetwork id (2k = A, 2k+1 = B), -1 outside; plus subnetwork sizes."""
    lut = np.full(int(np.prod(shape)), -1, int)
    sizes = np.zeros(32, int)
    for k in range(1, 17):
        f = [x for x in glob.glob(os.path.join(atlas_dir, "2mm_parcellations", kind, "*"))
             if os.path.basename(x).startswith(f"{k}_")][0]
        m = nib.load(f).get_fdata().astype(int).ravel()
        v1, v2 = (m == 1).sum(), (m == 2).sum()
        a, b = (1, 2) if v1 >= v2 else (2, 1)
        lut[m == a], lut[m == b] = 2 * (k - 1), 2 * (k - 1) + 1
        sizes[2 * (k - 1)], sizes[2 * (k - 1) + 1] = (m == a).sum(), (m == b).sum()
    return lut, np.round(0.05 * sizes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True, help="glob of NIfTI maps in MNI 2 mm (91x109x91)")
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    files = sorted(glob.glob(a.maps))
    if not files:
        raise SystemExit("no maps matched --maps")
    shape = nib.load(files[0]).shape[:3]
    luts = {kind: lookup(a.atlas_dir, kind, shape) for kind in ("receptor", "genetics")}
    n = len(files)
    vol = np.zeros(n)
    cats = {kind: np.zeros((n, 16), int) for kind in luts}
    freq = np.zeros(int(np.prod(shape)), np.float32)
    for i, f in enumerate(files):
        idx = np.flatnonzero(np.asarray(nib.load(f).dataobj).ravel() > a.threshold)
        vol[i] = len(idx) * 8 / 1000.0
        freq[idx] += 1
        for kind, (lut, thr) in luts.items():
            lab = lut[idx]
            hit = np.bincount(lab[lab >= 0], minlength=32) > thr
            cats[kind][i] = hit[0::2] * 1 + hit[1::2] * 2
    os.makedirs(a.out, exist_ok=True)
    bins = np.logspace(-2, 3, 41)
    summary = {"n_maps": n, "threshold": a.threshold,
               "volume_ml_quantiles": {q: float(np.percentile(vol, q)) for q in (5, 25, 50, 75, 95)},
               "volume_ml_mean": float(vol.mean()), "volume_ml_max": float(vol.max()),
               "volume_hist_bins_ml": bins.tolist(), "volume_hist_counts": np.histogram(vol, bins)[0].tolist()}
    rows = ["split,network,neither,A_only,B_only,both,enrolled,enrolled_pct"]
    for kind, c in cats.items():
        e = (c > 0).sum(1)
        summary[f"trials_per_map_{kind}"] = {str(t): int((e == t).sum()) for t in range(int(e.max()) + 1)}
        summary[f"median_volume_ml_no_trial_{kind}"] = float(np.median(vol[e == 0])) if (e == 0).any() else None
        summary[f"median_volume_ml_in_trial_{kind}"] = float(np.median(vol[e > 0])) if (e > 0).any() else None
        for k in range(16):
            cc = np.bincount(c[:, k], minlength=4)
            rows.append(f"{kind},{NAMES[k]},{cc[0]},{cc[1]},{cc[2]},{cc[3]},{cc[1:].sum()},{100 * cc[1:].sum() / n:.1f}")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    open(os.path.join(a.out, "per_network.csv"), "w").write("\n".join(rows) + "\n")
    ref = nib.load(files[0])
    nib.save(nib.Nifti1Image(freq.reshape(shape), ref.affine), os.path.join(a.out, "freq_map.nii.gz"))
    print(f"wrote aggregates for {n} maps to {a.out}")


if __name__ == "__main__":
    main()
