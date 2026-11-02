"""Does the representation keep how much of each network half a map covers? (2 Nov)

The ideal-cell effect of a patient is fixed by whether the binarised
disconnectome covers more than 5 % of each half of a network (giles_replica:
ROI_THRESH). A probe on the E1 latents separates the response regions (AUC
0.96 to 0.98), yet a learner fitted to the true effect from the same latents
stays far above one given the anatomy. This tests the explanation that the
latents keep where the map lies but not how much of each half it covers, on
the benchmark folds (single-acquisition test patients, each fold's own
encoder):

A. probe: per split, network and half, the coverage fraction predicted from
   the latents (ridge and gradient boosting fitted on the fold's training
   side), pooled R^2 over the test folds, and how often the predicted
   coverage puts the patient on the right side of the 5 % threshold, near the
   threshold (within half of it) and away from it;
B. errors: per trial, gradient boosting fitted to the true effect from the
   latents (the oracle setting of the latent reference, with a weaker
   learner); the squared error of the test patients by their distance to the
   threshold, as shares of the patients and of the error.

Aggregates only: no file name, no row of a patient is written or printed.

    PYTHONPATH=src:<perfold> python scripts/audit_20261028/coverage_diagnostic.py \\
        --reps <outputs_perfold_ws>/E1/E1/seed0/folds --images-dir "<data>/Full data/disconnectomes" \\
        --atlas-dir <export_bcb>/atlases --groups <export_bcb>/fold_table.csv --out <dir>
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

from neurocausalpfn.prior import giles_replica as gr

SPLITS = ("receptor", "genetics")
BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, np.inf)]   # |cov - t| / t

_MASKS = None


def _init(masks):
    global _MASKS
    _MASKS = masks


def _coverage(path):
    """Coverage fraction of every half (split, network, W1/W0) for one map."""
    import nibabel as nib

    flat = (np.asarray(nib.load(path).get_fdata()) > gr.DISCO_THRESH).ravel()
    idx = np.flatnonzero(flat)
    out = np.zeros(len(_MASKS), dtype=np.float32)
    for j, (m, vol) in enumerate(_MASKS):
        out[j] = m[idx].sum() / vol if len(idx) else 0.0
    return out


def half_masks(atlas_dir):
    """[(flat boolean mask, volume)] in the order split, network 1..16, (W1, W0),
    the treatments as label_images assigns them."""
    masks, keys = [], []
    for s in SPLITS:
        pairs = gr.load_atlas_pairs(atlas_dir, s)
        for k in range(1, 17):
            p = pairs[k]
            by_t = {t: (m, v) for m, v, t in zip(p.masks, p.vols, p.treatments)}
            for t in ("1", "0"):
                m, v = by_t[t]
                masks.append((np.asarray(m, dtype=bool).ravel(), int(v)))
                keys.append((s, k, t))
    return masks, keys


def hit_threshold(vol):
    """The fraction above which label_images counts a hit: overlap > round(0.05 vol)."""
    return int(np.round(vol * gr.ROI_THRESH)) / vol


def load_folds(rep_dir, names):
    folds = []
    for path in sorted(glob.glob(os.path.join(glob.escape(rep_dir), "fold*.npz"))):
        with np.load(path, allow_pickle=False) as z:
            if "files" in z.files:
                stored = [os.path.basename(str(f)) for f in z["files"]]
                if stored != list(names):
                    sys.exit(f"{os.path.basename(path)}: the stored listing differs from the image folder")
            folds.append((z["tr_idx"].copy(), z["te_idx"].copy(), z["Ztr"].copy(), z["Zte"].copy()))
    if len(folds) != 10:
        sys.exit(f"expected 10 folds, found {len(folds)}")
    return folds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from ncpfold.folds import load_groups
    from ncpfold.score import singles_mask

    files = sorted(glob.glob(os.path.join(glob.escape(a.images_dir), "*.nii*")))
    names = [os.path.basename(f) for f in files]
    folds = load_folds(a.reps, names)
    single = singles_mask(names, load_groups(a.groups))
    masks, keys = half_masks(a.atlas_dir)
    thr = np.array([hit_threshold(v) for _, v in masks])
    print(f"{len(files)} maps, {int(single.sum())} single-acquisition, {len(keys)} halves; computing coverage ...", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(masks,)) as pool:
        cov = np.stack(pool.map(_coverage, files, chunksize=16))            # [n, 64]
    hit = cov > thr

    # ---- A. probe of the coverage of every half
    rows_a = []
    for j, (s, k, t) in enumerate(keys):
        y_true, p_r, p_g = [], [], []
        for tr, te, Ztr, Zte in folds:
            keep = single[te]
            if not keep.any():
                continue
            sc = StandardScaler().fit(Ztr)
            Xtr, Xte = sc.transform(Ztr), sc.transform(Zte[keep])
            y_true.append(cov[te[keep], j])
            p_r.append(Ridge(alpha=1.0).fit(Xtr, cov[tr, j]).predict(Xte))
            p_g.append(HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Xtr, cov[tr, j]).predict(Xte))
        y, pr, pg = np.concatenate(y_true), np.concatenate(p_r), np.concatenate(p_g)
        sst = ((y - y.mean()) ** 2).sum()
        dist = np.abs(y - thr[j]) / thr[j]
        side_ok = (pg > thr[j]) == (y > thr[j])
        near = dist < 0.5
        rows_a.append(dict(split=s, network=gr.DEFICIT_NAMES[k - 1], half=f"W{t}",
                           share_hit=float((y > thr[j]).mean()), mean_cov=float(y.mean()),
                           r2_ridge=float(1 - ((y - pr) ** 2).sum() / sst) if sst > 0 else np.nan,
                           r2_gbm=float(1 - ((y - pg) ** 2).sum() / sst) if sst > 0 else np.nan,
                           share_near=float(near.mean()),
                           side_acc_near=float(side_ok[near].mean()) if near.any() else np.nan,
                           side_acc_far=float(side_ok[~near].mean()) if (~near).any() else np.nan))

    # ---- B. errors of a latent learner fitted to the true effect, by distance to the threshold
    rows_b, pehe_cells = [], {s: [] for s in SPLITS}
    acc = {s: {b: [0, 0.0] for b in range(len(BINS))} for s in SPLITS}
    for s in SPLITS:
        for k in range(1, 17):
            j1, j0 = keys.index((s, k, "1")), keys.index((s, k, "0"))
            for tr, te, Ztr, Zte in folds:
                keep = single[te]
                te_s, Zte_s = te[keep], Zte[keep]
                en_tr = hit[tr, j1] | hit[tr, j0]
                en_te = hit[te_s, j1] | hit[te_s, j0]
                if en_tr.sum() < 15 or en_te.sum() == 0:
                    continue
                tau = lambda ix: hit[ix, j1].astype(float) - hit[ix, j0].astype(float)
                sc = StandardScaler().fit(Ztr[en_tr])
                m = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(sc.transform(Ztr[en_tr]), tau(tr[en_tr]))
                ix = te_s[en_te]
                err = (m.predict(sc.transform(Zte_s[en_te])) - tau(ix)) ** 2
                pehe_cells[s].append(float(np.sqrt(err.mean())))
                d = np.minimum(np.abs(cov[ix, j1] - thr[j1]) / thr[j1], np.abs(cov[ix, j0] - thr[j0]) / thr[j0])
                for b, (lo, hi) in enumerate(BINS):
                    sel = (d >= lo) & (d < hi)
                    acc[s][b][0] += int(sel.sum()); acc[s][b][1] += float(err[sel].sum())
        n_tot = sum(v[0] for v in acc[s].values()); e_tot = sum(v[1] for v in acc[s].values())
        for b, (lo, hi) in enumerate(BINS):
            n_b, e_b = acc[s][b]
            rows_b.append(dict(split=s, distance_bin=f"{lo:g}-{hi:g}", share_patients=n_b / n_tot,
                               share_sq_error=e_b / e_tot if e_tot > 0 else np.nan,
                               mse=e_b / n_b if n_b else np.nan, n_patient_trials=n_b))

    import pandas as pd

    os.makedirs(a.out, exist_ok=True)
    da, db = pd.DataFrame(rows_a), pd.DataFrame(rows_b)
    da.to_csv(os.path.join(a.out, "coverage_probe_by_half.csv"), index=False)
    db.to_csv(os.path.join(a.out, "effect_error_by_threshold_distance.csv"), index=False)
    print("\nA. coverage probe (64 halves), median over halves per split:")
    print(da.groupby("split")[["r2_ridge", "r2_gbm", "share_near", "side_acc_near", "side_acc_far"]].median().round(3).to_string())
    print("\nB. gradient boosting fitted to the true effect, PEHE (mean over trial x fold cells):")
    for s in SPLITS:
        print(f"  {s}: {np.mean(pehe_cells[s]):.3f} over {len(pehe_cells[s])} cells")
    print("   squared error by distance to the threshold (|cov - t| / t, nearest half):")
    print(db.round(3).to_string(index=False))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
