"""Resolution of the coverage near the 5 % threshold, per representation (2 Nov).

Follows coverage_diagnostic.py. With a step effect any imperfect predictor of
coverage concentrates its error at the threshold; what says something about
a representation is its resolution there: the error of the predicted coverage,
in coverage units, inside the band |cov - t| < 0.25 t, against the half-width
of that band (0.25 t). Per representation (E1, a 100-latent variant, NMF-50
refitted in each fold) and half: gradient boosting regresses the coverage on
the fold's training side; on the single-acquisition test patients it reports
RMSE in the band, that RMSE over 0.25 t, and the share of band patients put on
the wrong side of the threshold.

It also checks the input of the encoders: the maps' grid against the VAE's
input shape (train_vae.full_config: pad or crop, no resampling); when the
maps fit inside it, the encoder sees every voxel and no coverage changes side
before the network.

Aggregates only. --cov-cache: the coverage matrix of two_stage_diagnostic.py.

    PYTHONPATH=src:<perfold> python scripts/audit_20261028/band_diagnostic.py \\
        --rep E1=<.../E1/E1/seed0/folds> --rep E5=<.../E5/.../seed0/folds> --nmf \\
        --images-dir ... --atlas-dir ... --groups ... --cov-cache ... --out ...
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coverage_diagnostic as cd  # noqa: E402


def load_some_folds(rep_dir, names):
    """[(fold id, tr, te, Ztr, Zte)] for the folds present, sorted by fold id."""
    out = []
    for path in sorted(glob.glob(os.path.join(glob.escape(rep_dir), "fold*.npz"))):
        with np.load(path, allow_pickle=False) as z:
            if "files" in z.files and [os.path.basename(str(f)) for f in z["files"]] != list(names):
                sys.exit(f"{os.path.basename(path)}: the stored listing differs from the image folder")
            k = int(z["fold"]) if "fold" in z.files else int(os.path.basename(path)[4:-4])
            out.append((k, z["tr_idx"].copy(), z["te_idx"].copy(), z["Ztr"].copy(), z["Zte"].copy()))
    if not out:
        sys.exit(f"no fold*.npz in {rep_dir}")
    return sorted(out, key=lambda f: f[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rep", action="append", required=True, metavar="TAG=FOLDS_DIR")
    ap.add_argument("--nmf", action="store_true", help="also NMF-50 refitted on each fold's training side")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--cov-cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import nibabel as nib
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    from ncpfold.folds import load_groups
    from ncpfold.score import _nmf_perfold, singles_mask
    from neurocausalpfn.train.train_vae import full_config

    files = sorted(glob.glob(os.path.join(glob.escape(a.images_dir), "*.nii*")))
    names = [os.path.basename(f) for f in files]
    shape = tuple(int(s) for s in nib.load(files[0]).shape[:3])
    in_shape = tuple(int(s) for s in full_config()["data"]["resolution"])
    fits = all(s <= t for s, t in zip(shape, in_shape))
    print(f"maps {shape}, encoder input {in_shape}: "
          f"{'padding only, every voxel reaches the encoder' if fits else 'CROPPED: voxels are lost before the encoder'}",
          flush=True)
    single = singles_mask(names, load_groups(a.groups))
    masks, keys = cd.half_masks(a.atlas_dir)
    thr = np.array([cd.hit_threshold(v) for _, v in masks])
    with np.load(a.cov_cache) as z:
        cov = z["cov"]
    if cov.shape != (len(files), len(keys)):
        sys.exit("the coverage cache does not match the image folder")

    # a representation may hold fewer than ten folds (only some were copied to
    # this machine); it is then compared with the first one, and NMF-50, on
    # the same folds (tags "<first>@<folds>", "NMF-50@<folds>")
    reps, fold_sets = [], []
    for item in a.rep:
        tag, _, path = item.partition("=")
        folds = load_some_folds(path, names)
        reps.append((tag, [f[1:] for f in folds]))
        ids = tuple(f[0] for f in folds)
        print(f"  {tag}: folds {list(ids)}", flush=True)
        if len(ids) < 10 and ids not in fold_sets:
            fold_sets.append(ids)
    ref_tag, ref = reps[0][0], dict(zip(range(10), reps[0][1]))
    if len(ref) != 10:
        sys.exit(f"the first representation ({ref_tag}) must hold the ten folds")
    # every representation must split the patients as the first one does
    # (the same group table and fold scheme), fold by fold
    for item in a.rep[1:]:
        tag, _, path = item.partition("=")
        for k, tr, te, _, _ in load_some_folds(path, names):
            if not (np.array_equal(np.sort(te), np.sort(ref[k][1])) and np.array_equal(np.sort(tr), np.sort(ref[k][0]))):
                sys.exit(f"{tag} fold {k}: its training and test sides differ from {ref_tag}'s (other group table or fold scheme)")
        print(f"  {tag}: same training and test sides as {ref_tag} in every fold it holds", flush=True)
    for ids in fold_sets:
        reps.append((f"{ref_tag}@{','.join(map(str, ids))}", [ref[k] for k in ids]))
    if a.nmf:
        refit = _nmf_perfold(files)
        nmf = {}
        for k in range(10):
            tr, te = ref[k][0], ref[k][1]
            Ztr, Zte = refit(tr, te)
            nmf[k] = (tr, te, Ztr, Zte)
            print(f"  NMF-50 fold {k + 1}/10 refitted", flush=True)
        reps.append(("NMF-50", [nmf[k] for k in range(10)]))
        for ids in fold_sets:
            reps.append((f"NMF-50@{','.join(map(str, ids))}", [nmf[k] for k in ids]))

    rows = []
    for tag, folds in reps:
        for j, (s, k, t) in enumerate(keys):
            y_all, p_all = [], []
            for tr, te, Ztr, Zte in folds:
                keep = single[te]
                if not keep.any():
                    continue
                sc = StandardScaler().fit(Ztr)
                m = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(sc.transform(Ztr), cov[tr, j])
                y_all.append(cov[te[keep], j]); p_all.append(m.predict(sc.transform(Zte[keep])))
            y, p = np.concatenate(y_all), np.concatenate(p_all)
            band = np.abs(y - thr[j]) < 0.25 * thr[j]
            sst = ((y - y.mean()) ** 2).sum()
            rmse_b = float(np.sqrt(((p[band] - y[band]) ** 2).mean())) if band.any() else np.nan
            rows.append(dict(rep=tag, split=s, half=f"{k}W{t}", threshold=float(thr[j]),
                             r2=float(1 - ((p - y) ** 2).sum() / sst) if sst > 0 else np.nan,
                             share_band=float(band.mean()), rmse_band=rmse_b,
                             rmse_band_over_halfwidth=rmse_b / (0.25 * thr[j]) if band.any() else np.nan,
                             wrong_side_band=float(((p[band] > thr[j]) != (y[band] > thr[j])).mean()) if band.any() else np.nan))
        print(f"  {tag} done", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "band_resolution_by_half.csv"), index=False)
    print("\nmedian over the 32 halves of each split:")
    print(df.groupby(["rep", "split"])[["r2", "share_band", "rmse_band", "rmse_band_over_halfwidth",
                                        "wrong_side_band"]].median().round(3).to_string())
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
