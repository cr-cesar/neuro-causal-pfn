"""Does resolving the threshold band buy PEHE, and how much of the band is lost to resolution? (2 Nov)

Follows band_diagnostic.py, where NMF-50 (fitted on the binarised maps)
resolved the 5 % coverage band better than E1 although its trial PEHE is
worse (0.328 against 0.295 under LR/ET). Two parts, on the benchmark folds
(single-acquisition test patients, true effect known: oracle settings that
measure what a representation holds, not methods):

1. Oracles per representation (E1 and NMF-50 refitted on each fold's training
   side): PEHE of a learner fitted to the true effect from the latents
   (gradient boosting; TabPFN regressor with --tabpfn, the setting of the
   thesis's latent reference, 0.243 for E1 over three seeds) and of the
   two-stage classifier of two_stage_diagnostic.py. If NMF's oracles fall
   below E1's, the band resolution is worth PEHE.
2. Resolution floor: each map is smoothed to 1/f of its resolution (linear
   down- and up-sampling, f in --factors) and binarised at 0.5 as the
   benchmark does; the coverage of every half is recomputed. Reported per f:
   the share of band patients whose side of the threshold changes, and the
   PEHE of the effect read from the degraded coverage (no learner): the best
   any representation holding only that resolution could do. The encoder's
   input is the full grid, so f = 1 is zero by construction.

Aggregates only. --cov-cache: the full-resolution coverage of
two_stage_diagnostic.py.
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import band_diagnostic as bd  # noqa: E402
import coverage_diagnostic as cd  # noqa: E402
import two_stage_diagnostic as ts  # noqa: E402

from neurocausalpfn.prior import giles_replica as gr  # noqa: E402

_STATE = {}


def _init(masks, factors):
    _STATE["masks"], _STATE["factors"] = masks, factors


def _degraded_coverage(path):
    """[len(factors), n_halves] coverage of the map smoothed to 1/f resolution."""
    import nibabel as nib
    from scipy.ndimage import zoom

    vol = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
    out = np.zeros((len(_STATE["factors"]), len(_STATE["masks"])), dtype=np.float32)
    for a, f in enumerate(_STATE["factors"]):
        low = zoom(vol, 1.0 / f, order=1)
        back = zoom(low, [s / l for s, l in zip(vol.shape, low.shape)], order=1)[: vol.shape[0], : vol.shape[1], : vol.shape[2]]
        if back.shape != vol.shape:
            pad = [(0, s - b) for s, b in zip(vol.shape, back.shape)]
            back = np.pad(back, pad)
        idx = np.flatnonzero(back.ravel() > gr.DISCO_THRESH)
        for j, (m, v) in enumerate(_STATE["masks"]):
            out[a, j] = m[idx].sum() / v if len(idx) else 0.0
    return out


def trial_cells(folds, single, hit, keys):
    """[(split, k, tr, te_s, en_tr, en_te, j1, j0, Xtr_idx...)] helper data per fold and trial."""
    for f, (tr, te, Ztr, Zte) in enumerate(folds):
        keep = single[te]
        if not keep.any():
            continue
        yield f, tr, te[keep], Ztr, Zte[keep]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rep", required=True, metavar="TAG=FOLDS_DIR", help="the reference representation (E1)")
    ap.add_argument("--tabpfn", action="store_true", help="also the TabPFN regressor oracle (slow on CPU)")
    ap.add_argument("--factors", type=float, nargs="*", default=[2, 4, 8])
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--cov-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    from ncpfold.folds import load_groups
    from ncpfold.score import _nmf_perfold, singles_mask

    files = sorted(glob.glob(os.path.join(glob.escape(a.images_dir), "*.nii*")))
    names = [os.path.basename(f) for f in files]
    single = singles_mask(names, load_groups(a.groups))
    masks, keys = cd.half_masks(a.atlas_dir)
    thr = np.array([cd.hit_threshold(v) for _, v in masks])
    with np.load(a.cov_cache) as z:
        cov = z["cov"]
    if cov.shape != (len(files), len(keys)):
        sys.exit("the coverage cache does not match the image folder")
    hit = cov > thr
    os.makedirs(a.out, exist_ok=True)

    # ---- 2. resolution floor (no learner)
    rows_f = []
    if a.factors:
        print(f"degrading {len(files)} maps at factors {a.factors} ...", flush=True)
        with Pool(a.workers, initializer=_init, initargs=(masks, list(a.factors))) as pool:
            deg = np.stack(pool.map(_degraded_coverage, files, chunksize=16))     # [n, F, halves]
        tag, path = a.rep.partition("=")[0], a.rep.partition("=")[2]
        folds = [f[1:] for f in bd.load_some_folds(path, names)]
        test = np.zeros(len(files), dtype=bool)
        for tr, te, _, _ in folds:
            test[te[single[te]]] = True
        for ai, fct in enumerate(a.factors):
            hit_f = deg[:, ai, :] > thr
            band_flip = []
            for j in range(len(keys)):
                y = cov[test, j]
                band = np.abs(y - thr[j]) < 0.25 * thr[j]
                if band.any():
                    band_flip.append(float((hit_f[test, j][band] != hit[test, j][band]).mean()))
            pehe = {s: [] for s in cd.SPLITS}
            for s in cd.SPLITS:
                for k in range(1, 17):
                    j1, j0 = keys.index((s, k, "1")), keys.index((s, k, "0"))
                    for tr, te, _, _ in folds:
                        te_s = te[single[te]]
                        en = hit[te_s, j1] | hit[te_s, j0]
                        if not en.any():
                            continue
                        ix = te_s[en]
                        tau = hit[ix, j1].astype(float) - hit[ix, j0].astype(float)
                        tau_f = hit_f[ix, j1].astype(float) - hit_f[ix, j0].astype(float)
                        pehe[s].append(float(np.sqrt(((tau_f - tau) ** 2).mean())))
            rows_f.append(dict(factor=fct, voxel_mm=2 * fct, band_flip_median=float(np.median(band_flip)),
                               pehe_receptor=float(np.mean(pehe["receptor"])),
                               pehe_genetics=float(np.mean(pehe["genetics"]))))
            print(f"  factor {fct:g} done", flush=True)
        pd.DataFrame(rows_f).to_csv(os.path.join(a.out, "resolution_floor.csv"), index=False)

    # ---- 1. oracles on E1 and NMF-50, same folds
    tag, path = a.rep.partition("=")[0], a.rep.partition("=")[2]
    ref = [f[1:] for f in bd.load_some_folds(path, names)]
    refit = _nmf_perfold(files)
    nmf = []
    for k, (tr, te, _, _) in enumerate(ref):
        Ztr, Zte = refit(tr, te)
        nmf.append((tr, te, Ztr, Zte))
        print(f"  NMF-50 fold {k + 1}/10 refitted", flush=True)
    tabpfn = None
    if a.tabpfn:
        from tabpfn import TabPFNRegressor
        tabpfn = TabPFNRegressor
    gbm = dict(max_iter=200, random_state=0)
    rows_o = []
    for rtag, folds in ((tag, ref), ("NMF-50", nmf)):
        cells = {s: {"gbm_direct": [], "two_stage_clf": [], "tabpfn_direct": []} for s in cd.SPLITS}
        for f, tr, te_s, Ztr, Zte_s in trial_cells(folds, single, hit, keys):
            sc = StandardScaler().fit(Ztr)
            Xtr, Xte = sc.transform(Ztr), sc.transform(Zte_s)
            p_clf = {}
            for j in range(len(keys)):
                y = hit[tr, j]
                p_clf[j] = (np.full(len(te_s), float(y.mean())) if (y.all() or not y.any()) else
                            HistGradientBoostingClassifier(**gbm).fit(Xtr, y).predict_proba(Xte)[:, 1])
            for s in cd.SPLITS:
                for k in range(1, 17):
                    j1, j0 = keys.index((s, k, "1")), keys.index((s, k, "0"))
                    en_tr, en_te = hit[tr, j1] | hit[tr, j0], hit[te_s, j1] | hit[te_s, j0]
                    if en_tr.sum() < 15 or en_te.sum() == 0:
                        continue
                    t_tr = hit[tr[en_tr], j1].astype(float) - hit[tr[en_tr], j0].astype(float)
                    t_te = hit[te_s[en_te], j1].astype(float) - hit[te_s[en_te], j0].astype(float)
                    rm = lambda p: float(np.sqrt(((p - t_te) ** 2).mean()))
                    cells[s]["gbm_direct"].append(rm(HistGradientBoostingRegressor(**gbm).fit(Xtr[en_tr], t_tr).predict(Xte[en_te])))
                    cells[s]["two_stage_clf"].append(rm(ts.enrolled_effect(p_clf[j1][en_te], p_clf[j0][en_te])))
                    if tabpfn is not None:
                        m = tabpfn(random_state=0).fit(Xtr[en_tr], t_tr)
                        cells[s]["tabpfn_direct"].append(rm(m.predict(Xte[en_te])))
            print(f"  {rtag} fold {f + 1}/10 done", flush=True)
        for s in cd.SPLITS:
            for m, v in cells[s].items():
                if v:
                    rows_o.append(dict(rep=rtag, split=s, oracle=m, pehe=float(np.mean(v)), n_cells=len(v)))
    df = pd.DataFrame(rows_o)
    df.to_csv(os.path.join(a.out, "oracles_e1_nmf.csv"), index=False)
    print("\n1. oracles (true effect known), PEHE, mean over trial x fold cells:")
    piv = df.pivot_table(index=["rep", "oracle"], columns="split", values="pehe")
    piv["mean"] = piv.mean(axis=1)
    print(piv.round(3).to_string())
    if rows_f:
        print("\n2. resolution floor (map smoothed to 1/f, binarised at 0.5; no learner):")
        print(pd.DataFrame(rows_f).round(3).to_string(index=False))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
