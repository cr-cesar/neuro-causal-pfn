"""Is the missing precision in the latents or in how the estimators use them? (2 Nov)

coverage_diagnostic.py found that the E1 latents recover the coverage of every
network half (R^2 about 0.93) and the side of the 5 % threshold away from it
(99 %), but only about 77 % of the sides near it, where most of the error of
a learner fitted to the true effect lies. This scores, on the same benchmark
folds and with the true effect known (the oracle setting of the latent
reference), three ways of predicting the effect of trial (split, network)
from the latents:

- direct: gradient boosting regressed on the true effect (as in part B there);
- two-stage, classifier: per half, gradient boosting classifies "covered above
  the threshold" on every training image; with p1, p0 the two probabilities,
  the effect of an enrolled patient is E[h1 - h0 | h1 or h0]
  = (p1 - p0) / (1 - (1 - p1)(1 - p0)) (halves taken as independent);
- two-stage, coverage: per half, gradient boosting regresses the coverage;
  p = Phi((predicted - threshold) / s), s the residual SD on a held-out fifth
  of the training side; then as above.

The two-stage estimators use the atlas halves that define the effect, so they
measure what the latents hold, not a method for the benchmark. If they fall
well below the direct learner, the information is in the latents and the
estimators do not use it; if they do not, the representation is the limit.
Aggregates only. ``--cov-cache`` keeps the coverage matrix (derived from the
maps, stays on the machine) so that a second run skips reading the maps.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coverage_diagnostic as cd  # noqa: E402

from neurocausalpfn.prior import giles_replica as gr  # noqa: E402


def enrolled_effect(p1, p0):
    """E[h1 - h0 | h1 or h0] for independent hits with probabilities p1, p0."""
    den = 1.0 - (1.0 - p1) * (1.0 - p0)
    return np.where(den > 1e-9, (p1 - p0) / np.maximum(den, 1e-9), 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cov-cache", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    from multiprocessing import Pool

    from scipy.stats import norm
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    from ncpfold.folds import load_groups
    from ncpfold.score import singles_mask

    files = sorted(glob.glob(os.path.join(glob.escape(a.images_dir), "*.nii*")))
    names = [os.path.basename(f) for f in files]
    folds = cd.load_folds(a.reps, names)
    single = singles_mask(names, load_groups(a.groups))
    masks, keys = cd.half_masks(a.atlas_dir)
    thr = np.array([cd.hit_threshold(v) for _, v in masks])
    if a.cov_cache and os.path.exists(a.cov_cache):
        with np.load(a.cov_cache) as z:
            cov = z["cov"]
        if cov.shape != (len(files), len(keys)):
            sys.exit("the coverage cache does not match the image folder")
        print("coverage from the cache", flush=True)
    else:
        print(f"{len(files)} maps; computing coverage ...", flush=True)
        with Pool(a.workers, initializer=cd._init, initargs=(masks,)) as pool:
            cov = np.stack(pool.map(cd._coverage, files, chunksize=16))
        if a.cov_cache:
            np.savez_compressed(a.cov_cache, cov=cov)
    hit = cov > thr

    gbm = dict(max_iter=200, random_state=0)
    cells = {s: {"direct": [], "two_stage_clf": [], "two_stage_cov": []} for s in cd.SPLITS}
    for f, (tr, te, Ztr, Zte) in enumerate(folds):
        keep = single[te]
        if not keep.any():
            continue
        te_s, Zte_s = te[keep], Zte[keep]
        sc = StandardScaler().fit(Ztr)
        Xtr, Xte = sc.transform(Ztr), sc.transform(Zte_s)
        rng = np.random.default_rng(f)
        hold = rng.random(len(tr)) < 0.2
        # per half: probability of a hit on the test images, two ways
        p_clf, p_cov = {}, {}
        for j in range(len(keys)):
            y = hit[tr, j]
            if y.all() or not y.any():
                p_clf[j] = np.full(len(te_s), float(y.mean()))
            else:
                p_clf[j] = HistGradientBoostingClassifier(**gbm).fit(Xtr, y).predict_proba(Xte)[:, 1]
            r = HistGradientBoostingRegressor(**gbm).fit(Xtr[~hold], cov[tr[~hold], j])
            s = float(np.std(cov[tr[hold], j] - r.predict(Xtr[hold]))) or 1e-6
            r = HistGradientBoostingRegressor(**gbm).fit(Xtr, cov[tr, j])
            p_cov[j] = norm.cdf((r.predict(Xte) - thr[j]) / s)
        for s_ in cd.SPLITS:
            for k in range(1, 17):
                j1, j0 = keys.index((s_, k, "1")), keys.index((s_, k, "0"))
                en_tr = hit[tr, j1] | hit[tr, j0]
                en_te = hit[te_s, j1] | hit[te_s, j0]
                if en_tr.sum() < 15 or en_te.sum() == 0:
                    continue
                tau_tr = hit[tr[en_tr], j1].astype(float) - hit[tr[en_tr], j0].astype(float)
                tau_te = hit[te_s[en_te], j1].astype(float) - hit[te_s[en_te], j0].astype(float)
                d = HistGradientBoostingRegressor(**gbm).fit(Xtr[en_tr], tau_tr).predict(Xte[en_te])
                c = enrolled_effect(p_clf[j1][en_te], p_clf[j0][en_te])
                v = enrolled_effect(p_cov[j1][en_te], p_cov[j0][en_te])
                for name, pred in (("direct", d), ("two_stage_clf", c), ("two_stage_cov", v)):
                    cells[s_][name].append(float(np.sqrt(((pred - tau_te) ** 2).mean())))
        print(f"  fold {f + 1}/10 done", flush=True)

    import pandas as pd

    rows = [dict(split=s, estimator=m, pehe=float(np.mean(v)), n_cells=len(v))
            for s in cd.SPLITS for m, v in cells[s].items()]
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "two_stage_pehe.csv"), index=False)
    piv = df.pivot(index="estimator", columns="split", values="pehe")
    piv["mean"] = piv.mean(axis=1)
    print("\nPEHE with the true effect known (mean over trial x fold cells, single-acquisition test patients):")
    print(piv.round(3).to_string())
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
