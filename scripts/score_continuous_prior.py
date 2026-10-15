"""Score Phase-1 representations on the extended continuous prior (F32).

Robustness check: the same per-fold representations that the virtual trial of
Giles scores (E1, E3, E11b, NMF, lesion volume) are scored on trials drawn
from the theory family of the Neuro-Prior instead. For every fold and every
drawn process, the training patients receive a treatment from the process's
propensity and an outcome from its potential outcomes; a two-model estimator
(logistic regression and extra trees for binary outcomes, ridge and extra
trees regression for continuous ones) is fitted on (Z, W, Y) of the training
side and predicts the CATE of the test side, scored against the process's
true CATE mu1 - mu0 as an RMSE (PEHE on the continuous scale).

The anatomy of each patient comes from a prior cache built from the SAME
cohort's images (overlaps, centroids, volumes and file names), aligned to the
fold files by basename; the representation never enters the generator.

CAVEAT (audit, 15 Oct): in the current caches the two overlap columns are
the two halves of one atlas split, so this measures the generator as
implemented, not the one of the design document. Re-run after the cache
rebuild.

    python scripts/score_continuous_prior.py --reps <root>/E1/E1/seed0/folds \\
        --cache outputs/prior_cache/uclh.npz --atlas-dir <atlases> \\
        --hyper "two_arms=1 effect_scale=0.2:1" --processes 20 --label E1 --out cp_E1.csv
    python scripts/score_continuous_prior.py --reps <same folds> --volume --label volume ...

Aggregate output only (one row per fold, process and estimator).
"""
import argparse
import glob
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _estimators(binary: bool):
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    if binary:
        return {"logistic_regression": lambda: LogisticRegression(max_iter=2000),
                "extra_trees": lambda: ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0)}
    return {"ridge": lambda: Ridge(alpha=1.0),
            "extra_trees": lambda: ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0)}


def _predict(model, X, binary: bool) -> np.ndarray:
    if binary:
        p = model.predict_proba(X)
        return p[:, 1] if p.shape[1] == 2 else np.full(len(X), float(model.classes_[0]))
    return model.predict(X)


def two_model_cate(Ztr, W, Y, Zte, make, binary: bool) -> np.ndarray:
    """mu1_hat - mu0_hat from one model per arm; an arm with fewer than 5
    patients (or a single outcome class) predicts its arm mean."""
    preds = []
    for w in (1, 0):
        m = W == w
        if m.sum() < 5 or (binary and len(np.unique(Y[m])) < 2):
            preds.append(np.full(len(Zte), float(Y[m].mean()) if m.any() else float(Y.mean())))
            continue
        model = make().fit(Ztr[m], Y[m])
        preds.append(_predict(model, Zte, binary))
    return preds[0] - preds[1]


def _align(cache_files: np.ndarray, files: List[str]) -> np.ndarray:
    pos = {os.path.basename(str(f)): i for i, f in enumerate(cache_files)}
    missing = [f for f in files if os.path.basename(str(f)) not in pos]
    if missing:
        # no file name in the message: names can identify patients
        raise ValueError(f"{len(missing)} of {len(files)} fold files are not in the prior cache; "
                         "the cache must be built from the same cohort's images")
    return np.asarray([pos[os.path.basename(str(f))] for f in files])


def score_fold(fold_npz: str, cache: Dict, dgps, volume: bool, seed: int) -> List[Dict]:
    from neurocausalpfn.prior.intersynth_theory import make_theory_dataset

    z = np.load(fold_npz, allow_pickle=True)
    files = [str(f) for f in z["files"]]
    tr, te = np.asarray(z["tr_idx"]), np.asarray(z["te_idx"])
    rows_tr = _align(cache["files"], [files[i] for i in tr])
    rows_te = _align(cache["files"], [files[i] for i in te])
    if volume:
        v = np.log1p(np.asarray(cache["volumes"], dtype=np.float64))
        Ztr, Zte = v[rows_tr, None], v[rows_te, None]
    else:
        Ztr, Zte = np.asarray(z["Ztr"], np.float64), np.asarray(z["Zte"], np.float64)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
    ov, cen = cache["overlaps"], cache["centroids"]
    fold = int(z["fold"]) if "fold" in z.files else -1
    out = []
    for p, dgp in enumerate(dgps):
        rng = np.random.default_rng(seed + 1000 * p + max(fold, 0))
        data = make_theory_dataset(dgp, ov[rows_tr], cen[rows_tr], Ztr, ov[rows_te], cen[rows_te], Zte, rng)
        W, Y = np.asarray(data["Tc"]).astype(int), np.asarray(data["Yc"], np.float64)
        true = np.asarray(data["mu1"]) - np.asarray(data["mu0"])
        binary = bool(getattr(dgp, "binary", False))
        for name, make in _estimators(binary).items():
            cate = two_model_cate(Ztr, W, Y, Zte, make, binary)
            out.append({"fold": fold, "process": p, "estimator": name, "binary": binary,
                        "two_arms": bool(dgp.two_arms), "mechanism": dgp.mechanism, "gamma": float(dgp.gamma),
                        "n_train": int(len(W)), "treated_share": float(W.mean()),
                        "mean_abs_cate": float(np.mean(np.abs(true))),
                        "pehe": float(np.sqrt(np.mean((cate - true) ** 2))),
                        "pehe_zero": float(np.sqrt(np.mean(true ** 2)))})
    return out


def draw_processes(cache: Dict, atlas, hyper: Dict, n: int, seed: int):
    from neurocausalpfn.prior.intersynth_theory import InterSynthTheoryDGP, TheoryHyperPrior

    h = TheoryHyperPrior(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in hyper.items()})
    rng = np.random.default_rng(seed)
    return [InterSynthTheoryDGP(atlas, rng, h).calibrate(cache["overlaps"]) for _ in range(n)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", required=True, help="directory with fold*.npz (Ztr, Zte, tr_idx, te_idx, files)")
    ap.add_argument("--cache", required=True, help="prior cache npz of the SAME cohort (overlaps, centroids, volumes, files)")
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--modality", default="receptor")
    ap.add_argument("--hyper", default="two_arms=1 effect_scale=0.2:1")
    ap.add_argument("--processes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--volume", action="store_true", help="score log lesion volume from the cache instead of Z")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import pandas as pd

    from neurocausalpfn.prior.atlas import FunctionalAtlas
    from neurocausalpfn.train.train_pfn import parse_hyper_opts

    with np.load(args.cache, allow_pickle=True) as c:
        cache = {k: c[k] for k in ("overlaps", "centroids", "volumes", "files")}
    cache["overlaps"] = np.asarray(cache["overlaps"], np.float64)
    atlas = FunctionalAtlas.from_dir(args.atlas_dir, modality=args.modality)
    dgps = draw_processes(cache, atlas, parse_hyper_opts(args.hyper), args.processes, args.seed)
    folds = sorted(glob.glob(os.path.join(args.reps, "fold*.npz")))
    if not folds:
        raise SystemExit(f"no fold*.npz under {args.reps}")
    rows = []
    for f in folds:
        rows += score_fold(f, cache, dgps, args.volume, args.seed)
        print(f"  {os.path.basename(f)} done ({len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    df.insert(0, "representation", args.label or ("volume" if args.volume else os.path.basename(os.path.dirname(args.reps.rstrip("/")))))
    df.to_csv(args.out, index=False)
    # per process: mean over folds; then mean and 95 % CI over processes
    per = df.groupby(["estimator", "process"])["pehe"].mean().reset_index()
    agg = per.groupby("estimator")["pehe"].agg(["mean", "std", "count"])
    agg["ci_low"] = agg["mean"] - 1.96 * agg["std"] / np.sqrt(agg["count"])
    agg["ci_high"] = agg["mean"] + 1.96 * agg["std"] / np.sqrt(agg["count"])
    agg["pehe_zero"] = df.groupby("estimator")["pehe_zero"].mean()
    print(f"{df['representation'].iloc[0]}: {len(folds)} folds x {args.processes} processes")
    print(agg.round(4).to_string())
    agg.to_csv(args.out.replace(".csv", "_summary.csv"))
    print("wrote", args.out)
    return df


if __name__ == "__main__":
    main()
