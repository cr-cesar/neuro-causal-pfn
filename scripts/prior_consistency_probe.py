"""Does the scoring path see what the checkpoint was trained on? (audit of 19 Oct)

Draws processes from the checkpoint's own prior with known CATE and scores
them through ``PFNEstimator`` under the transformations the scoring path
applies, so that every variant is compared inside the prior, where the
truth is continuous and known:

- as_trained: covariates exactly as the prior emitted them (pool-standardised,
  rotated), no re-standardisation;
- slice_standardised: the scoring path's default, context mean and sd;
- rotated: a fixed extra orthogonal rotation (the augmentation group);
- binary_Y: the context outcomes drawn as Bernoulli(Y) (the trial's outcomes
  are binary, the theory prior's are continuous);
- extra_trees: a two-model ExtraTrees fitted on the same context, the
  yardstick for how much a learner can read from 1,024 rows.

Per variant: correlation with the true CATE, rmse, the oracle rescaling
factor (above 1 = the predictions are shrunk) and the slope of prediction on
truth; all averaged over processes. CPU, minutes.

    python scripts/prior_consistency_probe.py --ckpt outputs/pfn_reduced_<run>/pfn.pt \\
        --processes 40 --out outputs/prior_consistency_<run>.csv
"""
import argparse
import os

import numpy as np


def _metrics(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    ss = float((pred ** 2).sum())
    corr = float(np.corrcoef(pred, truth)[0, 1]) if pred.std() > 0 and truth.std() > 0 else float("nan")
    dt = truth - truth.mean()
    slope = float((dt * (pred - pred.mean())).sum() / (dt ** 2).sum()) if (dt ** 2).sum() > 0 else float("nan")
    return {"corr": corr, "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
            "rescale": float((pred * truth).sum() / ss) if ss > 0 else float("nan"), "slope": slope,
            "mean_abs_pred": float(np.abs(pred).mean()), "mean_abs_truth": float(np.abs(truth).mean())}


def et_two_model(Xc, T, Y, Xq, seed=0, n_jobs=4):
    from sklearn.ensemble import ExtraTreesRegressor

    p = []
    for w in (1, 0):
        m = ExtraTreesRegressor(200, min_samples_leaf=3, n_jobs=n_jobs, random_state=seed)
        p.append(m.fit(Xc[T == w], Y[T == w]).predict(Xq))
    return p[0] - p[1]


def run(ckpt: str, processes: int, n_context: int, seed: int, device: str = "cpu", n_jobs: int = 4):
    from neurocausalpfn.pfn.estimator import PFNEstimator
    from neurocausalpfn.train.train_pfn import _build_prior

    est = PFNEstimator(ckpt, device=device, standardize=False, context_cap=0)
    cfg = dict(est.cfg); cfg["seed"] = int(cfg.get("seed", 0)) + seed
    prior, d_x, is_intersynth = _build_prior(cfg, seed_offset=10_000)
    if not is_intersynth:
        raise SystemExit("the checkpoint was not trained on a Neuro-Prior cache")
    rng = np.random.default_rng(seed)
    R = np.linalg.qr(rng.normal(size=(d_x, d_x)))[0]
    rows = []
    for k in range(processes):
        b = prior.sample_batch(1, n_context=n_context)
        Xc, T, Y, Xq = (np.asarray(b[key][0]) for key in ("Xc", "Tc", "Yc", "Xq"))
        truth = np.asarray(b["mu1"][0]) - np.asarray(b["mu0"][0])
        ybin = (rng.uniform(size=len(Y)) < Y).astype(float)
        outs = {}
        est.standardize = False
        outs["as_trained"] = np.subtract(*est(Xc, T, Y, Xq))
        outs["rotated"] = np.subtract(*est(Xc @ R, T, Y, Xq @ R))
        outs["binary_Y"] = np.subtract(*est(Xc, T, ybin, Xq))
        est.standardize = True
        outs["slice_standardised"] = np.subtract(*est(Xc, T, Y, Xq))
        outs["slice_standardised_binary_Y"] = np.subtract(*est(Xc, T, ybin, Xq))
        est.standardize = False
        outs["extra_trees"] = et_two_model(Xc, T, Y, Xq, seed=seed, n_jobs=n_jobs)
        info = b["processes"][0] if "processes" in b else {}
        for name, pred in outs.items():
            rows.append({"process": k, "variant": name, "n_slice": info.get("n_slice", np.nan),
                         "family": info.get("family", ""), **_metrics(pred, truth)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--processes", type=int, default=40)
    ap.add_argument("--n-context", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import pandas as pd

    rows = pd.DataFrame(run(args.ckpt, args.processes, args.n_context, args.seed, args.device, args.jobs))
    summ = rows.groupby("variant")[["corr", "rmse", "rescale", "slope", "mean_abs_pred", "mean_abs_truth"]].mean()
    order = ["as_trained", "slice_standardised", "rotated", "binary_Y", "slice_standardised_binary_Y", "extra_trees"]
    summ = summ.loc[[v for v in order if v in summ.index]]
    print(f"checkpoint {os.path.basename(os.path.dirname(args.ckpt))}: {args.processes} held-out processes, "
          f"context {args.n_context}, median slice {rows.n_slice.median():.0f}")
    print(summ.round(3).to_string())
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        rows.to_csv(args.out, index=False)
        summ.to_csv(args.out.replace(".csv", "_summary.csv"))
        print("wrote", args.out)
    return summ


if __name__ == "__main__":
    main()
