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
  yardstick for how much a learner can read from 1,024 rows;
- linear_two_model: two-model ridge on the same context (the trial's
  logistic-regression two-model is this learner on binary outcomes);
- linear_dgp_* and step_dgp_*: the same covariates with a CATE of a simple
  shape at the prior's magnitude (a sparse linear score; three groups by that
  score, the virtual trial's shape, tau in {-1, 0, +1}), read by the
  checkpoint, by ridge and by ExtraTrees, with continuous and binary context
  outcomes. If the checkpoint reads the prior's own processes but not these,
  its prior lacks the simple function class the trial has.

The header line also reports the in-sample R² of a ridge fit of the prior's
true CATE on the latents: how linear the prior's effects are in latent space.

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


def linear_two_model(Xc, T, Y, Xq, alpha=1.0):
    """Two-model ridge regression on the same context: the linear yardstick
    (the trial's logistic regression two-model is this learner on binary Y)."""
    from sklearn.linear_model import Ridge

    p = [Ridge(alpha=alpha).fit(Xc[T == w], Y[T == w]).predict(Xq) for w in (1, 0)]
    return p[0] - p[1]


def linear_r2(X, truth, alpha=1.0):
    """How much of a process's true CATE a linear function of the covariates
    explains (ridge fitted on the whole slice, in-sample R²): the prior's
    function class in latent space, to compare with the trial's."""
    from sklearn.linear_model import Ridge

    if truth.std() == 0:
        return float("nan")
    pred = Ridge(alpha=alpha).fit(X, truth).predict(X)
    return float(1.0 - ((truth - pred) ** 2).sum() / ((truth - truth.mean()) ** 2).sum())


def synthetic_dgp(rng, Xc, Xq, kind: str, scale: float, p0_range=(0.2, 0.8), n_active=5):
    """A process on the SAME covariates with a CATE of a known, simple shape,
    at the prior's effect magnitude ``scale`` (mean |CATE|):

    - ``linear``: CATE proportional to a sparse linear score of the covariates;
    - ``step``: three groups by the same score (bottom quarter -scale, top
      quarter +scale, the rest 0), the shape of the virtual trial's CATE
      (tau in {-1, 0, +1} by anatomy).

    Returns context outcomes (continuous, in [0, 1]) and the query's true CATE.
    """
    d = Xc.shape[1]
    w = np.zeros(d); active = rng.choice(d, size=min(n_active, d), replace=False)
    w[active] = rng.normal(size=len(active))
    X = np.vstack([Xc, Xq]); z = X @ w; z = (z - z.mean()) / (z.std() + 1e-8)
    if kind == "linear":
        tau = z * scale / (np.abs(z).mean() + 1e-8)
    elif kind == "step":
        lo, hi = np.quantile(z, [0.25, 0.75])
        tau = np.where(z <= lo, -1.0, np.where(z >= hi, 1.0, 0.0)) * scale / 0.5
    else:
        raise ValueError(kind)
    v = rng.normal(size=d) / np.sqrt(d)
    p0 = np.clip(rng.uniform(*p0_range) + 0.15 * (X @ v), 0.05, 0.95)
    p1 = np.clip(p0 + tau, 0.0, 1.0)
    tau = p1 - p0
    nc = len(Xc)
    return p0[:nc], p1[:nc], tau[nc:]


def run(ckpt: str, processes: int, n_context: int, seed: int, device: str = "cpu", n_jobs: int = 4,
        dgps: bool = True):
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
        outs["linear_two_model"] = linear_two_model(Xc, T, Y, Xq)
        info = b["processes"][0] if "processes" in b else {}
        r2 = linear_r2(Xq, truth)        # how linear the prior's CATE is in latent space (query rows)
        common = {"process": k, "n_slice": info.get("n_slice", np.nan), "family": info.get("family", "")}
        for name, pred in outs.items():
            rows.append({**common, "variant": name, "linear_r2_truth": r2, **_metrics(pred, truth)})
        if not dgps:
            continue
        # the same covariates with a simple CATE at the prior's magnitude: can
        # the checkpoint read a linear or a three-group effect from context?
        scale = float(np.abs(truth).mean()) or 0.2
        for kind in ("linear", "step"):
            p0c, p1c, tau_q = synthetic_dgp(rng, Xc, Xq, kind, scale)
            Yk = np.where(T == 1, p1c, p0c)
            ybin_k = (rng.uniform(size=len(Yk)) < Yk).astype(float)
            dg = {f"{kind}_dgp_pfn": np.subtract(*est(Xc, T, Yk, Xq)),
                  f"{kind}_dgp_pfn_binary_Y": np.subtract(*est(Xc, T, ybin_k, Xq)),
                  f"{kind}_dgp_linear_two_model": linear_two_model(Xc, T, Yk, Xq),
                  f"{kind}_dgp_linear_two_model_binary_Y": linear_two_model(Xc, T, ybin_k, Xq),
                  f"{kind}_dgp_extra_trees": et_two_model(Xc, T, Yk, Xq, seed=seed, n_jobs=n_jobs)}
            for name, pred in dg.items():
                rows.append({**common, "variant": name, "linear_r2_truth": float("nan"), **_metrics(pred, tau_q)})
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
    ap.add_argument("--no-dgps", action="store_true",
                    help="skip the linear and step DGPs on the prior's covariates")
    args = ap.parse_args(argv)

    import pandas as pd

    rows = pd.DataFrame(run(args.ckpt, args.processes, args.n_context, args.seed, args.device, args.jobs,
                            dgps=not args.no_dgps))
    summ = rows.groupby("variant")[["corr", "rmse", "rescale", "slope", "mean_abs_pred", "mean_abs_truth"]].mean()
    order = ["as_trained", "slice_standardised", "rotated", "binary_Y", "slice_standardised_binary_Y",
             "extra_trees", "linear_two_model"]
    order += [v for v in summ.index if v.endswith("_dgp_pfn") or "_dgp_" in v]
    seen = set(); order = [v for v in order if not (v in seen or seen.add(v))]
    summ = summ.loc[[v for v in order if v in summ.index]]
    print(f"checkpoint {os.path.basename(os.path.dirname(args.ckpt))}: {args.processes} held-out processes, "
          f"context {args.n_context}, median slice {rows.n_slice.median():.0f}, "
          f"linear R2 of the prior's CATE on the latents {rows.linear_r2_truth.mean():.2f}")
    print(summ.round(3).to_string())
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        rows.to_csv(args.out, index=False)
        summ.to_csv(args.out.replace(".csv", "_summary.csv"))
        print("wrote", args.out)
    return summ


if __name__ == "__main__":
    main()
