"""Probe a trained Neuro-Causal-PFN checkpoint (CPU, minutes, no patient data).

Separates the two reasons a checkpoint can predict near-zero effects on the
virtual-trial evaluation:

1. Treatment sensitivity: contexts on the pool latents whose outcome is the
   treatment itself (Y = T) or its negation (Y = 1 - T). A model that reads
   the treatment column of the context predicts a CATE near +1 and near -1;
   a model that ignores it predicts about 0 in both.
2. Per prior family: held-out processes of each family (theory, giles) drawn
   on the same anatomy cache. RMSE and correlation of the predicted against
   the true CATE, mean |CATE| predicted against true, and the balanced
   accuracy of the sign where |true CATE| > 0.1. A checkpoint that is good on
   its own family and blind on the other has learnt its prior, and the prior
   does not cover the evaluation.

    python scripts/probe_pfn.py --ckpt outputs/pfn_reduced_kch_e1/pfn.pt
    python scripts/probe_pfn.py --ckpt a/pfn.pt b/pfn.pt --cache outputs/prior_cache/kch_e1.npz --out probe.csv
"""
import argparse
import copy
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SIGN_MIN = 0.1      # |true CATE| above which the sign is scored


def cate_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    pred, true = np.asarray(pred, float).ravel(), np.asarray(true, float).ravel()
    out = {"n": int(len(true)), "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
           "mean_abs_pred": float(np.mean(np.abs(pred))), "mean_abs_true": float(np.mean(np.abs(true))),
           "corr": float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 1e-9 and true.std() > 1e-9 else float("nan")}
    m = np.abs(true) > SIGN_MIN
    pos, neg = m & (true > 0), m & (true < 0)
    sens = float((pred[pos] > 0).mean()) if pos.any() else float("nan")
    spec = float((pred[neg] < 0).mean()) if neg.any() else float("nan")
    out.update({"n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "sign_sens": sens, "sign_spec": spec,
                "sign_balacc": float(np.nanmean([sens, spec])) if pos.any() or neg.any() else float("nan")})
    return out


def _cohort(cfg: Dict, family: str, n_context: int, n_query: int, seed_offset: int):
    from neurocausalpfn.train.train_pfn import _build_prior, with_family

    c = copy.deepcopy(cfg)
    with_family(c, family)
    c["prior"].pop("stages", None)
    c["pfn"]["context_max"], c["pfn"]["n_query"] = int(n_context), int(n_query)
    prior, _, _ = _build_prior(c, seed_offset=seed_offset)
    return prior


def probe_families(est, cfg: Dict, families: List[str], n_context: int, n_query: int,
                   batches: int, seed_offset: int = 10_000) -> List[Dict]:
    rows = []
    for fam in families:
        coh = _cohort(cfg, fam, n_context, n_query, seed_offset)
        preds, trues = [], []
        for _ in range(batches):
            b = coh.sample_batch(1, n_context=n_context)
            p1, p0 = est(b["Xc"][0], b["Tc"][0], b["Yc"][0], b["Xq"][0])
            preds.append(p1 - p0); trues.append(b["mu1"][0] - b["mu0"][0])
        rows.append({"probe": f"family:{fam}", **cate_metrics(np.concatenate(preds), np.concatenate(trues))})
    return rows


def probe_treatment(est, cfg: Dict, n_context: int, n_query: int, batches: int, seed: int = 0) -> List[Dict]:
    """Y = T (true CATE +1) and Y = 1 - T (true CATE -1) on pool latents."""
    coh = _cohort(cfg, cfg["prior"].get("family", "theory"), n_context, n_query, 20_000)
    X = coh.X
    rng = np.random.default_rng(seed)
    rows = []
    for name, sign in (("Y=T", 1.0), ("Y=1-T", -1.0)):
        preds = []
        for _ in range(batches):
            idx = rng.permutation(len(X))
            ci, qi = idx[:n_context], idx[n_context:n_context + n_query]
            T = (rng.uniform(size=len(ci)) < 0.5).astype(np.float32)
            Y = T if sign > 0 else 1.0 - T
            p1, p0 = est(X[ci], T, Y, X[qi])
            preds.append(p1 - p0)
        pred = np.concatenate(preds)
        rows.append({"probe": f"treatment:{name}", **cate_metrics(pred, np.full(len(pred), sign))})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--cache", default=None, help="anatomy cache npz (default: the one in the checkpoint cfg)")
    ap.add_argument("--atlas-dir", default=None, help="default: the one in the checkpoint cfg")
    ap.add_argument("--families", nargs="+", default=["theory", "giles"])
    ap.add_argument("--n-context", type=int, default=1000)
    ap.add_argument("--n-query", type=int, default=256)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="csv with one row per checkpoint and probe")
    args = ap.parse_args(argv)

    import pandas as pd

    from neurocausalpfn.pfn.estimator import PFNEstimator

    rows = []
    for ck in args.ckpt:
        est = PFNEstimator(ck, device=args.device)
        cfg = copy.deepcopy(est.cfg)
        if args.cache:
            cfg["prior"]["cache"] = args.cache
        if args.atlas_dir:
            cfg["prior"]["atlas_dir"] = args.atlas_dir
        print(f"{ck}: prior {cfg['prior'].get('family', 'theory')}, cache {cfg['prior'].get('cache')}, d_x {est.d_x}", flush=True)
        for r in (probe_treatment(est, cfg, args.n_context, args.n_query, args.batches)
                  + probe_families(est, cfg, args.families, args.n_context, args.n_query, args.batches)):
            rows.append({"ckpt": ck, "trained_on": cfg["prior"].get("family", "theory"), **r})
    df = pd.DataFrame(rows)
    cols = ["ckpt", "trained_on", "probe", "n", "rmse", "corr", "mean_abs_pred", "mean_abs_true",
            "n_pos", "n_neg", "sign_balacc"]
    show = df[cols].copy()
    show["ckpt"] = [os.path.basename(os.path.dirname(os.path.abspath(c))) for c in show["ckpt"]]
    print(show.round(3).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print("wrote", args.out)
    return df


if __name__ == "__main__":
    main()
