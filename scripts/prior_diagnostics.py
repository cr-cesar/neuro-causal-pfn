"""Diagnostics of the extended Neuro-Prior on a real anatomy cache (F34).

For N processes of the theory family drawn on the cache, reports over the
pool lesions and over the process's slice (the lesions it can affect):

- the fraction of lesions whose expected outcome is clipped at 1 under arm A
  (mu0 + effect > 1), under arm B, and under either;
- the effect lost to the clipping: mean |CATE| before and after clipping and
  the relative loss;
- the same by tertile of D (the first tertile holds the lesions that barely
  touch the networks, where mu0 is close to 1 and the clipping bites).

Aggregates only; nothing per patient is written.

    python scripts/prior_diagnostics.py --cache outputs/prior_cache/kch_e1_32d.npz --atlas-dir <atlases> \\
        --hyper "two_arms=1 effect_scale=0.2:1" --processes 200 --out outputs/prior_diag_2arms_scale.csv
    python scripts/prior_diagnostics.py --ckpt outputs/pfn_reduced_<run>/pfn.pt   # the checkpoint's own prior
"""
import argparse
import copy
import os
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _cfg_from_args(args) -> Dict:
    from neurocausalpfn.train.train_pfn import parse_hyper_opts

    if args.ckpt:
        import torch

        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = copy.deepcopy(ck["cfg"])
    else:
        if not (args.cache and args.atlas_dir):
            raise SystemExit("give --ckpt, or --cache and --atlas-dir")
        cfg = {"seed": 0, "prior": {"kind": "neuro_prior", "family": "theory"},
               "pfn": {"context_max": 1024, "n_query": 64}}
    cfg["prior"]["family"] = "theory"
    cfg["prior"].pop("stages", None)
    if args.cache:
        cfg["prior"]["cache"] = args.cache
    if args.atlas_dir:
        cfg["prior"]["atlas_dir"] = args.atlas_dir
    if args.hyper is not None:
        cfg["prior"]["hyper"] = parse_hyper_opts(args.hyper)
    cfg["pfn"]["n_query"] = 8
    return cfg


def _stats(mu0, raw_a, raw_b, mu_a, mu_b, mask) -> Dict[str, float]:
    """Clipping and effect-loss figures over the lesions in ``mask``."""
    if not mask.any():
        return {"n": 0}
    cate_raw = (raw_a - raw_b)[mask]
    cate = (mu_a - mu_b)[mask]
    abs_raw, abs_clip = float(np.mean(np.abs(cate_raw))), float(np.mean(np.abs(cate)))
    return {"n": int(mask.sum()),
            "frac_mu0_above_0.9": float(np.mean(mu0[mask] > 0.9)),
            "frac_clip_A": float(np.mean(raw_a[mask] > 1.0)),
            "frac_clip_B": float(np.mean(raw_b[mask] > 1.0)),
            "frac_clip_any": float(np.mean((raw_a[mask] > 1.0) | (raw_b[mask] > 1.0))),
            "mean_abs_cate_raw": abs_raw, "mean_abs_cate": abs_clip,
            "effect_lost_rel": float(1.0 - abs_clip / abs_raw) if abs_raw > 1e-9 else 0.0}


def process_diagnostics(dgp, overlaps: np.ndarray, slice_idx) -> Dict[str, float]:
    ov = np.asarray(overlaps, dtype=np.float64)
    D, S = dgp.scores(ov)
    Sb = dgp.scores_b(ov)
    mu0 = dgp.mu0(D)
    f = float(dgp.effect_factor) * float(dgp.beta)
    raw_a = mu0 + f * S * D                                   # before the clip at 1
    raw_b = mu0 + f * Sb * D if dgp.two_arms else mu0
    mu_a, mu_b = dgp.mu1(D, S), dgp.mu_b(D, Sb)               # as the prior uses them
    out = {"two_arms": int(dgp.two_arms), "beta": float(dgp.beta), "alpha": float(dgp.alpha),
           "effect_factor": float(dgp.effect_factor), "mechanism": dgp.mechanism}
    allm = np.ones(len(D), bool)
    sl = np.zeros(len(D), bool)
    if slice_idx is None:
        sl[:] = True
    else:
        sl[np.asarray(slice_idx)] = True
    for name, m in (("pool", allm), ("slice", sl)):
        for k, v in _stats(mu0, raw_a, raw_b, mu_a, mu_b, m).items():
            out[f"{name}_{k}"] = v
    # by tertile of D inside the slice (lesions barely touching the networks first)
    if sl.sum() >= 3:
        q = np.quantile(D[sl], [1 / 3, 2 / 3])
        tert = np.digitize(D, q)                              # 0, 1, 2
        for t in range(3):
            for k, v in _stats(mu0, raw_a, raw_b, mu_a, mu_b, sl & (tert == t)).items():
                if k in ("frac_clip_any", "effect_lost_rel", "n", "frac_mu0_above_0.9"):
                    out[f"slice_D_tertile{t + 1}_{k}"] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--atlas-dir", default=None)
    ap.add_argument("--hyper", default=None, help='e.g. "two_arms=1 effect_scale=0.2:1"')
    ap.add_argument("--processes", type=int, default=200)
    ap.add_argument("--seed-offset", type=int, default=30_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import pandas as pd

    from neurocausalpfn.train.train_pfn import _build_prior

    cfg = _cfg_from_args(args)
    prior, _, _ = _build_prior(cfg, seed_offset=args.seed_offset)
    rows = []
    for _ in range(args.processes):
        fam, dgp, sl = prior._draw_process()
        rows.append(process_diagnostics(dgp, prior._theory_ov(), sl))
    df = pd.DataFrame(rows)
    print(f"hyper: {cfg['prior'].get('hyper', {})}; cache {cfg['prior'].get('cache')}; {len(df)} processes, "
          f"pool {len(prior.overlaps)} lesions, two-arm share {df['two_arms'].mean():.2f}")
    num = df.select_dtypes("number")
    summ = pd.DataFrame({"mean": num.mean(), "median": num.median(), "p90": num.quantile(0.9)})
    keep = [c for c in summ.index if any(k in c for k in ("frac_clip", "effect_lost", "mean_abs_cate", "mu0_above", "effect_factor"))]
    print(summ.loc[keep].round(3).to_string())
    if args.out:
        df.to_csv(args.out, index=False)
        summ.to_csv(args.out.replace(".csv", "_summary.csv"))
        print("wrote", args.out)
    return df


if __name__ == "__main__":
    main()
