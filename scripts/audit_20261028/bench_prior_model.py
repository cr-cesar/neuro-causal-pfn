"""Checks 1-3 of the 28 Oct audit (read-only; CPU; aggregates only).

1. effect distribution and outcome type of the r7 and thr03 priors
2. slope of predicted on true CATE of the r7 fold-0 checkpoint, on its own prior
   and on the giles family with TE=1, RE=0 (pool vs context standardisation)
3. per-dimension mean/sd of the context slices after the pool standardisation

Environment: BENCH_CKPT (checkpoint), BENCH_CACHE (prior cache), BENCH_ATLAS (atlas dir),
BENCH_GILES_ONLY=1 (check 2 on the giles family only), BENCH_CONTEXT_REPLACE=0 (distinct
context rows; the default 1 draws them with replacement, as in the 28 Oct runs).
"""
import argparse, copy, json, logging, os, sys, time
import numpy as np

H = os.path.expanduser
CKPT = os.environ.get("BENCH_CKPT") or H("~/p2_bcb/pfn/kch-s0/fold0/pfn.pt")
CACHE = os.environ.get("BENCH_CACHE") or H("~/p2_bcb/pools/kch_s0/fold0.npz")
ATLAS = os.environ.get("BENCH_ATLAS") or H("~/neuro-causal-pfn/data/atlases")
CONTEXT_REPLACE = os.environ.get("BENCH_CONTEXT_REPLACE", "1") != "0"   # 0: distinct context rows (no replacement)
logging.disable(logging.INFO)

import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
from neurocausalpfn.train.train_pfn import _build_prior, build_model, parse_hyper_opts, _as_ranges
from neurocausalpfn.prior.neuro_prior import HyperPrior
from neurocausalpfn.prior.cohort import NeuroPriorCohort
from neurocausalpfn.prior.atlas import FunctionalAtlas
from neurocausalpfn.pfn.inference import predict_cate
from neurocausalpfn.prior.giles_replica import _fit_predict_two_model

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
CFG = ck["cfg"]


def r7_cfg(seed):
    c = copy.deepcopy(CFG); c["seed"] = seed
    c["prior"]["atlas_dir"] = ATLAS; c["prior"]["cache"] = CACHE
    return c


def thr03_cfg(seed):
    c = r7_cfg(seed); c["prior"]["family"] = "mixture"; c["prior"]["family_weights"] = [0.5, 0.5]
    c["prior"]["hyper_giles"] = parse_hyper_opts("random_partitions=1 threshold=0.03:0.08 p_te=0.5:1")
    return c


def is_binary(y):
    return bool(np.all((y == 0) | (y == 1)))


def fam_of(p):
    return p.get("family", "?") if isinstance(p, dict) else "?"


def check1(n_proc, bs=50, n_ctx=256):
    out = {}
    for name, mk in (("r7", r7_cfg), ("thr03", thr03_cfg)):
        prior, _, _ = _build_prior(mk(1234))
        taus, fams, binr, tau_fam = [], [], [], {}
        t0 = time.time()
        for _ in range(n_proc // bs):
            b = prior.sample_batch(bs, n_context=n_ctx)
            for i in range(bs):
                t = b["mu1"][i] - b["mu0"][i]; f = fam_of(b["processes"][i])
                taus.append(t); fams.append(f); binr.append(is_binary(b["Yc"][i]))
                tau_fam.setdefault(f, []).append(t)
        T = np.concatenate(taus); A = np.abs(T); nz = A > 1e-6
        edges = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0001]
        hist = np.histogram(A[nz], bins=edges)[0]
        fams = np.array(fams); binr = np.array(binr)
        res = {"n_proc": len(fams), "n_query": int(len(T)), "frac_query_tau_nonzero": float(nz.mean()),
               "hist_abs_tau_nonzero": {f"{edges[k]:.2f}-{min(edges[k+1],1):.2f}": round(float(h / nz.sum()), 4) for k, h in enumerate(hist)},
               "frac_gt0.9_of_nonzero": float((A[nz] > 0.9).mean()), "frac_gt0.9_of_all": float((A > 0.9).mean()),
               "median_abs_tau_nonzero": float(np.median(A[nz])), "p95_abs_tau_nonzero": float(np.quantile(A[nz], .95)),
               "frac_ctx_binary_Y": float(binr.mean()),
               "by_family": {}, "secs": round(time.time() - t0)}
        for f in sorted(set(fams)):
            tf = np.abs(np.concatenate(tau_fam[f])); z = tf > 1e-6
            res["by_family"][f] = {"share_proc": float((fams == f).mean()), "frac_binary": float(binr[fams == f].mean()),
                                   "frac_nonzero": float(z.mean()), "frac_gt0.9_of_nonzero": float((tf[z] > .9).mean()) if z.any() else None,
                                   "median_abs_nonzero": float(np.median(tf[z])) if z.any() else None}
        out[name] = res
        print(name, json.dumps(res, indent=1), flush=True)
    return out


def load_model():
    m = build_model(CFG, int(ck["state_dict"]["ctx_emb.weight"].shape[1]) - 2)
    m.load_state_dict(ck["state_dict"]); m.eval(); return m


def predict(model, b, restd=False):
    Xc, Xq = b["Xc"].astype(np.float32), b["Xq"].astype(np.float32)
    if restd:   # what PFNEstimator does at scoring time: standardise with the context statistics
        mu, sd = Xc.mean(1, keepdims=True), Xc.std(1, keepdims=True) + 1e-6
        Xc, Xq = (Xc - mu) / sd, (Xq - mu) / sd
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))
    with torch.no_grad():
        o = predict_cate(model, t(Xc), t(b["Tc"]), t(b["Yc"]), t(Xq))
    return o["cate"].numpy()


def slope(x, y):
    x = np.asarray(x); y = np.asarray(y)
    vx = x.var()
    return float(np.cov(x, y, bias=True)[0, 1] / vx) if vx > 1e-12 else float("nan")


def summarize(true, pred, tag):
    true = np.concatenate(true); pred = np.concatenate(pred)
    r = {"n_query": int(len(true)), "slope_pred_on_true": round(slope(true, pred), 3),
         "calib_slope_true_on_pred": round(slope(pred, true), 3),
         "corr": round(float(np.corrcoef(true, pred)[0, 1]), 3), "pehe": round(float(np.sqrt(np.mean((true - pred) ** 2))), 3)}
    for lv in (1.0, -1.0, 0.0):
        k = np.abs(true - lv) < 1e-6
        if k.sum() >= 20:
            r[f"mean_pred_at_true{lv:+.0f}"] = round(float(pred[k].mean()), 3); r[f"n_true{lv:+.0f}"] = int(k.sum())
    big = np.abs(true) > 0.5
    if big.sum() >= 20:
        r["slope_pred_on_true_|tau|>0.5"] = round(float(np.mean(pred[big] / true[big])), 3)
    print(tag, json.dumps(r), flush=True)
    return r


def giles_cohort(seed, n_ctx):
    atlas = FunctionalAtlas.from_dir(ATLAS, shape=tuple(CFG["prior"].get("atlas_shape", [48, 56, 48])), seed=seed,
                                     modality=CFG["prior"].get("modality", "receptor"))
    h = HyperPrior(p_te=(1.0, 1.0), p_re=(0.0, 0.0), bias=(0.0, 0.0), label_noise=(0.0, 0.0),
                   n_networks=(1, 1), threshold=(0.05, 0.05))
    return NeuroPriorCohort.from_cache(atlas, CACHE, seed=seed, n_context=n_ctx, n_query=64, hyper=h,
                                       augment=True, family="giles", context_pool="susceptible",
                                       context_replace=CONTEXT_REPLACE, standardize="pool")


def check2(n_proc, ctx_list, bs=8):
    model = load_model(); out = {}
    prior, _, _ = _build_prior(r7_cfg(4321))
    for n_ctx in ([] if os.environ.get("BENCH_GILES_ONLY") else ctx_list):
        acc = {}
        for _ in range(n_proc // bs):
            b = prior.sample_batch(bs, n_context=n_ctx); p = predict(model, b)
            for i in range(bs):
                f = fam_of(b["processes"][i]); bn = "bin" if is_binary(b["Yc"][i]) else "cont"
                for key in ("all", f, f"Y{bn}"):
                    acc.setdefault(key, ([], []))
                    acc[key][0].append(b["mu1"][i] - b["mu0"][i]); acc[key][1].append(p[i])
        out[f"r7prior_ctx{n_ctx}"] = {k: summarize(*v, f"r7prior ctx={n_ctx} [{k}]") for k, v in acc.items()}
    for n_ctx in ctx_list:
        g = giles_cohort(9876, n_ctx); acc = {"pool": ([], []), "context": ([], []), "LR_two": ([], [])}
        # distinct mode: the cohort tops a short slice up with replacement, so a
        # context with a repeated row is dropped and another process drawn
        # (T7, 28 Oct); the share kept says how often the slice was long enough
        n_seen = n_kept = 0
        for _ in range(n_proc // bs if CONTEXT_REPLACE else 50 * (n_proc // bs)):
            if not CONTEXT_REPLACE and n_kept >= n_proc:
                break
            b = g.sample_batch(bs, n_context=n_ctx)
            keep = np.ones(bs, dtype=bool)
            if not CONTEXT_REPLACE:
                keep = np.array([len(np.unique(b["Xc"][i], axis=0)) == n_ctx for i in range(bs)])
            n_seen += bs; n_kept += int(keep.sum())
            if not keep.any():
                continue
            tr = b["mu1"] - b["mu0"]
            for mode, restd in (("pool", False), ("context", True)):
                p = predict(model, b, restd=restd)
                acc[mode][0].extend(list(tr[keep])); acc[mode][1].extend(list(p[keep]))
            for i in np.flatnonzero(keep):   # reference: the replica's LR two-model on the same context
                p1, p0 = _fit_predict_two_model(b["Xc"][i], b["Tc"][i].astype(int), b["Yc"][i].astype(int), b["Xq"][i], "logistic_regression")
                acc["LR_two"][0].append(tr[i]); acc["LR_two"][1].append(p1 - p0)
        if not CONTEXT_REPLACE:
            print(f"giles ctx={n_ctx} distinct contexts: {n_kept} of {n_seen} drawn processes kept", flush=True)
        if not acc["LR_two"][0]:
            continue
        out[f"giles_TE1_RE0_ctx{n_ctx}"] = {k: summarize(*v, f"giles TE=1 RE=0 ctx={n_ctx} std={k}") for k, v in acc.items()}
        out[f"giles_TE1_RE0_ctx{n_ctx}"]["kept"] = {"processes_kept": n_kept, "processes_drawn": n_seen}
    return out


def check3(n_proc):
    """Context slices drawn as in training; their rows after the pool standardisation."""
    out = {}
    for name, mk in (("r7", r7_cfg), ("thr03", thr03_cfg)):
        prior, _, _ = _build_prior(mk(555)); base = getattr(prior, "base", prior)
        X = base.X  # already pool-standardised
        nbad_m, nbad_s, nbad_any, mus, sds, sizes = [], [], [], [], [], []
        for _ in range(n_proc):
            fam, dgp, sl = base._draw_process()
            Xc = X[sl]
            m, s = Xc.mean(0), Xc.std(0)
            nbad_m.append(int((np.abs(m) > 0.2).sum())); nbad_s.append(int((s < 0.8).sum()))
            nbad_any.append(int(((np.abs(m) > 0.2) | (s < 0.8)).sum()))
            mus.append(np.abs(m).mean()); sds.append(s.mean()); sizes.append(len(sl))
        q = lambda v: [round(float(x), 2) for x in np.quantile(v, [.1, .5, .9])]
        r = {"d_x": int(X.shape[1]), "n_slices": n_proc, "pool_check_mean_absmean": round(float(np.abs(X.mean(0)).mean()), 4),
             "slice_size_q10_50_90": q(sizes),
             "dims_|mean|>0.2_q10_50_90": q(nbad_m), "dims_sd<0.8_q10_50_90": q(nbad_s), "dims_either_q10_50_90": q(nbad_any),
             "mean_dims_either": round(float(np.mean(nbad_any)), 1),
             "avg_|mean|_q10_50_90": q(mus), "avg_sd_q10_50_90": q(sds),
             "frac_slices_with_>=10_bad_dims": round(float(np.mean(np.array(nbad_any) >= 10)), 3)}
        out[name] = r; print("check3", name, json.dumps(r), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", default="1,2,3"); ap.add_argument("--n1", type=int, default=2000)
    ap.add_argument("--n2", type=int, default=256); ap.add_argument("--ctx2", default="1024")
    ap.add_argument("--n3", type=int, default=400); ap.add_argument("--out", default=H("~/p2_bcb/audit/results.json"))
    a = ap.parse_args(); res = {}
    if "1" in a.checks: res["check1"] = check1(a.n1)
    if "3" in a.checks: res["check3"] = check3(a.n3)
    if "2" in a.checks: res["check2"] = check2(a.n2, [int(x) for x in a.ctx2.split(",")])
    json.dump(res, open(a.out, "w"), indent=1)
