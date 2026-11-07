"""Brief 9 task 1 (item 0a, CPU): what the final prior's virtual-trial family
actually trains on. Run under the final model's launch environment
(g4c_launch.sh builds the cfg and execs this script):

    PRIOR_CHECK_ONLY=1 PRIOR_CHECK_SCRIPT=scripts/audit_20261107/allocation_atoms_check.py \\
        PRIOR_CHECK_OUT=<dir> OBJECTIVE=new FAMILY_W="0 1" CACHE=<E13a KCH pool> \\
        bash scripts/audit_20261028/g4c_launch.sh

Every attempt of NeuroPriorCohort._one is logged through two wrappers that
change nothing the cohort draws: the process of the attempt (from
_draw_process) and the verifier's verdict (the module's
verify_identifiability, called with the same arguments).

A. Allocation types before and after the verifier, by strength. Before: every
   process the cohort attempts (each attempt redraws the process). After: the
   processes _one returns. Pass rates by type and by a bin of the drawn
   strength BIAS (0 to 1 in tenths). For 'agnostic' the confounding strength
   is |BIAS - 0.5| (BIAS < 0.5 allocates away from the better arm), for 'axis'
   and 'severity' BIAS is the slope of the allocation ramp. Only 'agnostic'
   is tested with U (the true side); the others for positivity only. For the
   agnostic attempts the verifier's statistic (mean |p(X,U) - p(X)| of two
   in-sample logistic fits) is also recomputed, to show where the tolerance
   1e-3 sits against zero strength (Brief 9 point 3, calibration).
   Also: share of returned processes that failed every attempt (the cohort
   returns the last draw after max_tries).
B. Cross-split near-copies over the drawn virtual networks: the share whose
   two sides hold halves of one network from different splits, and the share
   that sets a whole network against one of its own halves.
C. R^2 (5-fold ridge) of the lesion centroid (each axis) and the log volume on
   the latents.
D. Distinct context rows (Brief 9 point 6), by context-length bin: distinct
   rows of each attempted context against its length, the slice size, and the
   count expected for draws with replacement from the slice minus the queries,
   R (1 - (1 - 1/R)^n). Also the number of distinct latent rows of the pool.

Aggregates only: no lesion, row or file name is printed or written.
"""
import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

TYPES = ("axis", "severity", "agnostic")
CTX_BINS = ((256, 512), (512, 1024), (1024, 2048), (2048, 3001))


def base_cohort(prior):
    """The NeuroPriorCohort under an optional GenericEffectMixture."""
    return getattr(prior, "base", prior)


def classify_network(desc) -> set:
    out = set()
    for ca, ka, ha in desc["A"]:
        for cb, kb, hb in desc["B"]:
            if ka != kb:
                continue
            if ha in (0, 1) and hb in (0, 1) and ca != cb:
                out.add("cross_split_same_network")
            if (ha == -1) != (hb == -1):
                out.add("whole_vs_own_half")
    return out


def r2_cv(X, y, folds=5, alpha=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict

    pred = cross_val_predict(Ridge(alpha=alpha), X, y, cv=folds)
    return float(1.0 - np.mean((y - pred) ** 2) / np.var(y))


def verifier_stat(W, X, U):
    """The statistic verify_identifiability compares with tol (same fits)."""
    from sklearn.linear_model import LogisticRegression

    W = np.asarray(W).ravel()
    U = np.asarray(U).reshape(len(W), -1)
    base = LogisticRegression(max_iter=500).fit(X, W).predict_proba(X)[:, 1]
    aug = LogisticRegression(max_iter=500).fit(np.c_[X, U], W).predict_proba(np.c_[X, U])[:, 1]
    return float(np.mean(np.abs(aug - base)))


def instrument(coh, n_query, stat_every=1):
    """Wrap the cohort's process draw and the verifier; returns the log."""
    import neurocausalpfn.prior.cohort as cm

    state = {"cur": None, "n_agn": 0}
    log = []
    orig_draw, orig_verify, orig_one = coh._draw_process, cm.verify_identifiability, coh._one

    def draw():
        r = orig_draw()
        state["cur"] = r
        return r

    def verify(W, X, Y0=None, Y1=None, U=None, tol=1e-3):
        ok = orig_verify(W, X, Y0, Y1, U=U, tol=tol)
        fam, dgp, sl = state["cur"]
        n = len(np.asarray(W).ravel())
        rec = {"family": fam, "type": getattr(dgp, "bias_type", None) if fam == "giles" else None,
               "bias": float(getattr(dgp, "bias", np.nan)), "passed": bool(ok), "n_ctx": n,
               "n_slice": int(coh.m if sl is None else len(sl)),
               "distinct": int(len(np.unique(np.round(np.asarray(X), 6), axis=0))),
               "treated": float(np.mean(W)), "stat": np.nan, "returned": False}
        if U is not None:
            state["n_agn"] += 1
            if state["n_agn"] % stat_every == 0:
                rec["stat"] = verifier_stat(W, np.asarray(X), U)
        log.append(rec)
        return ok

    def one(n_context, drawn=None):
        start = len(log)
        data = orig_one(n_context, drawn)
        if len(log) > start:
            log[-1]["returned"] = True
            log[-1]["fallback"] = not log[-1]["passed"]
        return data

    coh._draw_process = draw
    coh._one = one
    cm.verify_identifiability = verify
    return log


def bin_of(b):
    return min(int(b * 10), 9) / 10 if np.isfinite(b) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=5000, help="virtual-trial processes drawn for B")
    ap.add_argument("--batches", type=int, default=400, help="batches for A and D")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--ctx-min", type=int, default=256)
    ap.add_argument("--ctx-max", type=int, default=3000)
    ap.add_argument("--stat-every", type=int, default=1, help="recompute the verifier statistic on every k-th agnostic attempt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from neurocausalpfn.train.launch_cfg import cfg_from_env
    from neurocausalpfn.train.train_pfn import _build_prior

    cfg = cfg_from_env()
    prior, _, _ = _build_prior(cfg)
    coh = base_cohort(prior)
    os.makedirs(a.out, exist_ok=True)
    rows = []

    hg = coh.hyper if coh.family == "giles" else coh.hyper_giles
    rows.append(("hyper_bias_range", str(tuple(hg.bias))))
    rows.append(("hyper_bias_weights", str(tuple(hg.bias_weights))))
    rows.append(("family_weights_theory_trial", str(tuple(np.round(coh.family_weights, 3)))))
    rows.append(("context_replace", int(coh.context_replace)))
    rows.append(("max_tries", int(coh.max_tries)))
    rows.append(("pool_rows", int(coh.m)))
    rows.append(("pool_distinct_latent_rows", int(len(np.unique(np.round(coh.X, 6), axis=0)))))

    # B (and the prior's draw shares before any verifier, as a cross-check)
    prior_draw, nets, kinds = Counter(), 0, Counter()
    for _ in range(a.draws):
        dgp = coh.sample_process("giles")
        prior_draw[dgp.bias_type] += 1
        for d in getattr(dgp, "virtual_desc", []) or []:
            nets += 1
            for k in classify_network(d):
                kinds[k] += 1
    for t in TYPES:
        rows.append(("A_sample_process_share_" + t, prior_draw[t] / max(1, sum(prior_draw.values()))))
    rows.append(("B_virtual_networks", nets))
    for k in ("cross_split_same_network", "whole_vs_own_half"):
        rows.append(("B_share_" + k, kinds[k] / max(1, nets)))

    # A and D: batches at log-uniform context lengths (the DISTINCT curriculum;
    # the final model's linear curriculum covers the same range)
    log = instrument(coh, int(cfg["pfn"]["n_query"]), a.stat_every)
    rng = np.random.default_rng(a.seed)
    lo, hi = math.log(a.ctx_min), math.log(a.ctx_max)
    for _ in range(a.batches):
        n_ctx = int(round(math.exp(rng.uniform(lo, hi))))
        prior.sample_batch(a.batch_size, n_context=n_ctx)
    giles = [r for r in log if r["family"] == "giles"]
    att = Counter(r["type"] for r in giles)
    ret = Counter(r["type"] for r in giles if r["returned"])
    for t in TYPES:
        rows.append(("A_before_attempt_share_" + t, att[t] / max(1, sum(att.values()))))
        rows.append(("A_after_returned_share_" + t, ret[t] / max(1, sum(ret.values()))))
        sel = [r for r in giles if r["type"] == t]
        rows.append(("A_pass_rate_" + t, float(np.mean([r["passed"] for r in sel])) if sel else float("nan")))
    rows.append(("A_attempts", len(log)))
    rows.append(("A_returned", sum(r["returned"] for r in log)))
    rows.append(("A_returned_after_failing_every_attempt", sum(r.get("fallback", False) for r in log)))
    agn = [r for r in giles if r["type"] == "agnostic" and r["returned"] and r["passed"]]
    rows.append(("A_after_agnostic_strength_median", float(np.median([abs(r["bias"] - 0.5) for r in agn])) if agn else float("nan")))

    by_bin = []
    for t in TYPES:
        for b in [x / 10 for x in range(10)]:
            sel = [r for r in giles if r["type"] == t and bin_of(r["bias"]) == b]
            st = [r["stat"] for r in sel if np.isfinite(r["stat"])]
            by_bin.append({"type": t, "bias_bin": f"{b:.1f}-{b + 0.1:.1f}", "attempts": len(sel),
                           "passed": int(sum(r["passed"] for r in sel)),
                           "pass_rate": round(float(np.mean([r["passed"] for r in sel])), 4) if sel else "",
                           "returned": int(sum(r["returned"] and r["passed"] for r in sel)),
                           "treated_share_median": round(float(np.median([r["treated"] for r in sel])), 4) if sel else "",
                           "verifier_stat_median": round(float(np.median(st)), 5) if st else "",
                           "verifier_stat_p10": round(float(np.percentile(st, 10)), 5) if st else "",
                           "verifier_stat_p90": round(float(np.percentile(st, 90)), 5) if st else ""})
    with open(os.path.join(a.out, "allocation_by_type_and_strength.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_bin[0]))
        w.writeheader()
        w.writerows(by_bin)

    # D
    nq = int(cfg["pfn"]["n_query"])
    ctx_rows = []
    for lo_c, hi_c in CTX_BINS:
        sel = [r for r in log if lo_c <= r["n_ctx"] < hi_c]
        if not sel:
            continue
        share = [r["distinct"] / r["n_ctx"] for r in sel]
        exp = []
        for r in sel:
            R = max(r["n_slice"] - nq, 1)
            e = R * (1 - (1 - 1 / R) ** r["n_ctx"]) if coh.context_replace else min(R, r["n_ctx"])
            exp.append(e / r["n_ctx"])
        ctx_rows.append({"ctx_bin": f"{lo_c}-{hi_c - 1}", "contexts": len(sel),
                         "slice_median": float(np.median([r["n_slice"] for r in sel])),
                         "slice_p10": float(np.percentile([r["n_slice"] for r in sel], 10)),
                         "slice_p90": float(np.percentile([r["n_slice"] for r in sel], 90)),
                         "distinct_median": float(np.median([r["distinct"] for r in sel])),
                         "distinct_share_median": round(float(np.median(share)), 4),
                         "expected_share_median": round(float(np.median(exp)), 4),
                         "share_ctx_longer_than_slice": round(float(np.mean([r["n_ctx"] > r["n_slice"] - nq for r in sel])), 4)})
    with open(os.path.join(a.out, "distinct_context_rows_by_length.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ctx_rows[0]))
        w.writeheader()
        w.writerows(ctx_rows)

    # C
    X = np.asarray(coh.X, dtype=float)
    for i, ax in enumerate("xyz"):
        rows.append((f"C_r2_centroid_{ax}_on_latents", r2_cv(X, np.asarray(coh.centroids)[:, i])))
    vol = np.asarray(coh.volumes, dtype=float)
    if np.isfinite(vol).all():
        rows.append(("C_r2_log_volume_on_latents", r2_cv(X, np.log1p(vol))))

    with open(os.path.join(a.out, "allocation_atoms_check.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "value"])
        w.writerows(rows)
    for k, v in rows:
        print(f"{k}: {v}")
    for r in by_bin:
        print(r)
    for r in ctx_rows:
        print(r)


if __name__ == "__main__":
    main()
