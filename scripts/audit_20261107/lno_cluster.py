"""Brief 9 task 3 and point 2: held against seen in the leave-networks-out
runs, with the training run as a random effect. Validation pseudo-folds only;
aggregates only (deficit-level means, no patient rows).

Design. Group g (networks 4g-3 .. 4g) is left out of run (g, s) of seed s.
For deficit d of group g(d) and seed s:
    held[d, s] = PEHE of d from run (g(d), s)        (never saw the network)
    seen[d, s] = mean over the three runs (g', s), g' != g(d)
    D[d, s]    = held - seen,  averaged over the two splits (primary) or per split
A run effect u_r (a run scoring every deficit a little better or worse) enters
D[d, s] as +u for the held run and -u/3 for each seen run. Over the 16
deficits of one seed the run effects cancel exactly in the mean of D (each
run is held for 4 deficits and seen, with weight 1/3, for 12), so they do not
bias the contrast; they do inflate the spread of D across deficits, which the
naive paired t treats as noise. The script reports:

  1. naive: paired t over the 16 deficits of the seed-averaged D (the
     dissertation's method; for the single-seed run of 2 Nov it must
     reproduce +0.0028 [-0.0012, +0.0067] on pehe_paper, a pipeline check);
  2. cluster: t over the held-run cluster means (4 per seed, df = runs - 1),
     the conservative reading of "clustered by run";
  3. REML: D = a + Z_run u + Z_def b + e with u ~ N(0, s_u^2) per run (the
     +1 / -1/3 design above), b ~ N(0, s_b^2) per deficit (only with two or
     more seeds: a deficit's own held-seen difference shared across seeds) and
     e ~ N(0, s_e^2); GLS estimate of a, its standard error and a t interval
     with df = runs - 1 (and the normal interval).

    python scripts/audit_20261107/lno_cluster.py --root $V --rep E13a/E13a/seed0 \\
        --label 'lno_g{g}_s{s}' --seeds 0 --atlas-dir ~/export_bcb/atlases --out lno_old.csv
    python scripts/audit_20261107/lno_cluster.py --root $V --rep E13a/E13a/seed0 \\
        --label 'lno3_g{g}_s{s}' --seeds 0 1 2 --atlas-dir ~/export_bcb/atlases --out lno3.csv
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "audit_20261028"))
sys.path.insert(0, os.path.join(HERE, ".."))
from lno_compose import deficit_names  # noqa: E402

SPLITS = ("receptor", "genetics")


def load_run(path, metric, filters):
    df = pd.read_csv(path)
    for k, v in filters.items():
        df = df[df[k].astype(str) == str(v)]
    if df.empty or metric not in df.columns:
        raise ValueError(f"{path}: no {metric} rows")
    return df.groupby("deficit")[metric].mean()


def reml(D, Zs, X=None):
    """REML for D = X a + sum_k Z_k u_k + e; returns (a, se, sigmas)."""
    n = len(D)
    X = np.ones((n, 1)) if X is None else X

    def parts(logs):
        V = np.exp(2 * logs[-1]) * np.eye(n)
        for k, Z in enumerate(Zs):
            V = V + np.exp(2 * logs[k]) * (Z @ Z.T)
        Vi = np.linalg.inv(V)
        XtViX = X.T @ Vi @ X
        a = np.linalg.solve(XtViX, X.T @ Vi @ D)
        r = D - X @ a
        return V, Vi, XtViX, a, r

    def nll(logs):
        V, Vi, XtViX, a, r = parts(logs)
        s1, ld1 = np.linalg.slogdet(V)
        s2, ld2 = np.linalg.slogdet(XtViX)
        return 0.5 * (ld1 + ld2 + float(r @ Vi @ r))

    sd0 = math.log(max(np.std(D), 1e-6))
    best = None
    for start in ([sd0] * (len(Zs) + 1), [sd0 - 2] * len(Zs) + [sd0]):
        res = optimize.minimize(nll, np.array(start, dtype=float), method="Nelder-Mead",
                                options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    V, Vi, XtViX, a, r = parts(best.x)
    se = float(math.sqrt(np.linalg.inv(XtViX)[0, 0]))
    return float(a[0]), se, np.exp(best.x)


def analyse(tab, groups, seeds, tag):
    """tab[(g, s)] = Series deficit -> metric (one split or the split mean)."""
    defs = sorted(groups)
    runs = [(g, s) for s in seeds for g in (1, 2, 3, 4)]
    rid = {r: i for i, r in enumerate(runs)}
    rows, Zr, Zd = [], [], []
    for s in seeds:
        for d in defs:
            g = groups[d]
            held = tab[(g, s)][d]
            seen = np.mean([tab[(h, s)][d] for h in (1, 2, 3, 4) if h != g])
            z = np.zeros(len(runs)); z[rid[(g, s)]] = 1.0
            for h in (1, 2, 3, 4):
                if h != g:
                    z[rid[(h, s)]] = -1.0 / 3
            zd = np.zeros(len(defs)); zd[defs.index(d)] = 1.0
            rows.append({"deficit": d, "seed": s, "group": g, "held": held, "seen": seen, "diff": held - seen})
            Zr.append(z); Zd.append(zd)
    per = pd.DataFrame(rows)
    D = per["diff"].to_numpy()
    out = {"contrast": tag, "seeds": " ".join(map(str, seeds)), "n_runs": len(runs), "n_deficits": len(defs),
           "mean_held": per["held"].mean(), "mean_seen": per["seen"].mean(), "diff_mean": float(D.mean())}

    # 1. naive: paired t over deficits of the seed-averaged difference
    dbar = per.groupby("deficit")["diff"].mean().to_numpy()
    n = len(dbar); se = dbar.std(ddof=1) / math.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    out.update({"naive_ci_low": dbar.mean() - t * se, "naive_ci_high": dbar.mean() + t * se,
                "naive_p": float(stats.ttest_1samp(dbar, 0).pvalue)})

    # 2. held-run cluster means
    cm = per.groupby(["group", "seed"])["diff"].mean().to_numpy()
    k = len(cm); se = cm.std(ddof=1) / math.sqrt(k); t = stats.t.ppf(0.975, k - 1)
    out.update({"cluster_n": k, "cluster_ci_low": cm.mean() - t * se, "cluster_ci_high": cm.mean() + t * se,
                "cluster_p": float(stats.ttest_1samp(cm, 0).pvalue)})

    # 3. REML with the run design (and a deficit effect with >= 2 seeds)
    Zs = [np.array(Zr)] + ([np.array(Zd)] if len(seeds) > 1 else [])
    a, se, sig = reml(D, Zs)
    t = stats.t.ppf(0.975, len(runs) - 1)
    out.update({"reml_est": a, "reml_se": se, "reml_ci_low": a - t * se, "reml_ci_high": a + t * se,
                "reml_ci_low_normal": a - 1.96 * se, "reml_ci_high_normal": a + 1.96 * se,
                "reml_sd_run": float(sig[0]), "reml_sd_deficit": float(sig[1]) if len(seeds) > 1 else float("nan"),
                "reml_sd_resid": float(sig[-1])})
    return out, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="outputs_val_clean folder ($V)")
    ap.add_argument("--rep", default="E13a/E13a/seed0")
    ap.add_argument("--label", required=True, help="run label with {g} and {s}, e.g. lno3_g{g}_s{s}")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--metrics", nargs="+", default=["pehe_paper", "prescriptive_balacc"])
    ap.add_argument("--filter", nargs="*", default=[], help="column=value filters on replica_results rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    names = deficit_names(a.atlas_dir)
    groups = {names[i]: (i - 1) // 4 + 1 for i in range(1, 17)}
    filters = dict(f.split("=", 1) for f in a.filter)
    results, pers = [], []
    for metric in a.metrics:
        tabs = {m: {} for m in SPLITS}
        for s in a.seeds:
            for g in (1, 2, 3, 4):
                lab = a.label.format(g=g, s=s)
                for m in SPLITS:
                    p = os.path.join(a.root, lab, m, "replica", a.rep, "disconnectome-extrasclean", "replica_results.csv")
                    tabs[m][(g, s)] = load_run(p, metric, filters)
        both = {r: (tabs["receptor"][r] + tabs["genetics"][r]) / 2 for r in tabs["receptor"]}
        for tag, tab in (("both", both), ("receptor", tabs["receptor"]), ("genetics", tabs["genetics"])):
            tab = {r: v.dropna() for r, v in tab.items()}
            common = set(groups)
            for v in tab.values():
                common &= set(v.index)
            if len(common) < 16:
                print(f"{metric} {tag}: {16 - len(common)} deficit(s) undefined in some run, left out")
            g2 = {d: groups[d] for d in common}
            out, per = analyse(tab, g2, a.seeds, tag)
            out["metric"] = metric
            per["metric"], per["split"] = metric, tag
            results.append(out); pers.append(per)
            print(f"{metric:>20} {tag:>8}: held-seen {out['diff_mean']:+.4f} | naive [{out['naive_ci_low']:+.4f}, "
                  f"{out['naive_ci_high']:+.4f}] | cluster(n={out['cluster_n']}) [{out['cluster_ci_low']:+.4f}, "
                  f"{out['cluster_ci_high']:+.4f}] | REML {out['reml_est']:+.4f} se {out['reml_se']:.4f} "
                  f"[{out['reml_ci_low']:+.4f}, {out['reml_ci_high']:+.4f}] sd_run {out['reml_sd_run']:.4f}")
    pd.DataFrame(results).to_csv(a.out, index=False)
    pd.concat(pers).to_csv(a.out.replace(".csv", "_per_deficit.csv"), index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
