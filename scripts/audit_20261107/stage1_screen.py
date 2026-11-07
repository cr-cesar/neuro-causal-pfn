"""Brief 9 task 6: the screening table of the stage 1 sweep, validation only.

Per level against R0 (final1M seeds unless --r0 says otherwise), on the
16 deficits, each deficit's PEHE averaged over folds and seeds:
  gain = R0 - level (positive = the level is better), per split and on the
  mean of splits; paired t over the deficits on the mean of splits; Holm over
  the levels screened. The screening rule of the protocol (Brief 9):
    pass = gain >= 0.005 on the mean of splits, same sign on both splits, and
           Holm-adjusted p < 0.05 (the interval excludes zero after Holm);
  0.02 stays the bar for claims.
With every factorial level present (F2-F8; F1 = R0), the main effects of
A (generic share), B (ranges), C (CATE_W) and D (learning rate), each from 12
runs against 12, with Holm over the four. Generator D = ABC: main effects are
aliased with three-factor interactions only; two-factor interactions are
aliased in pairs (AB = CD, AC = BD, AD = BC) and are reported as such.
Aggregates only.
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

SPLITS = ("receptor", "genetics")
SIGNS = {"F1": (-1, -1, -1, -1), "F2": (1, -1, -1, 1), "F3": (-1, 1, -1, 1), "F4": (1, 1, -1, -1),
         "F5": (-1, -1, 1, 1), "F6": (1, -1, 1, -1), "F7": (-1, 1, 1, -1), "F8": (1, 1, 1, 1)}
FACTORS = ("A_generic_0.75", "B_ranges_wide", "C_cate_w_0", "D_cosine")


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def level_table(root, rep, pattern, seeds, metric):
    """deficit x split means over folds and seeds."""
    out = {}
    for m in SPLITS:
        per = []
        for s in seeds:
            p = os.path.join(root, pattern.format(s=s), m, "replica", rep, "disconnectome-extrasclean", "replica_results.csv")
            per.append(pd.read_csv(p).groupby("deficit")[metric].mean())
        out[m] = pd.concat(per, axis=1).mean(axis=1)
    return pd.DataFrame(out)


def paired(gain):
    gain = gain.dropna().to_numpy()
    n = len(gain)
    se = gain.std(ddof=1) / math.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return float(gain.mean()), float(gain.mean() - t * se), float(gain.mean() + t * se), \
        float(stats.ttest_1samp(gain, 0).pvalue), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/neuro-causal-pfn-perfold/outputs_val_clean"))
    ap.add_argument("--rep", default="E13a/E13a/seed0")
    ap.add_argument("--r0", default="final1M_s{s}")
    ap.add_argument("--levels", nargs="+", required=True, help="NAME=label pattern with {s}")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--metric", default="pehe_paper")
    ap.add_argument("--screen", type=float, default=0.005)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lower_is_better = a.metric.startswith("pehe")
    r0 = level_table(a.root, a.rep, a.r0, a.seeds, a.metric)
    tabs = {"F1": r0}
    rows = []
    for lv in a.levels:
        name, pat = lv.split("=", 1)
        tab = level_table(a.root, a.rep, pat, a.seeds, a.metric)
        tabs[name] = tab
        g = (r0 - tab) if lower_is_better else (tab - r0)
        mean, lo, hi, p, n = paired(g.mean(axis=1))
        rows.append({"level": name, "n_deficits": n, "r0_mean": float(r0.mean(axis=1).mean()),
                     "level_mean": float(tab.mean(axis=1).mean()), "gain_receptor": float(g["receptor"].mean()),
                     "gain_genetics": float(g["genetics"].mean()), "gain_mean": mean, "ci_low": lo, "ci_high": hi,
                     "p": p, "n_deficits_better": int((g.mean(axis=1) > 0).sum())})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p"])
    df["same_sign_both_splits"] = np.sign(df["gain_receptor"]) == np.sign(df["gain_genetics"])
    df["passes_screen"] = (df["gain_mean"] >= a.screen) & df["same_sign_both_splits"] & (df["p_holm"] < 0.05)
    df["meets_claim_bar_0.02"] = df["passes_screen"] & (df["gain_mean"] >= 0.02)
    df.to_csv(a.out, index=False)
    print(df.round(4).to_string(index=False))

    if all(f in tabs for f in SIGNS):
        sm = {f: tabs[f].mean(axis=1) for f in SIGNS}
        eff = []
        contrasts = list(FACTORS) + ["AB=CD", "AC=BD", "AD=BC"]
        for ci, cname in enumerate(contrasts):
            if ci < 4:
                w = {f: SIGNS[f][ci] for f in SIGNS}
            else:
                i, j = {"AB=CD": (0, 1), "AC=BD": (0, 2), "AD=BC": (0, 3)}[cname]
                w = {f: SIGNS[f][i] * SIGNS[f][j] for f in SIGNS}
            plus = sum(sm[f] for f in SIGNS if w[f] > 0) / 4
            minus = sum(sm[f] for f in SIGNS if w[f] < 0) / 4
            g = (minus - plus) if lower_is_better else (plus - minus)
            mean, lo, hi, p, n = paired(g)
            eff.append({"effect": cname, "gain_high_level": mean, "ci_low": lo, "ci_high": hi, "p": p, "n_deficits": n})
        ef = pd.DataFrame(eff)
        ef["p_holm_main"] = np.nan
        ef.loc[:3, "p_holm_main"] = holm(ef.loc[:3, "p"])
        ef.to_csv(a.out.replace(".csv", "_factorial.csv"), index=False)
        print("\nfactorial (gain of the high level; two-factor interactions aliased in pairs):")
        print(ef.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
