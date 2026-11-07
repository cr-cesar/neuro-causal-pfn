"""Brief 9 task 2: one table per Wave 0 cell from wave0_cells.sh's scorings
(validation pseudo-folds only). Estimators are told apart by the
replica_results columns that name them (classifier, learner, estimator, when
present); the final model's three seeds are averaged per deficit. For every
cell and metric (pehe_paper against tau in {-1, 0, 1}; pehe_implied against
TE (1 - RE) tau): the mean per split and over splits per estimator, and the
paired difference final model minus each other estimator over the 16
deficits (mean of splits; 95 % t interval). Aggregates only.

    python scripts/audit_20261107/wave0_table.py --out $W/wave0/wave0_tables.csv
"""
import argparse
import glob
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

SPLITS = ("receptor", "genetics")
CELLS = ("noisy_te075", "noisy_te050", "noisy_te025", "location_bias_050", "location_bias_095", "location_bias",
         "responsiveness_060", "responsiveness_075", "responsiveness_090")
KEYS = ("classifier", "learner", "estimator")


def rr(root, label, split, rep):
    return os.path.join(root, label, split, "replica", rep, "disconnectome-extrasclean", "replica_results.csv")


def per_deficit(files, metric):
    """{estimator: DataFrame deficit x split} for the rows of the given files."""
    out = {}
    for split, paths in files.items():
        frames = [pd.read_csv(p) for p in paths if os.path.exists(p)]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        if metric not in df.columns:
            continue
        keys = [k for k in KEYS if k in df.columns]
        df["_est"] = df[keys].astype(str).agg("|".join, axis=1) if keys else "estimator"
        for est, g in df.groupby("_est"):
            out.setdefault(est, {})[split] = g.groupby("deficit")[metric].mean()
    return {e: pd.DataFrame(v) for e, v in out.items() if set(v) == set(SPLITS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/neuro-causal-pfn-perfold/outputs_val_clean"))
    ap.add_argument("--rep", default="E13a/E13a/seed0")
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--metrics", nargs="+", default=["pehe_paper", "pehe_implied", "prescriptive_balacc"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows, pairs = [], []
    for cell in a.cells:
        for metric in a.metrics:
            final = per_deficit({m: [rr(a.root, f"w0_{cell}_final1M_s{s}", m, a.rep) for s in a.seeds] for m in SPLITS},
                                metric)
            if not final:
                continue
            fin = sum(final.values()) / len(final)          # one PFN estimator key; mean if several
            others = {}
            for lab in (f"w0_{cell}_lret_causalpfn", f"w0_{cell}_tabpfn"):
                others.update(per_deficit({m: [rr(a.root, lab, m, a.rep)] for m in SPLITS}, metric))
            tabs = {"final1M (3 seeds)": fin, **others}
            for est, t in tabs.items():
                rows.append({"cell": cell, "metric": metric, "estimator": est, "n_deficits": len(t.dropna()),
                             "receptor": t["receptor"].mean(), "genetics": t["genetics"].mean(),
                             "mean": t.mean(axis=1).mean()})
            for est, t in others.items():
                d = (fin.mean(axis=1) - t.mean(axis=1)).dropna().to_numpy()
                if len(d) < 2:
                    continue
                se = d.std(ddof=1) / math.sqrt(len(d)); tc = stats.t.ppf(0.975, len(d) - 1)
                pairs.append({"cell": cell, "metric": metric, "final_minus": est, "diff": d.mean(),
                              "ci_low": d.mean() - tc * se, "ci_high": d.mean() + tc * se,
                              "p": float(stats.ttest_1samp(d, 0).pvalue), "n_deficits": len(d)})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    pd.DataFrame(pairs).to_csv(a.out.replace(".csv", "_paired.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        print(pd.DataFrame(pairs).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
