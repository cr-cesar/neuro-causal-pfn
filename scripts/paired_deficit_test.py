"""Paired tests over the 16 deficits of the virtual trial (F31).

Two configurations scored on the same folds and test sets (two estimators on
one representation, or two representations under one estimator) are compared
deficit by deficit: the PEHE of each deficit is its mean over folds (and over
seeds / files when several are given), the paired difference A - B is taken
over the deficits, and the script reports its mean, 95 % t confidence interval,
paired t-test p and Wilcoxon signed-rank p, always with n = the deficits
present in both. Overlapping confidence intervals of two means say nothing
about their difference; this does.

    python scripts/paired_deficit_test.py \\
        --a "outputs_perfold_pub_t4/replica/E1/E1/seed*/*-singles/replica_results.csv" --a-filter classifier=causalpfn \\
        --b "outputs_perfold_pub/replica/E1/E1/seed*/*-singles/replica_results.csv" --b-filter classifier=logistic_regression learner=two \\
        --label-a "CausalPFN" --label-b "LR two-model" --out paired_causalpfn_vs_lr.csv

Filters are column=value pairs applied to the replica_results rows; the
metric defaults to pehe_paper (the headline scale). Aggregate output only.
"""
import argparse
import glob
import math
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _load(patterns: List[str], filters: Dict[str, str], metric: str) -> pd.Series:
    files = sorted(f for pat in patterns for f in glob.glob(pat))
    if not files:
        # label folders such as E3[backbone=cnn] contain brackets, which glob
        # reads as a character class: retry with the brackets taken literally
        files = sorted(f for pat in patterns for f in glob.glob(pat.replace("[", "[[]")))
    if not files:
        raise FileNotFoundError(f"no file matches {patterns}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        for k, v in filters.items():
            if k not in df.columns:
                raise KeyError(f"{f}: no column {k!r} (columns: {list(df.columns)})")
            df = df[df[k].astype(str) == str(v)]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError(f"no rows left after filters {filters} in {len(files)} file(s)")
    if metric not in df.columns:
        raise KeyError(f"no column {metric!r}")
    # one value per deficit: mean over folds (and seeds / files)
    return df.groupby("deficit")[metric].mean()


def paired_summary(a: pd.Series, b: pd.Series, label_a: str = "A", label_b: str = "B") -> Dict[str, float]:
    from scipy import stats

    common = a.index.intersection(b.index)
    a, b = a.loc[common].astype(float), b.loc[common].astype(float)
    # a metric can be undefined for a deficit (balanced accuracy needs both
    # classes among the decidable patients): pair only the finite deficits and
    # report how many were left out, instead of letting one nan void the test
    finite = np.isfinite(a.to_numpy()) & np.isfinite(b.to_numpy())
    n_excluded = int((~finite).sum())
    a, b = a[finite], b[finite]
    d = (a - b).to_numpy()
    n = len(d)
    if n < 2:
        raise ValueError(f"need at least 2 common deficits, got {n}")
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    t_stat, t_p = stats.ttest_rel(a.to_numpy(), b.to_numpy())
    try:
        w_stat, w_p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    except ValueError:            # all differences zero
        w_stat, w_p = float("nan"), 1.0
    return {"label_a": label_a, "label_b": label_b, "n_deficits": int(n), "n_excluded_nan": n_excluded,
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff_mean": mean, "diff_ci_low": mean - tcrit * se, "diff_ci_high": mean + tcrit * se,
            "diff_sd": sd, "n_a_lower": int((d < 0).sum()), "n_b_lower": int((d > 0).sum()),
            "paired_t": float(t_stat), "paired_t_p": float(t_p),
            "wilcoxon_w": float(w_stat), "wilcoxon_p": float(w_p)}


def _filters(items: Optional[List[str]]) -> Dict[str, str]:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not k or not v:
            raise ValueError(f"bad filter {it!r}: expected column=value")
        out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", nargs="+", required=True, help="replica_results.csv file(s) or glob(s) of configuration A")
    ap.add_argument("--b", nargs="+", required=True, help="the same for configuration B")
    ap.add_argument("--a-filter", nargs="*", default=None, help="column=value filters for A (e.g. classifier=causalpfn)")
    ap.add_argument("--b-filter", nargs="*", default=None)
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--metric", default="pehe_paper")
    ap.add_argument("--out", default=None, help="csv: the summary row plus one row per deficit")
    args = ap.parse_args(argv)

    a = _load(args.a, _filters(args.a_filter), args.metric)
    b = _load(args.b, _filters(args.b_filter), args.metric)
    s = paired_summary(a, b, args.label_a, args.label_b)
    common = a.index.intersection(b.index)
    per = pd.DataFrame({"deficit": common, args.label_a: a.loc[common].to_numpy(),
                        args.label_b: b.loc[common].to_numpy()})
    per["diff"] = per[args.label_a] - per[args.label_b]
    print(f"{args.metric}: {s['label_a']} {s['mean_a']:.3f} vs {s['label_b']} {s['mean_b']:.3f} over n = {s['n_deficits']} deficits"
          + (f" ({s['n_excluded_nan']} left out: metric undefined)" if s["n_excluded_nan"] else ""))
    print(f"  difference A - B {s['diff_mean']:+.3f}  95% CI [{s['diff_ci_low']:+.3f}, {s['diff_ci_high']:+.3f}]  "
          f"(A lower in {s['n_a_lower']}, B lower in {s['n_b_lower']})")
    print(f"  paired t = {s['paired_t']:.2f}, p = {s['paired_t_p']:.4f};  Wilcoxon W = {s['wilcoxon_w']:.1f}, p = {s['wilcoxon_p']:.4f}")
    print(per.round(3).to_string(index=False))
    if args.out:
        pd.DataFrame([s]).to_csv(args.out, index=False)
        per.to_csv(args.out.replace(".csv", "_per_deficit.csv"), index=False)
        print("wrote", args.out)
    return s


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
