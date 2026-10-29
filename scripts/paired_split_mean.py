"""Paired test on the mean of the two splits (29 Oct), the single headline
number Giles et al. report instead of one per split.

paired_deficit_test.py writes, next to each summary, a *_per_deficit.csv with
one value per deficit for A and B (mean over folds). This takes the receptor
and the genetics files of one comparison, averages each deficit over the two
splits (a deficit undefined in either split is left out and counted, as in the
per-split test), and runs the same paired summary on the 16 averages::

    python scripts/paired_split_mean.py \\
        --receptor tables/b2b_paired_location_bias_025_receptor_pehe_paper_per_deficit.csv \\
        --genetics tables/b2b_paired_location_bias_025_genetics_pehe_paper_per_deficit.csv \\
        --label-a TabPFN-S --label-b CausalPFN --level location_bias_025 \\
        --out tables/b2b_both_location_bias_025_pehe_paper.csv

Holm across the combined rows goes through holm_adjust.py, as a family of its
own. Aggregate output only.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_deficit_test import paired_summary  # noqa: E402


def split_mean(receptor: pd.DataFrame, genetics: pd.DataFrame, label_a: str, label_b: str):
    r, g = receptor.set_index("deficit"), genetics.set_index("deficit")
    if sorted(r.index) != sorted(g.index):
        raise ValueError(f"the splits have different deficits ({len(r)} receptor, {len(g)} genetics)")
    a = pd.concat([r[label_a], g[label_a]], axis=1).mean(axis=1, skipna=False)
    b = pd.concat([r[label_b], g[label_b]], axis=1).mean(axis=1, skipna=False)
    return a, b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--receptor", required=True, help="*_per_deficit.csv of the receptor split")
    ap.add_argument("--genetics", required=True, help="*_per_deficit.csv of the genetics split")
    ap.add_argument("--label-a", required=True); ap.add_argument("--label-b", required=True)
    ap.add_argument("--level", default=None, help="optional label column (e.g. the bias level or learner)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    a, b = split_mean(pd.read_csv(args.receptor), pd.read_csv(args.genetics), args.label_a, args.label_b)
    s = {"split": "both", **({"level": args.level} if args.level else {}),
         **paired_summary(a, b, args.label_a, args.label_b)}
    pd.DataFrame([s]).to_csv(args.out, index=False)
    print(f"both{' ' + args.level if args.level else ''}: {s['label_a']} - {s['label_b']} {s['diff_mean']:+.4f} "
          f"over n = {s['n_deficits']} ({s['n_excluded_nan']} left out); "
          f"t p = {s['paired_t_p']:.2g}, Wilcoxon p = {s['wilcoxon_p']:.2g}; wrote {args.out}")
    return s


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
