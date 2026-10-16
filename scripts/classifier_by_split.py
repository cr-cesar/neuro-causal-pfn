"""PEHE and balanced accuracy of every classifier x learner, per atlas split (F28).

The leaderboard reports only the cell the headline selects. This table gives
all of them for each split, aggregated exactly as the headline does: per seed
(one replica_results.csv each), the mean over folds of each deficit, then the
mean over the 16 deficits; across seeds, the mean and the SD of those values.
A last block puts the splits side by side with their mean, the comparison
with a figure that averages both splits.

    PF=~/Scratch/neuro-causal-pfn-perfold
    python scripts/classifier_by_split.py \\
        --split receptor "$PF/outputs_perfold_pub/replica/E1/E1/seed*/*-singles/replica_results.csv" \\
        --split genetics "$PF/outputs_perfold_pub_ideal_genetics/replica/E1/E1/seed*/*-singles/replica_results.csv" \\
        --out status/<day>/f28/classifier_by_split_E1.csv

Aggregate output only.
"""
import argparse
import glob
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

METRICS = ("pehe_paper", "prescriptive_balacc")


def _files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        # label folders such as E3[backbone=cnn] contain brackets
        files = sorted(glob.glob(pattern.replace("[", "[[]")))
    return files


def per_seed(df: pd.DataFrame) -> pd.DataFrame:
    """One seed: classifier x learner -> mean over deficits of the fold means."""
    metrics = [m for m in METRICS if m in df.columns]
    per_def = df.groupby(["deficit", "classifier", "learner"])[metrics].mean()
    return per_def.groupby(["classifier", "learner"]).mean()


def table(splits: List[Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for name, pattern in splits:
        files = _files(pattern)
        if not files:
            raise FileNotFoundError(f"split {name}: no file matches {pattern}")
        seeds = [per_seed(pd.read_csv(f)) for f in files]
        stacked = pd.concat(seeds, keys=range(len(seeds)), names=["seed"])
        g = stacked.groupby(["classifier", "learner"])
        mean, sd = g.mean(), g.std(ddof=0)
        for (clf, lrn), r in mean.iterrows():
            row = {"split": name, "classifier": clf, "learner": lrn, "n_seeds": len(seeds)}
            for m in r.index:
                row[f"{m}_mean"] = float(r[m])
                row[f"{m}_seed_sd"] = float(sd.loc[(clf, lrn), m])
            rows.append(row)
    return pd.DataFrame(rows)


def side_by_side(t: pd.DataFrame) -> pd.DataFrame:
    wide = t.pivot_table(index=["classifier", "learner"], columns="split", values="pehe_paper_mean")
    if wide.shape[1] > 1:
        wide["mean_of_splits"] = wide.mean(axis=1)
    return wide


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", nargs=2, action="append", metavar=("NAME", "GLOB"), required=True,
                    help="a split name and the glob of its replica_results.csv files (one per seed)")
    ap.add_argument("--out", default=None, help="csv; the side-by-side table goes next to it")
    args = ap.parse_args(argv)

    t = table([tuple(s) for s in args.split])
    print(t.round(3).to_string(index=False))
    wide = side_by_side(t)
    print("\npehe_paper by split:")
    print(wide.round(3).to_string())
    if args.out:
        t.to_csv(args.out, index=False)
        wide.to_csv(args.out.replace(".csv", "_wide.csv"))
        print("wrote", args.out)
    return t


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
