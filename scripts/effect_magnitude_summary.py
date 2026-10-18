"""Audit H5: is an estimator's PEHE gap a matter of effect magnitude?

The replica writes, for every scored slice, the decomposition of
``giles_replica.effect_decomposition``: the mean predicted effect among the
participants with tau = +1, 0 and -1, the slope of tau_hat on tau, the split of
pehe_paper ** 2 into the error of those group means (between) and the spread
inside each group (within), and pehe_paper after the best single rescaling of
tau_hat (an oracle bound, fitted on the test labels). This script averages them
like the headline (mean over folds per deficit, then over deficits) for every
run, representation, classifier and learner:

    python scripts/effect_magnitude_summary.py ~/Scratch/neuro-causal-pfn-perfold/outputs_perfold_h5 \\
        --out status/h5_effect_magnitude/h5_effect_magnitude.csv

A run is the folder above ``replica/`` (e.g. one PFN checkpoint). ``<out>``
holds one row per scoring; ``<out>_by_run.csv`` the mean and SD over encoder
seeds of each run. A slope well below 1 with a small within part means the
estimator orders the participants but shrinks the effects. Aggregate output
only; files scored before the decomposition existed are skipped.
"""
import argparse
import glob
import os
import re
import sys
from typing import List

import numpy as np
import pandas as pd

METRICS = ["pehe_paper", "prescriptive_balacc", "tau_hat_pos", "tau_hat_zero", "tau_hat_neg",
           "tau_slope", "mse_between", "mse_within", "rescale_factor", "pehe_paper_rescaled"]
BY_RUN = ["pehe_paper", "prescriptive_balacc", "tau_slope", "tau_hat_pos", "tau_hat_neg",
          "between_share", "pehe_paper_rescaled"]


def find_results(paths: List[str]) -> List[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(glob.escape(p), "**", "replica_results.csv"), recursive=True)
        else:
            files += glob.glob(p) or glob.glob(p.replace("[", "[[]"))
    return sorted(set(files))


def run_label(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if "replica" in parts:
        i = len(parts) - 1 - parts[::-1].index("replica")
        if i > 0:
            return parts[i - 1]
    return parts[-2] if len(parts) > 1 else "."


def encoder_seed(representation: str):
    m = re.search(r"\|seed(\d+)\|", f"|{representation}|")
    return int(m.group(1)) if m else np.nan


def summarise_file(df: pd.DataFrame, run: str) -> List[dict]:
    rows = []
    for (rep, clf, lrn), g in df.groupby(["representation", "classifier", "learner"]):
        per_def = g.groupby("deficit")[METRICS].mean()
        row = {"run": run, "representation": rep, "encoder_seed": encoder_seed(rep),
               "classifier": clf, "learner": lrn, "n_deficits": int(len(per_def)),
               "n_slices": int(len(g))}
        row.update({c: float(per_def[c].mean()) for c in METRICS})
        tot = row["mse_between"] + row["mse_within"]
        row["between_share"] = row["mse_between"] / tot if tot > 0 else np.nan
        rows.append(row)
    return rows


def by_run(rows: pd.DataFrame) -> pd.DataFrame:
    fam = rows["representation"].str.replace(r"\|seed\d+\|", "|seed*|", regex=True)
    rows = rows.assign(representation=fam)
    out = []
    for key, g in rows.groupby(["run", "representation", "classifier", "learner"]):
        r = dict(zip(["run", "representation", "classifier", "learner"], key))
        r["n_seeds"] = int(len(g))
        r["encoder_seeds"] = " ".join(str(int(s)) for s in sorted(g["encoder_seed"].dropna()))
        for c in BY_RUN:
            r[c] = float(g[c].mean())
            r[f"{c}_sd"] = float(g[c].std(ddof=0))
        out.append(r)
    return pd.DataFrame(out)


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="output roots, replica_results.csv files or globs")
    ap.add_argument("--out", default=None, help="csv to write (plus <out>_by_run.csv)")
    args = ap.parse_args(argv)

    files = find_results(args.paths)
    if not files:
        sys.exit(f"no replica_results.csv under {args.paths}")
    rows, skipped = [], 0
    for f in files:
        df = pd.read_csv(f)
        if not set(METRICS) <= set(df.columns):
            skipped += 1
            continue
        rows += summarise_file(df, run_label(f))
    print(f"{len(files)} results files, {skipped} skipped (scored before the decomposition)")
    if not rows:
        sys.exit("nothing to summarise")
    per = pd.DataFrame(rows).sort_values(["run", "classifier", "learner", "encoder_seed"])
    runs = by_run(per)
    cols = ["run", "classifier", "learner", "n_seeds", "pehe_paper", "prescriptive_balacc",
            "tau_slope", "tau_hat_pos", "tau_hat_neg", "between_share", "pehe_paper_rescaled"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(runs[cols].round(3).to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        per.to_csv(args.out, index=False)
        by_path = re.sub(r"\.csv$", "", args.out) + "_by_run.csv"
        runs.to_csv(by_path, index=False)
        print(f"written {args.out} and {by_path}")
    return runs


if __name__ == "__main__":
    main()
