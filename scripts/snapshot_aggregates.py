"""Aggregate tables for the audit-log snapshot (scripts/myriad_phase1b.sh
snapshot): every replica_headline.csv row, and the per-deficit slice sizes
(n_train / n_test per deficit and fold) of the headline encoder's per-fold
scoring and of the replicas of this repository. Aggregate counts only: no
filenames, participant ids or latents are read.

    PERFOLD=~/Scratch/neuro-causal-pfn-perfold python scripts/snapshot_aggregates.py status/<date>
"""
import glob
import os
import sys

import pandas as pd


def main(out: str) -> None:
    perfold = os.path.expanduser(os.environ.get("PERFOLD", "~/Scratch/neuro-causal-pfn-perfold"))

    def rel(f):
        return os.path.relpath(f, perfold) if f.startswith(perfold) else f

    heads = []
    for f in sorted(glob.glob(f"{perfold}/outputs_perfold*/replica/**/replica_headline.csv", recursive=True)
                    + glob.glob("outputs/giles_replica_*/replica_headline.csv")):
        d = pd.read_csv(f)
        d.insert(0, "source", rel(f))
        heads.append(d)
    if heads:
        pd.concat(heads, ignore_index=True).to_csv(os.path.join(out, "headline_rows.csv"), index=False)

    sizes = []
    for f in sorted(glob.glob(f"{perfold}/outputs_perfold_pub/replica/E1/E1/seed0/**/replica_results.csv", recursive=True)
                    + glob.glob("outputs/giles_replica_*/replica_results.csv")):
        cols = ["deficit", "fold", "n_train", "n_test"]
        d = pd.read_csv(f, usecols=lambda c: c in cols)
        if d.empty or any(c not in d.columns for c in cols):
            # header-only or legacy results file (e.g. an aborted replica): nothing to aggregate
            print(f"skip {rel(f)}: no slice-size columns")
            continue
        d = d.drop_duplicates(["deficit", "fold"])[cols]
        d.insert(0, "source", rel(f))
        sizes.append(d)
    if sizes:
        allsz = pd.concat(sizes, ignore_index=True)
        allsz.to_csv(os.path.join(out, "slice_sizes.csv"), index=False)
        allsz.groupby("source")[["n_train", "n_test"]].agg(["min", "median", "max"]).to_csv(
            os.path.join(out, "slice_sizes_summary.csv"))
    print(f"wrote {len(heads)} headline files and {len(sizes)} slice-size files to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "status/latest")
