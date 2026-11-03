"""Leave-networks-out (2 Nov): one replica_results table from the PFNs that
each left a group of networks out of their prior.

Each group is "ids=path": the atlas network ids the PFN never saw, joined by
"+", and the replica_results.csv it was scored into (all 16 deficits). With
--keep held the output keeps, from each group's file, the rows of the deficits
of its own left-out networks, so every deficit is scored by the PFN that never
saw it; with --keep seen, the rows of the other networks, so every deficit is
scored by the PFNs that did. The output is a replica_results.csv for
scripts/paired_deficit_test.py (which averages the rows of a deficit).

Deficit names are read from the receptor parcellation's file names
("<id>_<deficit>_<receptors>"), which both splits share. Aggregates only.

    python scripts/audit_20261028/lno_compose.py --atlas-dir ~/export_bcb/atlases --keep held \\
        --group 1+2+3+4=g1/replica_results.csv ... --out lno_held/replica_results.csv
"""
import argparse
import glob
import os
import sys
from typing import Dict

import pandas as pd


def deficit_names(atlas_dir: str) -> Dict[int, str]:
    out = {}
    for f in glob.glob(os.path.join(atlas_dir, "2mm_parcellations", "receptor", "*.nii*")):
        stem = os.path.basename(f).split(".nii")[0]
        head, _, rest = stem.partition("_")
        if head.isdigit() and rest:
            out[int(head)] = rest.split("_")[0]
    if len(out) != 16:
        raise ValueError(f"expected 16 networks in {atlas_dir}, found {len(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--group", action="append", required=True, help="ids=replica_results.csv, ids joined by +")
    ap.add_argument("--keep", choices=("held", "seen"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    names = deficit_names(a.atlas_dir)
    frames, held_all = [], []
    for g in a.group:
        ids_txt, _, path = g.partition("=")
        ids = [int(x) for x in ids_txt.split("+")]
        held = {names[i] for i in ids}
        held_all += list(held)
        df = pd.read_csv(path)
        missing = held - set(df["deficit"])
        if missing:
            raise ValueError(f"{path}: no rows for {len(missing)} left-out deficit(s)")
        keep = df["deficit"].isin(held) if a.keep == "held" else ~df["deficit"].isin(held)
        frames.append(df[keep])
        print(f"group {ids_txt}: {int(keep.sum())} rows kept over {df[keep]['deficit'].nunique()} deficits")
    if len(held_all) != len(set(held_all)):
        sys.exit("a network is left out by more than one group")
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"{a.keep}: {len(out)} rows, {out['deficit'].nunique()} deficits -> {a.out}")


if __name__ == "__main__":
    main()
