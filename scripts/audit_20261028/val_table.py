"""Validation PEHE / balanced accuracy per model (aggregates only).
PEHE: mean over folds per deficit, then over deficits; bal.acc: mean of the defined rows.

    python val_table.py --run label=<root>/{m}/replica/E1/E1/seed0/*/replica_results.csv[:col=val,...] --folds 0 --out t.csv
"""
import argparse, glob
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--folds", default=None, help="comma list, e.g. 0; default all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); rows = []
    for spec in a.run:
        label, rest = spec.split("=", 1)
        pat, _, flt = rest.partition(":")
        filters = dict(kv.split("=") for kv in flt.split(",")) if flt else {}
        r = {"model": label}
        for m in ("receptor", "genetics"):
            d = pd.concat([pd.read_csv(f) for f in glob.glob(pat.format(m=m))])
            for k, v in filters.items():
                d = d[d[k].astype(str) == v]
            if a.folds is not None:
                d = d[d.fold.isin([int(x) for x in a.folds.split(",")])]
            r[f"{m}_pehe"] = round(d.groupby("deficit").pehe_paper.mean().mean(), 3)
            r[f"{m}_balacc"] = round(float(d.prescriptive_balacc.mean()), 3)
            r[f"{m}_balacc_undefined"] = int(d.prescriptive_balacc.isna().sum()); r[f"{m}_rows"] = len(d)
            r[f"{m}_folds"] = d.fold.nunique()
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(a.out, index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
