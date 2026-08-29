#!/usr/bin/env python3
"""Recompute the headline from an existing replica_results.csv (no simulation
rerun needed). Useful after aggregation changes: the per-(deficit, fold,
classifier, learner) results are the ground data; the headline is derived.

    python scripts/reaggregate_replica.py outputs/giles_replica_ideal [more dirs...]

Rewrites replica_headline.csv in each directory and prints the summary.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocausalpfn.utils.portability import configure_portable_runtime

configure_portable_runtime()

import pandas as pd                                       # noqa: E402

from neurocausalpfn.prior import giles_replica as gr      # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for out_dir in sys.argv[1:]:
        path = os.path.join(out_dir, "replica_results.csv")
        res = pd.read_csv(path)
        scenario_cols = [c for c in ("TE", "RE", "BIAS", "BIASTYPE") if c in res.columns]
        rows = []
        for name, sub in res.groupby("representation"):
            agg = gr.headline_aggregate(sub)
            scenario = {c: sub[c].iloc[0] for c in scenario_cols}
            # balanced accuracy of the SAME configuration the PEHE headline
            # picked (the paper's other calibration anchor: VAE-50 disco 0.875,
            # vascular baseline 0.546)
            if agg["classifier"]:
                cfg = sub[(sub["classifier"] == agg["classifier"]) &
                          (sub["learner"] == agg["learner"])]
                for col, key in (("prescriptive_balacc", "balacc_mean"),
                                 ("pehe_xor", "pehe_xor_mean")):
                    if col in cfg.columns:
                        agg[key] = float(cfg.groupby("deficit")[col].mean().mean())
            rows.append({"representation": name, **agg, **scenario})
            print(f"  {out_dir} :: {name:24s} PEHE {agg['pehe_mean']:.3f} "
                  f"(CI {agg['ci_low']:.3f}-{agg['ci_high']:.3f}, "
                  f"{agg['classifier']}/{agg['learner']}, "
                  f"per-deficit-best {agg['pehe_per_deficit_best']:.3f}, "
                  f"balacc {agg.get('balacc_mean', float('nan')):.3f}, "
                  f"pehe-xor {agg.get('pehe_xor_mean', float('nan')):.3f})")
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "replica_headline.csv"),
                                  index=False)


if __name__ == "__main__":
    main()
