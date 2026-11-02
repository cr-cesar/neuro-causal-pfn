"""What the final model's prior trains on, against what it is scored on (2 Nov).

Run under the launch environment of the final model (g4c_launch.sh with
PRIOR_CHECK_ONLY=1 builds the same cfg and calls this script instead of the
gate and the training), on CPU, in minutes. Four checks of the external
review of 2 Nov, all measured on the drawn tasks:

A. cache: channel, threshold and modality of the pool, the dtypes of the
   overlap channels, and the number of distinct anatomical atoms of the
   random-partition family (16 networks + 32 receptor halves + 32 genetics
   halves = 80 when the primary channel repeats one split exactly);
B. class shares by family, with the gate's default requirements (the trial's
   shares) and no override: share_zero, share_unit, share_small, and the
   realised share of each family among the returned processes against the
   printed composition (rejected processes redraw the family);
C. context scale: per training context, the RMS over dimensions of the
   context mean and the mean per-dimension variance (trace of the covariance
   over d), both invariant under the per-process rotation; a scored context
   is standardised with its own statistics, so it has 0 and 1 by
   construction;
D. context duplication: distinct rows per training context (a scored context
   never repeats a patient), as counts and as a share of the context length.

Aggregates only: no lesion, row or file name is printed or written.

    PRIOR_CHECK_ONLY=1 OBJECTIVE=new ... bash scripts/audit_20261028/g4c_launch.sh
    python scripts/audit_20261028/prior_checks.py --batches 25 --contexts 512 3000 --out <dir>
"""
import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, ".."))

from prior_gate import DEFAULT_REQUIRE, METRICS, process_stats  # noqa: E402


def cache_report(prior) -> Dict[str, object]:
    """A: the pool's metadata and the atom count of the random partitions."""
    from neurocausalpfn.prior.partitions import atom_table

    out: Dict[str, object] = {"channel": getattr(prior, "channel", None)}
    meta = getattr(prior, "cache_meta", {}) or {}
    out["threshold"] = meta.get("threshold")
    out["modality"] = meta.get("modality")
    out["images_dir_tail"] = os.path.basename(str(meta.get("images_dir", ""))) or None
    out["overlaps_dtype"] = str(np.asarray(prior.overlaps).dtype)
    extra = getattr(prior, "overlaps_extra", {}) or {}
    for k, v in sorted(extra.items()):
        out[f"{k}_dtype"] = str(np.asarray(v).dtype)
    labels, _ = atom_table(prior.overlaps, extra)
    out["n_atoms"] = len(labels)
    out["n_atoms_primary_kept"] = sum(1 for lab in labels if lab[0] == "primary")
    out["pool_size"] = int(prior.m)
    out["d_latent"] = int(prior.d_x)
    return out


def context_stats(batch: Dict) -> List[Dict]:
    """C and D, one row per process of a batch."""
    rows = []
    for i in range(len(batch["Xc"])):
        Xc = np.asarray(batch["Xc"][i], dtype=np.float64)
        n, d = Xc.shape
        mu = Xc.mean(0)
        rows.append({"rms_mean": float(np.sqrt((mu ** 2).mean())),
                     "mean_var": float(Xc.var(0).sum() / d),
                     "distinct": int(len(np.unique(Xc, axis=0))),
                     "distinct_share": float(len(np.unique(Xc, axis=0)) / n)})
    return rows


def _q(v: List[float], q: float) -> float:
    return float(np.quantile(v, q)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=25, help="batches per context length")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--contexts", type=int, nargs="+", default=[512, 3000])
    ap.add_argument("--out", required=True, help="directory of the aggregate CSVs")
    a = ap.parse_args()

    from neurocausalpfn.train.launch_cfg import cfg_from_env
    from neurocausalpfn.train.train_pfn import _build_prior

    cfg = cfg_from_env()
    prior, _, _ = _build_prior(cfg)
    os.makedirs(a.out, exist_ok=True)
    if hasattr(prior, "describe"):
        print(f"prior: {prior.describe()}")

    rep = cache_report(prior)
    print("A cache: " + ", ".join(f"{k} {v}" for k, v in rep.items()))
    with open(os.path.join(a.out, "prior_checks_cache.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "value"])
        w.writerows(rep.items())

    rows = []
    for n_ctx in a.contexts:
        for _ in range(a.batches):
            b = prior.sample_batch(a.batch_size, n_context=n_ctx)
            for g, c in zip(process_stats(b), context_stats(b)):
                rows.append({**g, **c, "n_ctx": n_ctx})

    fams = ["all"] + sorted({r["family"] for r in rows})
    keys = ("n_ctx", "family", "processes", "process_share", *METRICS, "median_slice",
            "rms_mean_median", "mean_var_median", "distinct_p10", "distinct_median", "distinct_share_median")
    out_rows = []
    for n_ctx in a.contexts:
        sel_n = [r for r in rows if r["n_ctx"] == n_ctx]
        for fam in fams:
            sel = [r for r in sel_n if fam == "all" or r["family"] == fam]
            if not sel:
                continue
            row = {"n_ctx": n_ctx, "family": fam, "processes": len(sel),
                   "process_share": len(sel) / len(sel_n)}
            for m in METRICS:
                row[m] = float(np.nanmean([r[m] for r in sel]))
            row["median_slice"] = float(np.median([r["n_slice"] for r in sel if r["n_slice"] is not None]))
            row["rms_mean_median"] = _q([r["rms_mean"] for r in sel], 0.5)
            row["mean_var_median"] = _q([r["mean_var"] for r in sel], 0.5)
            row["distinct_p10"] = _q([r["distinct"] for r in sel], 0.1)
            row["distinct_median"] = _q([r["distinct"] for r in sel], 0.5)
            row["distinct_share_median"] = _q([r["distinct_share"] for r in sel], 0.5)
            out_rows.append(row)
            print(f"n_ctx {n_ctx:>5} {fam:>16}: " + ", ".join(
                f"{k} {row[k]:.3f}" if isinstance(row[k], float) else f"{k} {row[k]}" for k in keys[2:]))
    with open(os.path.join(a.out, "prior_checks_by_family.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keys))
        w.writeheader()
        for row in out_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})

    # B: the gate's default requirements, without the launch's overrides
    allrows = [r for r in out_rows if r["family"] == "all"]
    fails = []
    for row in allrows:
        for m, (lo, hi) in DEFAULT_REQUIRE.items():
            if not (lo <= row[m] <= hi):
                fails.append(f"n_ctx {row['n_ctx']}: {m} {row[m]:.3f} outside [{lo}, {hi}]")
    print("B default gate: " + ("passes" if not fails else "FAILS: " + "; ".join(fails)))
    print("C/D scored contexts: rms_mean 0, mean_var 1, distinct_share 1 by construction")


if __name__ == "__main__":
    main()
