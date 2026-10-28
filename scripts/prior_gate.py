"""Prior gate (28 Oct): before a training run, draw tasks from the prior the
run will train on and check that they look like the scored trial. Aggregates
only: shares and means over the drawn processes, never a lesion or a row.

The configuration is built from the environment exactly as
scripts/train_pfn_myriad.qsub.sh builds it (launch_cfg.cfg_from_env), so the
gate sees the training run's prior. For every drawn process the queries give
the true effect tau = mu1 - mu0 and the context its treatment and outcomes:

- ``share_zero``: queries with tau = 0 (the trial: 0.455 to 0.545 of the
  tested patients of a slice);
- ``share_unit``: queries with |tau| >= 0.95 (the trial: the rest, |tau| = 1);
- ``share_small``: queries with 0 < |tau| < 0.05, effects too small to read;
- ``pos_share``: tau > 0 among the non-null (the trial: both signs);
- ``ctx_binary``: processes whose context outcomes are all 0 or 1;
- ``ctx_treated``: share of treated context rows (the trial: 0.5).

The default requirements are the trial's, with a margin; a run whose prior
fails them does not train. ``--require metric=lo:hi`` replaces one, and
``--expect key.path=value`` also requires a value of the cfg (e.g.
``prior.hyper.generic_weight=0.8``), which replaces the per-run check
scripts. Run it on a compute node or the lab machine (the pool cache is
patient data and stays there):

    python scripts/prior_gate.py --batches 50 --contexts 128 512 3000 --out gate.csv
"""
import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

METRICS = ("share_zero", "share_unit", "share_small", "pos_share", "ctx_binary", "ctx_treated")
DEFAULT_REQUIRE = {"share_zero": (0.35, 0.65), "share_unit": (0.30, 0.65), "share_small": (0.0, 0.10),
                   "pos_share": (0.25, 0.75), "ctx_binary": (0.90, 1.0), "ctx_treated": (0.40, 0.60)}


def process_stats(batch: Dict) -> List[Dict]:
    """One row of metrics per process of a prior batch."""
    procs = batch.get("processes") or [{}] * len(batch["mu0"])
    rows = []
    for i in range(len(batch["mu0"])):
        tau = np.asarray(batch["mu1"][i], float) - np.asarray(batch["mu0"][i], float)
        a = np.abs(tau)
        nz = a > 1e-6
        yc = np.asarray(batch["Yc"][i], float)
        p = procs[i] if isinstance(procs[i], dict) else {}
        rows.append({"family": str(p.get("family", "?")), "n_slice": p.get("n_slice"),
                     "share_zero": float((~nz).mean()), "share_unit": float((a >= 0.95).mean()),
                     "share_small": float((nz & (a < 0.05)).mean()),
                     "pos_share": float((tau[nz] > 0).mean()) if nz.any() else np.nan,
                     "ctx_binary": float(np.all((yc == 0) | (yc == 1))),
                     "ctx_treated": float(np.mean(batch["Tc"][i]))})
    return rows


def summarise(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Mean of each metric over all processes and per family label."""
    out = {}
    for fam in ["all"] + sorted({r["family"] for r in rows}):
        sel = [r for r in rows if fam == "all" or r["family"] == fam]
        out[fam] = {"processes": float(len(sel)),
                    **{m: float(np.nanmean([r[m] for r in sel])) if sel else np.nan for m in METRICS}}
        sl = [r["n_slice"] for r in sel if r["n_slice"] is not None]
        out[fam]["median_slice"] = float(np.median(sl)) if sl else np.nan
    return out


def check(summary: Dict[str, Dict[str, float]], require: Dict) -> List[str]:
    errors = []
    s = summary["all"]
    for m, (lo, hi) in require.items():
        v = s.get(m, np.nan)
        if not (lo <= v <= hi):
            errors.append(f"{m} {v:.3f} outside [{lo}, {hi}]")
    return errors


def _flat(d: Dict, prefix: str = "") -> Dict[str, object]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def check_expect(cfg: Dict, expects: List[str]) -> List[str]:
    """``key.path=value`` items against the cfg; a range is lo:hi, compared
    element-wise with a list value."""
    flat, errors = _flat(cfg), []
    for item in expects:
        key, _, want = item.partition("=")
        have = flat.get(key, "<unset>")
        try:
            w = [float(x) for x in want.split(":")]
            h = [float(x) for x in (have if isinstance(have, (list, tuple)) else [have])]
            ok = len(w) == len(h) and np.allclose(w, h)
        except (TypeError, ValueError):
            ok = str(have) == want
        if not ok:
            errors.append(f"{key} is {have!r}, expected {want}")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=50, help="batches per context length")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 3000])
    ap.add_argument("--require", nargs="*", default=[], help="metric=lo:hi, replaces a default")
    ap.add_argument("--expect", nargs="*", default=[], help="cfg key.path=value that must hold")
    ap.add_argument("--out", default=None, help="CSV of the per-family summary")
    a = ap.parse_args()

    from neurocausalpfn.train.launch_cfg import cfg_from_env
    from neurocausalpfn.train.train_pfn import _build_prior

    cfg = cfg_from_env()
    errors = check_expect(cfg, a.expect)
    require = dict(DEFAULT_REQUIRE)
    for item in a.require:
        m, _, r = item.partition("=")
        if m not in METRICS:
            sys.exit(f"unknown metric {m!r}; choose from {METRICS}")
        lo, hi = (float(x) for x in r.split(":"))
        require[m] = (lo, hi)
    prior, _, _ = _build_prior(cfg)
    if hasattr(prior, "describe"):
        print(f"prior: {prior.describe()}")
    rows = []
    for n_ctx in a.contexts:
        for _ in range(a.batches):
            rows += [dict(r, n_ctx=n_ctx) for r in process_stats(prior.sample_batch(a.batch_size, n_context=n_ctx))]
    summary = summarise(rows)
    for fam, s in summary.items():
        print(f"{fam:>16}: " + ", ".join(f"{k} {v:.3f}" if k != "processes" else f"{int(v)} processes"
                                         for k, v in s.items()))
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["family", "processes", *METRICS, "median_slice"])
            for fam, s in summary.items():
                w.writerow([fam, int(s["processes"]), *[f"{s[m]:.4f}" for m in METRICS], s["median_slice"]])
    errors += check(summary, require)
    if errors:
        print("PRIOR GATE FAILED: " + "; ".join(errors))
        sys.exit(1)
    print("prior gate ok: the drawn tasks match the trial within the required ranges")


if __name__ == "__main__":
    main()
