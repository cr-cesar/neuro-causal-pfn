"""Consolidated reporting for the experiment programme.

Reads the per-run records the logger always writes to ``runs.jsonl`` and produces
a leaderboard (CSV and Markdown) grouped by arm, with the seed-aggregated tier
metrics and the stop/go outcome of each configuration. Also provides the
bootstrap-paired test on root-PEHE that section 14 prescribes for statistical
comparisons (1000 resamples).
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from .registry import TIER_GATES, get_experiment

TIER_METRICS = ["T1.dice", "T2.r2_nihss", "T3.active_dims", "T3.ioss", "T4.root_pehe",
                "T4.ate_bias", "T4.prescriptive_accuracy", "T4.ood_gap"]


def bootstrap_paired_pehe(pehe_a: np.ndarray, pehe_b: np.ndarray, n: int = 1000,
                          seed: int = 0) -> Dict:
    """Bootstrap-paired comparison of two configurations' per-query squared
    errors. Returns the mean root-PEHE difference (a - b), a 95% interval and the
    fraction of resamples in which A beats B (lower root-PEHE)."""
    a = np.asarray(pehe_a, dtype=np.float64).ravel()
    b = np.asarray(pehe_b, dtype=np.float64).ravel()
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    rng = np.random.default_rng(seed)
    diffs, a_wins = [], 0
    for _ in range(n):
        idx = rng.integers(0, m, m)
        ra = float(np.sqrt(np.mean(a[idx])))
        rb = float(np.sqrt(np.mean(b[idx])))
        diffs.append(ra - rb)
        a_wins += int(ra < rb)
    diffs = np.array(diffs)
    return {"mean_diff": float(diffs.mean()),
            "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
            "prob_a_better": a_wins / n}


def _read_runs(out_root: str) -> List[Dict]:
    path = os.path.join(out_root, "runs.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _leaderboard(rows: List[Dict]) -> List[Dict]:
    # last write wins per (eid, label, seed): a re-run of the same variant
    # replaces its stale row instead of stacking with it, matching
    # finalize_experiment's semantics (otherwise the seed counts double and
    # the means mix old and new training runs)
    latest: Dict = {}
    for r in rows:
        label = r.get("label")
        eid = str(r.get("kind", "")).split("/")[0]
        if not label or "seed" not in r:
            continue
        latest[(eid, label, int(r["seed"]))] = r

    # collect per-(eid,label) seed values
    buckets: Dict = defaultdict(lambda: defaultdict(list))
    passed: Dict = defaultdict(list)
    for (eid, label, _seed), r in latest.items():
        key = (eid, label)
        for m in TIER_METRICS:
            if isinstance(r.get(m), (int, float)):
                buckets[key][m].append(r[m])
        passed[key].append(1.0 if r.get("passed") else 0.0)

    board = []
    for (eid, label), metrics in buckets.items():
        try:
            arm = get_experiment(eid).arm
        except KeyError:
            arm = "?"
        entry = {"arm": arm, "eid": eid, "label": label,
                 "n_seeds": max((len(v) for v in metrics.values()), default=0),
                 "passed_frac": float(np.mean(passed[(eid, label)])) if passed[(eid, label)] else 0.0}
        for m, vals in metrics.items():
            entry[f"{m}.mean"] = float(np.mean(vals))
            entry[f"{m}.std"] = float(np.std(vals))
        board.append(entry)
    board.sort(key=lambda e: (e["arm"], e.get("T4.root_pehe.mean", float("inf"))))
    return board


def build_report(out_root: str) -> Dict[str, str]:
    rows = _read_runs(out_root)
    board = _leaderboard(rows)
    csv_path = os.path.join(out_root, "leaderboard.csv")
    md_path = os.path.join(out_root, "leaderboard.md")

    cols = ["arm", "eid", "label", "n_seeds", "passed_frac",
            "T1.dice.mean", "T2.r2_nihss.mean", "T3.active_dims.mean",
            "T3.ioss.mean", "T4.root_pehe.mean", "T4.root_pehe.std",
            "T4.prescriptive_accuracy.mean", "T4.ood_gap.mean"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in board:
            w.writerow({c: _round(e.get(c)) for c in cols})

    with open(md_path, "w") as f:
        f.write(_markdown(board))
    return {"csv": csv_path, "md": md_path}


def _round(v):
    return round(v, 4) if isinstance(v, float) else ("" if v is None else v)


def _markdown(board: List[Dict]) -> str:
    gate4 = TIER_GATES["T4"].threshold
    gate1 = TIER_GATES["T1"].threshold
    lines = [
        "# Table 9 - experiment leaderboard",
        "",
        f"Stop/go gates: T1 Dice >= {gate1}, T2 R2 >= {TIER_GATES['T2'].threshold}. "
        "T4 root-PEHE is the INTERNAL proxy: it ranks variants within one "
        "experiment only and is NOT comparable across representations or "
        "against the published 0.349 (external comparisons use the certified "
        "replica's decidable-subset PEHE). Metrics are seed-aggregated means.",
        "",
        "| Arm | Exp | Variant | Seeds | Pass% | T1 Dice | T2 R2 | T3 dims | T3 IOSS | "
        "T4 rootPEHE | Presc.acc | OOD gap |",
        "|-----|-----|---------|-------|-------|---------|-------|---------|---------|"
        "-------------|-----------|---------|",
    ]
    for e in board:
        lines.append("| {arm} | {eid} | {label} | {n} | {pf:.0%} | {t1} | {t2} | {t3d} | "
                     "{t3i} | {t4} | {pa} | {ood} |".format(
                        arm=e["arm"], eid=e["eid"], label=e["label"], n=e["n_seeds"],
                        pf=e["passed_frac"],
                        t1=_cell(e.get("T1.dice.mean")), t2=_cell(e.get("T2.r2_nihss.mean")),
                        t3d=_cell(e.get("T3.active_dims.mean")), t3i=_cell(e.get("T3.ioss.mean")),
                        t4=_cell(e.get("T4.root_pehe.mean")),
                        pa=_cell(e.get("T4.prescriptive_accuracy.mean")),
                        ood=_cell(e.get("T4.ood_gap.mean"))))
    lines.append("")
    lines.append("Generated from runs.jsonl (last write per variant and seed "
                 "wins). The proxy root-PEHE carries no absolute meaning; the "
                 "head-to-head against the Giles VAE-50 (0.349) lives in the "
                 "replica outputs (scripts/reaggregate_replica.py, pehe-xor).")
    return "\n".join(lines) + "\n"


def _cell(v):
    return "-" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="outputs/experiments")
    args = ap.parse_args()
    print(build_report(args.out_root))
