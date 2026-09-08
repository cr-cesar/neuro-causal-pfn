"""Consolidated reporting for the experiment programme.

Reads the per-run records the logger always writes to ``runs.jsonl`` and produces
a leaderboard (CSV and Markdown) grouped by arm, with the seed-aggregated tier
metrics and the stop/go outcome of each configuration. Also provides the
bootstrap-paired test on root-PEHE that section 14 prescribes for statistical
comparisons (1000 resamples).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
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


# --------------------- certified scores (virtual-trial replica) -------------
#
# The certified instrument is the Giles virtual-trial replica: fixed anatomical
# ground truth, identical folds and simulations for every representation,
# calibrated against the six published anchors. Its Methods-convention
# pehe-paper is the ONLY number comparable across representations and against
# the published 0.349, so when it exists it is the leaderboard's headline and
# sort key; the internal T4 proxy stays as an in-training diagnostic.

_SEED_RE = re.compile(r"_seed(\d+)")
_EID_RE = re.compile(r"^(E\d+[a-z]?)(?=_|$)")


def _stem_and_seed(name: str):
    """'E2_E2_w_dice=0.1_seed0_disco.npz' -> ('E2_E2_w_dice=0.1', 0, 0).

    The third element ranks the exported channel: 0 for disconnectome or
    unsuffixed latents, 1 for '_lesion'. An encoder exported on both channels
    yields the same stem twice, and the leaderboard's headline is the
    disconnectome-task league, so the disco export must win regardless of
    which headline file is newer (the lesion-task tables remain available in
    the replica outputs)."""
    name = re.sub(r"\.npz$", "", str(name))
    m = _SEED_RE.search(name)
    if not m:
        return None, None, None
    rest = _SEED_RE.sub("", name)
    rank = 1 if rest.endswith("_lesion") else 0
    stem = re.sub(r"_(disco|lesion)$", "", rest)
    return stem, int(m.group(1)), rank


def _read_certified(out_root: str) -> Dict[str, List[float]]:
    """Per-representation certified pehe-paper values from every
    ``giles_replica_*/replica_headline.csv`` next to the experiment outputs.
    Per (stem, seed): a disconnectome-channel export beats a lesion-channel
    one, and within a channel the newest headline file wins, so a clean
    re-run supersedes stale exports of the same variant."""
    parent = os.path.dirname(os.path.abspath(out_root))
    per: Dict = {}
    for path in glob.glob(os.path.join(parent, "giles_replica_*",
                                       "replica_headline.csv")):
        mtime = os.path.getmtime(path)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                stem, seed, rank = _stem_and_seed(row.get("representation", ""))
                if stem is None or _EID_RE.match(stem) is None:
                    continue
                try:
                    v = float(row.get("pehe_paper_mean", ""))
                except (TypeError, ValueError):
                    continue
                if v != v:                       # NaN: pre-metric run
                    continue
                key = (stem, seed)
                # lower rank (disco) always beats higher (lesion); then newest
                if key not in per or (rank, -mtime) < (per[key][0], -per[key][1]):
                    per[key] = (rank, mtime, v)
    stems: Dict[str, List[float]] = defaultdict(list)
    for (stem, _seed), (_r, _m, v) in sorted(per.items()):
        stems[stem].append(v)
    return stems


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _attach_certified(board: List[Dict], stems: Dict[str, List[float]]) -> None:
    """Match each certified stem to its leaderboard row. The variant part of
    the stem (after the eid) is compared against the variant part of the
    label: exact normalised match first, then substring, then the eid's only
    row. Unmatched stems are ignored rather than guessed."""
    by_eid: Dict[str, List[Dict]] = defaultdict(list)
    for e in board:
        by_eid[e["eid"]].append(e)
    for stem, vals in stems.items():
        eid = _EID_RE.match(stem).group(1)
        cands = by_eid.get(eid, [])
        variant = _norm(re.sub(rf"^({re.escape(eid)}_)+", "", stem))
        lnorms = [_norm(re.sub(rf"^{re.escape(eid)}", "", e["label"]))
                  for e in cands]
        entry = next((e for e, ln in zip(cands, lnorms) if ln == variant), None)
        if entry is None:
            entry = next((e for e, ln in zip(cands, lnorms)
                          if variant and ln and (variant in ln or ln in variant)),
                         None)
        if entry is None and len(cands) == 1:
            entry = cands[0]
        if entry is None:
            continue
        if entry.get("certified.n", 0) >= len(vals):
            continue                             # keep the fuller aggregate
        entry["certified.pehe_paper.mean"] = float(np.mean(vals))
        entry["certified.pehe_paper.std"] = float(np.std(vals))
        entry["certified.n"] = len(vals)


def build_report(out_root: str) -> Dict[str, str]:
    rows = _read_runs(out_root)
    board = _leaderboard(rows)
    _attach_certified(board, _read_certified(out_root))
    # certified rows lead their arm, ordered by the certified score; rows not
    # yet certified follow, ordered by the proxy (a within-experiment signal)
    board.sort(key=lambda e: (e["arm"],
                              0 if "certified.pehe_paper.mean" in e else 1,
                              e.get("certified.pehe_paper.mean",
                                    e.get("T4.root_pehe.mean", float("inf")))))
    csv_path = os.path.join(out_root, "leaderboard.csv")
    md_path = os.path.join(out_root, "leaderboard.md")

    cols = ["arm", "eid", "label", "n_seeds", "passed_frac",
            "certified.pehe_paper.mean", "certified.pehe_paper.std", "certified.n",
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
        "The HEADLINE column is the certified root-PEHE: the Methods-convention "
        "pehe-paper from the virtual-trial replica (fixed anatomical ground "
        "truth, identical folds for every representation), directly comparable "
        "across representations and against the published 0.349. T4 root-PEHE "
        "is the internal in-training proxy: a within-experiment diagnostic "
        "only. Metrics are seed-aggregated means.",
        "",
        "| Arm | Exp | Variant | Seeds | Pass% | Certified rootPEHE | T1 Dice | T2 R2 | "
        "T3 dims | T3 IOSS | T4 proxy | Presc.acc | OOD gap |",
        "|-----|-----|---------|-------|-------|--------------------|---------|-------|"
        "---------|---------|----------|-----------|---------|",
    ]
    for e in board:
        cm = e.get("certified.pehe_paper.mean")
        cert = "-" if cm is None else "{:.3f} ±{:.3f} (n={})".format(
            cm, e.get("certified.pehe_paper.std", 0.0), e.get("certified.n", 0))
        lines.append("| {arm} | {eid} | {label} | {n} | {pf:.0%} | {cert} | {t1} | {t2} | "
                     "{t3d} | {t3i} | {t4} | {pa} | {ood} |".format(
                        arm=e["arm"], eid=e["eid"], label=e["label"], n=e["n_seeds"],
                        pf=e["passed_frac"], cert=cert,
                        t1=_cell(e.get("T1.dice.mean")), t2=_cell(e.get("T2.r2_nihss.mean")),
                        t3d=_cell(e.get("T3.active_dims.mean")), t3i=_cell(e.get("T3.ioss.mean")),
                        t4=_cell(e.get("T4.root_pehe.mean")),
                        pa=_cell(e.get("T4.prescriptive_accuracy.mean")),
                        ood=_cell(e.get("T4.ood_gap.mean"))))
    lines.append("")
    lines.append("Generated from runs.jsonl (last write per variant and seed "
                 "wins). Certified scores are read from "
                 "outputs/giles_replica_*/replica_headline.csv (newest file "
                 "wins per variant and seed); reference points on that scale: "
                 "published VAE-50 0.349, NMF-50 0.320, volume baseline 0.519. "
                 "A '-' means the variant's latents have not been scored on "
                 "the replica yet (scripts/export_latents.py, then "
                 "scripts/run_giles_replica_myriad.qsub.sh).")
    return "\n".join(lines) + "\n"


def _cell(v):
    return "-" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="outputs/experiments")
    args = ap.parse_args()
    print(build_report(args.out_root))
