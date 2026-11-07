"""Brief 8 audit (7 Nov), rows 2 and 15: what the final prior's virtual-trial
family actually trains on, on CPU, in minutes. Run under the final model's
launch environment (g4c_launch.sh builds the cfg and calls this script):

    PRIOR_CHECK_ONLY=1 PRIOR_CHECK_SCRIPT=scripts/audit_20261107/allocation_atoms_check.py \\
        PRIOR_CHECK_OUT=<dir> OBJECTIVE=new FAMILY_W="0 1" CACHE=<E13a KCH pool> \\
        bash scripts/audit_20261028/g4c_launch.sh

A. Allocation types, before and after the identifiability verifier. Before:
   the bias type of every virtual-trial process sample_process draws (the
   hyper's bias_weights, 0.6 : 0.3 : 0.1 unless the launch overrides them).
   After: the bias type of the processes sample_batch returns, which have
   passed verify_identifiability (an 'agnostic' process is tested with the
   true side as U; the others for positivity only). The audit's simulation
   predicts that almost no agnostic process survives, so allocation towards
   the better arm is not in the trained prior. Also the strength of the
   surviving agnostic draws, |bias - 0.5|.
B. Cross-split near-copies (row 2). Over the drawn virtual networks: the
   share whose two sides hold halves of the same network from different
   splits (receptor on one side, genetics on the other), which
   is_benchmark_pair does not block, and the share that sets a whole network
   against one of its own halves.
C. Latent-space confounding of the observed allocation types: R^2 of the
   lesion centroid (each axis) and of the volume on the latents, by 5-fold
   ridge. Below 1, 'axis' and 'severity' allocation also depend on anatomy
   the model does not see.

Aggregates only: no lesion, row or file name is printed or written.
"""
import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))


def base_cohort(prior):
    """The NeuroPriorCohort under an optional GenericEffectMixture."""
    return getattr(prior, "base", prior)


def classify_network(desc) -> set:
    out = set()
    for ca, ka, ha in desc["A"]:
        for cb, kb, hb in desc["B"]:
            if ka != kb:
                continue
            if ha in (0, 1) and hb in (0, 1) and ca != cb:
                out.add("cross_split_same_network")
            if (ha == -1) != (hb == -1):
                out.add("whole_vs_own_half")
    return out


def r2_cv(X, y, folds=5, alpha=1.0, seed=0):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict

    pred = cross_val_predict(Ridge(alpha=alpha), X, y, cv=folds)
    return float(1.0 - np.mean((y - pred) ** 2) / np.var(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=5000, help="virtual-trial processes drawn for A-before and B")
    ap.add_argument("--batches", type=int, default=100, help="batches of returned processes for A-after")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--context", type=int, default=512)
    a = ap.parse_args()

    from neurocausalpfn.train.launch_cfg import cfg_from_env
    from neurocausalpfn.train.train_pfn import _build_prior

    cfg = cfg_from_env()
    prior, _, _ = _build_prior(cfg)
    coh = base_cohort(prior)
    os.makedirs(a.out, exist_ok=True)
    rows = []

    # A, before the verifier, and B
    before, nets, kinds = Counter(), 0, Counter()
    for _ in range(a.draws):
        dgp = coh.sample_process("giles")
        before[dgp.bias_type] += 1
        for d in getattr(dgp, "virtual_desc", []) or []:
            nets += 1
            for k in classify_network(d):
                kinds[k] += 1
    for t in ("axis", "severity", "agnostic"):
        rows.append(("A_before_share_" + t, before[t] / max(1, sum(before.values()))))
    rows.append(("B_virtual_networks", nets))
    for k in ("cross_split_same_network", "whole_vs_own_half"):
        rows.append(("B_share_" + k, kinds[k] / max(1, nets)))

    # A, after the verifier
    rej0 = int(getattr(coh, "n_rejected", 0))
    after, strength, n_proc = Counter(), [], 0
    for _ in range(a.batches):
        b = prior.sample_batch(a.batch_size, n_context=a.context)
        for p in b.get("processes", []):
            n_proc += 1
            t = p.get("bias_type")
            if t is None:
                continue
            after[t] += 1
            if t == "agnostic":
                strength.append(abs(float(p.get("bias", 0.5)) - 0.5))
    tot = max(1, sum(after.values()))
    for t in ("axis", "severity", "agnostic"):
        rows.append(("A_after_share_" + t, after[t] / tot))
    rows.append(("A_after_processes", n_proc))
    rows.append(("A_rejections_during_after", int(getattr(coh, "n_rejected", 0)) - rej0))
    rows.append(("A_after_agnostic_strength_median", float(np.median(strength)) if strength else float("nan")))

    # C
    X = np.asarray(coh.X, dtype=float)
    for i, ax in enumerate("xyz"):
        rows.append((f"C_r2_centroid_{ax}_on_latents", r2_cv(X, np.asarray(coh.centroids)[:, i])))
    vol = np.asarray(getattr(coh, "volumes", np.full(len(X), np.nan)), dtype=float)
    if np.isfinite(vol).all():
        rows.append(("C_r2_log_volume_on_latents", r2_cv(X, np.log1p(vol))))

    for k, v in rows:
        print(f"{k}: {v}")
    with open(os.path.join(a.out, "allocation_atoms_check.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "value"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
