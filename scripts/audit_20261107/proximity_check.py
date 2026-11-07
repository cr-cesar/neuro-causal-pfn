"""Brief 9 task 9 (Brief 8 section 4.8): proximity of the prior's virtual
trials to the 32 benchmark trials, measured on effect vectors, not on voxels.
CPU, no training, no UCLH image: the anatomy is the KCH pool the prior trains
on. Run under the launch environment of the prior to check (g4c_launch.sh
builds the cfg and execs this script), e.g. the final model's:

    PRIOR_CHECK_ONLY=1 PRIOR_CHECK_SCRIPT=scripts/audit_20261107/proximity_check.py \\
        PRIOR_CHECK_OUT=<dir> PRIOR_CHECK_ARGS="--draws 10000" OBJECTIVE=new FAMILY_W="0 1" \\
        CACHE=<E13a KCH pool> bash scripts/audit_20261028/g4c_launch.sh

Benchmark trial (split, network k): a pool lesion is enrolled when it covers
either half of k above the benchmark's threshold (round(5 % of the half's
voxels)); tau = +1 / -1 for one half only (the larger half is treatment 1),
0 for both halves or none. Prior draw: a process from the cohort's own
_draw_process (the trained distribution, slice rule included; the verifier
only filters allocation, which does not change tau); tau per lesion from the
process's virtual networks (first network hit, the DGP's susceptibility rule,
sign of TE (1 - RE) times the optimal arm). Each virtual network is also taken
alone (a single-network trial, the benchmark's shape).

Per benchmark trial, over the draws (the arm labels of a draw are arbitrary,
so the sign of a draw is chosen to fit best):
  - max |correlation| of the tau vectors over the whole pool, for processes
    and for single virtual networks, and the number of draws above 0.8 / 0.9;
  - the best agreement on the benchmark trial's enrolled lesions (share with
    the same tau);
  - for reference, the best |correlation| with another benchmark trial (the
    same network under the other split is the closest by construction).
A threshold on these numbers is fixed in advance by the plan, not here.

Aggregates only: no lesion, row or file name is printed or written.
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
SPLITS = ("receptor", "genetics")


def benchmark_tau(cache, atlas_dir):
    """{(split, k): tau [m]} with the benchmark's thresholds and orientation."""
    import nibabel as nib

    out = {}
    for split in SPLITS:
        ov = np.asarray(cache[f"overlaps_{split}"], dtype=float)
        for f in glob.glob(os.path.join(atlas_dir, "2mm_parcellations", split, "*.nii*")):
            k = int(os.path.basename(f).split("_")[0])
            d = np.rint(np.asarray(nib.load(f).dataobj)).astype(int)
            vol = [int((d == 1).sum()), int((d == 2).sum())]
            thr = [int(np.round(v * 0.05)) / v for v in vol]
            h = [ov[:, k - 1, j] > thr[j] for j in (0, 1)]
            big = 0 if vol[0] >= vol[1] else 1
            tau = np.zeros(len(ov))
            tau[h[big] & ~h[1 - big]] = 1.0
            tau[h[1 - big] & ~h[big]] = -1.0
            out[(split, k)] = (tau, h[0] | h[1])
    return out


def process_tau(coh, dgp, alone=None):
    """tau [m] of a virtual-trial process (vectorised susceptibility rule);
    ``alone`` = index of one of its networks taken as a single-network trial."""
    ov = coh._giles_ov(dgp)
    nets = dgp.networks if alone is None else [dgp.networks[alone]]
    m = ov.shape[0]
    tau = np.zeros(m)
    done = np.zeros(m, dtype=bool)
    eff = dgp.p_te * (1.0 - dgp.p_re)
    for n in nets:
        i = dgp.atlas.networks.index(n)
        ha, hb = ov[:, i, 0] > dgp.threshold, ov[:, i, 1] > dgp.threshold
        new = ~done & (ha | hb)
        oa = dgp.optimal_for_A[n]
        only_a, only_b = new & ha & ~hb, new & hb & ~ha
        tau[only_a] = eff * (1.0 if oa == 1 else -1.0)
        tau[only_b] = eff * (1.0 if oa == 0 else -1.0)
        done |= new
    return tau


def corr(a, b):
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--atlas-dir", default="data/atlases")
    ap.add_argument("--self-check", type=int, default=20, help="draws checked against the DGP's own cate()")
    a = ap.parse_args()

    import pandas as pd

    from neurocausalpfn.train.launch_cfg import cfg_from_env
    from neurocausalpfn.train.train_pfn import _build_prior

    cfg = cfg_from_env()
    prior, _, _ = _build_prior(cfg)
    coh = getattr(prior, "base", prior)
    caches = cfg["prior"]["cache"]
    path = caches if isinstance(caches, str) else list(caches)[0]
    with np.load(path, allow_pickle=True) as z:
        cache = {k: z[k] for k in ("overlaps_receptor", "overlaps_genetics") if k in z.files}
    bench = benchmark_tau(cache, a.atlas_dir)
    keys = sorted(bench)
    B = np.stack([bench[k][0] for k in keys])                  # [32, m]
    E = np.stack([bench[k][1] for k in keys])                  # enrolled masks
    Bc = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-12)
    os.makedirs(a.out, exist_ok=True)

    best_p = np.zeros(len(keys)); best_v = np.zeros(len(keys)); agree = np.zeros(len(keys))
    n08 = np.zeros(len(keys), dtype=int); n09 = np.zeros(len(keys), dtype=int)
    n_proc = n_vnet = 0
    for d in range(a.draws):
        fam, dgp, _ = coh._draw_process()
        if fam != "giles" or getattr(dgp, "virtual_overlaps", None) is None:
            continue
        if d < a.self_check:
            ov = coh._giles_ov(dgp)
            ref = np.array([dgp.cate(*dgp.susceptibility(o)) if dgp.susceptibility(o)[0] is not None else 0.0
                            for o in ov])
            if not np.allclose(ref, process_tau(coh, dgp)):
                sys.exit("self-check failed: the vectorised tau differs from the DGP's cate()")
        vecs = [("process", process_tau(coh, dgp))]
        vecs += [("vnet", process_tau(coh, dgp, alone=j)) for j in range(len(dgp.networks))]
        for kind, t in vecs:
            if t.std() == 0:
                continue
            tc = (t - t.mean()) / t.std()
            r = np.abs(Bc @ tc) / len(tc)                       # |correlation| with each benchmark trial
            s = np.sign(t)
            ag = np.array([max((s[E[i]] == B[i, E[i]]).mean(), (-s[E[i]] == B[i, E[i]]).mean())
                           if E[i].any() else 0.0 for i in range(len(keys))])
            if kind == "process":
                n_proc += 1
                best_p = np.maximum(best_p, r)
            else:
                n_vnet += 1
                best_v = np.maximum(best_v, r)
                n08 += r >= 0.8; n09 += r >= 0.9
            agree = np.maximum(agree, ag)
        if (d + 1) % 1000 == 0:
            print(f"  {d + 1} draws", flush=True)

    ref_bb = np.abs(np.corrcoef(B))
    np.fill_diagonal(ref_bb, 0)
    rows = []
    for i, (split, k) in enumerate(keys):
        other = "genetics" if split == "receptor" else "receptor"
        rows.append({"split": split, "network": k, "enrolled": int(E[i].sum()),
                     "max_abs_corr_process": round(float(best_p[i]), 4),
                     "max_abs_corr_virtual_network": round(float(best_v[i]), 4),
                     "virtual_networks_corr_ge_0.8": int(n08[i]), "virtual_networks_corr_ge_0.9": int(n09[i]),
                     "best_agreement_on_enrolled": round(float(agree[i]), 4),
                     "ref_corr_same_network_other_split": round(float(abs(corr(B[i], B[keys.index((other, k))]))), 4),
                     "ref_max_corr_other_benchmark_trial": round(float(ref_bb[i].max()), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "proximity_by_trial.csv"), index=False)
    summ = {"draws": a.draws, "processes": n_proc, "virtual_networks": n_vnet,
            "median_max_corr_virtual_network": float(df["max_abs_corr_virtual_network"].median()),
            "trials_with_a_virtual_network_corr_ge_0.9": int((df["virtual_networks_corr_ge_0.9"] > 0).sum()),
            "trials_with_a_virtual_network_corr_ge_0.8": int((df["virtual_networks_corr_ge_0.8"] > 0).sum())}
    pd.DataFrame([summ]).to_csv(os.path.join(a.out, "proximity_summary.csv"), index=False)
    print(df.to_string(index=False))
    print(summ)


if __name__ == "__main__":
    main()
