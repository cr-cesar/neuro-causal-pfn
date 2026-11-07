"""Brief 9 task 4: learning curve of the latent reference, on the validation
pseudo-folds only (never the UCLH test folds).

The latent reference (dissertation: 0.243 on E1 over three seeds, benchmark
folds) is a TabPFN regressor fitted to the TRUE effect of each trial from the
latents: an oracle setting that measures what the representation holds. Here
the same oracle is refitted with its training side subsampled to n enrolled
patients per trial cell, n in --sizes plus "all", R repeats per size, to see
whether the reference is limited by sample size. A curve still falling at
"all" says the latents hold more than the trial sizes let a learner read.

Validation pseudo-folds as ncpfold.score builds them (--test-extras,
--extras-context clean): pseudo-fold k tests the never-tested images whose
patient hashes to k; its training side is stored fold k's training side
without every image of the queried patients; both sides' latents are rows of
stored fold k's Ztr (the extras train in every fold).

True effect per trial (split, network): enrolled when either half is covered
above its threshold (coverage > round(5 % of the half) voxels), tau = h1 - h0.
Coverage comes from --cov-cache (two_stage_diagnostic.py's), computed from the
maps and written there when the file does not exist (stays on the machine).

Output: per-cell PEHE (split, network, pseudo-fold, enrolled training size,
n, repeat) and a summary by n, over (a) every cell with at least n enrolled
training patients and (b) the matched cells with at least max(--sizes).
Aggregates only.

    PYTHONPATH=$PF:src python scripts/audit_20261107/learning_curve_oracle.py \\
        --rep E1=$HOME/neuro-causal-pfn-perfold/outputs_perfold_ws/E1/E1/seed0/folds \\
        --images-dir "$BCB/Full data" --atlas-dir $BCB/atlases --groups $BCB/fold_table.csv \\
        --cov-cache $HOME/p2_bcb/audit/cov_cache.npz --out <dir>
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "audit_20261028"))
import band_diagnostic as bd  # noqa: E402
import coverage_diagnostic as cd  # noqa: E402


def coverage(files, masks, cache, workers):
    if os.path.exists(cache):
        with np.load(cache) as z:
            return z["cov"]
    from multiprocessing import Pool

    print(f"{len(files)} maps; computing coverage ...", flush=True)
    with Pool(workers, initializer=cd._init, initargs=(masks,)) as pool:
        cov = np.stack(pool.map(cd._coverage, files, chunksize=16))
    np.savez_compressed(cache, cov=cov)
    return cov


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rep", required=True, metavar="TAG=FOLDS_DIR")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--cov-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000])
    ap.add_argument("--repeats", type=int, default=3, help="subsamples per size below 'all'")
    ap.add_argument("--learner", choices=("tabpfn", "gbm"), default="tabpfn")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--min-enrolled", type=int, default=15, help="as oracle_nmf_floor.py")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    from ncpfold.folds import load_groups
    from ncpfold.score import extras_folds, extras_mask

    files = sorted(glob.glob(os.path.join(glob.escape(a.images_dir), "*.nii*")))
    names = [os.path.basename(f) for f in files]
    groups = load_groups(a.groups)
    masks, keys = cd.half_masks(a.atlas_dir)
    thr = np.array([cd.hit_threshold(v) for _, v in masks])
    cov = coverage(files, masks, a.cov_cache, a.workers)
    if cov.shape != (len(files), len(keys)):
        sys.exit("the coverage cache does not match the image folder")
    hit = cov > thr

    tag, path = a.rep.split("=", 1)
    stored = bd.load_some_folds(path, names)
    if len(stored) != 10:
        sys.exit(f"{path}: {len(stored)} stored folds, expected 10")
    val = extras_folds([(tr, te) for _, tr, te, _, _ in stored], extras_mask(names, groups), names, groups,
                       context="clean")

    if a.learner == "tabpfn":
        import tabpfn
        from tabpfn import TabPFNRegressor

        version = getattr(tabpfn, "__version__", "unknown")
        make = lambda r: TabPFNRegressor(random_state=r, device=a.device)
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor

        version = "sklearn HGB"
        make = lambda r: HistGradientBoostingRegressor(max_iter=200, random_state=r)
    print(f"learner {a.learner} ({version}); sizes {a.sizes} + all; repeats {a.repeats}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    rows = []
    t0 = time.time()
    for k, (rest, te_k) in enumerate(val):
        _, tr_k, _, Ztr_k, _ = stored[k]
        pos = {int(i): p for p, i in enumerate(tr_k)}
        Xtr = Ztr_k[[pos[int(i)] for i in rest]]
        Xte = Ztr_k[[pos[int(i)] for i in te_k]]
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        for s in cd.SPLITS:
            for net in range(1, 17):
                j1, j0 = keys.index((s, net, "1")), keys.index((s, net, "0"))
                en_tr = hit[rest, j1] | hit[rest, j0]
                en_te = hit[te_k, j1] | hit[te_k, j0]
                N = int(en_tr.sum())
                if N < a.min_enrolled or not en_te.any():
                    continue
                itr = np.flatnonzero(en_tr)
                t_tr = hit[rest[itr], j1].astype(float) - hit[rest[itr], j0].astype(float)
                t_te = hit[te_k[en_te], j1].astype(float) - hit[te_k[en_te], j0].astype(float)
                plan = [(n, r) for n in a.sizes if n < N for r in range(a.repeats)] + [("all", 0)]
                for n, r in plan:
                    rng = np.random.default_rng(1000 * k + 37 * net + r)
                    sub = np.arange(N) if n == "all" else np.sort(rng.choice(N, size=n, replace=False))
                    y = t_tr[sub]
                    if np.all(y == y[0]):
                        pred = np.full(int(en_te.sum()), y[0])
                    else:
                        pred = make(r).fit(Xtr[itr[sub]], y).predict(Xte[en_te])
                    rows.append({"rep": tag, "split": s, "network": net, "pseudo_fold": k, "n_train_enrolled": N,
                                 "n_test_enrolled": int(en_te.sum()), "n": n if n == "all" else int(n), "repeat": r,
                                 "pehe": float(np.sqrt(((pred - t_te) ** 2).mean()))})
        print(f"  pseudo-fold {k + 1}/10 done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, f"learning_curve_cells_{tag}.csv"), index=False)
    big = max(a.sizes)
    summ = []
    for n in [*a.sizes, "all"]:
        for scope in ("cells_with_at_least_n", f"matched_cells_N_ge_{big}"):
            sel = df[df["n"].astype(str) == str(n)]
            if scope.startswith("cells") and n != "all":
                sel = sel[sel["n_train_enrolled"] > n]
            if scope.startswith("matched"):
                sel = sel[sel["n_train_enrolled"] > big]
            if sel.empty:
                continue
            cell = sel.groupby(["split", "network", "pseudo_fold"])["pehe"].mean().reset_index()
            per_split = cell.groupby("split")["pehe"].mean()
            summ.append({"rep": tag, "learner": a.learner, "version": version, "n": n, "scope": scope,
                         "n_cells": len(cell), "pehe_receptor": per_split.get("receptor", np.nan),
                         "pehe_genetics": per_split.get("genetics", np.nan), "pehe_mean": float(per_split.mean()),
                         "median_train_enrolled": float(sel.groupby(["split", "network", "pseudo_fold"])
                                                        ["n_train_enrolled"].first().median())})
    sd = pd.DataFrame(summ)
    sd.to_csv(os.path.join(a.out, f"learning_curve_summary_{tag}.csv"), index=False)
    print(sd.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
