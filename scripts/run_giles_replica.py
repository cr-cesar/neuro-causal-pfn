#!/usr/bin/env python3
"""Score representations on the Giles virtual-trial scale and export the
simulations with their outcomes.

Runs the replica of Giles et al. (2025): labels each lesion against the 16
functional networks, simulates the virtual trials of the requested scenario,
fits the one-/two-model classifiers per fold, and reports PEHE on the same
scale as the published headline (VAE-50 disconnectome: 0.349 random
allocation / 0.305 location bias).

Representations scored:
  --latents one or more .npz files with array Z (rows aligned with the sorted
            file listing of --images-dir); each is a column in the report.
  built-ins: "volume" (single feature) and "nmf50" (sklearn NMF), for context.

Outputs in --out:
  replica_results.csv     per (representation, deficit, fold, classifier, learner)
  replica_headline.csv    the paper-style aggregate per representation
  simulations.csv         one row per simulated patient-trial: filename,
                          deficit, fold, y_true (ground-truth ITE), W
                          (assigned treatment), Y (observed outcome), scenario

Examples (Myriad, CPU-only — fine as a login-node-polite job or a short
qsub without GPU):
    python scripts/run_giles_replica.py \\
        --images-dir "data/Full data/disconnectomes" \\
        --atlas-dir data/atlases --modality receptor \\
        --scenario ideal \\
        --latents outputs/vae_full_disconnectome/representation_*.npz
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocausalpfn.utils.portability import configure_portable_runtime

configure_portable_runtime()

import numpy as np                                        # noqa: E402

from neurocausalpfn.prior import giles_replica as gr      # noqa: E402


def _sparse_voxel_matrix(files):
    import nibabel as nib
    from scipy import sparse
    rows, cols, n = [], [], len(files)
    shape = None
    for i, path in enumerate(files):
        img = nib.load(path).get_fdata()
        shape = img.shape
        nz = np.flatnonzero(img.ravel() > gr.DISCO_THRESH)
        rows.append(np.full(len(nz), i)); cols.append(nz)
    return sparse.csr_matrix((np.ones(sum(map(len, cols)), dtype=np.float32),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, int(np.prod(shape))))


def _nmf(k):
    from sklearn.decomposition import NMF
    return NMF(n_components=k, init="nndsvd", max_iter=200,
               random_state=0, tol=1e-3)


def _dense_voxel_matrix(files, idx, icv_mask):
    """Continuous voxel values (no binarisation) inside the ICV mask, exactly
    as Giles' reduce_nimfa_nmf loads them."""
    import nibabel as nib
    out = np.zeros((len(idx), int(icv_mask.sum())), dtype=np.float64)
    for row, i in enumerate(idx):
        out[row] = nib.load(files[i]).get_fdata().ravel()[icv_mask]
    return out


def _nimfa_nmf_builtin(files, icv_path, k):
    """Giles' exact NMF (representation.py reduce_nimfa_nmf): nimfa with
    seed='random_vcol' and library defaults, fitted per fold on the train
    side's continuous ICV-masked voxels; the test embedding is the linear
    projection X_test @ pinv(H). Deliberately NOT our sklearn variant — this
    is the faithful anchor against the published pipeline."""
    import nibabel as nib

    if not hasattr(np, "mat"):     # nimfa 1.4 predates NumPy 2.0
        np.mat = np.asmatrix
    import nimfa

    icv_mask = nib.load(icv_path).get_fdata().ravel().astype(bool)

    def _refit(tr_idx, te_idx):
        Xtr = _dense_voxel_matrix(files, tr_idx, icv_mask)
        fitted = nimfa.Nmf(V=Xtr, seed="random_vcol", rank=k)()
        W = np.asarray(fitted.fit.W)
        H = np.asarray(fitted.fit.H)
        Xte = _dense_voxel_matrix(files, te_idx, icv_mask)
        Zte = Xte @ np.linalg.pinv(H)
        return W.astype(np.float32), np.asarray(Zte, dtype=np.float32)

    return _refit


def _fold_latent_representation(fold_dir, n):
    """Per-fold latents saved by train_giles_style_vae50.py, matched to the
    replica's folds by their test indices (never by call order)."""
    by_key = {}
    for path in sorted(glob.glob(os.path.join(fold_dir, "fold*.npz"))):
        with np.load(path, allow_pickle=False) as z:
            tr, te = z["tr_idx"], z["te_idx"]
            if len(tr) + len(te) != n:
                sys.exit(f"{path}: covers {len(tr) + len(te)} images, expected {n}")
            by_key[(len(te), int(te[0]), int(te[-1]))] = (
                z["Ztr"].copy(), z["Zte"].copy(), tr.copy(), te.copy())
    if not by_key:
        sys.exit(f"no fold*.npz in {fold_dir}")

    def _lookup(tr_idx, te_idx):
        key = (len(te_idx), int(te_idx[0]), int(te_idx[-1]))
        if key not in by_key:
            sys.exit(f"{fold_dir}: no stored fold matches the requested split "
                     "(check --folds and the image listing)")
        Ztr, Zte, tr, te = by_key[key]
        if not (np.array_equal(tr, tr_idx) and np.array_equal(te, te_idx)):
            sys.exit(f"{fold_dir}: stored fold indices differ from the "
                     "replica's (different file listing or fold count)")
        return Ztr, Zte

    return _lookup


def _built_in_representations(labels_df, files, which, nmf_per_fold=False,
                              icv_path=None):
    reps = {}
    if "volume" in which:
        v = labels_df["vol"].to_numpy(dtype=float)
        reps["volume"] = (v[:, None] / max(v.max(), 1.0))
    if "nmf50_nimfa" in which:
        if not icv_path or not os.path.exists(icv_path):
            sys.exit("nmf50_nimfa needs the ICV mask: pass --icv-mask or place "
                     "icv_mask_2mm.nii.gz in the atlas dir (it ships with the "
                     "Giles repository)")
        reps["nmf50_nimfa"] = _nimfa_nmf_builtin(files, icv_path,
                                                 min(50, len(files) - 1))
    if "nmf50" in which:
        X = _sparse_voxel_matrix(files)
        k = min(50, len(files) - 1)
        if nmf_per_fold:
            # the paper's protocol: fit the reduction on the train side of each
            # fold only, transform the held-out side (no anatomy leakage)
            def _refit(tr_idx, te_idx, X=X, k=k):
                m = _nmf(k)
                Ztr = m.fit_transform(X[tr_idx])
                return Ztr.astype(np.float32), m.transform(X[te_idx]).astype(np.float32)
            reps["nmf50_perfold"] = _refit
        else:
            reps["nmf50"] = _nmf(k).fit_transform(X)
    return reps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True,
                    help="lesion or disconnectome niftis (91x109x91 MNI 2mm)")
    ap.add_argument("--atlas-dir", required=True, help="dir containing 2mm_parcellations/")
    ap.add_argument("--modality", default="receptor", choices=["receptor", "genetics"])
    ap.add_argument("--scenario", default="ideal",
                    choices=sorted(gr.HEADLINE_SCENARIOS) + ["custom"])
    ap.add_argument("--te", type=float, default=None)
    ap.add_argument("--re", type=float, default=None)
    ap.add_argument("--bias", type=float, default=None)
    ap.add_argument("--biastype", default=None, choices=["observed", "unobserved"])
    ap.add_argument("--latents", nargs="*", default=[],
                    help=".npz files with Z aligned to the sorted images")
    ap.add_argument("--fold-latents", nargs="*", default=[],
                    help="dirs of fold{K}.npz (Ztr/Zte/tr_idx/te_idx) from "
                         "per-fold encoders (train_giles_style_vae50.py); "
                         "each dir becomes one representation")
    ap.add_argument("--builtin", nargs="*", default=["volume"],
                    choices=["volume", "nmf50", "nmf50_nimfa"],
                    help="reference representations (nmf50_nimfa = Giles' exact "
                         "nimfa NMF: continuous ICV-masked voxels, per-fold fit, "
                         "pinv projection for test)")
    ap.add_argument("--nmf-per-fold", action="store_true",
                    help="refit sklearn NMF on each fold's train side only (the "
                         "paper's protocol; nmf50_nimfa is always per-fold)")
    ap.add_argument("--icv-mask", default=None,
                    help="intracranial mask nifti for nmf50_nimfa "
                         "(default: <atlas-dir>/icv_mask_2mm.nii.gz)")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--deficits", type=int, nargs="*", default=None, help="subset 1..16")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images (smoke)")
    ap.add_argument("--out", default="outputs/giles_replica")
    args = ap.parse_args()

    scenario = dict(gr.HEADLINE_SCENARIOS.get(args.scenario, gr.HEADLINE_SCENARIOS["ideal"]))
    for key, val in (("TE", args.te), ("RE", args.re), ("BIAS", args.bias),
                     ("BIASTYPE", args.biastype)):
        if val is not None:
            scenario[key] = val

    files = sorted(glob.glob(os.path.join(args.images_dir, "*.nii*")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no niftis in {args.images_dir}")
    print(f"{len(files)} images | scenario {scenario} | modality {args.modality}")

    pairs = gr.load_atlas_pairs(args.atlas_dir, args.modality)
    labels = gr.label_images(files, pairs)

    icv_path = args.icv_mask
    if icv_path is None:
        for cand in ("icv_mask_2mm.nii.gz", "icv_mask_2mm.nii"):
            p = os.path.join(args.atlas_dir, cand)
            if os.path.exists(p):
                icv_path = p
                break
    reps = _built_in_representations(labels, files, args.builtin,
                                     nmf_per_fold=args.nmf_per_fold,
                                     icv_path=icv_path)
    for path in args.latents:
        with np.load(path) as z:
            Z = z["Z"]
        if len(Z) != len(files):
            sys.exit(f"{path}: Z has {len(Z)} rows but there are {len(files)} images")
        reps[os.path.basename(path)] = Z
    for d in args.fold_latents:
        reps[os.path.basename(os.path.normpath(d))] = _fold_latent_representation(
            d, len(files))

    os.makedirs(args.out, exist_ok=True)
    import pandas as pd
    all_results, headline_rows, sims = [], [], []
    for name, Z in reps.items():
        collect = sims if not sims else None   # export the simulations once
        res = gr.evaluate_representation(Z, labels, pairs, scenario,
                                         n_folds=args.folds, deficits=args.deficits,
                                         collect_sims=collect)
        res.insert(0, "representation", name)
        all_results.append(res)
        agg = gr.headline_aggregate(res)
        headline_rows.append({"representation": name, **agg, **scenario})
        print(f"  {name:40s} PEHE {agg['pehe_mean']:.3f} "
              f"(CI {agg['ci_low']:.3f}-{agg['ci_high']:.3f}, {agg['n_deficits']} deficits, "
              f"{agg['classifier']}/{agg['learner']})")

    pd.concat(all_results).to_csv(os.path.join(args.out, "replica_results.csv"), index=False)
    pd.DataFrame(headline_rows).to_csv(os.path.join(args.out, "replica_headline.csv"), index=False)
    pd.DataFrame(sims).to_csv(os.path.join(args.out, "simulations.csv"), index=False)
    print(f"escritos en {args.out}: replica_results.csv, replica_headline.csv, "
          f"simulations.csv ({len(sims)} filas de simulacion)")


if __name__ == "__main__":
    main()
