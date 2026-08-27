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


def _built_in_representations(labels_df, files, which):
    reps = {}
    if "volume" in which:
        v = labels_df["vol"].to_numpy(dtype=float)
        reps["volume"] = (v[:, None] / max(v.max(), 1.0))
    if "nmf50" in which:
        import nibabel as nib
        from scipy import sparse
        from sklearn.decomposition import NMF
        rows, cols, n = [], [], len(files)
        shape = None
        for i, path in enumerate(files):
            img = nib.load(path).get_fdata()
            shape = img.shape
            nz = np.flatnonzero(img.ravel() > gr.DISCO_THRESH)
            rows.append(np.full(len(nz), i)); cols.append(nz)
        X = sparse.csr_matrix((np.ones(sum(map(len, cols)), dtype=np.float32),
                               (np.concatenate(rows), np.concatenate(cols))),
                              shape=(n, int(np.prod(shape))))
        k = min(50, n - 1)
        reps["nmf50"] = NMF(n_components=k, init="nndsvd", max_iter=200,
                            random_state=0, tol=1e-3).fit_transform(X)
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
    ap.add_argument("--builtin", nargs="*", default=["volume"],
                    choices=["volume", "nmf50"], help="reference representations")
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

    reps = _built_in_representations(labels, files, args.builtin)
    for path in args.latents:
        with np.load(path) as z:
            Z = z["Z"]
        if len(Z) != len(files):
            sys.exit(f"{path}: Z has {len(Z)} rows but there are {len(files)} images")
        reps[os.path.basename(path)] = Z

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
              f"(CI {agg['ci_low']:.3f}-{agg['ci_high']:.3f}, {agg['n_deficits']} deficits)")

    pd.concat(all_results).to_csv(os.path.join(args.out, "replica_results.csv"), index=False)
    pd.DataFrame(headline_rows).to_csv(os.path.join(args.out, "replica_headline.csv"), index=False)
    pd.DataFrame(sims).to_csv(os.path.join(args.out, "simulations.csv"), index=False)
    print(f"escritos en {args.out}: replica_results.csv, replica_headline.csv, "
          f"simulations.csv ({len(sims)} filas de simulacion)")


if __name__ == "__main__":
    main()
