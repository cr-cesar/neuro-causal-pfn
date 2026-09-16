"""Recover the E11a equity breakdown from its trained checkpoints (no retraining).

The audit computed stratified root-PEHE per subgroup, but ``TierReport.summary``
kept only scalar metrics, so the nested ``equity`` dict never reached
runs.jsonl (fixed alongside this script). The checkpoints and the dataset are
deterministic given the seed, so the breakdown is re-derived exactly: rebuild
the datasets, encode with the stored checkpoints, and re-run the Tier-4
evaluation with the same seeds and strata.

Usage (a short GPU job; encodes 2 x 4,119 volumes per seed):

    python scripts/recover_e11a_equity.py [--out-root outputs/experiments]

Writes <out-root>/E11a/equity.json and prints the per-stratum table. The
pre-registered criterion is a worst-to-best root-PEHE ratio below 2x.
"""
import argparse
import copy
import json
import os

import numpy as np

from neurocausalpfn.experiments import artifacts as art
from neurocausalpfn.experiments.estimators import tier4_semisynthetic
from neurocausalpfn.experiments.runner import _strata_from, load_context
from neurocausalpfn.train.run_stage2_real import load_vae
from neurocausalpfn.train.train_vae import _build_dataset, full_config


def _modality_artifacts(seed_dir: str, representation: str, subdir: str,
                        ckpt_name: str, base_cfg: dict, device: str):
    ckpt = os.path.join(seed_dir, subdir, ckpt_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    model = load_vae(ckpt, device=device)
    cfg = copy.deepcopy(base_cfg)
    cfg["representation"] = representation
    in_shape = tuple(cfg["data"]["resolution"])
    dataset, _, _ = _build_dataset(cfg, representation, in_shape, use_daft=False)
    return art.vae_artifacts(model, dataset, device=device,
                             batch_size=cfg["vae"]["batch_size"],
                             representation=representation, use_daft=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="outputs/experiments")
    ap.add_argument("--eid", default="E11a")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ctx = load_context(args.out_root)
    cfg = full_config()
    # mirror the audit dispatch: winner backbone and w_dice, no DAFT/ARD/PNS
    cfg["vae"]["backbone"] = ctx.get("backbone", "resnet")
    cfg["vae"]["w_dice"] = float(ctx.get("w_dice", 1.0))

    per_seed = {}
    for seed in range(args.seeds):
        seed_dir = os.path.join(args.out_root, args.eid, args.eid, f"seed{seed}")
        cfg["seed"] = seed
        lesion = _modality_artifacts(seed_dir, "lesion", "lesion",
                                     "vae_lesion.pt", cfg, device)
        disco = _modality_artifacts(seed_dir, "disconnectome", "disco",
                                    "vae_disconnectome.pt", cfg, device)
        fused = art.VaeArtifacts(
            Z=np.concatenate([lesion.Z, disco.Z], axis=1),
            logvar=np.concatenate([lesion.logvar, disco.logvar], axis=1),
            dice=lesion.dice, bce=lesion.bce, clinical=lesion.clinical,
            volume=lesion.volume, prior_var=None, has_posterior=True,
            meta={"fusion_mode": "both", "zdim": int(lesion.Z.shape[1] + disco.Z.shape[1])})
        strata = _strata_from(fused)
        res = tier4_semisynthetic(fused.Z, seed=seed, strata=strata)
        per_seed[f"seed{seed}"] = {"root_pehe": res["root_pehe"],
                                   "equity": res["equity"]}
        print(f"seed {seed}  root_pehe={res['root_pehe']:.4f}")
        for stratum, eq in res["equity"].items():
            groups = {k: v for k, v in eq.items()
                      if k not in ("all", "max_min_ratio", "passes")}
            gtxt = "  ".join(f"{k}:{v:.4f}" for k, v in sorted(groups.items()))
            print(f"  {stratum:16s} ratio={eq['max_min_ratio']:.3f} "
                  f"{'PASS' if eq['passes'] else 'FAIL'}  ({gtxt})")

    strata_names = list(next(iter(per_seed.values()))["equity"].keys())
    means = {s: float(np.mean([per_seed[k]["equity"][s]["max_min_ratio"]
                               for k in per_seed])) for s in strata_names}
    print("mean worst-to-best ratio per stratum (criterion < 2.0):")
    for s, m in means.items():
        print(f"  {s:16s} {m:.3f}")

    out_path = os.path.join(args.out_root, args.eid, "equity.json")
    with open(out_path, "w") as f:
        json.dump({"per_seed": per_seed, "mean_max_min_ratio": means}, f, indent=2)
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
