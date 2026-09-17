"""Active latent dimensions per channel for E5 (ARD prior), from frozen checkpoints.

runs.jsonl stored only the joint active-dimension count over the concatenated
200-dim latent (189 +/- 1); it did not persist the per-dimension KL vector, so
the split between the lesion channel and the disconnectome channel is recomputed
here from the frozen checkpoints (no retraining). For each seed it loads
vae_lesion.pt and vae_disconnectome.pt, encodes the cohort, forms each channel's
ARD-aware per-dimension KL and counts the dimensions above the 0.01 threshold --
the same metric evaluate_t3 uses. The disconnectome count is the one that
matters: it is the channel the frozen chain actually ships.

Usage (short CPU/GPU job):

    python scripts/active_dims_per_channel.py [--out-root outputs/experiments] [--eid E5]

Prints a per-channel table and writes <out-root>/<eid>/active_dims.json.
Sanity anchor: lesion_active + disco_active per seed should equal the joint
count already in the leaderboard (~189).
"""
import argparse
import copy
import json
import os

import numpy as np

from neurocausalpfn.experiments import artifacts as art
from neurocausalpfn.eval.latent_quality import active_dimensions
from neurocausalpfn.train.run_stage2_real import load_vae
from neurocausalpfn.train.train_vae import _build_dataset, full_config


def _channel_active(seed_dir, representation, subdir, ckpt_name, base_cfg, device):
    ckpt = os.path.join(seed_dir, subdir, ckpt_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    model = load_vae(ckpt, device=device)
    cfg = copy.deepcopy(base_cfg)
    cfg["representation"] = representation
    in_shape = tuple(cfg["data"]["resolution"])
    dataset, _, _ = _build_dataset(cfg, representation, in_shape, use_daft=False)
    a = art.vae_artifacts(model, dataset, device=device,
                          batch_size=cfg["vae"]["batch_size"],
                          representation=representation, use_daft=False)
    pdk = a.per_dim_kl
    if pdk is None:
        raise RuntimeError(f"no posterior KL for {ckpt} (use_ard not restored?)")
    zdim = int(a.Z.shape[1])
    active = active_dimensions(pdk, threshold=0.01)
    return {"zdim": zdim, "active": int(active),
            "kl_mean": float(np.mean(pdk)), "kl_max": float(np.max(pdk))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="outputs/experiments")
    ap.add_argument("--eid", default="E5")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = full_config()

    per_seed = {}
    for seed in range(args.seeds):
        seed_dir = os.path.join(args.out_root, args.eid, args.eid, f"seed{seed}")
        cfg["seed"] = seed
        les = _channel_active(seed_dir, "lesion", "lesion", "vae_lesion.pt", cfg, device)
        dis = _channel_active(seed_dir, "disconnectome", "disco",
                              "vae_disconnectome.pt", cfg, device)
        joint = les["active"] + dis["active"]
        per_seed[f"seed{seed}"] = {"lesion": les, "disconnectome": dis, "joint_active": joint}
        print(f"seed {seed}  lesion {les['active']}/{les['zdim']} active | "
              f"disco {dis['active']}/{dis['zdim']} active | joint {joint}")

    def summ(channel, key="active"):
        vals = [per_seed[k][channel][key] for k in per_seed]
        return float(np.mean(vals)), float(np.std(vals))

    lm, ls = summ("lesion")
    dm, ds = summ("disconnectome")
    print(f"\nlesion channel:        {lm:.1f} +/- {ls:.2f} active of 100")
    print(f"disconnectome channel: {dm:.1f} +/- {ds:.2f} active of 100  <- the shipped chain")

    out_path = os.path.join(args.out_root, args.eid, "active_dims.json")
    with open(out_path, "w") as f:
        json.dump({"per_seed": per_seed,
                   "lesion_active_mean": lm, "disconnectome_active_mean": dm}, f, indent=2)
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
