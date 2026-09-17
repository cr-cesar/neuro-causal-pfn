"""E5-disco control: ARD prior on a disconnectome-only VAE trained from scratch.

E5 applied the ARD prior with BOTH channels present (two independent VAEs) and
recorded 189/200 joint active dimensions. The shipped chain uses the
disconnectome channel only. Because the two VAEs share no parameters, the disco
encoder should behave identically whether or not the lesion VAE sits beside it --
scripts/active_dims_per_channel.py verifies that from the existing checkpoints.
This script is the stronger, dedicated control: it trains a disconnectome-only
ARD VAE from scratch (100 dims, ResNet backbone, the chain's settings) and
reports its active-dimension count, so the claim rests on an experiment, not on
an independence argument.

One seed per invocation (train an array over seeds). Writes
<out-root>/E5_disco_ard/seed{K}/active_dims_disco.json and the checkpoint.

    python scripts/e5_disco_ard_control.py --seed 0 [--zdim 100] [--out-root outputs/experiments]

Pre-registered expectation: the from-scratch disco active-dim count matches the
disconnectome share recovered by active_dims_per_channel.py, within seed noise.
A material difference would mean the joint training interacted across channels
and is itself a finding.
"""
import argparse
import json
import os

import numpy as np

from neurocausalpfn.experiments import artifacts as art
from neurocausalpfn.eval.latent_quality import active_dimensions
from neurocausalpfn.train.train_vae import _build_dataset, full_config, run_vae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--zdim", type=int, default=100)
    ap.add_argument("--out-root", default="outputs/experiments")
    ap.add_argument("--backbone", default="resnet")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, "E5_disco_ard", f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    cfg = full_config()
    cfg["seed"] = args.seed
    cfg["out_dir"] = out_dir
    cfg["export"] = False
    cfg["representation"] = "disconnectome"
    cfg["vae"]["zdim"] = args.zdim
    cfg["vae"]["use_ard"] = True
    cfg["vae"]["backbone"] = args.backbone

    model, hist = run_vae(cfg)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_shape = tuple(cfg["data"]["resolution"])
    dataset, _, _ = _build_dataset(cfg, "disconnectome", in_shape, use_daft=False)
    a = art.vae_artifacts(model, dataset, device=device,
                          batch_size=cfg["vae"]["batch_size"],
                          representation="disconnectome", use_daft=False)
    pdk = a.per_dim_kl
    if pdk is None:
        raise RuntimeError("no posterior KL (use_ard not active?)")
    result = {
        "seed": args.seed, "zdim": int(a.Z.shape[1]),
        "active_dims": int(active_dimensions(pdk, threshold=0.01)),
        "kl_mean": float(np.mean(pdk)), "kl_max": float(np.max(pdk)),
        "final_total": float(hist[-1].get("total", float("nan"))),
    }
    path = os.path.join(out_dir, "active_dims_disco.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"seed {args.seed}: disco-only ARD -> "
          f"{result['active_dims']}/{result['zdim']} active dims "
          f"(kl_mean {result['kl_mean']:.3f})")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
