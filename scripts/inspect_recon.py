#!/usr/bin/env python3
"""Diagnose a trained lesion VAE's reconstructions.

Distinguishes the two failure modes behind a Dice of 0.000 at the T1 gate:
(a) true collapse: the decoder outputs near-zero probability everywhere, so
    nothing survives any threshold;
(b) threshold miss: the decoder does mark the lesion but with probabilities
    below 0.5, so the 0.5-binarised Dice is 0 while lower thresholds recover
    overlap.

For a handful of real volumes it prints the sigmoid-output statistics and the
Dice at several thresholds. Light enough for a login node (CPU, few volumes).

Usage:
    python scripts/inspect_recon.py outputs/experiments/E1/E1/seed0/lesion/vae_lesion.pt
    python scripts/inspect_recon.py <ckpt> --tier trial --n 16
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocausalpfn.utils.portability import configure_portable_runtime

configure_portable_runtime()

import numpy as np                             # noqa: E402
import torch                                   # noqa: E402

torch.set_num_threads(2)                       # login-node safe

from neurocausalpfn.data.nifti_dataset import LesionMaskDataset      # noqa: E402
from neurocausalpfn.data.paths import lesion_root                    # noqa: E402
from neurocausalpfn.train.run_stage2_real import load_vae            # noqa: E402

THRESHOLDS = (0.5, 0.3, 0.1, 0.05, 0.01)


def dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum()
    return 1.0 if float(denom) == 0 else float(2.0 * inter / denom)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="path to a vae_lesion.pt checkpoint")
    ap.add_argument("--tier", default="full", choices=["trial", "full"])
    ap.add_argument("--n", type=int, default=8, help="volumes to inspect")
    args = ap.parse_args()

    model = load_vae(args.ckpt, device="cpu")
    res = tuple(torch.load(args.ckpt, map_location="cpu")["cfg"]["data"]["resolution"])
    ds = LesionMaskDataset(root=lesion_root(args.tier), in_shape=res,
                           n_synth=args.n, seed=0, binarize=True)
    n = min(args.n, len(ds))
    print(f"checkpoint: {args.ckpt}")
    print(f"data: {lesion_root(args.tier)} ({'synthetic fallback' if ds.synthetic else 'real'}), "
          f"{n} volumes at {res}")

    probs_max, probs_mean_fg, dices = [], [], {t: [] for t in THRESHOLDS}
    with torch.no_grad():
        for i in range(n):
            item = ds[i]
            x = (item[0] if isinstance(item, (tuple, list)) else item).unsqueeze(0)
            logits, _, _, _ = model(x)
            p = torch.sigmoid(logits)[0, 0]
            t = x[0, 0]
            probs_max.append(float(p.max()))
            fg = p[t > 0.5]
            probs_mean_fg.append(float(fg.mean()) if fg.numel() else float("nan"))
            for thr in THRESHOLDS:
                dices[thr].append(dice((p > thr).float(), t))

    print(f"\nsigmoid output:  max per volume: {np.nanmean(probs_max):.4f} "
          f"(range {min(probs_max):.4f} to {max(probs_max):.4f})")
    print(f"mean prob INSIDE the true lesion: {np.nanmean(probs_mean_fg):.4f}")
    print("\nDice by binarisation threshold (mean over volumes):")
    for thr in THRESHOLDS:
        print(f"  p > {thr:<4}: {np.mean(dices[thr]):.4f}")
    print("\nreading: max prob ~0 everywhere -> true collapse (the loss never "
          "rewarded the foreground); good Dice at low thresholds only -> the "
          "signal exists but sits under 0.5, a calibration rather than a "
          "capacity problem.")


if __name__ == "__main__":
    main()
