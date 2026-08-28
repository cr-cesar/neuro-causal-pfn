#!/usr/bin/env python3
"""Export frozen latents from experiment checkpoints, aligned with the Giles
replica.

Loads one or more VAE checkpoints (the ``vae_{representation}.pt`` files that
the experiment runner leaves under outputs/experiments/...), encodes every
image of --images-dir with the frozen encoder, and writes one .npz per
checkpoint with the array Z. Rows follow the SORTED listing of --images-dir —
exactly the ordering scripts/run_giles_replica.py uses — so the outputs can be
fed straight to its --latents to score our encoders on the published scale
(VAE-50 disconnectome: 0.349 ideal / 0.305 location bias).

The preprocessing replicates training: pad_or_crop to the checkpoint's
resolution, binarized input for the lesion representation, continuous for the
disconnectome. Early-fusion checkpoints take the lesion from --images-dir and
the matching disconnectome (same basename) from --disco-dir.

Examples:
    python scripts/export_latents.py \\
        --checkpoints "outputs/experiments/E2/E2[w_dice=0.5]/seed0/disco/vae_disconnectome.pt" \\
        --images-dir "data/Full data/disconnectomes" --out outputs/latents

    python scripts/run_giles_replica.py --images-dir "data/Full data/disconnectomes" \\
        --atlas-dir data/atlases --latents outputs/latents/*.npz ...
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocausalpfn.utils.portability import configure_portable_runtime

configure_portable_runtime()

import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402

from neurocausalpfn.data.transforms import binarize, pad_or_crop   # noqa: E402
from neurocausalpfn.vae.conv3d_vae import ConvVAE3D      # noqa: E402


def load_frozen_vae(ckpt_path: str, device: str = "cpu"):
    """Rebuilds the ConvVAE3D of a checkpoint and returns (model, representation,
    resolution). The runner stores the full config plus the architecture flags
    inside the checkpoint, so nothing else is needed."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    rep = ckpt.get("representation", cfg.get("representation", "lesion"))
    if ckpt.get("use_daft"):
        raise SystemExit(f"{ckpt_path}: DAFT-conditioned encoder — the export "
                         "needs the clinical vector per lesion, which we do not "
                         "have yet (pending the clinical CSV from Giles' team)")
    model = ConvVAE3D(in_channels=ckpt.get("in_channels", 2 if rep == "early_fusion" else 1),
                      zdim=cfg["vae"]["zdim"],
                      in_shape=tuple(cfg["data"]["resolution"]),
                      channels=tuple(cfg["vae"]["channels"]),
                      backbone=ckpt.get("backbone", cfg["vae"].get("backbone", "cnn")),
                      use_ard=ckpt.get("use_ard", False)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, rep, tuple(cfg["data"]["resolution"])


def _load_input(path, rep, resolution, disco_dir):
    import nibabel as nib

    vol = pad_or_crop(np.asarray(nib.load(path).get_fdata(), dtype=np.float32), resolution)
    if rep == "lesion":
        return binarize(vol)[None]
    if rep == "disconnectome":
        return vol.astype(np.float32)[None]
    # early fusion: lesion channel from --images-dir, disconnectome by basename
    dpath = os.path.join(disco_dir, os.path.basename(path))
    disco = pad_or_crop(np.asarray(nib.load(dpath).get_fdata(), dtype=np.float32), resolution)
    return np.stack([binarize(vol), disco.astype(np.float32)])


@torch.no_grad()
def export_latents(model, rep, resolution, files, disco_dir=None,
                   batch_size=8, device="cpu") -> np.ndarray:
    codes = []
    for start in range(0, len(files), batch_size):
        chunk = [_load_input(p, rep, resolution, disco_dir)
                 for p in files[start:start + batch_size]]
        x = torch.from_numpy(np.stack(chunk)).to(device)
        codes.append(model.encode_mean(x).cpu().numpy())
    return np.concatenate(codes, axis=0).astype(np.float32)


def default_name(ckpt_path: str) -> str:
    """A readable npz name from the runner's layout, e.g.
    outputs/experiments/E2/E2[w_dice=0.5]/seed0/disco/vae_disconnectome.pt
      -> E2_w_dice=0.5_seed0_disco.npz"""
    parts = os.path.normpath(ckpt_path).split(os.sep)
    keep = [re.sub(r"[\[\]\s]+", "_", p).strip("_") for p in parts
            if re.match(r"^E\d|^seed\d+$|^(disco|lesion)$", p)]
    stem = "_".join(dict.fromkeys(keep)) or os.path.splitext(os.path.basename(ckpt_path))[0]
    return stem + ".npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="vae_*.pt files (globs allowed)")
    ap.add_argument("--images-dir", required=True,
                    help="same directory later given to run_giles_replica.py")
    ap.add_argument("--disco-dir", default=None,
                    help="disconnectome dir (only for early_fusion checkpoints)")
    ap.add_argument("--out", default="outputs/latents", help="output directory")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="first N images (smoke)")
    args = ap.parse_args()

    from neurocausalpfn.utils.portability import resolve_device
    device = resolve_device(args.device)

    files = sorted(glob.glob(os.path.join(args.images_dir, "*.nii*")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no niftis in {args.images_dir}")

    ckpts = []
    for pat in args.checkpoints:
        hits = sorted(glob.glob(pat))
        if not hits:
            sys.exit(f"no checkpoint matches {pat}")
        ckpts.extend(hits)

    os.makedirs(args.out, exist_ok=True)
    print(f"{len(files)} images | {len(ckpts)} checkpoint(s) | device {device}")
    for ckpt_path in ckpts:
        model, rep, resolution = load_frozen_vae(ckpt_path, device)
        if rep == "early_fusion" and not args.disco_dir:
            sys.exit(f"{ckpt_path}: early_fusion checkpoint needs --disco-dir")
        Z = export_latents(model, rep, resolution, files,
                           disco_dir=args.disco_dir,
                           batch_size=args.batch_size, device=device)
        path = os.path.join(args.out, default_name(ckpt_path))
        np.savez(path, Z=Z,
                 files=np.array([os.path.basename(f) for f in files]),
                 checkpoint=np.array(ckpt_path),
                 representation=np.array(rep))
        print(f"  {ckpt_path} ({rep}) -> {path}  Z{Z.shape}")


if __name__ == "__main__":
    main()
