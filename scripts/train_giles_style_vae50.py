#!/usr/bin/env python3
"""Plan B: a per-fold VAE-50 following the paper's published description.

The released Giles repository does not include the VAE stage (representation.py
implements only the NMF and PCA reductions; prescription.py reads pre-computed
vae_means columns), so the headline representation cannot be re-run from public
code. This trains the closest thing the paper's text permits: a convolutional
VAE with the Giles ResNet blocks (our "resnet" backbone, the E1 baseline),
50 latent dimensions, trained on the TRAIN side of each replica fold with the
published budget (batch 10, min 16 / max 32 epochs, early stopping after 4
epochs without validation improvement), then used frozen to encode both sides.

Documented deviations (their training code being unpublished): our pad_or_crop
to 96x112x96, Adam 1e-4, and a beta warm-up over the first fifth of training.

One fold per invocation so the folds shard across cluster jobs:
    python scripts/train_giles_style_vae50.py \\
        --images-dir "data/Full data/disconnectomes" --fold 0 --out outputs/giles_style_vae50

Folds are KFold(n_splits, shuffle=True, random_state=0) over the SORTED file
listing - exactly the replica's folds. The output fold{K}.npz (Ztr, Zte,
tr_idx, te_idx) feeds scripts/run_giles_replica.py --fold-latents.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocausalpfn.utils.portability import configure_portable_runtime

configure_portable_runtime()

import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402
from torch.utils.data import DataLoader, Subset          # noqa: E402

from neurocausalpfn.data.nifti_dataset import LesionMaskDataset   # noqa: E402
from neurocausalpfn.utils.portability import resolve_device        # noqa: E402
from neurocausalpfn.utils.seed import set_seed                     # noqa: E402
from neurocausalpfn.vae.conv3d_vae import ConvVAE3D                # noqa: E402
from neurocausalpfn.vae.losses import vae_loss, vae_loss_mse       # noqa: E402


def fold_indices(n, fold, n_folds):
    from sklearn.model_selection import KFold
    for k, (tr, te) in enumerate(KFold(n_splits=n_folds, shuffle=True,
                                       random_state=0).split(np.arange(n))):
        if k == fold:
            return tr, te
    raise ValueError(f"fold {fold} out of range for {n_folds} folds")


@torch.no_grad()
def encode(model, dataset, idx, batch_size, device):
    model.eval()
    loader = DataLoader(Subset(dataset, list(idx)), batch_size=batch_size)
    return np.concatenate([model.encode_mean(x.to(device)).cpu().numpy()
                           for x in loader]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--fold", type=int, required=True, help="0-based fold index")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--modality", default="disconnectome",
                    choices=["disconnectome", "lesion"])
    ap.add_argument("--backbone", default="resnet")
    ap.add_argument("--batch-size", type=int, default=10)     # published
    ap.add_argument("--min-epochs", type=int, default=16)     # published
    ap.add_argument("--max-epochs", type=int, default=32)     # published
    ap.add_argument("--early-stop", type=int, default=4)      # published
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="first N images (smoke)")
    ap.add_argument("--out", default="outputs/giles_style_vae50")
    args = ap.parse_args()

    device = resolve_device(args.device)
    set_seed(args.fold)   # per-fold seed, deterministic across resubmissions

    files = sorted(glob.glob(os.path.join(args.images_dir, "*.nii*")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no niftis in {args.images_dir}")

    binarize = args.modality == "lesion"
    ds = LesionMaskDataset(root=args.images_dir, in_shape=(96, 112, 96),
                           n_synth=0, seed=0, binarize=binarize)
    if args.limit:
        ds.paths = ds.paths[:args.limit]
        ds.n = len(ds.paths)
    assert [os.path.basename(p) for p in ds.paths] == \
           [os.path.basename(f) for f in files], "dataset/replica ordering differs"

    tr_idx, te_idx = fold_indices(len(files), args.fold, args.folds)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(tr_idx))
    n_val = max(1, int(args.val_frac * len(tr_idx)))
    fit_idx, val_idx = tr_idx[perm[n_val:]], tr_idx[perm[:n_val]]
    print(f"fold {args.fold}/{args.folds}: {len(fit_idx)} fit, {len(val_idx)} val, "
          f"{len(te_idx)} test | dim {args.dim} | {args.backbone} | {device}")

    model = ConvVAE3D(in_channels=1, zdim=args.dim, in_shape=(96, 112, 96),
                      backbone=args.backbone).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = vae_loss if binarize else vae_loss_mse
    fit_loader = DataLoader(Subset(ds, list(fit_idx)), batch_size=args.batch_size,
                            shuffle=True)
    val_loader = DataLoader(Subset(ds, list(val_idx)), batch_size=args.batch_size)

    warmup = max(1, int(0.2 * args.max_epochs))
    best_val, best_state, stale = float("inf"), None, 0
    for epoch in range(args.max_epochs):
        beta = min(1.0, (epoch + 1) / warmup)
        model.train()
        for x in fit_loader:
            x = x.to(device)
            logits, mu, logvar, _ = model(x)
            loss, _ = loss_fn(logits, x, mu, logvar, beta=beta)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vals = []
            for x in val_loader:
                x = x.to(device)
                logits, mu, logvar, _ = model(x)
                l, _ = loss_fn(logits, x, mu, logvar, beta=beta)
                vals.append(float(l))
            val = float(np.mean(vals))
        if val < best_val - 1e-6:
            best_val, stale = val, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(f"  epoch {epoch + 1}/{args.max_epochs}  beta={beta:.2f}  "
              f"val={val:.5f}  best={best_val:.5f}  stale={stale}")
        if epoch + 1 >= args.min_epochs and stale >= args.early_stop:
            print("  early stop (published rule)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    Ztr = encode(model, ds, tr_idx, args.batch_size, device)
    Zte = encode(model, ds, te_idx, args.batch_size, device)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"fold{args.fold}.npz")
    np.savez(path, Ztr=Ztr, Zte=Zte, tr_idx=tr_idx, te_idx=te_idx,
             files=np.array([os.path.basename(f) for f in files]),
             fold=args.fold, n_folds=args.folds, dim=args.dim,
             modality=np.array(args.modality), backbone=np.array(args.backbone))
    print(f"escrito {path}  Ztr{Ztr.shape} Zte{Zte.shape} (best val {best_val:.5f})")


if __name__ == "__main__":
    main()
