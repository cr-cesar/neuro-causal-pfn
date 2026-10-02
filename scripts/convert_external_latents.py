"""Convert an externally produced embedding file into the replica's latent format.

The replica (``run_giles_replica.py --latents``) expects an .npz with a ``Z``
matrix whose rows follow the SORTED file listing of ``--images-dir``. External
collaborators send embeddings keyed by file name in their own order, so this
script aligns them by name, checks that every image has exactly one row, and
writes ``Z`` (float32), ``files`` and, when present, the per-image split label
the encoder's checkpoint had for that image (``ckpt_split``: which images the
external encoder saw during its own training; the leakage the replica cannot
remove on our side).

Optionally the embedding is reshaped as ``(tokens, dim)`` and averaged over the
tokens (``--token-mean T``), which turns a 16 x 256 flattened bottleneck into a
256-dimensional vector without any fitting.

    python scripts/convert_external_latents.py in.npz --images-dir DIR --out out.npz \
        [--embedding-key embedding --name-key file_name] [--token-mean 16] [--only-split unseen test]
"""
import argparse
import glob
import os
import sys

import numpy as np


def align(names_ext, Z_ext, image_files):
    """Rows of ``Z_ext`` reordered to follow ``image_files`` (matched by basename).
    Fails loudly on missing or duplicated names."""
    base = [os.path.basename(f) for f in image_files]
    index = {}
    for i, n in enumerate(names_ext):
        n = os.path.basename(str(n))
        if n in index:
            sys.exit(f"duplicated embedding for {n}")
        index[n] = i
    missing = [b for b in base if b not in index]
    if missing:
        sys.exit(f"{len(missing)} images without embedding, e.g. {missing[:3]}")
    extra = len(index) - len(base)
    rows = [index[b] for b in base]
    return Z_ext[rows], base, extra


def token_mean(Z, tokens):
    n, d = Z.shape
    if d % tokens:
        sys.exit(f"embedding dim {d} is not divisible by {tokens} tokens")
    return Z.reshape(n, tokens, d // tokens).mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--images-dir", required=True, help="folder whose sorted listing defines the row order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--embedding-key", default="embedding")
    ap.add_argument("--name-key", default="file_name")
    ap.add_argument("--split-key", default="ckpt_split")
    ap.add_argument("--token-mean", type=int, default=0,
                    help="reshape each row as (T, d/T) and average over T")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images_dir, "*.nii*")))
    if not files:
        sys.exit(f"no niftis in {args.images_dir}")
    z = np.load(args.npz, allow_pickle=True)
    Z = np.asarray(z[args.embedding_key], dtype=np.float32)
    names = z[args.name_key]
    split = np.asarray(z[args.split_key]).astype(str) if args.split_key in z.files else None

    Z_al, base, extra = align(names, Z, files)
    if split is not None:
        split_al, _, _ = align(names, split[:, None], files)
        split_al = split_al[:, 0]
    if args.token_mean:
        Z_al = token_mean(Z_al, args.token_mean)
    if not np.isfinite(Z_al).all():
        sys.exit("non-finite values in the embedding")

    out = {"Z": Z_al.astype(np.float32), "files": np.array(base),
           "source": np.array(os.path.basename(args.npz))}
    if split is not None:
        out["ckpt_split"] = split_al
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    msg = f"wrote {args.out}: Z {Z_al.shape} aligned to {len(base)} images"
    if extra:
        msg += f" ({extra} embeddings without image, ignored)"
    if split is not None:
        vals, cnt = np.unique(split_al, return_counts=True)
        msg += " | ckpt_split " + ", ".join(f"{v}={c}" for v, c in zip(vals, cnt))
    print(msg)


if __name__ == "__main__":
    main()
