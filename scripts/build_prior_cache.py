#!/usr/bin/env python3
"""Precompute the anatomy the Neuro-Prior needs from an image directory.

For every image of --images-dir (a binary lesion mask, or a disconnectome
binarised at --threshold): overlap fractions with the two subnetworks of each
atlas network [m, K, 2], centroid [m, 3] and volume [m]; optionally the
encoder latents of the same files (an npz with Z and files, as written by
export_latents.py) aligned by basename. The result is one npz that
``NeuroPriorCohort.from_cache`` loads, so the transformer trains without the
images in memory.

The images must be the ones the latents were exported from. The overlaps
decide which lesions a process can affect, the latents are the covariates the
transformer reads, and the evaluation labels the scored images with the same
overlap rule; a cohort keeps binary lesion masks and continuous disconnectomes
under the same basenames, and a cache built from the masks pairs with
disconnectome latents without any error, with per-network responsive
fractions of about 0.01-0.12 instead of 0.1-0.8. The script therefore detects
the channel of the images (every checked volume binary -> "lesion", otherwise
"disconnectome"; neurocausalpfn/prior/channel.py), reads the channel of the
latents (their recorded image channel, else the encoder representation, else
the _disco/_lesion suffix of the file name) and stops on a mismatch unless
--allow-channel-mismatch is given. The channel, the images directory, the
threshold and the atlas modality are stored in the cache, and the per-network
responsive fractions at 5 % are printed next to the hit rate.

    python scripts/build_prior_cache.py --images-dir "data/Full data/disconnectomes" \\
        --atlas-dir data/atlases --latents outputs/latents_E1/E1_seed0_disco.npz \\
        --out outputs/prior_cache/uclh_E1_seed0.npz

--lesions-dir is the former name of --images-dir and is still accepted.
"""
import argparse
import glob
import os
import sys

import numpy as np

from neurocausalpfn.prior.channel import (CHANNELS, DISCONNECTOME, LESION, N_CHECK,
                                           REPRESENTATION_CHANNEL, format_summary, is_binary_volume,
                                           latents_channel, summarize_overlaps)

_DESCRIBE = {LESION: "binary masks", DISCONNECTOME: "continuous maps"}


def _load_latents(path):
    """(Z, basenames, metadata) of a latents npz; the metadata holds every
    key but Z and files (representation, images_dir, images_channel, ... when
    export_latents wrote them)."""
    with np.load(path, allow_pickle=True) as z:
        Z = np.asarray(z["Z"], dtype=np.float32)
        names = [os.path.basename(str(n)) for n in z["files"]]
        meta = {k: z[k] for k in z.files if k not in ("Z", "files")}
    return Z, names, meta


def _scalar(value):
    arr = np.asarray(value)
    return str(arr.item()) if arr.ndim == 0 else str(value)


def check_channels(images_channel: str, images_dir: str, latents: str, latents_meta,
                   allow_mismatch: bool) -> None:
    """Compares the channel of the images with the channel of the latents.
    Raises SystemExit on a mismatch unless ``allow_mismatch``; prints what it
    could and could not verify."""
    latent_ch, how = latents_channel(latents_meta, latents)
    rep = _scalar(latents_meta["representation"]) if "representation" in latents_meta else None
    rec = _scalar(latents_meta["images_channel"]) if "images_channel" in latents_meta else None
    if rep in REPRESENTATION_CHANNEL and rec in CHANNELS and REPRESENTATION_CHANNEL[rep] != rec:
        print(f"note: the latents were encoded from {rec} images by a {rep} encoder "
              f"(export_latents recorded both)")
    if latent_ch is None:
        print(f"warning: cannot verify the channel of the latents {latents} ({how}); "
              f"the cache is built from {images_channel} images ({_DESCRIBE[images_channel]})")
        return
    if latent_ch == images_channel:
        print(f"latents channel {latent_ch} ({how}) matches the images")
        return
    msg = (f"channel mismatch: the images of {images_dir} are {images_channel} "
           f"({_DESCRIBE[images_channel]}) but the latents {latents} come from {latent_ch} images "
           f"({how}). The prior's responsiveness, the covariates and the evaluation's labels must "
           f"come from the same channel: build the cache from the directory the latents were "
           f"exported from, or pass --allow-channel-mismatch to proceed anyway.")
    if not allow_mismatch:
        raise SystemExit(msg)
    print("warning (--allow-channel-mismatch): " + msg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", dest="images_dir", default=None,
                    help="directory of the pool's images: the one the latents were exported from")
    ap.add_argument("--lesions-dir", dest="images_dir", help="former name of --images-dir")
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--modality", default="receptor", choices=["receptor", "genetics"])
    ap.add_argument("--latents", default=None, help="npz with Z [m, d] and files (optional)")
    ap.add_argument("--threshold", type=float, default=0.5, help="binarisation of continuous maps")
    ap.add_argument("--allow-channel-mismatch", action="store_true",
                    help="build the cache although the latents come from the other channel")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if not args.images_dir:
        ap.error("--images-dir is required")
    if "--lesions-dir" in (sys.argv[1:] if argv is None else list(argv)):
        print("note: --lesions-dir is the former name of --images-dir; the cache must be built "
              "from the directory the latents were exported from")

    import nibabel as nib

    from neurocausalpfn.prior.atlas import FunctionalAtlas, _centroid
    from neurocausalpfn.prior.intersynth_atlas import compute_overlaps

    images_dir = os.path.abspath(os.path.expanduser(args.images_dir))
    files = sorted(glob.glob(os.path.join(images_dir, "*.nii*")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"no niftis in {args.images_dir}")
    latents = None
    if args.latents:
        latents = _load_latents(args.latents)
        rec_dir = _scalar(latents[2]["images_dir"]) if "images_dir" in latents[2] else ""
        if rec_dir and os.path.realpath(os.path.expanduser(rec_dir)) != os.path.realpath(images_dir):
            print(f"warning: the latents were exported from {rec_dir}, the cache is built from {images_dir}")
    atlas = FunctionalAtlas.from_dir(args.atlas_dir, modality=args.modality)
    n_check = min(N_CHECK, len(files))
    channel, all_binary = None, True
    overlaps, centroids, volumes = [], [], []
    for i, f in enumerate(files):
        raw = np.asarray(nib.load(f).get_fdata())
        if i < n_check:
            all_binary = all_binary and is_binary_volume(raw)
        if i + 1 == n_check:
            # the channel is settled before the bulk of the work, so a
            # mismatch stops the job after the first files
            channel = LESION if all_binary else DISCONNECTOME
            print(f"images: {channel} ({_DESCRIBE[channel]}; {n_check} of {len(files)} volumes checked)")
            if latents is not None:
                check_channels(channel, images_dir, args.latents, latents[2], args.allow_channel_mismatch)
        m = (raw > args.threshold).astype(np.float32)
        if m.shape != atlas.shape:
            sys.exit(f"{os.path.basename(f)}: shape {m.shape} differs from the atlas {atlas.shape}")
        overlaps.append(compute_overlaps(atlas, m)); centroids.append(_centroid(m)); volumes.append(float(m.sum()))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)
    out = {"overlaps": np.stack(overlaps), "centroids": np.stack(centroids),
           "volumes": np.asarray(volumes), "files": np.asarray([os.path.basename(f) for f in files]),
           "networks": np.asarray(atlas.networks),
           "channel": np.array(channel), "images_dir": np.array(images_dir),
           "threshold": np.array(float(args.threshold)), "modality": np.array(args.modality)}
    if latents is not None:
        Z, names, meta = latents
        pos = {n: k for k, n in enumerate(names)}
        missing = [n for n in out["files"] if n not in pos]
        if missing:
            sys.exit(f"{len(missing)} images without a latent, e.g. {missing[0]}")
        out["Z"] = Z[[pos[n] for n in out["files"]]]
        out["latents"] = np.array(os.path.abspath(args.latents))
        out["latents_channel"] = np.array(latents_channel(meta, args.latents)[0] or "")
        for key in ("representation", "checkpoint"):
            if key in meta:
                out["latents_" + key] = np.array(_scalar(meta[key]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}: {len(files)} {channel} images, {atlas.n_networks} networks, "
          + format_summary(summarize_overlaps(out["overlaps"]))
          + (f", Z {out['Z'].shape}" if "Z" in out else ""))


if __name__ == "__main__":
    main()
