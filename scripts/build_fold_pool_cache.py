"""Per-fold anatomy pools for the Neuro-Prior from caches that already exist.

PFN k must pre-train only on the training side of fold k, encoded with fold
k's own encoder, so that no tested patient enters its pool. This script takes
the cohort's anatomy cache (scripts/build_prior_cache.py, every image of the
cohort) and the fold files of a per-fold representation (``fold<k>.npz`` with
``Ztr``, ``tr_idx`` and ``files``), and writes one cache per fold holding the
anatomy rows of the training-side images and their fold-k latents as ``Z``.
Optionally the rows of a second cohort (e.g. KCH) are appended, with latents
from the fold-k encoder given as an npz per fold (``{fold}`` in the pattern):

    python scripts/build_fold_pool_cache.py --cache outputs/prior_cache/uclh_disco_x.npz \\
        --fold-dir ~/Scratch/neuro-causal-pfn-perfold/outputs_perfold_pub/E1/E1/seed0/folds \\
        --folds 0 1 --out-dir outputs/prior_cache/fold_pools/e1_s0 \\
        [--extra-cache outputs/prior_cache/kch_e1_32d_x.npz --extra-latents "outputs/latents_pf/fold{fold}/*.npz"]

Rows are matched by file name; a training-side image without an anatomy row,
or an extra row without a latent, stops the script. Channel, threshold and
atlas modality of the two caches must agree. Prints counts only.
"""
import argparse
import glob
import os
import sys
from typing import Dict, List, Optional

import numpy as np

SCALARS = ("channel", "threshold", "modality", "networks")


def _load(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _rows(cache: Dict[str, np.ndarray]) -> List[str]:
    """Keys indexed by image (first dimension = number of files), Z excluded:
    the latents are replaced by the fold encoder's."""
    m = len(cache["files"])
    return [k for k, v in cache.items()
            if k != "Z" and not k.startswith("latents") and np.ndim(v) >= 1 and len(v) == m]


def _select(cache: Dict[str, np.ndarray], names: List[str], what: str) -> Dict[str, np.ndarray]:
    pos = {os.path.basename(str(n)): i for i, n in enumerate(cache["files"])}
    missing = [n for n in names if n not in pos]
    if missing:
        sys.exit(f"{len(missing)} of {len(names)} {what} images have no row in the anatomy cache")
    idx = [pos[n] for n in names]
    return {k: np.asarray(cache[k])[idx] for k in _rows(cache)}


def _check_same(a: Dict, b: Dict) -> None:
    for k in ("channel", "threshold", "modality"):
        if k in a and k in b and str(np.asarray(a[k]).item()) != str(np.asarray(b[k]).item()):
            sys.exit(f"the two caches differ in {k}: {np.asarray(a[k]).item()} against {np.asarray(b[k]).item()}")
    if "networks" in a and "networks" in b and list(a["networks"]) != list(b["networks"]):
        sys.exit("the two caches were built with different atlases")


def build_fold(cache: Dict, fold_path: str, out_path: str,
               extra: Optional[Dict] = None, extra_latents: Optional[str] = None) -> Dict[str, int]:
    with np.load(fold_path, allow_pickle=False) as z:
        files = [str(n) for n in z["files"]]
        tr_idx, Ztr = z["tr_idx"], np.asarray(z["Ztr"], dtype=np.float32)
        fold = int(z["fold"]) if "fold" in z.files else -1
    names = [os.path.basename(files[i]) for i in tr_idx]
    if len(names) != len(Ztr):
        sys.exit(f"{fold_path}: {len(names)} training images but {len(Ztr)} latent rows")
    out = _select(cache, names, "training-side")
    out["Z"] = Ztr
    counts = {"fold": fold, "n_main": len(names), "n_extra": 0}
    if extra is not None:
        if not extra_latents:
            sys.exit("--extra-cache needs --extra-latents")
        hits = sorted(glob.glob(extra_latents))
        if len(hits) != 1:
            sys.exit(f"{extra_latents}: {len(hits)} files match, need exactly one")
        with np.load(hits[0], allow_pickle=True) as z:
            Zx = np.asarray(z["Z"], dtype=np.float32)
            xnames = [os.path.basename(str(n)) for n in z["files"]]
        if Zx.shape[1] != Ztr.shape[1]:
            sys.exit(f"extra latents have {Zx.shape[1]} dims, the fold's have {Ztr.shape[1]}")
        xrows = _select(extra, xnames, "extra")
        for k in list(out):
            if k == "Z":
                continue
            if k not in xrows:
                sys.exit(f"extra cache lacks {k}")
            out[k] = np.concatenate([out[k], xrows[k]])
        out["Z"] = np.concatenate([Ztr, Zx])
        counts["n_extra"] = len(xnames)
    for k in SCALARS:
        if k in cache:
            out[k] = cache[k]
    srcs = [str(np.asarray(cache.get("images_dir", "")).item() or "")]
    if extra is not None:
        srcs.append(str(np.asarray(extra.get("images_dir", "")).item() or ""))
    out["images_dir"] = np.array(" + ".join(s for s in srcs if s))
    out["fold"] = np.array(fold)
    out["fold_file"] = np.array(os.path.abspath(fold_path))
    out["n_main"], out["n_extra"] = np.array(counts["n_main"]), np.array(counts["n_extra"])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **out)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="anatomy cache of the cohort the folds are over")
    ap.add_argument("--fold-dir", required=True, help="folder of fold<k>.npz files")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--extra-cache", default=None, help="anatomy cache of a cohort appended to every pool")
    ap.add_argument("--extra-latents", default=None,
                    help="npz of the extra cohort's latents from the fold encoder; '{fold}' is replaced per fold")
    args = ap.parse_args(argv)

    cache = _load(args.cache)
    extra = _load(args.extra_cache) if args.extra_cache else None
    if extra is not None:
        _check_same(cache, extra)
    for k in args.folds:
        fold_path = os.path.join(args.fold_dir, f"fold{k}.npz")
        if not os.path.exists(fold_path):
            sys.exit(f"no fold file for fold {k} in {args.fold_dir}")
        pattern = args.extra_latents.replace("{fold}", str(k)) if args.extra_latents else None
        out_path = os.path.join(args.out_dir, f"fold{k}.npz")
        c = build_fold(cache, fold_path, out_path, extra, pattern)
        print(f"fold {k}: {c['n_main']} training-side rows"
              + (f" + {c['n_extra']} extra" if c["n_extra"] else "") + f" -> {out_path}")


if __name__ == "__main__":
    main()
