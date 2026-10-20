#!/usr/bin/env python3
"""Which disconnectome generator reproduces the published trial sizes (F42).

Giles et al. (2025), Supplementary Table 2, give for each of the 16 networks
and both splits the number of the 4,119 patients whose disconnectome,
binarised at p > 0.5, covers more than 5% of at least one subnetwork. The
lesion masks reproduce Supplementary Table 1 exactly, but the disconnectomes
we received do not reproduce Table 2, so the published trials came from other
maps. QuickDisco (Functionnectome) offers several sets of priors; this script
draws a fixed random sample of lesion masks, builds their disconnectomes with
each candidate set, optionally reads the received maps of the same lesions as
the reference, applies the released enrolment rule (the PI's
scripts/disco_descriptives.py) and compares each network's enrolment fraction
with the published one.

    python scripts/disco_prior_match.py --lesions "data/Full data/lesions" \\
        --received "data/Full data/disconnectomes" \\
        --priors V2.P.WB=$HOME/Scratch/fcntm_priors/priors_full_proba_3T_comp9_thr0p01.h5 \\
        --priors V2.D.WB=$HOME/Scratch/fcntm_priors/priors_full_deter3T.h5 \\
        --atlas-dir data/atlases --work $HOME/Scratch/disco_prior_match \\
        --n 200 --procs 4 --out status/<day>/f42_prior_match.csv

A directory of maps already built elsewhere (for example the BCBtoolkit maps)
enters with --maps-dir LABEL=DIR; only the sampled lesions found in it count.
--sample-from DIR draws the sample only among the lesions that have a map in
DIR, so a partial set of maps built elsewhere is compared on its own patients
(with --n at least its size, all of them):

    python scripts/disco_prior_match.py --lesions "data/Full data/lesions" \\
        --received "data/Full data/disconnectomes" --maps-dir BCB178=$HOME/Scratch/uclh_disco \\
        --sample-from $HOME/Scratch/uclh_disco --n 5000 --atlas-dir data/atlases \\
        --out status/<day>/f42_bcb_match.csv

Output: aggregate enrolment fractions per network only. No file name, map or
per-patient value is printed or written, and QuickDisco's own output (which
names files) is captured, not echoed.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

N_PUBLISHED = 4119
# Giles et al. (2025), Supplementary Table 2 (disconnectomes, N = 4,119,
# threshold 0.05, p > 0.5): patients meeting the threshold for at least one
# subnetwork, and the share responsive to both, network by network.
PUBLISHED = {
    "receptor": {
        "N": [868, 688, 508, 853, 364, 978, 1503, 325, 1806, 3078, 1854, 335, 407, 1011, 1230, 1202],
        "overlap": [0.614, 0.660, 0.545, 0.573, 0.604, 0.409, 0.645, 0.698, 0.761, 0.553, 0.658,
                    0.167, 0.469, 0.408, 0.461, 0.452],
    },
    "genetics": {
        "N": [887, 851, 637, 829, 381, 955, 1346, 378, 1872, 2781, 1993, 297, 376, 863, 1394, 1427],
        "overlap": [0.512, 0.009, 0.223, 0.743, 0.588, 0.708, 0.800, 0.616, 0.675, 0.630, 0.550,
                    0.215, 0.234, 0.407, 0.157, 0.158],
    },
}
NIFTI = (".nii", ".nii.gz")


def _descriptives():
    """The PI's F33 script, so the enrolment rule is the same code."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "disco_descriptives.py")
    spec = importlib.util.spec_from_file_location("disco_descriptives", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strip(name: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def lesion_key(name: str) -> str:
    """Pairing key of a lesion or disconnectome file: the id of
    lesion{id}_{age}_{sex}, the same rule as the paired dataset of the chain
    (any 'qdisco_' or 'disco_' prefix removed first)."""
    stem = _strip(os.path.basename(name))
    for pre in ("qdisco_", "disco_"):
        if stem.startswith(pre):
            stem = stem[len(pre):]
    parts = stem.split("_")
    idp = "_".join(parts[:-2]) if len(parts) >= 3 else parts[0]
    return idp[6:] if idp.lower().startswith("lesion") else idp


def list_maps(folder: str) -> List[str]:
    folder = os.path.expanduser(folder)
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(NIFTI))


def sample(files: Sequence[str], n: int, seed: int) -> List[str]:
    if n >= len(files):
        return list(files)
    idx = np.random.default_rng(seed).choice(len(files), size=n, replace=False)
    return [files[i] for i in sorted(idx)]


def enrolment(maps: Sequence[str], luts: Dict, threshold: float) -> Dict[str, np.ndarray]:
    """Per split, the (n_maps, 16) category of every map: 0 none, 1 A only,
    2 B only, 3 both, exactly as disco_descriptives.py computes it."""
    import nibabel as nib

    cats = {kind: np.zeros((len(maps), 16), int) for kind in luts}
    for i, f in enumerate(maps):
        idx = np.flatnonzero(np.asarray(nib.load(f).dataobj).ravel() > threshold)
        for kind, (lut, thr) in luts.items():
            lab = lut[idx]
            hit = np.bincount(lab[lab >= 0], minlength=32) > thr
            cats[kind][i] = hit[0::2] * 1 + hit[1::2] * 2
    return cats


def compare(cats: Dict[str, np.ndarray], names: Sequence[str], label: str) -> List[Dict]:
    rows = []
    for kind, c in cats.items():
        n = len(c)
        for k, net in enumerate(names):
            enrolled = int((c[:, k] > 0).sum())
            both = int((c[:, k] == 3).sum())
            pub_frac = PUBLISHED[kind]["N"][k] / N_PUBLISHED
            frac = enrolled / n if n else float("nan")
            rows.append({"source": label, "split": kind, "network": net, "n_maps": n,
                         "enrolled_frac": frac, "published_frac": pub_frac,
                         "ratio": frac / pub_frac if pub_frac else float("nan"),
                         "both_share": both / enrolled if enrolled else float("nan"),
                         "published_both_share": PUBLISHED[kind]["overlap"][k]})
    return rows


def summarise(rows: List[Dict]) -> List[Dict]:
    """One line per source and split: how far the 16 fractions are from the
    published ones (median |log2 ratio|; 0 is a perfect match, 1 is a factor
    of two) and how many networks sit within 25% of them."""
    out = []
    keys = sorted({(r["source"], r["split"]) for r in rows})
    for src, kind in keys:
        rr = [r for r in rows if r["source"] == src and r["split"] == kind]
        lr = np.array([np.log2(r["ratio"]) for r in rr if r["ratio"] > 0])
        get = {r["network"]: r["ratio"] for r in rr}
        out.append({"source": src, "split": kind, "n_maps": rr[0]["n_maps"],
                    "median_abs_log2_ratio": float(np.median(np.abs(lr))) if len(lr) else float("nan"),
                    "within_25pct": int(np.sum(np.abs(lr) <= np.log2(1.25))),
                    "ratio_language": get.get("language"), "ratio_memory": get.get("memory"),
                    "ratio_aversion": get.get("aversion")})
    return out


def run_quickdisco(lesions: Sequence[str], priors: str, out_dir: str, procs: int,
                   exe: str = "QuickDisco") -> Tuple[List[str], int]:
    """Disconnectomes of the given lesions with one set of priors; reuses the
    maps already in out_dir. Returns the map paths and QuickDisco's exit code.
    QuickDisco's output names files, so it is captured and never printed."""
    os.makedirs(out_dir, exist_ok=True)
    maps = [os.path.join(out_dir, "qdisco_" + os.path.basename(f)) for f in lesions]
    todo = [f for f, m in zip(lesions, maps) if not os.path.exists(m)]
    code = 0
    if todo:
        res = subprocess.run([exe, "-i", *todo, "-p", os.path.expanduser(priors), "-o", out_dir,
                              "-pp", str(procs)], capture_output=True, text=True)
        code = res.returncode
    return [m for m in maps if os.path.exists(m)], code


def _pairs(arg: Optional[List[str]], what: str) -> List[Tuple[str, str]]:
    out = []
    for item in arg or []:
        if "=" not in item:
            raise SystemExit(f"{what} expects LABEL=PATH")
        lab, path = item.split("=", 1)
        out.append((lab.strip(), os.path.expanduser(path.strip())))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lesions", required=True, help="folder of lesion masks (MNI 2 mm)")
    ap.add_argument("--priors", action="append", help="LABEL=path of a QuickDisco priors .h5 (repeatable)")
    ap.add_argument("--maps-dir", action="append", help="LABEL=folder of maps built elsewhere (repeatable)")
    ap.add_argument("--received", default=None, help="folder of the received disconnectomes (reference)")
    ap.add_argument("--atlas-dir", required=True, help="folder containing 2mm_parcellations/")
    ap.add_argument("--work", default=os.path.expanduser("~/Scratch/disco_prior_match"))
    ap.add_argument("--sample-from", default=None,
                    help="draw the sample only among lesions that have a map in this folder")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--quickdisco", default="QuickDisco", help="QuickDisco executable")
    ap.add_argument("--out", default=None, help="csv of the per-network rows; the summary goes next to it")
    args = ap.parse_args(argv)

    import nibabel as nib

    dd = _descriptives()
    lesions_all = list_maps(args.lesions)
    if not lesions_all:
        raise SystemExit("no lesion masks in --lesions")
    if args.sample_from:
        have = {lesion_key(f) for f in list_maps(args.sample_from)}
        lesions_all = [f for f in lesions_all if lesion_key(f) in have]
        if not lesions_all:
            raise SystemExit("no lesion in --lesions has a map in --sample-from")
    lesions = sample(lesions_all, args.n, args.seed)
    keys = {lesion_key(f) for f in lesions}
    shape = nib.load(lesions[0]).shape[:3]
    luts = {kind: dd.lookup(args.atlas_dir, kind, shape) for kind in ("receptor", "genetics")}
    print(f"sample: {len(lesions)} of {len(lesions_all)} lesion masks (seed {args.seed})")

    sources: List[Tuple[str, List[str]]] = []
    if args.received:
        rec = [f for f in list_maps(args.received) if lesion_key(f) in keys]
        sources.append(("received", rec))
    for lab, folder in _pairs(args.maps_dir, "--maps-dir"):
        sources.append((lab, [f for f in list_maps(folder) if lesion_key(f) in keys]))
    for lab, h5 in _pairs(args.priors, "--priors"):
        if not os.path.exists(h5):
            print(f"{lab}: priors file not found, skipped")
            continue
        maps, code = run_quickdisco(lesions, h5, os.path.join(os.path.expanduser(args.work), lab),
                                    args.procs, args.quickdisco)
        if code:
            print(f"{lab}: QuickDisco exited with code {code}; scoring the {len(maps)} maps it wrote")
        sources.append((lab, maps))

    rows = []
    for lab, maps in sources:
        print(f"{lab}: {len(maps)} maps")
        if maps:
            rows += compare(enrolment(maps, luts, args.threshold), dd.NAMES, lab)
    if not rows:
        raise SystemExit("nothing to score")

    import pandas as pd

    summ = pd.DataFrame(summarise(rows))
    print("\nDistance from Supplementary Table 2 (median |log2 ratio|: 0 = same; "
          "within_25pct: networks out of 16):")
    print(summ.round(3).to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        summ.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
        print("wrote", args.out)
    return summ


if __name__ == "__main__":
    main()
