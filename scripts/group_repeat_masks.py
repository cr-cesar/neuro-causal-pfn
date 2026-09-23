"""Group lesion masks that are repeat acquisitions of the same lesion (trial).

Two acquisitions of one lesion a few weeks apart share territory, overlap
heavily and have similar volume; the two demographic fields encoded in the
filename (``lesion{id}_{age}_{sex}``) are the same or, for the age, one year
apart. This groups such masks so that cross-validation folds can keep every
acquisition of a lesion on the same side of the split (group-aware folds).

Method: block by sex and age (|difference| <= --age-tol); inside a block,
pre-filter pairs by centroid distance and volume ratio, compute the Dice
overlap of the two binary masks, and union the pairs whose Dice reaches the
threshold. Groups use complete linkage by default (a group is a clique of
mutually overlapping masks), because single linkage chains large lesions of
different origin into giant components. All masks must share one voxel grid.

    python scripts/group_repeat_masks.py "data/Full data/lesions" -o groups.csv \\
        --dice-thr 0.3 --expect-groups 2830 --expect-extra 1289

Outputs: ``groups.csv`` (filename, group, group_size), ``groups_pairs.csv``
(every candidate pair with dice, centroid distance and volume ratio, for
review) and a threshold sweep printed to stdout so the threshold can be
calibrated against an expected group count when one is known. Merging two
distinct lesions into one group only makes folds more conservative; missing a
true repeat leaves leakage, so err on the low side of the threshold.

Requires numpy + nibabel.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from neurocausalpfn.data.clinical import parse_lesion_filename  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def dice_sorted(a: np.ndarray, b: np.ndarray) -> float:
    """Dice of two masks given as SORTED flat voxel-index arrays."""
    if a.size == 0 or b.size == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return 2.0 * inter / (a.size + b.size)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)

    def groups(self) -> np.ndarray:
        roots = [self.find(i) for i in range(len(self.parent))]
        remap = {r: k for k, r in enumerate(sorted(set(roots)))}
        return np.array([remap[r] for r in roots], dtype=int)


def blocks(ages: List[Optional[float]], sexes: List[Optional[str]], age_tol: float) -> List[Tuple[int, int]]:
    """Candidate index pairs: same sex (or unknown) and |age diff| <= tol
    (an unknown age matches any age)."""
    order = sorted(range(len(ages)), key=lambda i: (str(sexes[i]), -1.0 if ages[i] is None else ages[i]))
    pairs = []
    by_sex: Dict[str, List[int]] = {}
    for i in order:
        by_sex.setdefault("?" if sexes[i] is None else str(sexes[i]), []).append(i)
    unknown = by_sex.pop("?", [])
    for sex, idx in by_sex.items():
        members = idx + unknown
        for a in range(len(members)):
            i = members[a]
            for b in range(a + 1, len(members)):
                j = members[b]
                if ages[i] is None or ages[j] is None or abs(ages[i] - ages[j]) <= age_tol:
                    pairs.append((min(i, j), max(i, j)))
    return sorted(set(pairs))


def group_summary(groups: np.ndarray) -> Dict[str, int]:
    sizes = np.bincount(groups)
    return {"n_images": int(groups.size), "n_groups": int(sizes.size),
            "n_extra": int(groups.size - sizes.size),
            "max_group": int(sizes.max()) if sizes.size else 0,
            "n_groups_ge2": int((sizes >= 2).sum())}


def cluster(n: int, pairs, thr: float, linkage: str = "complete") -> np.ndarray:
    """Group indices from scored pairs ``(i, j, dice, ...)``.

    ``single``: union-find over every pair with dice >= thr (transitive: A~B
    and B~C put A and C together even if they do not overlap, which chains
    large territorial lesions of different origin into giant components).
    ``complete``: two groups merge only if EVERY cross pair has dice >= thr,
    so a group is a clique of mutually overlapping masks and its size stays
    at the natural repeat count."""
    if linkage == "single":
        uf = UnionFind(n)
        for i, j, d, *_ in pairs:
            if d >= thr:
                uf.union(i, j)
        return uf.groups()
    if linkage != "complete":
        raise ValueError("linkage must be 'single' or 'complete'")
    dice = {(min(i, j), max(i, j)): d for i, j, d, *_ in pairs}
    group_of = list(range(n))
    members = {i: [i] for i in range(n)}
    for i, j, d, *_ in sorted(pairs, key=lambda t: -t[2]):
        if d < thr:
            break
        gi, gj = group_of[i], group_of[j]
        if gi == gj:
            continue
        if all(dice.get((min(a, b), max(a, b)), 0.0) >= thr
               for a in members[gi] for b in members[gj]):
            keep, drop = (gi, gj) if len(members[gi]) >= len(members[gj]) else (gj, gi)
            for m in members.pop(drop):
                group_of[m] = keep
                members[keep].append(m)
    remap = {g: k for k, g in enumerate(sorted(members))}
    return np.array([remap[g] for g in group_of], dtype=int)


def dice_histogram(pairs, edges=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)) -> List[Tuple[float, float, int]]:
    d = np.array([t[2] for t in pairs], dtype=float)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append((lo, min(hi, 1.0), int(((d >= lo) & (d < hi)).sum())))
    return out


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_masks(paths: List[str]):
    import nibabel as nib

    flats, cents, vols, shape, zooms = [], [], [], None, None
    for k, p in enumerate(paths):
        img = nib.as_closest_canonical(nib.load(p))
        data = np.asanyarray(img.dataobj)
        if shape is None:
            shape, zooms = data.shape, np.asarray(img.header.get_zooms()[:3], dtype=float)
        elif data.shape != shape:
            raise SystemExit(f"{p}: shape {data.shape} differs from {shape}; one grid is required")
        mask = data > 0
        flat = np.flatnonzero(mask).astype(np.int32)
        flats.append(flat)
        vols.append(int(flat.size))
        cents.append(np.argwhere(mask).mean(axis=0) * zooms if flat.size else np.full(3, np.nan))
        if (k + 1) % 500 == 0:
            print(f"  loaded {k + 1}/{len(paths)}", flush=True)
    return flats, np.array(cents), np.array(vols), shape, zooms


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="folder of binary masks (*.nii, *.nii.gz), one grid")
    ap.add_argument("-o", "--out", default="groups.csv")
    ap.add_argument("--dice-thr", type=float, default=0.6, help="Dice at or above which two masks are grouped")
    ap.add_argument("--centroid-mm", type=float, default=20.0, help="pre-filter: max centroid distance")
    ap.add_argument("--vol-ratio", type=float, default=5.0, help="pre-filter: max volume ratio (larger/smaller)")
    ap.add_argument("--age-tol", type=float, default=1.0, help="block: max age difference (years)")
    ap.add_argument("--linkage", default="complete", choices=["complete", "single"],
                    help="complete = groups are cliques (no chaining, default); single = union-find")
    ap.add_argument("--sweep", type=float, nargs="*", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    help="Dice thresholds to report group counts for")
    ap.add_argument("--expect-groups", type=int, default=None, help="expected number of groups, if known")
    ap.add_argument("--expect-extra", type=int, default=None, help="expected number of repeat images, if known")
    ap.add_argument("--limit", type=int, default=0, help="first N files (smoke)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.directory, "*.nii*")))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no masks under {args.directory}")
    names = [os.path.basename(p) for p in paths]
    meta = [parse_lesion_filename(p) for p in paths]
    ages = [m["age"] for m in meta]
    sexes = [m["sex"] for m in meta]

    print(f"{len(paths)} masks; loading ...", flush=True)
    flats, cents, vols, shape, zooms = load_masks(paths)
    print(f"grid {shape}, voxel {tuple(float(z) for z in zooms)} mm")

    cand = blocks(ages, sexes, args.age_tol)
    print(f"{len(cand)} candidate pairs after the sex/age blocking")

    pairs = []      # (i, j, dice, dist_mm, vol_ratio)
    for n, (i, j) in enumerate(cand):
        if vols[i] == 0 or vols[j] == 0:
            continue
        ratio = max(vols[i], vols[j]) / min(vols[i], vols[j])
        if ratio > args.vol_ratio:
            continue
        dist = float(np.linalg.norm(cents[i] - cents[j]))
        if dist > args.centroid_mm:
            continue
        d = dice_sorted(flats[i], flats[j])
        if d > 0:
            pairs.append((i, j, d, dist, ratio))
        if (n + 1) % 100000 == 0:
            print(f"  {n + 1}/{len(cand)} pairs screened, {len(pairs)} overlapping", flush=True)
    print(f"{len(pairs)} overlapping pairs within the pre-filters")

    stem = os.path.splitext(args.out)[0]
    with open(stem + "_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a_file", "b_file", "dice", "centroid_mm", "vol_ratio", "age_a", "age_b", "sex_a", "sex_b"])
        for i, j, d, dist, ratio in sorted(pairs, key=lambda t: -t[2]):
            w.writerow([names[i], names[j], f"{d:.4f}", f"{dist:.2f}", f"{ratio:.3f}",
                        ages[i], ages[j], sexes[i], sexes[j]])

    print("\nDice histogram of the overlapping pairs (a separate bump at the top "
          "is the repeat-acquisition population):")
    for lo, hi, c in dice_histogram(pairs):
        print(f"  [{lo:.1f}, {hi:.1f})  {c:6d}")
    high = [t for t in pairs if t[2] >= 0.7]
    same_age = sum(1 for i, j, *_ in high if ages[i] == ages[j])
    print(f"  pairs with Dice >= 0.7: {len(high)} ({same_age} with identical age)")

    print(f"\nthreshold sweep (Dice >= thr, {args.linkage} linkage):")
    print(f"{'thr':>5s} {'groups':>7s} {'extra':>6s} {'ge2':>5s} {'max':>4s}")
    for thr in sorted(set(args.sweep) | {args.dice_thr}):
        s = group_summary(cluster(len(paths), pairs, thr, args.linkage))
        flag = " <-- chosen" if abs(thr - args.dice_thr) < 1e-9 else ""
        print(f"{thr:5.2f} {s['n_groups']:7d} {s['n_extra']:6d} {s['n_groups_ge2']:5d} {s['max_group']:4d}{flag}")
    if args.expect_groups is not None or args.expect_extra is not None:
        print(f"expected: groups {args.expect_groups}, extra {args.expect_extra}")

    groups = cluster(len(paths), pairs, args.dice_thr, args.linkage)
    sizes = np.bincount(groups)
    hist = np.bincount(sizes)
    print("group-size histogram (size: count): " +
          ", ".join(f"{k}: {int(c)}" for k, c in enumerate(hist) if k >= 1 and c > 0))
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "group", "group_size"])
        for k, name in enumerate(names):
            w.writerow([name, int(groups[k]), int(sizes[groups[k]])])
    s = group_summary(groups)
    print(f"\nwrote {args.out}: {s['n_groups']} groups for {s['n_images']} masks "
          f"({s['n_extra']} repeats, {s['n_groups_ge2']} groups with >= 2, largest {s['max_group']}); "
          f"pairs in {stem}_pairs.csv")


if __name__ == "__main__":
    main()
