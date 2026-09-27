"""Screen an external lesion cohort against the training cohort (trial).

An external validation set is only external if none of its images is linked to
the images the encoders were trained on. Three links are checked, from
cheapest to strongest:

1. exact: identical voxels (same acquisition, same segmentation);
2. near-duplicate: Dice above --dice-thr after blocking by centroid distance
   and volume ratio (same acquisition, different segmentation or export);
3. subject: optional; a table of the new cohort with a subject key is hashed
   with the SAME salt used for the training group table and intersected with
   it (same subject, any acquisition). This stage runs where the salt lives;
   only the resulting keep/drop list needs to travel.

Both image sets must be on one voxel grid (MNI 2 mm here).

    python scripts/dedup_external_cohort.py --new-dir <new masks> \\
        --ref-dir "data/Full data/lesions" -o external_screen.csv \\
        [--new-table <csv> --file-col <col> --id-col <col> --site-col <col> \\
         --salt "<same secret as the group table>" --groups outputs/groups_public.csv]

Output: one row per new image with exact_match, best_dice, best_dice_file,
subject_match, site and keep (1 = no link found). A per-site summary and
per-site best-Dice histograms are printed, with the keep counts at several
thresholds. Nothing from the new table other than the site label is copied.

    --resummarise <screen.csv> --dice-thr 0.8   re-prints the summary of a
        finished screen at another threshold without recomputing Dice.
    --new-dir X --ref-dir X --exclude-same --groups <group table>   scores
        every training image against every OTHER training subject: the
        within-cohort null distribution of best Dice between different
        patients, against which an external cohort's histogram is judged.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from group_repeat_masks import dice_sorted, load_masks  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def mask_hash(flat: np.ndarray, shape: Sequence[int]) -> str:
    """Content hash of a binary mask: sorted flat voxel indices plus the grid."""
    h = hashlib.sha1(np.asarray(shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(flat, dtype=np.int32).tobytes())
    return h.hexdigest()


def best_overlaps(new_flats: List[np.ndarray], new_cents: np.ndarray, new_vols: np.ndarray,
                  ref_flats: List[np.ndarray], ref_cents: np.ndarray, ref_vols: np.ndarray,
                  centroid_mm: float = 20.0, vol_ratio: float = 3.0,
                  new_keys: Optional[Sequence[str]] = None,
                  ref_keys: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """For every new mask, the highest Dice against the reference masks among
    the candidates within ``centroid_mm`` and a volume ratio of ``vol_ratio``.
    When ``new_keys``/``ref_keys`` are given, a reference sharing the new
    mask's (non-empty) key is skipped: this turns the screen into the
    within-cohort null (each training image against every OTHER subject).
    Returns (best_dice, best_index); index -1 when no candidate exists."""
    best_d = np.zeros(len(new_flats))
    best_i = np.full(len(new_flats), -1, dtype=int)
    ref_vols = np.asarray(ref_vols, dtype=float)
    ref_keys_arr = np.asarray(ref_keys, dtype=object) if ref_keys is not None else None
    for i, (flat, c, v) in enumerate(zip(new_flats, new_cents, new_vols)):
        if flat.size == 0 or not np.all(np.isfinite(c)):
            continue
        dist = np.linalg.norm(ref_cents - c, axis=1)
        ratio = np.maximum(ref_vols, v) / np.maximum(np.minimum(ref_vols, v), 1)
        ok = (dist <= centroid_mm) & (ratio <= vol_ratio) & (ref_vols > 0)
        if new_keys is not None and ref_keys_arr is not None and new_keys[i]:
            ok &= ref_keys_arr != new_keys[i]
        for j in np.flatnonzero(ok):
            d = dice_sorted(flat, ref_flats[j])
            if d > best_d[i]:
                best_d[i], best_i[i] = d, j
    return best_d, best_i


DICE_BINS = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01]
DICE_BIN_LABEL = "[0-.3,.3-.5,.5-.7,.7-.8,.8-.9,.9-.95,.95-1]"


def summarise(rows: List[Dict], dice_thr: float) -> Dict[str, Dict]:
    """Per-site counts (key '' = all rows) at the given Dice threshold: n,
    exact, near-dup (Dice >= thr and not exact), subject, keep, and the
    best-Dice histogram over DICE_BINS. ``rows`` are the screen CSV rows."""
    out: Dict[str, Dict] = {}
    sites = [""] + sorted({str(r.get("site", "")) for r in rows if str(r.get("site", ""))})
    for s in sites:
        sub = rows if s == "" else [r for r in rows if str(r.get("site", "")) == s]
        ex = [bool(str(r["exact_match"]).strip()) for r in sub]
        bd = np.array([float(r["best_dice"]) for r in sub], dtype=float)
        sj = [str(r.get("subject_match", "")).strip() == "1" for r in sub]
        keep = [decide(e, d, dice_thr, j) for e, d, j in zip(ex, bd, sj)]
        out[s] = {"n": len(sub), "exact": int(sum(ex)),
                  "near_dup": int(sum(1 for e, d in zip(ex, bd) if not e and d >= dice_thr)),
                  "subject": int(sum(sj)), "keep": int(sum(keep)),
                  "hist": np.histogram(bd, bins=DICE_BINS)[0].tolist() if len(sub) else []}
    return out


def print_summary(rows: List[Dict], dice_thr: float) -> None:
    summ = summarise(rows, dice_thr)
    for s, c in summ.items():
        print(f"{s or 'all':12s} n {c['n']:5d} | exact {c['exact']:4d} | "
              f"near-dup (Dice>={dice_thr}) {c['near_dup']:4d} | subject {c['subject']:4d} | "
              f"keep {c['keep']:5d}")
    print(f"best-Dice histogram {DICE_BIN_LABEL}")
    for s, c in summ.items():
        print(f"   {s or 'all':12s} {c['hist']}")
    if len(summ) > 1:
        print("keep per site at other thresholds:")
        for thr in (0.5, 0.7, 0.8, 0.9, 0.95):
            alt = summarise(rows, thr)
            print(f"   Dice>={thr:<5}" + "  ".join(f"{s or 'all'} {c['keep']}" for s, c in alt.items()))


def hash_subject(salt: str, sid: str) -> str:
    return hashlib.sha256((salt + "|" + str(sid).strip()).encode()).hexdigest()[:12]


def subject_matches(new_rows: List[Dict[str, str]], file_col: str, id_col: str, salt: str,
                    training_groups: set) -> Dict[str, bool]:
    """filename -> True when the hashed subject key of the new image is one of
    the training cohort's groups."""
    out: Dict[str, bool] = {}
    for r in new_rows:
        name = os.path.basename(str(r[file_col]).replace("\\", "/"))
        out[name] = hash_subject(salt, r[id_col]) in training_groups
    return out


def decide(exact: bool, best_dice: float, dice_thr: float, subject: Optional[bool]) -> int:
    """1 = keep for external validation, 0 = linked to the training cohort."""
    if exact or best_dice >= dice_thr or subject:
        return 0
    return 1


# --------------------------------------------------------------------------- #
def _list(d: str) -> List[str]:
    return sorted(glob.glob(os.path.join(d, "*.nii.gz")) + glob.glob(os.path.join(d, "*.nii")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-dir", required=True, help="external masks (one voxel grid)")
    ap.add_argument("--ref-dir", required=True, help="training masks, same grid")
    ap.add_argument("--dice-thr", type=float, default=0.9)
    ap.add_argument("--centroid-mm", type=float, default=20.0)
    ap.add_argument("--vol-ratio", type=float, default=3.0)
    ap.add_argument("--new-table", default=None, help="CSV of the new cohort (optional)")
    ap.add_argument("--file-col", default="filename")
    ap.add_argument("--id-col", default=None, help="subject key column of --new-table")
    ap.add_argument("--site-col", default=None, help="site label column of --new-table")
    ap.add_argument("--site-regex", default=r"^sub-([A-Za-z]+)",
                    help="regex whose group 1 is the site label inside the new filename "
                         "(default: the letters after 'sub-'); --site-col overrides it")
    ap.add_argument("--salt", default=None, help="the SAME secret used for the group table")
    ap.add_argument("--groups", default=None, help="training group table (filename, group, rank)")
    ap.add_argument("--exclude-same", action="store_true",
                    help="skip references that are the same file or, with --groups, the same "
                         "group as the new image (within-cohort null: --new-dir == --ref-dir)")
    ap.add_argument("--resummarise", default=None, metavar="CSV",
                    help="only re-print the summary of an existing screen CSV at --dice-thr")
    ap.add_argument("-o", "--out", default="external_screen.csv")
    args = ap.parse_args()

    if args.resummarise:
        with open(args.resummarise, newline="") as f:
            rows = list(csv.DictReader(f))
        print_summary(rows, args.dice_thr)
        return

    new_paths, ref_paths = _list(args.new_dir), _list(args.ref_dir)
    if not new_paths or not ref_paths:
        sys.exit("empty image folder")
    print(f"new {len(new_paths)} | reference {len(ref_paths)}; loading ...", flush=True)
    nf, nc, nv, nshape, _ = load_masks(new_paths)
    rf, rc, rv, rshape, _ = load_masks(ref_paths)
    if tuple(nshape) != tuple(rshape):
        sys.exit(f"grids differ: new {nshape} vs reference {rshape}; resample first")

    # key of an image for --exclude-same: its group when the table knows it, else its name
    new_keys = ref_keys = None
    if args.exclude_same:
        grp: Dict[str, str] = {}
        if args.groups:
            with open(args.groups, newline="") as f:
                grp = {os.path.basename(r["filename"]): r["group"] for r in csv.DictReader(f)}
        key = lambda p: grp.get(os.path.basename(p), os.path.basename(p))  # noqa: E731
        new_keys = [key(p) for p in new_paths]
        ref_keys = [key(p) for p in ref_paths]

    ref_hash: Dict[str, List[Tuple[str, str]]] = {}
    for k, (p, flat) in enumerate(zip(ref_paths, rf)):
        ref_hash.setdefault(mask_hash(flat, rshape), []).append(
            (os.path.basename(p), ref_keys[k] if ref_keys else ""))
    exact = []
    for k, flat in enumerate(nf):
        hits = ref_hash.get(mask_hash(flat, nshape), [])
        if new_keys:
            hits = [h for h in hits if h[1] != new_keys[k]]
        exact.append(hits[0][0] if hits else "")

    print("Dice stage ...", flush=True)
    best_d, best_i = best_overlaps(nf, nc, nv, rf, rc, rv, args.centroid_mm, args.vol_ratio,
                                   new_keys, ref_keys)

    site: Dict[str, str] = {}
    subj: Dict[str, bool] = {}
    if args.site_regex:
        pat = re.compile(args.site_regex)
        for p in new_paths:
            m = pat.search(os.path.basename(p))
            if m:
                site[os.path.basename(p)] = m.group(1).lower()
    if args.new_table:
        with open(args.new_table, newline="") as f:
            rows = list(csv.DictReader(f))
        if args.site_col:
            site = {os.path.basename(str(r[args.file_col]).replace("\\", "/")): r[args.site_col]
                    for r in rows}
        if args.id_col and args.salt and args.groups:
            with open(args.groups, newline="") as f:
                training = {r["group"] for r in csv.DictReader(f)}
            subj = subject_matches(rows, args.file_col, args.id_col, args.salt, training)

    out_rows = []
    for k, p in enumerate(new_paths):
        name = os.path.basename(p)
        s = subj.get(name) if subj else None
        out_rows.append({
            "filename": name, "site": site.get(name, ""),
            "n_voxels": int(nv[k]), "exact_match": exact[k],
            "best_dice": round(float(best_d[k]), 4),
            "best_dice_file": os.path.basename(ref_paths[best_i[k]]) if best_i[k] >= 0 else "",
            "subject_match": "" if s is None else int(s),
            "keep": decide(bool(exact[k]), float(best_d[k]), args.dice_thr, s)})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    print_summary(out_rows, args.dice_thr)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
