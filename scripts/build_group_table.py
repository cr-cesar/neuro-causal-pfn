"""Derive a minimal grouping table for group-aware folds (trial).

Inputs: a crosswalk mapping the public filenames to the original filenames
(a_file, b_file; from match_lesion_datasets.py) and a metadata table that
carries, per original file, a subject key and an acquisition date. Output: one
row per public file with an OPAQUE group id (a salted hash of the subject key)
and the acquisition rank inside the group (0 = earliest). Nothing else from
the metadata table is copied, so the output can travel to the cluster.

    python scripts/build_group_table.py crosswalk_orig.csv info.csv \\
        --path-col <path column> --id-col <subject column> --date-col <date column> \\
        --salt "<a long secret phrase, never committed>" -o groups_public.csv \\
        --expect-groups 2830 --expect-extra 1289

The path column is reduced to its basename (with and without the NIfTI
extension) before matching the crosswalk's b_file. Dates are compared as
strings in YYYYMMDD form; ties fall back to the filename so the rank is
deterministic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import defaultdict


def _strip_ext(name: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def _keys(name: str):
    base = os.path.basename(str(name).replace("\\", "/"))
    return {base, _strip_ext(base)}


def build(crosswalk_rows, info_rows, path_col, id_col, date_col, salt, rank_col=None,
          all_rows=None, singles_only=False):
    """all_rows: the FULL metadata table (before any selection); when given,
    n_all = acquisitions of the subject in it is written per row, and with
    singles_only the rank-0 (testable) status is granted only to subjects with
    a single acquisition there; every other subject becomes train-only."""
    n_all = {}
    if all_rows is not None:
        for r in all_rows:
            sid = str(r[id_col]).strip()
            n_all[sid] = n_all.get(sid, 0) + 1
    info = {}
    for r in info_rows:
        first = ""
        if rank_col:
            first = "0" if str(r.get(rank_col, "")).strip() in ("1", "True", "true", "yes") else "1"
        for k in _keys(r[path_col]):
            info[k] = (str(r[id_col]).strip(), first + str(r[date_col]).strip())
    # one public file may match several originals (the same mask stored under
    # more than one folder): resolve per public file, require one subject
    hits_by_public = defaultdict(list)
    for r in crosswalk_rows:
        hit = next((info[k] for k in _keys(r["b_file"]) if k in info), None)
        if hit is not None:
            hits_by_public[r["a_file"]].append(hit)
        else:
            hits_by_public.setdefault(r["a_file"], [])
    linked, unmatched, conflicts = [], [], []
    for a_file, hits in hits_by_public.items():
        if not hits:
            unmatched.append(a_file)
            continue
        subjects = {sid for sid, _ in hits}
        if len(subjects) > 1:
            conflicts.append(a_file)
        sid, date = sorted(hits, key=lambda t: t[1])[0]        # earliest date wins
        linked.append((a_file, sid, date))
    build.conflicts = conflicts
    by_subject = defaultdict(list)
    for a_file, sid, date in linked:
        by_subject[sid].append((date, a_file))
    out = []
    for sid, items in by_subject.items():
        group = hashlib.sha256((salt + "|" + sid).encode()).hexdigest()[:12]
        n_full = n_all.get(sid, len(items)) if all_rows is not None else len(items)
        offset = 1 if (singles_only and n_full != 1) else 0   # train-only: no rank-0 image
        for rank, (_, a_file) in enumerate(sorted(items)):
            out.append({"filename": a_file, "group": group, "rank": rank + offset,
                        "n_in_group": len(items), "n_all": n_full})
    out.sort(key=lambda d: d["filename"])
    return out, unmatched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crosswalk", help="a_file (public), b_file (original) from match_lesion_datasets.py")
    ap.add_argument("info", help="metadata table with the subject key and the date")
    ap.add_argument("--path-col", required=True)
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--date-col", required=True)
    ap.add_argument("--salt", required=True, help="secret phrase; keep it out of every repository")
    ap.add_argument("--sep", default=None, help="delimiter of the metadata table (auto: tab or comma)")
    ap.add_argument("--all-info", default=None,
                    help="the FULL metadata table (before selection): adds n_all per row")
    ap.add_argument("--singles-only", action="store_true",
                    help="with --all-info: only subjects with one acquisition in the full table "
                         "are testable (rank 0); the others are train-only in every fold")
    ap.add_argument("--rank-col", default=None,
                    help="optional column that already marks each group's earliest image (truthy "
                         "value); when given it overrides the date ordering")
    ap.add_argument("-o", "--out", default="groups_public.csv")
    ap.add_argument("--expect-groups", type=int, default=None)
    ap.add_argument("--expect-extra", type=int, default=None)
    args = ap.parse_args()

    with open(args.crosswalk, newline="") as f:
        cw = list(csv.DictReader(f))
    with open(args.info, newline="") as f:
        head = f.read(8192)
        f.seek(0)
        sep = args.sep or ("\t" if head.count("\t") > head.count(",") else ",")
        info = list(csv.DictReader(f, delimiter=sep))
    for col in (args.path_col, args.id_col, args.date_col):
        if col not in info[0]:
            sys.exit(f"column {col!r} not in {args.info}; columns: {list(info[0])}")

    all_rows = None
    if args.all_info:
        with open(args.all_info, newline="") as f:
            head = f.read(8192)
            f.seek(0)
            sep_all = args.sep or ("\t" if head.count("\t") > head.count(",") else ",")
            all_rows = list(csv.DictReader(f, delimiter=sep_all))
    rows, unmatched = build(cw, info, args.path_col, args.id_col, args.date_col, args.salt,
                            rank_col=args.rank_col, all_rows=all_rows,
                            singles_only=args.singles_only)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "group", "rank", "n_in_group", "n_all"])
        w.writeheader()
        w.writerows(rows)

    n_groups = len({r["group"] for r in rows})
    n_extra = sum(1 for r in rows if r["rank"] > 0)
    n_testable = len(rows) - n_extra
    if all_rows is not None:
        singles = sum(1 for r in rows if r["n_all"] == 1)
        print(f"subjects' acquisitions in the full table: {singles} rows from single-acquisition "
              f"subjects, {len(rows) - singles} from multi-acquisition subjects")
    print(f"testable rows (rank 0) {n_testable} | train-only rows {n_extra}")
    sizes = defaultdict(int)
    for r in rows:
        if r["rank"] == 0:
            sizes[r["n_in_group"]] += 1
    print(f"crosswalk rows {len(cw)} | public files {len(rows) + len(unmatched)} | "
          f"linked {len(rows)} | unmatched {len(unmatched)} | "
          f"subject conflicts {len(getattr(build, 'conflicts', []))}")
    print(f"groups {n_groups} | repeat images (rank > 0) {n_extra}")
    print("group sizes:", ", ".join(f"{k}: {v}" for k, v in sorted(sizes.items())))
    if args.expect_groups is not None:
        print(f"expected groups {args.expect_groups}: {'OK' if n_groups == args.expect_groups else 'DIFFERS'}")
    if args.expect_extra is not None:
        print(f"expected repeats {args.expect_extra}: {'OK' if n_extra == args.expect_extra else 'DIFFERS'}")
    if unmatched:
        print("first unmatched:", unmatched[:5])
    print(f"wrote {args.out} (filename, group, rank, n_in_group, n_all only)")


if __name__ == "__main__":
    main()
