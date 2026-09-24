"""Link a participants table to the public image listing through the
original-filename crosswalk, and export only the clinical columns (trial).

Inputs:
  crosswalk   a_file (public), b_file (original) from match_lesion_datasets.py
  info        metadata table with the original path column and a subject key
  participants  table with a participant key and the clinical columns

The participant key and the subject key rarely share a format, so the script
tries several bridges and reports how many public files each one links:

  A  numeric part of the participant key == numeric part of the subject key
  B  numeric part of the participant key == one of the digit runs found in
     the original path (each run position is reported separately)
  C  exact string equality between the participant key and the subject key

With --emit and a chosen bridge (A, C, or B:<run index>), it writes one row per
public file: filename, site (prefix of the participant key), the requested
clinical columns, and the sum of the NIHSS items when they are among them.
No identifiers, dates or paths are copied.

    python scripts/link_participants.py crosswalk.csv info.csv participants.tsv \\
        --path-col <path col> --id-col <subject col> --pkey-col <participant col> \\
        --check
    python scripts/link_participants.py ... --emit A --cols S1AgeOnArrival,S1Gender \\
        --nihss-prefix S2Nihss -o clinical_public.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


def _read(path: str, sep: Optional[str] = None) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        head = f.read(8192)
        f.seek(0)
        s = sep or ("\t" if head.count("\t") > head.count(",") else ",")
        return list(csv.DictReader(f, delimiter=s))


def _keys(name: str):
    base = os.path.basename(str(name).replace("\\", "/"))
    out = {base}
    for ext in (".nii.gz", ".nii"):
        if base.lower().endswith(ext):
            out.add(base[: -len(ext)])
    return out


def numeric(s: str) -> Optional[int]:
    """The integer written by the digits of ``s`` (None when there are none)."""
    d = re.sub(r"\D", "", str(s))
    return int(d) if d else None


def digit_runs(path: str, min_len: int = 3) -> List[int]:
    return [int(r) for r in re.findall(r"\d{%d,}" % min_len, os.path.basename(str(path)))]


def site_of(pkey: str) -> str:
    m = re.match(r"\s*([A-Za-z]+)", str(pkey))
    return m.group(1).lower() if m else ""


def public_to_original(crosswalk_rows, info_rows, path_col, id_col) -> Dict[str, Tuple[str, str]]:
    """public filename -> (subject key, original path); the first hit wins."""
    info = {}
    for r in info_rows:
        for k in _keys(r[path_col]):
            info.setdefault(k, (str(r[id_col]).strip(), str(r[path_col])))
    out = {}
    for r in crosswalk_rows:
        hit = next((info[k] for k in _keys(r["b_file"]) if k in info), None)
        if hit is not None:
            out.setdefault(r["a_file"], hit)
    return out


def bridges(pub2orig: Dict[str, Tuple[str, str]], participants, pkey_col: str):
    """Return {bridge name: {public filename: participant row}} for every
    bridge, so their coverage can be compared before choosing one."""
    by_num: Dict[int, List[dict]] = defaultdict(list)
    by_str: Dict[str, List[dict]] = defaultdict(list)
    for p in participants:
        k = str(p[pkey_col]).strip()
        by_str[k].append(p)
        n = numeric(k)
        if n is not None:
            by_num[n].append(p)
    out: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for pub, (sid, path) in pub2orig.items():
        n = numeric(sid)
        if n is not None and len(by_num.get(n, [])) == 1:
            out["A"][pub] = by_num[n][0]
        if len(by_str.get(sid, [])) == 1:
            out["C"][pub] = by_str[sid][0]
        for i, run in enumerate(digit_runs(path)):
            if len(by_num.get(run, [])) == 1:
                out[f"B:{i}"][pub] = by_num[run][0]
    return out


def emit_rows(link: Dict[str, dict], pkey_col: str, cols: List[str], nihss_prefix: Optional[str]):
    rows = []
    for pub in sorted(link):
        p = link[pub]
        row = {"filename": pub, "site": site_of(p[pkey_col])}
        for c in cols:
            row[c] = p.get(c, "")
        if nihss_prefix:
            items = [c for c in p if c.startswith(nihss_prefix)]
            vals = [p[c] for c in items]
            complete = items and all(str(v).strip() != "" for v in vals)
            row["nihss_total"] = int(sum(float(v) for v in vals)) if complete else ""
            row["nihss_n_items"] = len(items)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crosswalk")
    ap.add_argument("info")
    ap.add_argument("participants")
    ap.add_argument("--path-col", required=True)
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--pkey-col", default="participant_id")
    ap.add_argument("--check", action="store_true", help="report the coverage of every bridge")
    ap.add_argument("--emit", default=None, help="bridge to use: A, C or B:<run index>")
    ap.add_argument("--cols", default="", help="comma-separated clinical columns to copy")
    ap.add_argument("--nihss-prefix", default=None, help="prefix of the NIHSS item columns")
    ap.add_argument("-o", "--out", default="clinical_public.csv")
    args = ap.parse_args()

    cw = _read(args.crosswalk)
    info = _read(args.info)
    parts = _read(args.participants)
    for col, table, name in ((args.path_col, info, args.info), (args.id_col, info, args.info),
                             (args.pkey_col, parts, args.participants)):
        if col not in table[0]:
            sys.exit(f"column {col!r} not in {name}; columns: {list(table[0])}")

    pub2orig = public_to_original(cw, info, args.path_col, args.id_col)
    n_pub = len({r["a_file"] for r in cw})
    print(f"public files {n_pub} | linked to an original {len(pub2orig)} | participants {len(parts)}")
    br = bridges(pub2orig, parts, args.pkey_col)
    if args.check or not args.emit:
        for name in sorted(br):
            sites = Counter(site_of(p[args.pkey_col]) for p in br[name].values())
            print(f"bridge {name:5s} links {len(br[name]):5d} public files | by site: "
                  + ", ".join(f"{k}: {v}" for k, v in sorted(sites.items())))
        if not br:
            print("no bridge links anything; the participant key does not derive from the "
                  "subject key or the original path")
    if args.emit:
        link = br.get(args.emit, {})
        if not link:
            sys.exit(f"bridge {args.emit!r} links nothing; run --check first")
        cols = [c for c in args.cols.split(",") if c]
        rows = emit_rows(link, args.pkey_col, cols, args.nihss_prefix)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        n_tot = sum(1 for r in rows if r.get("nihss_total", "") != "")
        print(f"wrote {args.out}: {len(rows)} rows, columns {list(rows[0])}"
              + (f", nihss_total present in {n_tot}" if args.nihss_prefix else ""))


if __name__ == "__main__":
    main()
