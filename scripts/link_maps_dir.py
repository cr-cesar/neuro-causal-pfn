"""A data folder whose disconnectomes are another set of maps of the same lesions.

The per-fold scorer reads ``<data>/lesions`` and ``<data>/disconnectomes`` and
matches images to the stored folds by their sorted listing, and the group
table by file name. To score the existing folds on trials defined by other
maps (F42: BCBtoolkit maps of the same lesions), this builds a folder of
symlinks that keeps the received file names, each pointing at the map of the
same lesion in the other set:

    python scripts/link_maps_dir.py --data "data/Full data" --maps ~/Scratch/uclh_disco \\
        --out ~/Scratch/data_bcb/Full\\ data

Every received disconnectome must have a map in ``--maps`` (matched by the
lesion id of the file name), else nothing is written. Prints counts only.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.disco_prior_match import lesion_key, list_maps  # noqa: E402


def build(data: str, maps: str, out: str) -> int:
    received = list_maps(os.path.join(data, "disconnectomes"))
    lesions = os.path.join(data, "lesions")
    if not received or not os.path.isdir(lesions):
        raise SystemExit(f"{data}: need lesions/ and a non-empty disconnectomes/")
    by_key = {}
    for f in list_maps(maps):
        k = lesion_key(f)
        if k in by_key:
            raise SystemExit(f"{maps}: two maps share lesion id {k}")
        by_key[k] = f
    missing = [f for f in received if lesion_key(f) not in by_key]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(received)} received maps have no map in {maps}; nothing written")
    disco = os.path.join(out, "disconnectomes")
    os.makedirs(disco, exist_ok=True)
    link = os.path.join(out, "lesions")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.abspath(lesions), link)
    for f in received:
        dst = os.path.join(disco, os.path.basename(f))
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(by_key[lesion_key(f)]), dst)
    return len(received)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the tier folder holding lesions/ and disconnectomes/")
    ap.add_argument("--maps", required=True, help="folder of the other maps of the same lesions")
    ap.add_argument("--out", required=True, help="tier folder to create (lesions/ link + disconnectomes/ links)")
    args = ap.parse_args(argv)
    n = build(args.data, args.maps, args.out)
    print(f"{args.out}: lesions linked, {n} disconnectome links to {args.maps}")


if __name__ == "__main__":
    main()
