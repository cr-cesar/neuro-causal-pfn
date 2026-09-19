"""Cross two lesion-signature CSVs (from lesion_signatures.py) to find files
present in both copies of a dataset.

Two-stage match:
 1. exact sha1 -- definitive when both copies share the voxel grid;
 2. fallback for files with no hash match: same volume (within a relative
    tolerance) AND centroid within a few mm -- a content heuristic that survives
    a grid/resolution difference between copies.

    python scripts/match_lesion_datasets.py a_sig.csv b_sig.csv -o crosswalk.csv

The crosswalk maps a_file (first CSV) <-> b_file (second CSV) so a per-file
table keyed on one copy can be joined onto the other. Eyeball the shape columns
first: if the two copies' shapes differ, the exact-hash stage cannot fire and
everything rests on the fuzzy stage.
"""
import argparse
import csv
import math


def load(path: str) -> list:
    with open(path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a_csv")
    ap.add_argument("b_csv")
    ap.add_argument("-o", "--out", default="crosswalk.csv")
    ap.add_argument("--vol-tol", type=float, default=0.02,
                    help="relative volume tolerance for the fuzzy stage")
    ap.add_argument("--centroid-tol", type=float, default=2.0,
                    help="max MNI centroid distance (mm) for the fuzzy stage")
    args = ap.parse_args()

    A, B = load(args.a_csv), load(args.b_csv)
    shapes_a = {r["shape"] for r in A}
    shapes_b = {r["shape"] for r in B}
    print(f"A={len(A)} rows, shapes {sorted(shapes_a)}")
    print(f"B={len(B)} rows, shapes {sorted(shapes_b)}")
    if shapes_a.isdisjoint(shapes_b):
        print("WARNING: no shared shape -> exact-hash stage cannot match; "
              "relying on the volume+centroid fallback only.")

    b_by_hash: dict = {}
    for r in B:
        b_by_hash.setdefault(r["sha1"], []).append(r)

    rows, matched_a, matched_b = [], set(), set()
    for r in A:                                            # stage 1: exact
        for h in b_by_hash.get(r["sha1"], []):
            rows.append({"a_file": r["filename"], "b_file": h["filename"],
                         "method": "exact_hash", "detail": "identical voxels"})
            matched_a.add(r["filename"])
            matched_b.add(h["filename"])

    rem_b = [r for r in B if r["filename"] not in matched_b and int(r["n_voxels"]) > 0]
    for r in A:                                            # stage 2: fuzzy
        if r["filename"] in matched_a or int(r["n_voxels"]) == 0:
            continue
        va = float(r["vol_mm3"])
        ca = (float(r["cx_mm"]), float(r["cy_mm"]), float(r["cz_mm"]))
        best, best_d = None, float("inf")
        for h in rem_b:
            vb = float(h["vol_mm3"])
            if va == 0 or abs(va - vb) / va > args.vol_tol:
                continue
            d = math.dist(ca, (float(h["cx_mm"]), float(h["cy_mm"]), float(h["cz_mm"])))
            if d < best_d:
                best_d, best = d, h
        if best is not None and best_d <= args.centroid_tol:
            rows.append({"a_file": r["filename"], "b_file": best["filename"],
                         "method": "vol+centroid", "detail": f"{best_d:.2f} mm"})
            matched_b.add(best["filename"])

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["a_file", "b_file", "method", "detail"])
        w.writeheader()
        w.writerows(rows)

    exact = sum(1 for x in rows if x["method"] == "exact_hash")
    fuzzy = len(rows) - exact
    print(f"matched {len(rows)} images (exact={exact}, fuzzy={fuzzy}); "
          f"of A's {len(A)}, {len(matched_a) + fuzzy} linked -> {args.out}")


if __name__ == "__main__":
    main()
