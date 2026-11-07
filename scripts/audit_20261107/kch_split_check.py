"""Brief 9 task 5 (and point 8): checks before a KCH split becomes the fresh
test. CPU, seconds. Aggregates only: counts per network, no lesion, row or
file name is printed or written to the CSVs.

1. One lesion per patient? The pool has no patient column, so the check is
   indirect: identical latent rows and identical overlap rows (exact repeats
   of one lesion), and, when the cache holds a ``files`` listing and
   --patient-regex is given (a regex whose first group is the patient id in a
   file name), the number of files against distinct patients. The regex and
   the names stay on the machine; only the two counts are written.
2. A fixed random split of the pool, by lesion (or by patient when the regex
   is given, so a patient never sits in both halves), into a training half A
   and a test half B (--seed, default 0). Writes A and B as pool npz files
   (every key sliced by row; scalar metadata kept) for R0 training on A
   (sweep_axis.sh with POOL=<A>, POOL_MD5=<printed md5>), and prints their
   md5.
3. Trial sizes per network and split in each half and in the whole pool: the
   lesions that enter trial (split, network), i.e. cover either half above
   its threshold (round(5 % of the half's voxels), the benchmark's rule, on the
   cache's overlap fractions), split into only-half-1, only-half-2 and both.
   Point 8: a test half of about 600 lesions may leave small trials.

    python scripts/audit_20261107/kch_split_check.py --pool $W/pools/kch_e13a_s0/fold0.npz \\
        --atlas-dir ~/export_bcb/atlases --out-dir $W/pools/kch_e13a_s0_split --report <dir>
"""
import argparse
import glob
import hashlib
import os
import re

import numpy as np

SPLITS = ("receptor", "genetics")


def half_thresholds(atlas_dir, split):
    """[16, 2] hit thresholds (fractions) of halves 1 and 2 of each network."""
    import nibabel as nib

    out = np.zeros((16, 2))
    for f in glob.glob(os.path.join(atlas_dir, "2mm_parcellations", split, "*.nii*")):
        k = int(os.path.basename(f).split("_")[0])
        d = np.rint(np.asarray(nib.load(f).dataobj)).astype(int)
        for h, lab in ((0, 1), (1, 2)):
            v = int((d == lab).sum())
            out[k - 1, h] = int(np.round(v * 0.05)) / v
    return out


def trial_sizes(ov, thr):
    """ov [m, 16, 2] overlap fractions; thr [16, 2]."""
    hit = ov > thr[None]
    a, b = hit[:, :, 0], hit[:, :, 1]
    return {"enrolled": (a | b).sum(0), "only_half1": (a & ~b).sum(0), "only_half2": (b & ~a).sum(0),
            "both": (a & b).sum(0)}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="where the half pools A and B are written (stays on the machine)")
    ap.add_argument("--report", required=True, help="where the aggregate CSVs go")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patient-regex", default=None)
    a = ap.parse_args()

    import pandas as pd

    with np.load(a.pool, allow_pickle=True) as z:
        cache = {k: z[k] for k in z.files}
    m = len(cache["overlaps"])
    rowkeys = [k for k, v in cache.items() if np.asarray(v).ndim >= 1 and len(v) == m]
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.report, exist_ok=True)
    rows = [("pool_md5", md5(a.pool)), ("pool_lesions", m), ("row_keys", " ".join(sorted(rowkeys)))]
    # which maps the overlaps were read from (lesion masks or disconnectomes):
    # the benchmark enrols on disconnectomes, so trial sizes depend on it
    for k in ("modality", "threshold", "channel"):
        if k in cache and np.asarray(cache[k]).ndim == 0:
            rows.append((f"cache_{k}", str(np.asarray(cache[k]).item())))

    # 1
    if "Z" in cache:
        rows.append(("distinct_latent_rows", int(len(np.unique(np.round(np.asarray(cache["Z"], float), 6), axis=0)))))
    ovs = np.concatenate([np.asarray(cache[k], float).reshape(m, -1) for k in
                          ("overlaps_network", "overlaps_receptor", "overlaps_genetics") if k in cache], axis=1)
    rows.append(("distinct_overlap_rows", int(len(np.unique(np.round(ovs, 6), axis=0)))))
    rows.append(("lesions_hitting_no_network_half", int((ovs.max(1) == 0).sum())))
    unit = np.arange(m)
    if "files" in cache:
        files = [os.path.basename(str(f)) for f in cache["files"]]
        rows.append(("files_listed", len(files)))
        rows.append(("distinct_file_names", len(set(files))))
        if a.patient_regex:
            rx = re.compile(a.patient_regex)
            ids = []
            for f in files:
                mt = rx.search(f)
                ids.append(mt.group(1) if mt else f)
            rows.append(("files_without_regex_match", sum(rx.search(f) is None for f in files)))
            rows.append(("distinct_patients", len(set(ids))))
            uniq = {p: i for i, p in enumerate(sorted(set(ids)))}
            unit = np.array([uniq[p] for p in ids])
    else:
        rows.append(("files_listed", "none in the cache: patient check not possible from the pool"))

    # 2
    rng = np.random.default_rng(a.seed)
    units = np.unique(unit)
    in_a = set(rng.permutation(units)[: len(units) // 2].tolist())
    mask_a = np.array([u in in_a for u in unit])
    for tag, mask in (("A_train", mask_a), ("B_test", ~mask_a)):
        part = {k: (v[mask] if k in rowkeys else v) for k, v in cache.items()}
        path = os.path.join(a.out_dir, f"kch_half_{tag}_seed{a.seed}.npz")
        np.savez(path, **part)
        rows.append((f"half_{tag}_lesions", int(mask.sum())))
        rows.append((f"half_{tag}_md5", md5(path)))
        print(f"wrote half {tag}: {int(mask.sum())} lesions, md5 {md5(path)}")

    # 3
    out = []
    for split in SPLITS:
        key = f"overlaps_{split}"
        if key not in cache:
            continue
        thr = half_thresholds(a.atlas_dir, split)
        ov = np.asarray(cache[key], float)
        for tag, mask in (("whole", np.ones(m, bool)), ("A_train", mask_a), ("B_test", ~mask_a)):
            ts = trial_sizes(ov[mask], thr)
            for k in range(16):
                out.append({"split": split, "network": k + 1, "part": tag,
                            **{c: int(v[k]) for c, v in ts.items()}})
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(a.report, "kch_trial_sizes.csv"), index=False)
    b = df[df["part"] == "B_test"]
    for c in ("enrolled", "only_half1", "only_half2"):
        rows.append((f"B_test_min_{c}", int(b[c].min())))
        rows.append((f"B_test_median_{c}", float(b[c].median())))
    rows.append(("B_test_trials_below_50_enrolled", int((b["enrolled"] < 50).sum())))
    rows.append(("B_test_trials_with_a_class_below_10", int(((b["only_half1"] < 10) | (b["only_half2"] < 10)).sum())))
    pd.DataFrame(rows, columns=["item", "value"]).to_csv(os.path.join(a.report, "kch_split_check.csv"), index=False)
    for k, v in rows:
        print(f"{k}: {v}")
    print(df[df["part"] == "B_test"].to_string(index=False))


if __name__ == "__main__":
    main()
