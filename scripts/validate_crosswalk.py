"""Validate a file crosswalk against a per-file attribute table (trial).

The first dataset's filenames encode two attributes (e.g. lesion0335_93_F ->
93, F); the second dataset's attribute table holds the same two fields per id.
A genuine cross-dataset match must agree on both; a coincidental geometric match
usually will not. This cross-checks every crosswalk row and flags agree /
disagree / unknown, broken down by match method, so the exact-hash rows serve as
a correctness check (they must all agree) and the fuzzy rows are filtered to the
ones that also agree on the attributes.

    python scripts/validate_crosswalk.py crosswalk.csv participants.tsv -o crosswalk_validated.csv
"""
import argparse
import csv
import re


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("crosswalk")                       # a_file,b_file,method,detail
    ap.add_argument("attrs")                           # comma-separated table with participant_id
    ap.add_argument("-o", "--out", default="crosswalk_validated.csv")
    ap.add_argument("--age-col", default="S1AgeOnArrival")
    ap.add_argument("--sex-col", default="S1Gender")
    args = ap.parse_args()

    attr = {}
    with open(args.attrs) as f:
        for r in csv.DictReader(f, delimiter=","):
            attr[r["participant_id"]] = (str(r.get(args.age_col, "")).strip(),
                                         str(r.get(args.sex_col, "")).strip())

    rows, agree, disagree, unknown = [], 0, 0, 0
    with open(args.crosswalk) as f:
        for r in csv.DictReader(f):
            m = re.search(r"_(\d+|NA)_([MF]|NA)\.nii", r["a_file"])
            pid = r["b_file"].replace("sub-", "").replace(".nii.gz", "")
            a_age, a_sex = (m.group(1), m.group(2)) if m else ("", "")
            b_age, b_sex = attr.get(pid, ("", ""))
            if not m or a_age == "NA" or pid not in attr:
                verdict, unknown = "unknown", unknown + 1
            elif a_age == b_age and a_sex == b_sex:
                verdict, agree = "agree", agree + 1
            else:
                verdict, disagree = "disagree", disagree + 1
            r.update(a_age=a_age, a_sex=a_sex, b_age=b_age, b_sex=b_sex, agecheck=verdict)
            rows.append(r)

    cols = ["a_file", "b_file", "method", "detail", "a_age", "a_sex", "b_age", "b_sex", "agecheck"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"total {len(rows)}: agree={agree} disagree={disagree} unknown={unknown} -> {args.out}")
    for meth in ("exact_hash", "vol+centroid"):
        sub = [x for x in rows if x["method"] == meth]
        a = sum(1 for x in sub if x["agecheck"] == "agree")
        d = sum(1 for x in sub if x["agecheck"] == "disagree")
        u = sum(1 for x in sub if x["agecheck"] == "unknown")
        print(f"  {meth}: agree={a} disagree={d} unknown={u} of {len(sub)}")


if __name__ == "__main__":
    main()
