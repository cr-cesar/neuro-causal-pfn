"""Clinical validation of frozen representations on a cohort with real outcomes.

For every representation (latent .npz files aligned to the sorted image listing,
plus the lesion volume and a clinical-only baseline) a logistic regression is
fitted and evaluated with repeated stratified k-fold cross-validation to predict
each binary outcome (e.g. good functional outcome, mRS 0-2 at 3 months; severe
deficit, NIHSS at 24 h above a threshold). Reported per representation and
outcome: balanced accuracy and ROC AUC (mean over folds and repeats, with the
standard deviation over repeats), the number of patients and the class balance.
Optionally every representation is also combined with the clinical covariates.

The cohort CSV needs an id column whose value is a prefix of the image file
names (``sub-stroke0001`` for ``sub-stroke0001_lesion-msk_mni2mm.nii.gz``), the
outcome columns, and the covariates; rows without the outcome are dropped for
that outcome only.

    python scripts/clinical_validation.py --cohort outputs/isles24_cohort.csv --id-col sub \
        --images-dir ~/Scratch/isles_masks_144 \
        --latents outputs/latents_isles_disco/*.npz outputs/latents_zicheng/zicheng_isles24_*D.npz \
        --outcome "mRS 3 months:<=2" --outcome "NIHSS 24h:>=5" \
        --covariates Age Sex "NIHSS at admission" --volume-col vol_mni_ml \
        --out outputs/clinical_isles.csv
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd


def parse_outcome(spec: str):
    """``"mRS 3 months:<=2"`` -> (column, function mapping values to 0/1)."""
    col, _, rule = spec.rpartition(":")
    m = re.fullmatch(r"(<=|>=|<|>|==)\s*(-?\d+(?:\.\d+)?)", rule.strip())
    if not col or not m:
        sys.exit(f"outcome spec must look like 'column:<=2', got {spec!r}")
    op, thr = m.group(1), float(m.group(2))
    fn = {"<=": lambda v: v <= thr, ">=": lambda v: v >= thr, "<": lambda v: v < thr,
          ">": lambda v: v > thr, "==": lambda v: v == thr}[op]
    return col, rule.strip(), fn


def binarise(series: pd.Series, fn):
    v = pd.to_numeric(series, errors="coerce")
    y = np.where(v.isna(), np.nan, fn(v).astype(float))
    return y


def encode_covariates(df: pd.DataFrame, cols):
    """Numeric matrix for the covariates: numbers as is, two-level strings
    (e.g. Sex M/F) as 0/1; rows with a missing covariate get the column mean."""
    out = []
    for c in cols:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() < len(s) * 0.5 and not pd.api.types.is_numeric_dtype(s):
            levels = sorted(x for x in s.dropna().unique())
            if len(levels) != 2:
                sys.exit(f"covariate {c!r} is categorical with {len(levels)} levels; encode it first")
            num = s.map({levels[0]: 0.0, levels[1]: 1.0}).astype(float)
        out.append(num.fillna(num.mean()).to_numpy(dtype=float))
    return np.column_stack(out) if out else np.zeros((len(df), 0))


def match_rows(ids, files):
    """Row index of each image in the cohort table (by id prefix of the file name)."""
    pos = {str(i): k for k, i in enumerate(ids)}
    rows = []
    for f in files:
        base = os.path.basename(f)
        hit = [i for i in pos if base.startswith(i)]
        if len(hit) != 1:
            sys.exit(f"{base}: {len(hit)} cohort rows match its id prefix")
        rows.append(pos[hit[0]])
    return np.array(rows)


def cv_scores(X, y, n_splits=10, n_repeats=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_min = int(min(np.sum(y == 1), np.sum(y == 0)))
    k = max(2, min(n_splits, n_min))
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=seed)
    bal, auc = [], []
    for tr, te in rskf.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        bal.append(balanced_accuracy_score(y[te], p >= 0.5))
        auc.append(roc_auc_score(y[te], p) if len(np.unique(y[te])) == 2 else np.nan)
    bal, auc = np.array(bal).reshape(n_repeats, k), np.array(auc).reshape(n_repeats, k)
    rb, ra = bal.mean(axis=1), np.nanmean(auc, axis=1)
    return {"bal_acc": float(rb.mean()), "bal_acc_sd": float(rb.std(ddof=1)) if n_repeats > 1 else 0.0,
            "auc": float(ra.mean()), "auc_sd": float(ra.std(ddof=1)) if n_repeats > 1 else 0.0,
            "folds": k}


def cv_regress(X, y, n_splits=10, n_repeats=5, seed=0):
    """Tier-2 probing of a continuous clinical variable: ridge regression with
    repeated k-fold cross-validation; R2 and MAE (mean, SD over repeats)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import RepeatedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    k = max(2, min(n_splits, len(y) // 5))
    rkf = RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=seed)
    r2, mae = [], []
    for tr, te in rkf.split(X):
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[tr], y[tr])
        p = m.predict(X[te])
        r2.append(r2_score(y[te], p)); mae.append(mean_absolute_error(y[te], p))
    r2, mae = np.array(r2).reshape(n_repeats, k), np.array(mae).reshape(n_repeats, k)
    rr, rm = r2.mean(axis=1), mae.mean(axis=1)
    return {"r2": float(rr.mean()), "r2_sd": float(rr.std(ddof=1)) if n_repeats > 1 else 0.0,
            "mae": float(rm.mean()), "mae_sd": float(rm.std(ddof=1)) if n_repeats > 1 else 0.0, "folds": k}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--id-col", default="sub")
    ap.add_argument("--images-dir", required=True, help="the listing the latents are aligned to")
    ap.add_argument("--latents", nargs="*", default=[])
    ap.add_argument("--outcome", action="append", default=[], help="'column:<=2' (repeatable)")
    ap.add_argument("--regress", action="append", default=[],
                    help="continuous column probed by ridge regression, R2 and MAE (repeatable; Tier 2)")
    ap.add_argument("--covariates", nargs="*", default=[])
    ap.add_argument("--volume-col", default=None)
    ap.add_argument("--with-covariates", action="store_true",
                    help="also evaluate every representation concatenated with the covariates")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.images_dir), "*.nii*")))
    if not files:
        sys.exit(f"no niftis in {args.images_dir}")
    df = pd.read_csv(args.cohort)
    rows = match_rows(df[args.id_col].astype(str).tolist(), files)
    cohort = df.iloc[rows].reset_index(drop=True)
    print(f"{len(files)} images matched to cohort rows")

    reps = {}
    if args.covariates:
        reps["clinical (" + ", ".join(args.covariates) + ")"] = encode_covariates(cohort, args.covariates)
    if args.volume_col:
        reps["volume"] = np.log1p(pd.to_numeric(cohort[args.volume_col], errors="coerce")
                                  .fillna(0).to_numpy(dtype=float))[:, None]
    for pattern in args.latents:
        for path in sorted(glob.glob(pattern)):
            with np.load(path, allow_pickle=True) as z:
                Z = np.asarray(z["Z"], dtype=float)
                if "files" in z.files:
                    names = [os.path.basename(str(n)) for n in z["files"]]
                    if names != [os.path.basename(f) for f in files]:
                        sys.exit(f"{path}: file order differs from {args.images_dir}")
            if len(Z) != len(files):
                sys.exit(f"{path}: {len(Z)} rows for {len(files)} images")
            reps[os.path.basename(path)] = Z
    if args.with_covariates and args.covariates:
        C = encode_covariates(cohort, args.covariates)
        for name in [n for n in reps if not n.startswith("clinical")]:
            reps[name + " + clinical"] = np.column_stack([reps[name], C])

    if not args.outcome and not args.regress:
        sys.exit("give at least one --outcome or --regress")
    results = []
    for col in args.regress:
        if col not in cohort.columns:
            sys.exit(f"column {col!r} not in cohort")
        v = pd.to_numeric(cohort[col], errors="coerce").to_numpy(dtype=float)
        keep = ~np.isnan(v); y = v[keep]
        print(f"\n== regress {col}: n = {int(keep.sum())}, mean {y.mean():.2f}, sd {y.std():.2f}")
        if keep.sum() < 20:
            print("   too few values; skipped"); continue
        for name, X in reps.items():
            s = cv_regress(X[keep], y, n_repeats=args.repeats)
            results.append({"outcome": f"regress {col}", "n": int(keep.sum()), "positives": np.nan,
                            "representation": name, "dim": X.shape[1], **s})
            print(f"   {name:48s} dim {X.shape[1]:4d}  R2 {s['r2']:.3f} ±{s['r2_sd']:.3f}  MAE {s['mae']:.2f} ±{s['mae_sd']:.2f}")
    for spec in args.outcome:
        col, rule, fn = parse_outcome(spec)
        if col not in cohort.columns:
            sys.exit(f"outcome column {col!r} not in cohort")
        y_all = binarise(cohort[col], fn)
        keep = ~np.isnan(y_all)
        y = y_all[keep].astype(int)
        n1 = int(y.sum()); n = int(len(y))
        print(f"\n== {col} {rule}: n = {n}, positives = {n1} ({100 * n1 / max(n, 1):.0f} %)")
        if min(n1, n - n1) < 5:
            print("   too few cases in one class; skipped"); continue
        for name, X in reps.items():
            s = cv_scores(X[keep], y, n_repeats=args.repeats)
            results.append({"outcome": f"{col} {rule}", "n": n, "positives": n1, "representation": name,
                            "dim": X.shape[1], **s})
            print(f"   {name:48s} dim {X.shape[1]:4d}  bal.acc {s['bal_acc']:.3f} ±{s['bal_acc_sd']:.3f}  "
                  f"AUC {s['auc']:.3f} ±{s['auc_sd']:.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
