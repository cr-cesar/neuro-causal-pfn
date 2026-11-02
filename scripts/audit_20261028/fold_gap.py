"""Does a prior trained on one encoder's pool lose on the folds of the others? (2 Nov)

The final model's pool is KCH encoded by the fold-0 encoder; validation
pseudo-fold k is scored with fold k's encoder. A model trained on the ten
pools (one per fold encoder) should gain on folds 1 to 9 relative to fold 0
if the encoder geometry matters. Per deficit, the difference A - B of
pehe_paper (mean over seeds) is taken on the reference fold and on the other
folds; the difference-in-differences (others - reference) is tested over the
deficits, so the difficulty of each fold cancels out. Aggregates only.

    python scripts/audit_20261028/fold_gap.py --a "<V>/finalKpools10*_s*/receptor/.../replica_results.csv" \\
        --b "<V>/finalK_lin4p8M_s*/receptor/.../replica_results.csv" --label-a pools10 --label-b finalK --out gap.csv
"""
import argparse
import glob
import math

import numpy as np
import pandas as pd


def _per_fold(pattern: str, metric: str) -> pd.Series:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no file matches {pattern}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.groupby(["deficit", "fold"])[metric].mean()


def main():
    from scipy import stats

    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--metric", default="pehe_paper")
    ap.add_argument("--ref-fold", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = (_per_fold(a.a, a.metric) - _per_fold(a.b, a.metric)).dropna().unstack("fold")
    ref = d[a.ref_fold]
    others = d.drop(columns=a.ref_fold).mean(axis=1)
    did = (others - ref).to_numpy()
    n = len(did)
    se = did.std(ddof=1) / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    row = {"label_a": a.label_a, "label_b": a.label_b, "metric": a.metric, "n_deficits": n,
           "n_other_folds": d.shape[1] - 1, "diff_ref_fold": float(ref.mean()), "diff_other_folds": float(others.mean()),
           "did_mean": float(did.mean()), "did_ci_low": float(did.mean() - tcrit * se),
           "did_ci_high": float(did.mean() + tcrit * se), "did_t_p": float(stats.ttest_1samp(did, 0.0).pvalue),
           "n_deficits_did_negative": int((did < 0).sum())}
    print(f"{a.metric} {a.label_a} - {a.label_b}: fold {a.ref_fold} {row['diff_ref_fold']:+.4f}, "
          f"other folds {row['diff_other_folds']:+.4f}; others - fold {a.ref_fold} {row['did_mean']:+.4f} "
          f"[{row['did_ci_low']:+.4f}, {row['did_ci_high']:+.4f}], p {row['did_t_p']:.3g}, "
          f"{row['n_deficits_did_negative']}/{n} deficits negative")
    per_fold = d.mean(axis=0)
    print("mean difference by fold: " + " ".join(f"f{k} {v:+.4f}" for k, v in per_fold.items()))
    if a.out:
        pd.DataFrame([row]).to_csv(a.out, index=False)
        per_fold.rename("diff_mean").reset_index().to_csv(a.out.replace(".csv", "_by_fold.csv"), index=False)


if __name__ == "__main__":
    main()
