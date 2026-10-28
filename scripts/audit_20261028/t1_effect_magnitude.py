"""T1: effect-magnitude columns of the validation leaderboards, per deficit plus the mean
(mean over folds per deficit, then over deficits), fold 0 and all folds. Aggregates only.

    python t1_effect_magnitude.py --run C1=<root>/{m}/.../replica_results.csv \
        --run "LR_two=<root>/{m}/.../replica_results.csv:classifier=logistic_regression,learner=two" --out t1.csv
"""
import argparse
import pandas as pd

COLS = ["tau_hat_zero", "tau_hat_pos", "tau_hat_neg", "tau_slope", "pehe_paper", "pehe_paper_rescaled", "pehe_paper_isotonic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=path_with_{m}[:col=val,col=val]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = []
    for spec in a.run:
        label, rest = spec.split("=", 1)
        path, _, flt = rest.partition(":")
        filters = dict(kv.split("=") for kv in flt.split(",")) if flt else {}
        for m in ("receptor", "genetics"):
            d = pd.read_csv(path.format(m=m))
            for k, v in filters.items():
                d = d[d[k].astype(str) == v]
            for tag, dd in (("fold0", d[d.fold == 0]), (f"{d.fold.nunique()}folds", d)):
                g = dd.groupby("deficit")[COLS].mean(); g.loc["mean"] = g.mean()
                g.insert(0, "folds", tag); g.insert(0, "split", m); g.insert(0, "model", label)
                out.append(g.reset_index())
    t = pd.concat(out); t.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(t[t.deficit == "mean"].drop(columns="deficit").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
