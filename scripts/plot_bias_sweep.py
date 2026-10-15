"""Location-bias sweep as a curve (F30).

Reads the per-fold leaderboards of the ideal cell (BIAS 0) and of the
location-bias cells (0.25, 0.5, 0.75, 0.95 and 1.0) for the classifier of
Giles (LR / ET) and for CausalPFN, and plots PEHE and balanced accuracy
against BIAS with the 95 % CI over deficits. BIAS = 1 is deterministic
allocation with positivity violated, the worst case; the intermediate cells
show how fast each estimator degrades before it.

    python scripts/plot_bias_sweep.py --perfold ~/Scratch/neuro-causal-pfn-perfold \\
        --root outputs_perfold_pub --eid E1 --out outputs/bias_sweep_E1.png

Leaderboard locations (as the scenario block writes them):
    BIAS 0      <root>/leaderboard.csv            <root>_t4/leaderboard.csv
    BIAS b      <root>_location_bias_<b>/...      <root>_location_bias_<b>_t4/...
    BIAS 1      <root>_location_bias/...          <root>_location_bias_t4/...
"""
import argparse
import os
import sys

import numpy as np

CELLS = [(0.0, ""), (0.25, "_location_bias_025"), (0.5, "_location_bias_050"),
         (0.75, "_location_bias_075"), (0.95, "_location_bias_095"), (1.0, "_location_bias")]


def _row(path: str, eid: str, budget: str):
    import pandas as pd

    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["eid"].astype(str) == eid]
    if "budget" in df.columns and budget:
        df = df[df["budget"].astype(str) == budget]
    if "channel" in df.columns:
        df = df[df["channel"].astype(str) == "disconnectome"]
    return None if df.empty else df.iloc[0]


def collect(perfold: str, root: str, eid: str, budget: str):
    import pandas as pd

    rows = []
    for bias, suf in CELLS:
        for est, t4 in (("LR / ET", ""), ("CausalPFN", "_t4")):
            r = _row(os.path.join(perfold, f"{root}{suf}{t4}", "leaderboard.csv"), eid, budget)
            if r is None:
                rows.append({"bias": bias, "estimator": est, "pehe": np.nan})
                continue
            rows.append({"bias": bias, "estimator": est, "pehe": float(r["pehe_paper_mean"]),
                         "ci_low": float(r.get("pehe_paper_ci_low", np.nan)),
                         "ci_high": float(r.get("pehe_paper_ci_high", np.nan)),
                         "balacc": float(r.get("balacc_mean", np.nan))})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--perfold", required=True)
    ap.add_argument("--root", default="outputs_perfold_pub")
    ap.add_argument("--eid", default="E1")
    ap.add_argument("--budget", default="published")
    ap.add_argument("--out", required=True, help="png; the table goes next to it as csv")
    args = ap.parse_args(argv)

    df = collect(args.perfold, args.root, args.eid, args.budget)
    df.to_csv(os.path.splitext(args.out)[0] + ".csv", index=False)
    missing = df[df["pehe"].isna()]
    if len(missing):
        print("missing cells (not scored yet):")
        print(missing[["bias", "estimator"]].to_string(index=False))
    print(df.round(3).to_string(index=False))
    try:                     # the table is written even where matplotlib is missing
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: table written, no figure")
        return df

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    colors = {"LR / ET": "#4C72B0", "CausalPFN": "#C44E52"}
    for est, g in df.dropna(subset=["pehe"]).groupby("estimator"):
        g = g.sort_values("bias")
        axes[0].plot(g["bias"], g["pehe"], "o-", color=colors[est], label=est)
        if g["ci_low"].notna().all():
            axes[0].fill_between(g["bias"], g["ci_low"], g["ci_high"], color=colors[est], alpha=0.15, lw=0)
        axes[1].plot(g["bias"], g["balacc"], "o-", color=colors[est], label=est)
    axes[0].set_ylabel("PEHE (paper scale)"); axes[1].set_ylabel("Balanced accuracy")
    for ax in axes:
        ax.set_xlabel("Location bias (BIAS)"); ax.set_xticks([b for b, _ in CELLS])
        ax.grid(alpha=0.3, lw=0.5); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{args.eid}, per-fold, {args.budget} budget: observed location bias sweep", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print("wrote", args.out)
    return df


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
