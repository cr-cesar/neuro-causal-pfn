"""Bench summary (giles TE=1 RE=0): slope, calibration, PEHE, corr per checkpoint and context. Aggregates only.

    python bench_table.py label=<bench.json> [...] --out t.csv
"""
import argparse, json
import pandas as pd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); rows = []
    for spec in a.runs:
        label, path = spec.split("=", 1)
        c2 = json.load(open(path))["check2"]
        for key, v in c2.items():
            if not key.startswith("giles_TE1_RE0_ctx"):
                continue
            ctx = int(key.split("ctx")[-1])
            kept = v.get("kept", {})            # distinct-context runs (T7): processes kept / drawn
            for std, r in v.items():
                if "slope_pred_on_true" not in r:
                    continue
                rows.append({"model": label if std != "LR_two" else "LR_two", "ctx": ctx, "std": std,
                             "slope": r["slope_pred_on_true"], "calib": r["calib_slope_true_on_pred"],
                             "pehe": r["pehe"], "corr": r["corr"],
                             "mean_at_0": r.get("mean_pred_at_true+0"), "mean_at_+1": r.get("mean_pred_at_true+1"),
                             "mean_at_-1": r.get("mean_pred_at_true-1"),
                             "processes_kept": kept.get("processes_kept"), "processes_drawn": kept.get("processes_drawn")})
    t = pd.DataFrame(rows).drop_duplicates(["model", "ctx", "std"])
    t.to_csv(a.out, index=False); print(t.to_string(index=False))


if __name__ == "__main__":
    main()
