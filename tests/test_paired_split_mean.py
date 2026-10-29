"""Paired test on the mean of the two splits (scripts/paired_split_mean.py)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from paired_split_mean import main, split_mean  # noqa: E402


def _per(seed, nan_at=None):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.2, 0.4, 16)
    df = pd.DataFrame({"deficit": [f"d{i}" for i in range(16)], "A": a, "B": a + 0.02 + 0.005 * rng.normal(size=16)})
    if nan_at is not None:
        df.loc[nan_at, "A"] = np.nan
    df["diff"] = df["A"] - df["B"]
    return df


def test_split_mean_averages_each_deficit_and_drops_an_undefined_one(tmp_path):
    r, g = _per(0, nan_at=3), _per(1)
    a, b = split_mean(r, g.iloc[::-1], "A", "B")           # row order does not matter
    assert np.isnan(a["d3"]) and np.isclose(a["d0"], (r.A[0] + g.A[0]) / 2)
    r.to_csv(tmp_path / "r.csv", index=False); g.to_csv(tmp_path / "g.csv", index=False)
    s = main(["--receptor", str(tmp_path / "r.csv"), "--genetics", str(tmp_path / "g.csv"),
              "--label-a", "A", "--label-b", "B", "--level", "lvl", "--out", str(tmp_path / "o.csv")])
    assert s["split"] == "both" and s["level"] == "lvl"
    assert s["n_deficits"] == 15 and s["n_excluded_nan"] == 1 and s["diff_mean"] < 0
    assert list(pd.read_csv(tmp_path / "o.csv").columns[:2]) == ["split", "level"]


def test_split_mean_refuses_different_deficits():
    with pytest.raises(ValueError):
        split_mean(_per(0), _per(1).iloc[:15], "A", "B")
