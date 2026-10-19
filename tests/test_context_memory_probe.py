"""Phase 2 audit question 4: the context probe runs a few steps per cell and reports them."""
import csv

import pytest

from scripts import context_memory_probe as cmp_


def test_probe_reports_one_row_per_context_on_cpu(tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "probe.csv"
    rows = cmp_.main(["--contexts", "32", "48", "--batch-sizes", "2", "--iters", "3",
                      "--device", "cpu", "--out", str(out)])
    assert [r["context"] for r in rows] == [32, 48] and all(r["status"] == "ok" for r in rows)
    assert all(r["s_per_iter"] > 0 for r in rows)
    with open(out) as f:
        assert len(list(csv.DictReader(f))) == 2
