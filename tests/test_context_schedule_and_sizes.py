"""Log-uniform context schedule and the lesion-size report (28 Oct)."""
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_loguniform_schedule_covers_the_range_throughout_and_is_reproducible():
    from neurocausalpfn.train.train_pfn import _context_length

    cfg = {"seed": 0, "pfn": {"curriculum": "loguniform", "context_min": 128, "context_max": 3000, "iters": 20000}}
    n = np.array([_context_length(cfg, it) for it in range(20000)])
    assert n.min() >= 128 and n.max() <= 3000
    late = n[15000:]                                   # short contexts are still drawn at the end of the run
    assert (late < 300).mean() > 0.2 and (late > 1500).mean() > 0.15
    med = np.median(n)
    assert abs(np.log(med) - 0.5 * (np.log(128) + np.log(3000))) < 0.05
    assert [_context_length(cfg, it) for it in range(5)] == list(n[:5])
    linear = dict(cfg, pfn=dict(cfg["pfn"], curriculum="context", context_min=256))
    assert _context_length(linear, 0) == 256 and _context_length(linear, 15000) == 3000
    with pytest.raises(ValueError):
        _context_length(dict(cfg, pfn=dict(cfg["pfn"], curriculum="spiral")), 0)


def test_launch_cfg_reads_the_curriculum():
    from neurocausalpfn.train.launch_cfg import cfg_from_env

    p = cfg_from_env({"FAMILY": "mixture", "CURRICULUM": "loguniform", "CONTEXT_MIN": "128",
                      "CONTEXT_MAX": "3000", "CACHE": "x.npz"})["pfn"]
    assert p["curriculum"] == "loguniform" and (p["context_min"], p["context_max"]) == (128, 3000)


def test_lesion_size_report_prints_quantiles_only(tmp_path):
    rng = np.random.default_rng(0)
    for name, scale in (("small", 1.0), ("large", 4.0)):
        np.savez(tmp_path / f"{name}.npz", volumes=rng.lognormal(7, 1, 500) * scale,
                 files=np.array([f"secret{i}" for i in range(500)]))
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "audit_20261028", "lesion_sizes.py"),
                          str(tmp_path / "small.npz"), str(tmp_path / "large.npz"), "--out", str(tmp_path / "s.csv")],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "secret" not in out.stdout
    lines = out.stdout.strip().splitlines()
    q50 = [float(l.split("q50 ")[1].split(",")[0]) for l in lines]
    assert len(lines) == 2 and q50[1] > 3 * q50[0]
