import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "compare_disconnectomes",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "compare_disconnectomes.py"))
cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cd)


def test_compare_pair_identical_and_scaled_maps():
    a = np.zeros((4, 4, 4))
    a[1:3, 1:3, 1:3] = np.linspace(0.1, 1.0, 8).reshape(2, 2, 2)
    s = cd.compare_pair(a, a)
    assert s["pearson"] == 1.0 and s["dice_support"] == 1.0 and s["max_abs_diff"] == 0.0
    # same shape, percent scale: correlation and support agree, scale does not
    s = cd.compare_pair(a * 100, a)
    assert s["pearson"] == 1.0 and s["dice_support"] == 1.0
    assert abs(s["sum_ratio"] - 100) < 1e-9 and s["max_new"] == 100.0
    # disjoint supports
    b = np.zeros_like(a)
    b[0, 0, 0] = 1.0
    s = cd.compare_pair(a, b)
    assert s["dice_support"] == 0.0 and s["dice_core"] == 0.0
