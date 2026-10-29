"""Holm step-down correction (scripts/holm_adjust.py), checked against the
values stored for the 2a family (4 tests) of 29 Oct."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from holm_adjust import holm  # noqa: E402


def test_holm_matches_the_stored_2a_family_and_handles_nan():
    p = [3.796818888111425e-05, 0.6245135439822838, 1.080952968468469e-06, 0.0373252156982516]
    stored = [0.00011390456664334275, 0.6245135439822838, 4.323811873873876e-06, 0.0746504313965032]
    np.testing.assert_allclose(holm(p), stored, rtol=1e-12)
    out = holm([0.01, np.nan, 0.04])
    assert np.isnan(out[1]) and np.allclose(out[[0, 2]], [0.02, 0.04])
    assert np.all(holm([0.5, 0.9]) <= 1.0)
