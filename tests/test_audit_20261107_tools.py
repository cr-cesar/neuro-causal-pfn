"""Synthetic checks of the Brief 9 analysis scripts (no data needed)."""
import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "scripts", "audit_20261107")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lno_tabs(effect, run_sd, seeds, rng):
    defs = [f"d{i:02d}" for i in range(16)]
    groups = {d: i // 4 + 1 for i, d in enumerate(defs)}
    mu = {d: rng.uniform(0.2, 0.4) for d in defs}
    tab = {}
    for s in seeds:
        for g in (1, 2, 3, 4):
            u = rng.normal(0, run_sd)
            tab[(g, s)] = pd.Series({d: mu[d] + u + (effect if groups[d] == g else 0.0) + rng.normal(0, 0.002)
                                     for d in defs})
    return tab, groups


def test_run_effects_cancel_in_held_minus_seen():
    lc = _load("lno_cluster")
    rng = np.random.default_rng(0)
    tab, groups = _lno_tabs(0.0, 0.05, [0], rng)
    out, per = lc.analyse(tab, groups, [0], "both")
    # large run effects, no held effect: the mean is free of them (only the
    # deficit-level noise remains), while the naive interval is inflated
    assert abs(out["diff_mean"]) < 0.003
    assert out["naive_ci_high"] - out["naive_ci_low"] > 0.02


def test_reml_recovers_the_held_effect():
    lc = _load("lno_cluster")
    rng = np.random.default_rng(1)
    tab, groups = _lno_tabs(0.004, 0.01, [0, 1, 2], rng)
    out, _ = lc.analyse(tab, groups, [0, 1, 2], "both")
    assert out["reml_ci_low"] < 0.004 < out["reml_ci_high"]
    assert out["reml_ci_low"] > 0


def test_holm():
    sc = _load("stage1_screen")
    adj = sc.holm([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])


def test_factorial_signs_are_a_resolution_iv_design():
    sc = _load("stage1_screen")
    S = np.array(list(sc.SIGNS.values()))
    assert (S[:, 3] == S[:, 0] * S[:, 1] * S[:, 2]).all()          # D = ABC
    assert (S.T @ S == 8 * np.eye(4)).all()                          # orthogonal main effects
