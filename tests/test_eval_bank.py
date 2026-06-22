import numpy as np

from neurocausalpfn.eval.equity import stratified_pehe
from neurocausalpfn.eval.latent_quality import (active_dimensions, ioss, mcc,
                                                per_dim_kl)
from neurocausalpfn.eval.probes import linear_probe, mlp_probe, stratified_probe


def test_active_dimensions_counts_used_axes():
    rng = np.random.default_rng(0)
    n, d = 200, 8
    mu = np.zeros((n, d))
    mu[:, :3] = rng.normal(0, 2.0, size=(n, 3))   # three informative axes
    logvar = np.zeros((n, d))                      # unit posterior variance
    pdk = per_dim_kl(mu, logvar, None)
    assert active_dimensions(pdk, threshold=0.01) == 3


def test_ioss_larger_when_dependent():
    rng = np.random.default_rng(1)
    n = 300
    indep = rng.normal(size=(n, 4))
    base = rng.normal(size=(n, 1))
    dependent = np.repeat(base, 4, axis=1) + 0.01 * rng.normal(size=(n, 4))
    assert ioss(dependent) > ioss(indep)


def test_mcc_recovers_matched_factors():
    rng = np.random.default_rng(2)
    n = 300
    V = rng.normal(size=(n, 4))
    perm = [2, 0, 3, 1]
    Z = V[:, perm] * np.array([1.5, -2.0, 0.7, 3.0]) + 0.01 * rng.normal(size=(n, 4))
    Z_random = rng.normal(size=(n, 4))
    assert mcc(Z, V) > 0.9
    assert mcc(Z, V) > mcc(Z_random, V)


def test_linear_probe_recovers_signal():
    rng = np.random.default_rng(3)
    n, d = 120, 10
    Z = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = Z @ w + 0.1 * rng.normal(size=n)
    m = linear_probe(Z, y)
    assert m["r2"] > 0.8
    assert m["mae"] >= 0.0
    noise = linear_probe(Z, rng.normal(size=n))
    assert m["r2"] > noise["r2"]


def test_mlp_probe_runs():
    rng = np.random.default_rng(4)
    n, d = 80, 6
    Z = rng.normal(size=(n, d))
    y = (Z[:, 0] ** 2 + Z[:, 1]) + 0.1 * rng.normal(size=n)
    m = mlp_probe(Z, y, hidden=32, max_iter=300)
    assert set(m.keys()) == {"r2", "spearman", "mae"}


def test_stratified_probe_reports_groups():
    rng = np.random.default_rng(5)
    n, d = 120, 8
    Z = rng.normal(size=(n, d))
    y = Z @ rng.normal(size=d) + 0.1 * rng.normal(size=n)
    groups = rng.integers(0, 2, size=n)
    out = stratified_probe(Z, y, groups, kind="linear")
    assert "all" in out and "0" in out and "1" in out
    assert "r2" in out["all"]


def test_stratified_pehe_equity():
    rng = np.random.default_rng(6)
    n = 200
    truth = rng.normal(size=n)
    groups = np.array([0] * (n // 2) + [1] * (n // 2))
    pred = truth.copy()
    pred[groups == 1] += rng.normal(0, 1.0, size=(groups == 1).sum())  # group 1 much worse
    out = stratified_pehe(pred, truth, groups)
    assert "all" in out and "0" in out and "1" in out
    assert out["1"] > out["0"]
    assert out["max_min_ratio"] > 1.0
    assert isinstance(out["passes"], bool)
