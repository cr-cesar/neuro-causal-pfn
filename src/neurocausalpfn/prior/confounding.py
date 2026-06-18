"""Confounding mechanisms and a generator with unobserved confounding.

The main generator (intersynth) produces processes that are ignorable by
construction. This module provides the deliberate counterexample: a process in
which the treatment depends on an unobserved confounder U that also affects the
outcome. It serves to check that the identifiability verifier rejects it, which
is the test that operationalizes the convergence requirement.
"""
import numpy as np

from .intersynth import _sigmoid

MECHANISMS = ("severity", "location", "network", "mixed")


def make_unobserved_confounded(d_x: int, n: int, rng: np.random.Generator,
                               strength: float = 2.0):
    """Returns (W, X, Y0, Y1, U) with W dependent on U in addition to X."""
    scale = 1.0 / np.sqrt(d_x)
    X = rng.normal(0.0, 1.0, size=(n, d_x))
    U = rng.normal(0.0, 1.0, size=(n, 1))
    w = rng.normal(0.0, 1.0, d_x) * scale
    # the propensity depends on U, which violates ignorability
    e = _sigmoid(X @ w + strength * U[:, 0])
    W = (rng.uniform(size=n) < e).astype(np.float64)
    base = X @ w
    Y0 = np.clip(_sigmoid(base + 0.5 * U[:, 0]), 0.0, 1.0)
    Y1 = np.clip(_sigmoid(base + 0.4 + 0.5 * U[:, 0]), 0.0, 1.0)
    return W, X, Y0, Y1, U
