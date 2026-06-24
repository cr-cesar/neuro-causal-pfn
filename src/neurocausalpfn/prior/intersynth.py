"""InterSynth-style synthetic generator adapted to the VAE latents.

Each data-generating process samples an outcome model under control and under
treatment, and a propensity model that depends only on the observed covariates,
which guarantees ignorability by construction. The treatment effect is
heterogeneous. Confounding is induced through the observed covariates by
correlating the propensity with the prognosis, according to the chosen
mechanism (severity, location, network or a mixture).

The expected conditional potential outcomes mu_0(x) and mu_1(x) are sigmoids,
so they live in the unit interval and serve both as a continuous target in
training and as the probability of a good outcome in deployment.
"""
from typing import Dict

import numpy as np

MECHANISMS = ("severity", "location", "network", "mixed")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class SyntheticDGP:
    def __init__(self, d_x: int, rng: np.random.Generator,
                 mechanism: str = "mixed", confound_strength: float = 1.0,
                 effect_scale: float = 1.0):
        self.d_x = int(d_x)
        self.mechanism = mechanism
        scale = 1.0 / np.sqrt(self.d_x)
        # outcome model under control and heterogeneous treatment effect
        self.w0 = rng.normal(0.0, 1.0, self.d_x) * scale
        self.b0 = float(rng.normal(0.0, 0.5))
        self.delta = rng.normal(0.0, 1.0, self.d_x) * scale
        self.b1 = self.b0 + float(rng.normal(0.0, 0.5))
        if effect_scale != 1.0:
            # scale the treatment effect (the E12 curriculum varies its size);
            # applied after sampling so effect_scale == 1.0 reproduces the draw
            self.delta = self.delta * effect_scale
            self.b1 = self.b0 + (self.b1 - self.b0) * effect_scale
        # the propensity depends only on X (ignorable). The observed confounding
        # is induced by aligning part of w_prop with the prognosis.
        if mechanism in ("severity", "mixed"):
            direction = self.w0.copy()
        else:
            direction = rng.normal(0.0, 1.0, self.d_x) * scale
        self.w_prop = confound_strength * direction + rng.normal(0.0, 1.0, self.d_x) * scale
        self.b_prop = float(rng.normal(0.0, 0.3))
        self.noise = 0.05

    def mu(self, X: np.ndarray, t: int) -> np.ndarray:
        w = self.w0 + (self.delta if t == 1 else 0.0)
        b = self.b1 if t == 1 else self.b0
        return _sigmoid(X @ w + b)

    def propensity(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.w_prop + self.b_prop)

    def sample_observed(self, X: np.ndarray, rng: np.random.Generator):
        e = self.propensity(X)
        t = (rng.uniform(size=len(X)) < e).astype(np.float64)
        mu_t = np.where(t == 1, self.mu(X, 1), self.mu(X, 0))
        y = np.clip(mu_t + rng.normal(0.0, self.noise, size=len(X)), 0.0, 1.0)
        return t, y


def sample_covariates(n: int, d_x: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=(n, d_x))


def make_dataset(dgp: SyntheticDGP, n_context: int, n_query: int,
                 rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Builds an observational context and a set of queries with the true
    expected conditional potential outcome for the queried treatment, plus
    mu_0 and mu_1 to evaluate the treatment effect."""
    Xc = sample_covariates(n_context, dgp.d_x, rng)
    Tc, Yc = dgp.sample_observed(Xc, rng)
    Xq = sample_covariates(n_query, dgp.d_x, rng)
    Tq = rng.integers(0, 2, size=n_query).astype(np.float64)
    mu0 = dgp.mu(Xq, 0)
    mu1 = dgp.mu(Xq, 1)
    mu_q = np.where(Tq == 1, mu1, mu0)
    return {"Xc": Xc, "Tc": Tc, "Yc": Yc, "Xq": Xq, "Tq": Tq,
            "mu_q": mu_q, "mu0": mu0, "mu1": mu1}
