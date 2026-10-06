"""InterSynth Neuro-Prior as written in the design document (Section 4.1).

Two anatomical scores drive the outcomes of every process: a functional
disruption score D in [0, 1] (weighted mean over the atlas networks of the
fraction of each network's receptomic subnetwork the lesion covers) and a
treatment-susceptibility score S in [0, 1] (the same over the transcriptomic
subnetworks). The potential outcomes are

    mu0 = (1 - D) + alpha * D,      Y0 = clip(mu0 + eps0, 0, 1)
    mu1 = mu0 + beta * S * D,       Y1 = clip(mu1 + eps1, 0, 1)

so the CATE beta * S * D is non-negative and bounded by beta. Treatment is
W ~ Bernoulli(sigmoid(slope * gamma * g(anatomy) + noise)) with one of four
mechanisms g: severity-driven (D), location-driven (cortical vs subcortical,
read as the depth of the lesion centroid), network-driven (coverage of one or
two priority networks) and mixed. gamma = 0 is a randomised trial, gamma = 1
near-deterministic allocation; the noise term keeps every propensity away
from 0 and 1 (positivity). Treatment depends on observed anatomy only, so
strong ignorability holds by construction (R1); mu0 and mu1 are bounded and
measurable (R2); the parameter space is finite-dimensional with the ranges
of the design, gamma in [0, 1], beta in [0.05, 0.55], alpha in [0.05, 0.45]
(R3).

The per-network weights of D and S are drawn per process (Dirichlet), so the
prior covers many ways of reading the same anatomy; both scores are rescaled
per process so that the 95th percentile of the covariate pool maps to 1 (raw
overlap fractions are small for most lesions). The covariate shown to the
transformer is the encoder latent of the pool, never D or S.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .atlas import FunctionalAtlas

MECHANISMS = ("severity", "location", "network", "mixed")


@dataclass
class TheoryHyperPrior:
    """Ranges of the process parameters (uniform unless stated); the defaults
    are the ranges of the design document (R3)."""
    alpha: Tuple[float, float] = (0.05, 0.45)       # residual recovery of a fully disrupted brain
    beta: Tuple[float, float] = (0.05, 0.55)        # treatment effect scale
    gamma: Tuple[float, float] = (0.0, 1.0)         # confounding strength
    mechanisms: Sequence[str] = MECHANISMS
    mechanism_weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25)
    noise_y: Tuple[float, float] = (0.02, 0.10)     # sd of eps0, eps1
    noise_w: Tuple[float, float] = (0.5, 1.5)       # sd of the logit noise (positivity)
    slope: float = 6.0                              # logit scale at gamma = 1 on a standardised g
    n_priority: Tuple[int, int] = (1, 2)            # priority networks of the network mechanism
    concentration: float = 1.0                      # Dirichlet concentration of the D and S weights
    quantile: float = 0.95                          # pool quantile of the raw scores mapped to 1

    def contains(self, alpha: float, beta: float, gamma: float) -> bool:
        return bool(self.alpha[0] <= alpha <= self.alpha[1] and self.beta[0] <= beta <= self.beta[1]
                    and self.gamma[0] <= gamma <= self.gamma[1])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _standardise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    sd = float(v.std())
    return (v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v)


class InterSynthTheoryDGP:
    """One process of the design's Neuro-Prior on an anatomical substrate."""

    def __init__(self, atlas: FunctionalAtlas, rng: np.random.Generator,
                 hyper: Optional[TheoryHyperPrior] = None, *,
                 alpha: Optional[float] = None, beta: Optional[float] = None, gamma: Optional[float] = None,
                 mechanism: Optional[str] = None, noise_y: Optional[float] = None,
                 noise_w: Optional[float] = None, w_d: Optional[np.ndarray] = None,
                 w_s: Optional[np.ndarray] = None, priority: Optional[Sequence[int]] = None,
                 sign: Optional[int] = None):
        h = hyper or TheoryHyperPrior()
        self.atlas, self.hyper = atlas, h
        K = atlas.n_networks
        self.alpha = float(rng.uniform(*h.alpha)) if alpha is None else float(alpha)
        self.beta = float(rng.uniform(*h.beta)) if beta is None else float(beta)
        self.gamma = float(rng.uniform(*h.gamma)) if gamma is None else float(gamma)
        if mechanism is None:
            w = np.asarray(h.mechanism_weights, dtype=float)[: len(h.mechanisms)]
            mechanism = str(rng.choice(list(h.mechanisms), p=w / w.sum()))
        if mechanism not in MECHANISMS:
            raise ValueError(f"unknown mechanism {mechanism!r}")
        self.mechanism = mechanism
        self.noise_y = float(rng.uniform(*h.noise_y)) if noise_y is None else float(noise_y)
        self.noise_w = float(rng.uniform(*h.noise_w)) if noise_w is None else float(noise_w)
        conc = np.full(K, float(h.concentration))
        self.w_d = np.asarray(rng.dirichlet(conc) if w_d is None else w_d, dtype=np.float64)
        self.w_s = np.asarray(rng.dirichlet(conc) if w_s is None else w_s, dtype=np.float64)
        if priority is None:
            k = int(rng.integers(h.n_priority[0], h.n_priority[1] + 1))
            priority = rng.choice(K, size=min(k, K), replace=False)
        self.priority = [int(i) for i in priority]           # positions in atlas.networks order
        self.sign = int(rng.choice([-1, 1])) if sign is None else int(sign)
        self.mixed_w = rng.dirichlet(np.ones(3)) if mechanism == "mixed" else None
        self.scale_d = self.scale_s = 1.0
        self.centre = np.asarray(atlas.shape, dtype=np.float64) / 2.0

    # ------------------------------------------------------------- anatomy
    def raw_scores(self, overlaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """overlaps [n, K, 2] -> unscaled (D, S): receptomic subnetwork for D,
        transcriptomic for S."""
        ov = np.asarray(overlaps, dtype=np.float64)
        return ov[:, :, 0] @ self.w_d, ov[:, :, 1] @ self.w_s

    def calibrate(self, overlaps_pool: np.ndarray) -> "InterSynthTheoryDGP":
        """Fix the per-process scales so that the pool's upper quantile of each
        raw score maps to 1. Called once per process by the cohort."""
        d, s = self.raw_scores(overlaps_pool)
        q = self.hyper.quantile
        self.scale_d = max(float(np.quantile(d, q)), 1e-6)
        self.scale_s = max(float(np.quantile(s, q)), 1e-6)
        return self

    def scores(self, overlaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d, s = self.raw_scores(overlaps)
        return np.clip(d / self.scale_d, 0.0, 1.0), np.clip(s / self.scale_s, 0.0, 1.0)

    # -------------------------------------------------------- ground truth
    def mu0(self, D: np.ndarray) -> np.ndarray:
        return (1.0 - D) + self.alpha * D

    def mu1(self, D: np.ndarray, S: np.ndarray) -> np.ndarray:
        return self.mu0(D) + self.beta * S * D

    def cate(self, D: np.ndarray, S: np.ndarray) -> np.ndarray:
        return self.beta * S * D

    # ---------------------------------------------------------- allocation
    def g(self, D: np.ndarray, overlaps: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Standardised allocation feature of the mechanism (sign random per
        process so that both directions of confounding occur)."""
        ov = np.asarray(overlaps, dtype=np.float64)
        depth = -np.linalg.norm(np.asarray(centroids, dtype=np.float64) - self.centre, axis=1)
        feats = {"severity": _standardise(D),
                 "location": _standardise(depth),
                 "network": _standardise(ov[:, self.priority, :].sum(axis=(1, 2)))}
        if self.mechanism == "mixed":
            g = sum(w * feats[k] for w, k in zip(self.mixed_w, ("severity", "location", "network")))
            g = _standardise(g)
        else:
            g = feats[self.mechanism]
        return self.sign * g

    def propensity(self, D: np.ndarray, overlaps: np.ndarray, centroids: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
        """P(W = 1 | anatomy, noise): the noise is part of the assignment, not
        of the covariates, and keeps the propensity inside (0, 1)."""
        logit = self.hyper.slope * self.gamma * self.g(D, overlaps, centroids)
        logit = logit + self.noise_w * rng.normal(size=len(D))
        return np.clip(_sigmoid(logit), 0.01, 0.99)

    def describe(self) -> Dict:
        return {"family": "theory", "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
                "mechanism": self.mechanism, "noise_y": self.noise_y, "noise_w": self.noise_w,
                "priority": list(self.priority), "sign": self.sign,
                "scale_d": self.scale_d, "scale_s": self.scale_s}


def make_theory_dataset(dgp: InterSynthTheoryDGP,
                        overlaps_ctx: np.ndarray, cent_ctx: np.ndarray, X_ctx: np.ndarray,
                        overlaps_qry: np.ndarray, cent_qry: np.ndarray, X_qry: np.ndarray,
                        rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Observational context (treatment and continuous outcome in [0, 1] drawn
    from the process) and query set with the known potential outcomes; same
    keys as ``neuro_prior.make_prior_dataset``."""
    Dc, Sc = dgp.scores(overlaps_ctx)
    e = dgp.propensity(Dc, overlaps_ctx, cent_ctx, rng)
    Tc = (rng.uniform(size=len(Dc)) < e).astype(np.float64)
    mu_c = np.where(Tc == 1, dgp.mu1(Dc, Sc), dgp.mu0(Dc))
    Yc = np.clip(mu_c + dgp.noise_y * rng.normal(size=len(Dc)), 0.0, 1.0)

    Dq, Sq = dgp.scores(overlaps_qry)
    Tq = rng.integers(0, 2, size=len(Dq)).astype(np.float64)
    mu0, mu1 = dgp.mu0(Dq), dgp.mu1(Dq, Sq)
    return {"Xc": np.asarray(X_ctx, np.float64), "Tc": Tc, "Yc": Yc,
            "Xq": np.asarray(X_qry, np.float64), "Tq": Tq,
            "mu_q": np.where(Tq == 1, mu1, mu0), "mu0": mu0, "mu1": mu1,
            "e_ctx": e, "D_ctx": Dc, "S_ctx": Sc}
