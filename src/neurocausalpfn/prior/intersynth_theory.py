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

Three extensions of the design, each off by default (TheoryHyperPrior): a
second active treatment with its own susceptibility, so that the CATE
beta * D * (S_A - S_B) carries a sign (the virtual trial of Giles is the limit
with binary susceptibilities and effect 1); a per-process effect scale that
maps the pool's upper quantile of |CATE| to a drawn target; and binary
outcomes Y ~ Bernoulli(mu) for a fraction of the processes, as in the
evaluation. mu0 and mu1 (the targets) are always the clipped expected
outcomes under T = 0 and T = 1.
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
    # Extensions of the design (each off by default, so the defaults reproduce
    # the design's prior exactly):
    # two_arms: probability that a process has two active treatments, A (T = 1)
    #   and B (T = 0), each responsive on its own susceptibility, so that the
    #   CATE beta * D * (S_A - S_B) carries a sign; the virtual trial of Giles is
    #   the limit with binary S and effect 1
    two_arms: float = 0.0
    # effect_scale: range of the pool's upper quantile of |CATE| after a
    #   per-process rescaling (None keeps the raw beta * D * S scale, whose
    #   upper quantile is typically 0.1-0.3)
    effect_scale: Optional[Tuple[float, float]] = None
    # binary_outcome: probability that a process draws Y ~ Bernoulli(mu)
    #   instead of mu + Gaussian noise (the evaluation's outcomes are binary)
    binary_outcome: float = 0.0

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
        self.scale_d = self.scale_s = self.scale_b = 1.0
        # extensions (see TheoryHyperPrior): a second arm with its own susceptibility
        # weights, a per-process effect scale fixed by calibrate(), binary outcomes
        self.two_arms = bool(rng.uniform() < float(h.two_arms))
        self.w_s2 = np.asarray(rng.dirichlet(conc), dtype=np.float64) if self.two_arms else None
        self.effect_target = (float(np.exp(rng.uniform(np.log(h.effect_scale[0]), np.log(h.effect_scale[1]))))
                              if h.effect_scale else None)
        self.effect_factor = 1.0
        self.binary = bool(rng.uniform() < float(h.binary_outcome))
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
        if self.two_arms:
            ov = np.asarray(overlaps_pool, dtype=np.float64)
            self.scale_b = max(float(np.quantile(ov[:, :, 0] @ self.w_s2, q)), 1e-6)
        if self.effect_target is not None:
            # the upper quantile of |CATE| over the pool maps to the drawn target
            self.effect_factor = 1.0
            D, S = self.scores(overlaps_pool)
            raw = np.abs(self.beta * D * (S - self.scores_b(overlaps_pool)))
            self.effect_factor = self.effect_target / max(float(np.quantile(raw, q)), 1e-6)
        return self

    def scores(self, overlaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d, s = self.raw_scores(overlaps)
        return np.clip(d / self.scale_d, 0.0, 1.0), np.clip(s / self.scale_s, 0.0, 1.0)

    def scores_b(self, overlaps: np.ndarray) -> np.ndarray:
        """Susceptibility to treatment B of a two-arm process: the second
        weighting of the first subnetwork column (B helps where other networks
        of the disrupted tissue are hit); zero for a single-arm process."""
        if not self.two_arms:
            return np.zeros(len(overlaps))
        ov = np.asarray(overlaps, dtype=np.float64)
        return np.clip((ov[:, :, 0] @ self.w_s2) / self.scale_b, 0.0, 1.0)

    # -------------------------------------------------------- ground truth
    def mu0(self, D: np.ndarray) -> np.ndarray:
        return (1.0 - D) + self.alpha * D

    def mu_arm(self, D: np.ndarray, S_arm: np.ndarray) -> np.ndarray:
        """Expected outcome under an arm whose susceptibility is S_arm."""
        return np.clip(self.mu0(D) + self.effect_factor * self.beta * S_arm * D, 0.0, 1.0)

    def mu1(self, D: np.ndarray, S: np.ndarray) -> np.ndarray:
        return self.mu_arm(D, S)

    def mu_b(self, D: np.ndarray, S_b: np.ndarray) -> np.ndarray:
        """Expected outcome under T = 0: the control (mu0) of a single-arm
        process, the B arm of a two-arm process."""
        return self.mu_arm(D, S_b) if self.two_arms else self.mu0(D)

    def cate(self, D: np.ndarray, S: np.ndarray, S_b: Optional[np.ndarray] = None) -> np.ndarray:
        S_b = np.zeros_like(D) if S_b is None else S_b
        return self.mu1(D, S) - self.mu_b(D, S_b)

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
                "scale_d": self.scale_d, "scale_s": self.scale_s,
                "two_arms": self.two_arms, "effect_factor": self.effect_factor, "binary": self.binary}


def make_theory_dataset(dgp: InterSynthTheoryDGP,
                        overlaps_ctx: np.ndarray, cent_ctx: np.ndarray, X_ctx: np.ndarray,
                        overlaps_qry: np.ndarray, cent_qry: np.ndarray, X_qry: np.ndarray,
                        rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Observational context (treatment and outcome drawn from the process:
    continuous in [0, 1], or binary when the process says so) and query set with the known potential outcomes; same
    keys as ``neuro_prior.make_prior_dataset``."""
    Dc, Sc = dgp.scores(overlaps_ctx)
    e = dgp.propensity(Dc, overlaps_ctx, cent_ctx, rng)
    Tc = (rng.uniform(size=len(Dc)) < e).astype(np.float64)
    mu_c = np.where(Tc == 1, dgp.mu1(Dc, Sc), dgp.mu_b(Dc, dgp.scores_b(overlaps_ctx)))
    if dgp.binary:
        Yc = (rng.uniform(size=len(Dc)) < mu_c).astype(np.float64)
    else:
        Yc = np.clip(mu_c + dgp.noise_y * rng.normal(size=len(Dc)), 0.0, 1.0)

    Dq, Sq = dgp.scores(overlaps_qry)
    Tq = rng.integers(0, 2, size=len(Dq)).astype(np.float64)
    mu0, mu1 = dgp.mu_b(Dq, dgp.scores_b(overlaps_qry)), dgp.mu1(Dq, Sq)
    return {"Xc": np.asarray(X_ctx, np.float64), "Tc": Tc, "Yc": Yc,
            "Xq": np.asarray(X_qry, np.float64), "Tq": Tq,
            "mu_q": np.where(Tq == 1, mu1, mu0), "mu0": mu0, "mu1": mu1,
            "e_ctx": e, "D_ctx": Dc, "S_ctx": Sc}
