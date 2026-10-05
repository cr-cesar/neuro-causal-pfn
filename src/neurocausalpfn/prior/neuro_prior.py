"""Neuro-Prior v1: a distribution over virtual-trial generators.

The Phase 1 evaluation uses one generator, the virtual trial of the reference
paper (``giles_replica.GilesTrial``): sixteen deficits defined by the overlap of
the lesion with the two subnetworks of a functional network, a treatment that
works only when it matches the subnetwork the lesion hits, a fixed response
probability (TE), a fixed spontaneous-recovery probability (RE) and an
allocation bias (BIAS) along the spatial axis that separates the two
subnetworks. A prior-fitted network needs many such generators, so this module
samples them: every draw is one data-generating process on the same anatomical
substrate, with its own set of causal networks, overlap threshold, response and
recovery probabilities, allocation mechanism and strength, and label noise.
The reference generator is one point of the distribution and is recovered by
``NeuroPriorDGP.from_giles``; a test checks that the headline scenarios of the
paper lie inside the hyper-prior ranges.

Allocation mechanisms. ``axis``: the paper's observed confounding, a ramp of
the treatment probability along the rank of the lesion centroid on the axis
with the largest separation between the two subnetwork centroids.
``severity``: the same ramp on lesion volume. ``agnostic``: the paper's
"unobserved" type, allocation towards the true susceptibility, which the
identifiability verifier must reject when the strength is not zero.

The covariate shown to the transformer is never produced here: it is the
encoder latent (or the overlap fractions) of the lesion pool managed by
``cohort.NeuroPriorCohort``. This module only turns anatomy into ground truth.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .atlas import FunctionalAtlas

BIAS_TYPES = ("axis", "severity", "agnostic")


@dataclass
class HyperPrior:
    """Ranges of the process hyper-parameters (uniform unless stated)."""
    n_networks: Tuple[int, int] = (1, 3)              # causal networks per process (inclusive)
    p_te: Tuple[float, float] = (0.3, 1.0)            # probability of response when treated correctly
    p_re: Tuple[float, float] = (0.0, 0.5)            # probability of spontaneous recovery
    bias: Tuple[float, float] = (0.0, 1.0)            # allocation strength (paper's BIAS)
    bias_types: Sequence[str] = BIAS_TYPES
    bias_weights: Sequence[float] = (0.6, 0.3, 0.1)   # sampling weights of bias_types
    threshold: Tuple[float, float] = (0.02, 0.10)     # overlap fraction that counts as a hit
    label_noise: Tuple[float, float] = (0.0, 0.05)    # probability of flipping an observed outcome
    both_hit_value: float = 0.5                       # paper's y_true when both subnetworks are hit

    def contains_giles(self, TE: float, RE: float, BIAS: float, BIASTYPE: str) -> bool:
        """True when a paper scenario is a possible draw of this hyper-prior."""
        lo, hi = self.p_te
        ok = lo <= TE <= hi
        ok &= self.p_re[0] <= RE <= self.p_re[1]
        ok &= self.bias[0] <= BIAS <= self.bias[1]
        ok &= self.threshold[0] <= 0.05 <= self.threshold[1]
        ok &= self.n_networks[0] <= 1 <= self.n_networks[1]
        ok &= ("axis" if BIASTYPE == "observed" else "agnostic") in tuple(self.bias_types)
        return bool(ok)


def _ramp(rank: np.ndarray, n: int, strength: float) -> np.ndarray:
    """Treatment probability along a rank 0..n-1: 0.5 at the middle, 0.5 +/-
    strength at the ends, clipped to [0, 1]. For strength <= 0.5 this is the
    paper's linspace(0.5 - BIAS, 0.5 + BIAS); above it the extremes saturate
    (the paper fixes a deterministic block there)."""
    if n <= 1:
        return np.full(n, 0.5)
    centred = 2.0 * rank / (n - 1) - 1.0
    return np.clip(0.5 + strength * centred, 0.0, 1.0)


class NeuroPriorDGP:
    """One process of the Neuro-Prior on an anatomical substrate."""

    def __init__(self, atlas: FunctionalAtlas, rng: np.random.Generator,
                 hyper: Optional[HyperPrior] = None, *,
                 networks: Optional[Sequence[int]] = None, optimal_for_A: Optional[Sequence[int]] = None,
                 p_te: Optional[float] = None, p_re: Optional[float] = None,
                 bias: Optional[float] = None, bias_type: Optional[str] = None,
                 threshold: Optional[float] = None, label_noise: Optional[float] = None,
                 severity_sign: Optional[int] = None):
        h = hyper or HyperPrior()
        self.atlas, self.hyper = atlas, h
        k = int(rng.integers(h.n_networks[0], h.n_networks[1] + 1))
        if networks is None:
            networks = list(rng.choice(atlas.networks, size=min(k, atlas.n_networks), replace=False))
        self.networks: List[int] = [int(n) for n in networks]
        if optimal_for_A is None:
            optimal_for_A = [int(rng.integers(0, 2)) for _ in self.networks]
        self.optimal_for_A: Dict[int, int] = {n: int(o) for n, o in zip(self.networks, optimal_for_A)}
        self.p_te = float(rng.uniform(*h.p_te)) if p_te is None else float(p_te)
        self.p_re = float(rng.uniform(*h.p_re)) if p_re is None else float(p_re)
        self.bias = float(rng.uniform(*h.bias)) if bias is None else float(bias)
        if bias_type is None:
            w = np.asarray(h.bias_weights, dtype=float)[: len(h.bias_types)]
            bias_type = str(rng.choice(list(h.bias_types), p=w / w.sum()))
        if bias_type not in BIAS_TYPES:
            raise ValueError(f"unknown bias type {bias_type!r}")
        self.bias_type = bias_type
        self.threshold = float(rng.uniform(*h.threshold)) if threshold is None else float(threshold)
        self.label_noise = float(rng.uniform(*h.label_noise)) if label_noise is None else float(label_noise)
        self.severity_sign = int(rng.choice([-1, 1])) if severity_sign is None else int(severity_sign)
        self.both_hit_value = h.both_hit_value

    # ------------------------------------------------------------------ paper
    @classmethod
    def from_giles(cls, atlas: FunctionalAtlas, network: int, TE: float, RE: float,
                   BIAS: float, BIASTYPE: str = "observed", rng: Optional[np.random.Generator] = None):
        """The reference virtual trial for one deficit as a process of the prior.
        The paper maps the larger subnetwork to treatment "1"; the overlap
        threshold is 5 % and the observed bias runs along the centroid axis."""
        rng = rng or np.random.default_rng(0)
        a, b = atlas.subnetworks[network]
        larger_is_A = float(a.sum()) >= float(b.sum())
        # optimal treatment for a lesion hitting A: "1" if A is the larger subnetwork
        return cls(atlas, rng, networks=[network], optimal_for_A=[1 if larger_is_A else 0],
                   p_te=TE, p_re=RE, bias=BIAS, bias_type="axis" if BIASTYPE == "observed" else "agnostic",
                   threshold=0.05, label_noise=0.0, severity_sign=1)

    # ------------------------------------------------------------ ground truth
    def susceptibility(self, overlaps: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """overlaps: [K, 2] fractions for every atlas network (atlas.networks
        order). Returns (network, s) for the first causal network the lesion
        hits, with s = 0 (subnetwork A), 1 (B) or both_hit_value (both); or
        (None, None) when it hits none."""
        for n in self.networks:
            i = self.atlas.networks.index(n)
            oa, ob = float(overlaps[i, 0]), float(overlaps[i, 1])
            ha, hb = oa > self.threshold, ob > self.threshold
            if ha and hb:
                return n, self.both_hit_value
            if ha:
                return n, 0.0
            if hb:
                return n, 1.0
        return None, None

    def optimal_treatment(self, network: Optional[int], s: Optional[float]) -> Optional[int]:
        if network is None or s is None:
            return None
        if s == self.both_hit_value:
            return -1                                   # either treatment works
        return self.optimal_for_A[network] if s == 0.0 else 1 - self.optimal_for_A[network]

    def mu(self, network: Optional[int], s: Optional[float], t: int) -> float:
        """Expected potential outcome: probability of a good outcome under t."""
        opt = self.optimal_treatment(network, s)
        works = opt is not None and (opt == -1 or opt == int(t))
        p_treat = self.p_te if works else 0.0
        return 1.0 - (1.0 - p_treat) * (1.0 - self.p_re)

    def cate(self, network, s) -> float:
        return self.mu(network, s, 1) - self.mu(network, s, 0)

    # -------------------------------------------------------------- allocation
    def _axis(self, network: int) -> Tuple[int, int]:
        """Axis with the largest separation of the two subnetwork centroids and
        the sign that sends treatment "1" towards its own subnetwork."""
        ca, cb = self.atlas.centroid(network, 0), self.atlas.centroid(network, 1)
        axis = int(np.argmax(np.abs(ca - cb)))
        c1 = ca if self.optimal_for_A[network] == 1 else cb       # centroid of the subnetwork treated by "1"
        c0 = cb if self.optimal_for_A[network] == 1 else ca
        return axis, (1 if c1[axis] >= c0[axis] else -1)

    def propensity(self, centroids: np.ndarray, volumes: np.ndarray,
                   networks: Sequence[Optional[int]], s: Sequence[Optional[float]]) -> np.ndarray:
        """P(T = 1) for a cohort. ``axis`` and ``severity`` depend only on
        observed geometry; ``agnostic`` depends on the true susceptibility."""
        n = len(centroids)
        if self.bias_type == "agnostic":
            p = np.full(n, 0.5)
            for i, si in enumerate(s):
                if si is None or si == self.both_hit_value:
                    continue
                opt = self.optimal_treatment(networks[i], si)
                p[i] = self.bias if opt == 1 else 1.0 - self.bias
            return p
        if self.bias_type == "axis":
            axis, sign = self._axis(self.networks[0])
            key = sign * np.asarray(centroids, dtype=float)[:, axis]
        else:
            key = self.severity_sign * np.asarray(volumes, dtype=float)
        rank = np.argsort(np.argsort(key, kind="stable"), kind="stable")
        return _ramp(rank, n, self.bias)

    def describe(self) -> Dict:
        return {"networks": list(self.networks), "optimal_for_A": dict(self.optimal_for_A),
                "p_te": self.p_te, "p_re": self.p_re, "bias": self.bias, "bias_type": self.bias_type,
                "threshold": self.threshold, "label_noise": self.label_noise}


def make_prior_dataset(dgp: NeuroPriorDGP,
                       overlaps_ctx: np.ndarray, cent_ctx: np.ndarray, vol_ctx: np.ndarray, X_ctx: np.ndarray,
                       overlaps_qry: np.ndarray, cent_qry: np.ndarray, vol_qry: np.ndarray, X_qry: np.ndarray,
                       rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Observational context (treatment and binary outcome drawn from the
    process) and query set with the known potential outcomes. overlaps_* are
    [n, K, 2]; cent_* [n, 3]; vol_* [n]; X_* [n, d_x]. Also returns the
    susceptibility indicator ``u_ctx`` (1 when the lesion hits a causal network)
    for the identifiability verifier."""
    sus_c = [dgp.susceptibility(o) for o in overlaps_ctx]
    nets_c, s_c = [a for a, _ in sus_c], [b for _, b in sus_c]
    e = dgp.propensity(cent_ctx, vol_ctx, nets_c, s_c)
    Tc = (rng.uniform(size=len(s_c)) < e).astype(np.float64)
    mu_c = np.array([dgp.mu(nets_c[i], s_c[i], int(Tc[i])) for i in range(len(s_c))])
    Yc = (rng.uniform(size=len(s_c)) < mu_c).astype(np.float64)
    if dgp.label_noise > 0:
        flip = rng.uniform(size=len(Yc)) < dgp.label_noise
        Yc = np.where(flip, 1.0 - Yc, Yc)

    sus_q = [dgp.susceptibility(o) for o in overlaps_qry]
    Tq = rng.integers(0, 2, size=len(sus_q)).astype(np.float64)
    mu0 = np.array([dgp.mu(n, s, 0) for n, s in sus_q])
    mu1 = np.array([dgp.mu(n, s, 1) for n, s in sus_q])
    u_ctx = np.array([0.0 if s is None else (1.0 if dgp.optimal_treatment(n, s) == 1 else -1.0)
                      for n, s in sus_c])
    return {"Xc": np.asarray(X_ctx, np.float64), "Tc": Tc, "Yc": Yc,
            "Xq": np.asarray(X_qry, np.float64), "Tq": Tq,
            "mu_q": np.where(Tq == 1, mu1, mu0), "mu0": mu0, "mu1": mu1,
            "e_ctx": e, "u_ctx": u_ctx}
