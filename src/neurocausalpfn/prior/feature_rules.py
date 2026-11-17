"""Volume and centroid rules (16 Nov): two more process families.

The other families set the effect by the coverage of anatomical regions (the
theory and virtual-trial families) or by a score of the latents (the generic
share). Here one scalar of the lesion sets it: its volume (family
``volume``), or the coordinate of its centroid on the first principal axis of
the enrolled lesions' centroids (family ``centroid``). Per process the rule
has one of two shapes over the enrolled lesions:

- ``threshold``: the top third of the scalar has effect +1, the bottom third
  -1, the middle 0;
- ``graded``: the effect is linear in the rank, from -1 (lowest) to +1
  (highest).

Lesions with the same value have the same effect: ranks are average ranks, and
the thirds are cut on them (``rule_effect``). A coin per process decides which
arm the high end responds to (``sign``), as a coin decides it for each side of
a virtual-trial network.

Everything else follows the virtual-trial family (``neuro_prior``) and is read
from the same ``HyperPrior``. With e the effect of a lesion, arm 1 suits it
with probability 0.5 + e / 2 and arm 0 with 0.5 - e / 2, and the expected
outcome under arm t is mu_t = 1 - (1 - TE r_t)(1 - RE), the virtual trial's
formula with its response indicator replaced by that probability r_t. With
TE = 1 and RE = 0 this is 0.5 + e / 2 under arm 1 and 0.5 - e / 2 under arm 0;
in general mu_1 - mu_0 = TE (1 - RE) e. Context outcomes are Bernoulli draws
(or the probability itself for the ``continuous_outcome`` share), with the
same label noise; treatment follows three allocation mechanisms with the
virtual trial's names, weights and strength range: a ramp along the rank of
the centroid coordinate on the principal axis (``axis``) or of the volume
(``severity``), each with a random direction, or towards the arm that suits
the lesion (``agnostic``, which the identifiability verifier rejects unless
its strength is negligible).

Allocation differs from the virtual trial in the ``axis`` mechanism. There
the ramp runs along the image axis (x, y or z) that best separates the two
halves of the process's first network, towards the half that responds to
arm 1 (``NeuroPriorDGP._axis``): the direction is tied to the effect. Here
the ramp runs along the principal coordinate of the centroids and its
direction is a coin of its own, independent of the rule's sign, as the
direction of the ``severity`` ramp is in both families. So in a centroid
process the ramp is aligned with the effect or against it, half the time
each; in the virtual trial it is always aligned.

Options of their own, in the same hyper-prior:

- ``rule_graded``: the share of processes with the graded shape (default 0.5);
- ``rule_enrol_by``: who a process enrols. ``share`` (default): a uniformly
  random part of the pool, of the share drawn from ``rule_enrol`` (default 1
  to 1, the whole pool). ``network``: the lesions a virtual-trial process
  drawn from the same ranges would enrol, i.e. those its networks reach above
  its threshold (the slice of the other families);
- ``rule_axis``: the centroids whose first principal axis the ``centroid``
  family (and the ``axis`` allocation ramp) uses. ``enrolled`` (default): the
  enrolled lesions'. ``pool``: the whole pool's, whoever is enrolled.

The thirds and the ranks are always those of the enrolled lesions. With the
defaults every process of a family is fixed on the same lesions, so only its
shape and its sign distinguish its effects: four functions per family.

A lesion without a finite volume and centroid cannot be ranked: enrolling one
is an error (NaN would otherwise sort last and take the top effect).

The direction of a principal axis is arbitrary; ``principal_coordinate`` fixes
it by making the axis's largest component positive, which is deterministic for
given centroids. The rule's own sign is a coin per process, so the convention
does not change which processes are drawn, only how ``sign`` reads.
"""
from typing import Dict, Optional

import numpy as np

from .neuro_prior import BIAS_TYPES, HyperPrior, _ramp

RULE_FAMILIES = ("volume", "centroid")
RULE_SHAPES = ("threshold", "graded")
ENROL_BY = ("share", "network")
RULE_AXES = ("enrolled", "pool")


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks 1..n of ``values``; equal values share the mean of their positions."""
    v = np.asarray(values, dtype=np.float64).ravel()
    _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)                       # last position of each distinct value
    return (last - (counts - 1) / 2.0)[inverse]


def rule_effect(values: np.ndarray, shape: str) -> np.ndarray:
    """The effect in [-1, 1] of each lesion from its scalar, before the
    process's sign. ``threshold``: with r the average rank of a lesion among n,
    +1 when (r - 0.5) / n is above 2/3, -1 when it is below 1/3, else 0 (for n
    a multiple of 3 and no ties: n / 3 lesions each; a group of equal values
    goes where its mean rank falls, to 0 when it sits exactly on a cut).
    ``graded``: 2 (r - 1) / (n - 1) - 1, computed as (2 r - n - 1) / (n - 1)
    so that reversing the order of the values flips the sign exactly."""
    if shape not in RULE_SHAPES:
        raise ValueError(f"unknown rule shape {shape!r}; choose from {RULE_SHAPES}")
    bad = int((~np.isfinite(np.asarray(values, dtype=np.float64))).sum())
    if bad:
        raise ValueError(f"{bad} of {np.size(values)} values are not finite: they cannot be ranked")
    r = average_ranks(values)
    n = len(r)
    twice = np.rint(2.0 * r).astype(np.int64)      # 2 r is an integer: no rounding below
    if shape == "graded":
        return np.zeros(n) if n < 2 else (twice - n - 1) / float(n - 1)
    pos = 3 * (twice - 1)                          # 6 (r - 0.5), against 6 n / 3 and 6 n 2 / 3
    return np.where(pos > 4 * n, 1.0, np.where(pos < 2 * n, -1.0, 0.0))


def principal_coordinate(centroids: np.ndarray) -> np.ndarray:
    """Coordinate of each centroid on the first principal axis of the given
    centroids (centred). The axis's direction is fixed by making its largest
    component positive, so the coordinate does not depend on the solver."""
    c = np.asarray(centroids, dtype=np.float64)
    if not np.isfinite(c).all():
        raise ValueError(f"{int((~np.isfinite(c)).any(axis=-1).sum())} of {len(c)} centroids are not finite")
    c = c - c.mean(axis=0, keepdims=True)
    if len(c) < 2:
        return np.zeros(len(c))
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    axis = vt[0]
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    return c @ axis


def _enrol_range(h) -> tuple:
    v = np.atleast_1d(np.asarray(getattr(h, "rule_enrol", (1.0, 1.0)), dtype=float))
    lo, hi = float(v[0]), float(v[-1])
    if not 0.0 < lo <= hi <= 1.0:
        raise ValueError("rule_enrol must be a range inside (0, 1]")
    return lo, hi


class FeatureRuleDGP:
    """One process of the ``volume`` or ``centroid`` family. The parameters
    are drawn from the virtual-trial hyper-prior; ``fit`` then fixes the
    effect of every enrolled lesion of the pool."""

    def __init__(self, feature: str, rng: np.random.Generator, hyper: Optional[HyperPrior] = None, *,
                 shape: Optional[str] = None, sign: Optional[int] = None,
                 p_te: Optional[float] = None, p_re: Optional[float] = None,
                 bias: Optional[float] = None, bias_type: Optional[str] = None,
                 label_noise: Optional[float] = None, enrol: Optional[float] = None,
                 enrol_by: Optional[str] = None, axis_from: Optional[str] = None):
        if feature not in RULE_FAMILIES:
            raise ValueError(f"unknown rule family {feature!r}; choose from {RULE_FAMILIES}")
        h = hyper or HyperPrior()
        self.feature, self.hyper = feature, h
        graded = float(getattr(h, "rule_graded", 0.5))
        if not 0.0 <= graded <= 1.0:
            raise ValueError("rule_graded must be in [0, 1]")
        if shape is None:
            shape = "graded" if rng.uniform() < graded else "threshold"
        if shape not in RULE_SHAPES:
            raise ValueError(f"unknown rule shape {shape!r}; choose from {RULE_SHAPES}")
        self.shape = shape
        self.sign = int(rng.choice([-1, 1])) if sign is None else int(sign)
        # efficacy, recovery, allocation and label noise as NeuroPriorDGP draws them
        self.p_te = float(rng.uniform(*h.p_te)) if p_te is None else float(p_te)
        self.p_re = float(rng.uniform(*h.p_re)) if p_re is None else float(p_re)
        if p_re is None and float(getattr(h, "te_re_coupled", 0.0) or 0.0) > 0 \
                and rng.uniform() < float(h.te_re_coupled):
            self.p_re = 1.0 - self.p_te
        self.bias = float(rng.uniform(*h.bias)) if bias is None else float(bias)
        if bias_type is None:
            w = np.asarray(h.bias_weights, dtype=float)[: len(h.bias_types)]
            bias_type = str(rng.choice(list(h.bias_types), p=w / w.sum()))
        if bias_type not in BIAS_TYPES:
            raise ValueError(f"unknown bias type {bias_type!r}")
        self.bias_type = bias_type
        self.label_noise = float(rng.uniform(*h.label_noise)) if label_noise is None else float(label_noise)
        self.ramp_sign = int(rng.choice([-1, 1]))          # direction of the axis / severity ramp
        self.continuous = bool(float(getattr(h, "continuous_outcome", 0.0) or 0.0) > 0
                               and rng.uniform() < float(h.continuous_outcome))
        self.enrol = float(rng.uniform(*_enrol_range(h))) if enrol is None else float(enrol)
        self.enrol_by = str(getattr(h, "rule_enrol_by", "share") if enrol_by is None else enrol_by)
        if self.enrol_by not in ENROL_BY:
            raise ValueError(f"unknown rule_enrol_by {self.enrol_by!r}; choose from {ENROL_BY}")
        if self.enrol_by == "network" and _enrol_range(h) != (1.0, 1.0):
            raise ValueError("rule_enrol is the share of the pool rule_enrol_by=share enrols; "
                             "leave it out with rule_enrol_by=network")
        self.axis_from = str(getattr(h, "rule_axis", "enrolled") if axis_from is None else axis_from)
        if self.axis_from not in RULE_AXES:
            raise ValueError(f"unknown rule_axis {self.axis_from!r}; choose from {RULE_AXES}")
        self.effect = self.coordinate = None               # per pool lesion, set by fit

    def fit(self, volumes: np.ndarray, centroids: np.ndarray, enrolled: np.ndarray) -> "FeatureRuleDGP":
        """Fix the effect of the enrolled lesions (indices into the pool):
        ranks and thirds are theirs, and so is the principal axis unless
        ``rule_axis`` is ``pool``. Lesions outside the enrolment get NaN, so
        using one is an error, not a silent zero; an enrolled lesion without
        a finite volume and centroid is an error too."""
        enrolled = np.asarray(enrolled, dtype=np.int64)
        vol = np.asarray(volumes, dtype=np.float64)[enrolled]
        cen = np.asarray(centroids, dtype=np.float64)[enrolled]
        bad = int((~(np.isfinite(vol) & np.isfinite(cen).all(axis=1))).sum())
        if bad:
            raise ValueError(f"{bad} of {len(enrolled)} enrolled lesions have no finite volume and centroid: "
                             f"the {self.feature} rule cannot rank them")
        coord = principal_coordinate(centroids)[enrolled] if self.axis_from == "pool" else principal_coordinate(cen)
        values = vol if self.feature == "volume" else coord
        self.effect = np.full(len(volumes), np.nan)
        self.effect[enrolled] = self.sign * rule_effect(values, self.shape)
        self.coordinate = np.full(len(volumes), np.nan)
        self.coordinate[enrolled] = coord
        return self

    def mu(self, effect: np.ndarray, t: int) -> np.ndarray:
        """Expected outcome under arm ``t`` (1 or 0) of lesions with the given effect."""
        e = np.asarray(effect, dtype=np.float64)
        suits = 0.5 + (e if int(t) == 1 else -e) / 2.0
        return 1.0 - (1.0 - self.p_te * suits) * (1.0 - self.p_re)

    def propensity(self, effect: np.ndarray, coordinate: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """P(T = 1) for a cohort, by the virtual trial's three mechanisms."""
        e = np.asarray(effect, dtype=np.float64)
        n = len(e)
        if self.bias_type == "agnostic":
            return np.where(e > 0, self.bias, np.where(e < 0, 1.0 - self.bias, 0.5))
        key = np.asarray(coordinate if self.bias_type == "axis" else volumes, dtype=np.float64)
        rank = np.argsort(np.argsort(self.ramp_sign * key, kind="stable"), kind="stable")
        return _ramp(rank, n, self.bias)

    def describe(self) -> Dict:
        return {"feature": self.feature, "shape": self.shape, "sign": self.sign,
                "p_te": self.p_te, "p_re": self.p_re, "bias": self.bias, "bias_type": self.bias_type,
                "label_noise": self.label_noise, "enrol": self.enrol, "enrol_by": self.enrol_by,
                "axis": self.axis_from,
                "outcome": "continuous" if self.continuous else "binary"}


def make_rule_dataset(dgp: FeatureRuleDGP, ctx: np.ndarray, qry: np.ndarray, volumes: np.ndarray,
                      X_ctx: np.ndarray, X_qry: np.ndarray, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Context (treatment and outcome drawn from the process) and queries with
    both expected outcomes, as ``neuro_prior.make_prior_dataset`` returns
    them. ``ctx`` and ``qry`` are pool indices of enrolled lesions; ``u_ctx``
    is the sign of each context lesion's effect, for the verifier."""
    e_c, e_q = dgp.effect[ctx], dgp.effect[qry]
    if np.isnan(e_c).any() or np.isnan(e_q).any():
        raise ValueError("a lesion outside the process's enrolment was drawn")
    p = dgp.propensity(e_c, dgp.coordinate[ctx], np.asarray(volumes)[ctx])
    Tc = (rng.uniform(size=len(e_c)) < p).astype(np.float64)
    mu_c = np.where(Tc == 1, dgp.mu(e_c, 1), dgp.mu(e_c, 0))
    if dgp.continuous:
        Yc = mu_c.astype(np.float64)
    else:
        Yc = (rng.uniform(size=len(e_c)) < mu_c).astype(np.float64)
        if dgp.label_noise > 0:
            flip = rng.uniform(size=len(Yc)) < dgp.label_noise
            Yc = np.where(flip, 1.0 - Yc, Yc)
    Tq = rng.integers(0, 2, size=len(e_q)).astype(np.float64)
    mu0, mu1 = dgp.mu(e_q, 0), dgp.mu(e_q, 1)
    return {"Xc": np.asarray(X_ctx, np.float64), "Tc": Tc, "Yc": Yc,
            "Xq": np.asarray(X_qry, np.float64), "Tq": Tq,
            "mu_q": np.where(Tq == 1, mu1, mu0), "mu0": mu0, "mu1": mu1,
            "e_ctx": p, "u_ctx": np.sign(e_c)}
