"""Generic effect processes mixed into a Neuro-Prior cohort (23 Oct).

Audit of 22 to 23 Oct: the checkpoints read their own prior well (correlation
0.80 with the true effect, above ridge and ExtraTrees on the same context)
but cannot read a simple effect on the same covariates: a sparse linear
score, or three groups by that score (the virtual trial's shape, tau in
{-1, 0, +1} with about 60 % zeros), gives correlation 0.5 against 0.87 to
0.98 for ridge. The anatomy prior is one function class; the model learned
that class and nothing wider.

This wrapper keeps the cohort's covariates, treatment and slices and, for a
fraction ``weight`` of the processes, replaces the outcome model by a process
of a simple shape in latent space:

- ``linear``: CATE proportional to a sparse linear score of the covariates;
- ``step``: a null fraction (drawn from ``null``) with zero effect, the rest
  split into a positive and a negative group by the same score;
- ``mlp`` and ``mlp_step``: the same two shapes on a random one-hidden-layer
  network score (the non-linear part of a real effect).

``generic_active=lo:hi`` is the number of latent dimensions a score uses
(default 2 to 8; up to d_x for dense effects); ``generic_baseline=lo:hi`` is
how much the baseline risk follows the effect's own score (default 0).

The baseline risk is a different random linear score; the effect size is
drawn from ``scale`` (mean |CATE| for ``linear``, the amplitude of the
affected patients for ``step``; the trial's amplitude is 1 on the probability
scale, so the default range reaches 0.9, clipped at the probability bounds). Outcomes are continuous
probabilities, as in the theory family, or Bernoulli draws for a fraction
``binary`` of these processes. The hyper options travel in the PRIOR_HYPER
string of the training jobs: ``generic_weight=0.5 generic_scale=0.1:0.9
generic_null=0.3:0.8 generic_kinds=linear+step+mlp+mlp_step generic_active=2:50
generic_baseline=0:0.7 generic_binary=0``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

GENERIC_KEYS = ("generic_weight", "generic_scale", "generic_null", "generic_kinds",
                "generic_binary", "generic_active", "generic_baseline")


def split_generic_hyper(hyper: Dict) -> Tuple[Dict, Dict]:
    """(hyper without the generic_* keys, the generic_* keys)."""
    gen = {k: v for k, v in hyper.items() if k in GENERIC_KEYS}
    rest = {k: v for k, v in hyper.items() if k not in GENERIC_KEYS}
    return rest, gen


def _rng_range(v, default):
    if v is None:
        return tuple(default)
    if np.isscalar(v):
        return (float(v), float(v))
    return (float(v[0]), float(v[-1]))


def _score(rng: np.random.Generator, X: np.ndarray, kind: str, n_active: int) -> np.ndarray:
    """A standardised score of the covariates: a sparse linear one, or a random
    one-hidden-layer tanh network on the same active dimensions (the
    non-linear part of a real effect; the trial's effects are 56 % linear in
    the latents, so a prior of linear scores alone covers half of them)."""
    d = X.shape[1]
    active = rng.choice(d, size=max(1, min(n_active, d)), replace=False)
    if kind.startswith("mlp"):
        h = int(rng.integers(4, 17))
        W1 = rng.normal(size=(len(active), h)) / np.sqrt(len(active)); b1 = rng.normal(size=h) * 0.5
        W2 = rng.normal(size=h) / np.sqrt(h)
        z = np.tanh(X[:, active] @ W1 + b1) @ W2
    else:
        z = X[:, active] @ rng.normal(size=len(active))
    return (z - z.mean()) / (z.std() + 1e-8)


def generic_process(rng: np.random.Generator, Xc: np.ndarray, Xq: np.ndarray, kind: str,
                    scale: float, null_frac: float, n_active: int,
                    p0_range=(0.2, 0.8), baseline_shared: float = 0.0) -> Dict[str, np.ndarray]:
    """Potential outcomes of a simple process on given covariates; returns
    p0 and p1 for the context and the query rows. ``kind`` is ``linear`` or
    ``step`` (sparse linear score), ``mlp`` or ``mlp_step`` (random network
    score); ``baseline_shared`` in [0, 1] is how much the baseline risk follows
    the effect's own score (in the trial the deficit and its treatment effect
    come from the same anatomy)."""
    if kind not in ("linear", "step", "mlp", "mlp_step"):
        raise ValueError(f"unknown generic kind {kind!r}")
    X = np.vstack([Xc, Xq]); d = X.shape[1]
    z = _score(rng, X, kind, n_active)
    if kind.endswith("step"):
        q = 1.0 - null_frac                       # non-null share
        r = rng.uniform(0.2, 0.8)                 # positive share of the non-null
        hi, lo = np.quantile(z, [1.0 - q * r, q * (1.0 - r)])
        tau = np.where(z >= hi, 1.0, np.where(z <= lo, -1.0, 0.0)) * scale   # amplitude per affected patient
    else:
        tau = z * scale / (np.abs(z).mean() + 1e-8)
    v = rng.normal(size=d) / np.sqrt(d)
    own = (X @ v); own = (own - own.mean()) / (own.std() + 1e-8)
    base_score = (1.0 - baseline_shared) * own + baseline_shared * z * rng.choice([-1.0, 1.0])
    p0 = np.clip(rng.uniform(*p0_range) + 0.15 * base_score, 0.05, 0.95)
    p1 = np.clip(p0 + tau, 0.0, 1.0)
    nc = len(Xc)
    return {"p0c": p0[:nc], "p1c": p1[:nc], "p0q": p0[nc:], "p1q": p1[nc:]}


class GenericEffectMixture:
    """A Neuro-Prior cohort whose processes are, with probability ``weight``,
    generic effect processes on the cohort's own covariates. Every other
    attribute (d_x, hyper, channel, ...) is the wrapped cohort's."""

    def __init__(self, base, weight: float, scale=(0.1, 0.9), null=(0.3, 0.8),
                 kinds: Sequence[str] = ("linear", "step"), n_active=(2, 8),
                 binary: float = 0.0, baseline_shared=(0.0, 0.0), seed: int = 0):
        self.base = base
        self.baseline_shared = _rng_range(baseline_shared, (0.0, 0.0))
        self.weight = float(weight)
        self.scale = _rng_range(scale, (0.1, 0.9))
        self.null = _rng_range(null, (0.3, 0.8))
        self.kinds = tuple(kinds)
        self.n_active = _rng_range(n_active, (2, 8))
        self.binary = float(binary)
        self.rng = np.random.default_rng(seed + 7919)
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("generic_weight must be in [0, 1]")
        if not self.kinds:
            raise ValueError("generic_kinds is empty")

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def hyper(self):
        return self.base.hyper

    @hyper.setter
    def hyper(self, value):
        self.base.hyper = value

    # the mixture's virtual-trial ranges, which a curriculum stage also sets:
    # without the setter the stage's value stayed on the wrapper
    @property
    def hyper_giles(self):
        return self.base.hyper_giles

    @hyper_giles.setter
    def hyper_giles(self, value):
        self.base.hyper_giles = value

    def describe(self) -> str:
        return (f"generic effect processes: weight {self.weight:.2f}, kinds {'+'.join(self.kinds)}, "
                f"effect size {self.scale[0]:.2f}-{self.scale[1]:.2f}, null share {self.null[0]:.2f}-{self.null[1]:.2f}, "
                f"active dims {int(self.n_active[0])}-{int(self.n_active[1])}, baseline shared "
                f"{self.baseline_shared[0]:.1f}-{self.baseline_shared[1]:.1f}, binary outcomes {self.binary:.2f}")

    def _replace(self, out: Dict, i: int) -> str:
        rng = self.rng
        kind = self.kinds[rng.integers(len(self.kinds))]
        pr = generic_process(rng, out["Xc"][i], out["Xq"][i], kind,
                             scale=rng.uniform(*self.scale), null_frac=rng.uniform(*self.null),
                             n_active=int(rng.integers(int(self.n_active[0]), int(self.n_active[1]) + 1)),
                             baseline_shared=rng.uniform(*self.baseline_shared))
        Tc, Tq = out["Tc"][i], out["Tq"][i]
        yc = np.where(Tc == 1, pr["p1c"], pr["p0c"])
        if self.binary > 0 and rng.uniform() < self.binary:
            yc = (rng.uniform(size=len(yc)) < yc).astype(yc.dtype)
        out["Yc"][i] = yc
        out["mu0"][i] = pr["p0q"]; out["mu1"][i] = pr["p1q"]
        out["mu_q"][i] = np.where(Tq == 1, pr["p1q"], pr["p0q"])
        return kind

    def sample_batch(self, batch_size: int, n_context: Optional[int] = None) -> Dict:
        out = self.base.sample_batch(batch_size, n_context=n_context) if n_context is not None \
            else self.base.sample_batch(batch_size)
        for k in ("Yc", "mu0", "mu1", "mu_q"):
            out[k] = np.array(out[k], dtype=float, copy=True)
        procs = out.get("processes")
        for i in range(batch_size):
            if self.rng.uniform() < self.weight:
                kind = self._replace(out, i)
                if procs is not None and isinstance(procs[i], dict):
                    procs[i] = {**procs[i], "family": f"generic-{kind}"}
        return out


def wrap_generic(prior, gen: Dict, seed: int = 0):
    """Wrap ``prior`` when the generic_* hyper options ask for it; else return it."""
    w = float(gen.get("generic_weight", 0.0) or 0.0)
    if w <= 0:
        return prior
    kinds = gen.get("generic_kinds", "linear+step")
    kinds = tuple(k for k in str(kinds).replace(",", "+").split("+") if k) if isinstance(kinds, str) else tuple(kinds)
    return GenericEffectMixture(prior, w, scale=gen.get("generic_scale"), null=gen.get("generic_null"),
                                kinds=kinds, n_active=gen.get("generic_active"),
                                binary=float(gen.get("generic_binary", 0.0) or 0.0),
                                baseline_shared=gen.get("generic_baseline"), seed=seed)
