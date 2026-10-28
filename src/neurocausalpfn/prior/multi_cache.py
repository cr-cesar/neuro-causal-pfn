"""Several anatomy caches behind one prior (multi-encoder pools, 28 Oct).

A per-fold PFN trained on one cache sees one latent geometry: its own
encoder's. The random rotation of the cohort (``augment``) covers orthogonal
changes of basis only; other encoders of the same patients (the other folds
and seeds of the variant) and other cohorts differ by more than a rotation:
scales mixed across axes, non-linear warps, another shape of the cloud. Each
cache here is a cohort of its own (same anatomy, another ``Z``, or another
cohort); one is drawn per batch, so a batch keeps a single geometry, as a
fold does at scoring time. The fold's own cache is the anchor and keeps a set
share of the batches (``cache_anchor_weight``), the rest is shared equally.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def cache_weights(n: int, weights: Optional[Sequence[float]] = None,
                  anchor_weight: Optional[float] = None) -> np.ndarray:
    """Draw probabilities of ``n`` caches: explicit ``weights``; else the
    first cache (the anchor) gets ``anchor_weight`` and the others share the
    rest equally; else uniform."""
    if n < 1:
        raise ValueError("no caches")
    if weights is not None:
        w = np.asarray(list(weights), dtype=float)
        if len(w) != n:
            raise ValueError(f"{len(w)} cache weights for {n} caches")
    elif anchor_weight is not None and n > 1:
        a = float(anchor_weight)
        if not 0.0 <= a <= 1.0:
            raise ValueError("cache_anchor_weight must be in [0, 1]")
        w = np.r_[a, np.full(n - 1, (1.0 - a) / (n - 1))]
    else:
        w = np.ones(n)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("cache weights must be non-negative with a positive sum")
    return w / w.sum()


class MultiCachePrior:
    """One Neuro-Prior cohort per cache; ``sample_batch`` draws a cohort, then
    a batch from it. Reads of other attributes (family, channel, ...) come
    from the first cohort; the hyper-priors a curriculum stage sets reach
    every cohort."""

    def __init__(self, priors: Sequence, weights: Optional[Sequence[float]] = None, seed: int = 0):
        priors = list(priors)
        if not priors:
            raise ValueError("no cohorts")
        dims = sorted({int(p.d_x) for p in priors})
        if len(dims) != 1:
            raise ValueError(f"the caches have different latent sizes: {dims}")
        object.__setattr__(self, "priors", priors)
        object.__setattr__(self, "weights", cache_weights(len(priors), weights))
        object.__setattr__(self, "rng", np.random.default_rng(seed + 104729))
        object.__setattr__(self, "counts", np.zeros(len(priors), dtype=int))

    @property
    def d_x(self) -> int:
        return int(self.priors[0].d_x)

    def __getattr__(self, name):
        return getattr(self.priors[0], name)

    def __setattr__(self, name, value):
        if name in ("hyper", "hyper_giles"):
            for p in self.priors:
                setattr(p, name, value)
        else:
            object.__setattr__(self, name, value)

    def sample_batch(self, batch_size: int, n_context: Optional[int] = None) -> Dict:
        i = int(self.rng.choice(len(self.priors), p=self.weights))
        self.counts[i] += 1
        p = self.priors[i]
        return p.sample_batch(batch_size, n_context=n_context) if n_context is not None \
            else p.sample_batch(batch_size)

    def describe(self) -> str:
        first = self.priors[0].describe() if hasattr(self.priors[0], "describe") else ""
        return (f"{len(self.priors)} caches, anchor share {self.weights[0]:.2f}"
                + (f"; {first}" if first else ""))


def build_multi_cache(priors: List, pr: Dict, seed: int):
    """The single cohort when there is one cache, else a MultiCachePrior with
    the configured shares (``cache_weights`` or ``cache_anchor_weight``)."""
    if len(priors) == 1:
        return priors[0]
    return MultiCachePrior(priors, cache_weights(len(priors), pr.get("cache_weights"),
                                                 pr.get("cache_anchor_weight")), seed=seed)
