"""Neuro-Prior cohort.

Iterates over synthetic data-generating processes, filters them with the
identifiability verifier and stacks contexts and queries into batches to train
the transformer. Returns numpy arrays; the conversion to tensors is done in the
model layer so that this module does not depend on torch.
"""
from typing import Dict, Sequence

import numpy as np

from .atlas import FunctionalAtlas, _centroid
from .intersynth import MECHANISMS, SyntheticDGP, make_dataset
from .intersynth_atlas import InterSynthDGP, compute_overlaps, make_intersynth_dataset
from .verify_identifiability import verify_identifiability


class NeuroPrior:
    def __init__(self, d_x: int, n_context: int, n_query: int, seed: int = 0,
                 mechanisms: Sequence[str] = MECHANISMS,
                 confound_range=None, effect_range=None):
        self.d_x = int(d_x)
        self.n_context = int(n_context)
        self.n_query = int(n_query)
        self.rng = np.random.default_rng(seed)
        self.mechanisms = tuple(mechanisms)
        self.confound_range = confound_range   # (lo, hi) for the confounding strength, or None
        self.effect_range = effect_range       # (lo, hi) for the effect scale, or None

    def _one(self) -> Dict[str, np.ndarray]:
        # retries until obtaining a process that passes the R1/R2 gate.
        # The processes are ignorable by construction (U=None); the filter
        # mostly rejects positivity violations in the sample.
        for _ in range(64):
            mech = str(self.rng.choice(self.mechanisms))
            cs = float(self.rng.uniform(*self.confound_range)) if self.confound_range else 1.0
            es = float(self.rng.uniform(*self.effect_range)) if self.effect_range else 1.0
            dgp = SyntheticDGP(self.d_x, self.rng, mechanism=mech,
                               confound_strength=cs, effect_scale=es)
            data = make_dataset(dgp, self.n_context, self.n_query, self.rng)
            if verify_identifiability(data["Tc"], data["Xc"], None, None, U=None):
                return data
        return data  # returns the last one if the retries are exhausted

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        items = [self._one() for _ in range(batch_size)]
        keys = ("Xc", "Tc", "Yc", "Xq", "Tq", "mu_q", "mu0", "mu1")
        return {k: np.stack([it[k] for it in items], axis=0) for k in keys}


def build_synthetic_lesion_pool(n: int, shape=(48, 56, 48), seed: int = 0) -> np.ndarray:
    """Set of synthetic masks to run InterSynth without real data.
    In the real pipeline, this set is replaced by the Giles masks."""
    from ..data.nifti_dataset import LesionMaskDataset

    ds = LesionMaskDataset(root=None, in_shape=shape, n_synth=n, seed=seed)
    return np.stack([ds._load_volume(i) for i in range(n)], axis=0)


class NeuroPriorInterSynth:
    """Neuro-Prior cohort with anatomical substrate (InterSynth).

    Precomputes, only once, the overlaps and centroids of each lesion in the
    set. Each batch samples a condition (a process of the prior) and, from
    indices of the set, builds context and queries with known potential
    outcomes. The covariate X is the encoder latent if z_pool is provided;
    otherwise, the observed overlap fractions plus the normalized centroid,
    which is a purely observed covariate and therefore preserves ignorability
    by construction.
    """

    def __init__(self, atlas: FunctionalAtlas, lesion_pool: np.ndarray, seed: int = 0,
                 z_pool=None, n_context: int = 128, n_query: int = 16,
                 unobserved_strength: float = 0.0):
        self.atlas = atlas
        self.rng = np.random.default_rng(seed)
        self.n_context = int(n_context)
        self.n_query = int(n_query)
        self.unobserved_strength = float(unobserved_strength)
        self.m = len(lesion_pool)

        self.overlaps = np.stack([compute_overlaps(atlas, lesion_pool[i]) for i in range(self.m)], axis=0)  # [m, K, 2]
        self.centroids = np.stack([_centroid(lesion_pool[i]) for i in range(self.m)], axis=0)               # [m, 3]
        if z_pool is not None:
            self.X = np.asarray(z_pool, dtype=np.float64)
        else:
            geo = self.centroids / np.array(atlas.shape, dtype=np.float64)
            self.X = np.concatenate([self.overlaps.reshape(self.m, -1), geo], axis=1)                        # [m, 2K + 3]
        self.d_x = int(self.X.shape[1])

    def _indices(self, n: int) -> np.ndarray:
        return self.rng.integers(0, self.m, size=n)

    def _one(self, n_context: int) -> Dict[str, np.ndarray]:
        for _ in range(64):
            dgp = InterSynthDGP(self.atlas, self.rng, unobserved_strength=self.unobserved_strength)
            ci, qi = self._indices(n_context), self._indices(self.n_query)
            k = dgp.network_idx
            data = make_intersynth_dataset(
                dgp,
                self.overlaps[ci, k, :], self.centroids[ci], self.X[ci],
                self.overlaps[qi, k, :], self.centroids[qi], self.X[qi],
                self.rng)
            if verify_identifiability(data["Tc"], data["Xc"], None, None, U=None):
                return data
        return data

    def sample_batch(self, batch_size: int, n_context=None) -> Dict[str, np.ndarray]:
        n_ctx = self.n_context if n_context is None else int(n_context)
        items = [self._one(n_ctx) for _ in range(batch_size)]
        keys = ("Xc", "Tc", "Yc", "Xq", "Tq", "mu_q", "mu0", "mu1")
        return {k: np.stack([it[k] for it in items], axis=0) for k in keys}
