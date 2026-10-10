"""Neuro-Prior cohort.

Iterates over synthetic data-generating processes, filters them with the
identifiability verifier and stacks contexts and queries into batches to train
the transformer. Returns numpy arrays; the conversion to tensors is done in the
model layer so that this module does not depend on torch.
"""
from typing import Dict, Optional, Sequence

import numpy as np

from .atlas import FunctionalAtlas, _centroid
from .intersynth import MECHANISMS, SyntheticDGP, make_dataset
from .intersynth_atlas import InterSynthDGP, compute_overlaps, make_intersynth_dataset
from .verify_identifiability import verify_identifiability

# process families of NeuroPriorCohort (see its docstring)
FAMILIES = ("theory", "giles", "mixture")


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


class NeuroPriorCohort(NeuroPriorInterSynth):
    """Neuro-Prior cohort: like ``NeuroPriorInterSynth`` but every batch item
    is a process drawn from a hyper-prior. Two families share the same
    anatomy cache and latent pool:

    * ``family="theory"`` (default): the design's InterSynth, disruption and
      susceptibility scores D and S, outcomes (1 - D) + alpha D and
      + beta S D, four allocation mechanisms of strength gamma
      (``intersynth_theory``); ignorable by construction.
    * ``family="giles"``: the v1 distribution over virtual-trial generators of
      the reference paper (``neuro_prior``: causal networks, threshold, TE,
      RE, three allocation mechanisms, label noise). Processes with
      susceptibility-driven allocation go to the verifier with the
      susceptibility as the unobserved variable and are rejected unless their
      strength is negligible.
    * ``family="mixture"``: every process is drawn from one of the two
      families with probabilities ``family_weights`` (theory, giles). The
      theory family alone has a non-negative, continuous and mostly small
      effect, whereas the virtual trial of the evaluation has binary outcomes
      and effects of either sign; the mixture keeps the design's family and
      covers the evaluation's.

    ``hyper`` is a ``TheoryHyperPrior`` or a ``HyperPrior`` matching the
    family (None gives the family's defaults); for the mixture it holds the
    theory ranges (the curriculum stages narrow those) and ``hyper_giles``
    the virtual-trial ranges."""

    def __init__(self, atlas: FunctionalAtlas, lesion_pool: Optional[np.ndarray], seed: int = 0,
                 z_pool=None, n_context: int = 128, n_query: int = 16, hyper=None,
                 max_tries: int = 64, augment: bool = True, cache: Optional[Dict] = None,
                 family: str = "theory", hyper_giles=None,
                 family_weights: Sequence[float] = (0.5, 0.5)):
        from .intersynth_theory import TheoryHyperPrior
        from .neuro_prior import HyperPrior

        if family not in FAMILIES:
            raise ValueError(f"unknown prior family {family!r}; choose from {FAMILIES}")
        self.family = family
        w = np.asarray(family_weights, dtype=float)
        if w.shape != (2,) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("family_weights must be two non-negative numbers (theory, giles)")
        self.family_weights = w / w.sum()

        if cache is None:
            super().__init__(atlas, lesion_pool, seed=seed, z_pool=z_pool,
                             n_context=n_context, n_query=n_query)
            self.volumes = np.asarray([float(lesion_pool[i].sum()) for i in range(self.m)])
        else:
            # precomputed anatomy (scripts/build_prior_cache.py): no masks in memory
            self.atlas = atlas
            self.rng = np.random.default_rng(seed)
            self.n_context, self.n_query = int(n_context), int(n_query)
            self.unobserved_strength = 0.0
            self.overlaps = np.asarray(cache["overlaps"], dtype=np.float32)
            self.centroids = np.asarray(cache["centroids"], dtype=np.float32)
            self.volumes = np.asarray(cache["volumes"], dtype=np.float64)
            self.m = len(self.overlaps)
            if z_pool is None and "Z" in cache:
                z_pool = cache["Z"]
            if z_pool is not None:
                self.X = np.asarray(z_pool, dtype=np.float64)
            else:
                geo = self.centroids / np.array(atlas.shape, dtype=np.float64)
                self.X = np.concatenate([self.overlaps.reshape(self.m, -1), geo], axis=1)
            self.d_x = int(self.X.shape[1])
            if len(self.X) != self.m or self.overlaps.shape[1:] != (atlas.n_networks, 2):
                raise ValueError("prior cache does not match the atlas or the latent pool")
        self.hyper = hyper or (HyperPrior() if family == "giles" else TheoryHyperPrior())
        # the mixture's virtual-trial ranges (the giles family uses ``hyper``)
        self.hyper_giles = hyper_giles or HyperPrior()
        self.max_tries = int(max_tries)
        self.n_rejected = 0
        # Coordinate-free covariates: the per-fold encoders (and external
        # cohorts) each have their own latent basis, so the transformer must
        # not learn one. The pool is standardised once and every process
        # applies its own random orthogonal rotation to context and queries;
        # at scoring time PFNEstimator standardises with the context statistics.
        self.augment = bool(augment)
        if self.augment:
            mu, sd = self.X.mean(0, keepdims=True), self.X.std(0, keepdims=True) + 1e-6
            self.X = (self.X - mu) / sd

    @classmethod
    def from_cache(cls, atlas: FunctionalAtlas, cache_path: str, **kw):
        """Cohort from an npz written by scripts/build_prior_cache.py (keys
        overlaps, centroids, volumes, optional Z and files)."""
        with np.load(cache_path, allow_pickle=True) as z:
            cache = {k: z[k] for k in z.files}
        return cls(atlas, None, cache=cache, **kw)

    def _rotation(self) -> np.ndarray:
        q, r = np.linalg.qr(self.rng.normal(size=(self.d_x, self.d_x)))
        return q * np.sign(np.diag(r))[None, :]

    def _draw_family(self) -> str:
        if self.family != "mixture":
            return self.family
        return str(self.rng.choice(("theory", "giles"), p=self.family_weights))

    def sample_process(self, family: Optional[str] = None):
        family = family or self._draw_family()
        if family == "theory":
            from .intersynth_theory import InterSynthTheoryDGP

            # the D and S scales are fixed on the whole pool, so the ground
            # truth of a process does not depend on the batch it is sampled in
            return InterSynthTheoryDGP(self.atlas, self.rng, self.hyper).calibrate(self.overlaps)
        from .neuro_prior import NeuroPriorDGP

        return NeuroPriorDGP(self.atlas, self.rng, self.hyper if self.family == "giles" else self.hyper_giles)

    def _one(self, n_context: int) -> Dict[str, np.ndarray]:
        from .intersynth_theory import make_theory_dataset
        from .neuro_prior import make_prior_dataset

        data = None
        for _ in range(self.max_tries):
            fam = self._draw_family()
            dgp = self.sample_process(fam)
            ci, qi = self._indices(n_context), self._indices(self.n_query)
            Xc, Xq = self.X[ci], self.X[qi]
            if self.augment:
                R = self._rotation(); Xc, Xq = Xc @ R, Xq @ R
            if fam == "theory":
                data = make_theory_dataset(dgp, self.overlaps[ci], self.centroids[ci], Xc,
                                           self.overlaps[qi], self.centroids[qi], Xq, self.rng)
                U = None
            else:
                data = make_prior_dataset(dgp, self.overlaps[ci], self.centroids[ci], self.volumes[ci], Xc,
                                          self.overlaps[qi], self.centroids[qi], self.volumes[qi], Xq, self.rng)
                U = data["u_ctx"][:, None] if dgp.bias_type == "agnostic" else None
            if verify_identifiability(data["Tc"], data["Xc"], None, None, U=U):
                data["process"] = {**dgp.describe(), "family": fam}
                return data
            self.n_rejected += 1
        data["process"] = {**dgp.describe(), "family": fam}
        return data

    def sample_batch(self, batch_size: int, n_context=None) -> Dict[str, np.ndarray]:
        n_ctx = self.n_context if n_context is None else int(n_context)
        items = [self._one(n_ctx) for _ in range(batch_size)]
        keys = ("Xc", "Tc", "Yc", "Xq", "Tq", "mu_q", "mu0", "mu1")
        out = {k: np.stack([it[k] for it in items], axis=0) for k in keys}
        out["processes"] = [it["process"] for it in items]
        return out
