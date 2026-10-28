"""Neuro-Prior cohort.

Iterates over synthetic data-generating processes, filters them with the
identifiability verifier and stacks contexts and queries into batches to train
the transformer. Returns numpy arrays; the conversion to tensors is done in the
model layer so that this module does not depend on torch.
"""
from typing import Dict, Optional, Sequence

import numpy as np

from ..utils.logging_utils import get_logger
from .atlas import FunctionalAtlas, _centroid
from .channel import cache_channel, describe_cache
from .intersynth import MECHANISMS, SyntheticDGP, make_dataset
from .intersynth_atlas import InterSynthDGP, compute_overlaps, make_intersynth_dataset
from .verify_identifiability import verify_identifiability

log = get_logger()

# process families of NeuroPriorCohort and where a process draws its context (see its docstring)
FAMILIES = ("theory", "giles", "mixture")
CONTEXT_POOLS = ("susceptible", "all")


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
    the virtual-trial ranges.

    A cohort built from a cache exposes ``channel`` ("lesion" for binary
    masks, "disconnectome" for binarised continuous maps, None when the cache
    predates the record) and ``cache_meta`` (images_dir, threshold, modality
    when recorded), and logs the channel with the per-network responsive
    fractions at load time: the overlaps, and with them which lesions a
    process can affect, are a function of the channel the cache was built
    from, which must be the one the latents were exported from.

    ``context_pool`` says which lesions a process draws its context and
    queries from. ``"susceptible"`` (default) restricts both to the lesions the
    process can affect, as the virtual trial does when it evaluates one
    deficit on the patients susceptible to it: for the giles family the
    lesions hitting one of the process's causal networks above its threshold,
    for the theory family the lesions with a non-zero disruption score.
    The queries are drawn without replacement from that slice and the
    context, of the requested length, from the rest of the slice with
    replacement: a lesion that appears twice is two patients with the same
    anatomy and their own treatment and outcome draws. Processes whose slice
    is smaller than ``min_slice`` are redrawn. On a real pool most lesions
    hit no network, so
    ``"all"`` (the pilot) yields processes whose effect is zero for almost
    every patient, and a model trained on them predicts no effect anywhere."""

    def __init__(self, atlas: FunctionalAtlas, lesion_pool: Optional[np.ndarray], seed: int = 0,
                 z_pool=None, n_context: int = 128, n_query: int = 16, hyper=None,
                 max_tries: int = 64, augment: bool = True, cache: Optional[Dict] = None,
                 family: str = "theory", hyper_giles=None,
                 family_weights: Sequence[float] = (0.5, 0.5), context_pool: str = "susceptible",
                 min_slice: int = 32, context_replace: bool = True, standardize: str = "pool"):
        from .intersynth_theory import TheoryHyperPrior
        from .neuro_prior import HyperPrior

        # audit of 19 Oct: (i) a context drawn with replacement shows the same
        # patient under both arms, which never happens at scoring;
        # context_replace=False draws distinct lesions and caps the context at
        # the slice; (ii) scoring standardises with the context's statistics
        # while training standardised the pool once; standardize="context"
        # does the same in training, so the two paths match by construction.
        self.context_replace = bool(context_replace)
        if standardize not in ("pool", "context"):
            raise ValueError("standardize must be 'pool' or 'context'")
        self.standardize = standardize
        if family not in FAMILIES:
            raise ValueError(f"unknown prior family {family!r}; choose from {FAMILIES}")
        self.family = family
        w = np.asarray(family_weights, dtype=float)
        if w.shape != (2,) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("family_weights must be two non-negative numbers (theory, giles)")
        self.family_weights = w / w.sum()
        if context_pool not in CONTEXT_POOLS:
            raise ValueError(f"unknown context_pool {context_pool!r}; choose from {CONTEXT_POOLS}")

        self.channel, self.cache_meta = None, {}
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
            # extra anatomy channels (scripts/build_prior_cache.py, 15 Oct):
            # whole-network overlap and the halves of both splits
            self.overlaps_extra = {k: cache[k] for k in ("overlaps_network", "overlaps_receptor", "overlaps_genetics")
                                   if k in cache}
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
            self.channel = cache_channel(cache)
            self.cache_meta = {k: np.asarray(cache[k]).item() for k in ("images_dir", "threshold", "modality")
                               if k in cache and np.asarray(cache[k]).ndim == 0}
            log.info(describe_cache(cache))
        # a slice must hold the queries plus a context worth inferring from
        self.context_pool, self.min_slice = context_pool, max(int(min_slice), self.n_query + 8)
        self.hyper = hyper or (HyperPrior() if family == "giles" else TheoryHyperPrior())
        # the mixture's virtual-trial ranges (the giles family uses ``hyper``)
        self.hyper_giles = hyper_giles or HyperPrior()
        # the theory family's anatomy view: the cache's two split columns by
        # default; three named channels (D, S_A, S_B sources) when asked
        self.overlaps_theory = self._theory_view()
        self.max_tries = int(max_tries)
        self.n_rejected = 0
        self.overlaps_extra = getattr(self, "overlaps_extra", {})
        self._atoms = None                 # (labels, table) of the anatomical atoms, built on first use
        # Coordinate-free covariates: the per-fold encoders (and external
        # cohorts) each have their own latent basis, so the transformer must
        # not learn one. The pool is standardised once and every process
        # applies its own random orthogonal rotation to context and queries;
        # at scoring time PFNEstimator standardises with the context statistics.
        self.augment = bool(augment)
        if self.augment and self.standardize == "pool":
            mu, sd = self.X.mean(0, keepdims=True), self.X.std(0, keepdims=True) + 1e-6
            self.X = (self.X - mu) / sd

    def _theory_view(self) -> Optional[np.ndarray]:
        """[m, K, 3] overlaps of the D, S_A and S_B sources named by the theory
        hyper-prior, or None when it keeps the defaults (the cache's two split
        columns, the behaviour of every run before 15 Oct)."""
        from .intersynth_atlas import channel_column

        h = self.hyper if self.family in ("theory", "mixture") else None
        if h is None or not hasattr(h, "uses_extra_channels") or not h.uses_extra_channels():
            return None

        def col(name: str) -> np.ndarray:
            if name in ("split_1", "split_2"):
                return self.overlaps[:, :, int(name[-1]) - 1]
            if not getattr(self, "overlaps_extra", None):
                raise ValueError(f"the prior asks for the anatomy channel {name!r} but the cache has no extra "
                                 "channels; rebuild it with scripts/build_prior_cache.py --augment <cache>")
            return channel_column(self.overlaps_extra, name)

        sb = h.s_b_source or (h.d_source if int(h.s_b_column) == 0 else h.s_source)
        view = np.stack([col(h.d_source), col(h.s_source), col(sb)], axis=2).astype(np.float32)
        log.info("theory anatomy view: D <- %s, S_A <- %s, S_B <- %s", h.d_source, h.s_source, sb)
        return view

    def _theory_ov(self) -> np.ndarray:
        return self.overlaps if getattr(self, "overlaps_theory", None) is None else self.overlaps_theory

    @classmethod
    def from_cache(cls, atlas: FunctionalAtlas, cache_path: str, **kw):
        """Cohort from an npz written by scripts/build_prior_cache.py (keys
        overlaps, centroids, volumes, optional Z and files; channel,
        images_dir, threshold and modality when the builder recorded them)."""
        with np.load(cache_path, allow_pickle=True) as z:
            cache = {k: z[k] for k in z.files}
        return cls(atlas, None, cache=cache, **kw)

    def _rotation(self) -> np.ndarray:
        q, r = np.linalg.qr(self.rng.normal(size=(self.d_x, self.d_x)))
        return q * np.sign(np.diag(r))[None, :]

    def _slice(self, dgp, family: str) -> Optional[np.ndarray]:
        """Indices of the pool lesions the process can affect (None = the
        whole pool, when context_pool is "all")."""
        if self.context_pool == "all":
            return None
        if family == "giles":
            ov = self._giles_ov(dgp)
            idx = [dgp.atlas.networks.index(n) for n in dgp.networks]
            hit = (ov[:, idx, :] > dgp.threshold).any(axis=(1, 2))
        else:
            hit = dgp.raw_scores(self._theory_ov())[0] > 0.0
        return np.flatnonzero(hit)

    def _indices_from(self, pool: Optional[np.ndarray], n: int) -> np.ndarray:
        if pool is None:
            return self._indices(n)
        return pool[self.rng.permutation(len(pool))[:n]]

    def _draw_process(self):
        """(family, dgp, slice) of one process whose slice holds at least
        min_slice lesions (None slice = whole pool)."""
        fam = self._draw_family()            # the family is drawn once: rejections must not shift the mixture weights
        for _ in range(self.max_tries):
            dgp = self.sample_process(fam)
            sl = self._slice(dgp, fam)
            if sl is None or len(sl) >= self.min_slice:
                return fam, dgp, sl
            self.n_rejected += 1
        raise RuntimeError(f"no process with at least {self.min_slice} susceptible lesions in {self.max_tries} draws")

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
            return InterSynthTheoryDGP(self.atlas, self.rng, self.hyper).calibrate(self._theory_ov())
        from .neuro_prior import NeuroPriorDGP

        h = self.hyper if self.family == "giles" else self.hyper_giles
        if float(getattr(h, "random_partitions", 0.0) or 0.0) > 0 and self.rng.uniform() < float(h.random_partitions):
            # route C: the process's networks are random partitions of the
            # cache's anatomical atoms, never the benchmark's own halves
            from .partitions import VirtualAtlas, atom_table, draw_virtual_networks

            if self._atoms is None:
                self._atoms = atom_table(self.overlaps, self.overlaps_extra)
            labels, table = self._atoms
            k = int(self.rng.integers(h.n_networks[0], h.n_networks[1] + 1))
            vov, desc = draw_virtual_networks(self.rng, labels, table, k, atoms_per_side=h.atoms_per_side)
            atlas = VirtualAtlas(vov, self.centroids)
            dgp = NeuroPriorDGP(atlas, self.rng, h, networks=list(atlas.networks))
            dgp.virtual_overlaps = vov
            dgp.virtual_desc = desc
            return dgp
        return NeuroPriorDGP(self.atlas, self.rng, h)

    def _giles_ov(self, dgp) -> np.ndarray:
        """The [m, K, 2] overlaps a virtual-trial process reads: the cache's
        atlas halves, or the process's own virtual networks."""
        v = getattr(dgp, "virtual_overlaps", None)
        return self.overlaps if v is None else v

    def _one(self, n_context: int, drawn=None) -> Dict[str, np.ndarray]:
        from .intersynth_theory import make_theory_dataset
        from .neuro_prior import make_prior_dataset

        data = None
        for attempt in range(self.max_tries):
            fam, dgp, sl = drawn if (drawn is not None and attempt == 0) else self._draw_process()
            if sl is None:
                ci, qi = self._indices(n_context), self._indices(self.n_query)
            else:
                # queries without replacement; the context of the requested
                # length from the rest of the slice, with replacement
                perm = sl[self.rng.permutation(len(sl))]
                qi, rest = perm[: self.n_query], perm[self.n_query:]
                if self.context_replace:
                    ci = rest[self.rng.integers(0, len(rest), size=n_context)]
                else:
                    # distinct lesions first; a slice shorter than the context is
                    # topped up with replacement so every item of a batch has
                    # the same length (audit of 27 Oct: a short slice broke the
                    # batch stacking)
                    ci = rest[: n_context]
                    if len(ci) < n_context:
                        ci = np.concatenate([ci, rest[self.rng.integers(0, len(rest), size=n_context - len(ci))]])
            Xc, Xq = self.X[ci], self.X[qi]
            if self.standardize == "context":
                mu, sd = Xc.mean(0, keepdims=True), Xc.std(0, keepdims=True) + 1e-6
                Xc, Xq = (Xc - mu) / sd, (Xq - mu) / sd
            if self.augment:
                R = self._rotation(); Xc, Xq = Xc @ R, Xq @ R
            if fam == "theory":
                ov_t = self._theory_ov()
                data = make_theory_dataset(dgp, ov_t[ci], self.centroids[ci], Xc,
                                           ov_t[qi], self.centroids[qi], Xq, self.rng)
                U = None
            else:
                ov_g = self._giles_ov(dgp)
                data = make_prior_dataset(dgp, ov_g[ci], self.centroids[ci], self.volumes[ci], Xc,
                                          ov_g[qi], self.centroids[qi], self.volumes[qi], Xq, self.rng)
                U = data["u_ctx"][:, None] if dgp.bias_type == "agnostic" else None
            n_slice = int(self.m if sl is None else len(sl))
            if verify_identifiability(data["Tc"], data["Xc"], None, None, U=U):
                data["process"] = {**dgp.describe(), "family": fam, "n_slice": n_slice}
                return data
            self.n_rejected += 1
        data["process"] = {**dgp.describe(), "family": fam, "n_slice": n_slice}
        return data

    def sample_batch(self, batch_size: int, n_context=None) -> Dict[str, np.ndarray]:
        n_ctx = self.n_context if n_context is None else int(n_context)
        items = [self._one(n_ctx) for _ in range(batch_size)]
        keys = ("Xc", "Tc", "Yc", "Xq", "Tq", "mu_q", "mu0", "mu1")
        out = {k: np.stack([it[k] for it in items], axis=0) for k in keys}
        out["processes"] = [it["process"] for it in items]
        return out
