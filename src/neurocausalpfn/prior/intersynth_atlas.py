"""InterSynth with anatomical substrate (the faithful version of the mechanism).

Unlike the lightweight generator in intersynth.py (which samples Gaussian
covariates from scratch), this module crosses each lesion with the functional
parcellation to fabricate the semi-synthetic ground truth, following the Giles
framework:

- Deficit: a lesion that covers at least 5% of a subnetwork causes the
  corresponding deficit.
- True treatment susceptibility: depends on which of the two subnetworks
  (transcriptomic or receptomic) of the causal network the lesion
  predominantly touches. This susceptibility is never shown to the model, it is
  only ground truth.
- Outcome: a treatment effect (TE) and a spontaneous recovery (SR) are combined
  as Bernoulli variables, which gives known expected conditional potential
  outcomes mu_0 and mu_1, and therefore a known CATE.
- Treatment assignment: observed confounding by the distance from the lesion
  centroid to the sensitive subnetwork (hyperparameter b); optionally unobserved
  confounding by leakage from the true susceptibility (which violates
  identifiability and must be rejected by the verifier).

Each sampled condition (causal network, subnetwork-to-treatment mapping, TE, SR,
assignment bias) is a process of the prior. The covariate X seen by the
transformer is the lesion representation: the frozen encoder latent if provided,
or, otherwise, the observed overlap fractions.
"""
import numpy as np

from .atlas import FunctionalAtlas, _centroid


def compute_overlaps(atlas: FunctionalAtlas, lesion: np.ndarray) -> np.ndarray:
    """Overlap fractions [K, 2] of the lesion with the two subnetworks of each
    network in the atlas."""
    out = np.zeros((atlas.n_networks, 2), dtype=np.float32)
    for i, k in enumerate(atlas.networks):
        out[i] = atlas.subnetwork_overlap(lesion, k)
    return out


class InterSynthDGP:
    """A condition (a process of the prior) sampled from the InterSynth framework."""

    def __init__(self, atlas: FunctionalAtlas, rng: np.random.Generator,
                 unobserved_strength: float = 0.0):
        self.atlas = atlas
        self.network_idx = int(rng.integers(0, atlas.n_networks))
        self.network = atlas.networks[self.network_idx]
        self.optimal_for_A = int(rng.integers(0, 2))       # subnetwork A responds to this treatment
        self.p_te = float(rng.uniform(0.4, 0.8))           # prob of treatment effect
        self.p_re = float(rng.uniform(0.1, 0.4))           # prob of spontaneous recovery
        self.bias_b = float(rng.uniform(0.5, 2.0))         # strength of the observed confounding
        self.unobserved_strength = float(unobserved_strength)
        self.deficit_threshold = 0.05
        self._scale = float(np.prod(atlas.shape) ** (1.0 / 3.0))

    def susceptibility(self, oa: float, ob: float):
        """0 if the lesion predominantly touches subnetwork A, 1 if B, None if
        it does not touch the network (not differentiable, the treatment does not
        change the outcome)."""
        if max(oa, ob) < self.deficit_threshold:
            return None
        return 0 if oa >= ob else 1

    def optimal_treatment(self, s):
        if s is None:
            return None
        return self.optimal_for_A if s == 0 else (1 - self.optimal_for_A)

    def mu(self, s, t: int) -> float:
        """Expected conditional potential outcome: probability of a good outcome
        under treatment t. Combines TE (if the treatment is the right one for the
        susceptibility) and SR (spontaneous recovery)."""
        works = (s is not None) and (t == self.optimal_treatment(s))
        p_treat = self.p_te if works else 0.0
        return 1.0 - (1.0 - p_treat) * (1.0 - self.p_re)

    def cate(self, s) -> float:
        return self.mu(s, 1) - self.mu(s, 0)

    def propensity(self, lesion_centroid: np.ndarray, s) -> float:
        """Probability of receiving treatment 1. The observed confounding
        depends only on the lesion geometry (its centroid); the unobserved
        leakage, optional, depends on the true susceptibility."""
        ca = self.atlas.centroid(self.network, 0)
        cb = self.atlas.centroid(self.network, 1)
        d = float(np.linalg.norm(lesion_centroid - ca) - np.linalg.norm(lesion_centroid - cb))
        signal = -self.bias_b * d / self._scale
        if self.unobserved_strength > 0.0 and s is not None:
            signal += self.unobserved_strength * (1.0 if s == 1 else -1.0)
        return float(1.0 / (1.0 + np.exp(-signal)))


def make_intersynth_dataset(dgp: InterSynthDGP,
                            overlaps_ctx: np.ndarray, cent_ctx: np.ndarray, X_ctx: np.ndarray,
                            overlaps_qry: np.ndarray, cent_qry: np.ndarray, X_qry: np.ndarray,
                            rng: np.random.Generator):
    """Observational context dataset and query set with the known potential
    outcomes. overlaps_* are [n, 2] (the two subnetworks of the process's causal
    network); cent_* are [n, 3]; X_* are [n, d_x]."""
    s_ctx = [dgp.susceptibility(o[0], o[1]) for o in overlaps_ctx]
    e = np.array([dgp.propensity(cent_ctx[i], s_ctx[i]) for i in range(len(s_ctx))])
    Tc = (rng.uniform(size=len(s_ctx)) < e).astype(np.float64)
    mu_assigned = np.array([dgp.mu(s_ctx[i], int(Tc[i])) for i in range(len(s_ctx))])
    Yc = (rng.uniform(size=len(s_ctx)) < mu_assigned).astype(np.float64)

    s_qry = [dgp.susceptibility(o[0], o[1]) for o in overlaps_qry]
    Tq = rng.integers(0, 2, size=len(s_qry)).astype(np.float64)
    mu0 = np.array([dgp.mu(s, 0) for s in s_qry])
    mu1 = np.array([dgp.mu(s, 1) for s in s_qry])
    mu_q = np.where(Tq == 1, mu1, mu0)

    return {"Xc": np.asarray(X_ctx, np.float64), "Tc": Tc, "Yc": Yc,
            "Xq": np.asarray(X_qry, np.float64), "Tq": Tq,
            "mu_q": mu_q, "mu0": mu0, "mu1": mu1}
