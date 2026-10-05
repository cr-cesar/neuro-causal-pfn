"""Neuro-Prior v1: the reference virtual trial is one draw of the hyper-prior,
the allocation mechanisms behave as specified, and the cohort sampler yields
identifiable processes. CPU only, no torch (the lesion pool is built here)."""
import numpy as np
import pytest

from neurocausalpfn.prior.atlas import FunctionalAtlas
from neurocausalpfn.prior.cohort import NeuroPriorCohort
from neurocausalpfn.prior.giles_replica import HEADLINE_SCENARIOS
from neurocausalpfn.prior.intersynth_atlas import compute_overlaps
from neurocausalpfn.prior.neuro_prior import (BIAS_TYPES, HyperPrior, NeuroPriorDGP, _ramp,
                                              make_prior_dataset)
from neurocausalpfn.prior.verify_identifiability import verify_identifiability

SHAPE = (24, 28, 24)


def _pool(atlas, n, seed=0):
    """Ellipsoidal lesions centred on random atlas voxels so that many hit a network."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.indices(atlas.shape)
    fg = np.argwhere(atlas.network_labels > 0)
    out = np.zeros((n,) + atlas.shape, dtype=np.float32)
    for i in range(n):
        c = fg[rng.integers(0, len(fg))] if rng.uniform() < 0.8 else rng.integers(0, atlas.shape)
        r = rng.integers(2, 6, size=3)
        out[i] = ((((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2) <= 1).astype(np.float32)
    return out


@pytest.fixture(scope="module")
def atlas():
    return FunctionalAtlas.synthetic(shape=SHAPE, n_networks=6, seed=0)


@pytest.fixture(scope="module")
def pool(atlas):
    return _pool(atlas, 160)


def test_hyper_prior_contains_the_paper_scenarios():
    h = HyperPrior()
    for name, sc in HEADLINE_SCENARIOS.items():
        assert h.contains_giles(**sc), name
    assert not HyperPrior(p_te=(0.4, 0.8)).contains_giles(**HEADLINE_SCENARIOS["ideal"])


def test_ramp_matches_the_paper_linspace_and_saturates():
    r = np.arange(10)
    assert np.allclose(_ramp(r, 10, 0.3), np.linspace(0.2, 0.8, 10))
    assert np.allclose(_ramp(r, 10, 0.0), 0.5)
    hi = _ramp(r, 10, 1.0)
    assert hi[0] == 0.0 and hi[-1] == 1.0 and np.all(np.diff(hi) >= 0)


def test_from_giles_ideal_reproduces_the_virtual_trial(atlas):
    net = atlas.networks[0]
    dgp = NeuroPriorDGP.from_giles(atlas, net, **HEADLINE_SCENARIOS["ideal"])
    assert dgp.networks == [net] and dgp.threshold == 0.05 and dgp.label_noise == 0.0
    opt_a = dgp.optimal_treatment(net, 0.0)
    assert dgp.mu(net, 0.0, opt_a) == 1.0 and dgp.mu(net, 0.0, 1 - opt_a) == 0.0   # TE 1, RE 0
    assert dgp.mu(None, None, 0) == 0.0 and dgp.cate(None, None) == 0.0
    assert dgp.mu(net, 0.5, 0) == 1.0 and dgp.mu(net, 0.5, 1) == 1.0               # both subnetworks hit
    # the paper maps the larger subnetwork to treatment "1"
    a, b = atlas.subnetworks[net]
    assert opt_a == (1 if a.sum() >= b.sum() else 0)
    # BIAS 0: random allocation
    p = dgp.propensity(np.zeros((7, 3)), np.ones(7), [None] * 7, [None] * 7)
    assert np.allclose(p, 0.5)


def test_location_bias_is_monotone_along_the_separating_axis(atlas):
    net = atlas.networks[1]
    dgp = NeuroPriorDGP.from_giles(atlas, net, **HEADLINE_SCENARIOS["location_bias"])
    axis, sign = dgp._axis(net)
    cent = np.zeros((11, 3)); cent[:, axis] = np.arange(11) * sign
    p = dgp.propensity(cent, np.ones(11), [None] * 11, [None] * 11)
    assert p[0] == 0.0 and p[-1] == 1.0 and np.all(np.diff(p) >= 0)
    # the subnetwork treated by "1" lies on the high side of the ramp
    c1 = atlas.centroid(net, 0 if dgp.optimal_for_A[net] == 1 else 1)
    c0 = atlas.centroid(net, 1 if dgp.optimal_for_A[net] == 1 else 0)
    assert sign * c1[axis] >= sign * c0[axis]


def test_susceptibility_uses_the_first_causal_network_hit(atlas):
    rng = np.random.default_rng(3)
    nets = atlas.networks[:2]
    dgp = NeuroPriorDGP(atlas, rng, networks=nets, optimal_for_A=[1, 0], threshold=0.05)
    K = atlas.n_networks
    ov = np.zeros((K, 2)); ov[1, 1] = 0.3                      # hits only the second causal network (B side)
    assert dgp.susceptibility(ov) == (nets[1], 1.0)
    ov[0, 0] = 0.2                                              # now also the first one: it takes priority
    assert dgp.susceptibility(ov) == (nets[0], 0.0)
    ov[0, 1] = 0.2
    assert dgp.susceptibility(ov) == (nets[0], 0.5)
    assert dgp.susceptibility(np.zeros((K, 2))) == (None, None)


def test_sampled_processes_stay_inside_the_hyper_prior(atlas):
    rng = np.random.default_rng(0)
    h = HyperPrior()
    seen = set()
    for _ in range(200):
        d = NeuroPriorDGP(atlas, rng, h)
        assert h.n_networks[0] <= len(d.networks) <= h.n_networks[1] and len(set(d.networks)) == len(d.networks)
        assert h.p_te[0] <= d.p_te <= h.p_te[1] and h.p_re[0] <= d.p_re <= h.p_re[1]
        assert h.bias[0] <= d.bias <= h.bias[1] and h.threshold[0] <= d.threshold <= h.threshold[1]
        assert 0 <= d.label_noise <= h.label_noise[1]
        seen.add(d.bias_type)
    assert seen == set(BIAS_TYPES)


def test_outcomes_follow_te_and_re_in_expectation(atlas, pool):
    rng = np.random.default_rng(1)
    net = atlas.networks[0]
    dgp = NeuroPriorDGP.from_giles(atlas, net, TE=0.6, RE=0.2, BIAS=0.0, rng=rng)
    ov = np.stack([compute_overlaps(atlas, m) for m in pool])
    cent = np.stack([np.argwhere(m > 0.5).mean(0) for m in pool]); vol = pool.reshape(len(pool), -1).sum(1)
    X = ov.reshape(len(pool), -1)
    idx = rng.integers(0, len(pool), size=4000)
    d = make_prior_dataset(dgp, ov[idx], cent[idx], vol[idx], X[idx], ov[:8], cent[:8], vol[:8], X[:8], rng)
    nets = [dgp.susceptibility(o)[0] for o in ov[idx]]
    s = [dgp.susceptibility(o)[1] for o in ov[idx]]
    correct = np.array([dgp.optimal_treatment(n, si) in (-1, int(t)) and n is not None
                        for n, si, t in zip(nets, s, d["Tc"])])
    assert correct.sum() > 200 and (~correct).sum() > 200
    assert abs(d["Yc"][correct].mean() - (1 - 0.4 * 0.8)) < 0.05        # 1-(1-TE)(1-RE) = 0.68
    assert abs(d["Yc"][~correct].mean() - 0.2) < 0.05                   # RE only
    assert set(np.unique(d["Tc"])) <= {0.0, 1.0} and abs(d["Tc"].mean() - 0.5) < 0.05


def test_cohort_batches_are_identifiable_and_agnostic_bias_is_rejected(atlas, pool):
    prior = NeuroPriorCohort(atlas, pool, seed=5, n_context=96, n_query=8)
    batch = prior.sample_batch(4, n_context=96)
    assert batch["Xc"].shape == (4, 96, prior.d_x) and batch["mu_q"].shape == (4, 8)
    assert ((batch["Yc"] == 0) | (batch["Yc"] == 1)).all()
    assert len(batch["processes"]) == 4 and all("bias_type" in p for p in batch["processes"])
    # a susceptibility-driven allocation of full strength carries information
    # about an unobserved cause of the outcome: the verifier must reject it
    rng = np.random.default_rng(7)
    dgp = NeuroPriorDGP(atlas, rng, networks=atlas.networks[:3], optimal_for_A=[0, 1, 0],
                        bias=1.0, bias_type="agnostic", threshold=0.03, p_te=0.9, p_re=0.1, label_noise=0.0)
    idx = rng.integers(0, len(pool), size=600)
    d = make_prior_dataset(dgp, prior.overlaps[idx], prior.centroids[idx], prior.volumes[idx], prior.X[idx],
                           prior.overlaps[:4], prior.centroids[:4], prior.volumes[:4], prior.X[:4], rng)
    assert (d["u_ctx"] != 0).sum() > 50
    assert not verify_identifiability(d["Tc"], d["Xc"], None, None, U=d["u_ctx"][:, None])
    # observed mechanisms depend on geometry only and pass without U
    assert verify_identifiability(d["Tc"], d["Xc"], None, None, U=None)
    # the sampler therefore keeps (almost) only observed mechanisms; a
    # susceptibility-driven process survives only when its strength is near 0.5
    big = prior.sample_batch(12, n_context=96)
    for p in big["processes"]:
        assert p["bias_type"] != "agnostic" or abs(p["bias"] - 0.5) < 0.15, p


def test_hyper_prior_can_be_narrowed_for_a_curriculum_stage(atlas, pool):
    prior = NeuroPriorCohort(atlas, pool, seed=9, n_context=64, n_query=4,
                             hyper=HyperPrior(p_te=(0.9, 1.0), p_re=(0.0, 0.0), bias=(0.0, 0.0),
                                              bias_types=("axis",), bias_weights=(1.0,), label_noise=(0.0, 0.0)))
    b = prior.sample_batch(3, n_context=64)
    for p in b["processes"]:
        assert p["bias_type"] == "axis" and p["bias"] == 0.0 and p["p_re"] == 0.0 and p["p_te"] >= 0.9
