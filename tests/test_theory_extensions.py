"""The three extensions of the design's Neuro-Prior (intersynth_theory): a
second active treatment, a per-process effect scale and binary outcomes. Off
by default, they must leave the design's prior untouched; on, they must do
what their names say."""
import numpy as np
import pytest

from neurocausalpfn.prior.atlas import FunctionalAtlas
from neurocausalpfn.prior.intersynth_theory import InterSynthTheoryDGP, TheoryHyperPrior, make_theory_dataset
from neurocausalpfn.train.train_pfn import parse_hyper_opts


def _pool(n=300, seed=0):
    atlas = FunctionalAtlas.synthetic(shape=(16, 18, 16), seed=seed)
    rng = np.random.default_rng(seed)
    # overlap fractions [n, K, 2] with a long tail, as real lesions give
    ov = np.clip(rng.exponential(0.05, size=(n, atlas.n_networks, 2)), 0, 1).astype(np.float64)
    cent = rng.uniform(0, 16, size=(n, 3))
    X = rng.normal(size=(n, 8))
    return atlas, ov, cent, X


def _dgp(atlas, ov, hyper, seed=1):
    return InterSynthTheoryDGP(atlas, np.random.default_rng(seed), hyper).calibrate(ov)


def _dataset(dgp, ov, cent, X, seed=2):
    rng = np.random.default_rng(seed)
    return make_theory_dataset(dgp, ov[:200], cent[:200], X[:200], ov[200:], cent[200:], X[200:], rng)


def test_defaults_reproduce_the_design_prior():
    atlas, ov, cent, X = _pool()
    a = _dgp(atlas, ov, TheoryHyperPrior())
    b = _dgp(atlas, ov, TheoryHyperPrior(two_arms=0.0, effect_scale=None, binary_outcome=0.0))
    assert not a.two_arms and not a.binary and a.effect_factor == 1.0
    da, db = _dataset(a, ov, cent, X), _dataset(b, ov, cent, X)
    assert np.allclose(da["mu1"], db["mu1"]) and np.allclose(da["Yc"], db["Yc"])
    D, S = a.scores(ov)
    assert np.allclose(a.cate(D, S), a.beta * S * D)            # the design's CATE, never negative
    assert (a.cate(D, S) >= 0).all()
    assert not set(np.unique(da["Yc"])) <= {0.0, 1.0}             # continuous outcomes


def test_two_arms_give_signed_effects_and_keep_the_targets_consistent():
    atlas, ov, cent, X = _pool()
    dgp = _dgp(atlas, ov, TheoryHyperPrior(two_arms=1.0))
    assert dgp.two_arms and dgp.w_s2 is not None
    d = _dataset(dgp, ov, cent, X)
    tau = d["mu1"] - d["mu0"]
    assert (tau < 0).any() and (tau > 0).any()                    # both signs occur
    assert np.allclose(d["mu_q"], np.where(d["Tq"] == 1, d["mu1"], d["mu0"]))
    assert (d["mu0"] >= 0).all() and (d["mu1"] <= 1).all()
    D, S = dgp.scores(ov[200:]); Sb = dgp.scores_b(ov[200:])
    assert np.allclose(dgp.cate(D, S, Sb), tau)


def test_effect_scale_maps_the_pool_quantile_to_the_drawn_target():
    atlas, ov, cent, X = _pool()
    dgp = _dgp(atlas, ov, TheoryHyperPrior(effect_scale=(0.5, 0.5)))
    assert dgp.effect_target == pytest.approx(0.5) and dgp.effect_factor != 1.0
    D, S = dgp.scores(ov)
    raw = np.abs(dgp.effect_factor * dgp.beta * D * S)              # before clipping to [0, 1]
    assert np.quantile(raw, dgp.hyper.quantile) == pytest.approx(0.5, rel=1e-6)
    plain = _dgp(atlas, ov, TheoryHyperPrior())
    assert np.quantile(np.abs(plain.cate(*plain.scores(ov))), 0.95) < 0.5


def test_binary_outcomes_are_bernoulli_draws_of_mu():
    atlas, ov, cent, X = _pool()
    dgp = _dgp(atlas, ov, TheoryHyperPrior(binary_outcome=1.0))
    assert dgp.binary
    d = _dataset(dgp, ov, cent, X)
    assert set(np.unique(d["Yc"])) <= {0.0, 1.0}
    assert not set(np.unique(d["mu_q"])) <= {0.0, 1.0}             # the targets stay expected outcomes


def test_parse_hyper_opts_and_hyperprior_fields():
    opts = parse_hyper_opts("two_arms=1 effect_scale=0.2:1 binary_outcome=0.5")
    assert opts == {"two_arms": 1.0, "effect_scale": [0.2, 1.0], "binary_outcome": 0.5}
    h = TheoryHyperPrior(**{k: tuple(v) if isinstance(v, list) else v for k, v in opts.items()})
    assert h.two_arms == 1.0 and h.effect_scale == (0.2, 1.0) and h.binary_outcome == 0.5
    assert parse_hyper_opts("") == {}
    with pytest.raises(ValueError):
        parse_hyper_opts("two_arms")


def test_effect_shapes_mix_discrete_and_null_processes():
    atlas, ov, cent, X = _pool()
    hyper = TheoryHyperPrior(two_arms=1.0, effect_scale=(0.2, 1.0), effect_headroom=True,
                             shape_weights=(0.0, 1.0, 0.0), disc_frac=(0.3, 0.3), disc_effect_scale=(0.6, 0.6))
    dgp = _dgp(atlas, ov, hyper)
    assert dgp.shape == "discrete" and dgp.thr_s is not None and dgp.thr_b is not None
    D, S = dgp.scores(ov)
    S_b = dgp.scores_b(ov)
    assert set(np.unique(S)) <= {0.0, 1.0} and set(np.unique(S_b)) <= {0.0, 1.0}
    assert abs(S.mean() - 0.3) < 0.05 and abs(S_b.mean() - 0.3) < 0.05      # the drawn fraction is susceptible
    tau = dgp.cate(D, S, S_b)
    zero = np.isclose(tau, 0)
    assert 0.4 < zero.mean() < 0.75                                           # both or neither susceptible
    assert np.allclose(np.abs(tau[~zero]), dgp.effect_factor * dgp.beta * D[~zero])
    assert abs(np.quantile(np.abs(tau), 0.95) - 0.6) < 0.05 or dgp.effect_factor * dgp.beta <= 1 - dgp.alpha + 1e-9
    assert dgp.describe()["shape"] == "discrete"
    # null shape: no effect anywhere
    null = _dgp(atlas, ov, TheoryHyperPrior(two_arms=1.0, shape_weights=(0.0, 0.0, 1.0)), seed=3)
    assert null.shape == "null" and np.allclose(null.cate(*null.scores(ov), null.scores_b(ov)), 0)
    # default weights: the design's prior, same draws as before
    a = InterSynthTheoryDGP(atlas, np.random.default_rng(5), TheoryHyperPrior())
    b = InterSynthTheoryDGP(atlas, np.random.default_rng(5), TheoryHyperPrior(shape_weights=(1.0, 0.0, 0.0)))
    assert a.shape == "continuous" and a.beta == b.beta and np.allclose(a.w_s, b.w_s)
    # the hyper string of the jobs parses into the three weights
    opts = parse_hyper_opts("shape_weights=0.4:0.4:0.2 binary_outcome=1 disc_frac=0.2:0.6")
    assert opts["shape_weights"] == [0.4, 0.4, 0.2] and opts["disc_frac"] == [0.2, 0.6]
    mixed = TheoryHyperPrior(**{k: tuple(v) if isinstance(v, list) else v for k, v in opts.items()})
    shapes = {InterSynthTheoryDGP(atlas, np.random.default_rng(i), mixed).shape for i in range(40)}
    assert shapes == {"continuous", "discrete", "null"}
