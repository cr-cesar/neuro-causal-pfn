"""Audit of 28 Oct (H4): the theory family's effect target can fix another
quantile of |CATE|, over the lesions of the process's slice; the default keeps
the design's calibration."""
import numpy as np
import pytest

from neurocausalpfn.prior.atlas import FunctionalAtlas
from neurocausalpfn.prior.intersynth_theory import InterSynthTheoryDGP, TheoryHyperPrior


def _pool(n=400, seed=0):
    atlas = FunctionalAtlas.synthetic(shape=(16, 18, 16), seed=seed)
    rng = np.random.default_rng(seed)
    ov = np.clip(rng.exponential(0.05, size=(n, atlas.n_networks, 2)), 0, 1).astype(np.float64)
    return atlas, ov


def _dgp(atlas, ov, hyper, seed=1):
    return InterSynthTheoryDGP(atlas, np.random.default_rng(seed), hyper).calibrate(ov)


def _slice_effects(dgp, ov):
    # the calibrated effect before the [0, 1] clip of the arms
    D, S = dgp.scores(ov)
    cate = np.abs(dgp.effect_factor * dgp.beta * D * (S - dgp.scores_b(ov)))
    hit = (dgp.raw_scores(ov)[0] > 0) & (cate > 0)
    return cate[hit]


def test_default_reproduces_the_design_calibration():
    h = TheoryHyperPrior()
    assert h.effect_quantile is None and h.effect_on_slice is False
    atlas, ov = _pool()
    base = dict(two_arms=1.0, effect_scale=(0.5, 0.5))
    a = _dgp(atlas, ov, TheoryHyperPrior(**base), seed=4)
    b = _dgp(atlas, ov, TheoryHyperPrior(**base, effect_quantile=None, effect_on_slice=False), seed=4)
    assert a.effect_factor == b.effect_factor


def test_slice_median_maps_to_the_target():
    atlas, ov = _pool()
    for seed in range(3, 9):
        h = TheoryHyperPrior(two_arms=1.0, effect_scale=(0.3, 0.3), effect_quantile=0.5, effect_on_slice=True)
        dgp = _dgp(atlas, ov, h, seed=seed)
        eff = _slice_effects(dgp, ov)
        assert np.median(eff) == pytest.approx(0.3, rel=0.05)
        assert dgp.describe()["effect_on_slice"] is True and dgp.describe()["effect_quantile"] == 0.5


def test_slice_median_calibration_gives_larger_typical_effects_than_the_pool_q95():
    atlas, ov = _pool()
    meds = {}
    for name, kw in (("pool", {}), ("slice", dict(effect_quantile=0.5, effect_on_slice=True))):
        h = TheoryHyperPrior(two_arms=1.0, effect_scale=(0.5, 0.5), **kw)
        meds[name] = np.median([np.median(_slice_effects(_dgp(atlas, ov, h, seed=s), ov)) for s in range(3, 9)])
    assert meds["slice"] > 2 * meds["pool"]


def test_options_from_the_prior_hyper_string():
    from neurocausalpfn.train.train_pfn import _as_ranges, parse_hyper_opts

    kw = parse_hyper_opts("two_arms=1 effect_scale=0.2:1 effect_quantile=0.5 effect_on_slice=1")
    h = TheoryHyperPrior(**_as_ranges(kw))
    assert h.effect_quantile == 0.5 and bool(h.effect_on_slice)
