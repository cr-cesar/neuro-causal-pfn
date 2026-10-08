"""The design's InterSynth Neuro-Prior (D, S, alpha, beta, gamma, four
mechanisms): R1 ignorability and positivity, R2 bounded outcomes and the
closed-form CATE, R3 the parameter ranges of the design; the cohort sampler
yields identifiable processes with it. CPU only, no torch."""
import numpy as np
import pytest

from neurocausalpfn.prior.atlas import FunctionalAtlas
from neurocausalpfn.prior.cohort import NeuroPriorCohort
from neurocausalpfn.prior.intersynth_atlas import compute_overlaps
from neurocausalpfn.prior.intersynth_theory import (MECHANISMS, InterSynthTheoryDGP, TheoryHyperPrior,
                                                    make_theory_dataset)
from neurocausalpfn.prior.verify_identifiability import verify_identifiability

from test_neuro_prior import SHAPE, _pool


@pytest.fixture(scope="module")
def atlas():
    return FunctionalAtlas.synthetic(shape=SHAPE, n_networks=6, seed=1)


@pytest.fixture(scope="module")
def pool(atlas):
    return _pool(atlas, 200, seed=1)


@pytest.fixture(scope="module")
def anatomy(atlas, pool):
    from neurocausalpfn.prior.atlas import _centroid

    ov = np.stack([compute_overlaps(atlas, m) for m in pool])
    cen = np.stack([_centroid(m) for m in pool])
    return ov, cen


def test_r3_default_ranges_are_the_design_ranges():
    h = TheoryHyperPrior()
    assert h.gamma == (0.0, 1.0) and h.beta == (0.05, 0.55) and h.alpha == (0.05, 0.45)
    assert tuple(h.mechanisms) == MECHANISMS == ("severity", "location", "network", "mixed")
    assert h.contains(0.2, 0.3, 0.5) and not h.contains(0.5, 0.3, 0.5) and not h.contains(0.2, 0.6, 0.5)


def test_sampled_processes_stay_inside_the_ranges(atlas):
    rng = np.random.default_rng(0)
    h = TheoryHyperPrior()
    seen = set()
    for _ in range(200):
        d = InterSynthTheoryDGP(atlas, rng, h)
        assert h.contains(d.alpha, d.beta, d.gamma)
        assert d.mechanism in MECHANISMS and 1 <= len(d.priority) <= 2
        assert np.isclose(d.w_d.sum(), 1.0) and np.isclose(d.w_s.sum(), 1.0)
        seen.add(d.mechanism)
    assert seen == set(MECHANISMS)


def test_r2_outcomes_bounded_and_cate_is_beta_s_d(atlas, anatomy):
    ov, cen = anatomy
    rng = np.random.default_rng(1)
    d = InterSynthTheoryDGP(atlas, rng, alpha=0.2, beta=0.4, gamma=0.5, mechanism="severity").calibrate(ov)
    D, S = d.scores(ov)
    assert D.min() >= 0 and D.max() <= 1 and S.min() >= 0 and S.max() <= 1
    assert np.isclose(np.quantile(d.raw_scores(ov)[0], 0.95) / d.scale_d, 1.0)   # calibration maps q95 -> 1
    mu0, mu1 = d.mu0(D), d.mu1(D, S)
    assert np.all((0 <= mu0) & (mu0 <= 1)) and np.all((0 <= mu1) & (mu1 <= 1))
    assert np.allclose(mu1 - mu0, 0.4 * S * D) and np.all(mu1 - mu0 >= 0) and np.all(mu1 - mu0 <= 0.4)
    # a fully disrupted brain keeps the residual alpha under control
    assert np.isclose(d.mu0(np.array([1.0]))[0], 0.2) and np.isclose(d.mu0(np.array([0.0]))[0], 1.0)


def test_r1_positivity_and_randomised_trial_at_gamma_zero(atlas, anatomy):
    ov, cen = anatomy
    rng = np.random.default_rng(2)
    for mech in MECHANISMS:
        d = InterSynthTheoryDGP(atlas, rng, gamma=1.0, mechanism=mech, noise_w=1.0).calibrate(ov)
        D, _ = d.scores(ov)
        e = d.propensity(D, ov, cen, rng)
        assert e.min() >= 0.01 and e.max() <= 0.99 and e.std() > 0.1      # confounded but positive
    d0 = InterSynthTheoryDGP(atlas, rng, gamma=0.0, mechanism="severity", noise_w=0.5).calibrate(ov)
    D, _ = d0.scores(ov)
    e0 = d0.propensity(D, ov, cen, rng)
    assert abs(float(np.corrcoef(e0, D)[0, 1])) < 0.2                       # gamma = 0: no confounding


def test_severity_mechanism_is_monotone_in_d(atlas, anatomy):
    ov, cen = anatomy
    rng = np.random.default_rng(3)
    d = InterSynthTheoryDGP(atlas, rng, gamma=1.0, mechanism="severity", sign=1, noise_w=0.5).calibrate(ov)
    D, _ = d.scores(ov)
    e = np.mean([d.propensity(D, ov, cen, rng) for _ in range(50)], axis=0)   # average out the noise
    assert np.corrcoef(e, D)[0, 1] > 0.8


def test_dataset_keys_and_observed_outcome_in_unit_interval(atlas, anatomy):
    ov, cen = anatomy
    rng = np.random.default_rng(4)
    d = InterSynthTheoryDGP(atlas, rng).calibrate(ov)
    X = rng.normal(size=(len(ov), 7))
    data = make_theory_dataset(d, ov[:150], cen[:150], X[:150], ov[150:], cen[150:], X[150:], rng)
    for k in ("Xc", "Tc", "Yc", "Xq", "Tq", "mu_q", "mu0", "mu1", "e_ctx"):
        assert k in data
    assert data["Xc"].shape == (150, 7) and data["mu0"].shape == (50,)
    assert data["Yc"].min() >= 0 and data["Yc"].max() <= 1 and set(np.unique(data["Tc"])) <= {0.0, 1.0}
    assert np.allclose(data["mu_q"], np.where(data["Tq"] == 1, data["mu1"], data["mu0"]))


def test_cohort_theory_family_is_identifiable_and_describes_processes(atlas, pool):
    coh = NeuroPriorCohort(atlas, pool, seed=0, n_context=96, n_query=8, family="theory", augment=False)
    assert isinstance(coh.hyper, TheoryHyperPrior)
    batch = coh.sample_batch(4)
    assert batch["Xc"].shape == (4, 96, coh.d_x) and batch["mu1"].shape == (4, 8)
    assert all(p["family"] == "theory" and p["mechanism"] in MECHANISMS for p in batch["processes"])
    for i in range(4):
        assert verify_identifiability(batch["Tc"][i], batch["Xc"][i], None, None, U=None)
    # the giles family is still available on the same cache
    coh_g = NeuroPriorCohort(atlas, pool, seed=0, n_context=64, n_query=8, family="giles", augment=False)
    assert "p_te" in coh_g.sample_batch(1)["processes"][0]
    with pytest.raises(ValueError):
        NeuroPriorCohort(atlas, pool, family="other")


def test_curriculum_stage_narrows_gamma_and_beta(atlas, pool):
    h = TheoryHyperPrior(gamma=(0.0, 0.3), beta=(0.2, 0.4))
    coh = NeuroPriorCohort(atlas, pool, seed=0, n_context=64, n_query=8, hyper=h, augment=False)
    for p in coh.sample_batch(6)["processes"]:
        assert 0.0 <= p["gamma"] <= 0.3 and 0.2 <= p["beta"] <= 0.4


def test_batch_metadata_is_not_a_tensor_field():
    """The cohort batch carries ``processes`` (one dict per item); the tensor
    conversion must skip it. Torch-free check of the predicate."""
    import importlib.util
    import sys, types

    if importlib.util.find_spec("torch") is None:
        # import the predicate without torch: load the module with a stub
        stub = types.ModuleType("torch"); stub.float32 = None; stub.dtype = object; stub.Tensor = object
        sys.modules.setdefault("torch", stub)
    from neurocausalpfn.pfn.tokens import is_tensor_field

    assert not is_tensor_field("processes", [{"family": "theory"}])
    assert not is_tensor_field("anything", [{"a": 1}]) and not is_tensor_field("x", "str")
    assert is_tensor_field("Xc", np.zeros((2, 3))) and is_tensor_field("Tc", [0.0, 1.0])


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="needs torch")
def test_pfn_trains_on_neuro_prior_cache(tmp_path):
    """End to end as on the cluster: anatomy cache -> neuro_prior kind (family
    theory) -> a few iterations of the transformer; the batch metadata must
    not break the tensor conversion."""
    import math

    from neurocausalpfn.prior.atlas import _centroid
    from neurocausalpfn.train.train_pfn import prototype_config, run_pfn

    # the training config builds its own synthetic atlas (atlas_dir None,
    # cfg seed), so the cache must come from that same atlas
    atlas_t = FunctionalAtlas.from_dir(None, shape=SHAPE, seed=0)
    pool = _pool(atlas_t, 60, seed=3)
    ov = np.stack([compute_overlaps(atlas_t, m) for m in pool])
    cache = tmp_path / "cache.npz"
    np.savez(cache, overlaps=ov, centroids=np.stack([_centroid(m) for m in pool]),
             volumes=np.array([float(m.sum()) for m in pool]),
             Z=np.random.default_rng(0).normal(size=(len(pool), 6)).astype(np.float32),
             files=np.array([f"l{i}" for i in range(len(pool))]))
    cfg = prototype_config()
    cfg["prior"] = {"kind": "neuro_prior", "family": "theory", "cache": str(cache),
                    "atlas_shape": list(SHAPE), "augment": True}
    cfg["pfn"].update({"iters": 3, "batch_size": 2, "context_min": 24, "context_max": 48, "n_query": 6})
    cfg["out_dir"] = str(tmp_path / "pfn")
    model, history = run_pfn(cfg)
    assert all(math.isfinite(h["loss"]) for h in history)


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="needs torch")
def test_e12_variants_and_curriculum():
    from neurocausalpfn.train.train_pfn import E12_VARIANTS, _context_length, _stage_index, e12_config

    assert E12_VARIANTS == ("nocurr", "ctx", "ctx+stages")
    c0, c1, c2 = (e12_config(v) for v in E12_VARIANTS)
    n = c1["pfn"]["iters"]
    assert _context_length(c0, 0) == c0["pfn"]["context_max"]                 # no curriculum: full context at once
    assert _context_length(c1, 0) == c1["pfn"]["context_min"] + (c1["pfn"]["context_max"] - c1["pfn"]["context_min"]) // (n // 2)
    assert _context_length(c1, n) == c1["pfn"]["context_max"]
    assert _stage_index(c1, 0) == -1 and _stage_index(c2, 0) == 0 and _stage_index(c2, n - 1) == 1
    assert c2["prior"]["stages"][0]["hyper"]["gamma"] == [0.0, 0.3]
    with pytest.raises(ValueError):
        e12_config("bogus")
