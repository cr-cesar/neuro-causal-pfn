"""Tests of the InterSynth mechanism with an anatomical substrate."""
import math

import numpy as np

from neurocausalpfn.prior.atlas import FunctionalAtlas
from neurocausalpfn.prior.cohort import NeuroPriorInterSynth, build_synthetic_lesion_pool
from neurocausalpfn.prior.intersynth_atlas import InterSynthDGP, compute_overlaps
from neurocausalpfn.prior.verify_identifiability import verify_identifiability


def test_atlas_synthetic_shapes():
    atlas = FunctionalAtlas.synthetic(shape=(32, 36, 32), n_networks=6, seed=0)
    assert atlas.n_networks == 6
    rng = np.random.default_rng(0)
    lesion = (rng.uniform(size=(32, 36, 32)) > 0.7).astype(np.float32)
    ov = compute_overlaps(atlas, lesion)
    assert ov.shape == (6, 2)
    assert (ov >= 0).all() and (ov <= 1).all()


def test_susceptibility_and_threshold():
    atlas = FunctionalAtlas.synthetic(shape=(32, 36, 32), seed=1)
    dgp = InterSynthDGP(atlas, np.random.default_rng(1))
    assert dgp.susceptibility(0.5, 0.0) == 0          # mostly affects subnetwork A
    assert dgp.susceptibility(0.0, 0.5) == 1          # mostly affects subnetwork B
    assert dgp.susceptibility(0.01, 0.0) is None      # below the 5% threshold


def test_mu_ordering_and_cate_sign():
    atlas = FunctionalAtlas.synthetic(shape=(32, 36, 32), seed=2)
    dgp = InterSynthDGP(atlas, np.random.default_rng(2))
    s = 0
    t_star = dgp.optimal_treatment(s)
    assert dgp.mu(s, t_star) > dgp.mu(s, 1 - t_star)  # the appropriate treatment improves the outcome
    assert dgp.mu(None, 0) == dgp.mu(None, 1)          # without susceptibility, the treatment changes nothing
    assert abs(dgp.cate(None)) < 1e-9                  # CATE zero if not susceptible


def test_intersynth_cohort_batch_and_ignorability():
    atlas = FunctionalAtlas.synthetic(shape=(32, 36, 32), seed=3)
    pool = build_synthetic_lesion_pool(48, shape=(32, 36, 32), seed=3)
    prior = NeuroPriorInterSynth(atlas, pool, seed=3, n_context=64, n_query=8)
    batch = prior.sample_batch(3, n_context=48)
    assert batch["Xc"].shape == (3, 48, prior.d_x)
    assert batch["mu_q"].shape == (3, 8)
    assert ((batch["Yc"] == 0) | (batch["Yc"] == 1)).all()   # binary outcome
    # the default prior is ignorable by construction
    assert verify_identifiability(batch["Tc"][0], batch["Xc"][0], None, None, U=None)


def test_pfn_trains_on_intersynth():
    from neurocausalpfn.train.train_pfn import prototype_config, run_pfn

    cfg = prototype_config()
    cfg["prior"] = {"kind": "intersynth", "atlas_shape": [24, 28, 24], "pool_size": 40}
    cfg["pfn"]["iters"] = 8
    cfg["pfn"]["batch_size"] = 3
    cfg["pfn"]["context_max"] = 48
    cfg["pfn"]["n_query"] = 8
    cfg["out_dir"] = "outputs/test_intersynth_pfn"
    model, history = run_pfn(cfg)
    assert all(math.isfinite(h["loss"]) for h in history)
