"""Neuro-Prior families for training the transformer: the mixture draws its
processes from the theory and the virtual-trial (giles) families with the
given weights, the curriculum stages narrow the ranges of the family in use,
and the probe separates a checkpoint that ignores the treatment from one that
has only learnt its own family. numpy-only tests first; torch-only last."""
import importlib.util
import os
import sys

import numpy as np
import pytest

from neurocausalpfn.prior.atlas import FunctionalAtlas
from neurocausalpfn.prior.cohort import FAMILIES, NeuroPriorCohort
from neurocausalpfn.prior.intersynth_theory import TheoryHyperPrior
from neurocausalpfn.prior.neuro_prior import HyperPrior

from test_neuro_prior import SHAPE, _pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.fixture(scope="module")
def atlas():
    return FunctionalAtlas.synthetic(shape=SHAPE, n_networks=6, seed=1)


@pytest.fixture(scope="module")
def pool(atlas):
    return _pool(atlas, 160, seed=4)


def test_mixture_draws_both_families_with_the_weights(atlas, pool):
    assert FAMILIES == ("theory", "giles", "mixture")
    coh = NeuroPriorCohort(atlas, pool, seed=0, n_context=48, n_query=8, family="mixture",
                           family_weights=(0.25, 0.75), augment=False)
    assert isinstance(coh.hyper, TheoryHyperPrior) and isinstance(coh.hyper_giles, HyperPrior)
    fams = [p["family"] for _ in range(20) for p in coh.sample_batch(4)["processes"]]
    share = fams.count("giles") / len(fams)
    assert set(fams) == {"theory", "giles"} and 0.6 < share < 0.9
    # giles processes carry signed effects and binary outcomes, theory ones non-negative effects
    b = coh.sample_batch(16)
    for i, p in enumerate(b["processes"]):
        cate = b["mu1"][i] - b["mu0"][i]
        if p["family"] == "theory":
            assert cate.min() >= -1e-12
        else:
            assert set(np.unique(b["Yc"][i])) <= {0.0, 1.0}


def test_single_families_and_weights_validation(atlas, pool):
    for fam in ("theory", "giles"):
        b = NeuroPriorCohort(atlas, pool, seed=1, n_context=32, n_query=4, family=fam,
                             augment=False).sample_batch(3)
        assert all(p["family"] == fam for p in b["processes"])
    with pytest.raises(ValueError):
        NeuroPriorCohort(atlas, pool, family="mixture", family_weights=(1.0,))
    with pytest.raises(ValueError):
        NeuroPriorCohort(atlas, pool, family="mixture", family_weights=(0.0, 0.0))


def test_giles_family_reads_the_swapped_hyper(atlas, pool):
    """The curriculum swaps ``hyper`` in place; the giles family must draw from
    the swapped ranges, not from a copy taken at construction."""
    coh = NeuroPriorCohort(atlas, pool, seed=2, n_context=32, n_query=4, family="giles", augment=False)
    coh.hyper = HyperPrior(bias=(0.0, 0.1))
    assert all(p["bias"] <= 0.1 for _ in range(5) for p in coh.sample_batch(4)["processes"])


def test_probe_metrics():
    from probe_pfn import cate_metrics

    true = np.array([1.0, 1.0, -1.0, -1.0, 0.0])
    perfect = cate_metrics(true, true)
    assert perfect["rmse"] == 0 and perfect["sign_balacc"] == 1.0 and perfect["corr"] == pytest.approx(1.0)
    blind = cate_metrics(np.zeros(5), true)                      # a model that ignores the treatment
    assert blind["mean_abs_pred"] == 0 and blind["sign_balacc"] == 0.0 and np.isnan(blind["corr"])
    positive = cate_metrics(np.full(5, 0.3), true)               # never predicts a negative effect
    assert positive["sign_sens"] == 1.0 and positive["sign_spec"] == 0.0 and positive["sign_balacc"] == 0.5


# ----------------------------------------------------------------- torch only
def _cache(tmp_path, n=60, d=6):
    from neurocausalpfn.prior.atlas import _centroid
    from neurocausalpfn.prior.intersynth_atlas import compute_overlaps

    atlas_t = FunctionalAtlas.from_dir(None, shape=SHAPE, seed=0)
    pool = _pool(atlas_t, n, seed=3)
    path = tmp_path / "cache.npz"
    np.savez(path, overlaps=np.stack([compute_overlaps(atlas_t, m) for m in pool]),
             centroids=np.stack([_centroid(m) for m in pool]),
             volumes=np.array([float(m.sum()) for m in pool]),
             Z=np.random.default_rng(0).normal(size=(n, d)).astype(np.float32),
             files=np.array([f"l{i}" for i in range(n)]))
    return str(path)


def _tiny_cfg(cache, out_dir, family, variant=None):
    from neurocausalpfn.train.train_pfn import e12_config, prototype_config, with_family

    cfg = prototype_config()
    if variant:
        cfg["prior"] = e12_config(variant, family)["prior"]
        cfg["pfn"]["curriculum"] = e12_config(variant, family)["pfn"]["curriculum"]
    cfg["prior"].update({"kind": "neuro_prior", "cache": cache, "atlas_dir": None,
                         "atlas_shape": list(SHAPE), "augment": True})
    with_family(cfg, family)
    cfg["pfn"].update({"iters": 4, "batch_size": 2, "context_min": 24, "context_max": 48, "n_query": 6})
    cfg["out_dir"] = str(out_dir)
    return cfg


@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
def test_e12_config_per_family():
    from neurocausalpfn.train.train_pfn import e12_config, with_family, reduced_config

    t, g, m = (e12_config("ctx+stages", f) for f in ("theory", "giles", "mixture"))
    assert t["prior"]["stages"][0]["hyper"] == {"gamma": [0.0, 0.3], "beta": [0.2, 0.4]}
    assert g["prior"]["family"] == "giles" and g["prior"]["stages"][0]["hyper"] == {"bias": [0.0, 0.3]}
    assert m["prior"]["stages"][0]["hyper_giles"] == {"bias": [0.0, 0.3]} and m["prior"]["family_weights"] == [0.5, 0.5]
    m["prior"]["stages"][0]["hyper"]["gamma"][1] = 9.0                # configs do not share the stage ranges
    assert e12_config("ctx+stages", "mixture")["prior"]["stages"][0]["hyper"]["gamma"] == [0.0, 0.3]
    assert e12_config("ctx", "mixture", [0.2, 0.8])["prior"]["family_weights"] == [0.2, 0.8]
    with pytest.raises(ValueError):
        with_family(reduced_config(), "other")


@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
@pytest.mark.parametrize("family", ["giles", "mixture"])
def test_staged_training_runs_for_each_family(tmp_path, family):
    """ctx+stages swaps the ranges at half the iterations; with the giles
    family the theory keys (gamma, beta) would not exist."""
    import math

    from neurocausalpfn.train.train_pfn import run_pfn

    cfg = _tiny_cfg(_cache(tmp_path), tmp_path / "pfn", family, variant="ctx+stages")
    model, history = run_pfn(cfg)
    assert len(history) == 4 and all(math.isfinite(h["loss"]) for h in history)


@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
def test_probe_runs_on_a_checkpoint(tmp_path):
    from neurocausalpfn.train.train_pfn import run_pfn
    from probe_pfn import main

    cfg = _tiny_cfg(_cache(tmp_path), tmp_path / "pfn", "theory")
    run_pfn(cfg)
    df = main(["--ckpt", str(tmp_path / "pfn" / "pfn.pt"), "--n-context", "30", "--n-query", "10",
               "--batches", "2", "--out", str(tmp_path / "probe.csv")])
    assert list(df["probe"]) == ["treatment:Y=T", "treatment:Y=1-T", "family:theory", "family:giles"]
    assert df["n"].tolist() == [20, 20, 20, 20] and (tmp_path / "probe.csv").exists()
