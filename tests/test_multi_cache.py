"""Multi-encoder pools: several caches behind one prior, the anchor's share,
the hyper-priors a curriculum stage sets, and the re-latented caches."""
import importlib.util
import math
import os
import sys

import numpy as np
import pytest

from neurocausalpfn.prior.multi_cache import MultiCachePrior, cache_weights

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
HAS_TORCH = importlib.util.find_spec("torch") is not None


class _Fake:
    def __init__(self, tag, d_x=4):
        self.tag, self.d_x, self.family = tag, d_x, "mixture"
        self.hyper = self.hyper_giles = None
        self.calls = []

    def sample_batch(self, batch_size, n_context=None):
        self.calls.append((batch_size, n_context))
        return {"tag": self.tag}


def test_cache_weights_anchor_explicit_and_uniform():
    np.testing.assert_allclose(cache_weights(4, anchor_weight=0.5), [0.5, 1 / 6, 1 / 6, 1 / 6])
    np.testing.assert_allclose(cache_weights(3), [1 / 3] * 3)
    np.testing.assert_allclose(cache_weights(2, weights=[3, 1]), [0.75, 0.25])
    np.testing.assert_allclose(cache_weights(1, anchor_weight=0.3), [1.0])
    for bad in (dict(weights=[1]), dict(anchor_weight=1.5), dict(weights=[0, 0])):
        with pytest.raises(ValueError):
            cache_weights(2, **bad)


def test_one_cohort_per_batch_with_the_anchor_share():
    fakes = [_Fake(i) for i in range(5)]
    mp = MultiCachePrior(fakes, cache_weights(5, anchor_weight=0.6), seed=0)
    tags = [mp.sample_batch(2, n_context=32)["tag"] for _ in range(4000)]
    share = np.bincount(tags, minlength=5) / len(tags)
    assert abs(share[0] - 0.6) < 0.03 and np.all(np.abs(share[1:] - 0.1) < 0.02)
    assert all(c == (2, 32) for f in fakes for c in f.calls)
    assert mp.counts.sum() == 4000 and mp.d_x == 4 and mp.family == "mixture"
    mp.sample_batch(3)
    assert any(c == (3, None) for f in fakes for c in f.calls)


def test_stage_hyper_priors_reach_every_cohort():
    fakes = [_Fake(i) for i in range(3)]
    mp = MultiCachePrior(fakes)
    mp.hyper, mp.hyper_giles = "h", "hg"
    assert all(f.hyper == "h" and f.hyper_giles == "hg" for f in fakes)
    with pytest.raises(ValueError):
        MultiCachePrior([_Fake(0, d_x=4), _Fake(1, d_x=5)])


def test_generic_wrapper_forwards_the_virtual_trial_ranges():
    from neurocausalpfn.prior.generic_mix import GenericEffectMixture

    base = _Fake(0)
    g = GenericEffectMixture(base, 0.5)
    g.hyper_giles = "stage"
    assert base.hyper_giles == "stage" and g.hyper_giles == "stage"


def test_relatent_aligns_rows_by_name_and_refuses_mismatches(tmp_path):
    from relatent_cache import main, relatent

    files = np.array([f"img{i}.nii.gz" for i in range(5)])
    cache = {"overlaps": np.arange(10.0).reshape(5, 2), "files": files, "Z": np.zeros((5, 3), np.float32)}
    order = np.array([3, 0, 4, 1, 2])
    lat = {"Z": np.arange(15, dtype=np.float32).reshape(5, 3)[order], "files": files[order]}
    out = relatent(cache, lat, "x")
    np.testing.assert_array_equal(out["Z"], np.arange(15, dtype=np.float32).reshape(5, 3))
    np.testing.assert_array_equal(out["overlaps"], cache["overlaps"])
    for bad in ({"Z": lat["Z"][:4], "files": lat["files"][:4]},
                {"Z": np.full((5, 3), np.nan, np.float32), "files": files},
                {"Z": np.zeros((5, 2), np.float32), "files": files}):
        with pytest.raises(ValueError):
            relatent(cache, bad, "x")
    np.savez(tmp_path / "base.npz", **cache)
    for s in (0, 1):
        d = tmp_path / "lat" / f"seed{s}" / "fold2"
        d.mkdir(parents=True)
        np.savez(d / "E1_disco.npz", **lat)
    main(["--cache", str(tmp_path / "base.npz"), "--latents", str(tmp_path / "lat" / "seed*" / "fold*" / "*.npz"),
          "--out-dir", str(tmp_path / "out")])
    assert sorted(os.listdir(tmp_path / "out")) == ["base__seed0_fold2.npz", "base__seed1_fold2.npz"]


@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
def test_training_on_a_two_cache_pool_runs_and_keeps_one_cache_unchanged(tmp_path):
    from relatent_cache import relatent
    from test_prior_families import _cache, _tiny_cfg

    from neurocausalpfn.prior.cohort import NeuroPriorCohort
    from neurocausalpfn.train.train_pfn import _build_prior, run_pfn

    c1 = _cache(tmp_path)
    with np.load(c1, allow_pickle=True) as z:
        base = {k: z[k] for k in z.files}
    other = relatent(base, {"Z": np.random.default_rng(5).normal(size=base["Z"].shape).astype(np.float32),
                            "files": base["files"]}, "other encoder")
    c2 = str(tmp_path / "cache_other.npz")
    np.savez(c2, **other)

    single = _tiny_cfg(c1, tmp_path / "one", "mixture")
    assert isinstance(_build_prior(single)[0], NeuroPriorCohort)          # one cache: unchanged path

    cfg = _tiny_cfg([c1, c2], tmp_path / "pool", "mixture", variant="ctx+stages")
    cfg["prior"]["cache_anchor_weight"] = 0.5
    prior, d_x, _ = _build_prior(cfg)
    assert isinstance(prior, MultiCachePrior) and len(prior.priors) == 2 and d_x == base["Z"].shape[1]
    np.testing.assert_allclose(prior.weights, [0.5, 0.5])
    model, history = run_pfn(cfg)
    assert len(history) == 4 and all(math.isfinite(h["loss"]) for h in history)
