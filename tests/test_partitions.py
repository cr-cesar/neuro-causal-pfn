"""Route C: virtual networks from random unions of the cache's anatomical
atoms, never the benchmark's own halves, driving the virtual-trial family."""
import numpy as np
import pytest

from neurocausalpfn.prior.atlas import FunctionalAtlas
from neurocausalpfn.prior.cohort import NeuroPriorCohort
from neurocausalpfn.prior.neuro_prior import HyperPrior
from neurocausalpfn.prior.partitions import (VirtualAtlas, atom_table, draw_virtual_networks,
                                             is_benchmark_pair)

from test_neuro_prior import SHAPE, _pool


def test_atoms_and_virtual_networks_avoid_benchmark_pairs():
    rng = np.random.default_rng(0)
    m, K = 50, 4
    primary = rng.uniform(size=(m, K, 2)).astype(np.float32)
    extra = {"overlaps_network": rng.uniform(size=(m, K)), "overlaps_genetics": rng.uniform(size=(m, K, 2))}
    labels, table = atom_table(primary, extra)
    assert table.shape == (m, 2 * K + K + 2 * K) and len(labels) == table.shape[1]
    assert ("primary", 0, 0) in labels and ("network", 2, -1) in labels and ("genetics", 3, 1) in labels
    assert is_benchmark_pair([0], [1], labels) and not is_benchmark_pair([0], [2], labels)
    assert not is_benchmark_pair([0, 2], [1], labels)                       # a union is not the benchmark pair
    vov, desc = draw_virtual_networks(rng, labels, table, n_networks=3, atoms_per_side=(1, 2))
    assert vov.shape == (m, 3, 2) and len(desc) == 3
    for d in desc:
        a, b = d["A"], d["B"]
        assert 1 <= len(a) <= 2 and 1 <= len(b) <= 2 and not set(a) & set(b)
        assert not (len(a) == 1 and len(b) == 1 and a[0][:2] == b[0][:2] and {a[0][2], b[0][2]} == {0, 1})
    # a side's overlap is the largest among its atoms
    ia = [labels.index(x) for x in desc[0]["A"]]
    np.testing.assert_allclose(vov[:, 0, 0], table[:, ia].max(axis=1))
    va = VirtualAtlas(vov, rng.normal(size=(m, 3)))
    assert va.networks == [0, 1, 2] and va.centroid(0, 1).shape == (3,)
    # all atoms identical to one benchmark pair: drawing must fail rather than leak it
    labels2 = [("primary", 0, 0), ("primary", 0, 1)]
    with pytest.raises(RuntimeError):
        draw_virtual_networks(rng, labels2, table[:, :2], 1, atoms_per_side=(1, 1))


def test_cohort_draws_random_partition_processes():
    atlas = FunctionalAtlas.synthetic(shape=SHAPE, n_networks=6, seed=1)
    pool = _pool(atlas, 48, seed=0)
    h = HyperPrior(random_partitions=1.0, atoms_per_side=(1, 2), threshold=(0.01, 0.2))
    coh = NeuroPriorCohort(atlas, pool, seed=3, n_context=32, n_query=4, family="giles", hyper=h,
                           context_pool="all", augment=False)
    b = coh.sample_batch(6)
    procs = b["processes"]
    assert all(p["partitions"] == "random" for p in procs)
    assert b["Yc"].shape == (6, 32) and set(np.unique(b["Yc"])) <= {0.0, 1.0}
    tau = b["mu1"] - b["mu0"]
    assert np.isfinite(tau).all() and (np.abs(tau) > 0).any()                  # all-or-none effects appear
    # the mixture reaches the option through hyper_giles
    mix = NeuroPriorCohort(atlas, pool, seed=4, n_context=32, n_query=4, family="mixture",
                           family_weights=(0.0, 1.0), hyper_giles=h, context_pool="all", augment=False)
    assert all(p["partitions"] == "random" for p in mix.sample_batch(3)["processes"])
    # off by default
    plain = NeuroPriorCohort(atlas, pool, seed=5, n_context=32, n_query=4, family="giles",
                             context_pool="all", augment=False)
    assert all(p["partitions"] == "atlas" for p in plain.sample_batch(3)["processes"])
