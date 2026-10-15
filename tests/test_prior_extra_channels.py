"""Extra anatomy channels of the prior cache (15 Oct): whole-network overlap
and both atlas splits, the augment mode of the builder, and the theory family
reading D, S_A and S_B from named channels. The defaults must reproduce the
behaviour of every earlier run exactly."""
import os

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from neurocausalpfn.prior.atlas import FunctionalAtlas
from neurocausalpfn.prior.intersynth_theory import TheoryHyperPrior
from neurocausalpfn.train.train_pfn import parse_hyper_opts

SHAPE = (12, 8, 8)


def _atlas_dir(tmp_path):
    """Two networks; the receptor split cuts each along y, the genetics split
    along x, so the two splits are different cuts of the same networks."""
    d = tmp_path / "atlas"
    aff = np.eye(4)
    net = np.zeros(SHAPE, np.int16); net[:6] = 1; net[6:] = 2
    nib.save(nib.Nifti1Image(net, aff), str(d.mkdir() or d / "functional_parcellation_2mm.nii.gz"))
    for mod in ("receptor", "genetics"):
        (d / "2mm_parcellations" / mod).mkdir(parents=True)
    for k, xs in ((1, slice(0, 6)), (2, slice(6, 12))):
        rec = np.zeros(SHAPE, np.int16); gen = np.zeros(SHAPE, np.int16)
        rec[xs, :4] = 1; rec[xs, 4:] = 2                       # y cut
        lo = xs.start; gen[lo:lo + 2] = 1; gen[lo + 2:xs.stop] = 2   # x cut, unequal halves
        gen[net != k] = 0; rec[net != k] = 0
        nib.save(nib.Nifti1Image(rec, aff), str(d / "2mm_parcellations" / "receptor" / f"{k}_net.nii.gz"))
        nib.save(nib.Nifti1Image(gen, aff), str(d / "2mm_parcellations" / "genetics" / f"{k}_net.nii.gz"))
    return str(d)


def _images_dir(tmp_path, n=60, seed=0):
    d = tmp_path / "images"; d.mkdir()
    rng = np.random.default_rng(seed)
    for i in range(n):
        m = np.zeros(SHAPE, np.float32)
        c = rng.integers([1, 1, 1], [11, 7, 7]); r = rng.integers(1, 4)
        m[max(c[0] - r, 0):c[0] + r, max(c[1] - r, 0):c[1] + r, max(c[2] - r, 0):c[2] + r] = 1.0
        nib.save(nib.Nifti1Image(m, np.eye(4)), str(d / f"img{i:03d}.nii.gz"))
    return str(d)


def _build(atlas_dir, images_dir, out, *extra):
    from scripts.build_prior_cache import main
    main(["--images-dir", images_dir, "--atlas-dir", atlas_dir, "--out", out, *extra])
    with np.load(out, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def test_builder_adds_the_extra_channels_and_keeps_overlaps(tmp_path):
    a, im = _atlas_dir(tmp_path), _images_dir(tmp_path)
    c = _build(a, im, str(tmp_path / "c.npz"))
    n, K = len(c["files"]), 2
    assert c["overlaps"].shape == (n, K, 2)
    assert c["overlaps_network"].shape == (n, K)
    assert c["overlaps_receptor"].shape == c["overlaps_genetics"].shape == (n, K, 2)
    # the cache was built with the receptor split: its overlaps are that split
    np.testing.assert_allclose(c["overlaps"], c["overlaps_receptor"], atol=1e-6)
    # each split partitions the network, so the whole-network overlap is the
    # volume-weighted mean of its two halves, for either split
    atl = FunctionalAtlas.from_dir(a, modality="genetics")
    for i, k in enumerate(atl.networks):
        va, vb = (float(m.sum()) for m in atl.subnetworks[k])
        want = (c["overlaps_genetics"][:, i, 0] * va + c["overlaps_genetics"][:, i, 1] * vb) / (va + vb)
        np.testing.assert_allclose(c["overlaps_network"][:, i], want, atol=1e-5)
    # the two splits are different cuts
    assert not np.allclose(c["overlaps_receptor"], c["overlaps_genetics"])


def test_augment_adds_only_the_extra_channels(tmp_path):
    a, im = _atlas_dir(tmp_path), _images_dir(tmp_path)
    plain = _build(a, im, str(tmp_path / "plain.npz"), "--no-extra-channels")
    assert "overlaps_network" not in plain
    full = _build(a, im, str(tmp_path / "full.npz"))
    aug = _build(a, im, str(tmp_path / "aug.npz"), "--augment", str(tmp_path / "plain.npz"))
    for k in plain:                                   # nothing that was there changes
        np.testing.assert_array_equal(np.asarray(aug[k]), np.asarray(plain[k]))
    for k in ("overlaps_network", "overlaps_receptor", "overlaps_genetics"):
        np.testing.assert_allclose(aug[k], full[k], atol=1e-6)


def _cohort(cache_path, atlas_dir, hyper, seed=0):
    from neurocausalpfn.prior.cohort import NeuroPriorCohort

    atlas = FunctionalAtlas.from_dir(atlas_dir, modality="receptor")
    return NeuroPriorCohort.from_cache(atlas, cache_path, seed=seed, n_context=20, n_query=4,
                                       hyper=hyper, family="theory", min_slice=8)


def test_default_sources_reproduce_the_old_prior_exactly(tmp_path):
    a, im = _atlas_dir(tmp_path), _images_dir(tmp_path)
    _build(a, im, str(tmp_path / "plain.npz"), "--no-extra-channels")
    _build(a, im, str(tmp_path / "full.npz"))
    h = TheoryHyperPrior(two_arms=1.0, effect_scale=(0.2, 1.0))
    old, new = _cohort(str(tmp_path / "plain.npz"), a, h), _cohort(str(tmp_path / "full.npz"), a, h)
    assert new.overlaps_theory is None
    b0, b1 = old.sample_batch(3), new.sample_batch(3)
    for k in ("Xc", "Tc", "Yc", "mu0", "mu1"):
        np.testing.assert_array_equal(b0[k], b1[k])


def test_named_sources_build_the_view_and_sample(tmp_path):
    a, im = _atlas_dir(tmp_path), _images_dir(tmp_path)
    _build(a, im, str(tmp_path / "full.npz"))
    h = TheoryHyperPrior(two_arms=1.0, effect_scale=(0.2, 1.0),
                         d_source="network", s_source="genetics_1", s_b_source="genetics_2")
    coh = _cohort(str(tmp_path / "full.npz"), a, h)
    v = coh.overlaps_theory
    assert v.shape == (coh.m, 2, 3)
    with np.load(str(tmp_path / "full.npz")) as z:
        np.testing.assert_allclose(v[:, :, 0], z["overlaps_network"], atol=1e-6)
        np.testing.assert_allclose(v[:, :, 1], z["overlaps_genetics"][:, :, 0], atol=1e-6)
        np.testing.assert_allclose(v[:, :, 2], z["overlaps_genetics"][:, :, 1], atol=1e-6)
    b = coh.sample_batch(2)
    assert np.isfinite(b["mu1"]).all() and b["Xc"].shape[1] == 20
    assert b["processes"][0]["d_source"] == "network"


def test_named_source_without_extra_channels_fails_clearly(tmp_path):
    a, im = _atlas_dir(tmp_path), _images_dir(tmp_path)
    _build(a, im, str(tmp_path / "plain.npz"), "--no-extra-channels")
    with pytest.raises(ValueError, match="--augment"):
        _cohort(str(tmp_path / "plain.npz"), a, TheoryHyperPrior(d_source="network"))


def test_prior_hyper_string_accepts_channel_names():
    h = parse_hyper_opts("two_arms=1 effect_scale=0.2:1 d_source=network s_source=genetics_1")
    assert h == {"two_arms": 1.0, "effect_scale": [0.2, 1.0], "d_source": "network", "s_source": "genetics_1"}
