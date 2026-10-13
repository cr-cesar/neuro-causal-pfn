"""The channel guard of the prior cache: scripts/build_prior_cache.py detects
whether a pool is binary lesion masks or continuous maps, refuses latents of
the other channel, records the channel in the cache, and its overlaps agree
with the labels of the virtual-trial replica on the same images. CPU, no
torch: the atlas is the synthetic one, written to disk so the script loads it
through the real path."""
import importlib.util
import logging
import os
import sys

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pd = pytest.importorskip("pandas")
ndimage = pytest.importorskip("scipy.ndimage")

from neurocausalpfn.prior import channel as ch                     # noqa: E402
from neurocausalpfn.prior.atlas import FunctionalAtlas             # noqa: E402
from neurocausalpfn.prior.cohort import NeuroPriorCohort           # noqa: E402
from neurocausalpfn.prior.giles_replica import (DEFICIT_NAMES, RoiPair,   # noqa: E402
                                                label_images)

_spec = importlib.util.spec_from_file_location(
    "build_prior_cache", os.path.join(os.path.dirname(__file__), "..", "scripts", "build_prior_cache.py"))
bpc = importlib.util.module_from_spec(_spec)
sys.modules["build_prior_cache"] = bpc
_spec.loader.exec_module(bpc)

SHAPE = (24, 28, 24)
N_NETWORKS = 6
N_LESIONS = 80


def _masks(atlas, n, seed=0):
    """Ellipsoidal binary lesions centred on random atlas voxels."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.indices(atlas.shape)
    fg = np.argwhere(atlas.network_labels > 0)
    out = np.zeros((n,) + atlas.shape, dtype=np.float32)
    for i in range(n):
        c = fg[rng.integers(0, len(fg))] if rng.uniform() < 0.8 else rng.integers(0, atlas.shape)
        r = rng.integers(2, 6, size=3)
        out[i] = ((((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2
                   + ((xx - c[2]) / r[2]) ** 2) <= 1).astype(np.float32)
    return out


def _continuous(mask):
    """A map in [0, 1] decaying with the distance to the mask: 1 inside,
    1 - d/4 outside, so its support above 0.5 is the mask dilated by two
    voxels and it holds fractional values on every boundary."""
    d = ndimage.distance_transform_edt(mask == 0)
    return np.clip(1.0 - d / 4.0, 0.0, 1.0).astype(np.float32)


def _write(dir_, vols):
    os.makedirs(dir_, exist_ok=True)
    files = []
    for i, v in enumerate(vols):
        p = os.path.join(dir_, f"lesion{i:04d}.nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(v, dtype=np.float32), np.eye(4)), p)
        files.append(p)
    return files


def _write_atlas(dir_, atlas):
    """The synthetic atlas in the layout FunctionalAtlas._load expects."""
    os.makedirs(os.path.join(dir_, "2mm_parcellations", "receptor"), exist_ok=True)
    nib.save(nib.Nifti1Image(atlas.network_labels.astype(np.int16), np.eye(4)),
             os.path.join(dir_, "functional_parcellation_2mm.nii.gz"))
    for k, (a, b) in atlas.subnetworks.items():
        lab = (a > 0).astype(np.int16) * 1 + (b > 0).astype(np.int16) * 2
        nib.save(nib.Nifti1Image(lab, np.eye(4)),
                 os.path.join(dir_, "2mm_parcellations", "receptor", f"{k}_network.nii.gz"))
    return dir_


def _pairs(atlas):
    """RoiPair per network from the atlas subnetworks, as load_atlas_pairs
    builds them from the files: both sides then see identical masks."""
    pairs = {}
    for k in atlas.networks:
        a, b = atlas.subnetworks[k]
        masks = [(a > 0).astype(int), (b > 0).astype(int)]
        vols = [int(m.sum()) for m in masks]
        treatments = ["1", "0"] if int(np.argmax(vols)) == 0 else ["0", "1"]
        cx, cy, cz = [], [], []
        for m in masks:
            inds = np.where(m == 1)
            cx.append(float(inds[0].mean())); cy.append(float(inds[1].mean())); cz.append(float(inds[2].mean()))
        pairs[k] = RoiPair(masks, vols, treatments, (cx, cy, cz))
    return pairs


@pytest.fixture(scope="module")
def atlas():
    return FunctionalAtlas.synthetic(shape=SHAPE, n_networks=N_NETWORKS, seed=0)


@pytest.fixture(scope="module")
def data(tmp_path_factory, atlas):
    root = tmp_path_factory.mktemp("pool")
    masks = _masks(atlas, N_LESIONS)
    return {"atlas_dir": _write_atlas(str(root / "atlases"), atlas),
            "lesions": _write(str(root / "lesions"), masks),
            "discos": _write(str(root / "disconnectomes"), [_continuous(m) for m in masks]),
            "root": root}


def _build(data, images, out, latents=None, extra=()):
    argv = ["--images-dir", os.path.dirname(images[0]), "--atlas-dir", data["atlas_dir"], "--out", str(out)]
    if latents:
        argv += ["--latents", str(latents)]
    bpc.main(argv + list(extra))
    with np.load(out, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _latents(path, files, **meta):
    Z = np.random.default_rng(0).normal(size=(len(files), 5)).astype(np.float32)
    np.savez(path, Z=Z, files=np.array([os.path.basename(f) for f in files]),
             **{k: np.array(v) for k, v in meta.items()})
    return Z


# --------------------------------------------------------------------------- #
# the detector
# --------------------------------------------------------------------------- #
def test_detector_rule():
    assert ch.is_binary_volume(np.zeros((3, 3, 3))) and ch.is_binary_volume(np.array([0.0, 1.0, 1.0]))
    assert not ch.is_binary_volume(np.array([0.0, 0.5, 1.0])) and not ch.is_binary_volume(np.array([0, 2]))
    vols = {"a": np.ones(4), "b": np.zeros(4), "c": np.array([0.0, 0.3, 1.0, 1.0])}
    load = vols.__getitem__
    assert ch.detect_channel(["a", "b"], loader=load) == "lesion"
    assert ch.detect_channel(["a", "b", "c"], loader=load) == "disconnectome"
    assert ch.detect_channel(["a", "b", "c"], n_check=2, loader=load) == "lesion"   # only the first n_check are read
    assert ch.channel_from_basename("outputs/latents/E1_seed0_disco.npz") == "disconnectome"
    assert ch.channel_from_basename("E2_E2_w_dice=0.5_seed0_lesion.npz") == "lesion"
    assert ch.channel_from_basename("pool.npz") is None
    # precedence: recorded image channel, then the encoder representation, then the name
    assert ch.latents_channel({"images_channel": "lesion", "representation": "disconnectome"}, "x_disco.npz")[0] == "lesion"
    assert ch.latents_channel({"representation": np.array("disconnectome")}, "x_lesion.npz")[0] == "disconnectome"
    assert ch.latents_channel({"representation": "early_fusion"}, "x_disco.npz")[0] == "disconnectome"
    assert ch.latents_channel({"representation": "early_fusion"}, "pool.npz")[0] is None
    assert ch.latents_channel({}, "pool.npz")[0] is None


# --------------------------------------------------------------------------- #
# (a) binary masks -> lesion cache; (b) continuous maps -> disconnectome cache
# --------------------------------------------------------------------------- #
def test_binary_masks_give_a_lesion_cache_with_provenance(data, tmp_path, capsys):
    cache = _build(data, data["lesions"], tmp_path / "lesion.npz")
    assert str(cache["channel"]) == "lesion"
    assert str(cache["images_dir"]) == os.path.abspath(os.path.dirname(data["lesions"][0]))
    assert float(cache["threshold"]) == 0.5 and str(cache["modality"]) == "receptor"
    assert cache["overlaps"].shape == (N_LESIONS, N_NETWORKS, 2)
    out = capsys.readouterr().out
    assert "images: lesion" in out
    assert "hit at least one subnetwork above 5 %" in out and "per-network responsive fraction min/median/max" in out


def test_continuous_maps_give_a_denser_disconnectome_cache(data, tmp_path):
    lesion = _build(data, data["lesions"], tmp_path / "lesion.npz")
    disco = _build(data, data["discos"], tmp_path / "disco.npz")
    assert str(disco["channel"]) == "disconnectome"
    f_les, f_dis = ch.responsive_fractions(lesion["overlaps"]), ch.responsive_fractions(disco["overlaps"])
    assert f_les.shape == (N_NETWORKS,) and f_les.max() > 0
    # the binarised support of the maps contains the mask, so every network is
    # at least as responsive and the pool as a whole is denser
    assert np.all(f_dis >= f_les) and f_dis.mean() > f_les.mean()
    assert np.all(disco["volumes"] > lesion["volumes"])
    assert ch.summarize_overlaps(disco["overlaps"])["any_hit"] > ch.summarize_overlaps(lesion["overlaps"])["any_hit"]


def test_lesions_dir_alias_still_works(data, tmp_path, capsys):
    out = tmp_path / "alias.npz"
    bpc.main(["--lesions-dir", os.path.dirname(data["lesions"][0]), "--atlas-dir", data["atlas_dir"], "--out", str(out)])
    assert "former name of --images-dir" in capsys.readouterr().out
    with np.load(out) as z:
        assert str(z["channel"]) == "lesion"
    with pytest.raises(SystemExit):
        bpc.main(["--atlas-dir", data["atlas_dir"], "--out", str(out)])     # neither flag: refused


# --------------------------------------------------------------------------- #
# (c) the mismatch abort
# --------------------------------------------------------------------------- #
def test_mismatch_aborts_unless_allowed(data, tmp_path):
    lat = tmp_path / "pool_latents.npz"
    Z = _latents(lat, data["lesions"], representation="disconnectome")
    with pytest.raises(SystemExit, match=r"(?s)lesion.*disconnectome") as e:
        _build(data, data["lesions"], tmp_path / "bad.npz", latents=lat)
    assert "channel mismatch" in str(e.value) and "--allow-channel-mismatch" in str(e.value)
    assert not os.path.exists(tmp_path / "bad.npz")
    cache = _build(data, data["lesions"], tmp_path / "forced.npz", latents=lat, extra=["--allow-channel-mismatch"])
    assert str(cache["channel"]) == "lesion" and str(cache["latents_channel"]) == "disconnectome"
    assert str(cache["latents_representation"]) == "disconnectome"
    np.testing.assert_array_equal(cache["Z"], Z)      # aligned by basename, same sorted order
    # the reverse pairing is refused too
    lat2 = tmp_path / "mask_latents.npz"
    _latents(lat2, data["discos"], representation="lesion")
    with pytest.raises(SystemExit, match=r"(?s)disconnectome.*lesion"):
        _build(data, data["discos"], tmp_path / "bad2.npz", latents=lat2)


def test_matching_channels_pass_and_are_recorded(data, tmp_path, capsys):
    lat = tmp_path / "disco_latents.npz"
    _latents(lat, data["discos"], representation="disconnectome",
             images_dir=os.path.dirname(data["discos"][0]), images_channel="disconnectome")
    cache = _build(data, data["discos"], tmp_path / "ok.npz", latents=lat)
    assert str(cache["channel"]) == "disconnectome" and str(cache["latents_channel"]) == "disconnectome"
    assert str(cache["latents"]) == os.path.abspath(lat)
    assert "matches the images" in capsys.readouterr().out
    # the recorded image channel decides, not the encoder's representation:
    # masks encoded by a disconnectome encoder pair with a lesion cache
    lat2 = tmp_path / "masks_through_disco_encoder.npz"
    _latents(lat2, data["lesions"], representation="disconnectome", images_channel="lesion")
    cache = _build(data, data["lesions"], tmp_path / "ok2.npz", latents=lat2)
    assert str(cache["channel"]) == "lesion" and str(cache["latents_channel"]) == "lesion"


def test_without_metadata_the_name_decides_else_only_a_warning(data, tmp_path, capsys):
    named = tmp_path / "E1_seed0_disco.npz"
    _latents(named, data["lesions"])
    with pytest.raises(SystemExit, match=r"(?s)lesion.*disconnectome"):
        _build(data, data["lesions"], tmp_path / "bad.npz", latents=named)
    named_ok = tmp_path / "E1_seed0_lesion.npz"
    _latents(named_ok, data["lesions"])
    assert str(_build(data, data["lesions"], tmp_path / "ok.npz", latents=named_ok)["latents_channel"]) == "lesion"
    capsys.readouterr()
    anon = tmp_path / "pool.npz"
    _latents(anon, data["lesions"])
    cache = _build(data, data["lesions"], tmp_path / "warned.npz", latents=anon)
    assert str(cache["channel"]) == "lesion" and str(cache["latents_channel"]) == ""
    assert "cannot verify the channel" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# (d) consistency with the evaluation's labels on the same images
# --------------------------------------------------------------------------- #
def test_cache_overlaps_agree_with_the_replica_labels(data, atlas, tmp_path):
    """On the continuous maps, giles_replica.label_images (count > round(0.05 *
    subnetwork volume) after binarising at 0.5) and the cache (fraction >
    0.05 after the same binarisation) must agree lesion by lesion and
    subnetwork by subnetwork. The one admissible difference is the boundary
    cell where 0.05 * volume rounds up: the replica then needs one voxel more
    than the strict fraction (count == round(0.05 * vol) > 0.05 * vol), a
    prior hit and a replica miss, never the reverse."""
    cache = _build(data, data["discos"], tmp_path / "disco.npz")
    pairs = _pairs(atlas)
    df = label_images(data["discos"], pairs, roi_thresh=0.05, disco_thresh=0.5, progress=False)
    assert list(df["filename"]) == [str(f) for f in cache["files"]]
    np.testing.assert_array_equal(df["vol"].to_numpy(), cache["volumes"])
    np.testing.assert_allclose(df[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(), cache["centroids"], atol=1e-4)
    n_cells = n_hits = n_boundary = 0
    for ki, k in enumerate(atlas.networks):
        pair, name = pairs[k], DEFICIT_NAMES[k - 1]
        for s in range(2):
            vol = pair.vols[s]
            replica = df[f"{name}_W{pair.treatments[s]}"].to_numpy().astype(bool)
            frac = cache["overlaps"][:, ki, s].astype(np.float64)
            prior = frac > 0.05
            count = np.rint(frac * vol).astype(int)
            boundary = prior & ~replica & (count == int(np.round(vol * 0.05))) & (count > 0.05 * vol)
            agree = (prior == replica) | boundary
            assert agree.all(), f"network {k} subnetwork {s}: {np.flatnonzero(~agree)}"
            assert not np.any(replica & ~prior)        # a replica hit is always a prior hit
            n_cells += len(prior); n_hits += int(replica.sum()); n_boundary += int(boundary.sum())
    assert n_hits > 0 and n_cells == N_LESIONS * N_NETWORKS * 2
    assert n_boundary <= 0.02 * n_cells


# --------------------------------------------------------------------------- #
# the cohort exposes the channel and logs it at load time
# --------------------------------------------------------------------------- #
def test_cohort_exposes_the_channel_and_logs_the_fractions(data, atlas, tmp_path, caplog):
    path = tmp_path / "lesion.npz"
    _build(data, data["lesions"], path)
    with caplog.at_level(logging.INFO, logger="neurocausalpfn"):
        cohort = NeuroPriorCohort.from_cache(atlas, str(path), family="giles", seed=0,
                                             n_context=16, n_query=4, augment=False, min_slice=8)
    assert cohort.channel == "lesion"
    assert cohort.cache_meta["threshold"] == 0.5 and cohort.cache_meta["modality"] == "receptor"
    assert cohort.cache_meta["images_dir"] == os.path.abspath(os.path.dirname(data["lesions"][0]))
    lines = [r.getMessage() for r in caplog.records if "prior cache" in r.getMessage()]
    assert len(lines) == 1 and "channel lesion" in lines[0] and "per-network responsive fraction" in lines[0]
    assert ch.format_summary(ch.summarize_overlaps(cohort.overlaps)) in lines[0]
    # a cache written before the channel was recorded still loads and says so
    old = tmp_path / "old.npz"
    with np.load(path) as z:
        np.savez(old, overlaps=z["overlaps"], centroids=z["centroids"], volumes=z["volumes"], files=z["files"])
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="neurocausalpfn"):
        legacy = NeuroPriorCohort.from_cache(atlas, str(old), family="giles", seed=0,
                                             n_context=16, n_query=4, augment=False, min_slice=8)
    assert legacy.channel is None and legacy.cache_meta == {}
    assert any("channel unrecorded" in r.getMessage() for r in caplog.records)
