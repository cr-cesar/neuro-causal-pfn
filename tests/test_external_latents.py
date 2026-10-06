"""External embeddings: alignment by file name, token averaging, per-fold PCA."""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_align_reorders_rows_by_basename_and_reports_extras():
    conv = _load("convert_external_latents")
    images = ["/d/b.nii.gz", "/d/a.nii.gz", "/d/c.nii.gz"]          # replica order (as given)
    names = np.array(["c.nii.gz", "a.nii.gz", "zz.nii.gz", "b.nii.gz"])
    Z = np.arange(8, dtype=np.float32).reshape(4, 2)
    Z_al, base, extra = conv.align(names, Z, images)
    assert base == ["b.nii.gz", "a.nii.gz", "c.nii.gz"]
    assert np.array_equal(Z_al, Z[[3, 1, 0]]) and extra == 1


def test_align_fails_on_missing_or_duplicated_names():
    conv = _load("convert_external_latents")
    with pytest.raises(SystemExit):
        conv.align(np.array(["a.nii.gz"]), np.zeros((1, 2)), ["/d/a.nii.gz", "/d/b.nii.gz"])
    with pytest.raises(SystemExit):
        conv.align(np.array(["a.nii.gz", "a.nii.gz"]), np.zeros((2, 2)), ["/d/a.nii.gz"])


def test_token_mean_averages_the_flattened_layout():
    conv = _load("convert_external_latents")
    Z = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 12)      # 4 tokens x 3 dims per row
    M = conv.token_mean(Z, 4)
    assert M.shape == (2, 3)
    assert np.allclose(M[0], Z[0].reshape(4, 3).mean(axis=0))
    with pytest.raises(SystemExit):
        conv.token_mean(Z, 5)


def test_pca_per_fold_fits_on_train_rows_only():
    pytest.importorskip("sklearn")
    rep = _load("run_giles_replica")
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 40)).astype(np.float32)
    refit = rep.pca_per_fold(Z, 5)
    tr, te = np.arange(0, 45), np.arange(45, 60)
    Ztr, Zte = refit(tr, te)
    assert Ztr.shape == (45, 5) and Zte.shape == (15, 5) and Ztr.dtype == np.float32
    # the fit is a function of the train rows only: shuffling test rows changes nothing in Ztr
    Z2 = Z.copy(); Z2[te] = rng.normal(size=(15, 40))
    Ztr2, _ = rep.pca_per_fold(Z2, 5)(tr, te)
    assert np.allclose(Ztr, Ztr2)


def test_ensemble_latents_concatenates_members_and_checks_rows(tmp_path):
    rep = _load("run_giles_replica")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(20, 5)).astype(np.float32), rng.normal(size=(20, 7)).astype(np.float32)
    np.savez(tmp_path / "s0.npz", Z=a); np.savez(tmp_path / "s1.npz", Z=b)
    Z = rep.ensemble_latents([str(tmp_path / "s0.npz"), str(tmp_path / "s1.npz")], 20)
    assert Z.shape == (20, 12) and np.allclose(Z[:, :5], a) and np.allclose(Z[:, 5:], b)
    # a full-listing export is subset like --latents
    Zs = rep.ensemble_latents([str(tmp_path / "s0.npz")], 4, sub_idx=np.array([0, 2, 4, 6]), n_all=20)
    assert np.allclose(Zs, a[[0, 2, 4, 6]])
    with pytest.raises(SystemExit):
        rep.ensemble_latents([str(tmp_path / "s0.npz")], 19)
