"""Per-fold pools: training-side rows of the cohort cache with fold-k latents."""
import os

import numpy as np
import pytest

from scripts import build_fold_pool_cache as bfp


def _cache(path, names, d_over=(3, 2), seed=0, channel="disconnectome"):
    rng = np.random.default_rng(seed)
    m = len(names)
    np.savez(path, overlaps=rng.random((m,) + d_over).astype(np.float32),
             overlaps_network=rng.random((m, 3)).astype(np.float32),
             centroids=rng.random((m, 3)), volumes=rng.random(m) * 100,
             files=np.array(names), networks=np.array(["a", "b", "c"]),
             channel=np.array(channel), images_dir=np.array("/img/" + os.path.basename(path)),
             threshold=np.array(0.5), modality=np.array("receptor"),
             Z=rng.random((m, 7)).astype(np.float32))           # some other encoder's latents


def _fold(path, names, k, tr_idx, Z):
    np.savez(path, Ztr=Z[tr_idx], Zte=np.zeros((1, Z.shape[1])), tr_idx=np.array(tr_idx),
             te_idx=np.array([i for i in range(len(names)) if i not in tr_idx]),
             files=np.array(names), fold=k, n_folds=2)


def test_fold_pool_holds_training_rows_with_fold_latents(tmp_path, capsys):
    names = [f"lesion{i}_NA_NA.nii.gz" for i in range(8)]
    _cache(tmp_path / "uclh.npz", names)
    Z = np.random.default_rng(1).normal(size=(8, 5)).astype(np.float32)
    fd = tmp_path / "folds"; fd.mkdir()
    tr0 = [0, 2, 3, 5, 6, 7]; tr1 = [1, 4] + [0, 2]
    _fold(fd / "fold0.npz", names, 0, tr0, Z)
    _fold(fd / "fold1.npz", names, 1, tr1, Z)
    bfp.main(["--cache", str(tmp_path / "uclh.npz"), "--fold-dir", str(fd), "--folds", "0", "1",
              "--out-dir", str(tmp_path / "pools")])
    out = capsys.readouterr().out
    assert "fold 0: 6 training-side rows" in out and "fold 1: 4 training-side rows" in out
    with np.load(tmp_path / "pools" / "fold0.npz", allow_pickle=True) as p:
        assert list(p["files"]) == [names[i] for i in tr0]
        np.testing.assert_array_equal(p["Z"], Z[tr0])            # the fold encoder's, not the cache's
        with np.load(tmp_path / "uclh.npz") as c:
            np.testing.assert_array_equal(p["overlaps"], c["overlaps"][tr0])
            np.testing.assert_array_equal(p["volumes"], c["volumes"][tr0])
        assert str(p["channel"]) == "disconnectome" and float(p["threshold"]) == 0.5
        assert int(p["fold"]) == 0 and int(p["n_main"]) == 6 and int(p["n_extra"]) == 0
        assert "latents" not in p.files


def test_extra_cohort_is_appended_with_its_fold_latents(tmp_path):
    names = [f"lesion{i}_NA_NA.nii.gz" for i in range(6)]
    knames = [f"kch{i}.nii.gz" for i in range(4)]
    _cache(tmp_path / "uclh.npz", names)
    _cache(tmp_path / "kch.npz", knames, seed=3)
    Z = np.random.default_rng(1).normal(size=(6, 5)).astype(np.float32)
    fd = tmp_path / "folds"; fd.mkdir()
    _fold(fd / "fold0.npz", names, 0, [0, 1, 2, 3], Z)
    Zk = np.random.default_rng(2).normal(size=(4, 5)).astype(np.float32)
    lat = tmp_path / "lat" / "fold0"; lat.mkdir(parents=True)
    np.savez(lat / "E1_fold0_disco.npz", Z=Zk[::-1], files=np.array(knames[::-1]))   # other order
    bfp.main(["--cache", str(tmp_path / "uclh.npz"), "--fold-dir", str(fd), "--folds", "0",
              "--out-dir", str(tmp_path / "pools"), "--extra-cache", str(tmp_path / "kch.npz"),
              "--extra-latents", str(tmp_path / "lat" / "fold{fold}" / "*.npz")])
    with np.load(tmp_path / "pools" / "fold0.npz", allow_pickle=True) as p:
        assert len(p["files"]) == 8 and int(p["n_extra"]) == 4
        # extra rows follow the extra latents' order, anatomy aligned by name
        assert list(p["files"][4:]) == knames[::-1]
        np.testing.assert_array_equal(p["Z"][4:], Zk[::-1])
        with np.load(tmp_path / "kch.npz") as c:
            np.testing.assert_array_equal(p["overlaps"][4:], c["overlaps"][::-1])
        assert "uclh.npz" in str(p["images_dir"]) and "kch.npz" in str(p["images_dir"])


def test_mismatches_stop_the_build(tmp_path):
    names = [f"lesion{i}_NA_NA.nii.gz" for i in range(4)]
    _cache(tmp_path / "uclh.npz", names[:3])                      # one image has no anatomy row
    Z = np.zeros((4, 5), np.float32)
    fd = tmp_path / "folds"; fd.mkdir()
    _fold(fd / "fold0.npz", names, 0, [0, 1, 3], Z)
    with pytest.raises(SystemExit, match="have no row"):
        bfp.main(["--cache", str(tmp_path / "uclh.npz"), "--fold-dir", str(fd), "--folds", "0",
                  "--out-dir", str(tmp_path / "pools")])
    _cache(tmp_path / "kch_lesion.npz", ["k1.nii.gz"], channel="lesion")
    _cache(tmp_path / "uclh.npz", names)
    with pytest.raises(SystemExit, match="differ in channel"):
        bfp.main(["--cache", str(tmp_path / "uclh.npz"), "--fold-dir", str(fd), "--folds", "0",
                  "--out-dir", str(tmp_path / "pools"), "--extra-cache", str(tmp_path / "kch_lesion.npz"),
                  "--extra-latents", "x{fold}.npz"])
    with pytest.raises(SystemExit, match="no fold file"):
        bfp.main(["--cache", str(tmp_path / "uclh.npz"), "--fold-dir", str(fd), "--folds", "7",
                  "--out-dir", str(tmp_path / "pools")])
