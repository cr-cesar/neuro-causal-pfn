"""nmf50_both: one per-fold NMF per channel, factors concatenated."""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("sklearn")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_giles_replica", os.path.join(ROOT, "scripts", "run_giles_replica.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_channel(folder, n, seed, shape=(8, 8, 8)):
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    files = []
    for i in range(n):
        arr = (rng.random(shape) > 0.7).astype(np.float32)
        path = os.path.join(folder, f"img{i:03d}.nii.gz")
        nib.save(nib.Nifti1Image(arr, np.eye(4)), path)
        files.append(path)
    return files


def test_nmf50_both_concatenates_two_per_fold_factorisations(tmp_path):
    mod = _load_script()
    n = 12
    files = _write_channel(tmp_path / "disco", n, seed=0)
    second_dir = tmp_path / "lesion"
    _write_channel(second_dir, n, seed=1)
    second = mod._second_channel_files(files, str(second_dir))
    assert [os.path.basename(p) for p in second] == [os.path.basename(p) for p in files]

    labels = pd.DataFrame({"vol": np.arange(1, n + 1)})
    reps = mod._built_in_representations(labels, files, ["volume", "nmf50_both"],
                                         second_files=second)
    assert set(reps) == {"volume", "nmf50_both_perfold"}
    refit = reps["nmf50_both_perfold"]
    tr, te = np.arange(0, 9), np.arange(9, n)
    Ztr, Zte = refit(tr, te)
    k = min(50, n - 1, len(tr))          # capped by the fold's train size
    assert Ztr.shape == (9, 2 * k) and Zte.shape == (3, 2 * k)
    assert Ztr.dtype == np.float32 and np.isfinite(Ztr).all() and (Ztr >= 0).all()


def test_nmf50_both_requires_the_second_channel(tmp_path):
    mod = _load_script()
    files = _write_channel(tmp_path / "disco", 6, seed=0)
    labels = pd.DataFrame({"vol": np.arange(1, 7)})
    with pytest.raises(SystemExit):
        mod._built_in_representations(labels, files, ["nmf50_both"])
    with pytest.raises(SystemExit):
        mod._second_channel_files(files, str(tmp_path / "missing"))
