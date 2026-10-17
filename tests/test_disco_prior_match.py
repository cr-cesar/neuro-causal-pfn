"""F42: comparing disconnectome generators with the published trial sizes."""
import os
import stat
import sys

import numpy as np
import pytest

from scripts import disco_prior_match as dpm


def test_lesion_key_pairs_lesions_with_their_disconnectomes():
    assert dpm.lesion_key("lesion12_NA_NA.nii.gz") == "12"
    assert dpm.lesion_key("qdisco_lesion12_70_M.nii.gz") == "12"
    assert dpm.lesion_key("/x/disco_lesion12_NA_F.nii") == "12"
    assert dpm.lesion_key("lesion12_NA_NA.nii.gz") != dpm.lesion_key("lesion120_NA_NA.nii.gz")


def test_sample_is_fixed_by_the_seed_and_keeps_listing_order():
    files = [f"f{i:03d}" for i in range(50)]
    a, b = dpm.sample(files, 10, 0), dpm.sample(files, 10, 0)
    assert a == b and a == sorted(a) and len(set(a)) == 10
    assert dpm.sample(files, 10, 1) != a
    assert dpm.sample(files, 80, 0) == files


def test_published_table_has_sixteen_networks_per_split():
    for kind in ("receptor", "genetics"):
        assert len(dpm.PUBLISHED[kind]["N"]) == 16 and len(dpm.PUBLISHED[kind]["overlap"]) == 16
    assert dpm.PUBLISHED["receptor"]["N"][1] == 688 and dpm.PUBLISHED["receptor"]["N"][5] == 978


def test_compare_and_summarise_measure_the_distance_to_the_table():
    names = dpm._descriptives().NAMES
    # 4,119 maps whose enrolment equals the published counts exactly
    cats = {}
    for kind in ("receptor", "genetics"):
        c = np.zeros((dpm.N_PUBLISHED, 16), int)
        for k, n in enumerate(dpm.PUBLISHED[kind]["N"]):
            c[:n, k] = 1
        cats[kind] = c
    s = dpm.summarise(dpm.compare(cats, names, "exact"))
    assert all(r["median_abs_log2_ratio"] == pytest.approx(0, abs=1e-9) and r["within_25pct"] == 16 for r in s)
    # halving every trial is one unit of |log2 ratio| and leaves none within 25%
    halved = {}
    for kind in cats:
        c = np.zeros_like(cats[kind])
        for k, n in enumerate(dpm.PUBLISHED[kind]["N"]):
            c[:n // 2, k] = 1
        halved[kind] = c
    s2 = dpm.summarise(dpm.compare(halved, names, "half"))
    assert all(r["median_abs_log2_ratio"] == pytest.approx(1, abs=0.01) and r["within_25pct"] == 0 for r in s2)
    assert all(r["ratio_language"] == pytest.approx(0.5, abs=0.01) for r in s2)


def _atlas(tmp_path, nib, shape):
    for kind in ("receptor", "genetics"):
        d = tmp_path / "atlas" / "2mm_parcellations" / kind
        d.mkdir(parents=True)
        for k in range(1, 17):
            m = np.zeros(shape, np.int16)
            m[k % 8, :4, :] = 1
            m[k % 8, 4:6, :] = 2
            nib.save(nib.Nifti1Image(m, np.eye(4)), str(d / f"{k}_net.nii.gz"))


def _fake_quickdisco(tmp_path):
    """Writes qdisco_<name> into -o with every lesion voxel at 0.9, and prints
    the file names as the real tool does."""
    exe = tmp_path / "QuickDisco"
    exe.write_text(
        f"#!{sys.executable}\n"
        "import sys, os, numpy as np, nibabel as nib\n"
        "a = sys.argv[1:]\n"
        "files = a[a.index('-i') + 1:a.index('-p')]\n"
        "out = a[a.index('-o') + 1]\n"
        "for f in files:\n"
        "    img = nib.load(f)\n"
        "    v = (np.asarray(img.dataobj) > 0) * 0.9\n"
        "    nib.save(nib.Nifti1Image(v.astype(np.float32), img.affine), os.path.join(out, 'qdisco_' + os.path.basename(f)))\n"
        "    print('wrote', f)\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def test_end_to_end_with_a_fake_quickdisco_prints_no_file_name(tmp_path, capsys):
    nib = pytest.importorskip("nibabel")
    pytest.importorskip("pandas")
    shape = (8, 8, 8)
    _atlas(tmp_path, nib, shape)
    les, rec = tmp_path / "lesions", tmp_path / "received"
    les.mkdir()
    rec.mkdir()
    for i in range(12):
        v = np.zeros(shape, np.uint8)
        v[i % 8, :, :] = 1
        nib.save(nib.Nifti1Image(v, np.eye(4)), str(les / f"lesion{i}_NA_NA.nii.gz"))
        nib.save(nib.Nifti1Image(v.astype(np.float32) * 0.8, np.eye(4)), str(rec / f"lesion{i}_NA_NA.nii.gz"))
    priors = tmp_path / "p.h5"
    priors.write_bytes(b"")
    out = tmp_path / "res" / "match.csv"
    summ = dpm.main(["--lesions", str(les), "--received", str(rec), "--atlas-dir", str(tmp_path / "atlas"),
                     "--priors", f"FAKE={priors}", "--priors", f"MISSING={tmp_path / 'none.h5'}",
                     "--work", str(tmp_path / "work"), "--n", "8", "--quickdisco", _fake_quickdisco(tmp_path),
                     "--out", str(out)])
    printed = capsys.readouterr().out
    import re
    assert not re.search(r"lesion\d|_NA_NA|qdisco_", printed)        # no file names
    assert "MISSING: priors file not found" in printed
    assert set(summ["source"]) == {"received", "FAKE"} and set(summ["split"]) == {"receptor", "genetics"}
    assert (summ["n_maps"] == 8).all()
    assert out.exists() and os.path.exists(str(out).replace(".csv", "_summary.csv"))
    # the fake maps equal the received ones above threshold, so both score the same
    import pandas as pd
    rows = pd.read_csv(out)
    a = rows[rows.source == "FAKE"].enrolled_frac.to_numpy()
    b = rows[rows.source == "received"].enrolled_frac.to_numpy()
    assert np.allclose(a, b)
    # a second run reuses the maps already built
    built = sorted(os.listdir(tmp_path / "work" / "FAKE"))
    dpm.main(["--lesions", str(les), "--atlas-dir", str(tmp_path / "atlas"), "--priors", f"FAKE={priors}",
              "--work", str(tmp_path / "work"), "--n", "8", "--quickdisco", "/nonexistent/QuickDisco"])
    assert sorted(os.listdir(tmp_path / "work" / "FAKE")) == built
