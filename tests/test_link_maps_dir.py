"""F42: a data folder whose disconnectomes point at another map set."""
import os

import pytest

from scripts import link_maps_dir as lmd


def _touch(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()


def test_links_keep_received_names_and_point_at_the_other_maps(tmp_path, capsys):
    data, maps, out = tmp_path / "Full data", tmp_path / "bcb", tmp_path / "out" / "Full data"
    for i in (3, 12, 120):
        _touch(str(data / "lesions" / f"lesion{i}_NA_NA.nii.gz"))
        _touch(str(data / "disconnectomes" / f"disco_lesion{i}_NA_NA.nii.gz"))
        _touch(str(maps / f"lesion{i}_70_M.nii.gz"))
    lmd.main(["--data", str(data), "--maps", str(maps), "--out", str(out)])
    assert "3 disconnectome links" in capsys.readouterr().out
    assert sorted(os.listdir(out / "disconnectomes")) == sorted(os.listdir(data / "disconnectomes"))
    assert os.path.realpath(out / "disconnectomes" / "disco_lesion12_NA_NA.nii.gz") == str(maps / "lesion12_70_M.nii.gz")
    assert os.path.realpath(out / "lesions") == str(data / "lesions")
    # a second run replaces the links instead of failing
    lmd.main(["--data", str(data), "--maps", str(maps), "--out", str(out)])


def test_refuses_when_a_received_map_has_no_counterpart(tmp_path):
    data, maps, out = tmp_path / "Full data", tmp_path / "bcb", tmp_path / "out"
    for i in (1, 2):
        _touch(str(data / "lesions" / f"lesion{i}_NA_NA.nii.gz"))
        _touch(str(data / "disconnectomes" / f"lesion{i}_NA_NA.nii.gz"))
    _touch(str(maps / "lesion1_NA_NA.nii.gz"))
    with pytest.raises(SystemExit, match="1 of 2"):
        lmd.main(["--data", str(data), "--maps", str(maps), "--out", str(out)])
    assert not os.path.exists(out)
