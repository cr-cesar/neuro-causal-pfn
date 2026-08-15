"""The tiered data layout: data/Trial data + data/Full data + data/atlases."""
import os

import pytest

from neurocausalpfn.data import paths


@pytest.fixture()
def in_tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(paths.TIER_ENV_VAR, raising=False)
    return tmp_path


def test_default_tier_is_full(in_tmp_cwd):
    assert paths.current_tier() == "full"
    assert paths.lesion_root() == os.path.join("data", "Full data", "lesions")
    assert paths.disconnectome_root() == os.path.join("data", "Full data", "disconnectomes")


def test_env_var_selects_trial(in_tmp_cwd, monkeypatch):
    monkeypatch.setenv(paths.TIER_ENV_VAR, "trial")
    assert paths.current_tier() == "trial"
    assert paths.lesion_root() == os.path.join("data", "Trial data", "lesions")


def test_explicit_tier_argument_wins(in_tmp_cwd):
    assert paths.lesion_root("trial") == os.path.join("data", "Trial data", "lesions")
    assert paths.disconnectome_root("full") == os.path.join("data", "Full data", "disconnectomes")


def test_set_tier_validates(in_tmp_cwd):
    with pytest.raises(ValueError):
        paths.set_tier("production")
    assert paths.set_tier("trial") == "trial"
    assert paths.current_tier() == "trial"


def test_case_insensitive_folder_match(in_tmp_cwd):
    os.makedirs(os.path.join("data", "trial data", "lesions"))
    assert paths.lesion_root("trial") == os.path.join("data", "trial data", "lesions")


def test_legacy_flat_layout_fallback(in_tmp_cwd):
    os.makedirs(os.path.join("data", "lesions"))
    assert paths.lesion_root("full") == os.path.join("data", "lesions")
    # the tiered folder wins as soon as it exists
    os.makedirs(os.path.join("data", "Full data", "lesions"))
    assert paths.lesion_root("full") == os.path.join("data", "Full data", "lesions")


def test_missing_everything_returns_canonical(in_tmp_cwd):
    # nothing on disk: the canonical tiered path is returned so the dataset
    # loader can report it (and prototype mode falls back to synthetic masks)
    assert paths.lesion_root("full") == os.path.join("data", "Full data", "lesions")


def test_atlas_dir_is_shared(in_tmp_cwd):
    assert paths.ATLAS_DIR == os.path.join("data", "atlases")
