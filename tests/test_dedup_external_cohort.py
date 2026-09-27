import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "dedup_external_cohort",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "dedup_external_cohort.py"))
dx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dx)


def test_mask_hash_depends_on_voxels_and_grid_only():
    a = np.array([3, 4, 5], dtype=np.int32)
    assert dx.mask_hash(a, (4, 4, 4)) == dx.mask_hash(a.copy(), (4, 4, 4))
    assert dx.mask_hash(a, (4, 4, 4)) != dx.mask_hash(a, (5, 4, 4))
    assert dx.mask_hash(a, (4, 4, 4)) != dx.mask_hash(np.array([3, 4], dtype=np.int32), (4, 4, 4))


def test_best_overlaps_blocks_by_centroid_and_volume():
    ref = [np.arange(0, 100, dtype=np.int32), np.arange(1000, 1100, dtype=np.int32),
           np.arange(2000, 2010, dtype=np.int32)]
    ref_c = np.array([[0, 0, 0], [50, 0, 0], [0, 0, 0]], dtype=float)
    ref_v = np.array([100, 100, 10])
    new = [np.arange(0, 90, dtype=np.int32),          # 90 % overlap with ref 0
           np.arange(1000, 1100, dtype=np.int32),      # identical to ref 1 but 50 mm away
           np.array([], dtype=np.int32)]              # empty mask
    new_c = np.array([[1, 0, 0], [0, 0, 0], [np.nan] * 3])
    new_v = np.array([90, 100, 0])
    d, i = dx.best_overlaps(new, new_c, new_v, ref, ref_c, ref_v, centroid_mm=20, vol_ratio=3)
    assert i[0] == 0 and abs(d[0] - 2 * 90 / 190) < 1e-9
    assert i[1] == -1 and d[1] == 0.0                  # blocked by the centroid distance
    assert i[2] == -1                                   # empty masks never match
    # ref 2 (10 voxels) is excluded from new 0 by the volume ratio (90 / 10 > 3)


def test_subject_matches_use_the_salted_hash_and_decide_combines_stages():
    training = {dx.hash_subject("s", "P1")}
    rows = [{"f": "a.nii.gz", "pid": "P1"}, {"f": "sub/b.nii.gz", "pid": "P2 "}]
    m = dx.subject_matches(rows, "f", "pid", "s", training)
    assert m == {"a.nii.gz": True, "b.nii.gz": False}
    assert dx.decide(False, 0.2, 0.9, None) == 1
    assert dx.decide(True, 0.0, 0.9, None) == 0
    assert dx.decide(False, 0.95, 0.9, False) == 0
    assert dx.decide(False, 0.1, 0.9, True) == 0


def test_best_overlaps_skips_references_with_the_same_key():
    ref = [np.arange(0, 100, dtype=np.int32), np.arange(0, 90, dtype=np.int32)]
    ref_c = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    ref_v = np.array([100, 90])
    new = [np.arange(0, 100, dtype=np.int32)]           # identical to ref 0
    new_c = np.array([[0, 0, 0]], dtype=float)
    new_v = np.array([100])
    d, i = dx.best_overlaps(new, new_c, new_v, ref, ref_c, ref_v)
    assert i[0] == 0 and d[0] == 1.0
    # same group as ref 0 -> ref 0 skipped, ref 1 (Dice 2*90/190) is the best other subject
    d, i = dx.best_overlaps(new, new_c, new_v, ref, ref_c, ref_v,
                            new_keys=["g1"], ref_keys=["g1", "g2"])
    assert i[0] == 1 and abs(d[0] - 2 * 90 / 190) < 1e-9
    # an empty key never excludes anything
    d, i = dx.best_overlaps(new, new_c, new_v, ref, ref_c, ref_v,
                            new_keys=[""], ref_keys=["", ""])
    assert i[0] == 0


def test_summarise_counts_per_site_and_threshold():
    rows = [{"site": "k", "exact_match": "", "best_dice": "0.62", "subject_match": ""},
            {"site": "k", "exact_match": "", "best_dice": "0.20", "subject_match": ""},
            {"site": "u", "exact_match": "x.nii.gz", "best_dice": "1.0", "subject_match": ""},
            {"site": "u", "exact_match": "", "best_dice": "0.93", "subject_match": "1"}]
    s = dx.summarise(rows, 0.5)
    assert s[""]["n"] == 4 and s["k"]["keep"] == 1 and s["u"]["keep"] == 0
    assert s["k"]["near_dup"] == 1 and s["u"]["near_dup"] == 1 and s["u"]["exact"] == 1
    assert s["k"]["hist"] == [1, 0, 1, 0, 0, 0, 0]
    s = dx.summarise(rows, 0.8)
    assert s["k"]["keep"] == 2 and s["k"]["near_dup"] == 0
    assert s["u"]["keep"] == 0                          # exact + subject links stay
