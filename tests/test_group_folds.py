"""Group-aware folds for the virtual-trial replica and the grouping table."""
import importlib.util
import os

import numpy as np
import pytest

from neurocausalpfn.prior.giles_replica import group_folds, image_folds

_spec = importlib.util.spec_from_file_location(
    "build_group_table",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "build_group_table.py"))
bgt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bgt)


def _cohort():
    # 12 files: groups A (3 acquisitions), B (2), C..I singles
    files = [f"f{i:02d}.nii.gz" for i in range(12)]
    groups = {"f00.nii.gz": ("A", 0), "f01.nii.gz": ("A", 1), "f02.nii.gz": ("A", 2),
              "f03.nii.gz": ("B", 0), "f04.nii.gz": ("B", 1)}
    for i in range(5, 12):
        groups[files[i]] = (f"G{i}", 0)
    return files, groups


def test_only_earliest_images_are_tested_and_all_of_them_once():
    files, groups = _cohort()
    for mode in ("giles", "strict"):
        folds = group_folds(files, groups, n_folds=3, mode=mode)
        tested = np.concatenate([te for _, te in folds])
        assert sorted(tested.tolist()) == [0, 3, 5, 6, 7, 8, 9, 10, 11]      # rank-0 only, once
        for tr, te in folds:
            assert not set(tr) & set(te)
            assert 1 not in te and 2 not in te and 4 not in te


def test_giles_mode_trains_repeats_everywhere_strict_keeps_groups_together():
    files, groups = _cohort()
    for tr, te in group_folds(files, groups, n_folds=3, mode="giles"):
        assert {1, 2, 4} <= set(tr)                       # repeats train in every fold
    for tr, te in group_folds(files, groups, n_folds=3, mode="strict"):
        if 0 in te:
            assert not {1, 2} & set(tr)                   # A's repeats leave with A's earliest
        else:
            assert {1, 2} <= set(tr)
        if 3 in te:
            assert 4 not in tr
        else:
            assert 4 in tr


def test_primary_split_matches_image_kfold_over_primaries():
    files, groups = _cohort()
    prim = np.array([0, 3, 5, 6, 7, 8, 9, 10, 11])
    ref = image_folds(len(prim), 3)
    got = group_folds(files, groups, n_folds=3, mode="giles")
    for (_, te_ref), (_, te) in zip(ref, got):
        assert np.array_equal(prim[te_ref], te)


def test_missing_and_malformed_tables_are_rejected():
    files, groups = _cohort()
    with pytest.raises(KeyError):
        group_folds(files + ["extra.nii.gz"], groups, n_folds=3)
    folds = group_folds(files + ["extra.nii.gz"], groups, n_folds=3, allow_missing=True)
    assert 12 in np.concatenate([te for _, te in folds])   # a lone file is its own group
    bad = dict(groups); bad["f05.nii.gz"] = ("A", 0)       # two earliest images in A
    with pytest.raises(ValueError):
        group_folds(files, bad, n_folds=3)


def test_group_without_earliest_image_is_train_only_everywhere():
    files, groups = _cohort()
    groups["f03.nii.gz"] = ("B", 1)                        # B has no rank-0 image any more
    for mode in ("giles", "strict"):
        folds = group_folds(files, groups, n_folds=3, mode=mode)
        for tr, te in folds:
            assert 3 in tr and 4 in tr and 3 not in te and 4 not in te
        tested = np.concatenate([te for _, te in folds])
        assert sorted(tested.tolist()) == [0, 5, 6, 7, 8, 9, 10, 11]


def test_build_group_table_ranks_by_date_and_hides_the_key():
    cw = [{"a_file": "pub1.nii.gz", "b_file": "orig_x.nii.gz"},
          {"a_file": "pub2.nii.gz", "b_file": "orig_y.nii.gz"},
          {"a_file": "pub3.nii.gz", "b_file": "orig_z.nii.gz"},
          {"a_file": "pub4.nii.gz", "b_file": "orig_missing.nii.gz"}]
    info = [{"seg": "sub/orig_x.nii.gz", "pid": "P1", "date": "20200105"},
            {"seg": "sub/orig_y", "pid": "P1", "date": "20191230"},
            {"seg": "sub/orig_z.nii.gz", "pid": "P2", "date": "20200101"}]
    rows, unmatched = bgt.build(cw, info, "seg", "pid", "date", salt="s3cret")
    by = {r["filename"]: r for r in rows}
    assert unmatched == ["pub4.nii.gz"]
    assert by["pub2.nii.gz"]["rank"] == 0 and by["pub1.nii.gz"]["rank"] == 1   # earlier date first
    assert by["pub1.nii.gz"]["group"] == by["pub2.nii.gz"]["group"] != by["pub3.nii.gz"]["group"]
    blob = str(rows)
    assert "P1" not in blob and "20200105" not in blob and "20191230" not in blob   # nothing sensitive copied
    assert by["pub1.nii.gz"]["n_in_group"] == 2


def test_build_group_table_marks_multi_acquisition_subjects_train_only():
    cw = [{"a_file": "p1", "b_file": "x.nii.gz"}, {"a_file": "p2", "b_file": "y.nii.gz"},
          {"a_file": "p3", "b_file": "z.nii.gz"}]
    info = [{"seg": "x.nii.gz", "pid": "A", "date": "20200101"},
            {"seg": "y.nii.gz", "pid": "A", "date": "20200301"},
            {"seg": "z.nii.gz", "pid": "B", "date": "20200101"}]
    full = info + [{"seg": "w.nii.gz", "pid": "B", "date": "20190101"}]   # B's other scan was excluded
    rows, _ = bgt.build(cw, info, "seg", "pid", "date", "s", all_rows=full, singles_only=True)
    by = {r["filename"]: r for r in rows}
    assert by["p1"]["rank"] == 1 and by["p2"]["rank"] == 2      # multi-acquisition: train-only
    assert by["p3"]["rank"] == 1 and by["p3"]["n_all"] == 2      # single kept, but multi in the full table
    rows, _ = bgt.build(cw, info, "seg", "pid", "date", "s", all_rows=full, singles_only=False)
    assert [r["rank"] for r in sorted(rows, key=lambda r: r["filename"])] == [0, 1, 0]
