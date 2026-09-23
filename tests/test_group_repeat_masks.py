import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "group_repeat_masks",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "group_repeat_masks.py"))
grm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grm)


def test_dice_on_sorted_index_arrays():
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    b = np.array([3, 4, 5, 6], dtype=np.int32)
    assert grm.dice_sorted(a, a) == 1.0
    assert grm.dice_sorted(a, b) == 0.5
    assert grm.dice_sorted(a, np.array([], dtype=np.int32)) == 0.0


def test_union_find_groups_are_transitive_and_compact():
    uf = grm.UnionFind(6)
    uf.union(0, 1); uf.union(1, 2); uf.union(4, 5)
    g = uf.groups()
    assert g[0] == g[1] == g[2] and g[4] == g[5] and g[3] not in (g[0], g[4])
    assert sorted(set(g.tolist())) == [0, 1, 2]
    assert grm.group_summary(g) == {"n_images": 6, "n_groups": 3, "n_extra": 3,
                                    "max_group": 3, "n_groups_ge2": 2}


def test_blocking_by_sex_and_age():
    ages = [60.0, 61.0, 63.0, 60.0, None]
    sexes = ["M", "M", "M", "F", "F"]
    pairs = grm.blocks(ages, sexes, age_tol=1.0)
    assert (0, 1) in pairs and (1, 2) not in pairs and (0, 3) not in pairs   # age band, sex block
    assert (3, 4) in pairs                                                   # unknown age matches
    assert (0, 4) not in pairs                                               # unknown age, other sex
