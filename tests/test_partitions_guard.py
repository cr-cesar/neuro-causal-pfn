"""Audit of 28 Oct: the random partitions of the virtual-trial family never
rebuild a benchmark network from its two halves, whether under two channel
names (the primary overlaps repeat one split) or inside a union of atoms."""
import numpy as np

from neurocausalpfn.prior.partitions import atom_table, draw_virtual_networks, is_benchmark_pair


def _cache(m=200, k=4, seed=0):
    rng = np.random.default_rng(seed)
    rec = rng.uniform(size=(m, k, 2)).astype(np.float32)
    gen = rng.uniform(size=(m, k, 2)).astype(np.float32)
    net = rng.uniform(size=(m, k)).astype(np.float32)
    # the primary overlaps are the receptor split, as in a cache built on it
    return rec.copy(), {"overlaps_receptor": rec, "overlaps_genetics": gen, "overlaps_network": net}


def test_duplicate_channels_are_one_atom():
    primary, extra = _cache()
    labels, table = atom_table(primary, extra)
    k = primary.shape[1]
    assert table.shape[1] == len(labels) == 2 * k + 2 * k + k     # receptor counted once
    assert len({table[:, i].tobytes() for i in range(table.shape[1])}) == table.shape[1]


def test_guard_blocks_halves_inside_unions():
    labels = [("primary", 0, 0), ("primary", 0, 1), ("genetics", 2, 1), ("network", 1, -1)]
    assert is_benchmark_pair([0], [1], labels)
    assert is_benchmark_pair([0, 2], [1], labels)          # union holding half 0 against half 1
    assert is_benchmark_pair([0, 3], [2, 1], labels)        # the pair sits inside both unions
    assert not is_benchmark_pair([3], [2, 1], labels)
    assert not is_benchmark_pair([0, 2], [3], labels)
    assert not is_benchmark_pair([0], [2], labels)


def test_drawn_networks_never_hold_a_benchmark_pair():
    primary, extra = _cache(k=3)
    labels, table = atom_table(primary, extra)
    rng = np.random.default_rng(1)
    for _ in range(300):
        vov, desc = draw_virtual_networks(rng, labels, table, 3, atoms_per_side=(1, 2))
        for d in desc:
            for ca, ka, ha in d["A"]:
                for cb, kb, hb in d["B"]:
                    assert not (ca == cb and ka == kb and {ha, hb} == {0, 1})
        assert vov.shape == (len(primary), 3, 2)
