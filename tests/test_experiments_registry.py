"""Registry integrity and builder logic for Table 9 (torch-free)."""
import numpy as np

from neurocausalpfn.experiments.registry import (ARM_A_ORDER, REGISTRY,
                                                 TIER_GATES, arm_experiments,
                                                 dependency_order, get_experiment)
from neurocausalpfn.experiments.runner import build_runs, scale_dim


def test_all_table9_experiments_present():
    ids = {e.eid for e in REGISTRY}
    expected = {"E1", "E2", "E3", "E4", "E5a", "E5b", "E5c", "E6", "E7",
                "E8a", "E8b", "E8c", "E9a", "E9b", "E10a", "E10b", "E10c",
                "E11", "E12"}
    assert expected <= ids


def test_dependencies_are_known_and_acyclic():
    ids = {e.eid for e in REGISTRY}
    for e in REGISTRY:
        for dep in e.depends_on:
            assert dep in ids, (e.eid, dep)
    order = dependency_order()          # raises on a cycle
    # every dependency appears before its dependent
    pos = {eid: i for i, eid in enumerate(order)}
    for e in REGISTRY:
        for dep in e.depends_on:
            assert pos[dep] < pos[e.eid]


def test_arm_a_order_matches_spec():
    assert dependency_order(list(ARM_A_ORDER)) == list(ARM_A_ORDER)
    # E7 (backbone) precedes E3 (dimensionality) precedes E4 (ARD)
    o = dependency_order(list(ARM_A_ORDER))
    assert o.index("E7") < o.index("E3") < o.index("E4")
    assert o.index("E6") < o.index("E9a") < o.index("E9b")


def test_tier_gates_semantics():
    assert TIER_GATES["T1"].passes(0.8) is True
    assert TIER_GATES["T1"].passes(0.5) is False
    assert TIER_GATES["T4"].passes(0.30) is True     # below 0.349 -> pass
    assert TIER_GATES["T4"].passes(0.40) is False
    assert TIER_GATES["T2"].kind == "deprioritize"
    assert TIER_GATES["T3"].passes(5.0) is None      # informational


def test_arm_experiments_grouping():
    a = [e.eid for e in arm_experiments("A")]
    assert set(a) == set(ARM_A_ORDER)
    assert all(get_experiment(e).arm == "E" for e in [x.eid for x in arm_experiments("E")])


def test_build_runs_expands_grids():
    ctx = {"backbone": "resnet", "w_dice": 1.0, "dims": (50, 50)}
    assert len(build_runs(get_experiment("E1"), "prototype", ctx)) == 1
    assert len(build_runs(get_experiment("E2"), "prototype", ctx)) == 3    # w_dice grid
    assert len(build_runs(get_experiment("E7"), "prototype", ctx)) == 4    # backbones
    assert len(build_runs(get_experiment("E3"), "prototype", ctx)) == 8    # dim sweep
    assert len(build_runs(get_experiment("E6"), "prototype", ctx)) == 3    # channels


def test_build_runs_uses_context_winner():
    ctx = {"backbone": "resnet50", "w_dice": 0.5, "dims": (75, 25)}
    runs = build_runs(get_experiment("E3"), "prototype", ctx)
    assert all(r.meta["backbone"] == "resnet50" for r in runs)
    # E5a inherits the E7 backbone winner from the context
    e5a = build_runs(get_experiment("E5a"), "prototype", ctx)[0]
    assert e5a.meta["backbone"] == "resnet50" and e5a.meta["use_daft"] is True


def test_scale_dim_prototype_and_full():
    assert [scale_dim("prototype", d) for d in (25, 50, 75, 100)] == [4, 8, 12, 16]
    assert scale_dim("full", 50) == 50
