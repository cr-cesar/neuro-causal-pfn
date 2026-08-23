"""E0 (no-VAE reference baselines) and E11b (backbone x loss 2x2 audit)."""
import numpy as np

from neurocausalpfn.experiments import runner
from neurocausalpfn.experiments.registry import get_experiment

_FAST = {"n_synth": 8, "resolution": [24, 28, 24]}


def test_e0_builds_three_reference_runs():
    specs = runner.build_runs(get_experiment("E0"), "prototype", {})
    assert [sp.meta["method"] for sp in specs] == ["nmf50", "nmf21", "volume"]
    assert all(sp.kind == "reference" for sp in specs)


def test_e0_volume_and_nmf_artifacts(tmp_path):
    spec_vol = runner.RunSpec(label="E0[method=volume]", kind="reference",
                              meta={"method": "volume"})
    a = runner._exec_reference(spec_vol, "prototype", 0, str(tmp_path), dict(_FAST))
    assert a.Z.shape == (8, 1)
    assert float(a.Z.max()) <= 1.0 and a.volume.sum() > 0

    spec_nmf = runner.RunSpec(label="E0[method=nmf50]", kind="reference",
                              meta={"method": "nmf50"})
    a = runner._exec_reference(spec_nmf, "prototype", 0, str(tmp_path), dict(_FAST))
    assert a.Z.shape[0] == 8 and 2 <= a.Z.shape[1] <= 8
    assert np.isfinite(a.Z).all() and a.Z.min() >= 0.0     # NMF is non-negative


def test_e11b_grid_uses_top2_rankings():
    context = {"dims": (50, 50),
               "ranking": {"E3": ["E3[backbone=resnet18]", "E3[backbone=wide]",
                                  "E3[backbone=cnn]"],
                           "E2": ["E2[w_dice=0.5]", "E2[w_dice=0.1]",
                                  "E2[w_dice=1.0]"]}}
    specs = runner.build_runs(get_experiment("E11b"), "full", context)
    labels = [sp.label for sp in specs]
    assert len(specs) == 4
    assert "E11b[backbone=resnet18,w_dice=0.5]" in labels
    assert "E11b[backbone=wide,w_dice=0.1]" in labels
    assert all(sp.meta["backbone"] in ("resnet18", "wide") for sp in specs)
    assert all(sp.meta["w_dice"] in (0.5, 0.1) for sp in specs)


def test_e11b_falls_back_without_rankings():
    specs = runner.build_runs(get_experiment("E11b"), "full",
                              {"backbone": "resnet", "w_dice": 1.0})
    assert len(specs) == 4                                  # 2 fallbacks x 2


def test_propagate_stores_ranking():
    exp = get_experiment("E3")
    agg = {"E3[backbone=cnn]": {"label": "E3[backbone=cnn]",
                                "T4.root_pehe.mean": 0.2},
           "E3[backbone=wide]": {"label": "E3[backbone=wide]",
                                 "T4.root_pehe.mean": 0.1}}
    context = {}
    winner = runner._select_winner(exp, agg)
    runner._propagate(exp, winner, agg, context)
    assert context["ranking"]["E3"] == ["E3[backbone=wide]", "E3[backbone=cnn]"]
    assert context["backbone"] == "wide"
