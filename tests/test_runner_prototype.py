"""End-to-end prototype runs of the orchestrator.

These exercise the real training executors, so they require torch and are skipped
where it is not installed (e.g. a numpy-only CI shard). They stay tiny: a single
seed, a 16^3 volume, a handful of synthetic masks and one epoch.
"""
import pytest

pytest.importorskip("torch")

from neurocausalpfn.experiments.runner import run_experiment   # noqa: E402

FAST = {"resolution": [16, 16, 16], "n_synth": 6, "val_frac": 0.5, "epochs": 1}


def test_e1_baseline_runs_and_reports_tiers(tmp_path):
    out = run_experiment("E1", mode="prototype", seeds=1, base_seed=0,
                         out_root=str(tmp_path), overrides=FAST)
    assert out["eid"] == "E1"
    agg = out["aggregate"]
    assert agg, "expected at least one aggregated variant"
    entry = next(iter(agg.values()))
    # E1 is judged on T1, T2, T4 -> a root-PEHE mean must be present
    assert "T4.root_pehe.mean" in entry


def test_e7_selects_a_backbone_winner_and_propagates(tmp_path):
    context = {"w_dice": 1.0}
    out = run_experiment("E7", mode="prototype", seeds=1, base_seed=0,
                         out_root=str(tmp_path), overrides=FAST, context=context)
    assert out["winner"] is not None
    # the winning backbone is written back into the shared context for E3, E5a...
    assert "backbone" in context
    assert context["backbone"] in {"cnn", "resnet18", "resnet50", "wide"}


def test_gate_short_circuits_are_recorded(tmp_path):
    out = run_experiment("E1", mode="prototype", seeds=1, base_seed=1,
                         out_root=str(tmp_path), overrides=FAST)
    entry = next(iter(out["aggregate"].values()))
    assert "passed_frac" in entry
