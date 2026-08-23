"""End-to-end prototype runs of the orchestrator.

These exercise the real training executors, so they require torch and are skipped
where it is not installed (e.g. a numpy-only CI shard). They stay tiny: a single
seed, a 16^3 volume, a handful of synthetic masks and one epoch.
"""
import pytest

pytest.importorskip("torch")

from neurocausalpfn.experiments.runner import run_experiment   # noqa: E402

# Tiny but not degenerate: 8 synth volumes at 50/50 split -> 4 train, batch size 2,
# so no size-1 batch can reach BatchNorm (which needs >1 value per channel), and a
# shallow 2-stage backbone keeps the spatial dims from collapsing to 1x1x1.
FAST = {"resolution": [16, 16, 16], "n_synth": 8, "val_frac": 0.5, "epochs": 1,
        "channels": [8, 16]}


def test_e1_baseline_runs_and_reports_tiers(tmp_path):
    out = run_experiment("E1", mode="prototype", seeds=1, base_seed=0,
                         out_root=str(tmp_path), overrides=FAST)
    assert out["eid"] == "E1"
    agg = out["aggregate"]
    assert agg, "expected at least one aggregated variant"
    entry = next(iter(agg.values()))
    # T1 (reconstruction) is a hard gate. A 1-epoch tiny VAE need not reach
    # Dice >= 0.70, in which case the pipeline correctly stops at T1 and never
    # computes T4 -- so assert the tier machinery ran and recorded a decision,
    # and only require T4 when the T1 gate was actually passed.
    assert "T1.dice.mean" in entry
    assert "passed_frac" in entry
    if entry.get("T1.dice.mean", 0.0) >= 0.70:
        assert "T4.root_pehe.mean" in entry


def test_e7_selects_a_backbone_winner_and_propagates(tmp_path):
    context = {"w_dice": 1.0}
    out = run_experiment("E3", mode="prototype", seeds=1, base_seed=0,
                         out_root=str(tmp_path), overrides=FAST, context=context)
    assert out["winner"] is not None
    # the winning backbone is written back into the shared context for E4, E8...
    assert "backbone" in context
    assert context["backbone"] in {"cnn", "resnet18", "resnet50", "wide"}


def test_gate_short_circuits_are_recorded(tmp_path):
    out = run_experiment("E1", mode="prototype", seeds=1, base_seed=1,
                         out_root=str(tmp_path), overrides=FAST)
    entry = next(iter(out["aggregate"].values()))
    assert "passed_frac" in entry
