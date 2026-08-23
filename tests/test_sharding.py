"""Sharded execution: --only filters variants and defers winner selection;
finalize_experiment re-derives the winner from runs.jsonl and updates the
cross-job context; load/save_context round-trips through out_root."""
import json
import os

import pytest

from neurocausalpfn.experiments import runner


def test_only_filters_variants_and_skips_winner(tmp_path):
    runner._MODALITY_CACHE.clear()
    out = runner.run_experiment(
        "E7", mode="prototype", seeds=1, out_root=str(tmp_path), only="cnn",
        overrides={"epochs": 1, "n_synth": 8, "resolution": [24, 28, 24],
                   "batch_size": 4, "channels": [8, 16, 32, 64]})
    assert list(out["aggregate"]) == ["E7[backbone=cnn]"]
    assert out["winner"] is None            # deferred to --finalize


def test_only_with_no_match_raises(tmp_path):
    with pytest.raises(ValueError):
        runner.run_experiment("E7", mode="prototype", seeds=1,
                              out_root=str(tmp_path), only="no-such-backbone")


def test_finalize_selects_winner_from_history(tmp_path):
    rows = []
    for backbone, pehe in (("cnn", 0.20), ("resnet18", 0.10)):
        for seed in (0, 1):
            rows.append({"label": f"E7[backbone={backbone}]", "seed": seed,
                         "T4.root_pehe": pehe + 0.01 * seed, "T1.dice": 0.8,
                         "passed": True, "meta.backbone": backbone})
    with open(os.path.join(tmp_path, "runs.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")

    context = {}
    out = runner.finalize_experiment("E7", context, out_root=str(tmp_path))
    assert out["winner"]["label"] == "E7[backbone=resnet18]"   # lower root-PEHE
    assert context["backbone"] == "resnet18"                   # propagated

    runner.save_context(str(tmp_path), context)
    assert runner.load_context(str(tmp_path))["backbone"] == "resnet18"


def test_context_dims_round_trip(tmp_path):
    runner.save_context(str(tmp_path), {"dims": (75, 25)})
    assert runner.load_context(str(tmp_path))["dims"] == (75, 25)
