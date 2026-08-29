"""Manual decisions pinned in the context survive finalize and job races."""
import json
import os

from neurocausalpfn.experiments import runner


PIN = {"backbone": {"value": "resnet", "reason": "incumbent, E3 ties within noise"},
       "winners.E3": {"value": "E3[backbone=resnet]", "reason": "same decision"}}


def test_apply_manual_pins_overrides_automated_winner():
    ctx = {"backbone": "resnet18", "winners": {"E3": "E3[backbone=resnet18]"},
           "manual": PIN}
    runner.apply_manual_pins(ctx)
    assert ctx["backbone"] == "resnet"
    assert ctx["winners"]["E3"] == "E3[backbone=resnet]"


def test_save_context_merges_pins_from_disk(tmp_path):
    out_root = str(tmp_path)
    # a curator records the pin on disk...
    runner.save_context(out_root, {"backbone": "resnet", "manual": PIN})
    # ...meanwhile a long job holds a stale, pin-less context and saves at exit
    stale = {"backbone": "resnet18", "winners": {"E3": "E3[backbone=resnet18]"},
             "dims": (25, 25)}
    runner.save_context(out_root, stale)
    ctx = runner.load_context(out_root)
    assert ctx["backbone"] == "resnet"                    # pin survived the race
    assert ctx["winners"]["E3"] == "E3[backbone=resnet]"
    assert ctx["dims"] == (25, 25)                        # job's real work kept
    assert "manual" in json.load(open(os.path.join(out_root, "context.json")))


def test_load_context_applies_pins():
    # even a hand-edited file with a contradictory value comes back pinned
    ctx = {"backbone": "resnet18", "manual": {"backbone": "resnet"}}
    runner.apply_manual_pins(ctx)
    assert ctx["backbone"] == "resnet"
