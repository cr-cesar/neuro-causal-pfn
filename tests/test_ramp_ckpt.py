"""The fixed context ramp and the intermediate checkpoints of the PFN trainer (3 Nov)."""
import os

import torch

from neurocausalpfn.train.launch_cfg import cfg_from_env
from neurocausalpfn.train.train_pfn import _context_length, prototype_config, run_pfn


def _cfg(iters, **pfn):
    cfg = {"seed": 0, "pfn": {"context_min": 256, "context_max": 3000, "iters": iters, **pfn}}
    return cfg


def test_default_ramp_is_half_of_training():
    # the final model: 250,000 steps, ramp over the first 125,000
    cfg = _cfg(250000)
    assert _context_length(cfg, 0) == _context_length(_cfg(250000, ramp_iters=125000), 0)
    for it in (0, 1000, 62499, 124999, 125000, 249999):
        assert _context_length(cfg, it) == _context_length(_cfg(375000, ramp_iters=125000), it)
    assert _context_length(cfg, 124999) == 3000


def test_fixed_ramp_does_not_scale_with_length():
    short, long = _cfg(250000, ramp_iters=125000), _cfg(375000, ramp_iters=125000)
    assert _context_length(long, 124999) == 3000
    assert _context_length(_cfg(375000), 124999) < 3000      # default: the first half, 187,500
    assert [_context_length(short, i) for i in range(0, 250000, 5000)] == \
           [_context_length(long, i) for i in range(0, 250000, 5000)]


def test_env_sets_ramp_and_checkpoints():
    cfg = cfg_from_env({"RAMP_ITERS": "125000", "CKPT_EVERY": "62500", "ITERS": "375000"})
    assert cfg["pfn"]["ramp_iters"] == 125000 and cfg["pfn"]["ckpt_every"] == 62500
    cfg = cfg_from_env({})
    assert "ramp_iters" not in cfg["pfn"] and "ckpt_every" not in cfg["pfn"]


def test_intermediate_checkpoints(tmp_path):
    cfg = prototype_config()
    cfg["out_dir"] = str(tmp_path)
    cfg["device"] = "cpu"
    cfg["pfn"]["iters"] = 6
    cfg["pfn"]["ckpt_every"] = 2
    run_pfn(cfg)
    names = sorted(f for f in os.listdir(tmp_path) if f.endswith(".pt"))
    assert names == ["pfn.pt", "pfn_it2.pt", "pfn_it4.pt"]
    assert torch.load(os.path.join(tmp_path, "pfn_it4.pt"), map_location="cpu")["updates"] == 4
    assert torch.load(os.path.join(tmp_path, "pfn.pt"), map_location="cpu")["updates"] == 6
