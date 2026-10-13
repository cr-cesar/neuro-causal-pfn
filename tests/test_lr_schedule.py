"""The learning-rate schedule of run_pfn: constant by default (the pilots'
behaviour), warmup then cosine decay when asked."""
import math

import pytest

from neurocausalpfn.train.train_pfn import _lr_schedule


def test_constant_is_the_default():
    lr_at = _lr_schedule({"lr": 2e-4, "iters": 100})
    assert lr_at(0) == lr_at(50) == lr_at(99) == 2e-4


def test_cosine_warms_up_then_decays_to_the_floor():
    p = {"lr": 1e-3, "iters": 1000, "lr_schedule": "cosine", "warmup": 100, "min_lr_frac": 0.1}
    lr_at = _lr_schedule(p)
    assert lr_at(0) == pytest.approx(1e-5)                 # first step: lr / warmup
    assert lr_at(99) == pytest.approx(1e-3)                # end of warmup: full lr
    assert lr_at(100) == pytest.approx(1e-3)               # cosine starts at lr
    assert lr_at(550) == pytest.approx(0.1e-3 + 0.5 * 0.9e-3 * (1 + math.cos(math.pi * 0.5)), rel=1e-3)
    assert lr_at(999) == pytest.approx(1e-4, rel=1e-3)     # floor at the last iteration
    assert all(lr_at(i) >= lr_at(i + 1) for i in range(100, 999))  # monotone after warmup


def test_defaults_and_unknown_kind():
    lr_at = _lr_schedule({"lr": 1e-3, "iters": 1000, "lr_schedule": "cosine"})
    assert lr_at(19) == pytest.approx(1e-3)                # default warmup: 2 % = 20 iterations
    with pytest.raises(ValueError):
        _lr_schedule({"lr": 1e-3, "iters": 10, "lr_schedule": "linear"})
