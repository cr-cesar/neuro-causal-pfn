import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from neurocausalpfn.utils.runtime import (autocast_ctx, make_grad_scaler,
                                          make_loader, optim_step,
                                          resolve_device, use_amp)


def test_resolve_device_without_cuda():
    # no CUDA in the test environment: every request resolves to cpu
    assert resolve_device({"device": "cpu"}) == "cpu"
    assert resolve_device({"device": "auto"}) == "cpu"
    assert resolve_device({"device": "cuda"}) == "cpu"


def test_amp_disabled_on_cpu():
    assert use_amp({"amp": True}, "cpu") is False
    assert use_amp({"amp": False}, "cpu") is False


def test_autocast_is_noop_on_cpu():
    x = torch.randn(2, 2)
    with autocast_ctx("cpu", True):
        y = x @ x
    assert y.dtype == torch.float32


def test_optim_step_updates_parameters():
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = make_grad_scaler(False)        # disabled -> plain step
    x, y = torch.randn(8, 4), torch.randn(8, 1)
    before = model.weight.detach().clone()
    optim_step(((model(x) - y) ** 2).mean(), opt, scaler)
    assert not torch.allclose(before, model.weight)


def test_optim_step_with_grad_clip_runs():
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = make_grad_scaler(False)
    x, y = torch.randn(8, 4), torch.randn(8, 1)
    optim_step(((model(x) - y) ** 2).mean(), opt, scaler,
               params=model.parameters(), grad_clip=1.0)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_make_loader_cpu_is_unpinned_single_process():
    ds = TensorDataset(torch.randn(6, 3))
    dl = make_loader(ds, batch_size=2, shuffle=False, device="cpu", num_workers=4)
    assert dl.num_workers == 0 and dl.pin_memory is False
    assert len(list(dl)) == 3
