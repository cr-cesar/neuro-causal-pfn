"""Runtime helpers for GPU nodes (the cluster uses NVIDIA V100).

The two changes that matter most on a V100 are automatic mixed precision (the
Tensor Cores run float16 matmuls roughly two to three times faster and halve the
activation memory) and cuDNN autotuning for the fixed-size 3D convolutions. These
helpers centralise device resolution, AMP and data-loading so the training loops
opt in with a one-line change. On CPU everything degrades to a no-op: the grad
scaler and autocast are disabled, the loader is unpinned and single-process, so
the numerics and the CPU test results are unchanged.
"""
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from .logging_utils import get_logger

log = get_logger()


def resolve_device(cfg) -> str:
    """Resolves the device from cfg ('auto', 'cuda', 'cpu'); enables cuDNN
    autotuning on CUDA, which speeds up the fixed-size 3D convolutions."""
    dev = cfg.get("device", "auto")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if str(dev).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    elif str(dev).startswith("cuda"):
        dev = "cpu"   # requested CUDA but none available
    return dev


def use_amp(cfg, device) -> bool:
    """Mixed precision is used on CUDA when cfg['amp'] is set (default True)."""
    return bool(cfg.get("amp", True)) and str(device).startswith("cuda")


def make_grad_scaler(enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled)


def autocast_ctx(device, enabled: bool):
    if enabled and str(device).startswith("cuda"):
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def optim_step(loss, opt, scaler, params=None, grad_clip=None):
    """AMP-aware optimiser step. With the scaler disabled (CPU) this is exactly
    zero_grad, backward, optional gradient clipping, step."""
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if grad_clip is not None and params is not None:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
    scaler.step(opt)
    scaler.update()


def make_loader(dataset, batch_size, shuffle, device, num_workers: int = 0):
    """DataLoader with V100-friendly settings: pinned memory and worker
    processes on CUDA, plain single-process loading on CPU."""
    pin = str(device).startswith("cuda")
    workers = int(num_workers) if pin else 0
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=pin,
                      persistent_workers=workers > 0)


def log_runtime(name: str, device: str, amp: bool):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("%s on %s (%s, %.0f GB), mixed precision=%s", name, device, gpu, mem, amp)
    else:
        log.info("%s on %s, mixed precision=%s", name, device, amp)
