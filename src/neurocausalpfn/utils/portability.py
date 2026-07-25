"""Portability helpers so the pipeline starts the same on a laptop (macOS / CPU),
a Linux workstation, and the cluster (NVIDIA V100).

The cluster path is handled by ``utils/runtime.py`` (automatic mixed precision and
cuDNN autotuning). This module covers the *other* end: making the code start
cleanly and observably on Apple Silicon and CPU, where the usual failure modes are
(a) an OpenMP / MKL duplicate-runtime abort ("OMP: Error #15"), (b) 3D
convolutions that are simply slow and look "hung", and (c) Metal (MPS) kernels
that error on unsupported 3D ops.

Nothing here changes numerics. ``configure_portable_runtime`` only sets a few
environment variables, and only where they are unset, so an explicit user or
cluster setting always wins. It must run *before* torch / faiss are imported to
take effect, so call it first in an entry point::

    from neurocausalpfn.utils.portability import configure_portable_runtime
    configure_portable_runtime()
    import torch  # noqa: E402

Or, with zero code changes, ``source scripts/portable_env.sh`` before running.
"""
from __future__ import annotations

import os
import platform
import threading
import time
from contextlib import contextmanager

try:  # reuse the project logger if available, else fall back to stdlib
    from .logging_utils import get_logger
    _log = get_logger()
except Exception:  # pragma: no cover - fallback only
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    _log = logging.getLogger("neurocausalpfn")


# Applied only when the variable is unset, so an explicit user/cluster setting
# always wins over these defaults.
_PORTABLE_ENV_DEFAULTS = {
    # Apple ships its own OpenMP; PyTorch/MKL ship theirs. Loading both aborts
    # unless this is set. Harmless on Linux/CUDA.
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    # One OpenMP thread avoids oversubscription/deadlocks with FAISS on macOS and
    # keeps CPU runs deterministic. Raise it on a big CPU box for throughput.
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    # If MPS is ever selected, let unsupported 3D ops fall back to CPU instead of
    # raising, so a run completes (slowly) rather than crashing.
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
}


def configure_portable_runtime(omp_num_threads: int | None = None,
                               verbose: bool = True) -> dict:
    """Set portability environment defaults (only where unset). Returns the dict
    of variables this call actually set. Call before importing torch / faiss."""
    if omp_num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(omp_num_threads)
    applied = {}
    for key, value in _PORTABLE_ENV_DEFAULTS.items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    if verbose and applied:
        _log.info("portability: set %s",
                  ", ".join(f"{k}={v}" for k, v in applied.items()))
    return applied


def resolve_device(prefer: str = "auto", allow_mps: bool = False) -> str:
    """Resolve a torch device string across platforms.

    ``prefer`` is 'auto' | 'cuda' | 'mps' | 'cpu'. 'auto' picks CUDA if present,
    then MPS only when ``allow_mps`` is set (3D-conv on MPS is still unreliable, so
    the safe default on Apple Silicon is CPU), then CPU. An unavailable explicit
    request degrades to CPU with a warning rather than raising."""
    import torch
    has_cuda = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    has_mps = mps_backend is not None and mps_backend.is_available()

    if prefer == "cuda":
        if has_cuda:
            return "cuda"
        _log.warning("portability: CUDA requested but unavailable; using CPU")
        return "cpu"
    if prefer == "mps":
        if has_mps:
            return "mps"
        _log.warning("portability: MPS requested but unavailable; using CPU")
        return "cpu"
    if prefer == "cpu":
        return "cpu"
    # auto
    if has_cuda:
        return "cuda"
    if has_mps and allow_mps:
        return "mps"
    return "cpu"


def describe_runtime(device: str | None = None) -> dict:
    """Log and return a small dict describing the runtime, for reproducibility and
    for diagnosing "will it run on the server?" questions."""
    import torch
    mps_backend = getattr(torch.backends, "mps", None)
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": mps_backend is not None and mps_backend.is_available(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "torch_threads": torch.get_num_threads(),
    }
    if device is not None:
        info["device"] = device
    if info["cuda_available"]:
        info["gpu"] = torch.cuda.get_device_name(0)
    _log.info("runtime: %s", ", ".join(f"{k}={v}" for k, v in info.items()))
    return info


@contextmanager
def heartbeat(label: str, interval: float = 30.0):
    """Emit a log line every ``interval`` seconds while the block runs, so a slow
    stage (for example 3D convolutions on CPU/MPS) is visibly progressing rather
    than apparently hung. The heartbeat thread is a daemon and never blocks exit.

    Usage::

        with heartbeat("stage-1 epoch", interval=30):
            train_one_epoch(...)
    """
    stop = threading.Event()
    start = time.time()

    def _beat():
        while not stop.wait(interval):
            _log.info("%s: still running (%.0fs elapsed)", label, time.time() - start)

    thread = threading.Thread(target=_beat, name="heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        _log.info("%s: done (%.0fs)", label, time.time() - start)


if __name__ == "__main__":  # `python -m neurocausalpfn.utils.portability`
    configure_portable_runtime()
    try:
        describe_runtime(resolve_device("auto"))
    except Exception as exc:  # torch not importable yet
        _log.info("portability configured; torch not importable yet (%s)", exc)
