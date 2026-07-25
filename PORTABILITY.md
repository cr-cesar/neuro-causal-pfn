# Portability: running the pipeline on a laptop, a workstation, and the cluster

The goal is that the code starts cleanly and *observably* everywhere, so a
reviewer on a Mac sees the same behaviour the cluster does — just slower. The
cluster performance path (mixed precision, cuDNN autotuning) lives in
`utils/runtime.py`; this note covers the laptop / CPU / Apple-Silicon side.

## The three failure modes on a Mac, and the fixes

1. **OpenMP / MKL duplicate-runtime abort** (`OMP: Error #15`). Apple and
   PyTorch/MKL each load an OpenMP runtime. Fix: `KMP_DUPLICATE_LIB_OK=TRUE`.
2. **A slow run that looks hung.** 3D convolutions on CPU/MPS are simply slow;
   the process is progressing, not stuck. Fix: a heartbeat log (below) and
   `OMP_NUM_THREADS=1` to avoid FAISS/OpenMP oversubscription deadlocks.
3. **MPS kernels erroring on unsupported 3D ops.** Fix: prefer CPU on Apple
   Silicon by default (3D-conv on MPS is still unreliable), and set
   `PYTORCH_ENABLE_MPS_FALLBACK=1` so any op that *is* sent to MPS falls back to
   CPU instead of raising.

## Two ways to apply the flags (pick one)

**A. Zero code change — source the script before running:**

```bash
source scripts/portable_env.sh
python -m neurocausalpfn.train.train_vae ...
```

**B. In-process — call it first in an entry point (before importing torch):**

```python
from neurocausalpfn.utils.portability import configure_portable_runtime
configure_portable_runtime()          # only sets vars that are unset
import torch                          # noqa: E402
```

Both are no-ops on the cluster if the variables are already set by your job
script, so an explicit cluster setting always wins.

## Device selection

`utils/portability.resolve_device(prefer="auto", allow_mps=False)` returns
`cuda` if present, then `cpu`. Pass `allow_mps=True` to opt into Metal on Apple
Silicon once you have verified the 3D ops you need are supported; until then CPU
is the safe default. This complements `utils/runtime.resolve_device`, which
handles the `cuda`/`cpu` cluster path.

## Seeing progress on a slow run

Wrap a slow stage so it prints a heartbeat instead of looking hung:

```python
from neurocausalpfn.utils.portability import heartbeat
with heartbeat("stage-1 epoch", interval=30):
    train_one_epoch(...)
```

## FAISS on Apple Silicon

`faiss-cpu` is a `baselines` extra (needed by `causalpfn`). On Apple Silicon
install it through **conda**, not pip, to get a wheel built against a compatible
OpenMP:

```bash
conda install -c pytorch faiss-cpu=1.8.0
```

## Quick self-check

```bash
python -m neurocausalpfn.utils.portability
```

prints the platform, torch version, resolved device and thread counts — useful
evidence when answering "will it run on the server?".
