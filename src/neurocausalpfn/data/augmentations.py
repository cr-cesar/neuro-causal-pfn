"""Binary-coherent augmentations for the contrastive arm (Arm C).

Physics-based MRI augmentations (Rician noise, k-space artefacts, bias fields)
exit the {0,1} manifold of a binary lesion mask, so the views are built only
from transformations that keep each modality on its own manifold:

- binary masks: axis flips, small integer-voxel shifts, one-step morphological
  dilation or erosion, and territory-style block masking; the output stays in
  {0, 1};
- continuous disconnectomes in [0, 1]: the same geometric transformations plus
  a mild global intensity scaling, clipped back to [0, 1].

The augmentation set is part of the hypothesis being tested, not a detail: a
contrastive objective only recovers useful structure when the augmentation
graph aligns with the task (Saunshi et al. 2022).
"""
from typing import Optional, Tuple

import numpy as np
import torch

try:                                    # scipy is a core dependency, but keep a
    from scipy import ndimage           # pure-numpy fallback for morphology-free
    _HAS_SCIPY = True                   # environments
except ImportError:                     # pragma: no cover
    _HAS_SCIPY = False


def _shift_zero_fill(vol: np.ndarray, shifts: Tuple[int, int, int]) -> np.ndarray:
    """Integer-voxel translation with zero fill (no wrap-around)."""
    out = np.zeros_like(vol)
    src, dst = [], []
    for size, sh in zip(vol.shape, shifts):
        if abs(sh) >= size:
            return out
        if sh >= 0:
            src.append(slice(0, size - sh))
            dst.append(slice(sh, size))
        else:
            src.append(slice(-sh, size))
            dst.append(slice(0, size + sh))
    out[tuple(dst)] = vol[tuple(src)]
    return out


def territory_mask(vol: np.ndarray, rng: np.random.Generator,
                   frac: float = 0.25) -> np.ndarray:
    """Zero out one contiguous block covering roughly ``frac`` of the volume.

    A cheap stand-in for masking a vascular territory: the block is axis-aligned
    and contiguous, so the removed region is spatially coherent rather than
    salt-and-pepper dropout.
    """
    out = vol.copy()
    side = float(np.clip(frac, 0.0, 1.0)) ** (1.0 / 3.0)
    starts, stops = [], []
    for size in vol.shape:
        ext = max(1, int(round(side * size)))
        start = int(rng.integers(0, max(1, size - ext + 1)))
        starts.append(start)
        stops.append(min(size, start + ext))
    out[starts[0]:stops[0], starts[1]:stops[1], starts[2]:stops[2]] = 0.0
    return out


def augment_volume(vol: np.ndarray, binary: bool = True,
                   rng: Optional[np.random.Generator] = None,
                   p_flip: float = 0.5, max_shift: int = 3,
                   p_morph: float = 0.3, p_territory: float = 0.2) -> np.ndarray:
    """One augmented view of a single volume [D, H, W].

    Every operation is manifold-preserving: a binary input returns a binary
    output and a [0, 1] input stays in [0, 1].
    """
    rng = np.random.default_rng() if rng is None else rng
    out = vol.astype(np.float32, copy=True)

    # left-right hemispheric flip (axis 0 in the MNI-ordered array)
    if rng.random() < p_flip:
        out = np.flip(out, axis=0).copy()

    # small integer-voxel translation, zero-filled
    if max_shift > 0:
        shifts = tuple(int(rng.integers(-max_shift, max_shift + 1)) for _ in range(3))
        if any(shifts):
            out = _shift_zero_fill(out, shifts)

    if binary:
        # one-step morphological perturbation of the lesion boundary
        if _HAS_SCIPY and rng.random() < p_morph and out.sum() > 0:
            if rng.random() < 0.5:
                out = ndimage.binary_dilation(out > 0.5).astype(np.float32)
            else:
                eroded = ndimage.binary_erosion(out > 0.5).astype(np.float32)
                if eroded.sum() > 0:           # never erase the lesion entirely
                    out = eroded
        out = (out > 0.5).astype(np.float32)
    else:
        # mild global intensity scaling, clipped back onto [0, 1]
        out = np.clip(out * float(rng.uniform(0.9, 1.1)), 0.0, 1.0)

    # territory-style contiguous block masking
    if rng.random() < p_territory:
        out = territory_mask(out, rng, frac=float(rng.uniform(0.05, 0.25)))
    return out


def augment_batch(batch: torch.Tensor, binary: bool = True,
                  seed: Optional[int] = None) -> torch.Tensor:
    """Augment a batch [B, C, D, H, W], returning a tensor on the same device.

    The transformations are integer-voxel numpy operations, so the batch is
    round-tripped through the CPU; at prototype resolutions this is negligible
    and at full resolution the loaders already stage batches on the CPU.
    """
    rng = np.random.default_rng(seed)
    arr = batch.detach().cpu().numpy()
    out = np.empty_like(arr, dtype=np.float32)
    for b in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[b, c] = augment_volume(arr[b, c], binary=binary, rng=rng)
    return torch.from_numpy(out).to(device=batch.device, dtype=batch.dtype)
