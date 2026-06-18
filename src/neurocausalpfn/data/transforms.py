"""Volume transformations: padding or cropping to a target shape and
binarization. They operate on numpy arrays [D, H, W]."""
from typing import Tuple

import numpy as np


def pad_or_crop(vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Centers the volume in a box of the target shape, padding with zeros or
    cropping as needed along each axis."""
    target = tuple(int(t) for t in target)
    out = np.zeros(target, dtype=vol.dtype)
    src_slices, dst_slices = [], []
    for s, t in zip(vol.shape, target):
        if s >= t:
            start = (s - t) // 2
            src_slices.append(slice(start, start + t))
            dst_slices.append(slice(0, t))
        else:
            start = (t - s) // 2
            src_slices.append(slice(0, s))
            dst_slices.append(slice(start, start + s))
    out[tuple(dst_slices)] = vol[tuple(src_slices)]
    return out


def binarize(vol: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (vol > threshold).astype(np.float32)
