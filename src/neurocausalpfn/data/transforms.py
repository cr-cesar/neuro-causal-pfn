"""Transformaciones de volumen: relleno o recorte a una forma objetivo y
binarizacion. Operan sobre arreglos numpy [D, H, W]."""
from typing import Tuple

import numpy as np


def pad_or_crop(vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Centra el volumen en una caja de la forma objetivo, rellenando con ceros
    o recortando segun haga falta en cada eje."""
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
