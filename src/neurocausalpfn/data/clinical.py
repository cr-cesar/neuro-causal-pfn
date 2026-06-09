"""Covariables clinicas que acompanan a cada lesion.

En modo completo se leerian de una tabla alineada con las mascaras (edad,
NIHSS, lateralidad, tiempo hasta el tratamiento, etc.). En modo prototipo se
sintetizan para que la tuberia corra sin datos reales.
"""
from typing import Optional

import numpy as np


def synthesize_clinical(n: int, d: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)


def load_clinical(path: Optional[str], n: int, d: int = 4, seed: int = 0) -> np.ndarray:
    """Lee una tabla de covariables si existe, en caso contrario sintetiza."""
    if path:
        import pandas as pd

        table = pd.read_csv(path)
        return table.to_numpy(dtype=np.float32)
    return synthesize_clinical(n, d, seed)
