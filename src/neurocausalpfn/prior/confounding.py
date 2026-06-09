"""Mecanismos de confusion y un generador con confusion no observada.

El generador principal (intersynth) produce procesos ignorables por
construccion. Este modulo aporta el contraejemplo deliberado: un proceso en el
que el tratamiento depende de un confundidor no observado U que tambien afecta
al resultado. Sirve para comprobar que el verificador de identificabilidad lo
rechaza, que es la prueba que operacionaliza el requisito de convergencia.
"""
import numpy as np

from .intersynth import _sigmoid

MECHANISMS = ("severity", "location", "network", "mixed")


def make_unobserved_confounded(d_x: int, n: int, rng: np.random.Generator,
                               strength: float = 2.0):
    """Devuelve (W, X, Y0, Y1, U) con W dependiente de U ademas de X."""
    scale = 1.0 / np.sqrt(d_x)
    X = rng.normal(0.0, 1.0, size=(n, d_x))
    U = rng.normal(0.0, 1.0, size=(n, 1))
    w = rng.normal(0.0, 1.0, d_x) * scale
    # la propension depende de U, lo que viola la ignorabilidad
    e = _sigmoid(X @ w + strength * U[:, 0])
    W = (rng.uniform(size=n) < e).astype(np.float64)
    base = X @ w
    Y0 = np.clip(_sigmoid(base + 0.5 * U[:, 0]), 0.0, 1.0)
    Y1 = np.clip(_sigmoid(base + 0.4 + 0.5 * U[:, 0]), 0.0, 1.0)
    return W, X, Y0, Y1, U
