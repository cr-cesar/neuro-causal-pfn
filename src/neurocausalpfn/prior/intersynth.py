"""Generador sintetico estilo InterSynth adaptado a los latentes del VAE.

Cada proceso generador de datos muestrea un modelo de resultado bajo control y
bajo tratamiento, y un modelo de propension que depende solo de las covariables
observadas, lo que garantiza la ignorabilidad por construccion. El efecto del
tratamiento es heterogeneo. La confusion se induce a traves de las covariables
observadas correlacionando la propension con el pronostico, segun el mecanismo
elegido (gravedad, localizacion, red o una mezcla).

Los resultados potenciales esperados condicionales mu_0(x) y mu_1(x) son
sigmoides, por lo que viven en el intervalo unidad y sirven tanto de objetivo
continuo en el entrenamiento como de probabilidad de buen desenlace en el
despliegue.
"""
from typing import Dict

import numpy as np

MECHANISMS = ("severity", "location", "network", "mixed")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class SyntheticDGP:
    def __init__(self, d_x: int, rng: np.random.Generator,
                 mechanism: str = "mixed", confound_strength: float = 1.0):
        self.d_x = int(d_x)
        self.mechanism = mechanism
        scale = 1.0 / np.sqrt(self.d_x)
        # modelo de resultado bajo control y efecto heterogeneo del tratamiento
        self.w0 = rng.normal(0.0, 1.0, self.d_x) * scale
        self.b0 = float(rng.normal(0.0, 0.5))
        self.delta = rng.normal(0.0, 1.0, self.d_x) * scale
        self.b1 = self.b0 + float(rng.normal(0.0, 0.5))
        # la propension depende solo de X (ignorable). La confusion observada se
        # induce alineando parte de w_prop con el pronostico.
        if mechanism in ("severity", "mixed"):
            direction = self.w0.copy()
        else:
            direction = rng.normal(0.0, 1.0, self.d_x) * scale
        self.w_prop = confound_strength * direction + rng.normal(0.0, 1.0, self.d_x) * scale
        self.b_prop = float(rng.normal(0.0, 0.3))
        self.noise = 0.05

    def mu(self, X: np.ndarray, t: int) -> np.ndarray:
        w = self.w0 + (self.delta if t == 1 else 0.0)
        b = self.b1 if t == 1 else self.b0
        return _sigmoid(X @ w + b)

    def propensity(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.w_prop + self.b_prop)

    def sample_observed(self, X: np.ndarray, rng: np.random.Generator):
        e = self.propensity(X)
        t = (rng.uniform(size=len(X)) < e).astype(np.float64)
        mu_t = np.where(t == 1, self.mu(X, 1), self.mu(X, 0))
        y = np.clip(mu_t + rng.normal(0.0, self.noise, size=len(X)), 0.0, 1.0)
        return t, y


def sample_covariates(n: int, d_x: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=(n, d_x))


def make_dataset(dgp: SyntheticDGP, n_context: int, n_query: int,
                 rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Construye un contexto observacional y un conjunto de consultas con el
    resultado potencial esperado condicional verdadero para el tratamiento
    consultado, mas mu_0 y mu_1 para evaluar el efecto del tratamiento."""
    Xc = sample_covariates(n_context, dgp.d_x, rng)
    Tc, Yc = dgp.sample_observed(Xc, rng)
    Xq = sample_covariates(n_query, dgp.d_x, rng)
    Tq = rng.integers(0, 2, size=n_query).astype(np.float64)
    mu0 = dgp.mu(Xq, 0)
    mu1 = dgp.mu(Xq, 1)
    mu_q = np.where(Tq == 1, mu1, mu0)
    return {"Xc": Xc, "Tc": Tc, "Yc": Yc, "Xq": Xq, "Tq": Tq,
            "mu_q": mu_q, "mu0": mu0, "mu1": mu1}
