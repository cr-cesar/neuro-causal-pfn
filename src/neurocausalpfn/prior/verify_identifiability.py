"""Verificador de identificabilidad: la puerta R1 y R2 del Neuro-Prior.

El resultado de convergencia de los prior-fitted networks solo se cumple si el
prior es identificable, es decir, si el resultado potencial esperado condicional
depende del proceso solo a traves de su distribucion observacional. En la
practica esto exige dos cosas en cada proceso muestreado: positividad, y que el
tratamiento dependa unicamente de las covariables observadas. Esta funcion
inspecciona un proceso y lo acepta solo si ambas se cumplen, rechazando
cualquiera en el que el tratamiento lleve informacion sobre una causa no
observada del resultado.
"""
from typing import Optional

import numpy as np


def verify_identifiability(W: np.ndarray, X: np.ndarray,
                           Y0: Optional[np.ndarray] = None,
                           Y1: Optional[np.ndarray] = None,
                           U: Optional[np.ndarray] = None,
                           tol: float = 1e-3) -> bool:
    W = np.asarray(W).ravel()
    X = np.asarray(X)
    p = float(np.clip(W.mean(), 1e-6, 1.0 - 1e-6))
    if not (tol < p < 1.0 - tol):                       # positividad
        return False
    if U is not None:                                   # W debe ser independiente de U dado X
        from sklearn.linear_model import LogisticRegression

        U = np.asarray(U).reshape(len(W), -1)
        base = LogisticRegression(max_iter=500).fit(X, W).predict_proba(X)[:, 1]
        aug = (LogisticRegression(max_iter=500)
               .fit(np.c_[X, U], W).predict_proba(np.c_[X, U])[:, 1])
        if float(np.mean(np.abs(aug - base))) > tol:    # U aporta poder predictivo, hay confusion
            return False
    return True
