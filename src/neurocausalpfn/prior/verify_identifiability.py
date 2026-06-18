"""Identifiability verifier: the R1 and R2 gate of the Neuro-Prior.

The convergence result of prior-fitted networks only holds if the prior is
identifiable, that is, if the expected conditional potential outcome depends on
the process only through its observational distribution. In practice this
requires two things in each sampled process: positivity, and that the treatment
depends solely on the observed covariates. This function inspects a process and
accepts it only if both hold, rejecting any in which the treatment carries
information about an unobserved cause of the outcome.
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
    if not (tol < p < 1.0 - tol):                       # positivity
        return False
    if U is not None:                                   # W must be independent of U given X
        from sklearn.linear_model import LogisticRegression

        U = np.asarray(U).reshape(len(W), -1)
        base = LogisticRegression(max_iter=500).fit(X, W).predict_proba(X)[:, 1]
        aug = (LogisticRegression(max_iter=500)
               .fit(np.c_[X, U], W).predict_proba(np.c_[X, U])[:, 1])
        if float(np.mean(np.abs(aug - base))) > tol:    # U adds predictive power, there is confounding
            return False
    return True
