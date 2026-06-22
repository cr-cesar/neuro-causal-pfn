"""Tier 2 evaluation: clinical alignment via probes.

A frozen encoder is judged by how well a simple probe predicts clinical scores
(NIHSS, mRS) from the latent alone. A low score means the encoder discarded
clinical information during compression. Reported globally and, for the equity
question, per subgroup (e.g. vascular territory, lesion-volume quartile, sex).
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor


def _metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    rho = spearmanr(y_true, y_pred).correlation if y_true.size > 1 else float("nan")
    return {"r2": float(r2_score(y_true, y_pred)),
            "spearman": float(rho),
            "mae": float(mean_absolute_error(y_true, y_pred))}


def _estimator(kind: str, hidden: int, seed: int, max_iter: int):
    if kind == "mlp":
        return MLPRegressor(hidden_layer_sizes=(hidden,), random_state=seed, max_iter=max_iter)
    return LinearRegression()


def _held_out_pred(Z, y, kind, n_splits, seed, hidden, max_iter):
    Z, y = np.asarray(Z, dtype=np.float64), np.asarray(y, dtype=np.float64)
    k = min(n_splits, Z.shape[0])
    cv = KFold(n_splits=max(2, k), shuffle=True, random_state=seed)
    est = _estimator(kind, hidden, seed, max_iter)
    return Z, y, cross_val_predict(est, Z, y, cv=cv)


def linear_probe(Z, y, n_splits: int = 5, seed: int = 0) -> dict:
    """Cross-validated linear probe predicting y from Z. Returns R2, Spearman, MAE."""
    _, y, pred = _held_out_pred(Z, y, "linear", n_splits, seed, 64, 500)
    return _metrics(y, pred)


def mlp_probe(Z, y, hidden: int = 64, n_splits: int = 5, seed: int = 0, max_iter: int = 500) -> dict:
    """Cross-validated MLP probe (one hidden layer) predicting y from Z."""
    _, y, pred = _held_out_pred(Z, y, "mlp", n_splits, seed, hidden, max_iter)
    return _metrics(y, pred)


def stratified_probe(Z, y, groups, kind: str = "linear", n_splits: int = 5,
                     seed: int = 0, hidden: int = 64, max_iter: int = 500) -> dict:
    """Probe trained with cross-validation on all patients, then scored overall and
    within each subgroup (the held-out predictions are reused per group). Returns a
    dict with one entry per group plus 'all'."""
    Z, y, pred = _held_out_pred(Z, y, kind, n_splits, seed, hidden, max_iter)
    groups = np.asarray(groups)
    out = {"all": _metrics(y, pred)}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 2:
            out[str(g)] = _metrics(y[m], pred[m])
    return out
