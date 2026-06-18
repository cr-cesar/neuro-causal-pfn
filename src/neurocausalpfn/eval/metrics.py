"""Stage 2 evaluation metrics."""
import numpy as np


def _np(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def root_pehe(cate_pred, cate_true) -> float:
    """Root of the expected squared error in treatment effect heterogeneity (PEHE)."""
    p, t = _np(cate_pred).ravel(), _np(cate_true).ravel()
    return float(np.sqrt(np.mean((p - t) ** 2)))


def prescriptive_accuracy(cate_pred, cate_true) -> float:
    """Fraction of queries in which the recommendation (sign of the effect)
    matches that of the true effect."""
    p, t = _np(cate_pred).ravel(), _np(cate_true).ravel()
    return float(np.mean((p > 0) == (t > 0)))


def picp(lo, hi, truth) -> float:
    """Prediction interval coverage probability."""
    lo, hi, truth = _np(lo).ravel(), _np(hi).ravel(), _np(truth).ravel()
    return float(np.mean((truth >= lo) & (truth <= hi)))
