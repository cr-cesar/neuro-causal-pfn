"""Metricas de evaluacion de la Etapa 2."""
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
    """Raiz del error cuadratico esperado de heterogeneidad del efecto."""
    p, t = _np(cate_pred).ravel(), _np(cate_true).ravel()
    return float(np.sqrt(np.mean((p - t) ** 2)))


def prescriptive_accuracy(cate_pred, cate_true) -> float:
    """Fraccion de consultas en las que la recomendacion (signo del efecto)
    coincide con la del efecto verdadero."""
    p, t = _np(cate_pred).ravel(), _np(cate_true).ravel()
    return float(np.mean((p > 0) == (t > 0)))


def picp(lo, hi, truth) -> float:
    """Probabilidad de cobertura del intervalo de prediccion."""
    lo, hi, truth = _np(lo).ravel(), _np(hi).ravel(), _np(truth).ravel()
    return float(np.mean((truth >= lo) & (truth <= hi)))
