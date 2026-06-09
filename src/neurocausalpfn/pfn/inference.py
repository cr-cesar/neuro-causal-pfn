"""Inferencia con el transformer entrenado.

Para un contexto observado y un conjunto de consultas, evalua la distribucion
predicha bajo tratamiento y bajo control. La media de cada una es la estimacion
puntual del resultado potencial esperado condicional, su diferencia es el efecto
del tratamiento individualizado, y los cuantiles de la distribucion por bins dan
intervalos creibles.
"""
from typing import Dict

import torch


@torch.no_grad()
def predict_cate(model, Xc: torch.Tensor, Tc: torch.Tensor, Yc: torch.Tensor,
                 Xq: torch.Tensor) -> Dict[str, torch.Tensor]:
    model.eval()
    B, n_qry, _ = Xq.shape
    t0 = torch.zeros(B, n_qry, device=Xq.device, dtype=Xq.dtype)
    t1 = torch.ones(B, n_qry, device=Xq.device, dtype=Xq.dtype)
    logits0 = model(Xc, Tc, Yc, Xq, t0)
    logits1 = model(Xc, Tc, Yc, Xq, t1)
    mu0 = model.head.mean(logits0)
    mu1 = model.head.mean(logits1)
    return {"mu0": mu0, "mu1": mu1, "cate": mu1 - mu0,
            "logits0": logits0, "logits1": logits1}


@torch.no_grad()
def quantile_from_logits(head, logits: torch.Tensor, p: float) -> torch.Tensor:
    """Cuantil de nivel p de la distribucion predicha por bins."""
    probs = torch.softmax(logits, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    idx = torch.argmax((cdf >= p).to(torch.float32), dim=-1)  # primer bin que supera p
    return head.centers[idx]


@torch.no_grad()
def credible_interval(head, logits: torch.Tensor, lo: float = 0.05, hi: float = 0.95):
    return quantile_from_logits(head, logits, lo), quantile_from_logits(head, logits, hi)
