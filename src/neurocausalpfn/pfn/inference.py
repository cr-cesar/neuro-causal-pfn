"""Inference with the trained transformer.

For an observed context and a set of queries, it evaluates the predicted
distribution under treatment and under control. The mean of each one is the point
estimate of the expected conditional potential outcome, their difference is the
individualized treatment effect, and the quantiles of the binned distribution
give credible intervals.
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
    """Level-p quantile of the binned predicted distribution."""
    probs = torch.softmax(logits, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    idx = torch.argmax((cdf >= p).to(torch.float32), dim=-1)  # first bin that exceeds p
    return head.centers[idx]


@torch.no_grad()
def credible_interval(head, logits: torch.Tensor, lo: float = 0.05, hi: float = 0.95):
    return quantile_from_logits(head, logits, lo), quantile_from_logits(head, logits, hi)
