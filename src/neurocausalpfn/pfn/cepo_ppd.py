"""Cabeza CEPO-PPD y perdida de histograma.

El eje del resultado potencial esperado condicional se discretiza en L bins.
Cada token de consulta se proyecta a L logits y un softmax forma una
distribucion sobre bins. El valor verdadero se convierte en un objetivo suave
colocando una gaussiana estrecha a su alrededor e integrando sobre los bins, y
el entrenamiento minimiza la entropia cruzada entre ese objetivo y la prediccion.
A medida que la gaussiana se estrecha y crece el numero de bins, esto recupera
la verosimilitud logaritmica negativa del valor verdadero.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CEPOHead(nn.Module):
    def __init__(self, d_model: int, n_bins: int = 1024,
                 lo: float = 0.0, hi: float = 1.0, sigma: float = 0.02):
        super().__init__()
        self.proj = nn.Linear(d_model, n_bins)
        self.register_buffer("edges", torch.linspace(lo, hi, n_bins + 1))
        self.sigma = float(sigma)
        self.n_bins = int(n_bins)

    @property
    def centers(self) -> torch.Tensor:
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    def target_hist(self, mu: torch.Tensor) -> torch.Tensor:
        """Objetivo suavizado con gaussiana, normalizado sobre los bins."""
        c = self.centers
        dist = torch.distributions.Normal(mu.unsqueeze(-1), self.sigma)
        w = torch.exp(dist.log_prob(c))
        return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)  # logits [..., n_bins]

    def loss(self, logits: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        log_q = F.log_softmax(logits, dim=-1)
        return -(self.target_hist(mu) * log_q).sum(dim=-1).mean()

    @torch.no_grad()
    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        return (F.softmax(logits, dim=-1) * self.centers).sum(dim=-1)
