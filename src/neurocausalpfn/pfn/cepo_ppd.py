"""CEPO-PPD head and histogram loss.

The expected conditional potential outcome axis is discretized into L bins. Each
query token is projected to L logits and a softmax forms a distribution over
bins. The ground-truth value is converted into a soft target by placing a narrow
gaussian around it and integrating over the bins, and training minimizes the
cross-entropy between that target and the prediction. As the gaussian narrows and
the number of bins grows, this recovers the negative log-likelihood of the
ground-truth value.
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
        """Gaussian-smoothed target, normalized over the bins."""
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
