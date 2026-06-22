"""Contrastive losses for Arm C.

- supcon_loss: Supervised Contrastive loss (Khosla et al. 2020, used by Tsai et
  al. 2024). Positives are all samples sharing the anchor's label, so the latent
  groups patients by outcome rather than by identity. Works with small batches,
  unlike SimCLR, which is the reason SupCon (not SimCLR) is chosen here.
- nt_xent_loss: the SimCLR objective, used as the intra-modal term where the two
  augmented views of a sample are the only positives.
"""
import torch
import torch.nn.functional as F


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss over L2-normalised features [M, d] with integer
    labels [M] (the same label repeated across a sample's views)."""
    device = features.device
    m = features.shape[0]
    sim = features @ features.T / tau
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()        # stability
    self_mask = torch.eye(m, dtype=torch.bool, device=device)
    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T) & (~self_mask)
    pos_count = pos_mask.sum(1)
    pos_log_prob = (log_prob * pos_mask).sum(1) / pos_count.clamp(min=1)
    valid = pos_count > 0
    if not valid.any():
        return features.sum() * 0.0
    return -pos_log_prob[valid].mean()


def nt_xent_loss(features: torch.Tensor, batch_size: int, tau: float = 0.1) -> torch.Tensor:
    """SimCLR NT-Xent over [2B, d] normalised features, where view 1 is the first
    B rows and view 2 the next B; the positive of sample i is its other view."""
    device = features.device
    m = 2 * batch_size
    sim = features @ features.T / tau
    sim = sim.masked_fill(torch.eye(m, dtype=torch.bool, device=device), -1e9)
    targets = torch.cat([torch.arange(batch_size, device=device) + batch_size,
                         torch.arange(batch_size, device=device)])
    return F.cross_entropy(sim, targets)
