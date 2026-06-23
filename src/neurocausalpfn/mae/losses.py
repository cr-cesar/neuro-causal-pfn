"""Reconstruction loss for the Hi-End MAE (Arm D).

Lesion-weighted BCE over the masked patches only. Inputs are binary {0, 1}, so
BCE (not MSE) is the right reconstruction objective; patches that contain any
lesion voxel are up-weighted (default 10x) so the model spends its capacity on
the anatomy that matters rather than on empty background.
"""
import torch
import torch.nn.functional as F


def masked_lesion_bce(pred_logits: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor, lesion_weight: float = 10.0) -> torch.Tensor:
    """pred_logits, target: [B, N, patch_dim]; mask: [B, N] with 1 for masked
    patches. The loss is averaged over masked patches, weighting lesion-bearing
    patches by lesion_weight."""
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none").mean(-1)  # [B, N]
    has_lesion = (target.sum(-1) > 0).float()
    weight = (1.0 + (lesion_weight - 1.0) * has_lesion) * mask
    return (bce * weight).sum() / (weight.sum() + 1e-8)
