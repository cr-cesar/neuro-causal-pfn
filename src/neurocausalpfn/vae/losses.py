"""Funciones de perdida de la Etapa 1.

El objetivo de reconstruccion combina entropia cruzada binaria con un termino
de Dice suave, porque los voxeles de primer plano son una fraccion minuscula del
volumen y una entropia cruzada pura es casi degenerada ante ese desequilibrio.
"""
import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Dice suave sobre probabilidades sigmoides. Vale 0 en mascaras identicas."""
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    num = 2.0 * (p * t).sum(1) + eps
    den = p.sum(1) + t.sum(1) + eps
    return (1.0 - num / den).mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                  w_bce: float = 1.0, w_dice: float = 1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = soft_dice_loss(logits, target)
    total = w_bce * bce + w_dice * dice
    return total, {"bce": float(bce.detach()), "dice": float(dice.detach())}


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Divergencia KL en forma cerrada frente a una normal estandar."""
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()


def vae_loss(logits: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, beta: float = 1.0,
             w_bce: float = 1.0, w_dice: float = 1.0):
    """Objetivo completo de la Etapa 1: L = L_rec + beta * D_KL."""
    rec, parts = bce_dice_loss(logits, target, w_bce, w_dice)
    kl = kl_standard_normal(mu, logvar)
    total = rec + beta * kl
    parts.update({"rec": float(rec.detach()), "kl": float(kl.detach()),
                  "beta": float(beta), "total": float(total.detach())})
    return total, parts


def mse_recon_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reconstruccion continua: MSE entre la salida sigmoide y el mapa objetivo.

    Para el disconnectoma el objetivo es un mapa de probabilidad continuo en
    [0, 1], no una mascara binaria, asi que la reconstruccion se mide con MSE
    sobre la probabilidad predicha (sigmoide de los logits) y no con BCE mas Dice.
    """
    return F.mse_loss(torch.sigmoid(logits), target)


def vae_loss_mse(logits: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
                 logvar: torch.Tensor, beta: float = 1.0):
    """Objetivo del VAE para entradas continuas: L = MSE + beta * D_KL."""
    rec = mse_recon_loss(logits, target)
    kl = kl_standard_normal(mu, logvar)
    total = rec + beta * kl
    parts = {"mse": float(rec.detach()), "rec": float(rec.detach()),
             "kl": float(kl.detach()), "beta": float(beta), "total": float(total.detach())}
    return total, parts
