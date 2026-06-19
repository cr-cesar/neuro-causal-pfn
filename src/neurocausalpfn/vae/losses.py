"""Stage 1 loss functions.

The reconstruction objective combines binary cross-entropy with a soft Dice
term, because the foreground voxels are a tiny fraction of the volume and a pure
cross-entropy is almost degenerate under that imbalance.
"""
import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice over sigmoid probabilities. Equals 0 for identical masks."""
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
    """Closed-form KL divergence against a standard normal."""
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()


def vae_loss(logits: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, beta: float = 1.0,
             w_bce: float = 1.0, w_dice: float = 1.0):
    """Full Stage 1 objective: L = L_rec + beta * D_KL."""
    rec, parts = bce_dice_loss(logits, target, w_bce, w_dice)
    kl = kl_standard_normal(mu, logvar)
    total = rec + beta * kl
    parts.update({"rec": float(rec.detach()), "kl": float(kl.detach()),
                  "beta": float(beta), "total": float(total.detach())})
    return total, parts


def mse_recon_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Continuous reconstruction: MSE between the sigmoid output and the target map.

    For the disconnectome the target is a continuous probability map in [0, 1],
    not a binary mask, so the reconstruction is measured with MSE over the
    predicted probability (sigmoid of the logits) and not with BCE plus Dice.
    """
    return F.mse_loss(torch.sigmoid(logits), target)


def vae_loss_mse(logits: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
                 logvar: torch.Tensor, beta: float = 1.0):
    """VAE objective for continuous inputs: L = MSE + beta * D_KL."""
    rec = mse_recon_loss(logits, target)
    kl = kl_standard_normal(mu, logvar)
    total = rec + beta * kl
    parts = {"mse": float(rec.detach()), "rec": float(rec.detach()),
             "kl": float(kl.detach()), "beta": float(beta), "total": float(total.detach())}
    return total, parts


def vae_loss_two_channel(logits: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
                         logvar: torch.Tensor, beta: float = 1.0,
                         w_bce: float = 1.0, w_dice: float = 1.0):
    """Early-fusion objective for a two-channel input (E9a).

    Channel 0 is the binary lesion (BCE plus Dice) and channel 1 is the
    continuous disconnectome (MSE). The two reconstruction terms are summed and
    the KL is shared, because a single VAE encodes both channels jointly.
    """
    rec_lesion, parts = bce_dice_loss(logits[:, 0:1], target[:, 0:1], w_bce, w_dice)
    rec_disc = mse_recon_loss(logits[:, 1:2], target[:, 1:2])
    rec = rec_lesion + rec_disc
    kl = kl_standard_normal(mu, logvar)
    total = rec + beta * kl
    parts.update({"mse": float(rec_disc.detach()), "rec": float(rec.detach()),
                  "kl": float(kl.detach()), "beta": float(beta), "total": float(total.detach())})
    return total, parts
