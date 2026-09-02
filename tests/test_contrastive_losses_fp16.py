"""The contrastive losses survive half precision (the V100 autocast path)."""
import torch

from neurocausalpfn.contrastive.losses import nt_xent_loss, supcon_loss


def test_nt_xent_half_precision_does_not_overflow():
    torch.manual_seed(0)
    feats = torch.nn.functional.normalize(torch.randn(8, 16), dim=1).half()
    loss = nt_xent_loss(feats, batch_size=4, tau=0.1)
    assert torch.isfinite(loss)


def test_supcon_half_precision_finite():
    torch.manual_seed(0)
    feats = torch.nn.functional.normalize(torch.randn(8, 16), dim=1).half()
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    assert torch.isfinite(supcon_loss(feats, labels, tau=0.1))
