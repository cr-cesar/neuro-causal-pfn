import torch

from neurocausalpfn.pfn.attention import context_only_mask


def test_query_attends_only_to_context():
    n_ctx, n_total = 5, 8
    mask = context_only_mask(n_ctx, n_total)

    # any token attending to a query position is blocked
    assert torch.isinf(mask[7, 6]) and mask[7, 6] < 0
    # attending to the context is allowed
    assert mask[7, 0].item() == 0.0
    # after the softmax, the total weight over the queries is zero
    weights = torch.softmax(mask[7], dim=-1)
    assert weights[n_ctx:].sum().item() < 1e-6
    # and no row is entirely minus infinity (stable softmax)
    assert torch.isfinite(torch.softmax(mask, dim=-1)).all()
