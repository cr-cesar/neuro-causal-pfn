"""The PNS surrogate survives half precision (E5b died on the first batch:
no SVD or linalg.solve kernel exists for Half on CUDA or CPU)."""
import torch

from neurocausalpfn.causal.pns import (_topk_components, soft_pns_per_dim,
                                       soft_pns_value)


def test_topk_components_accepts_half_input():
    z = torch.randn(32, 12).half()
    C = _topk_components(z, 5)
    assert C.shape == (32, 5)
    assert not C.requires_grad                    # detached common cause


def test_soft_pns_backpropagates_from_half_latents():
    mu = torch.randn(32, 12).half().requires_grad_(True)
    y = (torch.rand(32) > 0.5).float()
    val = soft_pns_value(mu, y, k=4)
    assert torch.isfinite(val)
    val.backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()


def test_half_and_float_paths_agree():
    torch.manual_seed(0)
    mu = torch.randn(64, 8)
    y = (torch.rand(64) > 0.5).float()
    a = soft_pns_per_dim(mu, y, k=3)
    b = soft_pns_per_dim(mu.half(), y, k=3)
    assert torch.allclose(a, b, atol=2e-3)        # only input-quantisation noise
