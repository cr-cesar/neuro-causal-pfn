import torch

from neurocausalpfn.pfn.attention import context_only_mask


def test_query_attends_only_to_context():
    n_ctx, n_total = 5, 8
    mask = context_only_mask(n_ctx, n_total)

    # cualquier token que atienda a una posicion de consulta esta bloqueado
    assert torch.isinf(mask[7, 6]) and mask[7, 6] < 0
    # atender al contexto esta permitido
    assert mask[7, 0].item() == 0.0
    # tras el softmax, el peso total sobre las consultas es cero
    weights = torch.softmax(mask[7], dim=-1)
    assert weights[n_ctx:].sum().item() < 1e-6
    # y ninguna fila es enteramente menos infinito (softmax estable)
    assert torch.isfinite(torch.softmax(mask, dim=-1)).all()
