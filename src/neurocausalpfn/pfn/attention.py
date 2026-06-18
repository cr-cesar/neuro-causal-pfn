"""Asymmetric attention masking for Stage 2.

Both context tokens and query tokens attend only to the context. This makes the
per-query predictions mutually independent and is consistent with the predictive
posterior semantics: each query is resolved from the context, not from the other
queries.
"""
import torch


def context_only_mask(n_context: int, n_total: int,
                      device=None, dtype=torch.float32) -> torch.Tensor:
    """Additive mask [n_total, n_total]. The query columns are blocked with
    negative infinity, so that no token can attend to a query position. The first
    n_context columns stay at zero, so no row is entirely negative infinity and
    the softmax remains stable."""
    mask = torch.zeros(n_total, n_total, device=device, dtype=dtype)
    mask[:, n_context:] = float("-inf")
    return mask
