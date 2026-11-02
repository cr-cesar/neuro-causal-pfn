"""The Stage 2 causal transformer.

Trained from scratch with the prior-fitted network methodology on the
Neuro-Prior. It embeds the context and query rows, concatenates them, applies a
transformer encoder under the asymmetric attention mask and projects the query
tokens to the binned CEPO-PPD distribution.

The default configuration (12 layers, 8 heads, width 512) is the one from the
plan and should be read as a hypothesis to be checked with the backbone ablation
(E7), not as a closed design. Prototype mode uses a much smaller network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import context_only_mask
from .cepo_ppd import CEPOHead

ATTN_IMPLS = ("masked", "sdpa")


def _context_attention(layer: nn.TransformerEncoderLayer, h: torch.Tensor, n_ctx: int) -> torch.Tensor:
    """The self-attention block of ``layer`` under the context-only mask,
    computed as one attention of every token (queries) over the context tokens
    (keys and values): the same softmax attention and the same weights, without
    the [n_total, n_total] mask, so the fused kernels keep memory linear in the
    context length (2 Nov)."""
    sa = layer.self_attn
    B, N, d = h.shape
    H = sa.num_heads
    w_q, w_k, w_v = sa.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = sa.in_proj_bias.chunk(3, dim=0) if sa.in_proj_bias is not None else (None, None, None)
    ctx = h[:, :n_ctx]
    q = F.linear(h, w_q, b_q).view(B, N, H, d // H).transpose(1, 2)
    k = F.linear(ctx, w_k, b_k).view(B, n_ctx, H, d // H).transpose(1, 2)
    v = F.linear(ctx, w_v, b_v).view(B, n_ctx, H, d // H).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=sa.dropout if sa.training else 0.0)
    return sa.out_proj(out.transpose(1, 2).reshape(B, N, d))


class NeuroCausalPFN(nn.Module):
    def __init__(self, d_x: int, d_model: int = 512, n_layers: int = 12,
                 n_heads: int = 8, n_bins: int = 1024, dim_feedforward: int = None,
                 dropout: float = 0.0, lo: float = 0.0, hi: float = 1.0,
                 sigma: float = 0.02, attn_impl: str = "masked"):
        super().__init__()
        if attn_impl not in ATTN_IMPLS:
            raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}, got {attn_impl!r}")
        self.attn_impl = attn_impl
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.ctx_emb = nn.Linear(d_x + 2, d_model)   # covariates, treatment, outcome
        self.qry_emb = nn.Linear(d_x + 1, d_model)   # covariates, treatment
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = CEPOHead(d_model, n_bins=n_bins, lo=lo, hi=hi, sigma=sigma)

    def forward(self, Xc: torch.Tensor, Tc: torch.Tensor, Yc: torch.Tensor,
                Xq: torch.Tensor, Tq: torch.Tensor) -> torch.Tensor:
        # Xc [B, n_ctx, d_x]; Tc, Yc [B, n_ctx]; Xq [B, n_qry, d_x]; Tq [B, n_qry]
        ctx = self.ctx_emb(torch.cat([Xc, Tc.unsqueeze(-1), Yc.unsqueeze(-1)], dim=-1))
        qry = self.qry_emb(torch.cat([Xq, Tq.unsqueeze(-1)], dim=-1))
        tokens = torch.cat([ctx, qry], dim=1)
        n_ctx = ctx.shape[1]
        n_total = tokens.shape[1]
        if self.attn_impl == "sdpa":
            # pre-norm layers, as nn.TransformerEncoderLayer(norm_first=True) runs them
            hidden = tokens
            for layer in self.encoder.layers:
                hidden = hidden + layer.dropout1(_context_attention(layer, layer.norm1(hidden), n_ctx))
                hidden = hidden + layer._ff_block(layer.norm2(hidden))
            if self.encoder.norm is not None:
                hidden = self.encoder.norm(hidden)
            return self.head(hidden[:, n_ctx:])
        mask = context_only_mask(n_ctx, n_total, device=tokens.device, dtype=tokens.dtype)
        hidden = self.encoder(tokens, mask=mask)
        return self.head(hidden[:, n_ctx:])  # logits [B, n_qry, n_bins]
