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

from .attention import context_only_mask
from .cepo_ppd import CEPOHead


class NeuroCausalPFN(nn.Module):
    def __init__(self, d_x: int, d_model: int = 512, n_layers: int = 12,
                 n_heads: int = 8, n_bins: int = 1024, dim_feedforward: int = None,
                 dropout: float = 0.0, lo: float = 0.0, hi: float = 1.0,
                 sigma: float = 0.02):
        super().__init__()
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
        mask = context_only_mask(n_ctx, n_total, device=tokens.device, dtype=tokens.dtype)
        hidden = self.encoder(tokens, mask=mask)
        return self.head(hidden[:, n_ctx:])  # logits [B, n_qry, n_bins]
