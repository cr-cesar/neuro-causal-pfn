"""El transformer causal de la Etapa 2.

Entrenado desde cero con la metodologia de prior-fitted network sobre el
Neuro-Prior. Embebe las filas de contexto y de consulta, las concatena, aplica
un codificador transformer bajo la mascara de atencion asimetrica y proyecta los
tokens de consulta a la distribucion CEPO-PPD por bins.

La configuracion por defecto (12 capas, 8 cabezas, ancho 512) es la del plan y
debe leerse como una hipotesis a comprobar con la ablacion de backbone (E7), no
como un diseno cerrado. El modo prototipo usa una red mucho mas pequena.
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
        self.ctx_emb = nn.Linear(d_x + 2, d_model)   # covariables, tratamiento, resultado
        self.qry_emb = nn.Linear(d_x + 1, d_model)   # covariables, tratamiento
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
