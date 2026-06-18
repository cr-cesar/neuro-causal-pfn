"""Stage 2 causal transformer with TabICL-style tabular encoding.

Instead of embedding each row with a single linear projection, the cohort is
treated as a table and reasoning proceeds in two stages:

1. Column-wise attention across the samples: for each variable (column), each
   cell attends to that same variable in the rest of the patients, so that its
   embedding becomes aware of the distribution of the whole variable.
2. Row-wise attention between patients: after collapsing the columns into a
   per-row vector, the patients interact in context for the prediction.

Both stages use the same asymmetric context-only mask along the sample axis, so
no query prediction depends on another query and there is no outcome leakage. The
outcome column in the query rows is unknown, so it is replaced by a learned mask
embedding.

Scaling note: standard attention is quadratic in the number of rows and columns;
for the large contexts of full mode it would be replaced by a more efficient
attention. The current skeleton uses dense attention, sufficient for the
prototype and for validating the architecture.
"""
import torch
import torch.nn as nn

from .attention import context_only_mask
from .cepo_ppd import CEPOHead


class NeuroCausalPFNTabICL(nn.Module):
    def __init__(self, d_x: int, d_model: int = 512, n_row_layers: int = 12,
                 n_col_layers: int = 3, n_heads: int = 8, n_bins: int = 1024,
                 dim_feedforward: int = None, dropout: float = 0.0,
                 lo: float = 0.0, hi: float = 1.0, sigma: float = 0.02):
        super().__init__()
        self.d_x = int(d_x)
        self.n_cols = int(d_x) + 2          # covariates, treatment, outcome
        self.y_col = int(d_x) + 1           # index of the outcome column
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.value_proj = nn.Linear(1, d_model)            # embeds the scalar value of each cell
        self.col_embed = nn.Parameter(torch.zeros(self.n_cols, d_model))  # identity of each column
        nn.init.normal_(self.col_embed, std=0.02)
        self.y_mask = nn.Parameter(torch.zeros(d_model))   # unknown outcome in the queries
        nn.init.normal_(self.y_mask, std=0.02)

        def _encoder(n_layers: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, num_layers=n_layers)

        self.col_encoder = _encoder(n_col_layers)
        self.row_encoder = _encoder(n_row_layers)
        self.head = CEPOHead(d_model, n_bins=n_bins, lo=lo, hi=hi, sigma=sigma)

    def _embed_cells(self, Xc, Tc, Yc, Xq, Tq):
        B, nc, _ = Xc.shape
        nq = Xq.shape[1]
        ctx = torch.cat([Xc, Tc.unsqueeze(-1), Yc.unsqueeze(-1)], dim=-1)   # [B, nc, n_cols]
        qzero = torch.zeros(B, nq, 1, device=Xq.device, dtype=Xq.dtype)
        qry = torch.cat([Xq, Tq.unsqueeze(-1), qzero], dim=-1)              # [B, nq, n_cols]
        vals = torch.cat([ctx, qry], dim=1)                                # [B, n_rows, n_cols]
        cells = self.value_proj(vals.unsqueeze(-1))                        # [B, n_rows, n_cols, d_model]
        cells = cells + self.col_embed.view(1, 1, self.n_cols, -1)
        cells = cells.clone()
        cells[:, nc:, self.y_col, :] = (self.y_mask + self.col_embed[self.y_col]).view(1, 1, -1)
        return cells, nc

    def forward(self, Xc: torch.Tensor, Tc: torch.Tensor, Yc: torch.Tensor,
                Xq: torch.Tensor, Tq: torch.Tensor) -> torch.Tensor:
        cells, nc = self._embed_cells(Xc, Tc, Yc, Xq, Tq)
        B, n_rows, n_cols, d_model = cells.shape
        mask = context_only_mask(nc, n_rows, device=cells.device, dtype=cells.dtype)

        # 1) column-wise attention across the samples
        col_in = cells.permute(0, 2, 1, 3).reshape(B * n_cols, n_rows, d_model)
        col_out = self.col_encoder(col_in, mask=mask)
        cells = col_out.reshape(B, n_cols, n_rows, d_model).permute(0, 2, 1, 3)

        # 2) column collapse and row-wise attention between patients
        rows = cells.mean(dim=2)                                            # [B, n_rows, d_model]
        rows = self.row_encoder(rows, mask=mask)
        return self.head(rows[:, nc:])                                      # [B, nq, n_bins]
