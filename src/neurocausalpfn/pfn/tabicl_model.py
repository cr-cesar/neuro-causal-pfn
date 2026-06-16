"""Transformer causal de la Etapa 2 con codificacion tabular estilo TabICL.

En lugar de embeber cada fila con una sola proyeccion lineal, se trata la
cohorte como una tabla y se razona en dos etapas:

1. Atencion por columna a traves de las muestras: para cada variable (columna),
   cada celda atiende a esa misma variable en el resto de pacientes, de modo que
   su embedding se vuelve consciente de la distribucion de toda la variable.
2. Atencion por fila entre pacientes: tras colapsar las columnas en un vector por
   fila, los pacientes interactuan en contexto para la prediccion.

Ambas etapas usan la misma mascara asimetrica de solo contexto a lo largo del
eje de muestras, asi que ninguna prediccion de consulta depende de otra consulta
y no hay fuga del resultado. La columna del resultado en las filas de consulta
es desconocida, por lo que se sustituye por un embedding de mascara aprendido.

Nota de escala: la atencion estandar es cuadratica en el numero de filas y de
columnas; para los contextos grandes del modo completo se sustituiria por una
atencion mas eficiente. El esqueleto actual usa atencion densa, suficiente para
el prototipo y para validar la arquitectura.
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
        self.n_cols = int(d_x) + 2          # covariables, tratamiento, resultado
        self.y_col = int(d_x) + 1           # indice de la columna del resultado
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.value_proj = nn.Linear(1, d_model)            # embebe el valor escalar de cada celda
        self.col_embed = nn.Parameter(torch.zeros(self.n_cols, d_model))  # identidad de cada columna
        nn.init.normal_(self.col_embed, std=0.02)
        self.y_mask = nn.Parameter(torch.zeros(d_model))   # resultado desconocido en las consultas
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

        # 1) atencion por columna a traves de las muestras
        col_in = cells.permute(0, 2, 1, 3).reshape(B * n_cols, n_rows, d_model)
        col_out = self.col_encoder(col_in, mask=mask)
        cells = col_out.reshape(B, n_cols, n_rows, d_model).permute(0, 2, 1, 3)

        # 2) colapso de columnas y atencion por fila entre pacientes
        rows = cells.mean(dim=2)                                            # [B, n_rows, d_model]
        rows = self.row_encoder(rows, mask=mask)
        return self.head(rows[:, nc:])                                      # [B, nq, n_bins]
