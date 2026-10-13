"""Stage 2 causal transformer with the TabICL architecture (Qu et al., 2025).

The cohort is a table: one row per patient, one column per covariate, plus the
treatment column and the outcome column. Reasoning proceeds in the three stages
of TabICL, re-implemented here against the reference package (tabicl 2.2.0,
``tabicl/_model``) and its minimal re-implementation (nanotabicl):

1. Column embedding (TF_col). Every cell is projected to ``embed_dim`` and each
   column is processed by a Set Transformer of induced self-attention blocks
   (ISAB): a small set of learned inducing points attends to the CONTEXT rows
   of the column, and every row (context and query) then attends to the
   inducing points. The embedding of a cell is therefore aware of the
   distribution of its column, at a cost linear in the number of rows, and no
   query row leaks into the context. The embedding is target-aware: the
   outcome of each context row is embedded and added to that row's cell in
   every column.
2. Row interaction (TF_row). For every row, a transformer over its cells plus
   ``n_cls`` learned CLS tokens; the CLS outputs are concatenated into the row
   representation of dimension ``n_cls * embed_dim``. Here the covariates, the
   treatment and the outcome interact within a patient.
3. In-context learning (TF_icl). The outcome of the context rows is embedded
   and added to their row representation once more, and a transformer over the
   rows lets every query attend to the context (context-only mask: no query
   attends to another query, so the per-query predictions are independent).
   The CEPO head reads the query rows.

Differences with the reference, all deliberate: the schema is fixed (the
columns are the same in every table), so a learned column embedding replaces
the rotary position encoding over features; the treatment is a column of the
table and the outcome column of the query rows carries a learned mask token;
the scalable softmax (SSMax) of the reference is not used, the contexts here
being at most a few thousand rows; the residual branches start at zero as in
TabICL v1. The former skeleton (dense column attention and a mean over the
columns as the row representation) is kept as ``tabicl_v0`` so that the
checkpoints trained with it can still be scored.
"""
from typing import Optional

import torch
import torch.nn as nn

from .attention import context_only_mask
from .cepo_ppd import CEPOHead


class _Block(nn.Module):
    """Pre-norm transformer block with cross-attention: ``forward(q, kv)``
    attends from ``q`` to ``kv`` (``kv`` defaults to ``q``: self-attention).
    The attention output projection and the second feed-forward linear start
    at zero, so the block is the identity at initialisation."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.0,
                 zero_init: bool = True):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.drop = nn.Dropout(dropout)
        if zero_init:
            nn.init.zeros_(self.attn.out_proj.weight); nn.init.zeros_(self.attn.out_proj.bias)
            nn.init.zeros_(self.ff[-1].weight); nn.init.zeros_(self.ff[-1].bias)

    def forward(self, q: torch.Tensor, kv: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qn = self.ln_q(q)
        kvn = qn if kv is None else self.ln_kv(kv)
        a, _ = self.attn(qn, kvn, kvn, attn_mask=attn_mask, need_weights=False)
        x = q + self.drop(a)
        return x + self.drop(self.ff(self.ln_ff(x)))


class _ISAB(nn.Module):
    """Induced self-attention block (Lee et al., 2019) as used by TabICL: the
    inducing points attend to the first ``n_context`` rows only, every row then
    attends to the inducing points."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, n_inds: int,
                 dropout: float = 0.0, zero_init: bool = True):
        super().__init__()
        self.to_inds = _Block(d_model, n_heads, dim_feedforward, dropout, zero_init)
        self.from_inds = _Block(d_model, n_heads, dim_feedforward, dropout, zero_init)
        self.inds = nn.Parameter(torch.empty(n_inds, d_model))
        nn.init.trunc_normal_(self.inds, std=0.02)

    def forward(self, src: torch.Tensor, n_context: int) -> torch.Tensor:
        inds = self.inds.unsqueeze(0).expand(src.shape[0], -1, -1)
        hidden = self.to_inds(inds, src[:, :n_context])
        return self.from_inds(src, hidden)


class NeuroCausalPFNTabICL(nn.Module):
    def __init__(self, d_x: int, embed_dim: int = 128, n_col_layers: int = 3, n_row_layers: int = 3,
                 n_icl_layers: int = 12, n_heads: int = 8, n_cls: int = 4, n_inds: int = 128,
                 ff_factor: int = 2, dropout: float = 0.0, n_bins: int = 1024,
                 lo: float = 0.0, hi: float = 1.0, sigma: float = 0.02, zero_init: bool = True):
        super().__init__()
        self.d_x = int(d_x)
        self.n_cols = int(d_x) + 2          # covariates, treatment, outcome
        self.y_col = int(d_x) + 1
        E, D = int(embed_dim), int(embed_dim) * int(n_cls)
        self.embed_dim, self.n_cls, self.d_row = E, int(n_cls), D

        # stage 1: cell value -> E, column identity, outcome (target-aware) and the
        # mask token of the unknown outcome of the queries
        self.value_proj = nn.Linear(1, E)
        self.col_embed = nn.Parameter(torch.zeros(self.n_cols, E)); nn.init.normal_(self.col_embed, std=0.02)
        self.y_mask = nn.Parameter(torch.zeros(E)); nn.init.normal_(self.y_mask, std=0.02)
        self.y_col_enc = nn.Linear(1, E)
        self.col_blocks = nn.ModuleList([_ISAB(E, n_heads, ff_factor * E, n_inds, dropout, zero_init)
                                         for _ in range(n_col_layers)])
        # stage 2: CLS tokens and the row transformer over cells + CLS
        self.cls = nn.Parameter(torch.empty(self.n_cls, E)); nn.init.trunc_normal_(self.cls, std=0.02)
        self.row_blocks = nn.ModuleList([_Block(E, n_heads, ff_factor * E, dropout, zero_init)
                                         for _ in range(n_row_layers)])
        self.row_ln = nn.LayerNorm(E)
        # stage 3: outcome added to the context rows again, transformer over the rows
        self.y_row_enc = nn.Linear(1, D)
        self.icl_blocks = nn.ModuleList([_Block(D, n_heads, ff_factor * D, dropout, zero_init)
                                         for _ in range(n_icl_layers)])
        self.icl_ln = nn.LayerNorm(D)
        self.head = CEPOHead(D, n_bins=n_bins, lo=lo, hi=hi, sigma=sigma)

    @staticmethod
    def _standardise(Xc: torch.Tensor, Xq: torch.Tensor):
        """Per-column standardisation with the context statistics (as TabICL
        does with the training rows)."""
        mu = Xc.mean(dim=1, keepdim=True)
        sd = Xc.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        return (Xc - mu) / sd, (Xq - mu) / sd

    def _embed_cells(self, Xc, Tc, Yc, Xq, Tq):
        B, nc, _ = Xc.shape
        nq = Xq.shape[1]
        Xc, Xq = self._standardise(Xc, Xq)
        ctx = torch.cat([Xc, Tc.unsqueeze(-1), Yc.unsqueeze(-1)], dim=-1)              # [B, nc, n_cols]
        qry = torch.cat([Xq, Tq.unsqueeze(-1), torch.zeros_like(Tq).unsqueeze(-1)], dim=-1)
        vals = torch.cat([ctx, qry], dim=1)                                           # [B, T, n_cols]
        cells = self.value_proj(vals.unsqueeze(-1)) + self.col_embed.view(1, 1, self.n_cols, -1)
        cells = cells.clone()
        cells[:, nc:, self.y_col, :] = (self.y_mask + self.col_embed[self.y_col]).view(1, 1, -1)
        # target-aware: the outcome of every context row enters every column of that row
        cells[:, :nc] = cells[:, :nc] + self.y_col_enc(Yc.unsqueeze(-1)).unsqueeze(2)
        return cells, nc

    def forward(self, Xc: torch.Tensor, Tc: torch.Tensor, Yc: torch.Tensor,
                Xq: torch.Tensor, Tq: torch.Tensor) -> torch.Tensor:
        cells, nc = self._embed_cells(Xc, Tc, Yc, Xq, Tq)
        B, T, H, E = cells.shape

        # 1) column embedding: ISAB over the rows of every column
        col = cells.permute(0, 2, 1, 3).reshape(B * H, T, E)
        for blk in self.col_blocks:
            col = blk(col, nc)
        cells = col.reshape(B, H, T, E).permute(0, 2, 1, 3)                            # [B, T, H, E]

        # 2) row interaction: CLS tokens + cells of each row
        rows = cells.reshape(B * T, H, E)
        cls = self.cls.unsqueeze(0).expand(B * T, -1, -1)
        tokens = torch.cat([cls, rows], dim=1)                                        # [B*T, C+H, E]
        for blk in self.row_blocks:
            tokens = blk(tokens)
        R = self.row_ln(tokens[:, : self.n_cls]).reshape(B, T, self.d_row)            # [B, T, C*E]

        # 3) in-context learning over the rows, outcome of the context rows added again
        R = R.clone()
        R[:, :nc] = R[:, :nc] + self.y_row_enc(Yc.unsqueeze(-1))
        mask = context_only_mask(nc, T, device=R.device, dtype=R.dtype)
        for blk in self.icl_blocks:
            R = blk(R, attn_mask=mask)
        return self.head(self.icl_ln(R[:, nc:]))                                     # [B, nq, n_bins]
