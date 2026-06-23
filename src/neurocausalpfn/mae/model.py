"""Vascular-constrained Hi-End MAE (Arm D, section 9.4).

A masked autoencoder over binary lesion masks. The encoder is a 3D ViT; the
decoder is hierarchical (Hi-End MAE, Tang et al. 2026): it receives skip
connections from every encoder layer, not only the bottleneck, so anatomy rather
than texture drives the reconstruction. Masking removes 75% of the patches in
contiguous blocks (a stand-in for an arterial-territory atlas until one is wired
in). The reconstruction loss is a lesion-weighted BCE on the masked patches only.
"""
import numpy as np
import torch
import torch.nn as nn


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """[B, 1, D, H, W] -> [B, N, p^3] with N the number of patches."""
    b, c, d, h, w = x.shape
    gd, gh, gw = d // p, h // p, w // p
    x = x.reshape(b, c, gd, p, gh, p, gw, p)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
    return x.reshape(b, gd * gh * gw, p * p * p * c)


def _grid(in_shape, p):
    return (in_shape[0] // p, in_shape[1] // p, in_shape[2] // p)


def _block_ids(grid, block, device) -> torch.Tensor:
    gd, gh, gw = grid
    bd, bh, bw = block
    idx = torch.arange(gd * gh * gw, device=device)
    i = idx // (gh * gw)
    rem = idx % (gh * gw)
    j = rem // gw
    k = rem % gw
    nbh = (gh + bh - 1) // bh
    nbw = (gw + bw - 1) // bw
    return (i // bd) * (nbh * nbw) + (j // bh) * nbw + (k // bw)


def block_masking(b, grid, block, mask_ratio, device):
    """Returns ids_keep, ids_restore, mask (1 = masked) with contiguous-block
    structure and a fixed number of visible patches per sample."""
    n = grid[0] * grid[1] * grid[2]
    len_keep = max(1, int(round(n * (1 - mask_ratio))))
    block_id = _block_ids(grid, block, device)
    n_blocks = int(block_id.max().item()) + 1
    block_scores = torch.rand(b, n_blocks, device=device)
    patch_scores = block_scores[:, block_id] + 1e-3 * torch.rand(b, n, device=device)
    ids_shuffle = torch.argsort(patch_scores, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones(b, n, device=device)
    mask.scatter_(1, ids_keep, 0.0)
    return ids_keep, ids_restore, mask


def _gather(x, ids):
    return torch.gather(x, 1, ids.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _block_layer(dim, heads):
    return nn.TransformerEncoderLayer(dim, heads, dim_feedforward=4 * dim, dropout=0.0,
                                      activation="gelu", batch_first=True, norm_first=True)


class HiEndMAE3D(nn.Module):
    def __init__(self, in_shape=(96, 112, 96), patch: int = 16, embed_dim: int = 384,
                 depth: int = 12, heads: int = 6, decoder_dim: int = 192,
                 decoder_depth: int = 8, decoder_heads: int = 6, zdim: int = 50,
                 mask_ratio: float = 0.75, block=(2, 2, 2)):
        super().__init__()
        self.in_shape = tuple(int(s) for s in in_shape)
        self.patch = patch
        self.grid = _grid(self.in_shape, patch)
        self.n_patches = self.grid[0] * self.grid[1] * self.grid[2]
        self.mask_ratio = mask_ratio
        self.block = tuple(block)
        self.patch_dim = patch ** 3
        self.zdim = zdim

        self.patch_embed = nn.Conv3d(1, embed_dim, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.enc_blocks = nn.ModuleList([_block_layer(embed_dim, heads) for _ in range(depth)])
        self.enc_norm = nn.LayerNorm(embed_dim)
        self.to_latent = nn.Linear(embed_dim, zdim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.skip_proj = nn.ModuleList([nn.Linear(embed_dim, decoder_dim) for _ in range(depth)])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, decoder_dim))
        self.dec_blocks = nn.ModuleList([_block_layer(decoder_dim, decoder_heads) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _embed(self, x):
        return self.patch_embed(x).flatten(2).transpose(1, 2)   # [B, N, E]

    def patchify(self, x):
        return patchify(x, self.patch)

    def forward(self, x):
        b = x.shape[0]
        tokens = self._embed(x) + self.pos_embed
        ids_keep, ids_restore, mask = block_masking(b, self.grid, self.block, self.mask_ratio, x.device)
        h = _gather(tokens, ids_keep)
        feats = []
        for blk in self.enc_blocks:
            h = blk(h)
            feats.append(h)
        h = self.enc_norm(h)
        # hierarchical skip: every encoder layer feeds the decoder (visible tokens)
        skip = sum(self.skip_proj[i](feats[i]) for i in range(len(feats)))
        dec_vis = self.decoder_embed(h) + skip
        n_mask = mask.shape[1] - dec_vis.shape[1]
        mask_tokens = self.mask_token.expand(b, n_mask, -1)
        x_ = torch.cat([dec_vis, mask_tokens], dim=1)
        x_ = _gather(x_, ids_restore) + self.decoder_pos_embed
        for blk in self.dec_blocks:
            x_ = blk(x_)
        pred = self.decoder_pred(self.decoder_norm(x_))         # [B, N, patch_dim]
        return pred, mask

    @torch.no_grad()
    def encode_z(self, x):
        """Deterministic latent for export: full volume, no masking, mean-pooled
        encoder features projected to zdim."""
        h = self._embed(x) + self.pos_embed
        for blk in self.enc_blocks:
            h = blk(h)
        h = self.enc_norm(h)
        return self.to_latent(h.mean(1))
