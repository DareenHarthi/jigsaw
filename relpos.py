# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# Written as an extension of DINOv3 (https://github.com/facebookresearch/dinov3).

"""Relative position: name the direction between a tile and one of its neighbours.

With `block_size > 1` the tiles are pooled into blocks first (global puzzle): the shuffle
stays at tile level, but the 8-way task runs on the coarser block grid and the arrangement
inside a block is never supervised.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

from views import block_members, gather_tiles, neighbour_pairs, pool_blocks, slot_to_home

N_DIRECTIONS = 8


class RelPosHead(nn.Module):
    """Projects a tile pair to an 8-way direction logit."""

    def __init__(self, embed_dim: int, proj_dim: int = 256, hidden_dim: int = 512, nlayers: int = 3):
        super().__init__()
        self.proj = nn.Linear(embed_dim, proj_dim)
        layers, d = [], 2 * proj_dim
        for _ in range(max(nlayers - 1, 0)):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            d = hidden_dim
        layers += [nn.Linear(d, N_DIRECTIONS)]
        self.mlp = nn.Sequential(*layers)

    def init_weights(self):
        self.apply(_init_linear)

    def forward(self, anchor: Tensor, partner: Tensor) -> Tensor:
        a, b = self.proj(anchor), self.proj(partner)
        return self.mlp(torch.cat([a, b], dim=-1))


def _init_linear(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def direction_targets(view, block_size: int = 1):
    """Partner index and validity for the grid the loss runs on."""
    if block_size == 1:
        return view["partner"], view["valid"], view["grid"]
    side = view["grid"] // block_size
    ident = torch.arange(side * side, device=view["mosaic"].device)
    ident = ident.unsqueeze(0).expand(view["mosaic"].shape[0], -1)
    partner, valid = neighbour_pairs(ident, side)
    return partner, valid, side


def tile_features(tokens: Tensor, view, block_size: int = 1):
    """Pool mosaic tokens into per-tile (or per-block) features, in home order."""
    tiles = gather_tiles(tokens, view["slot_index"]).mean(2)
    tiles = slot_to_home(tiles, view["tile_at_slot"])
    if block_size > 1:
        members = block_members(view["grid"], block_size, tokens.device)
        tiles = pool_blocks(tiles, members)
    return tiles


class RelPosLoss(nn.Module):
    """Cross-entropy over the 8 directions, averaged over valid pairs."""

    def forward(self, logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
        logp = logits.log_softmax(-1)
        picked = torch.gather(logp, -1, target.unsqueeze(-1)).squeeze(-1)
        return -(picked * valid).sum() / valid.sum().clamp(min=1)

    @staticmethod
    @torch.no_grad()
    def accuracy(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
        hit = (logits.argmax(-1) == target) & valid
        return hit.sum() / valid.sum().clamp(min=1)


def relpos_loss(student_tokens, teacher_tokens, view, head, block_size=1):
    """One anchor-neighbour direction loss. Teacher tiles come from the intact crop."""
    s = tile_features(student_tokens, view, block_size)
    t = gather_tiles(teacher_tokens, view["source_index"]).mean(2)
    if block_size > 1:
        t = pool_blocks(t, block_members(view["grid"], block_size, t.device))
    partner, valid, side = direction_targets(view, block_size)

    n = s.shape[1]
    anchor = s.unsqueeze(2).expand(-1, -1, N_DIRECTIONS, -1)
    idx = partner.unsqueeze(-1).expand(-1, -1, -1, t.shape[-1])
    nb = torch.gather(t.unsqueeze(1).expand(-1, n, -1, -1), 2, idx)
    logits = head(anchor.flatten(1, 2), nb.flatten(1, 2)).unflatten(1, (n, N_DIRECTIONS))
    target = torch.arange(N_DIRECTIONS, device=s.device).view(1, 1, -1).expand_as(valid)
    return logits, target, valid
