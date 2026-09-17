# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# Written as an extension of DINOv3 (https://github.com/facebookresearch/dinov3).

"""Tile cut, shuffle and pooling shared by the jigsaw objectives."""

import random
from typing import Dict, Optional

import torch
from torch import Tensor

OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def slot_token_index(grid: int, tile_patches: int, device=None) -> Tensor:
    """Mosaic tokens belonging to each slot. [grid^2, tile_patches^2]"""
    side = grid * tile_patches
    rows = torch.arange(grid, device=device).repeat_interleave(grid)
    cols = torch.arange(grid, device=device).repeat(grid)
    top_left = rows * tile_patches * side + cols * tile_patches
    steps = torch.arange(tile_patches, device=device)
    within = (steps.unsqueeze(1) * side + steps).reshape(-1)
    return top_left.unsqueeze(1) + within.unsqueeze(0)


def source_token_index(grid: int, tile_patches: int, cell_patches: int, tokens_per_side: int,
                       offset_row: int, offset_col: int, device=None) -> Tensor:
    """Intact-crop tokens of each tile, in home order. [grid^2, tile_patches^2]"""
    rows = torch.arange(grid, device=device).repeat_interleave(grid)
    cols = torch.arange(grid, device=device).repeat(grid)
    top_left = ((offset_row + rows * cell_patches) * tokens_per_side
                + offset_col + cols * cell_patches)
    steps = torch.arange(tile_patches, device=device)
    within = (steps.unsqueeze(1) * tokens_per_side + steps).reshape(-1)
    return top_left.unsqueeze(1) + within.unsqueeze(0)


def neighbour_pairs(tile_at_slot: Tensor, grid: int):
    """Home index of each slot's true neighbour per direction, and a validity mask."""
    rows = torch.div(tile_at_slot, grid, rounding_mode="floor")
    cols = tile_at_slot % grid
    off = torch.tensor(OFFSETS, device=tile_at_slot.device)
    r = rows.unsqueeze(-1) + off[:, 0]
    c = cols.unsqueeze(-1) + off[:, 1]
    valid = (r >= 0) & (r < grid) & (c >= 0) & (c < grid)
    partner = r.clamp(0, grid - 1) * grid + c.clamp(0, grid - 1)
    return partner, valid


def build_view(images: Tensor, grid: int, tile_patches: int, gap_patches: int, patch_size: int,
               random_offset: bool = True, perm: Optional[Tensor] = None) -> Dict[str, Tensor]:
    """Cut the crop into grid x grid tiles, shuffle them, reassemble into a mosaic.

    A gap of `gap_patches` is dropped between tiles so no two tiles continue each other.
    `perm` gives the arrangement explicitly; otherwise tiles are shuffled freely.
    """
    B, C, H, W = images.shape
    device = images.device
    n_tiles = grid * grid
    cell = tile_patches + gap_patches
    span = grid * cell
    tokens_per_side = H // patch_size
    assert span <= tokens_per_side, f"{grid}x{grid} of {tile_patches}p tiles needs {span} patches, crop has {tokens_per_side}"

    slack = tokens_per_side - span
    if random_offset and slack > 0:
        off_r, off_c = random.randrange(slack + 1), random.randrange(slack + 1)
    else:
        off_r = off_c = slack // 2

    cell_px, tile_px = cell * patch_size, tile_patches * patch_size
    region = images[:, :, off_r * patch_size:(off_r + span) * patch_size,
                          off_c * patch_size:(off_c + span) * patch_size]
    cells = region.unfold(2, cell_px, cell_px).unfold(3, cell_px, cell_px)
    tiles = cells[..., :tile_px, :tile_px].permute(0, 2, 3, 1, 4, 5).reshape(B, n_tiles, C, tile_px, tile_px)

    if perm is not None:
        tile_at_slot = perm.to(device)
    else:
        tile_at_slot = torch.rand(B, n_tiles, device=device).argsort(-1)

    placed = torch.gather(tiles, 1, tile_at_slot.view(B, n_tiles, 1, 1, 1).expand(-1, -1, C, tile_px, tile_px))
    mosaic = placed.reshape(B, grid, grid, C, tile_px, tile_px)
    mosaic = mosaic.permute(0, 3, 1, 4, 2, 5).reshape(B, C, grid * tile_px, grid * tile_px)
    partner, valid = neighbour_pairs(tile_at_slot, grid)

    return {
        "mosaic": mosaic,
        "slot_index": slot_token_index(grid, tile_patches, device),
        "source_index": source_token_index(grid, tile_patches, cell, tokens_per_side, off_r, off_c, device),
        "tile_at_slot": tile_at_slot,
        "partner": partner,
        "valid": valid,
        "offset": (off_r, off_c),
        "grid": grid,
    }


def gather_tiles(tokens: Tensor, index: Tensor) -> Tensor:
    """Patch tokens of each tile. [B, T, D] x [N, P] -> [B, N, P, D]"""
    B, _, D = tokens.shape
    idx = index.reshape(1, -1, 1).expand(B, -1, D)
    return torch.gather(tokens, 1, idx).reshape(B, index.shape[0], index.shape[1], D)


def slot_to_home(tiles_slot: Tensor, tile_at_slot: Tensor) -> Tensor:
    """Reorder per-slot features into home order."""
    slot_of_home = torch.argsort(tile_at_slot, dim=1)
    return torch.gather(tiles_slot, 1, slot_of_home.unsqueeze(-1).expand(-1, -1, tiles_slot.shape[-1]))


def block_members(grid: int, block: int, device=None) -> Tensor:
    """Home indices of the tiles in each block. [(grid/block)^2, block^2]"""
    assert grid % block == 0
    side = grid // block
    br = torch.arange(side, device=device).repeat_interleave(side)
    bc = torch.arange(side, device=device).repeat(side)
    dr = torch.arange(block, device=device).repeat_interleave(block)
    dc = torch.arange(block, device=device).repeat(block)
    rows = br.unsqueeze(1) * block + dr
    cols = bc.unsqueeze(1) * block + dc
    return rows * grid + cols


def pool_blocks(tiles_home: Tensor, members: Tensor) -> Tensor:
    """Mean the tiles of each block into one vector."""
    return tiles_home[:, members].mean(2)
