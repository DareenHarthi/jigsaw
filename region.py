# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# Written as an extension of DINOv3 (https://github.com/facebookresearch/dinov3).

"""Region prediction: locate a crop taken from inside the teacher's view.

A small crop is cut from anywhere in the view, tiled and shuffled. Each crop tile scores
against every teacher tile; the columns are summed within a region and averaged over crop
tiles. The target is the crop's normalised area overlap with each region, so nothing is
averaged in feature space.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn.init import trunc_normal_


class RegionHead(nn.Module):
    """Separate projections for crop tiles and teacher tiles, then all-pairs similarity."""

    def __init__(self, embed_dim: int, proj_dim: int = 256):
        super().__init__()
        self.proj_student = nn.Linear(embed_dim, proj_dim)
        self.proj_teacher = nn.Linear(embed_dim, proj_dim)
        self.scale = 1.0 / math.sqrt(proj_dim)

    def init_weights(self):
        self.apply(_init_linear)

    def forward(self, crop_tiles: Tensor, teacher_tiles: Tensor) -> Tensor:
        q = self.proj_student(crop_tiles)
        k = self.proj_teacher(teacher_tiles)
        return (q @ k.transpose(1, 2)) * self.scale


def _init_linear(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def region_targets(offset: Tensor, span: int, view_patches: int, grid: int, block: int) -> Tensor:
    """Normalised area overlap between the crop and each region. [B, (grid/block)^2]"""
    side = grid // block
    region_px = view_patches / side
    dev = offset.device
    lo = offset.float()
    hi = lo + span
    edges = torch.arange(side + 1, device=dev, dtype=torch.float32) * region_px
    r_lo, r_hi = edges[:-1], edges[1:]
    ov_r = (torch.minimum(hi[:, 0:1], r_hi) - torch.maximum(lo[:, 0:1], r_lo)).clamp(min=0)
    ov_c = (torch.minimum(hi[:, 1:2], r_hi) - torch.maximum(lo[:, 1:2], r_lo)).clamp(min=0)
    area = ov_r.unsqueeze(2) * ov_c.unsqueeze(1)
    area = area.reshape(area.shape[0], -1)
    return area / area.sum(-1, keepdim=True).clamp(min=1e-6)


class RegionLoss(nn.Module):
    """Soft cross-entropy of the region scores against the overlap target."""

    def forward(self, scores: Tensor, target: Tensor) -> Tensor:
        return -(target * scores.log_softmax(-1)).sum(-1).mean()

    @staticmethod
    @torch.no_grad()
    def accuracy(scores: Tensor, target: Tensor) -> Tensor:
        return (scores.argmax(-1) == target.argmax(-1)).float().mean()
