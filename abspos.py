# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# Written as an extension of DINOv3 (https://github.com/facebookresearch/dinov3).

"""Absolute position: place each shuffled tile back on the grid.

The target is smoothed over neighbouring cells with a Gaussian whose width anneals to zero,
so early training is graded by distance and late training asks for the exact cell.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn.init import trunc_normal_


class AbsPosHead(nn.Module):
    """Scores every slot against every home cell of the teacher."""

    def __init__(self, embed_dim: int, proj_dim: int = 256):
        super().__init__()
        self.proj_student = nn.Linear(embed_dim, proj_dim)
        self.proj_teacher = nn.Linear(embed_dim, proj_dim)
        self.scale = 1.0 / math.sqrt(proj_dim)

    def init_weights(self):
        self.apply(_init_linear)

    def forward(self, slots: Tensor, cells: Tensor) -> Tensor:
        return (self.proj_student(slots) @ self.proj_teacher(cells).transpose(1, 2)) * self.scale


def _init_linear(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def grid_distance2(grid: int, device=None) -> Tensor:
    """Squared grid distance between every pair of cells. [grid^2, grid^2]"""
    r = torch.arange(grid, device=device).repeat_interleave(grid).float()
    c = torch.arange(grid, device=device).repeat(grid).float()
    return (r[:, None] - r[None, :]) ** 2 + (c[:, None] - c[None, :]) ** 2


def smoothing_matrix(d2: Tensor, sigma: float) -> Tensor:
    """Row-normalised exp(-d^2 / sigma^2); a one-hot target as sigma -> 0."""
    if sigma <= 0:
        return torch.eye(d2.shape[0], device=d2.device, dtype=d2.dtype)
    w = torch.exp(-d2 / (sigma ** 2))
    return w / w.sum(-1, keepdim=True)


def anneal_sigma(it: int, start_iter: int, end_iter: int, sigma_start=1.0, sigma_end=0.0) -> float:
    if it <= start_iter:
        return sigma_start
    if it >= end_iter:
        return sigma_end
    f = (it - start_iter) / max(end_iter - start_iter, 1)
    return sigma_start + f * (sigma_end - sigma_start)


class AbsPosLoss(nn.Module):
    """Soft cross-entropy of the cell scores against the smoothed target."""

    def forward(self, logits: Tensor, home: Tensor, smooth: Tensor) -> Tensor:
        return -(smooth[home] * logits.log_softmax(-1)).sum(-1).mean()

    @staticmethod
    @torch.no_grad()
    def accuracies(logits: Tensor, home: Tensor, grid: int):
        pred = logits.argmax(-1)
        exact = (pred == home).float().mean()
        pr, pc = torch.div(pred, grid, rounding_mode="floor"), pred % grid
        hr, hc = torch.div(home, grid, rounding_mode="floor"), home % grid
        near = ((pr - hr).abs() <= 1) & ((pc - hc).abs() <= 1)
        return exact, near.float().mean()
