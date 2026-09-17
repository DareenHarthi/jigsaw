# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# Written as an extension of DINOv3 (https://github.com/facebookresearch/dinov3).

"""NeCo: patch nearest-neighbour ordering, after Pariza et al. (arXiv:2408.11054).

Student and teacher see two crops of the same image. Over the overlapping region, each patch
is ranked by similarity against a shared pool of reference patches, and the student is trained
to reproduce the teacher's ordering through a differentiable sort.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from torchvision.ops import roi_align


class NeCoHead(nn.Module):
    """One projector per model; the teacher's copy is the EMA of this one."""

    def __init__(self, embed_dim: int, proj_dim: int = 256, hidden_dim: int = 2048, nlayers: int = 3):
        super().__init__()
        layers, d = [], embed_dim
        for _ in range(max(nlayers - 1, 0)):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            d = hidden_dim
        layers += [nn.Linear(d, proj_dim)]
        self.mlp = nn.Sequential(*layers)

    def init_weights(self):
        self.apply(_init_linear)

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.normalize(self.mlp(x), dim=-1)


def _init_linear(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def overlap_rois(boxes: Tensor, crop_size: int):
    """Boxes of the shared region in each crop's own frame, or None if the crops miss."""
    (i0, j0, h0, w0, f0), (i1, j1, h1, w1, f1) = boxes[0].tolist(), boxes[1].tolist()
    top, left = max(i0, i1), max(j0, j1)
    bot, right = min(i0 + h0, i1 + h1), min(j0 + w0, j1 + w1)
    if bot <= top or right <= left:
        return None
    out = []
    for (i, j, h, w, flip) in ((i0, j0, h0, w0, f0), (i1, j1, h1, w1, f1)):
        x0, x1 = (left - j) * crop_size / w, (right - j) * crop_size / w
        y0, y1 = (top - i) * crop_size / h, (bot - i) * crop_size / h
        if flip:
            x0, x1 = crop_size - x1, crop_size - x0
        out.append([x0, y0, x1, y1])
    return out


def sample_references(tokens: Tensor, n_refs: int, generator=None) -> Tensor:
    """Reference patches pooled across the whole batch, not per image."""
    flat = tokens.reshape(-1, tokens.shape[-1])
    idx = torch.randperm(flat.shape[0], device=flat.device, generator=generator)[:n_refs]
    return flat[idx]


class NeCoLoss(nn.Module):
    """Cross-entropy between the student's and teacher's nearest-neighbour orderings."""

    def __init__(self, n_refs: int = 49, steepness: float = 100.0, sort_net: str = "bitonic"):
        super().__init__()
        from diffsort import DiffSortNet
        self.sorter = DiffSortNet(sorting_network_type=sort_net, size=n_refs, steepness=steepness)

    def forward(self, s_sim: Tensor, t_sim: Tensor) -> Tensor:
        _, s_perm = self.sorter(s_sim)
        with torch.no_grad():
            _, t_perm = self.sorter(t_sim)
        return -(t_perm * (s_perm + 1e-9).log()).sum(-1).mean()

    @staticmethod
    @torch.no_grad()
    def rank_agreement(s_sim: Tensor, t_sim: Tensor) -> Tensor:
        return (s_sim.argmax(-1) == t_sim.argmax(-1)).float().mean()


def aligned_patches(tokens: Tensor, boxes, crop_size: int, kernel: int = 7) -> Tensor:
    """Patch grid of the overlapping region, resampled to kernel x kernel."""
    B, P, D = tokens.shape
    g = int(math.sqrt(P))
    grid = tokens.transpose(1, 2).reshape(B, D, g, g).float()
    scale = g / crop_size
    rois = [torch.tensor([[b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale]],
                         device=tokens.device) for b in boxes]
    return roi_align(grid, rois, output_size=(kernel, kernel), aligned=True)
