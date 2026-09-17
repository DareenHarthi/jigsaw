# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# Written as an extension of DINOv3 (https://github.com/facebookresearch/dinov3).

"""Boundary-centric masked modelling, after LingBot-Vision (arXiv:2607.05247).

The teacher predicts a dense 4-channel boundary field (distance and three angles locating a
line segment), decodes it into segments anchored on corner points, keeps the ones an
a-contrario test can justify, and re-renders them. Tokens the surviving boundaries cross are
forced into iBOT's mask and additionally supervised with a categorical loss on the field.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

N_CHANNELS = 4
CIRCULAR = (False, True, False, False)
PHI_LIMIT = math.radians(88.0)
ALIGN_TOL = math.pi / 16
P_CHANCE = 1.0 / 16.0
RECT_WIDTH = 3
MIN_DENSITY = 0.5
GRAD_THRESHOLD = 2.0 / math.sin(ALIGN_TOL)


def normalise(d, theta, phi1, phi2, tau_d):
    """Pack the raw channels into [0, 1] on a channel axis at dim 1."""
    return torch.stack([
        (d / tau_d).clamp(0, 1),
        (theta % (2 * math.pi)) / (2 * math.pi),
        (phi1.clamp(-PHI_LIMIT, PHI_LIMIT) + math.pi / 2) / math.pi,
        (phi2.clamp(-PHI_LIMIT, PHI_LIMIT) + math.pi / 2) / math.pi,
    ], dim=1)


def denormalise(attr, tau_d):
    d, theta, phi1, phi2 = attr.unbind(dim=1)
    return d * tau_d, theta * 2 * math.pi, phi1 * math.pi - math.pi / 2, phi2 * math.pi - math.pi / 2


def render_field(segments: Tensor, valid: Tensor, size, tau_d: float, chunk: int = 8):
    """Rasterise segments into a dense field. Returns [B, 4, H, W] and a support mask."""
    B, S, _ = segments.shape
    H, W = size
    dev, dt = segments.device, segments.dtype
    P = H * W
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=dt),
                            torch.arange(W, device=dev, dtype=dt), indexing="ij")
    px, py = xs.reshape(1, P, 1), ys.reshape(1, P, 1)
    attr = torch.zeros(B, N_CHANNELS, P, device=dev, dtype=dt)
    got_all = torch.zeros(B, P, dtype=torch.bool, device=dev)

    for i in range(0, B, chunk):
        sl = slice(i, min(i + chunk, B))
        seg, ok = segments[sl], valid[sl]
        ax, ay = seg[:, None, :, 0], seg[:, None, :, 1]
        bx, by = seg[:, None, :, 2], seg[:, None, :, 3]
        vx, vy = bx - ax, by - ay
        L2 = (vx * vx + vy * vy).clamp_min(1e-6)
        L = L2.sqrt()
        wx, wy = px - ax, py - ay
        t = (wx * vx + wy * vy) / L2
        dist = (wx * vy - wy * vx).abs() / L
        inside = (t >= 0) & (t <= 1) & (dist <= tau_d) & ok[:, None, :]
        d_sel = torch.where(inside, dist, torch.full_like(dist, float("inf")))
        d_min, arg = d_sel.min(dim=2)
        got = torch.isfinite(d_min)

        g = arg.unsqueeze(-1)
        pick = lambda v: v.expand(-1, P, -1).gather(2, g).squeeze(-1)
        ax_g, ay_g, bx_g, by_g = pick(ax), pick(ay), pick(bx), pick(by)
        t_g = t.gather(2, g).squeeze(-1)
        pxs, pys = px.squeeze(-1), py.squeeze(-1)
        fx, fy = ax_g + t_g * (bx_g - ax_g), ay_g + t_g * (by_g - ay_g)
        ux, uy = fx - pxs, fy - pys
        dd = torch.hypot(ux, uy).clamp_min(1e-6)
        ux, uy = ux / dd, uy / dd
        theta = torch.atan2(uy, ux)
        ang = lambda ex, ey: torch.atan2(ux * (ey - pys) - uy * (ex - pxs),
                                         ux * (ex - pxs) + uy * (ey - pys))
        a = normalise(d_min.masked_fill(~got, 0.0), theta, ang(ax_g, ay_g), ang(bx_g, by_g), tau_d)
        attr[sl] = torch.where(got.unsqueeze(1), a, torch.zeros_like(a))
        got_all[sl] = got

    return attr.reshape(B, N_CHANNELS, H, W), got_all.reshape(B, H, W)


def decode_chords(attr: Tensor, tau_d: float) -> Tensor:
    """Read a full segment proposal out of every field position. [B, H*W, 4]"""
    B, _, H, W = attr.shape
    d, theta, phi1, phi2 = denormalise(attr, tau_d)
    ys, xs = torch.meshgrid(torch.arange(H, device=attr.device, dtype=attr.dtype),
                            torch.arange(W, device=attr.device, dtype=attr.dtype), indexing="ij")
    ux, uy = torch.cos(theta), torch.sin(theta)
    fx, fy = xs + d * ux, ys + d * uy
    nx, ny = -uy, ux
    lim = math.tan(PHI_LIMIT)
    t1, t2 = torch.tan(phi1).clamp(-lim, lim), torch.tan(phi2).clamp(-lim, lim)
    e1 = torch.stack([fx + d * t1 * nx, fy + d * t1 * ny], -1)
    e2 = torch.stack([fx + d * t2 * nx, fy + d * t2 * ny], -1)
    return torch.cat([e1, e2], -1).reshape(B, H * W, 4)


def soft_labels(values: Tensor, n_bins: int, sigma_bins: float = 1.5) -> Tensor:
    """Narrow categorical labels; theta uses circular bins. [..., 4, K]"""
    centres = (torch.arange(n_bins, device=values.device, dtype=values.dtype) + 0.5) / n_bins
    delta = values.unsqueeze(-1) - centres
    circ = torch.tensor(CIRCULAR, device=values.device).reshape(*([1] * (values.dim() - 1)), N_CHANNELS, 1)
    delta = torch.where(circ, (delta + 0.5) % 1.0 - 0.5, delta)
    return (-delta.pow(2) / (2.0 * (sigma_bins / n_bins) ** 2)).softmax(-1)


def boundary_tokens(valid: Tensor, patch_size: int, stride: int) -> Tensor:
    """A token is a boundary token when a boundary falls inside its patch."""
    r = patch_size // stride
    B, H, W = valid.shape
    return valid.reshape(B, H // r, r, W // r, r).amax(dim=(2, 4))


def shi_tomasi_corners(gray: Tensor, max_corners: int = 32, nms_radius: int = 6,
                       rel_threshold: float = 0.01):
    """Corner points to anchor the decoding on."""
    B, H, W = gray.shape
    g = gray.unsqueeze(1).float()
    kx = torch.tensor([[-1.0, 0.0, 1.0]], device=g.device).view(1, 1, 1, 3) / 2
    gx = F.conv2d(F.pad(g, (1, 1, 0, 0), mode="replicate"), kx)
    gy = F.conv2d(F.pad(g, (0, 0, 1, 1), mode="replicate"), kx.view(1, 1, 3, 1))
    box = lambda t: F.avg_pool2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    a, b, c = box(gx * gx), box(gx * gy), box(gy * gy)
    tr, det = a + c, a * c - b * b
    score = (tr / 2 - ((tr / 2) ** 2 - det).clamp_min(0).sqrt()).squeeze(1)
    peak = F.max_pool2d(score.unsqueeze(1), 2 * nms_radius + 1, 1, nms_radius).squeeze(1)
    keep = (score >= peak) & (score > rel_threshold * score.amax(dim=(1, 2), keepdim=True))
    flat = score.masked_fill(~keep, -1.0).reshape(B, -1)
    val, idx = flat.topk(min(max_corners, flat.shape[1]), dim=1)
    return torch.stack([idx % W, idx // W], -1).float(), val > 0


def level_line_field(gray: Tensor, grad_threshold: float = GRAD_THRESHOLD):
    """LSD's level-line orientation, and where the gradient is strong enough to trust it."""
    g = gray.float()
    a, b = g[:, :-1, :-1], g[:, :-1, 1:]
    c, d = g[:, 1:, :-1], g[:, 1:, 1:]
    gx = F.pad(((b - a) + (d - c)) / 2, (0, 1, 0, 1), mode="replicate")
    gy = F.pad(((c - a) + (d - b)) / 2, (0, 1, 0, 1), mode="replicate")
    return torch.atan2(gx, -gy), torch.hypot(gx, gy) > grad_threshold


def guide_theta(field: Tensor, angle: Tensor, defined: Tensor, weight: float) -> Tensor:
    """Borrow the image's level lines for the orientation channel while bootstrapping."""
    if weight <= 0:
        return field
    guided = ((angle - math.pi / 2) % (2 * math.pi)) / (2 * math.pi)
    take = defined & (torch.rand(angle.shape, device=angle.device) < weight)
    out = field.clone()
    out[:, 1] = torch.where(take, guided.to(field.dtype), field[:, 1])
    return out


def snap_and_vote(chords: Tensor, corners: Tensor, corner_valid: Tensor,
                  min_votes: int = 2, max_candidates: int = 256, chunk: int = 16):
    """Snap proposal endpoints to corners; corner pairs with enough votes are candidates."""
    B, P, _ = chords.shape
    C = corners.shape[1]
    dev = chords.device
    seg = torch.zeros(B, max_candidates, 4, device=dev, dtype=chords.dtype)
    ok = torch.zeros(B, max_candidates, dtype=torch.bool, device=dev)

    for i in range(0, B, chunk):
        sl = slice(i, min(i + chunk, B))
        ch, co, cv = chords[sl], corners[sl], corner_valid[sl]
        big = torch.finfo(ch.dtype).max / 4
        nearest = lambda pt: (pt.unsqueeze(2) - co.unsqueeze(1)).pow(2).sum(-1) \
            .masked_fill(~cv.unsqueeze(1), big).argmin(-1)
        i1, i2 = nearest(ch[..., :2]), nearest(ch[..., 2:])
        lo, hi = torch.minimum(i1, i2), torch.maximum(i1, i2)
        good = (lo != hi) & cv.gather(1, lo) & cv.gather(1, hi)
        pair = (lo * C + hi).masked_fill(~good, C * C)
        counts = torch.zeros(ch.shape[0], C * C + 1, device=dev, dtype=torch.long)
        counts.scatter_add_(1, pair, torch.ones_like(pair))
        n = min(max_candidates, C * C)
        v, flat = counts[:, :C * C].topk(n, dim=1)
        a_idx, b_idx = flat // C, flat % C
        pa = torch.gather(co, 1, a_idx.unsqueeze(-1).expand(-1, -1, 2))
        pb = torch.gather(co, 1, b_idx.unsqueeze(-1).expand(-1, -1, 2))
        seg[sl, :n] = torch.cat([pa, pb], -1)
        ok[sl, :n] = v >= min_votes
    return seg, ok


def _log_binom_tail(n: Tensor, k: Tensor, p: float, n_max: int) -> Tensor:
    i = torch.arange(n_max + 1, device=n.device, dtype=torch.float64)
    nn_, kk = n.unsqueeze(-1).double(), k.unsqueeze(-1).double()
    lg = torch.lgamma
    term = (lg(nn_ + 1) - lg(i + 1) - lg((nn_ - i).clamp_min(0) + 1)
            + i * math.log(p) + (nn_ - i) * math.log1p(-p))
    return torch.logsumexp(term.masked_fill((i < kk) | (i > nn_), float("-inf")), dim=-1)


def nfa_validate(segments: Tensor, valid: Tensor, angle: Tensor, defined: Tensor,
                 n_samples: int = 256, p_chance: float = P_CHANCE, align_tol: float = ALIGN_TOL):
    """Keep only segments whose support cannot be explained by chance (a-contrario test)."""
    B, N, _ = segments.shape
    H, W = angle.shape[-2:]
    a = segments[..., :2]
    d = segments[..., 2:] - a
    length = d.norm(dim=-1).clamp_min(1e-6)
    u = d / length.unsqueeze(-1)
    nrm = torch.stack([-u[..., 1], u[..., 0]], -1)
    t = torch.linspace(0, 1, n_samples, device=segments.device).view(1, 1, n_samples, 1)
    off = torch.arange(RECT_WIDTH, device=segments.device, dtype=segments.dtype) - (RECT_WIDTH - 1) / 2
    pts = (a[:, :, None, None, :] + t.unsqueeze(-2) * d[:, :, None, None, :]
           + off.view(1, 1, 1, RECT_WIDTH, 1) * nrm[:, :, None, None, :])
    xr, yr = pts[..., 0].round().long(), pts[..., 1].round().long()
    in_bounds = (xr >= 0) & (xr < W) & (yr >= 0) & (yr < H)
    flat = (yr.clamp(0, H - 1) * W + xr.clamp(0, W - 1)).reshape(B, -1)
    samp_a = torch.gather(angle.reshape(B, -1), 1, flat).reshape(B, N, n_samples, RECT_WIDTH)
    samp_d = torch.gather(defined.reshape(B, -1).float(), 1, flat).reshape(B, N, n_samples, RECT_WIDTH) > 0

    seg_angle = torch.atan2(u[..., 1], u[..., 0])[:, :, None, None]
    diff = (samp_a - seg_angle) % math.pi
    diff = torch.minimum(diff, math.pi - diff)
    hit = (samp_d & (diff <= align_tol) & in_bounds).any(dim=3)
    on = in_bounds.any(dim=3)

    # the width is a search band for the one-pixel gradient ridge, so correct the null for it
    p_eff = 1.0 - (1.0 - p_chance) ** RECT_WIDTH
    scale = length / n_samples
    n_pix = (on.sum(2).float() * scale).round()
    k_pix = (hit.sum(2).float() * scale).round()
    n_max = int(math.ceil(math.hypot(H, W))) + 1
    n_c = n_pix.clamp(0, n_max)
    log_nfa = 2.5 * math.log(float(H * W)) + _log_binom_tail(n_c, torch.minimum(k_pix, n_c), p_eff, n_max)
    density = k_pix / n_pix.clamp_min(1.0)
    return valid & (n_pix > 0) & (density >= MIN_DENSITY) & (log_nfa <= 0.0), log_nfa.float()


class BoundaryHead(nn.Module):
    """Per-token MLP unfolded to sub-token positions, scored against learned bin prototypes."""

    def __init__(self, in_dim: int, head_dim: int = 512, n_bins: int = 128,
                 patch_size: int = 16, stride: int = 2, nlayers: int = 3, hidden_dim: int = 2048):
        super().__init__()
        self.r = patch_size // stride
        self.n_positions = self.r * self.r
        self.n_bins = n_bins
        layers, d = [], in_dim
        for _ in range(max(nlayers - 1, 0)):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            d = hidden_dim
        layers += [nn.Linear(d, head_dim)]
        self.mlp = nn.Sequential(*layers)
        self.tile_pos = nn.Parameter(torch.zeros(self.n_positions, head_dim))
        self.mix = nn.Sequential(nn.Linear(head_dim, head_dim), nn.GELU(), nn.Linear(head_dim, head_dim))
        self.prototypes = nn.Parameter(torch.zeros(N_CHANNELS, n_bins, head_dim))

    def init_weights(self):
        self.apply(_init_linear)
        trunc_normal_(self.tile_pos, std=0.5)
        trunc_normal_(self.prototypes, std=0.02)

    def forward(self, tokens: Tensor, readout: bool = False, temperature: float = 0.08,
                chunk: int = 2048, standardize: bool = True) -> Tensor:
        if readout:
            return self._readout(tokens, temperature, chunk, standardize)
        return self._logits(tokens)

    def _logits(self, tokens: Tensor) -> Tensor:
        x = self.mix(self.mlp(tokens).unsqueeze(1) + self.tile_pos)
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.prototypes, dim=-1)
        return torch.einsum("npd,ckd->ncpk", x, w)

    @torch.no_grad()
    def _readout(self, tokens, temperature, chunk, standardize):
        """Continuous field values: the expectation over bin centres, circular for theta."""
        centres = (torch.arange(self.n_bins, device=tokens.device, dtype=torch.float32) + 0.5) / self.n_bins
        circ = torch.tensor(CIRCULAR, device=tokens.device).view(1, N_CHANNELS, 1)
        out = []
        for i in range(0, tokens.shape[0], chunk):
            lg = self._logits(tokens[i:i + chunk]).float()
            if standardize:
                lg = (lg - lg.mean(-1, keepdim=True)) / lg.std(-1, keepdim=True).clamp_min(1e-6)
            p = (lg / temperature).softmax(-1)
            ang = 2 * math.pi * centres
            s, c = (p * ang.sin()).sum(-1), (p * ang.cos()).sum(-1)
            out.append(torch.where(circ, (torch.atan2(s, c) % (2 * math.pi)) / (2 * math.pi), (p * centres).sum(-1)))
        return torch.cat(out, 0)


def _init_linear(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BoundaryLoss(nn.Module):
    """Cross-entropy of the student's field against the validated, re-rendered target."""

    def __init__(self, n_bins: int = 128, label_sigma_bins: float = 1.5, student_temp: float = 0.1):
        super().__init__()
        self.n_bins, self.sigma, self.temp = n_bins, label_sigma_bins, student_temp

    def build_targets(self, values: Tensor, in_support: Tensor) -> Tensor:
        """Positions off the boundary get a random label, not a constant background class."""
        v = values.permute(0, 2, 1)
        v = torch.where(in_support.unsqueeze(-1), v, torch.rand_like(v))
        return soft_labels(v, self.n_bins, self.sigma).permute(0, 2, 1, 3)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.numel() == 0:
            return logits.sum() * 0.0
        return -(targets * (logits / self.temp).log_softmax(-1)).sum(-1).mean()

    @staticmethod
    @torch.no_grad()
    def accuracy(logits: Tensor, targets: Tensor) -> Tensor:
        if logits.numel() == 0:
            return logits.new_zeros(())
        return (logits.argmax(-1) == targets.argmax(-1)).float().mean()
