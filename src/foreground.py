"""
Foreground / background separation.

Motivation (Multi-Flow, Real-IAD MVAD):
  Background regions (conveyor belts, jigs, uniform fixtures) produce
  spurious high distances in memory-bank models because the background
  appearance varies a lot between views/classes but is always "normal".
  Suppressing those pixels from the anomaly map gives a consistent P-AP
  boost (+1-3pp) with NO extra model or data.

We provide a zero-cost foreground signal: the CLS→patch attention from
DINOv2's FINAL block (extracted via a hook during the SAME forward pass
that produces the patch features). This is the original DINO self-
segmentation signal (Caron et al. 2021) and requires no extra compute.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def foreground_from_last_attn(last_attn: torch.Tensor,
                              out_size: int,
                              percentile: float = 35.0,
                              smooth_sigma: float = 2.0) -> torch.Tensor:
    """
    Build a soft foreground mask from DINOv2's final-block CLS attention.

    last_attn : (B, Hp, Wp) attention mass the CLS token puts on each patch
                (already averaged over heads).
    out_size  : target spatial size (H, W) for the mask, e.g. 448.

    Returns
    -------
    fg : (B, 1, out_size, out_size) soft mask in [0, 1], high = foreground.
    """
    if last_attn is None:
        return None
    B, Hp, Wp = last_attn.shape
    a = last_attn.float().unsqueeze(1)  # (B, 1, Hp, Wp)
    # Upsample to output resolution
    a = F.interpolate(a, size=(out_size, out_size), mode="bilinear",
                      align_corners=False)
    # Separable Gaussian smoothing to avoid patch-grid artefacts
    if smooth_sigma > 0:
        k = int(smooth_sigma * 4) | 1
        ax = torch.arange(k, device=a.device, dtype=a.dtype) - k // 2
        g = torch.exp(-0.5 * (ax / max(smooth_sigma, 1e-3)) ** 2)
        g = g / g.sum()
        a = F.conv2d(a, g.view(1, 1, -1, 1).repeat(1, 1, 1, 1),
                     padding=(k // 2, 0), groups=1)
        a = F.conv2d(a, g.view(1, 1, 1, -1).repeat(1, 1, 1, 1),
                     padding=(0, k // 2), groups=1)
    # Percentile-based soft threshold (Otsu-like): anything ABOVE percentile
    # is foreground. Sigmoid ramp width is tied to the std of the attention
    # map so the transition adapts per image.
    flat = a.reshape(B, -1)
    # kthvalue is 1-indexed
    kth = max(1, min(flat.shape[-1],
                     int(flat.shape[-1] * percentile / 100.0)))
    t = torch.kthvalue(flat, kth, dim=-1).values.reshape(B, 1, 1, 1)
    std = flat.std(dim=-1).reshape(B, 1, 1, 1).clamp(min=1e-4)
    fg = torch.sigmoid((a - t) / (std * 0.5))
    return fg  # (B,1,H,W)


def apply_foreground(anomaly_map: torch.Tensor,
                     fg_mask: torch.Tensor | None,
                     lam: float = 0.15) -> torch.Tensor:
    """
    Suppress background pixels in the anomaly map.

    anomaly_map : (B, H, W) or (V, H, W) calibrated anomaly scores [0,1]
    fg_mask     : (B, 1, H, W) soft foreground in [0,1], same batch size
                  as anomaly_map (or broadcastable -- when V views share
                  one foreground mask per micro-batch, pass fg already
                  reshaped).
    lam         : floor for background pixels (0 = zero background out,
                  1 = no suppression). Literature uses 0.1-0.25.

    Returns
    -------
    suppressed : same shape as anomaly_map, scores multiplied by
                 (lam + (1-lam)*fg) so background is suppressed but not
                 killed (preserves small defects near object borders).
    """
    if fg_mask is None:
        return anomaly_map
    if fg_mask.dim() == 4 and fg_mask.shape[1] == 1:
        if anomaly_map.dim() == 3:
            # (B,1,H,W) * (B,H,W) -> assume broadcast along B==V (micro-batch)
            m = fg_mask.squeeze(1)
        else:
            m = fg_mask
    else:
        m = fg_mask
    return anomaly_map * (lam + (1.0 - lam) * m)


# Pure-PyTorch connected-component size suppression (no scipy dependency).
# Uses a simple iterative label-propagation on GPU; fast enough for 448x448.
def _torch_connected_components(binary: torch.Tensor) -> torch.Tensor:
    """
    binary : (H, W) uint8/bool tensor on CPU or CUDA.
    Returns labels (H, W) int64 tensor (0 = background).
    Simple two-pass algorithm (no union-find flatten pass 2 -- labels may be
    non-canonical; we only need per-component sizes which are invariant
    after a single pass).
    """
    H, W = binary.shape
    lab = torch.zeros(H, W, dtype=torch.int64, device=binary.device)
    if not bool(binary.any()):
        return lab
    # Pass 1: label with left/up merge
    next_label = 1
    parent = {}
    b = binary.to(torch.bool)
    for y in range(H):
        for x in range(W):
            if not b[y, x]:
                continue
            nbs = []
            if y > 0 and lab[y-1, x] > 0:
                nbs.append(int(lab[y-1, x].item()))
            if x > 0 and lab[y, x-1] > 0:
                nbs.append(int(lab[y, x-1].item()))
            if not nbs:
                lab[y, x] = next_label
                parent[next_label] = next_label
                next_label += 1
            else:
                root = min(nbs)
                lab[y, x] = root
                for u in nbs:
                    # union: attach larger to smaller
                    ru = _find(parent, u)
                    rr = _find(parent, root)
                    if ru != rr:
                        parent[max(ru, rr)] = min(ru, rr)
    # Pass 2: canonicalise labels via parent flatten
    for y in range(H):
        for x in range(W):
            if lab[y, x] > 0:
                lab[y, x] = _find(parent, int(lab[y, x].item()))
    return lab


def _find(parent, x):
    # path compression
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def remove_small_components_torch(mask: torch.Tensor,
                                  threshold: float = 0.30,
                                  min_area: int = 25,
                                  min_area_kill: float = 0.1) -> torch.Tensor:
    """
    Soft-suppress tiny connected components in an anomaly map.

    mask : (H, W) float tensor in [0,1]
    threshold : binarisation threshold for component labelling.
    min_area  : components smaller than this are multiplied by min_area_kill.
    min_area_kill : multiplier for tiny components.

    Works on CPU or CUDA (CUDA path falls back to CPU + .cpu()/.to() which is
    cheap for a 448x448 map); no scipy dependency.
    """
    if mask.numel() == 0:
        return mask
    dev = mask.device
    m = mask.detach().float().cpu()
    binary = (m > threshold)
    lab = _torch_connected_components(binary)
    if lab.max() == 0:
        return mask
    # Compute sizes
    sizes = torch.bincount(lab.reshape(-1), minlength=int(lab.max().item()) + 1)
    # Build a multiplier map
    mult = torch.ones_like(m)
    for li in range(1, sizes.shape[0]):
        if sizes[li] < min_area and sizes[li] > 0:
            mult[lab == li] = min_area_kill
    out = m * mult
    return out.to(dev).clamp_(0.0, 1.0)
