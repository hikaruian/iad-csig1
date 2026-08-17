"""
Foreground / background separation for anomaly maps.

Motivation (Multi-Flow, Real-IAD MVAD):
  Background regions (conveyor belts, jigs, uniform fixtures) produce
  spurious high 1-NN distances in memory-bank models because the
  background appearance varies between views/classes but is always
  "normal". Soft-suppressing those pixels in the anomaly map gives a
  consistent P-AP boost without hurting image-level scoring.

We use a zero-cost signal: the cosine similarity between DINOv2's final-
norm CLS token and each final-norm patch token (the original DINO self-
segmentation signal, Caron et al. 2021). This is extracted during the
SAME forward pass that produces patch features -- no extra compute.

NOTE on why we do NOT use raw CLS->patch attention weights:
  DINOv2 was trained with iBOT-style masked image modelling, not with
  the DINOv1 self-distillation + centering/sharpening that made attention
  heads into clean segmenters. In practice raw attention on ViT-L/14 is
  very diffuse and often lights up large background regions; the final-
  feature cosine is the robust foreground cue.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def foreground_from_saliency(saliency: torch.Tensor,
                             out_size: int,
                             percentile: float = 35.0,
                             smooth_sigma: float = 2.0) -> torch.Tensor:
    """
    Build a soft foreground mask from the CLS↔patch cosine saliency.

    saliency   : (B, Hp, Wp) cosine similarity in roughly [-1, 1]
    out_size   : target (H, W) resolution for the mask (e.g. 448).
    percentile : pixels with saliency ABOVE this percentile are treated
                 as foreground; a soft sigmoid ramp is centred there.
    smooth_sigma : Gaussian sigma (in *output* pixels, not patch pixels)
                 used to avoid patch-grid blockiness.

    Returns (B, 1, out_size, out_size) soft mask in (0, 1), 1=foreground.
    """
    if saliency is None:
        return None
    B, Hp, Wp = saliency.shape
    a = saliency.float().unsqueeze(1)                         # (B,1,Hp,Wp)
    a = F.interpolate(a, size=(out_size, out_size),
                      mode="bilinear", align_corners=False)
    if smooth_sigma > 0:
        k = int(smooth_sigma * 4) | 1
        ax = torch.arange(k, device=a.device, dtype=a.dtype) - k // 2
        g = torch.exp(-0.5 * (ax / max(smooth_sigma, 1e-3)) ** 2)
        g = g / g.sum()
        a = F.conv2d(a, g.view(1, 1, -1, 1).expand(1, 1, -1, 1),
                     padding=(k // 2, 0), groups=1)
        a = F.conv2d(a, g.view(1, 1, 1, -1).expand(1, 1, 1, -1),
                     padding=(0, k // 2), groups=1)
    flat = a.reshape(B, -1)
    kth = max(1, min(flat.shape[-1],
                     int(flat.shape[-1] * percentile / 100.0)))
    t = torch.kthvalue(flat, kth, dim=-1).values.reshape(B, 1, 1, 1)
    std = flat.std(dim=-1).reshape(B, 1, 1, 1).clamp(min=1e-4)
    fg = torch.sigmoid((a - t) / (std * 0.5))
    return fg


def apply_foreground(anomaly_map: torch.Tensor,
                     fg_mask: torch.Tensor | None,
                     lam: float = 0.15) -> torch.Tensor:
    """
    Soft-suppress background pixels in the anomaly map only (NOT the
    image-level score -- that would break calibration).

    anomaly_map : (V, H, W) or (B, H, W) calibrated [0,1] anomaly map.
    fg_mask     : (V, 1, H, W) or broadcastable soft foreground in [0,1].
    lam         : floor multiplier for background pixels. 0.15 means a
                  pixel in the "most background" region retains 15% of
                  its anomaly score -- enough to preserve faint defects
                  on object edges while still knocking down jig/belt
                  false positives.

    Returns same-shape tensor:  map * (lam + (1-lam) * fg).
    """
    if fg_mask is None:
        return anomaly_map
    if fg_mask.dim() == 4 and fg_mask.shape[1] == 1 and anomaly_map.dim() == 3:
        m = fg_mask.squeeze(1)
    else:
        m = fg_mask
    return anomaly_map * (lam + (1.0 - lam) * m)


# ---------------------------------------------------------------------------
# Pure-PyTorch connected-component size suppression (no scipy dependency).
# Uses a simple iterative two-pass labelling. Good enough for 448x448 maps.
# ---------------------------------------------------------------------------
def _torch_connected_components(binary: torch.Tensor) -> torch.Tensor:
    H, W = binary.shape
    lab = torch.zeros(H, W, dtype=torch.int64, device=binary.device)
    if not bool(binary.any()):
        return lab
    parent: dict = {}
    b = binary.to(torch.bool)
    next_label = 1

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    for y in range(H):
        for x in range(W):
            if not b[y, x]:
                continue
            nbs = []
            if y > 0 and lab[y - 1, x] > 0:
                nbs.append(int(lab[y - 1, x].item()))
            if x > 0 and lab[y, x - 1] > 0:
                nbs.append(int(lab[y, x - 1].item()))
            if not nbs:
                lab[y, x] = next_label
                parent[next_label] = next_label
                next_label += 1
            else:
                root = min(nbs)
                lab[y, x] = root
                for u in nbs:
                    ru, rr = _find(u), _find(root)
                    if ru != rr:
                        parent[max(ru, rr)] = min(ru, rr)
    for y in range(H):
        for x in range(W):
            if lab[y, x] > 0:
                lab[y, x] = _find(int(lab[y, x].item()))
    return lab


def remove_small_components_torch(mask: torch.Tensor,
                                  threshold: float = 0.30,
                                  min_area: int = 25,
                                  min_area_kill: float = 0.1) -> torch.Tensor:
    """
    Soft-suppress tiny connected components in an anomaly map.

    mask : (H, W) float in [0,1].
    threshold : binarisation threshold for component labelling.
    min_area  : components smaller than this (in pixels @ map resolution)
                have their values multiplied by ``min_area_kill``.
    min_area_kill : multiplier for tiny components (0 = black them out,
                1 = no suppression; 0.1 is a gentle soft-kill).
    """
    if mask.numel() == 0 or min_area <= 1:
        return mask
    dev = mask.device
    m = mask.detach().float().cpu()
    binary = (m > threshold)
    lab = _torch_connected_components(binary)
    if lab.max() == 0:
        return mask
    sizes = torch.bincount(lab.reshape(-1),
                           minlength=int(lab.max().item()) + 1)
    mult = torch.ones_like(m)
    for li in range(1, sizes.shape[0]):
        if 0 < sizes[li] < min_area:
            mult[lab == li] = min_area_kill
    return (m * mult).to(dev).clamp_(0.0, 1.0)
