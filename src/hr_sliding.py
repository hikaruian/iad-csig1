"""
High-resolution sliding-window inference.

The CSIG / Real-IAD Variety images are up to 4400x4400. Directly resizing
to 448x448 destroys tiny defects (0.002mm level). The standard fix in all
top-performing entries (AnomalyDINO, EfficientAD, VisionAD, Dinomaly2) is
to run the feature extractor on overlapping WINDOWS and merge the resulting
patch-level scores with a Gaussian weight, while still running at ~448px
input for the backbone. This recovers the small-defect P-F1max / P-AUPR
that is lost under aggressive down-scaling.

Usage:
    from src.hr_sliding import SlidingWindowInferer
    sw = SlidingWindowInferer(window_size=448, stride=224)
    anom_map = sw.infer(image_hr, backbone, patchcore, cls_name)
    # image_hr: (3, H, W) tensor, original resolution (already normalised)
    # anom_map: (H, W) anomaly map at the same resolution
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


def _gaussian_window(size: int, sigma: float, device=None, dtype=torch.float32):
    """2D Gaussian kernel used for soft blending of sliding windows."""
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g2 = g1.unsqueeze(1) * g1.unsqueeze(0)
    return g2 / g2.sum()


class SlidingWindowInferer:
    def __init__(self, window_size: int = 448, stride: int = 224,
                 sigma: Optional[float] = None, patch_output_stride: int = 14,
                 map_size: int = 448):
        self.ws = window_size
        self.stride = stride
        self.sigma = sigma if sigma is not None else window_size / 4.0
        self.patch_stride = patch_output_stride  # ViT patch size -> 1 map pixel per patch
        self.map_size = map_size  # size of anomaly map produced per window

    @torch.no_grad()
    def infer(self, image_hr: torch.Tensor, backbone, bank, cls_name: str,
              clip_model=None, clip_weight: float = 0.0,
              clip_ref=None) -> torch.Tensor:
        """
        image_hr: (3, H, W) tensor, already normalised to backbone stats,
                  on the same device as the backbone.
        Returns: anomaly map (H, W) at the original HR resolution.
        """
        C, H, W = image_hr.shape
        device = image_hr.device
        dtype = image_hr.dtype

        # Accumulators in patch-grid coordinate space (lower resolution
        # than HR but higher than a single 448 window): we first build a
        # dense score map at stride = patch_stride relative to the original,
        # then bilinearly upsample to (H, W).
        H_p = H // self.patch_stride
        W_p = W // self.patch_stride
        score_acc = torch.zeros(1, 1, H_p, W_p, device=device, dtype=dtype)
        count_acc = torch.zeros(1, 1, H_p, W_p, device=device, dtype=dtype)

        # Gaussian weight per window (in patch-grid coordinates)
        win_p = self.ws // self.patch_stride  # number of patches per window
        gw = _gaussian_window(win_p, self.sigma / self.patch_stride,
                              device=device, dtype=dtype)
        gw = gw.reshape(1, 1, win_p, win_p)

        # Top-left corner positions (pixel level) in the original image
        ys = list(range(0, max(1, H - self.ws + 1), self.stride))
        xs = list(range(0, max(1, W - self.ws + 1), self.stride))
        # Always include a final window flush with the bottom/right edge
        if ys[-1] != H - self.ws: ys.append(H - self.ws)
        if xs[-1] != W - self.ws: xs.append(W - self.ws)

        for y in ys:
            for x in xs:
                patch = image_hr[:, y:y+self.ws, x:x+self.ws].unsqueeze(0)  # (1,3,ws,ws)
                # DINOv2 branch
                out = backbone(patch)
                res = bank.predict(cls_name, out["patch"], out["cls"],
                                   return_map=True)
                amap = res["anomaly_map"]  # (1, map_size, map_size)
                # Convert amap back to patch-grid size: map_size is 448,
                # each patch covers 14 pixels -> 32 patches
                amap_p = F.interpolate(amap.unsqueeze(1), size=(win_p, win_p),
                                       mode="bilinear", align_corners=False)
                # Clip weight
                if clip_model is not None and clip_weight > 0 and clip_ref is not None:
                    c_in = F.interpolate(patch, size=(336, 336),
                                         mode="bilinear", align_corners=False)
                    co = clip_model.encode_image(c_in)
                    from .zeroshot import winclip_score
                    wc = winclip_score(clip_model, co["patch"], co["cls"],
                                       cls_name, reference_patches=clip_ref,
                                       alpha=0.5)
                    cm = F.interpolate(wc["anomaly_map"].unsqueeze(1),
                                       size=(win_p, win_p),
                                       mode="bilinear", align_corners=False)
                    amap_p = (1 - clip_weight) * amap_p + clip_weight * cm

                # Splat into accumulators with Gaussian weights
                yp = y // self.patch_stride
                xp = x // self.patch_stride
                score_acc[:, :, yp:yp+win_p, xp:xp+win_p] += amap_p * gw
                count_acc[:, :, yp:yp+win_p, xp:xp+win_p] += gw

        count_acc = count_acc.clamp_min(1e-6)
        score_map_p = score_acc / count_acc  # (1,1,H_p,W_p)
        score_map = F.interpolate(score_map_p, size=(H, W),
                                  mode="bilinear", align_corners=False)
        return score_map[0, 0]
