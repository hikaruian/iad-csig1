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
        self.patch_stride = patch_output_stride  # ViT patch size
        self.map_size = map_size  # anomaly-map spatial size per window

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
        ps = self.patch_stride
        ws = self.ws
        win_p = ws // ps  # patches per window, e.g. 448/14 = 32

        # Gaussian blending kernel in patch-grid coordinates.
        gw = _gaussian_window(win_p, self.sigma / ps, device=device, dtype=dtype)
        gw = gw.reshape(1, 1, win_p, win_p)

        # Pad image if smaller than window, and round padded size up to a
        # multiple of ps so patch-grid indexing stays aligned.
        pad_h = max(ws - H, 0)
        pad_w = max(ws - W, 0)
        pad_h += (- (H + pad_h)) % ps
        pad_w += (- (W + pad_w)) % ps
        if pad_h > 0 or pad_w > 0:
            image_padded = F.pad(image_hr, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            image_padded = image_hr
        _, H_pad, W_pad = image_padded.shape

        H_p_pad = H_pad // ps
        W_p_pad = W_pad // ps
        score_acc = torch.zeros(1, 1, H_p_pad, W_p_pad, device=device, dtype=dtype)
        count_acc = torch.zeros(1, 1, H_p_pad, W_p_pad, device=device, dtype=dtype)

        # Window top-left positions (pixel).
        ys = list(range(0, max(1, H_pad - ws + 1), self.stride))
        xs = list(range(0, max(1, W_pad - ws + 1), self.stride))
        if ys[-1] != H_pad - ws:
            ys.append(H_pad - ws)
        if xs[-1] != W_pad - ws:
            xs.append(W_pad - ws)

        for y in ys:
            for x in xs:
                patch = image_padded[:, y:y+ws, x:x+ws].unsqueeze(0)  # (1,3,ws,ws)
                out = backbone(patch)
                res = bank.predict(cls_name, out["patch"], out["cls"],
                                   return_map=True)
                amap = res["anomaly_map"]  # (1, ws, ws)
                # Downscale each window's anomaly map to patch-grid.
                amap_p = F.interpolate(amap.unsqueeze(1), size=(win_p, win_p),
                                       mode="bilinear", align_corners=False)
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

                yp = y // ps
                xp = x // ps
                # Defensive: clip splat to accumulator bounds.
                y_end = min(yp + win_p, H_p_pad)
                x_end = min(xp + win_p, W_p_pad)
                dy = y_end - yp
                dx = x_end - xp
                if dy > 0 and dx > 0:
                    score_acc[:, :, yp:y_end, xp:x_end] += amap_p[:, :, :dy, :dx] * gw[:, :, :dy, :dx]
                    count_acc[:, :, yp:y_end, xp:x_end] += gw[:, :, :dy, :dx]

        count_acc = count_acc.clamp_min(1e-6)
        score_map_p = score_acc / count_acc
        # Upsample from patch grid to full padded resolution, then crop.
        score_map = F.interpolate(score_map_p, size=(H_pad, W_pad),
                                  mode="bilinear", align_corners=False)
        score_map = score_map[:, :, :H, :W]
        return score_map[0, 0]
