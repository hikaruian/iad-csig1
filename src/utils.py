"""
Misc utilities: calibration, mask I/O, CSV I/O, percentile normalisation.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Score calibration
# ---------------------------------------------------------------------------
class PerClassPercentileCalibrator:
    """
    Because anomaly scores from different classes live on different scales,
    we calibrate image-level scores to a roughly [0,1] range using statistics
    collected from the training set (normal-only). Specifically:

        calibrated = (s - p50_train) / (p95_train - p50_train + eps)
        then clamped to [0, 1] globally at submission time after mixing.
    """

    def __init__(self, eps: float = 1e-6):
        self.stats: Dict[str, Tuple[float, float]] = {}
        self.eps = eps

    @torch.no_grad()
    def update(self, cls: str, scores: torch.Tensor):
        s = scores.float().cpu().numpy().reshape(-1)
        p50 = float(np.percentile(s, 50))
        p95 = float(np.percentile(s, 95))
        if cls in self.stats:
            old_p50, old_p95 = self.stats[cls]
            # running average (ok for our small dataset)
            p50 = 0.5 * (old_p50 + p50)
            p95 = 0.5 * (old_p95 + p95)
        self.stats[cls] = (p50, p95)

    def apply(self, cls: str, scores: torch.Tensor) -> torch.Tensor:
        if cls not in self.stats:
            # unseen class: robust fallback
            return torch.sigmoid(5.0 * (scores - scores.median()))
        p50, p95 = self.stats[cls]
        z = (scores - p50) / (p95 - p50 + self.eps)
        return torch.clamp(z, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Mask I/O (submission format: 448x448 single-channel 8-bit PNG)
# ---------------------------------------------------------------------------
def save_mask_png(mask: np.ndarray | torch.Tensor, path: str | Path,
                  target_size: int = 448):
    """
    mask: (H, W) in arbitrary range. We normalise per-mask to [0,255] then
          write as 8-bit PNG.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.float().detach().cpu().numpy()
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    if mask.shape != (target_size, target_size):
        mask_pil = Image.fromarray(
            (mask - mask.min()) / (mask.max() - mask.min() + 1e-8) * 255
        ).convert("L")
        mask_pil = mask_pil.resize((target_size, target_size), Image.BILINEAR)
        mask = np.asarray(mask_pil, dtype=np.float32)
    # Per-mask min-max to [0, 255] so each mask uses the full dynamic range
    m_min, m_max = float(mask.min()), float(mask.max())
    if m_max - m_min < 1e-8:
        out = np.zeros_like(mask, dtype=np.uint8)
    else:
        out = ((mask - m_min) / (m_max - m_min) * 255.0).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="L").save(str(path))


def save_submission_csv(rows: Iterable[Tuple[str, float]], out_path: str | Path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_folder", "anomaly_score"])
        for gf, sc in rows:
            w.writerow([gf, f"{float(sc):.6f}"])


# ---------------------------------------------------------------------------
# TTA helpers
# ---------------------------------------------------------------------------
def tta_flips(x: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
    """Return a list of (name, tensor) for the 4 flip augmentations."""
    return [
        ("orig", x),
        ("h",    torch.flip(x, dims=[-1])),
        ("v",    torch.flip(x, dims=[-2])),
        ("hv",   torch.flip(x, dims=[-2, -1])),
    ]


def unflip_map(amap: torch.Tensor, aug_name: str) -> torch.Tensor:
    """Reverse the flip applied to an anomaly map (shape B,H,W or B,1,H,W)."""
    if aug_name == "orig":
        return amap
    dims = []
    if "h" in aug_name:
        dims.append(-1)
    if "v" in aug_name:
        dims.append(-2)
    return torch.flip(amap, dims=dims)


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.v = 0.0; self.n = 0
    def update(self, x, n: int = 1):
        self.v += float(x) * n; self.n += n
    @property
    def avg(self):
        return self.v / max(1, self.n)
