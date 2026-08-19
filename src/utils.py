"""
Misc utilities: calibration, mask I/O, CSV I/O, percentile normalisation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Score calibration
# ---------------------------------------------------------------------------
class PerClassPercentileCalibrator:
    """
    Per-class percentile calibration to a roughly [0, 1] range, using
    statistics collected from (normal) training samples:

        z = clamp((s - p50) / (p95 - p50 + eps), 0, 1)

    Usage:
        cal = PerClassPercentileCalibrator()
        for batch in train_loader:
            ...
            cal.update(cls, scores)         # accumulate raw scores
        cal.finalize()                      # compute percentiles once
        z = cal.apply(cls, test_scores)     # map to [0,1]

    We accumulate ALL scores (memory-light: 20 samples * 5 views = 100 floats
    per class) and compute percentiles at finalize() time rather than
    running-averaging them, because running averages of percentiles are
    mathematically ill-defined and produce biased estimates.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self._buf: Dict[str, List[float]] = {}
        self.stats: Dict[str, Tuple[float, float]] = {}

    def update(self, cls: str, scores):
        """Accumulate raw scores (scalar or tensor/array) for a class."""
        if isinstance(scores, torch.Tensor):
            s = scores.detach().float().cpu().numpy().reshape(-1)
        else:
            s = np.asarray(scores, dtype=np.float32).reshape(-1)
        self._buf.setdefault(cls, []).extend(s.tolist())

    def finalize(self):
        """Compute p50/p95 from accumulated scores. Called once after all
        training samples have been passed to update()."""
        for cls, vals in self._buf.items():
            s = np.asarray(vals, dtype=np.float64)
            p50 = float(np.percentile(s, 50))
            p95 = float(np.percentile(s, 95))
            # Guard against degenerate (all-identical) score distributions
            if p95 - p50 < self.eps:
                p95 = p50 + self.eps
            self.stats[cls] = (p50, p95)

    def apply(self, cls: str, scores: torch.Tensor) -> torch.Tensor:
        """Map raw scores to [0,1] using the stored p50/p95."""
        if not self.stats:
            # Auto-finalize if user forgot to call finalize()
            self.finalize()
        if cls not in self.stats:
            # Unseen class (zero-shot): robust fallback
            med = torch.quantile(scores.reshape(-1).float(), 0.5)
            return torch.sigmoid(5.0 * (scores - med)).clamp(0.0, 1.0)
        p50, p95 = self.stats[cls]
        z = (scores - p50) / (p95 - p50 + self.eps)
        return torch.clamp(z, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Mask I/O (submission format: 448x448 single-channel 8-bit PNG)
# ---------------------------------------------------------------------------
def save_mask_png(mask, path, target_size: int = 448,
                  global_lo: float = 0.0, global_hi: float = 1.0,
                  gamma: float = 1.0):
    """
    Save a single-channel anomaly mask as an 8-bit grayscale PNG at
    (target_size, target_size).

    IMPORTANT: we do NOT use per-mask min-max normalisation -- that destroys
    cross-sample score comparability and kills the P-AUPR / PRO metrics.
    Instead we clip to [global_lo, global_hi] (defaults [0,1] since the
    calibrator already maps scores to [0,1]) and linearly scale to [0,255].
    Gamma correction (default 2.0) suppresses weak mid-range false positives
    (background noise near ~0.3) while preserving strong anomaly responses
    near ~1.0 -- this gives a significant pixel-AP boost on benchmarks where
    most pixels are normal. Set gamma=1.0 to disable.

    mask       : numpy array or torch tensor (H, W), values assumed in [0,1]
                 after calibration; values outside are clipped.
    path       : destination path
    target_size: output PNG size
    global_lo/global_hi: mapping window; values map lo->0, hi->255.
    gamma      : optional gamma correction applied in [0,1] space.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().float().cpu().numpy()
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    mask = np.clip(mask, 0.0, 1.0)

    # Resize in FLOAT32 (bilinear) to avoid quantisation artefacts that
    # arise when you quantise to uint8 *before* interpolating – the latter
    # creates visible stairstepping at anomaly boundaries and kills PRO.
    if mask.shape != (target_size, target_size):
        mask_f = Image.fromarray(mask.astype(np.float32), mode="F")
        mask_f = mask_f.resize((target_size, target_size), Image.BILINEAR)
        mask = np.asarray(mask_f, dtype=np.float32)

    # Global window linear map, no per-mask renormalisation.
    z = (mask - global_lo) / (global_hi - global_lo + 1e-8)
    z = np.clip(z, 0.0, 1.0)
    if gamma != 1.0:
        z = np.power(z, gamma)
    out = np.clip(z * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out, mode="L").save(str(path))


def save_submission_csv(rows: Iterable[Tuple[str, float]], out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_folder", "anomaly_score"])
        for gf, sc in rows:
            w.writerow([gf, f"{float(sc):.6f}"])


# ---------------------------------------------------------------------------
# TTA helpers
# ---------------------------------------------------------------------------
def tta_flips(x: torch.Tensor):
    """Yield (name, flipped_tensor) for the 4 flip augmentations.
    Back to full 4-way TTA (orig/h/v/hv) -- v-flip does not hurt on
    Real-IAD Variety because the parts are roughly symmetric top/bottom
    in many classes and v-flip acts as a useful invariance prior."""
    yield ("orig", x)
    yield ("h",    torch.flip(x, dims=[-1]))
    yield ("v",    torch.flip(x, dims=[-2]))
    yield ("hv",   torch.flip(x, dims=[-2, -1]))


def unflip_map(amap: torch.Tensor, aug_name: str) -> torch.Tensor:
    """Reverse the flip applied to an anomaly map (shape ...,H,W)."""
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
    def __init__(self):
        self.reset()

    def reset(self):
        self.v = 0.0
        self.n = 0

    def update(self, x, n: int = 1):
        self.v += float(x) * n
        self.n += n

    @property
    def avg(self):
        return self.v / max(1, self.n)
