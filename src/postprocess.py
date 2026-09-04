"""Anomaly-map post-processing and mask serialization."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover
    gaussian_filter = None


def smooth_map(amap: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    if sigma <= 0:
        return amap
    if gaussian_filter is None:
        return amap
    if amap.ndim == 2:
        return gaussian_filter(amap, sigma=sigma)
    out = np.empty_like(amap)
    for i in range(amap.shape[0]):
        out[i] = gaussian_filter(amap[i], sigma=sigma)
    return out


def squash_score(score: float) -> float:
    """Map a non-negative raw score to [0, 1] with a strictly increasing transform.

    Raw scores are top-1% means of reconstruction / z-score maps, not probabilities.
    After z-score + gamma they routinely exceed 1. I-AUROC / I-AP only care about
    ranking, so this does not change those metrics.
    """
    s = float(score)
    if not np.isfinite(s):
        return 0.0
    s = max(0.0, s)
    return float(min(1.0, s / (1.0 + s)))


def sample_score_from_views(view_maps: np.ndarray, max_ratio: float = 0.01, reduce: str = "max") -> float:
    """view_maps: (5, H, W). Official recipe: mean of top-1% pixels per view."""
    scores = []
    for m in view_maps:
        flat = m.reshape(-1)
        k = max(1, int(flat.size * max_ratio))
        scores.append(float(np.partition(flat, -k)[-k:].mean()))
    if reduce == "max":
        return float(np.max(scores))
    if reduce == "mean":
        return float(np.mean(scores))
    if reduce == "lse":
        arr = np.asarray(scores, dtype=np.float64)
        return float(np.log(np.mean(np.exp(arr - arr.max()))) + arr.max())
    raise ValueError(reduce)


def maps_to_uint8(maps: np.ndarray, scale: float) -> np.ndarray:
    """Global linear scale (NOT per-image min-max) so pixel ranking is preserved."""
    arr = np.clip(np.asarray(maps, dtype=np.float32) * float(scale), 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def calibrate_scale(maps: Iterable[np.ndarray], target_q: float = 0.995, target_value: float = 0.55) -> float:
    """Pick a single global scale so train-set high-quantile maps sit mid-range.

    Anomalous pixels (larger than the train quantile) still have headroom up to 1.0.
    """
    buf = []
    for m in maps:
        flat = np.asarray(m, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            continue
        # subsample for speed
        if flat.size > 200_000:
            rng = np.random.default_rng(0)
            flat = rng.choice(flat, size=200_000, replace=False)
        buf.append(flat)
    if not buf:
        return 2.0
    all_pix = np.concatenate(buf, axis=0)
    q = float(np.quantile(all_pix, target_q))
    if q <= 1e-8:
        return 2.0
    return float(target_value / q)


def resize_to_448(arr: np.ndarray) -> np.ndarray:
    if arr.shape[-2:] == (448, 448):
        return arr
    from PIL import Image

    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.float32), mode="F")
        img = img.resize((448, 448), resample=Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)
    out = []
    for m in arr:
        img = Image.fromarray(m.astype(np.float32), mode="F")
        img = img.resize((448, 448), resample=Image.BILINEAR)
        out.append(np.asarray(img, dtype=np.float32))
    return np.stack(out, axis=0)

