"""Pixel-map refinement aimed at P-AP / P-F1max.

Macro P-AP / P-F1max are computed *inside each category*. They collapse when
high-scoring pixels are texture, object contour or canvas border rather than
defects. This module:

  1. z-scores each (class, view) map against TRAIN-only normal statistics
     so chronically hard textures are no longer "anomalous"
  2. drops negative z (more normal than the class mean)
  3. applies a mild power (gamma) to stretch true peaks
  4. fades the outer border (ViT interpolation ringing)
  5. optionally gates by a cheap foreground mask from RGB

Do NOT min-max per image — that destroys the ranking these metrics need.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import binary_dilation, gaussian_filter
except ImportError:  # pragma: no cover
    binary_dilation = None
    gaussian_filter = None

from .postprocess import smooth_map

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
STAT_HW = 64


def _resize(arr: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    if arr.shape[-2:] == hw:
        return arr.astype(np.float32, copy=False)
    from PIL import Image

    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((hw[1], hw[0]), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def denorm_rgb(chw) -> np.ndarray:
    """Normalized CHW tensor/ndarray -> HWC RGB in [0, 1]."""
    x = chw.detach().float().cpu().numpy() if hasattr(chw, "detach") else np.asarray(chw, dtype=np.float32)
    x = x.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x, 0.0, 1.0)


def foreground_mask(rgb01: np.ndarray, dilate: int = 10) -> np.ndarray:
    """1 on object, 0 on canvas. Falls back to all-ones if the heuristic is unsure."""
    gray = rgb01.mean(axis=-1)
    h, w = gray.shape
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    mask = np.abs(gray - bg) > max(0.07, 0.18 * float(gray.std() + 1e-6))
    frac = float(mask.mean())
    if frac < 0.04 or frac > 0.97:
        return np.ones((h, w), dtype=np.float32)
    if binary_dilation is not None:
        mask = binary_dilation(mask, iterations=max(1, dilate))
    elif gaussian_filter is not None:
        mask = gaussian_filter(mask.astype(np.float32), sigma=3.0) > 0.25
    return mask.astype(np.float32)


def border_weight(h: int, w: int, margin: int = 16) -> np.ndarray:
    if margin <= 0:
        return np.ones((h, w), dtype=np.float32)
    margin = min(margin, h // 4, w // 4)
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(margin, dtype=np.float32) / margin)
    wy[:margin] = ramp
    wy[-margin:] = ramp[::-1]
    wx[:margin] = ramp
    wx[-margin:] = ramp[::-1]
    return np.outer(wy, wx)


def refine_map(
    amap: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    rgb_chw=None,
    gamma: float = 1.4,
    border: int = 16,
    use_fg: bool = True,
    relu: bool = True,
) -> np.ndarray:
    """amap: (H, W) raw reconstruction error. Returns non-negative refined map."""
    m = np.asarray(amap, dtype=np.float32)
    h, w = m.shape
    if mean is not None and std is not None:
        mu = _resize(np.asarray(mean, dtype=np.float32), (h, w))
        sd = _resize(np.asarray(std, dtype=np.float32), (h, w))
        sd = np.maximum(sd, 1e-4)
        m = (m - mu) / sd
    if relu:
        m = np.maximum(m, 0.0)
    if gamma is not None and abs(gamma - 1.0) > 1e-6:
        # power on a floor-clipped map; no per-image max
        m = np.power(m, float(gamma), dtype=np.float32)
    m = m * border_weight(h, w, margin=border)
    if use_fg and rgb_chw is not None:
        fg = foreground_mask(denorm_rgb(rgb_chw))
        if fg.shape != m.shape:
            fg = _resize(fg, (h, w))
        m = m * (0.12 + 0.88 * fg)
    return m


class NormalStats:
    """Per-(category, view) spatial mean/std of *smoothed* train maps, plus a class scale."""

    def __init__(self, hw: int = STAT_HW):
        self.hw = int(hw)
        self.sum: Dict[str, np.ndarray] = {}
        self.sumsq: Dict[str, np.ndarray] = {}
        self.count: Dict[str, int] = {}
        self.class_peak: Dict[str, list] = {}
        self.view_scores: Dict[str, list] = {}

    @staticmethod
    def key(category: str, view: int) -> str:
        return f"{category}#{int(view)}"

    def update(self, category: str, view: int, amap: np.ndarray, view_score: Optional[float] = None):
        small = _resize(np.asarray(amap, dtype=np.float32), (self.hw, self.hw))
        k = self.key(category, view)
        if k not in self.sum:
            self.sum[k] = np.zeros((self.hw, self.hw), dtype=np.float64)
            self.sumsq[k] = np.zeros((self.hw, self.hw), dtype=np.float64)
            self.count[k] = 0
        self.sum[k] += small
        self.sumsq[k] += small.astype(np.float64) ** 2
        self.count[k] += 1
        self.class_peak.setdefault(category, []).append(float(np.quantile(small, 0.995)))
        if view_score is None:
            flat = small.reshape(-1)
            kk = max(1, int(flat.size * 0.01))
            view_score = float(np.partition(flat, -kk)[-kk:].mean())
        self.view_scores.setdefault(k, []).append(float(view_score))

    def mean_std(self, category: str, view: int):
        k = self.key(category, view)
        n = self.count.get(k, 0)
        if n < 2:
            return None, None
        mean = (self.sum[k] / n).astype(np.float32)
        var = self.sumsq[k] / n - mean.astype(np.float64) ** 2
        std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
        return mean, std

    def scale_for(self, category: str, target: float = 0.55) -> float:
        peaks = self.class_peak.get(category)
        if not peaks:
            return 2.0
        q = float(np.median(peaks))
        if q <= 1e-8:
            return 2.0
        return float(target / q)

    def score_mu_std(self, category: str, view: int):
        xs = self.view_scores.get(self.key(category, view)) or []
        if len(xs) < 2:
            return None, None
        arr = np.asarray(xs, dtype=np.float64)
        return float(arr.mean()), float(max(arr.std(), 1e-6))

    def view_gate(
        self,
        category: str,
        view: int,
        score: float,
        k: float = 1.25,
        temp: float = 0.35,
        floor: float = 0.0,
        hard: bool = True,
        mode: str = "max",
        margin: float = 1.12,
    ) -> float:
        xs = self.view_scores.get(self.key(category, view)) or []
        if len(xs) < 1:
            return 1.0
        sc = float(score)
        if mode == "max":
            keep = sc > float(np.max(xs)) * float(margin)
        else:
            mu, sd = self.score_mu_std(category, view)
            if mu is None:
                return 1.0
            z = (sc - mu) / sd
            if hard:
                keep = z > float(k)
            else:
                x = (z - float(k)) / max(float(temp), 1e-3)
                x = float(np.clip(x, -20.0, 20.0))
                return float(max(floor, 1.0 / (1.0 + np.exp(-x))))
        if keep:
            return 1.0
        return 0.0 if hard else float(floor)

    def save(self, path: str):
        keys = sorted(self.sum.keys())
        payload = {
            "hw": np.int32(self.hw),
            "keys": np.array(keys),
            "counts": np.array([self.count[k] for k in keys], dtype=np.int32),
        }
        for k in keys:
            payload[f"sum/{k}"] = self.sum[k]
            payload[f"sumsq/{k}"] = self.sumsq[k]
        cats = sorted(self.class_peak.keys())
        payload["peak_cats"] = np.array(cats)
        for c in cats:
            payload[f"peak/{c}"] = np.asarray(self.class_peak[c], dtype=np.float32)
        for k, xs in self.view_scores.items():
            payload[f"sc/{k}"] = np.asarray(xs, dtype=np.float32)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "NormalStats":
        z = np.load(path, allow_pickle=True)
        obj = cls(hw=int(z["hw"]) if "hw" in z.files else STAT_HW)
        keys = [str(k) for k in z["keys"]] if "keys" in z.files else []
        counts = z["counts"] if "counts" in z.files else np.zeros(len(keys), dtype=np.int32)
        for k, n in zip(keys, counts):
            if f"sum/{k}" in z.files:
                obj.sum[k] = z[f"sum/{k}"]
                obj.sumsq[k] = z[f"sumsq/{k}"]
                obj.count[k] = int(n)
        if "peak_cats" in z.files:
            for c in z["peak_cats"]:
                c = str(c)
                if f"peak/{c}" in z.files:
                    obj.class_peak[c] = list(np.asarray(z[f"peak/{c}"]).reshape(-1))
        for name in z.files:
            if name.startswith("sc/"):
                obj.view_scores[name[3:]] = [float(x) for x in np.asarray(z[name]).reshape(-1)]
        return obj


def apply_view_refine(
    maps: np.ndarray,
    images_chw,
    category: str,
    stats: Optional[NormalStats],
    sigma: float = 2.5,
    gamma: float = 1.4,
    border: int = 16,
    use_fg: bool = True,
) -> np.ndarray:
    """maps: (5,H,W). images_chw: (5,3,H,W) tensor or ndarray."""
    maps = smooth_map(np.asarray(maps, dtype=np.float32), sigma=sigma)
    out = np.empty_like(maps)
    for v in range(maps.shape[0]):
        mean = std = None
        if stats is not None:
            mean, std = stats.mean_std(category, v)
        rgb = images_chw[v] if images_chw is not None else None
        out[v] = refine_map(
            maps[v],
            mean=mean,
            std=std,
            rgb_chw=rgb,
            gamma=gamma,
            border=border,
            use_fg=use_fg,
        )
    return out


def apply_view_gate(
    maps: np.ndarray,
    category: str,
    stats: Optional[NormalStats],
    max_ratio: float = 0.01,
    k: float = 1.25,
    temp: float = 0.35,
    floor: float = 0.0,
    scores: Optional[list] = None,
    hard: bool = True,
    mode: str = "max",
    margin: float = 1.12,
) -> tuple:
    maps = np.asarray(maps, dtype=np.float32)
    gates = []
    used_scores = []
    out = maps.copy()
    for v in range(maps.shape[0]):
        if scores is not None:
            sc = float(scores[v])
        else:
            flat = maps[v].reshape(-1)
            kk = max(1, int(flat.size * max_ratio))
            sc = float(np.partition(flat, -kk)[-kk:].mean())
        used_scores.append(sc)
        g = 1.0 if stats is None else stats.view_gate(
            category, v, sc, k=k, temp=temp, floor=floor, hard=hard, mode=mode, margin=margin
        )
        gates.append(g)
        out[v] = maps[v] * g
    return out, gates, used_scores

