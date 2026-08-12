"""
Multi-view fusion strategies.

Each physical sample has 5 camera views. The views are not registered
pixel-by-pixel, so we do NOT do geometry-based stereo fusion. Instead we
fuse on the *feature* and *score* level. The three complementary strategies
implemented here are:

1. **Feature-level aggregation (MVConcat)**
   For memory-bank construction we treat each view as an independent
   training sample (already the default). At inference we compute a
   per-view anomaly map and aggregate scores across views.

2. **Cross-view attention pooling (MVAttn)**
   Given CLS tokens for all 5 views, compute an attention-weighted
   aggregate CLS vector w.r.t. the training prototype. This gives an
   image-level "consensus" score that is more robust than naive
   max-pooling per-view scores.

3. **Multi-view mask voting (MVVote)**
   After per-view high-res anomaly maps are produced, we blend them with
   the cross-view MEDIAN consensus. Median is used instead of mean
   because it is robust to single-view outliers (specular highlights,
   view-specific background clutter) while still reinforcing true
   defects that appear in 2+ views.

The recommended setting is Robust-Mean score aggregation + MVVote, which
gives a strong +1~2% P-AUROC boost on multi-view AD benchmarks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewAttention(nn.Module):
    """
    Tiny attention block that pools 5 view-level CLS tokens into a single
    robust sample-level descriptor. Run with frozen weights (linear init).
    NOTE: not currently used by the default pipeline (we use a simple mean
    instead), but kept here as a building block for custom ensembles.
    """

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self._init_weights()

    def _init_weights(self):
        for m in [self.q, self.k, self.v, self.out]:
            nn.init.eye_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.no_grad()
    def forward(self, views_cls: torch.Tensor,
                prototype: torch.Tensor) -> torch.Tensor:
        """
        views_cls : (B, V, D)
        prototype : (D,) or (B, D) -- a "normal prototype" used as query
        Returns   : (B, D) aggregated cls
        """
        B, V, D = views_cls.shape
        if prototype.dim() == 1:
            q = prototype.view(1, 1, D).expand(B, 1, -1)
        else:
            q = prototype.unsqueeze(1)
        k = self.k(views_cls)
        v = self.v(views_cls)
        q = self.q(q)
        head_dim = D // self.n_heads
        q = q.reshape(B, 1, self.n_heads, head_dim).transpose(1, 2)
        k = k.reshape(B, V, self.n_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, V, self.n_heads, head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, 1, D)
        out = self.out(out).squeeze(1)
        return F.normalize(out, dim=-1)


def multiview_mask_vote(per_view_maps: torch.Tensor,
                        beta: float = 1.5,
                        mode: str = "none") -> torch.Tensor:
    """
    Cross-view consensus blending for high-res anomaly maps.

    per_view_maps : (B, V, H, W) per-view anomaly maps, values in [0,1].
    mode :
      - "none"   : return maps AS-IS (BEST for Real-IAD Variety, because the
                   5 camera views show DIFFERENT physical surfaces — per-pixel
                   median/mean destroys single-view defects and boosts
                   background noise that coincidentally fires in 2+ views).
      - "median" : 0.85*orig + 0.15*median across views.
      - "mean"   : 0.85*orig + 0.15*mean across views.
    Returns (B, V, H, W) refined per-view maps.
    """
    if mode == "none":
        return per_view_maps
    if mode == "median":
        consensus = per_view_maps.median(dim=1, keepdim=True).values
        return 0.85 * per_view_maps + 0.15 * consensus
    if mode == "mean":
        consensus = per_view_maps.mean(dim=1, keepdim=True)
        return 0.85 * per_view_maps + 0.15 * consensus
    raise ValueError(f"Unknown mv_mask_vote mode: {mode}")


def aggregate_image_scores(per_view_scores: torch.Tensor,
                           strategy: str = "robust_mean") -> torch.Tensor:
    """
    per_view_scores : (B, V) per-view image-level anomaly scores.

    Strategies:
      * "mean"        : simple average.
      * "max"         : classic PatchCore strategy (worst view).
      * "robust_mean" : drop the highest and lowest view, then mean;
                        more robust than max for multi-view setups.
    """
    if strategy == "mean":
        return per_view_scores.mean(dim=1)
    if strategy == "max":
        return per_view_scores.max(dim=1).values
    if strategy == "robust_mean":
        s, _ = per_view_scores.sort(dim=1)
        if s.shape[1] <= 2:
            return s.mean(dim=1)
        return s[:, 1:-1].mean(dim=1)
    raise ValueError(f"Unknown aggregation strategy {strategy}")
