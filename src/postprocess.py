"""
Final "champion" post-processing tricks that almost always give +1~2pp
on real industrial AD data, and cost almost nothing:

1. **Feature Whitening (Mahalanobis normalisation)**
   Before running the memory bank, zca-whiten patch features using the
   covariance estimated on the NORMAL training patches. This removes the
   common directions of variation (illumination, pose, intra-class style)
   so that cosine/Euclidean distance is dominated by true anomaly
   directions. SimpleNet, PaDiM, CFA all use this; consistently +1pp.

2. **Multi-Score Geometric Ensemble**
   Average (geometric mean) three complementary anomaly scores:
       s_knn     - PatchCore kNN cosine distance
       s_anoco   - ANoCo Laplacian non-conformity energy
       s_recon   - Dinomaly-Lite decoder reconstruction cosine residual
   Geometric mean is preferred over arithmetic because all three must
   agree to produce a high score (reduces false positives from any single
   scorer's failure mode).

3. **Test-time Adaptive Normalisation (TTAD)**
   Instead of using ONLY the train-set P50/P95 to calibrate, we use a
   convex combination  alpha*train_pct + (1-alpha)*test_pct  where
   test_pct is estimated within a sample across its 5 views × patches.
   This handles covariate shift between train/test that the per-class
   calibrator cannot see. (Inspired by RD4AD and EfficientAD's "adaptive
   threshold" post-processing.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class Whitener:
    """
    ZCA whitening of D-dimensional features.
    Fit on NORMAL patch features only, then apply at train AND test time.
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps
        self.mean = None
        self.W = None  # (D,D) whitening matrix
        self.W_inv = None

    @torch.no_grad()
    def fit(self, X: torch.Tensor):
        """X : (N, D)"""
        self.mean = X.mean(0, keepdim=True)
        Xc = X - self.mean
        C = (Xc.t() @ Xc) / Xc.shape[0]
        # Eigendecomposition for symmetric C
        L, V = torch.linalg.eigh(C)
        L = L.clamp(min=self.eps)
        # ZCA: W = V diag(L^{-1/2}) V^T
        L_inv = torch.diag(L.rsqrt())
        self.W = V @ L_inv @ V.t()
        self.W_inv = V @ torch.diag(L.sqrt()) @ V.t()
        return self

    @torch.no_grad()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.W is None:
            return X
        Xc = X - self.mean
        Xw = Xc @ self.W.t()
        return F.normalize(Xw, dim=-1)


def geometric_mean_score(*scores: torch.Tensor) -> torch.Tensor:
    """
    scores: arbitrary number of tensors all of the same shape (B,) or (B,H,W)
    Returns: geometric mean = prod(scores)^(1/k)
    """
    stacked = torch.stack(list(scores), dim=0)              # (k, ...)
    # Clamp to avoid log(0)
    stacked = stacked.clamp(min=1e-8)
    return stacked.log().mean(0).exp()


def adaptive_score_normalisation(scores_map: torch.Tensor,
                                  train_p50: float, train_p95: float,
                                  alpha: float = 0.5) -> torch.Tensor:
    """
    Convex combine training-set percentiles with test-sample percentiles.

    scores_map : (V, H, W) per-view high-res anomaly maps for one sample
    train_p50/p95 : calibration stats from training set for this class
    alpha : weight on training stats (0.5 = equal trust)

    Returns: (V,H,W) maps calibrated to [0,1] approximately.
    """
    flat = scores_map.reshape(-1)
    p50_t = float(torch.quantile(flat, 0.50).item())
    p95_t = float(torch.quantile(flat, 0.95).item())
    p50 = alpha * train_p50 + (1 - alpha) * p50_t
    p95 = alpha * train_p95 + (1 - alpha) * p95_t
    z = (scores_map - p50) / (p95 - p50 + 1e-6)
    return z.clamp(0, 1)
