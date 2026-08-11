"""
PatchCore++ memory bank for multi-class industrial AD.

Standard PatchCore (Roth et al., 2022) stores dense patch features from
*all* training samples in a single memory bank, and scores test patches by
their nearest-neighbour distance to the bank. We extend it with:

1. **Per-class memory banks** – mandatory on 50-class Real-IAD to avoid
   cross-class feature collision. Defects that look "normal" for class A
   should not be explained by class B's bank.
2. **Multi-layer DINOv2 features** – we fuse the last 4 transformer blocks,
   matching INP-Former / Dinomaly.
3. **Greedy coreset subsampling** – deterministic (FPS-style) reduction so
   inference fits on a single GPU.
4. **Neighbourhood aggregation (bipartite neighbour smoothing)** – averages
   distances over a 3x3 patch window before upsampling, which dramatically
   improves P-AUROC/P-AUPR for tiny defects (cf. PaDiM, PatchCore paper).
5. **High-resolution re-upsampling** – we produce masks at 448x448 using
   Gaussian upsampling (sigma=~4.0) following Dinomaly/UnityAD.
6. **Optional reweighting** – combines CLS-level similarity score with the
   patch-level distance for a more robust image-level anomaly score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.random_projection import SparseRandomProjection
from torch import Tensor


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def gaussian_smooth2d(x: Tensor, kernel_size: int = 7, sigma: float = 4.0) -> Tensor:
    """Apply separable 2D Gaussian smoothing (used for mask refinement).

    Uses ``replicate`` (edge-value) padding so that scores near the image
    border are not artificially pulled toward zero (zero-padding would cause
    dark borders and suppress anomalies that happen to lie near an edge).
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = _fspecial_gauss_1d(kernel_size, sigma).to(x.device, x.dtype)
    pad = kernel_size // 2
    B, C, H, W = x.shape
    # Manually replicate-pad so the conv sees edge values instead of zeros
    x = F.pad(x, [pad, pad, pad, pad], mode="replicate")
    # Vertical pass (kernel over H, kW=1), then horizontal pass (kernel over W)
    x = F.conv2d(x, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1),
                 padding=(0, 0), groups=C)
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1),
                 padding=(0, 0), groups=C)
    return x


def bilinear_upsample(x: Tensor, size: int) -> Tensor:
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def reshape_patch_features(patch_feat: Tensor) -> Tuple[Tensor, int, int]:
    """
    patch_feat: (B, C, Hp, Wp) -> (B*Hp*Wp, C), Hp, Wp
    """
    B, C, Hp, Wp = patch_feat.shape
    feats = patch_feat.permute(0, 2, 3, 1).reshape(-1, C)
    return feats, Hp, Wp


# ---------------------------------------------------------------------------
# Greedy coreset (farthest-point sampling, deterministic)
# ---------------------------------------------------------------------------
def greedy_coreset(features: Tensor, n_select: int,
                   random_proj_dim: Optional[int] = None,
                   seed: int = 0,
                   device: str | torch.device | None = None) -> Tensor:
    """Farthest-point-sampling coreset (deterministic).

    features: (N, D) on CPU or CUDA
    Returns:  (n_select, D) on the SAME device as ``features``.

    Uses vectorised distance updates (no per-point Python loops) and runs
    on CUDA when the input is already on CUDA; for 100k points × 1024 dims
    this finishes in seconds rather than hours.
    """
    N = features.shape[0]
    n_select = min(n_select, N)
    if n_select == N:
        return features

    # Choose compute device -- prefer CUDA if available
    if device is None:
        device = features.device
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda")
    work_dev = torch.device(device)

    # Sparse random projection for speed (PatchCore default). We fit the
    # RP on CPU (sklearn) then move the projected points to work_dev.
    if random_proj_dim is not None and random_proj_dim < features.shape[1]:
        rp = SparseRandomProjection(n_components=random_proj_dim, random_state=seed)
        proj_np = rp.fit_transform(features.float().cpu().numpy()).astype(np.float32)
        proj = torch.from_numpy(proj_np).to(work_dev)
    else:
        proj = features.to(work_dev)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    first = torch.randint(0, N, (1,), generator=g).item()

    # Initialise: distance from every point to the first selected point
    selected_idx = [first]
    # min_dist[i] = squared distance to the closest selected point so far
    with torch.no_grad():
        # d2[i] = ||proj[i] - proj[first]||^2
        diff = proj - proj[first:first+1]
        min_d2 = (diff * diff).sum(-1)                         # (N,)

        for _ in range(n_select - 1):
            nxt = int(min_d2.argmax().item())
            selected_idx.append(nxt)
            # Update distances: distance to newly added point
            new = proj[nxt:nxt+1]
            d2_to_new = ((proj - new) ** 2).sum(-1)
            torch.minimum(min_d2, d2_to_new, out=min_d2)

    idx = torch.tensor(selected_idx, dtype=torch.long, device=features.device)
    return features.index_select(0, idx).contiguous()


# ---------------------------------------------------------------------------
# Per-class memory bank
# ---------------------------------------------------------------------------
@dataclass
class ClassBank:
    features: Tensor           # (M, D) memory bank, L2-normalised
    feat_dim: int              # D
    Hp: int
    Wp: int
    cls_prototype: Tensor      # (D,) mean CLS token, L2-normalised


class MultiClassPatchCore:
    """
    Multi-class PatchCore++ model.

    Workflow
    --------
    1. fit_class(cls_name, patch_features, cls_features) -> builds a per-class
       bank. Called during training (only normal samples available).
    2. build() -> runs coreset subsampling to shrink the bank for inference.
    3. predict(cls_name, patch_features, cls_features) -> returns an anomaly
       map (B, Hp, Wp) and an image-level score (B,).
    """

    def __init__(self,
                 coreset_ratio: float = 0.01,
                 coreset_max: int = 8192,
                 neighbourhood_size: int = 3,
                 n_neighbours: int = 5,
                 high_res: int = 448,
                 smooth_kernel: int = 9,
                 smooth_sigma: float = 4.0,
                 cls_bank_weight: float = 0.3,
                 random_proj_dim: int = 256):
        self.coreset_ratio = coreset_ratio
        self.coreset_max = coreset_max
        self.neighbourhood_size = neighbourhood_size
        self.n_neighbours = n_neighbours
        self.high_res = high_res
        self.smooth_kernel = smooth_kernel
        self.smooth_sigma = smooth_sigma
        self.cls_bank_weight = cls_bank_weight
        self.random_proj_dim = random_proj_dim
        self.banks: Dict[str, ClassBank] = {}
        self._raw_features: Dict[str, List[Tensor]] = {}
        self._raw_cls: Dict[str, List[Tensor]] = {}

    # ---- training -----------------------------------------------------------
    def add(self, cls: str, patch_feat: Tensor, cls_feat: Tensor):
        """
        Accumulate features for one class.

        patch_feat: (B, D, Hp, Wp) – L2-normalised fused multi-layer patches.
        cls_feat  : (B, D)         – L2-normalised CLS tokens.
        """
        B, D, Hp, Wp = patch_feat.shape
        flat = patch_feat.permute(0, 2, 3, 1).reshape(-1, D).cpu()
        if cls not in self._raw_features:
            self._raw_features[cls] = []
            self._raw_cls[cls] = []
            self.Hp, self.Wp = Hp, Wp
        self._raw_features[cls].append(flat)
        self._raw_cls[cls].append(cls_feat.cpu())

    def build(self, device: str | torch.device | None = None,
              bank_device: str | torch.device | None = "cpu"):
        """Finalize all per-class banks (coreset subsampling).

        Args:
            device: compute device used for coreset selection.
            bank_device: where to STORE the finalised bank tensors.
                "cpu" (default, safest): banks live on CPU and are lazily
                moved to the active GPU on each ``predict()`` call. Use
                ``banks_to_device()`` / ``banks_to_cpu()`` to explicitly
                prefetch / evict.
                "cuda": ~3 GB for 50 classes × 16k × 1024-d fp32; gives a
                small speedup if you have ≥24 GB GPU memory.
        """
        # If no device specified and CUDA is available, use it for coreset
        # selection (it's 10-100× faster than CPU).
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_dev = torch.device(device)

        for cls, feats in self._raw_features.items():
            all_feats = torch.cat(feats, dim=0)   # (N, D)
            all_cls = torch.cat(self._raw_cls[cls], dim=0)
            all_feats = all_feats[all_feats.abs().sum(-1) > 0]

            n_select = min(
                self.coreset_max,
                max(1024, int(self.coreset_ratio * all_feats.shape[0])),
            )
            bank = greedy_coreset(all_feats, n_select=n_select,
                                  random_proj_dim=self.random_proj_dim,
                                  device=compute_dev)
            bank = F.normalize(bank, dim=-1).contiguous()
            proto = F.normalize(all_cls.to(compute_dev).mean(0, keepdim=True),
                                dim=-1).squeeze(0)
            self.banks[cls] = ClassBank(
                features=bank.to(bank_device).contiguous(),
                feat_dim=bank.shape[1],
                Hp=self.Hp, Wp=self.Wp,
                cls_prototype=proto.to(bank_device).contiguous(),
            )
        self._raw_features.clear()
        self._raw_cls.clear()

    def banks_to_device(self, device: str | torch.device = "cuda"):
        """Prefetch all banks to ``device``. Call once before a prediction
        loop to get the best throughput; pairs with ``banks_to_cpu()``."""
        dev = torch.device(device)
        for bank in self.banks.values():
            if bank.features.device != dev:
                bank.features = bank.features.to(dev)
                bank.cls_prototype = bank.cls_prototype.to(dev)

    def banks_to_cpu(self):
        """Evict all banks back to CPU to free GPU memory."""
        for bank in self.banks.values():
            if bank.features.device.type != "cpu":
                bank.features = bank.features.to("cpu")
                bank.cls_prototype = bank.cls_prototype.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- inference ----------------------------------------------------------
    def _nearest_neighbour_distance(self, q: Tensor, bank: Tensor) -> Tensor:
        """
        q   : (B*N, D) L2-normalised
        bank: (M, D)   L2-normalised
        returns (B*N,) distance = (1 - cos_sim) per query patch.
        Batched matmul over chunks to avoid OOM.
        """
        chunk = 4096
        d_min = torch.empty(q.shape[0], device=q.device, dtype=q.dtype)
        for i in range(0, q.shape[0], chunk):
            qc = q[i:i+chunk]  # (c, D)
            sim = qc @ bank.t()  # (c, M) cosine similarity
            topk_sim = sim.topk(self.n_neighbours, dim=1).values.mean(1)
            d_min[i:i+chunk] = 1.0 - topk_sim
        return d_min

    @torch.no_grad()
    def predict(self, cls: str, patch_feat: Tensor,
                cls_feat: Tensor,
                return_map: bool = True) -> Dict[str, Tensor]:
        """
        patch_feat: (B, D, Hp, Wp) fused multi-layer patches
        cls_feat  : (B, D)         CLS tokens
        Returns dict with image_scores (B,) and anomaly_map (B, high_res, high_res).
        """
        assert cls in self.banks, f"Class {cls} not in memory bank"
        bank = self.banks[cls]
        dev = patch_feat.device
        # Lazy GPU migration of this class' bank. We do NOT auto-evict
        # here -- call banks_to_cpu() explicitly when you're done with
        # a prediction loop to free GPU memory.
        if bank.features.device != dev:
            bank.features = bank.features.to(dev)
            bank.cls_prototype = bank.cls_prototype.to(dev)
        bank_feats = bank.features

        # Make sure bank is in fp32 for the kNN matmul (fp16 matmul of
        # L2-normed vectors can give cos > 1 due to rounding, producing
        # negative distances).
        if bank_feats.dtype != torch.float32:
            bank_feats = bank_feats.float()
        if cls_feat.dtype != torch.float32:
            cls_feat = cls_feat.float()

        B, D, Hp, Wp = patch_feat.shape
        flat_p = patch_feat.float().permute(0, 2, 3, 1).reshape(-1, D)
        d = self._nearest_neighbour_distance(flat_p, bank_feats)
        amap = d.reshape(B, 1, Hp, Wp)

        # (1) Patch-level neighbourhood smoothing (3x3 avg)
        if self.neighbourhood_size > 1:
            pad = self.neighbourhood_size // 2
            amap = F.avg_pool2d(
                F.pad(amap, [pad]*4, mode="replicate"),
                kernel_size=self.neighbourhood_size, stride=1, padding=0
            )

        # (2) Upsample to high-res
        amap_hr = bilinear_upsample(amap, self.high_res)
        amap_hr = gaussian_smooth2d(amap_hr,
                                    kernel_size=self.smooth_kernel,
                                    sigma=self.smooth_sigma)

        # (3) Image score: weighted blend of (a) patch-level max distance
        # and (b) CLS-token distance to the class prototype.
        # Both terms are in the SAME scale ([0,2]) because features
        # and prototype are L2-normalised (dot = cos sim ∈ [-1,1]).
        patch_score = amap_hr.flatten(1).max(dim=1).values  # (B,)
        proto = bank.cls_prototype
        if proto.dim() == 1:
            proto = proto.unsqueeze(-1)  # (D,1)
        else:
            proto = proto.reshape(-1, 1)
        if proto.dtype != torch.float32:
            proto = proto.float()
        cls_dist = 1.0 - (cls_feat @ proto).squeeze(-1)  # (B,)
        img_score = (1.0 - self.cls_bank_weight) * patch_score \
                  + self.cls_bank_weight * cls_dist

        out = {
            "image_score": img_score,        # (B,)
            "anomaly_map": amap_hr[:, 0],    # (B, high_res, high_res)
        }

        if return_map:
            return out
        out.pop("anomaly_map")
        return out
