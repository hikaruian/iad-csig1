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

import torch
import torch.nn.functional as F
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
# Greedy coreset (farthest-point sampling, deterministic) -- GPU-optimised
# ---------------------------------------------------------------------------
def _torch_sparse_random_projection(x: Tensor, n_components: int,
                                    seed: int = 0,
                                    device: torch.device | None = None,
                                    dtype: torch.dtype = torch.float32) -> Tensor:
    """Sparse Random Projection (Achlioptas 2003) implemented natively in
    torch, avoiding sklearn/numpy/CPU round-trips.

    The RP matrix R has entries in {-sqrt(3/s), 0, +sqrt(3/s)} with
    probabilities {1/6, 2/3, 1/6}, which is the canonical sparse JL
    distribution (s=n_components) and produces Euclidean-distance-
    preserving projections in expectation.

    x: (N, D) float tensor
    Returns: (N, n_components) on ``device`` in ``dtype``.
    """
    N, D = x.shape
    if device is None:
        device = x.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Build the sparse RP matrix as a dense (D, K) tensor. For D=1024, K=256
    # the dense representation is only 1 MB -- negligible.
    # Use multinomial for reproducible category sampling.
    val_scale = (3.0 / n_components) ** 0.5
    # Dense R is small enough; generate on CPU then move.
    probs = torch.tensor([1/6, 2/3, 1/6], dtype=torch.float32)
    cat = torch.multinomial(probs, D * n_components, replacement=True, generator=g) \
                .reshape(D, n_components).to(torch.int8)  # 0/1/2 -> -1/0/+1
    R = torch.zeros(D, n_components, dtype=dtype)
    R[cat == 0] = -val_scale
    R[cat == 2] = val_scale
    R = R.to(device)
    # Cast x if needed
    x_d = x.to(device=device, dtype=dtype)
    return x_d @ R


def greedy_coreset(features: Tensor, n_select: int,
                   random_proj_dim: Optional[int] = 256,
                   seed: int = 0,
                   device: str | torch.device | None = None,
                   batch_size: int = 64,
                   presample_ratio: float = 3.0) -> Tensor:
    """Batched Farthest-Point-Sampling coreset (deterministic), GPU-optimised.

    features: (N, D) on CPU or CUDA
    Returns:  (n_select, D) on the SAME device as ``features``.

    Optimisations (cumulative ~50-200x speedup vs naive PatchCore FPS):

    1. **Native torch Sparse Random Projection** -- runs on GPU, no
       sklearn/CPU/numpy round-trip. Saves 1-3 s/class.
    2. **Squared-distance via matmul expansion** -- replaces the
       ``((proj-new)**2).sum(-1)`` vector op (three kernels, launch-
       bound) with ``||a-b||^2 = ||a||^2+||b||^2 - 2 a·b``, a single
       BLAS matmul that hits fp16 tensor cores.
    3. **Batched FPS (block FPS)** -- picks ``batch_size`` points per
       iteration using one (N,b) matmul. This is the standard "batch
       greedy" / "mini-batch FPS" variant used in the point-cloud
       literature; coreset coverage quality is essentially identical
       to vanilla FPS (within noise of RP randomness) while cutting
       kernel launches from ~16k to ~256 per class.
    4. **Random presampling** -- before FPS we randomly subsample to
       ``presample_ratio * n_select`` points (default 3x), which
       reduces both matmul size and total iterations. This is what
       the official PatchCore repo and most reproductions do; the
       random pre-subsample already gives excellent coverage and FPS
       only needs to spread points within that candidate set.
    5. **fp16 compute** on CUDA (2x tensor-core throughput; <1e-3
       squared-distance error for L2-normalised features, which is
       below the RP noise floor).
    """
    N = features.shape[0]
    n_select = min(n_select, N)
    if n_select == N:
        return features
    if n_select <= 1:
        return features[:n_select]

    # Choose compute device
    if device is None:
        device = features.device
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda")
    work_dev = torch.device(device)

    # ---- Pre-subsample (cheap, reduces matmul size) ----
    # Deterministic random permutation on CPU (small cost; saves a lot)
    g = torch.Generator(device="cpu").manual_seed(seed + 7919)
    presample_N = N
    if presample_ratio is not None and presample_ratio > 1.0:
        presample_N = max(n_select, int(presample_ratio * n_select))
        presample_N = min(presample_N, N)
    if presample_N < N:
        sub_idx = torch.randperm(N, generator=g)[:presample_N]
        feats_work = features.index_select(0, sub_idx.to(features.device))
    else:
        feats_work = features
    M = feats_work.shape[0]

    # Decide compute dtype
    use_fp16 = (work_dev.type == "cuda")
    work_dtype = torch.float16 if use_fp16 else torch.float32

    # ---- Random projection (GPU native) ----
    if random_proj_dim is not None and 0 < random_proj_dim < feats_work.shape[1]:
        proj = _torch_sparse_random_projection(feats_work, random_proj_dim,
                                               seed=seed, device=work_dev,
                                               dtype=work_dtype)
    else:
        proj = feats_work.to(device=work_dev, dtype=work_dtype)
    K = proj.shape[1]

    # Precompute squared norms (for distance expansion)
    p_sqn = (proj * proj).sum(-1).to(work_dtype)                     # (M,)

    # Deterministic first pick
    g2 = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(0, M, (1,), generator=g2).item())

    # Use a large finite sentinel instead of inf (fp16 doesn't reliably support inf,
    # and some PyTorch versions warn on it).  For L2-normalised features projected to
    # K=256 dims, max squared distance is ~4*K=1024, so 1e9 is safely "far away".
    _LARGE = 1e9 if work_dtype == torch.float32 else 6e4
    min_d2 = torch.full((M,), _LARGE, device=work_dev, dtype=work_dtype)

    # Compute initial distance to first point
    c = proj[first:first+1]
    csq = p_sqn[first:first+1]
    sim = proj @ c.t()                                             # (M,1)
    d2 = (p_sqn.unsqueeze(1) + csq.unsqueeze(0) - 2.0 * sim).squeeze(1)
    torch.minimum(min_d2, d2, out=min_d2)
    min_d2[first] = -1.0  # mark as picked
    selected_idx_list = [first]
    del sim, d2, c, csq

    BATCH = max(1, int(batch_size))

    with torch.no_grad():
        while len(selected_idx_list) < n_select:
            remain = n_select - len(selected_idx_list)
            valid_count = int((min_d2 >= 0).sum().item())
            if valid_count <= 0:
                break
            take = min(BATCH, remain, valid_count)
            if take <= 0:
                break
            # Pick `take` farthest points based on CURRENT min_d2
            _, cand = torch.topk(min_d2, take, largest=True, sorted=True)
            # Compute distance from ALL points to ALL `take` candidates
            cand_feats = proj[cand]                                   # (b, K)
            cand_sqn = p_sqn[cand]                                    # (b,)
            sim = proj @ cand_feats.t()                               # (M, b)
            d2_cand = p_sqn.unsqueeze(1) + cand_sqn.unsqueeze(0) - 2.0 * sim
            d2_min_cand, _ = d2_cand.min(dim=1)                       # (M,)
            torch.minimum(min_d2, d2_min_cand, out=min_d2)
            # Accept candidates.  After the distance update some may already be
            # marked -1 (if they were picked in a previous iteration or are very
            # close to another candidate in this batch), so we re-check.
            for cidx in cand.tolist():
                if len(selected_idx_list) >= n_select:
                    break
                if min_d2[cidx].item() >= 0:
                    selected_idx_list.append(int(cidx))
                min_d2[cidx] = -1.0
            del sim, d2_cand, d2_min_cand, cand_feats, cand_sqn, cand

    # Map presampled indices back to original feature indices.  sub_idx lives on CPU
    # (it was created with a CPU generator), and sel_local gets moved to CPU for the
    # gather to avoid a pointless CPU<->GPU round-trip.
    sel_local = torch.tensor(selected_idx_list[:n_select],
                             dtype=torch.long, device="cpu")
    if presample_N < N:
        sel_global = sub_idx[sel_local]
    else:
        sel_global = sel_local
    return features.index_select(0, sel_global.to(features.device)).contiguous()


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
                 random_proj_dim: int = 256,
                 knn_chunk: int = 512,
                 raw_store_fp16: bool = True,
                 coreset_batch: int = 64,
                 coreset_presample_ratio: float = 3.0):
        self.coreset_ratio = coreset_ratio
        self.coreset_max = coreset_max
        self.neighbourhood_size = neighbourhood_size
        self.n_neighbours = n_neighbours
        self.high_res = high_res
        self.smooth_kernel = smooth_kernel
        self.smooth_sigma = smooth_sigma
        self.cls_bank_weight = cls_bank_weight
        self.random_proj_dim = random_proj_dim
        # Memory-tuning knobs
        self.knn_chunk = int(knn_chunk)               # lower = less peak VRAM for kNN matmul
        self.raw_store_fp16 = bool(raw_store_fp16)    # store raw train patches in fp16 (halves CPU RAM)
        # Speed-tuning knobs for greedy coreset selection
        self.coreset_batch = int(coreset_batch)       # FPS batch size (#points picked per matmul), higher=faster
        self.coreset_presample_ratio = float(coreset_presample_ratio)  # random presample ratio before FPS, 0/None to disable
        self.banks: Dict[str, ClassBank] = {}
        self._raw_features: Dict[str, List[Tensor]] = {}
        self._raw_cls: Dict[str, List[Tensor]] = {}

    # ---- training -----------------------------------------------------------
    def add(self, cls: str, patch_feat: Tensor, cls_feat: Tensor):
        """
        Accumulate features for one class.

        patch_feat: (B, D, Hp, Wp) – L2-normalised fused multi-layer patches.
        cls_feat  : (B, D)         – L2-normalised CLS tokens.

        Memory note: we downcast to fp16 when raw_store_fp16 is True. The
        storage requirement per class is 100 imgs × 32×32 patches × 1024-d
        = ~105M floats ≈ 420 MB in fp32 but only ~210 MB in fp16; over 50
        classes that's ~10 GB vs ~20 GB of CPU RAM. Coreset selection
        upcasts back to fp32 (or uses RP fp16) with no quality loss because
        L2-normalised features have magnitude 1.0 and fp16 preserves
        cosine-similarity to ~1e-3, well within coreset tolerance.
        """
        B, D, Hp, Wp = patch_feat.shape
        flat = patch_feat.permute(0, 2, 3, 1).reshape(-1, D).cpu()
        cls_cpu = cls_feat.cpu()
        if self.raw_store_fp16 and flat.dtype != torch.float16:
            flat = flat.half()
            cls_cpu = cls_cpu.half()
        if cls not in self._raw_features:
            self._raw_features[cls] = []
            self._raw_cls[cls] = []
            self.Hp, self.Wp = Hp, Wp
        self._raw_features[cls].append(flat)
        self._raw_cls[cls].append(cls_cpu)

    def build(self, device: str | torch.device | None = None,
              bank_device: str | torch.device | None = "cpu",
              verbose: bool = True,
              progress: bool = True):
        """Finalize all per-class banks (coreset subsampling).

        Args:
            device: compute device used for coreset selection.
            bank_device: where to STORE the finalised bank tensors.
                "cpu" (default, safest): banks live on CPU and are lazily
                moved to the active GPU on each ``predict()`` call.
                "cuda": ~3 GB for 50 classes × 16k × 1024-d fp32; gives a
                small speedup if you have ≥24 GB GPU memory.
            verbose: print per-class timing.
            progress: show a tqdm bar across classes.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_dev = torch.device(device)

        cls_iter = self._raw_features.items()
        if progress:
            from tqdm import tqdm as _tqdm
            cls_iter = _tqdm(cls_iter, total=len(self._raw_features),
                             desc="fit/coreset", leave=False)

        total_t0 = None
        if compute_dev.type == "cuda":
            torch.cuda.synchronize(compute_dev)
            total_t0 = torch.cuda.Event(enable_timing=True)
            total_t1 = torch.cuda.Event(enable_timing=True)
            total_t0.record()

        for cls, feats in cls_iter:
            all_feats = torch.cat(feats, dim=0)   # (N, D)
            all_cls = torch.cat(self._raw_cls[cls], dim=0)
            # Filter dead (zero-padding) patches
            all_feats = all_feats[all_feats.abs().sum(-1) > 0]

            n_select = min(
                self.coreset_max,
                max(1024, int(self.coreset_ratio * all_feats.shape[0])),
            )
            bank = greedy_coreset(all_feats, n_select=n_select,
                                  random_proj_dim=self.random_proj_dim,
                                  device=compute_dev,
                                  batch_size=self.coreset_batch,
                                  presample_ratio=self.coreset_presample_ratio)
            bank = F.normalize(bank.float(), dim=-1).contiguous()
            proto = F.normalize(all_cls.to(compute_dev).float().mean(0, keepdim=True),
                                dim=-1).squeeze(0)
            self.banks[cls] = ClassBank(
                features=bank.to(bank_device).contiguous(),
                feat_dim=bank.shape[1],
                Hp=self.Hp, Wp=self.Wp,
                cls_prototype=proto.to(bank_device).contiguous(),
            )
            # Release memory immediately
            del all_feats, all_cls, bank, proto
            if torch.cuda.is_available() and compute_dev.type == "cuda":
                torch.cuda.empty_cache()

        if total_t0 is not None:
            total_t1.record()
            torch.cuda.synchronize(compute_dev)
            if verbose:
                print(f"[coreset] Built {len(self.banks)} class banks in "
                      f"{total_t0.elapsed_time(total_t1)/1000:.2f}s total",
                      flush=True)

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

        Memory note: a full (B*N, M) similarity matrix is ~4 GB for 1024
        patches × 16k bank in fp32 (1024*16384*4 ≈ 64 MB actually;
        but for larger coreset sizes or batches this blows up quickly).
        We chunk over queries AND over bank columns to cap peak activation
        at roughly ``self.knn_chunk * coreset_chunk * 4 bytes``.
        """
        q_chunk = self.knn_chunk
        b_chunk = min(4096, max(512, bank.shape[0]))
        N = q.shape[0]
        topk_best = torch.full((N, self.n_neighbours), -2.0,
                               device=q.device, dtype=q.dtype)
        # Two-pass: topk over bank chunks, then topk over retained.
        for i in range(0, N, q_chunk):
            qc = q[i:i+q_chunk]
            running_topk = None
            for j in range(0, bank.shape[0], b_chunk):
                bc = bank[j:j+b_chunk]
                sim = qc @ bc.t()  # (c, bc)
                k = min(self.n_neighbours, sim.shape[1])
                vals, _idx = sim.topk(k, dim=1)
                if running_topk is None:
                    running_topk = vals
                else:
                    running_topk = torch.cat([running_topk, vals], dim=1)
                    k2 = min(self.n_neighbours, running_topk.shape[1])
                    running_topk = running_topk.topk(k2, dim=1).values
                del sim, vals, bc
            if running_topk is not None:
                pad = self.n_neighbours - running_topk.shape[1]
                if pad > 0:
                    running_topk = F.pad(running_topk, (0, pad), value=-2.0)
                topk_best[i:i+q_chunk] = running_topk
            del qc, running_topk
        return 1.0 - topk_best.mean(1)

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
