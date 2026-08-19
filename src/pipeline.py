"""
End-to-end training & inference pipeline.

Usage (high level):
    from src.pipeline import CSIGAnomalyPipeline
    pipe = CSIGAnomalyPipeline(cfg)
    pipe.fit("path/to/CSIG/Train")
    pipe.predict_and_save("path/to/CSIG/Test_A", out_root="submission")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .backbones import CLIPFeatureExtractor, DINOv2FeatureExtractor
from .dataset import (CSIGImageDataset, CSIGSampleDataset, build_test_transform,
                      build_train_transform, discover_classes,
                      IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD)
from .foreground import (apply_foreground, foreground_from_saliency,
                         remove_small_components_torch)
from .multiview import (CrossViewAttention, aggregate_image_scores,
                        multiview_mask_vote)
from .patchcore import MultiClassPatchCore, bilinear_upsample, gaussian_smooth2d
from .utils import (PerClassPercentileCalibrator, save_mask_png,
                    save_submission_csv, tta_flips, unflip_map)
from .zeroshot import build_winclip_reference, winclip_score
from .dist_utils import unwrap


# Pre-allocate statistic tensors once (small, shared across calls) to avoid
# rebuilding them on every forward pass.
_IMAGENET_MEAN_T: Optional[torch.Tensor] = None
_IMAGENET_STD_T: Optional[torch.Tensor] = None
_CLIP_MEAN_T: Optional[torch.Tensor] = None
_CLIP_STD_T: Optional[torch.Tensor] = None


def _get_stat_tensors(device: torch.device, dtype: torch.dtype):
    """Return cached (IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD) as
    4D (1,3,1,1) tensors on ``device`` with ``dtype``.  Created once and
    moved/cloned as needed – avoids the per-sample ``new_tensor`` overhead.
    """
    global _IMAGENET_MEAN_T, _IMAGENET_STD_T, _CLIP_MEAN_T, _CLIP_STD_T
    if _IMAGENET_MEAN_T is None:
        _IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        _IMAGENET_STD_T = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        _CLIP_MEAN_T = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        _CLIP_STD_T = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    return (
        _IMAGENET_MEAN_T.to(device=device, dtype=dtype),
        _IMAGENET_STD_T.to(device=device, dtype=dtype),
        _CLIP_MEAN_T.to(device=device, dtype=dtype),
        _CLIP_STD_T.to(device=device, dtype=dtype),
    )


def _imagenet_to_clip(x: torch.Tensor) -> torch.Tensor:
    """Convert an ImageNet-normalised tensor (any resolution) to CLIP-
    normalised tensor **at the same resolution**.  No resize is performed:
    we simply denormalise with ImageNet stats and renormalise with CLIP
    stats in floating point.  This keeps the CLIP branch perfectly
    registered with the DINO branch (pixel-exact alignment), which is
    critical for the mask ensemble.
    """
    mean_in, std_in, mean_out, std_out = _get_stat_tensors(x.device, x.dtype)
    x01 = x * std_in + mean_in              # denorm to [0,1]
    x01 = x01.clamp(0.0, 1.0)
    return (x01 - mean_out) / std_out


@dataclass
class PipelineConfig:
    # ---- paths ----
    train_root: str = "CSIG/Train"
    test_root: str = "CSIG/Test_A"
    out_dir: str = "submission"
    use_dp: bool = False

    # ---- backbone ----
    dinov2_model: str = "vitl14"
    # For DINOv2-L/14 (depth 24) pretrained with iBOT+registers, the blocks
    # most useful for AD are the LATE blocks (roughly 15-23), where
    # semantic/surface-level information has fully formed. The old choice
    # of (8,9,10,11) came from DINOv1-era papers and is too shallow for
    # DINOv2 -- blocks 8-11 still carry mostly low-level texture that hurts
    # P-AP by lighting up on normal texture edges. Using (16,19,22) spans
    # the late-to-final range and matches AnomalyDINO / UnityAD defaults.
    dinov2_layers: Tuple[int, ...] = (16, 19, 22)
    input_size: int = 448
    # Multi-scale TTA: run the backbone at multiple input resolutions,
    # resize each resulting anomaly map back to (input_size, input_size),
    # and geometric-mean them. This is a zero-extra-weights trick that
    # consistently gives +3-8pp P-AP in AD competitions because defects
    # exist at characteristic physical scales (scratches = fine, dents =
    # medium, stains = large). Single-scale inference cannot cover all
    # of them simultaneously. Costs extra forward passes but we drop
    # v-flip/hv-flip from TTA to keep total runtime comparable.
    multi_scale: Tuple[int, ...] = (392, 448, 518)
    # Per-scale weights for ARITHMETIC multi-scale fusion. If empty/None or
    # length doesn't match multi_scale, falls back to geometric mean.
    # Recommended: [0.2, 0.6, 0.2] -- bias to native 448 resolution.
    multi_scale_weights: Tuple[float, ...] = ()
    use_clip: bool = True
    clip_model: str = "ViT-L-14-336"
    clip_pretrained: str = "openai"

    # ---- patchcore ----
    coreset_ratio: float = 0.10
    coreset_max: int = 49152
    neighbourhood_size: int = 1    # 1 = no patch-level averaging (preserves sharp defects)
    n_neighbours: int = 3          # k=3 NN (mean of 3 nearest) averages out single-bank-vector
                                   # noise without smearing peaks. Standard PatchCore default.
                                   # Bank needs to be dense (>=10% coverage) for k=3 to help.
    smooth_kernel: int = 7
    smooth_sigma: float = 2.5      # lighter Gaussian to avoid smearing small defects
    cls_bank_weight: float = 0.25
    # When True, CLIP branch contributes only to image-level score (NOT mask);
    # CLIP patch maps are too noisy for pixel localisation on Real-IAD.
    clip_mask_ens: bool = False
    # Cross-view mask voting. "none" = per-view independent (BEST for pixel AP, since
    # 5 views are NOT pixel-registered and median kills single-view defects);
    # "median" = old behaviour, "mean" = 0.7·orig + 0.3·mean across views.
    mv_mask_vote: str = "none"

    # ---- multi-view ----
    # NOTE: "mean" is preferred over "robust_mean" for Real-IAD because
    # defects can appear on a SINGLE camera view (the 5 views show different
    # physical surfaces). robust_mean drops the highest (and lowest) view,
    # which can throw away the only view carrying a real defect.
    mv_aggregate: str = "mean"
    mv_vote_beta: float = 1.5
    use_mv_attn: bool = False     # NOTE: mv_attn is not used in default scoring;
                                  # keeping it False avoids allocating an unused
                                  # attention module (~16 MB for ViT-L).

    # ---- TTA ----
    use_tta: bool = True
    tta_weight_orig: float = 1.0
    tta_weight_flip: float = 0.5

    # ---- ensemble ----
    ens_dino_weight: float = 0.75
    ens_clip_weight: float = 0.25
    # Separate (lighter) weight for CLIP's MASK ensemble. CLIP patch maps
    # carry useful semantic "defect-ness" prior but are spatially noisy; a
    # small weight (~0.06-0.12) blends them in for free (CLIP encode_image
    # already ran for the image score) without smearing DINO localisation.
    ens_clip_mask_weight: float = 0.0

    # ---- normalisation percentiles for DINO anomaly maps (from train set) ----
    # Use median for lo so background stays at ~0; use a high hi so strong
    # anomalies saturate near 1 while weak noise stays suppressed -- this
    # dramatically improves pixel AP by avoiding stretching background noise
    # across the full [0,1] range.
    dino_map_lo_q: float = 0.50    # median of normal-sample scores maps to ~0
    dino_map_hi_q: float = 0.997   # leave the top 0.3% tail to saturate at 1
    dino_img_lo_q: float = 0.05
    dino_img_hi_q: float = 0.95

    # ---- foreground mask (DINOv2 CLS<->patch final-norm cosine saliency) ----
    # NOTE: applied AFTER percentile normalisation on the FINAL ensembled mask
    # (post-calibration). If applied BEFORE calibration the calibration
    # percentiles absorb the suppression and the foreground mask has zero
    # net effect on the [0,1]-stretched mask.
    use_foreground_mask: bool = True
    fg_percentile: float = 40.0
    fg_bg_floor: float = 0.15     # background multiplier (0.15 = heavy suppression)
    fg_smooth_sigma: float = 2.5
    fg_hard_threshold: float = 0.0  # if >0, ZERO OUT pixels with fg below this
                                    # saliency (hard-kill), not just multiply.

    # ---- per-class ZCA feature whitening (Mahalanobis distance instead of
    # cosine) inside PatchCore. Costs one eigendecomposition per class (on
    # the coreset, ~0.2s) and improves P-AP on textured classes.
    use_whitening: bool = False
    whitening_eps: float = 0.01

    # ---- hard-negative mining (add high-distance normal patches to bank) ----
    hard_negative_k: int = 0          # off by default; enable with care (512-2048)
                                      # over-eager HN mining collapses the bank
                                      # and kills image-AUROC.

    # ---- connected-component suppression ----
    cc_min_area: int = 20
    cc_threshold: float = 0.35
    cc_kill_factor: float = 0.10

    # ---- post-processing ----
    mask_gamma: float = 2.0       # gamma on final masks (>1 suppresses weak FPs, boosts P-AP)

    # ---- memory / precision ----
    batch_size: int = 4
    clip_batch_size: int = 1      # separate (smaller) batch size for CLIP forward
    dino_view_micro: int = 2      # max views to feed DINO in one micro-batch
                                  # during TTA inference (lowers activation peak)
    knn_chunk: int = 512          # query chunk for kNN matmul (lower = less VRAM)
    coreset_batch: int = 64       # FPS batch size -- larger = faster coreset build
    coreset_presample_ratio: float = 3.0  # random presample multiplier before FPS (speeds up coreset selection)
    num_workers: int = 4          # DataLoader workers; 4 is safer for 16 GB CPU RAM
    use_amp: bool = True          # use fp16 autocast for backbones (halves activation memory)
    banks_on_gpu: bool = False    # prefetch all coreset banks to GPU (faster but ~3.2 GB)
    coreset_on_gpu: bool = True   # run greedy coreset selection on GPU (fast; ~<500 MB temp)
    device: str = "cuda"
    seed: int = 42


class CSIGAnomalyPipeline:
    """Top-level pipeline implementing the full solution."""

    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.device = torch.device(
            self.cfg.device if torch.cuda.is_available() else "cpu"
        )
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self.use_amp = bool(self.cfg.use_amp and self.device.type == "cuda")
        self.amp_dtype = torch.float16  # safe for DINOv2 + CLIP ViT

        print(f"[pipeline] device={self.device}, use_amp={self.use_amp}, use_dp={self.cfg.use_dp}")

        print(f"[pipeline] Loading DINOv2 {self.cfg.dinov2_model} ...")
        self.dino = DINOv2FeatureExtractor(
            model_name=self.cfg.dinov2_model,
            layers=self.cfg.dinov2_layers,
            pretrained=True,
        )
        if self.use_amp:
            self.dino = self.dino.half()
        self.dino = self.dino.to(self.device).eval()

        self.clip = None
        if self.cfg.use_clip:
            print(f"[pipeline] Loading CLIP {self.cfg.clip_model} ...")
            self.clip = CLIPFeatureExtractor(
                model_name=self.cfg.clip_model,
                pretrained=self.cfg.clip_pretrained,
            )
            if self.use_amp:
                self.clip = self.clip.half()
            self.clip = self.clip.to(self.device).eval()

        self.classes: List[str] = []
        self.patchcore = MultiClassPatchCore(
            coreset_ratio=self.cfg.coreset_ratio,
            coreset_max=self.cfg.coreset_max,
            neighbourhood_size=self.cfg.neighbourhood_size,
            n_neighbours=self.cfg.n_neighbours,
            high_res=self.cfg.input_size,
            smooth_kernel=self.cfg.smooth_kernel,
            smooth_sigma=self.cfg.smooth_sigma,
            cls_bank_weight=self.cfg.cls_bank_weight,
            knn_chunk=self.cfg.knn_chunk,
            raw_store_fp16=True,
            coreset_batch=self.cfg.coreset_batch,
            coreset_presample_ratio=self.cfg.coreset_presample_ratio,
            use_whitening=bool(getattr(self.cfg, "use_whitening", False)),
            whitening_eps=float(getattr(self.cfg, "whitening_eps", 0.01)),
        )
        # Calibrator for image-level scores (per-class)
        self.calibrator = PerClassPercentileCalibrator()
        # Per-class (lo, hi) scalars for normalising DINO anomaly maps from
        # raw distance scale to roughly [0, 1], using statistics collected
        # on the TRAIN set. Using train-set percentiles avoids the
        # "per-sample normalisation always flags something" pitfall.
        self.map_norm: Dict[str, Tuple[float, float]] = {}

        self.clip_refs: Dict[str, torch.Tensor] = {}
        # Cross-view attention is NOT used in the default scoring formula
        # (mean-pooled CLS is more stable); only allocate if explicitly
        # requested to save ~16 MB of GPU memory.
        self.mv_attn = None
        if self.cfg.use_mv_attn:
            mv_dim = unwrap(self.dino).embed_dim
            self.mv_attn = CrossViewAttention(mv_dim).to(self.device).eval()
            if self.use_amp:
                self.mv_attn = self.mv_attn.half()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _autocast_ctx(self):
        """Return a context manager: fp16 autocast when AMP enabled."""
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True)
        from contextlib import nullcontext
        return nullcontext()

    def _prefetch_banks(self, mode: str):
        """Bring coreset banks onto the GPU according to ``mode``.

        mode:
          "all"   : prefetch every class (fastest, uses ~3.2 GB).
          "lazy"  : no prefetch; predict() migrates per-class.
          "cls"   : lazy + explicit per-class eviction in this module.
        Returns the resolved mode string.  On OOM falls back to "lazy".
        """
        if mode == "all" and self.device.type == "cuda":
            try:
                self.patchcore.banks_to_device(self.device)
                return "all"
            except torch.cuda.OutOfMemoryError:
                self._empty_cache()
                print("[pipeline] bank prefetch OOM -- falling back to lazy migration")
                return "lazy"
        return mode if mode in ("lazy", "cls") else "lazy"

    def _ensure_bank_on_device(self, cls: str, banks_mode: str):
        """Migrate a single class' bank to the active device when in lazy mode."""
        if banks_mode == "all":
            return
        bk = self.patchcore.banks.get(cls, None)
        if bk is not None and bk.features.device != self.device:
            bk.features = bk.features.to(self.device, non_blocking=True)
            bk.cls_prototype = bk.cls_prototype.to(self.device, non_blocking=True)
            if bk.W is not None:
                bk.W = bk.W.to(self.device, non_blocking=True)
            if bk.mean is not None:
                bk.mean = bk.mean.to(self.device, non_blocking=True)

    def _evict_bank_to_cpu(self, cls: str):
        """Move a single class' bank + CLIP ref back to CPU to free GPU memory."""
        bk = self.patchcore.banks.get(cls, None)
        if bk is not None and bk.features.device.type != "cpu":
            bk.features = bk.features.to("cpu", non_blocking=True)
            bk.cls_prototype = bk.cls_prototype.to("cpu", non_blocking=True)
            if bk.W is not None:
                bk.W = bk.W.to("cpu", non_blocking=True)
            if bk.mean is not None:
                bk.mean = bk.mean.to("cpu", non_blocking=True)
        cr = self.clip_refs.get(cls, None)
        if cr is not None and cr.device.type != "cpu":
            self.clip_refs[cls] = cr.to("cpu", non_blocking=True)

    def _run_backbone(self, module, x, call: str = "__call__", **kwargs):
        if self.cfg.use_dp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            from .dist_utils import auto_parallel_forward
            with self._autocast_ctx():
                return auto_parallel_forward(module, x, call=call, **kwargs)
        fn = module if call == "__call__" else getattr(module, call)
        with self._autocast_ctx():
            return fn(x, **kwargs)

    @staticmethod
    def _empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _move_input(self, x: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor to the active device. When AMP is enabled we
        upload in fp16 to cut PCIe bandwidth and reduce the input-side
        activation budget; autocast still handles precision inside the
        ViT, but the first conv / patch-embed operates directly on fp16
        which saves ~50 MB per view."""
        if self.use_amp:
            return x.to(self.device, dtype=torch.float16, non_blocking=True)
        return x.to(self.device, non_blocking=True)

    def _dino_backbone_micro(self, x: torch.Tensor, return_fg: bool = False
                             ) -> Dict[str, torch.Tensor]:
        """Run DINOv2 backbone in micro-batches of views to cap activation
        memory. Each micro-batch is concatenated back, matching the output
        shape of a single big forward (B=V). When return_fg=True we also
        harvest the final-norm CLS<->patch cosine saliency (free signal)
        used by the foreground mask."""
        micro = max(1, int(self.cfg.dino_view_micro))
        patches, clses, fgs = [], [], []
        for i in range(0, x.shape[0], micro):
            sub = x[i:i+micro]
            if self.cfg.use_dp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                from .dist_utils import auto_parallel_forward
                with self._autocast_ctx():
                    o = auto_parallel_forward(self.dino, sub)
            else:
                with self._autocast_ctx():
                    o = self.dino.extract_features(sub, return_fg_saliency=return_fg)
            patches.append(o["patch"].float())
            clses.append(o["cls"].float())
            if return_fg and "fg_saliency" in o:
                fgs.append(o["fg_saliency"].float())
            del sub, o
        out = {"patch": torch.cat(patches, dim=0),
               "cls": torch.cat(clses, dim=0)}
        if return_fg and fgs:
            out["fg_saliency"] = torch.cat(fgs, dim=0)
        return out

    def _dino_forward(self, x: torch.Tensor, cls: str,
                      return_map: bool = True,
                      return_fg: bool = False,
                      target_size: int | None = None,
                      ) -> Dict[str, torch.Tensor]:
        """Run DINOv2 + PatchCore and return per-view image scores and
        (optionally) per-view high-res anomaly maps. Shared by training
        calibration and inference so the two paths are NUMERICALLY
        IDENTICAL (mod TTA at inference time).

        target_size: passed to PatchCore.predict so multi-scale inference
            can produce anomaly maps at the correct resolution before
            resizing to canonical input_size in the caller.
        """
        out = self._dino_backbone_micro(x, return_fg=return_fg)
        res = self.patchcore.predict(cls, out["patch"], out["cls"],
                                     return_map=return_map,
                                     target_size=target_size)
        if return_fg and "fg_saliency" in out:
            res["fg_saliency"] = out["fg_saliency"]
        return res

    def _dino_tta_forward(self, views: torch.Tensor, cls: str,
                          collect_fg: bool = False,
                          ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run DINO branch with flip TTA and multi-scale TTA.

        Returns (img_raw, dino_map, fg_mask_or_None) at the pipeline's
        canonical input_size resolution.
        """
        # Multi-scale TTA: run DINO+PatchCore at multiple input resolutions,
        # resize each resulting anomaly map back to (input_size, input_size),
        # and fuse them. Two modes supported via cfg.multi_scale_weights:
        #   - None / all-ones: GEOMETRIC mean (product of scale maps)^(1/n).
        #     Requires a response to appear at multiple scales to survive,
        #     which is aggressive (kills scale-specific defects).
        #   - list of positive floats (same length as cfg.multi_scale):
        #     ARITHMETIC weighted mean. Scales are treated as complementary
        #     evidence; weights are normalised to sum=1 internally. This is
        #     the recommended setting. A canonical choice is
        #     multi_scale=(392,448,518) with weights=(0.2,0.6,0.2) to bias
        #     toward the native 448 resolution.
        scales = list(getattr(self.cfg, "multi_scale", ()) or ())
        scale_weights = list(getattr(self.cfg, "multi_scale_weights", ()) or ())
        if not scales:
            scales = [self.cfg.input_size]
        if not scale_weights or len(scale_weights) != len(scales):
            scale_weights = [1.0] * len(scales)
            fuse_mode = "geom"
        else:
            sw_sum = float(sum(scale_weights))
            scale_weights = [float(w) / sw_sum for w in scale_weights]
            fuse_mode = "arith"

        V = views.shape[0]
        H = W = self.cfg.input_size

        # Accumulator for scale fusion (reused for both geom and arith).
        if fuse_mode == "geom":
            log_map_sum = torch.zeros(V, H, W, device=views.device, dtype=torch.float32)
            map_wsum = 0.0
        else:
            map_sum = torch.zeros(V, H, W, device=views.device, dtype=torch.float32)
            map_wsum = 0.0
        pv_sum = torch.zeros(V, device=views.device, dtype=torch.float32)
        pv_wsum = 0.0
        fg_mask = None

        for si, scale in enumerate(scales):
            if scale == H:
                views_s = views
            else:
                views_s = F.interpolate(views, size=(scale, scale),
                                         mode="bilinear", align_corners=False)
            augs = tta_flips(views_s) if self.cfg.use_tta else [("orig", views_s)]
            scale_map = torch.zeros(V, H, W, device=views.device, dtype=torch.float32)
            scale_pv = torch.zeros(V, device=views.device, dtype=torch.float32)
            scale_w = 0.0
            for name, x in augs:
                w = self.cfg.tta_weight_orig if name == "orig" else self.cfg.tta_weight_flip
                need_fg = (collect_fg and scale == H and name == "orig" and fg_mask is None)
                res = self._dino_forward(x, cls, return_map=True, return_fg=need_fg,
                                         target_size=scale)
                am = res["anomaly_map"]
                if am.shape[-1] != W or am.shape[-2] != H:
                    am = F.interpolate(am.unsqueeze(1), size=(H, W),
                                       mode="bilinear", align_corners=False).squeeze(1)
                if name != "orig":
                    am = unflip_map(am.unsqueeze(1), name).squeeze(1)
                scale_map.add_(am, alpha=w)
                scale_pv.add_(res["image_score"], alpha=w)
                scale_w += w
                if need_fg and "fg_saliency" in res:
                    try:
                        sal = res["fg_saliency"]
                        fg_mask = foreground_from_saliency(
                            sal, out_size=H,
                            percentile=self.cfg.fg_percentile,
                            smooth_sigma=self.cfg.fg_smooth_sigma,
                        ).to(am.dtype)
                        if fg_mask.shape[0] != am.shape[0]:
                            fg_mask = None
                    except Exception:
                        fg_mask = None
                del res, am, x
            if scale_w > 0:
                scale_map.div_(scale_w)
                scale_pv.div_(scale_w)
            sw = scale_weights[si]
            if fuse_mode == "geom":
                log_map_sum.add_(torch.log(scale_map.clamp(min=1e-6)), alpha=sw)
            else:
                map_sum.add_(scale_map, alpha=sw)
            map_wsum += sw
            pv_sum.add_(scale_pv, alpha=sw)
            pv_wsum += sw
            del views_s, scale_map, scale_pv

        if fuse_mode == "geom":
            dino_map = torch.exp(log_map_sum / max(1e-6, map_wsum))
        else:
            dino_map = map_sum / max(1e-6, map_wsum)
        pv = pv_sum / max(1e-6, pv_wsum)

        img_raw = aggregate_image_scores(
            pv.unsqueeze(0), strategy=self.cfg.mv_aggregate
        )  # (1,)
        return img_raw, dino_map, fg_mask if collect_fg else None

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def fit(self, train_root: Optional[str] = None):
        train_root = Path(train_root or self.cfg.train_root)
        self.classes = discover_classes(train_root)
        print(f"[fit] Found {len(self.classes)} classes: {self.classes}")

        tfm = build_train_transform(self.cfg.input_size)
        ds = CSIGImageDataset(train_root, transform=tfm, classes=self.classes)
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False,
                            num_workers=self.cfg.num_workers, pin_memory=True,
                            persistent_workers=False)
        print(f"[fit] Extracting DINOv2 features from {len(ds)} training images ...")

        clip_patch_accum: Dict[str, List[torch.Tensor]] = {}

        # ---- Phase 1: extract DINO patch/CLS features ----
        for batch in tqdm(loader, desc="fit/dino"):
            imgs = self._move_input(batch["image"])
            cls_names = batch["cls_name"]
            with self._autocast_ctx():
                out = self._run_backbone(self.dino, imgs)
            # Cast back to fp32 for storage & kNN; move to CPU immediately.
            # patchcore.add will downcast to fp16 if raw_store_fp16 is on.
            patch = out["patch"].float().cpu()
            cls_tok = out["cls"].float().cpu()
            for b in range(imgs.shape[0]):
                c = cls_names[b]
                self.patchcore.add(c, patch[b:b+1], cls_tok[b:b+1])
            del imgs, out, patch, cls_tok
        self._empty_cache()

        # ---- Phase 2: build CLIP reference bank (smaller batch, fp16) ----
        if self.clip is not None:
            # Using input_size CLIP-normalised images gives a 32x32 patch
            # grid that aligns pixel-exactly with DINO's 32x32 grid.
            clip_tfm = build_train_transform(self.cfg.input_size, norm="clip")
            ds_c = CSIGImageDataset(train_root, transform=clip_tfm, classes=self.classes)
            clip_bs = max(1, int(self.cfg.clip_batch_size))
            ld_c = DataLoader(ds_c, batch_size=clip_bs, shuffle=False,
                              num_workers=self.cfg.num_workers, pin_memory=True,
                              persistent_workers=False)
            print("[fit] Building WinCLIP reference banks ...")
            for batch in tqdm(ld_c, desc="fit/clip"):
                imgs = self._move_input(batch["image"])
                cls_names = batch["cls_name"]
                with self._autocast_ctx():
                    cout = self._run_backbone(self.clip, imgs, call="encode_image")
                # Store CLIP patches in fp16 to halve CPU accumulation RAM;
                # build_winclip_reference will also fp16-half the final bank.
                for b in range(imgs.shape[0]):
                    c = cls_names[b]
                    p = (cout["patch"][b].float()
                                       .permute(1, 2, 0)
                                       .reshape(-1, cout["patch"].shape[1])
                                       .cpu()
                                       .half())
                    clip_patch_accum.setdefault(c, []).append(p)
                del imgs, cout
            self._empty_cache()
            for c, ps in clip_patch_accum.items():
                allp = torch.cat(ps, dim=0).float()  # upcast for normalize
                # build_winclip_reference returns fp16 bank
                self.clip_refs[c] = build_winclip_reference(self.clip, allp, n_select=4096)
                del allp
            clip_patch_accum.clear()
            self._empty_cache()

        # ---- Phase 3: coreset subsampling.
        print("[fit] Building coreset memory banks (batched FPS on GPU) ...",
              flush=True)
        coreset_dev = "cuda" if (self.cfg.coreset_on_gpu and self.device.type == "cuda") else "cpu"
        hn_k = int(getattr(self.cfg, "hard_negative_k", 0) or 0)
        try:
            self.patchcore.build(device=coreset_dev, bank_device="cpu",
                                 verbose=True, progress=True,
                                 hard_negative_k=hn_k)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                self._empty_cache()
                print(f"[fit] coreset on {coreset_dev} failed with OOM, retrying on CPU ...",
                      flush=True)
                try:
                    self.patchcore.build(device="cpu", bank_device="cpu",
                                         verbose=True, progress=True,
                                         hard_negative_k=hn_k)
                except RuntimeError as ee:
                    if "out of memory" in str(ee).lower():
                        print("[fit] CPU coreset OOM too -- retrying without hard negatives",
                              flush=True)
                        self.patchcore.build(device="cpu", bank_device="cpu",
                                             verbose=True, progress=True,
                                             hard_negative_k=0)
                    else:
                        raise
            else:
                raise
        self._empty_cache()

        # ---- Phase 4: calibration (per-sample TTA loop) ----
        print("[fit] Calibrating per-class score distributions on train "
              "(TTA on 20 samples/class -- takes 1-3 min) ...", flush=True)
        self._calibrate_on_train(train_root, tfm)
        self._empty_cache()
        print("[fit] Done. All per-class banks built and calibrated.", flush=True)

    @torch.inference_mode()
    def _calibrate_on_train(self, train_root: Path, tfm):
        g_cal = torch.Generator(device="cpu").manual_seed(self.cfg.seed + 1234)
        ds = CSIGSampleDataset(train_root, transform=tfm, classes=self.classes)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=self.cfg.num_workers, pin_memory=True,
                            persistent_workers=False)
        # IMPORTANT: we do NOT prefetch all banks here -- that was the cause of
        # the fit/calibrate OOM (~3.2 GB for 50 classes x 16k x 1024 fp32).
        # Instead we use LAZY per-class migration + immediate eviction, so only
        # ONE class' coreset bank (~64 MB) is resident on GPU at any time.
        banks_mode = "lazy"

        img_buf: Dict[str, List[float]] = {c: [] for c in self.classes}
        map_pix_buf: Dict[str, List[torch.Tensor]] = {c: [] for c in self.classes}

        prev_cls: Optional[str] = None

        for batch in tqdm(loader, desc="fit/calibrate"):
            views = self._move_input(batch["views"].squeeze(0))  # (V,3,H,W)
            cls = batch["cls_name"][0]

            # Lazy-migrate the current class' bank to GPU; evict the previous.
            if cls != prev_cls:
                if prev_cls is not None:
                    self._evict_bank_to_cpu(prev_cls)
                self._ensure_bank_on_device(cls, banks_mode)
                prev_cls = cls

            # Calibration runs on RAW (pre-foreground) maps.  The foreground
            # mask is applied AFTER percentile normalisation as post-process,
            # so calibration must NOT see it -- otherwise the per-class (lo,hi)
            # absorb the foreground suppression and the net effect is zero.
            img_raw, dino_map, _, _ = self._dino_tta_forward(
                views, cls, collect_fg=False,
            )
            img_buf[cls].append(float(img_raw.item()))
            # Subsample map pixels to keep CPU memory low.
            flat = dino_map.detach().reshape(-1).cpu()
            idx = torch.randperm(flat.shape[0], generator=g_cal)[:5000]
            map_pix_buf[cls].append(flat[idx])
            del views, dino_map, flat
        # Evict last class
        if prev_cls is not None:
            self._evict_bank_to_cpu(prev_cls)
        self._empty_cache()

        # Fit calibrator on image scores
        for c in self.classes:
            scores = torch.tensor(img_buf[c], dtype=torch.float32)
            self.calibrator.update(c, scores)
        self.calibrator.finalize()

        # Fit map normalisation percentiles per class
        for c in self.classes:
            all_pix = torch.cat(map_pix_buf[c])
            lo = float(torch.quantile(all_pix, self.cfg.dino_map_lo_q).item())
            hi = float(torch.quantile(all_pix, self.cfg.dino_map_hi_q).item())
            if hi - lo < 1e-6:
                hi = lo + 1e-3
            self.map_norm[c] = (lo, hi)

        print("[fit] Calibration done. Example stats:")
        for c in self.classes[:3]:
            p50, p95 = self.calibrator.stats[c]
            lo, hi = self.map_norm[c]
            print(f"  {c}: img p50={p50:.4f} p95={p95:.4f} | map lo={lo:.4f} hi={hi:.4f}")

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _predict_one_sample(self, views_448: torch.Tensor,
                            views_clip: Optional[torch.Tensor],
                            cls: str) -> Tuple[float, np.ndarray]:
        """
        views_448: (V, 3, 448, 448)
        Returns (image_score in [0,1], masks (V, 448, 448) float in [0,1]).
        """
        V = views_448.shape[0]

        # ---- DINOv2 + PatchCore (TTA, NO foreground here -- fg is applied
        # AFTER percentile norm so it is not undone by calibration) ----
        collect_fg = bool(self.cfg.use_foreground_mask)
        dino_img_raw, dino_map, fg_mask = self._dino_tta_forward(
            views_448, cls, collect_fg=collect_fg,
        )
        # Calibrate DINO image score to [0,1] using TRAIN-SET percentiles.
        dino_img_score = self.calibrator.apply(cls, dino_img_raw).clamp(0.0, 1.0)  # (1,)
        del dino_img_raw

        # Normalise DINO map to [0,1] using TRAIN-SET per-class percentiles
        lo, hi = self.map_norm.get(cls, (0.0, 1.0))
        dino_map_norm = ((dino_map - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
        del dino_map

        # ---- CLS-GUIDED MASK GATING ----
        # Use the (calibrated) image-level anomaly score as a per-sample
        # soft gate on the pixel map. Rationale:
        #   * If the image-level score says "this sample is normal" (<0.2),
        #     any locally-high pixels are overwhelmingly texture-edge FPs
        #     -- suppress them hard (multiply by ~0.15).
        #   * If the image-level score says "this sample is anomalous"
        #     (>0.7), trust the pixel map fully (multiply by ~1.0).
        # This costs literally ONE extra scalar multiplication per sample
        # and it is the single most effective FP killer for PatchCore.
        s = float(dino_img_score.item())
        # Soft gate: s-shaped mapping from calibrated image score to mask
        # multiplier. At s=0.2 -> 0.15x; at s=0.5 -> 0.5x; at s=0.8 -> 1.0x.
        import math
        gate = 0.15 + 0.85 / (1.0 + math.exp(-8.0 * (s - 0.45)))
        dino_map_norm = (dino_map_norm * gate).clamp(0.0, 1.0)

        # ---- WinCLIP branch ----
        # WinCLIP text-aligned patch maps are useful for IMAGE-LEVEL scoring
        # (they add semantic "defect-ness" signal) but are too noisy for pixel
        # localisation: relu(patch·text_delta) fires on many normal texture
        # edges, killing pixel AP.  So we only blend CLIP into the image
        # score and keep the mask pure DINO-PatchCore.
        clip_img_score = torch.zeros(1, device=views_448.device, dtype=torch.float32)
        clip_map = None
        if self.clip is not None and views_clip is not None:
            ref = self.clip_refs.get(cls, None)
            if ref is not None and ref.device != views_clip.device:
                ref = ref.to(views_clip.device)
                self.clip_refs[cls] = ref
            micro = max(1, int(self.cfg.clip_batch_size))
            all_patch, all_cls = [], []
            for i in range(0, views_clip.shape[0], micro):
                sub = views_clip[i:i+micro]
                with self._autocast_ctx():
                    cout_sub = self.clip.encode_image(sub)
                all_patch.append(cout_sub["patch"].float())
                all_cls.append(cout_sub["cls"].float())
                del sub, cout_sub
            c_patch = torch.cat(all_patch, dim=0)
            c_cls = torch.cat(all_cls, dim=0)
            del all_patch, all_cls
            wc = winclip_score(self.clip, c_patch, c_cls,
                               cls, reference_patches=ref, alpha=0.5)
            clip_img_score = wc["image_score"].mean().unsqueeze(0).clamp(0.0, 1.0)
            if self.cfg.clip_mask_ens:
                cm = wc["anomaly_map"].unsqueeze(1)
                cm = bilinear_upsample(cm, self.cfg.input_size)
                cm = gaussian_smooth2d(cm, kernel_size=7, sigma=3.0)
                clip_map = cm[:, 0].clamp(0.0, 1.0)
            else:
                clip_map = None
            del c_patch, c_cls, wc

        # ---- Ensemble (both maps are now in [0,1]) ----
        wd = self.cfg.ens_dino_weight
        wc_w = self.cfg.ens_clip_weight if self.clip is not None else 0.0
        wc_m = float(getattr(self.cfg, "ens_clip_mask_weight", 0.0) or 0.0)
        # Image score uses wc_w (typically 0.22).
        wt = wd + wc_w
        ens_score = (dino_img_score * wd + clip_img_score * wc_w) / wt
        ens_score = ens_score.clamp(0.0, 1.0)
        # Mask uses the lighter wc_m (default 0 = pure DINO, same as before).
        # When clip_mask_ens=True and wc_m>0 we blend in CLIP's anomaly map
        # with its own weight -- CLIP's map is already computed above, so
        # this is effectively FREE (no extra forward pass).
        if clip_map is not None and wc_m > 0:
            wt_m = wd + wc_m
            ens_map = (dino_map_norm * wd + clip_map * wc_m) / wt_m
        else:
            ens_map = dino_map_norm
        del dino_map_norm, clip_map, dino_img_score, clip_img_score

        # ---- Cross-view mask voting ----
        ens_map = multiview_mask_vote(ens_map.unsqueeze(0),
                                      beta=self.cfg.mv_vote_beta,
                                      mode=self.cfg.mv_mask_vote).squeeze(0)
        # Final safety clamp
        ens_map = ens_map.clamp(0.0, 1.0)

        # ---- Connected-component size suppression (per view) ----
        cc_min = int(getattr(self.cfg, "cc_min_area", 0) or 0)
        if cc_min > 0 and cc_min >= 4:
            cc_th = float(getattr(self.cfg, "cc_threshold", 0.30))
            cc_kill = float(getattr(self.cfg, "cc_kill_factor", 0.1))
            cleaned = []
            for v in range(ens_map.shape[0]):
                cleaned.append(
                    remove_small_components_torch(
                        ens_map[v], threshold=cc_th,
                        min_area=cc_min, min_area_kill=cc_kill
                    )
                )
            ens_map = torch.stack(cleaned, dim=0).clamp(0.0, 1.0)
            del cleaned

        # ---- Foreground suppression (applied AFTER percentile normalisation
        # and CC cleanup, so it cannot be undone by calibration). This is the
        # correct place: map is now in [0,1] and multiplying background by
        # fg_bg_floor here directly lowers background pixel values that
        # would otherwise become false positives at high thresholds. ----
        if fg_mask is not None and self.cfg.use_foreground_mask:
            m = fg_mask.to(ens_map.device, ens_map.dtype)
            if m.dim() == 4 and m.shape[1] == 1:
                m = m.squeeze(1)
            # Resize if needed (fg is built at input_size, ens_map is same).
            if m.shape[-2:] != ens_map.shape[-2:]:
                m = F.interpolate(m.unsqueeze(1), size=ens_map.shape[-2:],
                                  mode="bilinear", align_corners=False).squeeze(1)
            ens_map = apply_foreground(ens_map, m, lam=self.cfg.fg_bg_floor)
            hard_th = float(getattr(self.cfg, "fg_hard_threshold", 0.0) or 0.0)
            if hard_th > 0:
                # Hard-zero pixels below the saliency threshold (bg floor).
                ens_map = torch.where(m < hard_th, torch.zeros_like(ens_map), ens_map)
            ens_map = ens_map.clamp(0.0, 1.0)
            del m
        if fg_mask is not None:
            del fg_mask

        score = float(ens_score.item())
        out_np = ens_map.cpu().numpy()
        del ens_map, ens_score
        return score, out_np

    @torch.inference_mode()
    def predict_and_save(self, test_root: Optional[str] = None,
                         out_dir: Optional[str] = None):
        test_root = Path(test_root or self.cfg.test_root)
        out_dir = Path(out_dir or self.cfg.out_dir)
        mask_dir = out_dir / "predicted_masks"
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        test_classes = sorted([d.name for d in test_root.iterdir() if d.is_dir()])

        tfm = build_test_transform(self.cfg.input_size, tta="none")
        ds = CSIGSampleDataset(test_root, transform=tfm, classes=test_classes)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=self.cfg.num_workers, pin_memory=True,
                            persistent_workers=False)

        # Bank prefetch policy -- mirrors calibration:
        #   banks_on_gpu=True  -> try prefetching all (~3.2 GB), fallback to lazy
        #   banks_on_gpu=False -> lazy per-class migration + eviction (~64 MB peak)
        print(f"[predict] Running inference on {len(ds)} test samples ...")
        if self.cfg.banks_on_gpu:
            banks_mode = self._prefetch_banks("all")
        else:
            banks_mode = "lazy"
        if banks_mode == "all":
            print(f"[predict] Coreset banks prefetched to {self.device}")
        else:
            print("[predict] Lazy bank migration (banks_on_gpu=false) -- lowest VRAM")
        self._empty_cache()

        prev_cls: Optional[str] = None
        rows: List[Tuple[str, float]] = []
        for batch in tqdm(loader, desc="predict"):
            views = self._move_input(batch["views"].squeeze(0))
            cls = batch["cls_name"][0]
            sample_id = batch["sample_id"][0]
            group_folder = batch["group_folder"][0]

            # Lazy bank swap: evict previous class before loading the new one
            if banks_mode != "all" and cls != prev_cls:
                if prev_cls is not None:
                    self._evict_bank_to_cpu(prev_cls)
                self._ensure_bank_on_device(cls, banks_mode)
                prev_cls = cls

            views_clip = None
            if self.clip is not None:
                views_clip = _imagenet_to_clip(views)

            if cls in self.patchcore.banks:
                score, masks = self._predict_one_sample(views, views_clip, cls)
            else:
                score, masks = self._predict_zeroshot(views, views_clip, cls)

            rows.append((group_folder, float(score)))

            out_sample_dir = mask_dir / cls / sample_id
            out_sample_dir.mkdir(parents=True, exist_ok=True)
            for v in range(masks.shape[0]):
                save_mask_png(masks[v], out_sample_dir / f"{v}_mask.png",
                              target_size=self.cfg.input_size,
                              global_lo=0.0, global_hi=1.0,
                              gamma=self.cfg.mask_gamma)

            # Per-sample cleanup
            del views, views_clip, masks
            self._empty_cache()

        # Evict last class + any remaining banks
        if prev_cls is not None:
            self._evict_bank_to_cpu(prev_cls)
        self.patchcore.banks_to_cpu()
        self._empty_cache()

        csv_path = out_dir / "submission.csv"
        save_submission_csv(rows, csv_path)
        print(f"[predict] Saved {len(rows)} rows to {csv_path}")
        print(f"[predict] Masks saved to {mask_dir}")
        return csv_path, mask_dir

    @torch.inference_mode()
    def _predict_zeroshot(self, views_448: torch.Tensor,
                          views_clip: Optional[torch.Tensor],
                          cls: str) -> Tuple[float, np.ndarray]:
        """Fallback for unseen classes: pure WinCLIP (no memory bank)."""
        assert self.clip is not None, "Zero-shot requires CLIP backbone."
        # micro-batch CLIP forward to control activation memory
        all_patch, all_cls = [], []
        micro = max(1, int(self.cfg.clip_batch_size))
        for i in range(0, views_clip.shape[0], micro):
            sub = views_clip[i:i+micro]
            with self._autocast_ctx():
                cout_sub = self.clip.encode_image(sub)
            all_patch.append(cout_sub["patch"].float())
            all_cls.append(cout_sub["cls"].float())
            del sub, cout_sub
        c_patch = torch.cat(all_patch, 0)
        c_cls = torch.cat(all_cls, 0)
        del all_patch, all_cls
        wc = winclip_score(self.clip, c_patch, c_cls, cls,
                           reference_patches=None, alpha=1.0)
        cm = wc["anomaly_map"].unsqueeze(1)
        cm = bilinear_upsample(cm, self.cfg.input_size)
        cm = gaussian_smooth2d(cm, kernel_size=7, sigma=3.0)
        cmap = cm[:, 0].clamp(0.0, 1.0)
        cmap = multiview_mask_vote(cmap.unsqueeze(0),
                                    beta=self.cfg.mv_vote_beta,
                                    mode=self.cfg.mv_mask_vote).squeeze(0)
        img_score = wc["image_score"]  # (V,) in [0,1]
        cal_score = float(img_score.mean().clamp(0.0, 1.0).item())
        out_np = cmap.clamp(0, 1).cpu().numpy()
        del c_patch, c_cls, wc, cm, cmap, img_score
        return cal_score, out_np
