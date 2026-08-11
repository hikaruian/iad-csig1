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
    dinov2_layers: Tuple[int, ...] = (8, 9, 10, 11)
    input_size: int = 448
    use_clip: bool = True
    clip_model: str = "ViT-L-14-336"
    clip_pretrained: str = "openai"

    # ---- patchcore ----
    coreset_ratio: float = 0.02
    coreset_max: int = 16384
    neighbourhood_size: int = 3
    n_neighbours: int = 5
    smooth_kernel: int = 9
    smooth_sigma: float = 4.0
    cls_bank_weight: float = 0.25

    # ---- multi-view ----
    mv_aggregate: str = "robust_mean"
    mv_vote_beta: float = 1.5
    use_mv_attn: bool = True

    # ---- TTA ----
    use_tta: bool = True
    tta_weight_orig: float = 1.0
    tta_weight_flip: float = 0.5

    # ---- ensemble ----
    ens_dino_weight: float = 0.75
    ens_clip_weight: float = 0.25

    # ---- normalisation percentiles for DINO anomaly maps (from train set) ----
    dino_map_lo_q: float = 0.05   # lower percentile = "background level" on normal samples
    dino_map_hi_q: float = 0.99   # upper percentile = "clearly anomalous" level
    dino_img_lo_q: float = 0.05
    dino_img_hi_q: float = 0.95

    # ---- memory / precision ----
    batch_size: int = 4
    clip_batch_size: int = 2      # separate (smaller) batch size for CLIP forward
    num_workers: int = 4
    use_amp: bool = True          # use fp16 autocast for backbones (halves activation memory)
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
        )
        # Calibrator for image-level scores (per-class)
        self.calibrator = PerClassPercentileCalibrator()
        # Per-class (lo, hi) scalars for normalising DINO anomaly maps from
        # raw distance scale to roughly [0, 1], using statistics collected
        # on the TRAIN set. Using train-set percentiles avoids the
        # "per-sample normalisation always flags something" pitfall.
        self.map_norm: Dict[str, Tuple[float, float]] = {}

        self.clip_refs: Dict[str, torch.Tensor] = {}
        # NOTE: mv_attn is intentionally not used in the default scoring
        # formula (we use mean-pooled CLS instead, which is more stable),
        # but kept as an attribute in case a custom pipeline wants it.
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

    def _dino_forward(self, x: torch.Tensor, cls: str,
                      return_map: bool = True) -> Dict[str, torch.Tensor]:
        """Run DINOv2 + PatchCore and return per-view image scores and
        (optionally) per-view high-res anomaly maps. Shared by training
        calibration and inference so the two paths are NUMERICALLY IDENTICAL
        (modulo TTA at inference time)."""
        out = self._run_backbone(self.dino, x)
        # Backbone may return fp16 when AMP is enabled; upcast for kNN
        # math and downstream stats to keep distances precise.
        patch = out["patch"].float()
        cls_tok = out["cls"].float()
        res = self.patchcore.predict(cls, patch, cls_tok, return_map=return_map)
        return res

    def _dino_tta_forward(self, views: torch.Tensor, cls: str
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run DINO branch with flip TTA, returning (dino_img_raw, dino_map)
        where dino_img_raw is (1,) sample-level raw score (pre-calibration)
        and dino_map is (V, H, W) fused per-view high-res map."""
        augs = tta_flips(views) if self.cfg.use_tta else [("orig", views)]
        img_acc = []
        map_acc = []
        w_list = []
        for name, x in augs:
            res = self._dino_forward(x, cls, return_map=True)
            amap = unflip_map(res["anomaly_map"].unsqueeze(1), name).squeeze(1)
            map_acc.append(amap)
            img_acc.append(res["image_score"])
            w_list.append(self.cfg.tta_weight_orig if name == "orig" else self.cfg.tta_weight_flip)
        w = torch.tensor(w_list, device=views.device, dtype=views.dtype)
        wsum = w.sum()
        w_map = w.view(-1, 1, 1, 1)
        stacked_maps = torch.stack(map_acc, 0)
        dino_map = (stacked_maps * w_map).sum(0) / wsum     # (V, H, W)
        w_img = w.view(-1, 1)
        stacked_imgs = torch.stack(img_acc, 0)
        pv = (stacked_imgs * w_img).sum(0) / wsum           # (V,)
        img_raw = aggregate_image_scores(
            pv.unsqueeze(0), strategy=self.cfg.mv_aggregate
        )  # (1,)
        del stacked_maps, stacked_imgs, map_acc, img_acc
        return img_raw, dino_map

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit(self, train_root: Optional[str] = None):
        train_root = Path(train_root or self.cfg.train_root)
        self.classes = discover_classes(train_root)
        print(f"[fit] Found {len(self.classes)} classes: {self.classes}")

        tfm = build_train_transform(self.cfg.input_size)
        ds = CSIGImageDataset(train_root, transform=tfm, classes=self.classes)
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False,
                            num_workers=self.cfg.num_workers, pin_memory=True)
        print(f"[fit] Extracting DINOv2 features from {len(ds)} training images ...")

        clip_patch_accum: Dict[str, List[torch.Tensor]] = {}

        # ---- Phase 1: extract DINO patch/CLS features ----
        for batch in tqdm(loader, desc="fit/dino"):
            imgs = batch["image"].to(self.device, non_blocking=True)
            cls_names = batch["cls_name"]
            with self._autocast_ctx():
                out = self._run_backbone(self.dino, imgs)
            # Cast back to fp32 for storage & kNN; move to CPU immediately.
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
            clip_bs = max(1, min(self.cfg.clip_batch_size, self.cfg.batch_size))
            ld_c = DataLoader(ds_c, batch_size=clip_bs, shuffle=False,
                              num_workers=self.cfg.num_workers, pin_memory=True)
            print("[fit] Building WinCLIP reference banks ...")
            for batch in tqdm(ld_c, desc="fit/clip"):
                imgs = batch["image"].to(self.device, non_blocking=True)
                cls_names = batch["cls_name"]
                with self._autocast_ctx():
                    cout = self._run_backbone(self.clip, imgs, call="encode_image")
                for b in range(imgs.shape[0]):
                    c = cls_names[b]
                    p = cout["patch"][b].float().permute(1, 2, 0).reshape(-1, cout["patch"].shape[1])
                    clip_patch_accum.setdefault(c, []).append(p.cpu())
                del imgs, cout
            self._empty_cache()
            for c, ps in clip_patch_accum.items():
                allp = torch.cat(ps, dim=0)
                # build_winclip_reference does random sampling (no GPU needed)
                self.clip_refs[c] = build_winclip_reference(self.clip, allp, n_select=2048)
            del clip_patch_accum
            self._empty_cache()

        # ---- Phase 3: coreset subsampling. Run on CPU by default
        # (greedy_coreset supports CUDA but on 16GB cards the intermediate
        # [N, 256] RP matrix + distance tensors can push us over).
        print("[fit] Building coreset memory banks ...")
        coreset_dev = "cpu"
        try:
            self.patchcore.build(device=coreset_dev, bank_device="cpu")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                self._empty_cache()
                print("[fit] coreset on CUDA failed, retrying on CPU ...")
                self.patchcore.build(device="cpu", bank_device="cpu")
            else:
                raise
        self._empty_cache()

        # ---- Phase 4: calibration (per-sample TTA loop) ----
        print("[fit] Calibrating per-class score distributions on train ...")
        self._calibrate_on_train(train_root, tfm)
        self._empty_cache()

    @torch.no_grad()
    def _calibrate_on_train(self, train_root: Path, tfm):
        g_cal = torch.Generator(device="cpu").manual_seed(self.cfg.seed + 1234)
        ds = CSIGSampleDataset(train_root, transform=tfm, classes=self.classes)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=self.cfg.num_workers, pin_memory=True)
        # Collect per-class image scores AND per-class map pixel values so
        # we can later normalise anomaly maps using TRAIN-SET statistics.
        # IMPORTANT: we run the SAME scoring formula here as at inference
        # (including TTA) so the percentiles reflect the actual test-time
        # score distribution.
        img_buf: Dict[str, List[float]] = {c: [] for c in self.classes}
        map_pix_buf: Dict[str, List[torch.Tensor]] = {c: [] for c in self.classes}

        for batch in tqdm(loader, desc="fit/calibrate"):
            views = batch["views"].squeeze(0).to(self.device)  # (V,3,H,W)
            cls = batch["cls_name"][0]

            # Run the identical DINO (with TTA) pipeline used at inference
            img_raw, dino_map = self._dino_tta_forward(views, cls)
            img_buf[cls].append(float(img_raw.item()))
            # Subsample map pixels (5 views × 448×448 ≈ 1M px; keep 20k per sample).
            # Use a deterministic CPU generator so calibration is reproducible
            # regardless of whether the maps land on GPU.
            flat = dino_map.detach().reshape(-1).cpu()
            idx = torch.randperm(flat.shape[0], generator=g_cal)[:20000]
            map_pix_buf[cls].append(flat[idx])

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
    @torch.no_grad()
    def _predict_one_sample(self, views_448: torch.Tensor,
                            views_clip: Optional[torch.Tensor],
                            cls: str) -> Tuple[float, np.ndarray]:
        """
        views_448: (V, 3, 448, 448)
        Returns (image_score in [0,1], masks (V, 448, 448) float in [0,1]).
        """
        V = views_448.shape[0]

        # ---- DINOv2 + PatchCore (with TTA) ----
        dino_img_raw, dino_map = self._dino_tta_forward(views_448, cls)
        # Calibrate DINO image score to [0,1] using TRAIN-SET percentiles.
        dino_img_score = self.calibrator.apply(cls, dino_img_raw).clamp(0.0, 1.0)  # (1,)

        # Normalise DINO map to [0,1] using TRAIN-SET per-class percentiles
        # (NOT per-sample). This is critical: per-sample normalisation
        # forces every image to span [0,1] and kills the ability to
        # distinguish all-normal images from defective ones at the mask
        # level. Using train-set statistics preserves relative intensity
        # across samples.
        lo, hi = self.map_norm.get(cls, (0.0, 1.0))
        dino_map_norm = ((dino_map - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)

        # ---- WinCLIP branch (no TTA for speed; optional) ----
        clip_img_score = torch.zeros_like(dino_img_score)
        clip_map = torch.zeros_like(dino_map_norm)
        if self.clip is not None and views_clip is not None:
            ref = self.clip_refs.get(cls, None)
            if ref is not None and ref.device != views_clip.device:
                ref = ref.to(views_clip.device)
                self.clip_refs[cls] = ref
            # Micro-batch CLIP forward to bound activation memory
            # (ViT-L/14 at 448px uses ~1.3 GB/view in fp16).
            all_patch = []
            all_cls = []
            micro = 2
            for i in range(0, views_clip.shape[0], micro):
                sub = views_clip[i:i+micro]
                with self._autocast_ctx():
                    cout_sub = self.clip.encode_image(sub)
                all_patch.append(cout_sub["patch"].float())
                all_cls.append(cout_sub["cls"].float())
                del sub, cout_sub
            cout = {
                "patch": torch.cat(all_patch, dim=0),
                "cls": torch.cat(all_cls, dim=0),
            }
            wc = winclip_score(self.clip, cout["patch"], cout["cls"],
                               cls, reference_patches=ref, alpha=0.5)
            cm = wc["anomaly_map"].unsqueeze(1)            # (V,1,Hp_c,Wp_c)
            cm = bilinear_upsample(cm, self.cfg.input_size)
            cm = gaussian_smooth2d(cm, kernel_size=7, sigma=3.0)
            clip_map = cm[:, 0].clamp(0.0, 1.0)            # (V,448,448)
            clip_img_score = wc["image_score"].mean().unsqueeze(0).clamp(0.0, 1.0)
            del cout, all_patch, all_cls, wc, cm

        # ---- Ensemble (both maps are now in [0,1]) ----
        wd = self.cfg.ens_dino_weight
        wc = self.cfg.ens_clip_weight if self.clip is not None else 0.0
        wt = wd + wc
        ens_map = (dino_map_norm * wd + clip_map * wc) / wt        # (V,448,448)
        ens_score = (dino_img_score * wd + clip_img_score * wc) / wt
        ens_score = ens_score.clamp(0.0, 1.0)

        # ---- Cross-view mask voting ----
        ens_map = multiview_mask_vote(ens_map.unsqueeze(0),
                                      beta=self.cfg.mv_vote_beta).squeeze(0)
        # Final safety clamp
        ens_map = ens_map.clamp(0.0, 1.0)
        return float(ens_score.item()), ens_map.cpu().numpy()

    @torch.no_grad()
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
                            num_workers=self.cfg.num_workers, pin_memory=True)

        # Prefetch memory banks to GPU once before the loop (faster).
        # On GPUs with <16GB memory, set bank_device="cpu" via the build
        # call and skip prefetch -- banks will be migrated lazily per
        # call and evicted below.
        print(f"[predict] Running inference on {len(ds)} test samples ...")
        try:
            # Try prefetching; if it OOMs, fall back to lazy migration.
            self.patchcore.banks_to_device(self.device)
            banks_prefetched = True
            print(f"[predict] Coreset banks prefetched to {self.device}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            banks_prefetched = False
            print("[predict] Prefetch OOM -- falling back to lazy bank migration")
        self._empty_cache()

        rows: List[Tuple[str, float]] = []
        for batch in tqdm(loader, desc="predict"):
            views = batch["views"].squeeze(0).to(self.device)
            cls = batch["cls_name"][0]
            sample_id = batch["sample_id"][0]
            group_folder = batch["group_folder"][0]

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
                              global_lo=0.0, global_hi=1.0)

            # Per-sample cleanup
            del views, views_clip
            if not banks_prefetched:
                # Evict this class' bank back to CPU after use so we only
                # ever have one class coreset resident on GPU at a time.
                bk = self.patchcore.banks.get(cls, None)
                if bk is not None and bk.features.device.type == "cuda":
                    bk.features = bk.features.to("cpu")
                    bk.cls_prototype = bk.cls_prototype.to("cpu")
            self._empty_cache()

        # Evict banks back to CPU at the end.
        self.patchcore.banks_to_cpu()
        self._empty_cache()

        csv_path = out_dir / "submission.csv"
        save_submission_csv(rows, csv_path)
        print(f"[predict] Saved {len(rows)} rows to {csv_path}")
        print(f"[predict] Masks saved to {mask_dir}")
        return csv_path, mask_dir

    def _predict_zeroshot(self, views_448: torch.Tensor,
                          views_clip: Optional[torch.Tensor],
                          cls: str) -> Tuple[float, np.ndarray]:
        """Fallback for unseen classes: pure WinCLIP (no memory bank)."""
        assert self.clip is not None, "Zero-shot requires CLIP backbone."
        # micro-batch CLIP forward to control activation memory
        all_patch, all_cls = [], []
        micro = 2
        for i in range(0, views_clip.shape[0], micro):
            sub = views_clip[i:i+micro]
            with self._autocast_ctx():
                cout_sub = self.clip.encode_image(sub)
            all_patch.append(cout_sub["patch"].float())
            all_cls.append(cout_sub["cls"].float())
            del sub, cout_sub
        cout = {
            "patch": torch.cat(all_patch, 0),
            "cls": torch.cat(all_cls, 0),
        }
        wc = winclip_score(self.clip, cout["patch"], cout["cls"], cls,
                           reference_patches=None, alpha=1.0)
        cm = wc["anomaly_map"].unsqueeze(1)
        cm = bilinear_upsample(cm, self.cfg.input_size)
        cm = gaussian_smooth2d(cm, kernel_size=7, sigma=3.0)
        cmap = cm[:, 0].clamp(0.0, 1.0)
        cmap = multiview_mask_vote(cmap.unsqueeze(0)).squeeze(0)
        img_score = wc["image_score"]  # (V,) in [0,1]
        cal_score = float(img_score.mean().clamp(0.0, 1.0).item())
        del cout, wc, cm, all_patch, all_cls
        return cal_score, cmap.clamp(0, 1).cpu().numpy()
