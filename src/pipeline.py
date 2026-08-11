"""
End-to-end training & inference pipeline.

Usage (high level):
    from src.pipeline import CSIGAnomalyPipeline
    pipe = CSIGAnomalyPipeline(cfg)
    pipe.fit("path/to/CSIG/Train")
    pipe.predict_and_save("path/to/CSIG/Test_A",
                          out_root="submission")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .backbones import CLIPFeatureExtractor, DINOv2FeatureExtractor
from .dataset import (CSIGImageDataset, CSIGSampleDataset, build_test_transform,
                      build_train_transform, discover_classes, get_patch_hw)
from .multiview import (CrossViewAttention, aggregate_image_scores,
                        multiview_mask_vote)
from .patchcore import MultiClassPatchCore, bilinear_upsample, gaussian_smooth2d
from .utils import (PerClassPercentileCalibrator, AverageMeter, save_mask_png,
                    save_submission_csv, tta_flips, unflip_map)
from .zeroshot import build_winclip_reference, winclip_score


@dataclass
class PipelineConfig:
    # ---- paths ----
    train_root: str = "CSIG/Train"
    test_root: str = "CSIG/Test_A"
    out_dir: str = "submission"

    # ---- backbone ----
    dinov2_model: str = "vitl14"            # vitb14 | vitl14 | vitg14
    dinov2_layers: Tuple[int, ...] = (8, 9, 10, 11)
    input_size: int = 448                   # must be divisible by 14 (patch)
    use_clip: bool = True                   # winclip zeroshot branch
    clip_model: str = "ViT-L-14-336"        # open_clip model name
    clip_pretrained: str = "openai"

    # ---- patchcore ----
    coreset_ratio: float = 0.02
    coreset_max: int = 16384
    neighbourhood_size: int = 3
    n_neighbours: int = 5
    smooth_kernel: int = 9
    smooth_sigma: float = 4.0
    cls_bank_weight: float = 0.25           # CLS-prototype score blend

    # ---- multi-view ----
    mv_aggregate: str = "robust_mean"       # mean | max | robust_mean
    mv_vote_beta: float = 1.5
    use_mv_attn: bool = True                # lightweight cross-view attn for CLS

    # ---- TTA ----
    use_tta: bool = True                    # flip TTA (4x)
    tta_weight_orig: float = 1.0            # original view gets more weight
    tta_weight_flip: float = 0.5

    # ---- ensemble ----
    ens_dino_weight: float = 0.75           # weight of DINOv2 patchcore branch
    ens_clip_weight: float = 0.25           # weight of WinCLIP branch

    # ---- misc ----
    batch_size: int = 1                     # per-view batch (i.e. 4 samples)
    num_workers: int = 2
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

        # ---- backbones ----
        print(f"[pipeline] Loading DINOv2 {self.cfg.dinov2_model} ...")
        self.dino = DINOv2FeatureExtractor(
            model_name=self.cfg.dinov2_model,
            layers=self.cfg.dinov2_layers,
            pretrained=True,
        ).to(self.device).eval()

        self.clip = None
        if self.cfg.use_clip:
            print(f"[pipeline] Loading CLIP {self.cfg.clip_model} ...")
            self.clip = CLIPFeatureExtractor(
                model_name=self.cfg.clip_model,
                pretrained=self.cfg.clip_pretrained,
            ).to(self.device).eval()

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
        self.calibrator = PerClassPercentileCalibrator()

        # WinCLIP reference banks per class (CLIP patch tokens)
        self.clip_refs: Dict[str, torch.Tensor] = {}

        # Cross-view attention pool (init after we know DINOv2 dim)
        self.mv_attn = None
        if self.cfg.use_mv_attn:
            self.mv_attn = CrossViewAttention(self.dino.embed_dim).to(self.device).eval()

    # ------------------------------------------------------------------
    # TRAINING: build per-class memory banks
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit(self, train_root: Optional[str] = None):
        train_root = Path(train_root or self.cfg.train_root)
        self.classes = discover_classes(train_root)
        print(f"[fit] Found {len(self.classes)} classes: {self.classes}")
        Hp, Wp = get_patch_hw(self.cfg.input_size, self.dino.patch_size)

        tfm = build_train_transform(self.cfg.input_size)
        ds = CSIGImageDataset(train_root, transform=tfm, classes=self.classes)
        loader = DataLoader(ds, batch_size=self.cfg.batch_size,
                            shuffle=False, num_workers=self.cfg.num_workers,
                            pin_memory=True, drop_last=False)
        print(f"[fit] Extracting DINOv2 features from {len(ds)} training images ...")

        clip_patch_accum: Dict[str, List[torch.Tensor]] = {}

        for batch in tqdm(loader, desc="fit/dino"):
            imgs = batch["image"].to(self.device, non_blocking=True)
            cls_names = batch["cls_name"]
            out = self.dino(imgs)
            # Group features by class (they may be mixed in a batch)
            patch = out["patch"].cpu()
            cls_tok = out["cls"].cpu()
            for b in range(imgs.shape[0]):
                c = cls_names[b]
                self.patchcore.add(c, patch[b:b+1], cls_tok[b:b+1])

            # WinCLIP reference accumulation (if enabled)
            if self.clip is not None:
                # Need to re-run at CLIP input size (336) for alignment
                # We use a separate transform; for simplicity here we resize
                # to clip's expected size by building a dedicated dataloader
                # below.
                pass

        # Build WinCLIP references in a separate pass at 336px
        if self.clip is not None:
            clip_size = 336
            clip_tfm = build_train_transform(clip_size)
            ds_c = CSIGImageDataset(train_root, transform=clip_tfm, classes=self.classes)
            ld_c = DataLoader(ds_c, batch_size=self.cfg.batch_size, shuffle=False,
                              num_workers=self.cfg.num_workers, pin_memory=True)
            print(f"[fit] Building WinCLIP reference banks ...")
            for batch in tqdm(ld_c, desc="fit/clip"):
                imgs = batch["image"].to(self.device, non_blocking=True)
                cls_names = batch["cls_name"]
                cout = self.clip.encode_image(imgs)
                for b in range(imgs.shape[0]):
                    c = cls_names[b]
                    p = cout["patch"][b].permute(1, 2, 0).reshape(-1, cout["patch"].shape[1])
                    clip_patch_accum.setdefault(c, []).append(p.cpu())
            for c, ps in clip_patch_accum.items():
                allp = torch.cat(ps, dim=0)
                self.clip_refs[c] = build_winclip_reference(self.clip, allp,
                                                            n_select=2048)

        # Coreset subsampling
        print("[fit] Building coreset memory banks ...")
        self.patchcore.build()

        # Calibrate image-score distribution using training set (all normal)
        print("[fit] Calibrating per-class score distributions on train ...")
        self._calibrate_on_train(train_root)

    def _calibrate_on_train(self, train_root: Path):
        tfm = build_train_transform(self.cfg.input_size)
        ds = CSIGSampleDataset(train_root, transform=tfm, classes=self.classes)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=self.cfg.num_workers, pin_memory=True)
        for batch in tqdm(loader, desc="fit/calibrate"):
            # Per-sample, 5 views
            views = batch["views"].squeeze(0).to(self.device)  # (5,3,H,W)
            cls = batch["cls_name"][0]
            out = self.dino(views)
            res = self.patchcore.predict(cls, out["patch"], out["cls"], return_map=False)
            # per-view scores
            pv_scores = res["image_score"].unsqueeze(0)  # (1,V)
            agg = aggregate_image_scores(pv_scores, strategy=self.cfg.mv_aggregate)
            self.calibrator.update(cls, agg)

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _predict_one_sample(self, views_448: torch.Tensor,
                            views_clip: Optional[torch.Tensor],
                            cls: str) -> Tuple[float, np.ndarray]:
        """
        views_448: (V, 3, 448, 448)
        Returns (image_score, final_mask_array (V, 448, 448)).
        """
        V = views_448.shape[0]

        # ---- DINOv2 branch (with TTA flips) ----
        augs = tta_flips(views_448) if self.cfg.use_tta else [("orig", views_448)]
        dino_image_scores = []
        dino_maps = []
        w_dino = []
        for name, x in augs:
            out = self.dino(x)
            res = self.patchcore.predict(cls, out["patch"], out["cls"], return_map=True)
            amap = unflip_map(res["anomaly_map"].unsqueeze(1), name).squeeze(1)
            dino_maps.append(amap)
            dino_image_scores.append(res["image_score"])
            w_dino.append(self.cfg.tta_weight_orig if name == "orig"
                          else self.cfg.tta_weight_flip)
        wsum = sum(w_dino)
        dino_maps = torch.stack(dino_maps, dim=0)  # (A, V, H, W)
        w_tensor = torch.tensor(w_dino, device=dino_maps.device,
                                dtype=dino_maps.dtype).view(-1, 1, 1, 1)
        dino_map = (dino_maps * w_tensor).sum(0) / wsum  # (V,H,W)
        dino_scores = torch.stack(dino_image_scores, dim=0)
        w_s = torch.tensor(w_dino, device=dino_scores.device,
                           dtype=dino_scores.dtype).view(-1, 1)
        dino_pv_scores = (dino_scores * w_s).sum(0) / wsum  # (V,)

        # Cross-view attention on CLS (for image score only)
        if self.mv_attn is not None:
            # Re-run without aug to get clean CLS tokens
            out_clean = self.dino(views_448)
            cls_tokens = out_clean["cls"].unsqueeze(0)     # (1,V,D)
            proto = self.patchcore.banks[cls].cls_prototype.to(self.device)
            agg_cls = self.mv_attn(cls_tokens, proto)      # (1,D)
            cls_dist = 1.0 - (agg_cls @ proto.unsqueeze(1)).squeeze(1)  # (1,)
            dino_image_max = dino_map.flatten(1).max(dim=1).values.mean().unsqueeze(0)
            dino_img_score = 0.7 * dino_image_max + 0.3 * cls_dist
        else:
            dino_img_score = aggregate_image_scores(
                dino_pv_scores.unsqueeze(0), strategy=self.cfg.mv_aggregate
            )

        # ---- WinCLIP branch (no TTA to keep it fast; optional) ----
        clip_img_score = torch.zeros_like(dino_img_score)
        clip_map = torch.zeros_like(dino_map)
        if self.clip is not None and views_clip is not None:
            cout = self.clip.encode_image(views_clip)
            ref = self.clip_refs.get(cls, None)
            wc = winclip_score(self.clip, cout["patch"], cout["cls"],
                               cls, reference_patches=ref, alpha=0.5)
            # up-sample map to 448
            cm = wc["anomaly_map"].unsqueeze(1)       # (V,1,Hp_c,Wp_c)
            cm = bilinear_upsample(cm, self.cfg.input_size)
            cm = gaussian_smooth2d(cm, kernel_size=7, sigma=3.0)
            clip_map = cm[:, 0]                       # (V,448,448)
            clip_img_score = wc["image_score"].max().unsqueeze(0)

        # ---- Ensemble ----
        wd = self.cfg.ens_dino_weight
        wc = self.cfg.ens_clip_weight if self.clip is not None else 0.0
        wt = wd + wc
        ens_map = (dino_map * wd + clip_map * wc) / wt  # (V,448,448)
        ens_score = (dino_img_score * wd + clip_img_score * wc) / wt

        # ---- Cross-view mask voting ----
        ens_map_ref = multiview_mask_vote(ens_map.unsqueeze(0),
                                          beta=self.cfg.mv_vote_beta).squeeze(0)

        # Calibrate image score to [0,1] using train-set percentiles
        cal_score = self.calibrator.apply(cls, ens_score).item()
        return cal_score, ens_map_ref.cpu().numpy()

    @torch.no_grad()
    def predict_and_save(self, test_root: Optional[str] = None,
                         out_dir: Optional[str] = None):
        test_root = Path(test_root or self.cfg.test_root)
        out_dir = Path(out_dir or self.cfg.out_dir)
        mask_dir = out_dir / "predicted_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        # Auto-discover classes present in test root (works for B-board too)
        test_classes = sorted([d.name for d in test_root.iterdir() if d.is_dir()])
        # For classes not in the trained bank (zero-shot), fall back to WinCLIP
        # only; handled below.

        tfm = build_test_transform(self.cfg.input_size, tta="none")
        clip_tfm = build_test_transform(336, tta="none") if self.clip else None
        ds = CSIGSampleDataset(test_root, transform=tfm, classes=test_classes)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=self.cfg.num_workers, pin_memory=True)

        rows: List[Tuple[str, float]] = []
        print(f"[predict] Running inference on {len(ds)} test samples ...")

        for batch in tqdm(loader, desc="predict"):
            views = batch["views"].squeeze(0).to(self.device)    # (V,3,448,448)
            cls = batch["cls_name"][0]
            sample_id = batch["sample_id"][0]
            group_folder = batch["group_folder"][0]

            # Prepare CLIP-views at 336 if needed
            views_clip = None
            if self.clip is not None:
                # Re-resize by using the clip_tfm on raw images (we already
                # have 448 tensors; simpler path: re-read but the dataset
                # ctor above used tfm, so we run an ad-hoc resize here)
                views_clip = F.interpolate(views, size=(336, 336),
                                           mode="bilinear", align_corners=False)

            if cls in self.patchcore.banks:
                score, masks = self._predict_one_sample(views, views_clip, cls)
            else:
                # Zero-shot class: use WinCLIP only
                score, masks = self._predict_zeroshot(views, views_clip, cls)

            rows.append((group_folder, float(score)))

            # Save per-view masks
            out_sample_dir = mask_dir / cls / sample_id
            out_sample_dir.mkdir(parents=True, exist_ok=True)
            for v in range(masks.shape[0]):
                save_mask_png(masks[v], out_sample_dir / f"{v}_mask.png",
                              target_size=self.cfg.input_size)

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
        cout = self.clip.encode_image(views_clip)
        wc = winclip_score(self.clip, cout["patch"], cout["cls"], cls,
                           reference_patches=None, alpha=1.0)
        cm = wc["anomaly_map"].unsqueeze(1)
        cm = bilinear_upsample(cm, self.cfg.input_size)
        cm = gaussian_smooth2d(cm, kernel_size=7, sigma=3.0)
        cmap = cm[:, 0]
        cmap = multiview_mask_vote(cmap.unsqueeze(0)).squeeze(0)
        img_score = wc["image_score"].max().unsqueeze(0)
        cal = torch.sigmoid(5.0 * (img_score - img_score.median())).item()
        return cal, cmap.cpu().numpy()
