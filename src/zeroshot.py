"""
WinCLIP-style zero-shot anomaly detection branch.

This is used as a secondary / ensemble branch when classes are truly novel
(zero-shot cold start). For the 50 *seen* classes it still provides a
useful complementary signal because CLIP's text-image alignment brings in
semantic knowledge that pure feature-memory banks do not have.

Implementation outline
----------------------
1. Build compositional text prompts for normal vs. defective states of a
   class. Following WinCLIP (CVPR 2023) and AdaCLIP we use prompt ensembling
   with state + defect-type templates to improve text embedding quality.
2. For each test image, compare CLIP patch tokens against (a) the class
   text embeddings and (b) a small normal reference bank (from training
   images) using cosine similarity to get a pixel-level anomaly heatmap.
3. Average over prompts / templates.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


# Compositional prompt templates (following WinCLIP + AdaCLIP)
_NORMAL_TEMPLATES = [
    "a crisp photo of a perfect {}.",
    "a sharp image of a flawless {}.",
    "a clear photo of a brand-new {}.",
    "a high-resolution photo of a defect-free {}.",
    "a close-up photo of a normal {}.",
    "a photo of a well-manufactured {}.",
]
_DEFECT_TEMPLATES = [
    "a photo of a {} with scratches.",
    "a photo of a {} with cracks.",
    "a photo of a {} with dents.",
    "a photo of a {} with stains.",
    "a photo of a damaged {}.",
    "a photo of a broken {}.",
    "a photo of a defective {}.",
]


def build_text_prompts(class_name: str) -> Tuple[List[str], List[str]]:
    norm = [t.format(class_name) for t in _NORMAL_TEMPLATES]
    defc = [t.format(class_name) for t in _DEFECT_TEMPLATES]
    return norm, defc


def _normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, dim=dim)


@torch.no_grad()
def winclip_score(clip_model, patch_feat: torch.Tensor,
                  cls_feat: torch.Tensor,
                  class_name: str,
                  reference_patches: torch.Tensor | None = None,
                  alpha: float = 0.5) -> Dict[str, torch.Tensor]:
    """
    Compute WinCLIP-style anomaly map / image score for ONE class.

    clip_model      : CLIPFeatureExtractor instance
    patch_feat      : (B, D, Hp, Wp) L2-normalised CLIP patch tokens
    cls_feat        : (B, D)         L2-normalised CLS tokens
    class_name      : e.g. "battery"
    reference_patches: (M, D) L2-normalised patch tokens from NORMAL training
                      images of the same class (optional but strongly
                      recommended – WinCLIP's "reference bank" gives much
                      better localisation).
    alpha           : weight blending text vs. reference-bank scores.

    Returns
    -------
    dict:
        image_score (B,)
        anomaly_map (B, Hp, Wp)
    """
    device = patch_feat.device
    norm_prompts, defc_prompts = build_text_prompts(class_name)
    norm_text = clip_model.encode_text(norm_prompts).mean(0)  # (D,)
    defc_text = clip_model.encode_text(defc_prompts).mean(0)  # (D,)
    norm_text = _normalize(norm_text, dim=0)
    defc_text = _normalize(defc_text, dim=0)
    delta = defc_text - norm_text  # defect direction in CLIP space

    B, D, Hp, Wp = patch_feat.shape
    flat = patch_feat.permute(0, 2, 3, 1).reshape(-1, D)  # (B*N, D)

    # Text-direction alignment (cosine with defect direction)
    sim_text = (flat @ delta.unsqueeze(1)).squeeze(1)  # (B*N,)
    map_text = sim_text.reshape(B, Hp, Wp)

    # Reference-bank alignment: distance to nearest normal patch
    if reference_patches is not None and reference_patches.numel() > 0:
        ref = reference_patches.to(device)
        sim_ref = flat @ ref.t()            # (B*N, M) cosine
        d_ref = 1.0 - sim_ref.max(1).values # (B*N,)
        map_ref = d_ref.reshape(B, Hp, Wp)
        amap = alpha * map_text + (1 - alpha) * map_ref
    else:
        amap = map_text

    # Image score = max over map
    img_score = amap.reshape(B, -1).max(dim=1).values

    # Normalise to [0,1] globally (per-batch percentile-clip is handled later)
    return {"image_score": img_score, "anomaly_map": amap}


def build_winclip_reference(clip_model, patch_feat_all: torch.Tensor,
                            n_select: int = 2048) -> torch.Tensor:
    """
    Build a small normal-reference patch bank for one class via random
    selection (deterministic seed). Used by the WinCLIP branch.

    patch_feat_all: (N_total_patches, D) L2-normalised.
    """
    n = patch_feat_all.shape[0]
    n_select = min(n_select, n)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(n, generator=g)[:n_select]
    return _normalize(patch_feat_all[idx], dim=-1).contiguous()
