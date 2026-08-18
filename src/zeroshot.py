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
    # Convert snake_case / directory names to natural English for CLIP.
    pretty = class_name.replace("_", " ").replace("-", " ").strip()
    norm = [t.format(pretty) for t in _NORMAL_TEMPLATES]
    defc = [t.format(pretty) for t in _DEFECT_TEMPLATES]
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
    # Canonical compute dtype: we do all WinCLIP math in fp32 for numerical
    # stability and to avoid any fp16/fp32 mismatch from AMP/half models.
    work_dtype = torch.float32
    patch_feat = patch_feat.to(dtype=work_dtype)
    cls_feat = cls_feat.to(dtype=work_dtype)

    # --- Text embedding cache (per-class, computed once per process) ----
    # NOTE: text embeddings are computed + cached in fp32 to avoid fp16/fp32
    # mismatch with the (fp32) patch features downstream.
    cache = getattr(winclip_score, "_text_cache", None)
    if cache is None:
        cache = {}
        winclip_score._text_cache = cache
    if class_name not in cache:
        with torch.no_grad():
            norm_prompts, defc_prompts = build_text_prompts(class_name)
            # .float() here is essential -- CLIP may be in fp16 (AMP) and we
            # must NOT cache fp16 text vectors, otherwise the matmul with
            # fp32 patches fails with "expected same dtype".
            norm_text = _normalize(clip_model.encode_text(norm_prompts).float().mean(0), dim=-1)
            defc_text = _normalize(clip_model.encode_text(defc_prompts).float().mean(0), dim=-1)
            delta = _normalize((defc_text - norm_text).float(), dim=-1)
        cache[class_name] = delta.detach().cpu()
    delta = cache[class_name].to(device=device, dtype=work_dtype)

    B, D, Hp, Wp = patch_feat.shape
    flat = patch_feat.permute(0, 2, 3, 1).reshape(-1, D)  # (B*N, D)

    # Text-direction alignment (cosine with defect direction). Range ~[-1,1].
    sim_text = (flat @ delta.unsqueeze(1)).squeeze(1)  # (B*N,)
    # Map to non-negative "anomalousness" via ReLU on the positive half (WinCLIP
    # trick: patches pointing toward the defect direction are anomalous).
    map_text = F.relu(sim_text).reshape(B, Hp, Wp)

    # Reference-bank alignment: distance to nearest normal patch, range [0,2]
    # (since cosine sim in [-1,1]); normalise to [0,1] by multiplying by 0.5.
    # Chunked matmul over bank columns to avoid a (B*N, M) temp when M=2048
    # (small but saves ~25 MB peak in fp32; matters for 8-GB cards).
    if reference_patches is not None and reference_patches.numel() > 0:
        ref = reference_patches.to(device)
        if ref.dtype != flat.dtype:
            ref = ref.to(flat.dtype)
        max_sim = torch.full((flat.shape[0],), -2.0, device=device, dtype=flat.dtype)
        bsz = 512
        for j in range(0, ref.shape[0], bsz):
            sub = ref[j:j+bsz]
            s = (flat @ sub.t()).max(1).values
            torch.maximum(max_sim, s, out=max_sim)
            del sub, s
        d_ref = (1.0 - max_sim) * 0.5  # (B*N,) in [0,1]
        map_ref = d_ref.reshape(B, Hp, Wp)
        amap = alpha * map_text + (1 - alpha) * map_ref
        del max_sim, d_ref, map_ref
    else:
        amap = map_text

    # Image score = per-view max over spatial map
    img_score = amap.reshape(B, -1).max(dim=1).values

    return {"image_score": img_score, "anomaly_map": amap}


def build_winclip_reference(clip_model, patch_feat_all: torch.Tensor,
                            n_select: int = 4096) -> torch.Tensor:
    """
    Build a small normal-reference patch bank for one class via random
    selection (deterministic seed). Used by the WinCLIP branch.

    Memory note: we store the reference bank in FP16 to halve CPU/GPU
    storage. Cosine-similarity tolerates fp16 rounding easily (~1e-3 rel
    error), and the chunked matmul in winclip_score upcasts implicitly.
    """
    n = patch_feat_all.shape[0]
    n_select = min(n_select, n)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(n, generator=g)[:n_select]
    bank = _normalize(patch_feat_all[idx], dim=-1).contiguous()
    return bank.half() if bank.dtype != torch.float16 else bank
