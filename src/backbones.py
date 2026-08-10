"""
Backbone feature extractors.

Two backbones are supported:

1. DINOv2 (default, primary)
   - facebookresearch/dinov2  (ViT-B/14, ViT-L/14, ViT-G/14)
   - Returns dense patch tokens + <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token.
   - Multi-layer feature harvesting (layers 8,9,10,11 for viT-L) is used
     to capture semantics + edges/textures, which is known to boost AD
     (cf. Dinomaly / INP-Former / PatchCore-ML).

2. OpenCLIP (secondary, for zero-shot WinCLIP-style ensemble)
   - open_clip ViT-L/14 @ 336px (e.g. ViT-L-14-336-quickgelu)
   - Provides CLS and patch tokens aligned with text embeddings,
     enabling WinCLIP-style compositional prompt ensembles.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DINOv2 multi-layer feature extractor
# ---------------------------------------------------------------------------
_DINOV2_URLS = {
    "vitb14": "dinov2_vitb14",
    "vitl14": "dinov2_vitl14",
    "vitg14": "dinov2_vitg14",
}


class DINOv2FeatureExtractor(nn.Module):
    """
    Extracts multi-scale dense features from a frozen DINOv2 ViT.

    Parameters
    ----------
    model_name : str
        One of "vitb14", "vitl14", "vitg14". ViT-L/14 is the recommended
        default for 50-class industrial AD (best speed/quality trade-off).
    layers : tuple[int, ...]
        Which transformer block outputs to harvest. Indices are 0-based.
        Default (8, 9, 10, 11) picks the last 4 blocks of a ViT-L/14 (depth 24
        indexes 0..23) – a good balance of semantic (last) and texture (earlier)
        features, matching the INP-Former / Dinomaly design.
    pretrained : bool
        If True, loads official torch.hub weights (facebookresearch/dinov2).
    """

    def __init__(self, model_name: str = "vitl14",
                 layers: Tuple[int, ...] = (8, 9, 10, 11),
                 pretrained: bool = True):
        super().__init__()
        assert model_name in _DINOV2_URLS, f"Unknown DINOv2 model {model_name}"
        self.model_name = model_name
        self.layers = tuple(layers)

        # Load the official DINOv2 model via torch hub
        # (The hub repo is cached locally after the first run.)
        self.model = torch.hub.load(
            "facebookresearch/dinov2", _DINOV2_URLS[model_name],
            pretrained=pretrained,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.patch_size = self.model.patch_size  # 14
        self.embed_dim = self.model.embed_dim
        self.num_layers = len(list(self.model.blocks))  # e.g. 24 for vitl14

        # Safety check on layer indices
        for li in self.layers:
            assert 0 <= li < self.num_layers, \
                f"layer index {li} out of range [0,{self.num_layers})"

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 3, H, W)

        Returns
        -------
        dict with:
            "cls"    : (B, D)           – <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token from the LAST block.
            "patch"  : (B, D, Hp, Wp)   – FUSED multi-layer patch features.
            "patch_layers": list[(B, D, Hp, Wp)] – per-layer patch features.
        """
        B, _, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"input H,W must be divisible by patch_size={self.patch_size}"
        Hp, Wp = H // self.patch_size, W // self.patch_size

        # Prepare tokens following DINOv2's forward
        x = self.model.prepare_tokens_with_masks(x)

        patch_layers: List[torch.Tensor] = []
        cls_token = None
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.layers:
                # tokens[:, 0] is CLS; tokens[:, 1:] are spatial patches
                tokens = x  # (B, 1+Hp*Wp, D)
                patch_layers.append(
                    tokens[:, 1:].reshape(B, Hp, Wp, -1)
                                  .permute(0, 3, 1, 2).contiguous()
                )
            if i == self.num_layers - 1:
                cls_token = x[:, 0]

        x_norm = self.model.norm(x)
        # After final norm, overwrite the CLS token with the normalized one
        cls_token = x_norm[:, 0]

        # Multi-layer fusion: channel-wise concatenation + 1x1 projection to D
        fused = torch.cat(patch_layers, dim=1)  # (B, k*D, Hp, Wp)
        # L2-normalise along channel to make cosine = dot in PatchCore
        fused = F.normalize(fused, dim=1)

        return {
            "cls": F.normalize(cls_token, dim=-1),
            "patch": fused,
            "patch_layers": patch_layers,
            "Hp": Hp, "Wp": Wp,
        }


# ---------------------------------------------------------------------------
# DINOv3 multi-layer feature extractor (Meta, arXiv:2508.10104)
# ---------------------------------------------------------------------------
# We support two flavours of DINOv3 because the flagship ViT-7B is extremely
# large (~6.7B params, patch size 16) and rarely fits a single GPU at 448px:
#
#   1. "official_7b":    facebookresearch/dinov3 reference implementation
#                        (requires manual install; gives best dense features).
#   2. "timm_l16":       timm's "vit_large_patch16_dinov3.sat493m"
#                        (~493M params, patch16, distilled, runs on 8GB GPUs).
#
# NOTE (empirical, Dinomaly author 2025-08): DINOv3 gives *better* pixel-level
# anomaly maps but *slightly worse* image-level CLS discrimination than
# DINOv2-L/14. The recommended recipe is to ENSEMBLE DINOv2+DINOv3 patches
# for the final score rather than replace DINOv2 outright. See
# configs/dinov3_ensemble.yaml for that setup.

_DINOV3_TIMM_VARIANTS = {
    "l16_493": "vit_large_patch16_dinov3.sat493m",
    "l16_493_in1k": "vit_large_patch16_dinov3.sat493m_in1k",
    "b16":  "vit_base_patch16_dinov3.lvd1689m",
    "convnext_l": "convnext_large.dinov3_lvd1689m",
}


class DINOv3FeatureExtractor(nn.Module):
    """
    DINOv3 feature extractor using the timm models (easiest to install).

    Parameters
    ----------
    variant : str
        Key into _DINOV3_TIMM_VARIANTS, or a full timm model name starting
        with "vit_".
    layers : tuple[int,...]
        Transformer block indices to harvest. Default is (18..23) for the
        SAT-distilled L/16 variant which has 24 blocks.
    """

    def __init__(self, variant: str = "l16_493",
                 layers: Tuple[int, ...] = (18, 19, 20, 21, 22, 23),
                 pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "DINOv3 via timm requires `pip install timm huggingface_hub`"
            ) from e

        self.variant = variant
        self.layers = tuple(layers)
        model_name = _DINOV3_TIMM_VARIANTS.get(variant, variant)
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       num_classes=0, dynamic_img_size=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Infer patch size / embed dim (works for ViT variants)
        self.patch_size = getattr(self.model, "patch_size", 16)
        if isinstance(self.patch_size, tuple):
            self.patch_size = self.patch_size[0]
        self.embed_dim = self.model.embed_dim
        self.num_layers = len(self.model.blocks)

        # Register forward hooks to capture intermediate block outputs
        self._captured: Dict[int, torch.Tensor] = {}
        for li in self.layers:
            assert 0 <= li < self.num_layers, \
                f"DINOv3 layer {li} out of [0,{self.num_layers})"
            self.model.blocks[li].register_forward_hook(
                self._make_hook(li)
            )

    def _make_hook(self, layer_idx: int):
        def _hook(module, inp, out):
            self._captured[layer_idx] = out
        return _hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        Hp, Wp = H // self.patch_size, W // self.patch_size

        self._captured.clear()
        # Use forward_features (returns patch tokens pre-logit)
        out = self.model.forward_features(x)

        # timm ViT returns a tensor of shape (B, 1+N, D) where token 0 is CLS
        if out.dim() == 3:
            cls_tok = out[:, 0]
            patch_base = out[:, 1:]
        else:
            # Some variants return a dict; take CLS and tokens accordingly
            cls_tok = out[:, 0]
            patch_base = out[:, 1:]

        patch_layers: List[torch.Tensor] = []
        for li in self.layers:
            t = self._captured[li]
            # tokens are (B, 1+N, D)
            if t.dim() == 3:
                pl = t[:, 1:]
            else:
                pl = t
            patch_layers.append(
                pl.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()
            )

        fused = torch.cat(patch_layers, dim=1)
        fused = F.normalize(fused, dim=1)
        cls_tok = F.normalize(cls_tok, dim=-1)

        return {
            "cls": cls_tok,
            "patch": fused,
            "patch_layers": patch_layers,
            "Hp": Hp, "Wp": Wp,
        }


# ---------------------------------------------------------------------------
# OpenCLIP feature extractor (for WinCLIP zero-shot branch)
# ---------------------------------------------------------------------------
class CLIPFeatureExtractor(nn.Module):
    """
    Light wrapper around open_clip to obtain CLS and patch tokens aligned
    with a text encoder for WinCLIP-style zero-shot AD.
    """

    def __init__(self, model_name: str = "ViT-L-14-336",
                 pretrained: str = "openai"):
        super().__init__()
        import open_clip
        self.model_name = model_name
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # infer patch size / grid
        if hasattr(self.model.visual, "patch_size"):
            self.patch_size = self.model.visual.patch_size
        else:
            self.patch_size = 14
        self.embed_dim = self.model.visual.output_dim  # text/CLS proj dim
        # visual token dim before projection
        if hasattr(self.model.visual, "transformer"):
            self.visual_dim = self.model.visual.ln_post.normalized_shape[0]
        else:
            self.visual_dim = self.embed_dim

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size

        visual = self.model.visual
        # Patch embedding
        x_vis = visual.conv1(x)  # (B, Dv, Hp, Wp)
        x_vis = x_vis.reshape(B, -1, Hp * Wp).permute(0, 2, 1)  # (B, N, Dv)

        # Prepend CLS token
        cls = visual.class_embedding.to(x.dtype).expand(B, 1, -1)
        x_vis = torch.cat([cls, x_vis], dim=1)
        x_vis = x_vis + visual.positional_embedding.to(x.dtype)
        x_vis = visual.patch_dropout(x_vis)
        x_vis = visual.ln_pre(x_vis)
        x_vis = x_vis.permute(1, 0, 2)  # (N+1, B, Dv) for transformer
        x_vis = visual.transformer(x_vis, attn_mask=visual.attn_mask)
        x_vis = x_vis.permute(1, 0, 2)  # (B, N+1, Dv)
        x_vis = visual.ln_post(x_vis)

        cls_tok = x_vis[:, 0] @ visual.proj  # (B, D)
        patch_tok = x_vis[:, 1:]             # (B, N, Dv)
        patch_tok = patch_tok @ visual.proj  # (B, N, D)
        patch_tok = F.normalize(patch_tok, dim=-1)
        cls_tok = F.normalize(cls_tok, dim=-1)

        return {
            "cls": cls_tok,
            "patch": patch_tok.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2),
            "Hp": Hp, "Wp": Wp,
        }

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        import open_clip
        tok = self.tokenizer(prompts).to(next(self.model.parameters()).device)
        feats = self.model.encode_text(tok)
        feats = F.normalize(feats, dim=-1)
        return feats
