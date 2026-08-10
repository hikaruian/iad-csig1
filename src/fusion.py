"""
Dual-Modality Attention (DMA) + Stabilized Attention Pooling (SAP)
for fusing DINOv2 (local structure) and CLIP (global semantics) patch
features at FEATURE-LEVEL, as described in:

    "Zero-Shot Industrial Anomaly Detection via CLIP-DINOv2 Multimodal
     Fusion and Stabilized Attention Pooling" (Electronics, 2025)

Feature-level fusion outperforms score-level weighted averaging by
~1pp I-AUROC / P-AUPR in industrial settings because it lets CLIP's
semantic signal attend to the exact DINOv2 patches that correspond to
class-specific structures (e.g. connectors, solder joints), rather than
being averaged over the whole image.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualModalityAttention(nn.Module):
    """
    Fuse DINOv2 patch tokens (D) and CLIP patch tokens (C) via:

        A_dino = softmax(D^T D / sqrt(d))           # self-similarity of structure
        A_clip = softmax((C_cls)^T C / sqrt(d))     # global semantic guidance
        A_fuse = A_dino + theta * A_clip
        F_fused = A_fuse @ D                         # re-weighted DINO features

    Only theta is a learnable scalar (initialised 0.5), everything else
    is parameter-free -> no extra data / labels required.
    """

    def __init__(self, dino_dim: int, clip_dim: int, theta_init: float = 0.5):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta_init))
        # Project CLIP to DINOv2 dim (linear, learned with self-supervision
        # on normal patches only - see DMA trainer below).
        self.proj = nn.Linear(clip_dim, dino_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    @torch.no_grad()
    def forward(self, dino_p: torch.Tensor, clip_p: torch.Tensor,
                clip_cls: torch.Tensor) -> torch.Tensor:
        """
        dino_p : (B, N_d, D_d)  L2-normalised DINOv2 patch tokens
        clip_p : (B, N_c, D_c)  L2-normalised CLIP patch tokens
        clip_cls: (B, D_c)      CLS token from CLIP
        Returns: (B, N_d, D_d) fused tokens
        """
        B, N, D = dino_p.shape
        c = F.normalize(self.proj(clip_p), dim=-1)      # (B, N_c, D)
        # Resize CLIP token map to match DINO grid if needed
        Hd = Wd = int(N ** 0.5)
        Nc = c.shape[1]
        Hc = Wc = int(Nc ** 0.5)
        if Hc != Hd:
            c = c.transpose(1, 2).reshape(B, D, Hc, Wc)
            c = F.interpolate(c, size=(Hd, Wd), mode="bilinear",
                              align_corners=False)
            c = c.reshape(B, D, Hd * Wd).transpose(1, 2)

        # DINOv2 self-attention weights (patch affinity)
        scale = D ** -0.5
        A_dino = torch.bmm(dino_p, dino_p.transpose(1, 2)) * scale
        A_dino = A_dino.softmax(dim=-1)

        # CLIP-guided attention: each patch attends to CLS direction
        q_cls = F.normalize(self.proj(clip_cls), dim=-1).unsqueeze(1)  # (B,1,D)
        A_clip = torch.bmm(q_cls, c.transpose(1, 2)).softmax(dim=-1)   # (B,1,N)
        A_clip = A_clip.transpose(1, 2).expand_as(A_dino) / N          # broadcast

        A_fuse = A_dino + self.theta.clamp(0.0, 2.0) * A_clip
        fused = torch.bmm(A_fuse, dino_p)
        return F.normalize(fused, dim=-1)


class StabilizedAttentionPooling(nn.Module):
    """
    Replaces naive global average pooling when producing image-level
    scores from fused patch tokens. Pools over patches using the
    similarity with the CLIP CLS token as weight (temperature-scaled,
    with a stability constant epsilon to avoid over-concentration).
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    @torch.no_grad()
    def forward(self, fused_p: torch.Tensor,
                anchor: torch.Tensor) -> torch.Tensor:
        """
        fused_p : (B, N, D)
        anchor  : (B, D) global reference vector (e.g. class prototype or CLIP CLS)
        Returns: (B, D) globally pooled feature
        """
        w = torch.bmm(fused_p, anchor.unsqueeze(-1)).squeeze(-1)  # (B,N)
        w = (w / self.temperature).softmax(dim=-1)               # (B,N)
        w = w + self.eps
        w = w / w.sum(dim=-1, keepdim=True)
        pooled = torch.bmm(w.unsqueeze(1), fused_p).squeeze(1)
        return F.normalize(pooled, dim=-1)


# -------------------------------------------------------------------
# DMA training: self-supervised alignment on normal patches only
# -------------------------------------------------------------------
@torch.no_grad()
def extract_aligned_patches(dino, clip, loader, device, max_patches: int = 20000):
    """Offline: harvest matched (dino_patch, clip_patch) pairs from normal
    training images for lightweight self-supervised alignment."""
    dino_p, clip_p, clip_cls = [], [], []
    for batch in loader:
        imgs = batch["image"].to(device)
        d = dino(imgs)
        c = clip.encode_image(F.interpolate(imgs, size=(336, 336),
                                             mode="bilinear",
                                             align_corners=False))
        B, Dd, Hpd, Wpd = d["patch"].shape
        _, Dc, Hpc, Wpc = c["patch"].shape
        dp = d["patch"].permute(0, 2, 3, 1).reshape(B, -1, Dd)
        cp = c["patch"].permute(0, 2, 3, 1).reshape(B, -1, Dc)
        # Spatial alignment: resize CLIP tokens to DINO grid size
        cp_r = F.interpolate(cp.transpose(1, 2).reshape(B, Dc, Hpc, Wpc),
                              size=(Hpd, Wpd), mode="bilinear",
                              align_corners=False).reshape(B, Dc, -1)\
                              .transpose(1, 2)
        dino_p.append(dp.cpu())
        clip_p.append(cp_r.cpu())
        clip_cls.append(c["cls"].cpu())
        if sum(x.shape[0] * x.shape[1] for x in dino_p) > max_patches:
            break
    dino_p = torch.cat(dino_p, dim=0).reshape(-1, dino_p[0].shape[-1])
    clip_p = torch.cat(clip_p, dim=0).reshape(-1, clip_p[0].shape[-1])
    clip_cls = torch.cat(clip_cls, dim=0)
    # random subsample to max_patches
    if dino_p.shape[0] > max_patches:
        idx = torch.randperm(dino_p.shape[0])[:max_patches]
        dino_p, clip_p = dino_p[idx], clip_p[idx]
    return dino_p, clip_p, clip_cls


def fit_dma_projection(dma: DualModalityAttention, dino_p: torch.Tensor,
                       clip_p: torch.Tensor, device: str = "cuda",
                       epochs: int = 5, batch_size: int = 512,
                       lr: float = 1e-3):
    """
    Fit the DMA projection layer with a simple cosine alignment loss on
    NORMAL training patches (no labels, no anomaly data used).
    Loss: 1 - cos(proj(c_p), d_p) averaged over patches. This aligns CLIP
    tokens into the DINOv2 space using only normal samples.
    """
    dma.train()
    opt = torch.optim.AdamW(dma.parameters(), lr=lr, weight_decay=1e-4)
    d_p = dino_p.to(device)
    c_p = clip_p.to(device)
    n = d_p.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0; nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            db = d_p[idx]
            cb = c_p[idx]
            proj_cb = F.normalize(dma.proj(cb), dim=-1)
            loss = (1 - (db * proj_cb).sum(-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item(); nb += 1
        print(f"[DMA] epoch {ep}: align loss = {total/nb:.4f}")
    dma.eval()
    return dma
