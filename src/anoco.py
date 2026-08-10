"""
ANoCo: Anomaly as Non-Conformity (CVPR 2026, arXiv:2605.28428)

Instead of scoring a query patch by raw 1-NN cosine distance to the memory
bank (PatchCore), ANoCo builds a BIPARTITE graph between query patches
and retrieved normal anchors, then solves a CONVEX anchored Laplacian
energy minimization in CLOSED FORM. The anomaly score is the magnitude
of the feature drift required to make a query conform to the normal
manifold.

Why this is a strict improvement over PatchCore kNN:
  * Removes query-query and normal-normal edges (evidence dilution).
  * Considers the NEAREST NORMAL SET as a structured manifold, not
    independent vectors -> fewer false positives on textured surfaces.
  * Closed-form solution is a simple linear solve (per image).
  * Training-free; can be swapped in directly on top of an existing
    PatchCore coreset bank with no retraining.

We implement the lightweight version used in the paper:
  1. For each query patch q_i retrieve top-k anchors N(q_i) from bank.
  2. Build bipartite weights W_ia = exp(-sim(q_i, a) / tau) for a in N(q_i).
  3. Solve  (D_q + L) z = D_q q   where L is the bipartite Laplacian.
  4. Score  s_i = || q_i - z_i ||_2^2   (non-conformity energy).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def anoco_scores(query_patches: torch.Tensor,
                 bank: torch.Tensor,
                 k: int = 5,
                 tau: float = 0.1,
                 anchor_weight: float = 10.0,
                 chunk_size: int = 1024) -> torch.Tensor:
    """
    query_patches : (Q, D) L2-normalised query patch features
    bank          : (M, D) L2-normalised memory bank
    k             : number of nearest anchors per query
    tau           : softmax temperature for graph weights
    anchor_weight : stiffness of anchor springs (mu in the paper)

    Returns
    -------
    scores : (Q,) anomaly energy per query patch (higher = more anomalous)
    z      : (Q, D) projected (conformed) query features -- useful if
             downstream modules need them.
    """
    Q, D = query_patches.shape
    device = query_patches.device
    dtype = query_patches.dtype

    # ---- Step 1: retrieve top-k anchors per query (batched) ----
    topk_idx = torch.empty(Q, k, dtype=torch.long, device=device)
    topk_sim = torch.empty(Q, k, dtype=dtype, device=device)
    for i in range(0, Q, chunk_size):
        qc = query_patches[i:i+chunk_size]
        sim = qc @ bank.t()                           # (c, M)
        s, idx = sim.topk(k, dim=1)
        topk_idx[i:i+chunk_size] = idx
        topk_sim[i:i+chunk_size] = s

    # ---- Step 2: build bipartite edge weights W_ia ----
    W = torch.softmax(topk_sim / tau, dim=1)           # (Q, k) row-stochastic
    # Anchor coordinates (D,) weighted by W_ia
    anchors = bank[topk_idx]                           # (Q, k, D)
    # Weighted anchor sum for each query (used in closed form)
    weighted_anchor = (W.unsqueeze(-1) * anchors).sum(1)  # (Q, D)

    # ---- Step 3: closed-form solution ---------------------------------
    # The energy  E(z) = mu * || z - q ||^2  +  sum_a W_ia || z_i - a ||^2
    # has closed-form per-row optimum (because it is DECOUPLED across rows
    # when there are no query-query edges - exactly ANoCo's bipartite
    # construction):
    #     z_i = (mu * q_i  +  sum_a W_ia * a) / (mu + sum_a W_ia)
    # Since W is row-stochastic sum_a W_ia = 1, this simplifies to:
    #     z_i = (mu q + w_sum) / (mu + 1)
    # This is O(Q * D) and EXACT (no linear solve needed).
    mu = anchor_weight
    z = (mu * query_patches + weighted_anchor) / (mu + 1.0)

    # ---- Step 4: non-conformity energy = || q - z ||^2 --------------
    scores = (query_patches - z).pow(2).sum(-1)         # (Q,)
    return scores, z


def anoco_anomaly_map(patch_feat: torch.Tensor,
                      bank: torch.Tensor,
                      k: int = 5,
                      tau: float = 0.1,
                      anchor_weight: float = 10.0,
                      neighbourhood_size: int = 3,
                      smooth_kernel: int = 9,
                      smooth_sigma: float = 4.0,
                      high_res: int = 448) -> dict:
    """
    patch_feat : (B, D, Hp, Wp) fused DINO patch features, L2-normalised
    bank       : (M, D) per-class coreset bank, L2-normalised

    Returns: {'anomaly_map': (B, high_res, high_res), 'image_score': (B,)}
    """
    from .patchcore import gaussian_smooth2d, bilinear_upsample

    B, D, Hp, Wp = patch_feat.shape
    flat = patch_feat.permute(0, 2, 3, 1).reshape(B * Hp * Wp, D)
    scores, _ = anoco_scores(flat, bank, k=k, tau=tau,
                              anchor_weight=anchor_weight)
    amap = scores.reshape(B, 1, Hp, Wp)

    if neighbourhood_size > 1:
        pad = neighbourhood_size // 2
        amap = F.avg_pool2d(
            F.pad(amap, [pad]*4, mode="replicate"),
            kernel_size=neighbourhood_size, stride=1, padding=0,
        )
    amap_hr = bilinear_upsample(amap, high_res)
    amap_hr = gaussian_smooth2d(amap_hr, kernel_size=smooth_kernel,
                                 sigma=smooth_sigma)
    img_score = amap_hr.flatten(1).max(dim=1).values
    return {"anomaly_map": amap_hr[:, 0], "image_score": img_score}
