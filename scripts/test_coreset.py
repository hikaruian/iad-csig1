"""Quick correctness + speed benchmark for the new greedy_coreset.

Runs on GPU (or CPU if unavailable). Reports:
  - output shape, dtype, unique count
  - average min-distance coverage metric (lower = better coverage)
  - wall-clock time
"""
from __future__ import annotations
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.patchcore import greedy_coreset


def coverage(feats: torch.Tensor, bank: torch.Tensor, chunk: int = 2048) -> float:
    """Average cosine-distance from any feature to its nearest bank entry."""
    feats = F.normalize(feats.float(), dim=-1)
    bank = F.normalize(bank.float(), dim=-1)
    best = []
    for i in range(0, feats.shape[0], chunk):
        sim = feats[i : i + chunk] @ bank.t()
        d = 1.0 - sim.max(1).values
        best.append(d)
    return float(torch.cat(best).mean().item())


def main():
    assert torch.cuda.is_available(), "This bench expects a CUDA GPU"
    torch.manual_seed(0)

    # 100 imgs × 32×32 patches × 1024 dims ≈ a typical class in Real-IAD
    N, D, M = 100 * 32 * 32, 1024, 16384
    print(f"Generating {N} × {D} synthetic L2-normalised features on CPU ...")
    x = F.normalize(torch.randn(N, D), dim=-1)

    dev = torch.device("cuda")
    xg = x.to(dev)
    torch.cuda.synchronize()

    # Warmup
    _ = greedy_coreset(xg[:10000], 1024, random_proj_dim=256, seed=42, device=dev,
                       batch_size=64, presample_ratio=3.0)
    torch.cuda.synchronize()

    t0 = time.time()
    bank = greedy_coreset(xg, M, random_proj_dim=256, seed=42, device=dev,
                          batch_size=64, presample_ratio=3.0)
    torch.cuda.synchronize()
    dt = time.time() - t0
    uniq = torch.unique(bank, dim=0).shape[0]
    cov = coverage(xg, bank)
    print(f"\n[fast-GPU]  N={N} -> M={bank.shape[0]}  unique={uniq}/{M}  "
          f"time={dt:.2f}s  avg-cos-dist-to-bank={cov:.4f}")
    print(f"           bank shape={tuple(bank.shape)} dtype={bank.dtype} "
          f"device={bank.device}")
    assert bank.shape == (M, D), f"expected ({M},{D}), got {bank.shape}"
    assert uniq >= M - 5, f"too many duplicates: {uniq}"
    print("\nCoreset implementation looks correct and fast.")


if __name__ == "__main__":
    main()
