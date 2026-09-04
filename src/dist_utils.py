"""Single-node multi-GPU helpers (torchrun / DDP)."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def env_world() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def init_distributed(backend: str = "nccl") -> DistInfo:
    from datetime import timedelta

    rank, local_rank, world_size = env_world()
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"WORLD_SIZE={world_size} but CUDA is not available. "
                "Do not use torchrun on a CPU-only machine."
            )
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            # First-run DINOv2 download can exceed the default 10 min NCCL timeout.
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(minutes=120),
            )
        device = torch.device("cuda", local_rank)
        return DistInfo(True, rank, local_rank, world_size, device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistInfo(False, 0, 0, 1, device)


def is_main(info: Optional[DistInfo] = None) -> bool:
    if info is None:
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    return info.rank == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = value.detach().clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t


def all_gather_object(obj: Any) -> list:
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def setup_seed(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # T4: benchmark helps more than bit-exact determinism
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def strip_module_prefix(state: dict) -> dict:
    if not state:
        return state
    keys = list(state.keys())
    if any(k.startswith("module.") for k in keys):
        return {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    return state


def make_autocast(enabled: bool, dtype: torch.dtype = torch.float16):
    """T4 has no bfloat16 tensor cores — always fp16 for AMP."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled and torch.cuda.is_available(), dtype=dtype)
    from torch.cuda.amp import autocast as cuda_autocast

    return cuda_autocast(enabled=enabled and torch.cuda.is_available(), dtype=dtype)


def make_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled and torch.cuda.is_available())
    from torch.cuda.amp import GradScaler

    return GradScaler(enabled=enabled and torch.cuda.is_available())
