"""
Distributed / multi-GPU utilities.

Three modes:

1. **Single GPU (default)** -- ``python run.py``.

2. **Auto multi-GPU forward (recommended for competition)** --
   ``auto_parallel_forward(module, x, call="__call__", **kwargs)``
   splits a batch across all visible CUDA devices at the CALL SITE and
   concatenates dict outputs. Works for ANY module method (including
   custom ones like ``encode_image``), requires NO wrapping, and is
   DETERMINISTIC. Enable via ``--dp`` flag on run.py. Gives near-linear
   speedup on frozen backbones because there are no gradients to sync.

3. **Torch DDP** -- launch via ``torchrun --nproc_per_node=N run_ddp.py``.
   True multi-process, recommended for 4+ GPUs. Each rank processes a
   shard of train/test samples and rank 0 merges csv/masks.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Callable, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel


# ---------------------------------------------------------------------
# device / rank helpers
# ---------------------------------------------------------------------
def get_device_info():
    """Returns (device_str, is_ddp, rank, world_size, local_rank)."""
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        ws = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return f"cuda:{local_rank}", True, rank, ws, local_rank
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda", False, 0, 1, 0
    return "cpu", False, 0, 1, 0


def unwrap(model):
    while isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module
    return model


def n_visible_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def is_main_process() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def barrier():
    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------
# Auto multi-GPU forward (preferred; works with custom methods)
# ---------------------------------------------------------------------
def auto_parallel_forward(module: torch.nn.Module,
                           x: torch.Tensor,
                           call: str = "__call__",
                           **kwargs) -> Dict[str, Any]:
    """
    Split batch ``x`` along dim=0 across all visible CUDA devices, run
    ``getattr(module, call)(chunk, **kwargs)`` on each, and concatenate
    dict outputs back.

    * Works with any method (forward, encode_image, ...) on frozen modules.
    * Module weights are deepcopy'd to each GPU on first call and cached.
    * Tensor kwargs (e.g., labels) are split/moved automatically;
      non-tensor kwargs are broadcast as-is.
    * Output dict values that are Tensors are concatenated on dim=0;
      lists of Tensors are concatenated element-wise; non-tensor values
      (e.g., Hp/Wp ints) are taken from replica 0.
    * Single-GPU fallback: runs on current device with no splitting.
    """
    n = n_visible_gpus()
    if n <= 1:
        fn = module if call == "__call__" else getattr(module, call)
        return fn(x, **kwargs)

    # Cache shallow copies of the unwrapped module per GPU.
    cache = getattr(auto_parallel_forward, "_cache", None)
    if cache is None:
        cache = {}
        auto_parallel_forward._cache = cache  # type: ignore[attr-defined]

    B = x.shape[0]
    if B == 0:
        fn = module if call == "__call__" else getattr(module, call)
        return fn(x, **kwargs)

    base_dev = x.device if x.device.type == "cuda" else torch.device("cuda:0")
    chunk_size = (B + n - 1) // n
    outputs = []
    for i in range(n):
        s, e = i * chunk_size, min(B, (i + 1) * chunk_size)
        if s >= e:
            continue
        dev = torch.device(f"cuda:{i}")
        key = (id(unwrap(module)), i, call)
        if key not in cache:
            m_copy = copy.deepcopy(unwrap(module)).to(dev).eval()
            for p in m_copy.parameters():
                p.requires_grad_(False)
            cache[key] = m_copy
        m_copy = cache[key]
        chunk = x[s:e].to(dev, non_blocking=True)
        kw_dev = {
            k: (v[s:e].to(dev, non_blocking=True)
                if isinstance(v, torch.Tensor) and v.shape[0] == B
                else (v.to(dev) if isinstance(v, torch.Tensor) else v))
            for k, v in kwargs.items()
        }
        fn = m_copy if call == "__call__" else getattr(m_copy, call)
        with torch.no_grad():
            out = fn(chunk, **kw_dev)
        outputs.append((dev, out))

    ref_dev, ref_out = outputs[0]
    merged: Dict[str, Any] = {}
    for k, v in ref_out.items():
        if isinstance(v, torch.Tensor):
            chunks = [out[k].to(ref_dev, non_blocking=True) for _, out in outputs]
            merged[k] = torch.cat(chunks, dim=0)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            new_list = []
            for li in range(len(v)):
                chunks = [out[k][li].to(ref_dev, non_blocking=True) for _, out in outputs]
                new_list.append(torch.cat(chunks, dim=0))
            merged[k] = new_list
        else:
            merged[k] = v
    torch.cuda.synchronize()
    return merged


# ---------------------------------------------------------------------
# DDP helpers (used by run_ddp.py)
# ---------------------------------------------------------------------
def ddp_setup():
    if dist.is_initialized():
        return
    if "RANK" not in os.environ:
        raise RuntimeError("DDP must be launched via torchrun")
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def ddp_gather_tensor(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return t
    ws = dist.get_world_size()
    if ws == 1:
        return t
    shape = list(t.shape)
    shape[0] *= ws
    out = torch.empty(shape, dtype=t.dtype, device=t.device)
    dist.all_gather_into_tensor(out, t.contiguous())
    return out


def ddp_wrap_module(module: torch.nn.Module, local_rank: int,
                     find_unused: bool = False) -> torch.nn.Module:
    module = module.to(local_rank)
    if dist.is_initialized() and dist.get_world_size() > 1:
        return DistributedDataParallel(
            module, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=find_unused,
        )
    return module


def maybe_wrap_dp(model: torch.nn.Module, device: str, use_dp: bool = True):
    """
    Legacy no-op retained for API compatibility. Multi-GPU parallelism is
    applied via auto_parallel_forward() at each call site, which works
    with any method and doesn't require wrapping.
    """
    return model
