"""StableAdamW + warmup cosine schedule (INP-Former training recipe)."""

from __future__ import annotations

import math
from typing import List

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class StableAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        clip_threshold: float = 1.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            clip_threshold=clip_threshold,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("clip_threshold", 1.0)

    @staticmethod
    def _rms(tensor: torch.Tensor) -> float:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.mul_(1 - group["lr"] * group["weight_decay"])
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("StableAdamW does not support sparse gradients")
                amsgrad = group["amsgrad"]
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                lr_scale = grad / denom
                rms = float(self._rms(lr_scale).item())
                lr_scale = max(1.0, rms / group["clip_threshold"])
                step_size = group["lr"] / bias_correction1 / lr_scale
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


class WarmCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=100, last_epoch=-1):
        self.base_value = base_value
        self.final_value = final_value
        self.total_iters = max(1, total_iters)
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        if step < self.warmup_iters:
            lr = self.base_value * float(step + 1) / float(max(1, self.warmup_iters))
        else:
            t = (step - self.warmup_iters) / float(max(1, self.total_iters - self.warmup_iters))
            t = min(1.0, max(0.0, t))
            lr = self.final_value + 0.5 * (self.base_value - self.final_value) * (1 + math.cos(math.pi * t))
        return [lr for _ in self.optimizer.param_groups]
