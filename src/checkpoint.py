"""Load / save helpers that survive DDP `module.` prefixes."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .dist_utils import strip_module_prefix
from .model import INPFormer, TRAINABLE_PREFIXES, build_model  # INPFormer used in return type


def torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state(ckpt: Dict[str, Any]) -> Tuple[dict, dict]:
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
        meta = ckpt
    else:
        state = ckpt
        meta = {}
    return strip_module_prefix(state), meta


def load_inpformer(ckpt_path: str, device, **overrides) -> Tuple[INPFormer, dict]:
    ckpt = torch_load(ckpt_path, map_location="cpu")
    state, meta = extract_state(ckpt)
    saved_args = dict(meta.get("args", {}) or {}) if isinstance(meta, dict) else {}

    encoder = overrides.get("encoder") or saved_args.get("encoder", "dinov2reg_vit_base_14")
    residual = bool(saved_args.get("residual", False)) or bool(overrides.get("residual", False))
    arch = {
        "encoder_name": encoder,
        "inp_num": int(saved_args.get("inp_num", overrides.get("inp_num", 6))),
        "decoder_depth": int(saved_args.get("decoder_depth", overrides.get("decoder_depth", 8))),
        "bottleneck_drop": float(saved_args.get("bottleneck_drop", overrides.get("bottleneck_drop", 0.0))),
        "residual": residual,
        "encoder_source": overrides.get("encoder_source") or saved_args.get("encoder_source", "auto"),
        "grad_checkpoint": False,
    }
    model = build_model(**arch)
    missing, unexpected = model.load_state_dict(state, strict=False)
    trainable_missing = [k for k in missing if k.startswith(TRAINABLE_PREFIXES) or k == "prototype_token"]
    if trainable_missing:
        raise RuntimeError(
            f"Checkpoint is missing trainable keys (decoder would be random): {trainable_missing[:12]}"
        )
    model.to(device).eval()
    return model, {"missing": list(missing), "unexpected": list(unexpected), "args": saved_args, "arch": arch}
