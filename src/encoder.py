"""Frozen DINOv2 encoder + trainable high-resolution texture branch."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .dist_utils import barrier, env_world, is_main


ENCODER_PRESETS = {
    "dinov2reg_vit_small_14": {
        "hub": "dinov2_vits14_reg",
        "timm": "vit_small_patch14_reg4_dinov2.lvd142m",
        "embed_dim": 384,
        "num_heads": 6,
        "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],
        "patch_size": 14,
    },
    "dinov2reg_vit_base_14": {
        "hub": "dinov2_vitb14_reg",
        "timm": "vit_base_patch14_reg4_dinov2.lvd142m",
        "embed_dim": 768,
        "num_heads": 12,
        "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],
        "patch_size": 14,
    },
    "dinov2reg_vit_large_14": {
        "hub": "dinov2_vitl14_reg",
        "timm": "vit_large_patch14_reg4_dinov2.lvd142m",
        "embed_dim": 1024,
        "num_heads": 16,
        "target_layers": [4, 6, 8, 10, 12, 14, 16, 18],
        "patch_size": 14,
    },
}


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()

        groups = min(8, out_channels)
        while groups > 1 and out_channels % groups:
            groups -= 1

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualTextureBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        groups = min(8, channels)
        while groups > 1 and channels % groups:
            groups -= 1

        self.conv1 = ConvGNAct(channels, channels)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
        )

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(
            x + self.conv2(self.conv1(x))
        )


class TextureEncoder(nn.Module):
    """Produces stride-4 and stride-8 high-resolution features."""

    def __init__(
        self,
        base_dim: int = 32,
        texture_dim: int = 64,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            ConvGNAct(
                3,
                base_dim,
                kernel_size=5,
                stride=2,
            ),
            ResidualTextureBlock(base_dim),
        )

        self.stage1 = nn.Sequential(
            ConvGNAct(
                base_dim,
                texture_dim,
                stride=2,
            ),
            ResidualTextureBlock(texture_dim),
        )

        self.stage2 = nn.Sequential(
            ConvGNAct(
                texture_dim,
                texture_dim,
                stride=2,
            ),
            ResidualTextureBlock(texture_dim),
        )

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

    def forward(self, x):
        x = self.stem(x)
        f4 = self.stage1(x)
        f8 = self.stage2(f4)
        return [f4, f8]


def prefetch_encoder_weights(
    name: str,
    source: str = "auto",
    stamp_dir: str = "runs/_prefetch",
) -> None:
    rank, _, world_size = env_world()

    if world_size <= 1:
        return

    stamp = Path(stamp_dir) / f"{name}.{source}.ready"
    fail = Path(stamp_dir) / f"{name}.{source}.fail"

    stamp.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if rank == 0:
        try:
            if fail.exists():
                fail.unlink()

            tmp = DinoV2Encoder(
                name,
                source=source,
                use_texture_branch=False,
            )

            del tmp

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            stamp.write_text(
                "ok",
                encoding="utf-8",
            )

        except Exception as exc:
            fail.write_text(
                repr(exc),
                encoding="utf-8",
            )
            raise

        return

    deadline = time.time() + 3600

    while time.time() < deadline:
        if stamp.exists():
            return

        if fail.exists():
            raise RuntimeError(
                fail.read_text(encoding="utf-8")
            )

        time.sleep(1)

    raise TimeoutError(
        f"Timed out prefetching {name}"
    )


class DinoV2Encoder(nn.Module):
    def __init__(
        self,
        name: str = "dinov2reg_vit_base_14",
        source: str = "auto",
        use_texture_branch: bool = True,
        texture_base_dim: int = 32,
        texture_dim: int = 64,
    ):
        super().__init__()

        if name not in ENCODER_PRESETS:
            raise ValueError(
                f"Unknown encoder: {name}"
            )

        self.cfg = ENCODER_PRESETS[name]
        self.name = name
        self.backend = None

        self.model = self._load_synced(source)

        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.texture_dim = texture_dim

        self.texture_branch = (
            TextureEncoder(
                texture_base_dim,
                texture_dim,
            )
            if use_texture_branch
            else None
        )

    @property
    def embed_dim(self):
        return int(self.cfg["embed_dim"])

    @property
    def num_heads(self):
        return int(self.cfg["num_heads"])

    @property
    def target_layers(self):
        return list(self.cfg["target_layers"])

    @property
    def patch_size(self):
        return int(self.cfg["patch_size"])

    def _load_synced(self, source):
        if is_main():
            try:
                model = self._load(source)
            finally:
                barrier()
            return model

        barrier()
        return self._load(source)

    def _load(self, source):
        if source not in ("auto", "hub", "timm"):
            raise ValueError(
                f"Unknown source: {source}"
            )

        order = (
            ["hub", "timm"]
            if source == "auto"
            else [source]
        )

        errors = []

        for kind in order:
            try:
                if kind == "hub":
                    model = torch.hub.load(
                        "facebookresearch/dinov2",
                        self.cfg["hub"],
                        pretrained=True,
                        trust_repo=True,
                    )

                    self.backend = "hub"
                    return model

                import timm

                model = timm.create_model(
                    self.cfg["timm"],
                    pretrained=True,
                    dynamic_img_size=True,
                    num_classes=0,
                )

                self.backend = "timm"
                return model

            except Exception as exc:
                errors.append(
                    f"{kind}: {exc}"
                )

        raise RuntimeError(
            "Failed loading DINOv2:\n"
            + "\n".join(errors)
        )

    def train(self, mode=True):
        # Set wrapper/texture branch to requested mode,
        # but always force frozen DINO to eval.
        super().train(mode)

        self.model.eval()

        if self.texture_branch is not None:
            self.texture_branch.train(mode)

        return self

    def forward_features(
        self,
        x: torch.Tensor,
        layers: Sequence[int] | None = None,
    ) -> List[torch.Tensor]:

        layers = (
            list(layers)
            if layers is not None
            else self.target_layers
        )

        with torch.no_grad():
            feats = self._forward_features(
                x,
                layers,
            )

        return [
            f.detach()
            for f in feats
        ]

    def _forward_features(self, x, layers):
        expected = (
            x.shape[-2] // self.patch_size
        ) * (
            x.shape[-1] // self.patch_size
        )

        if self.backend == "hub":
            feats = self.model.get_intermediate_layers(
                x,
                n=list(layers),
                reshape=False,
                return_class_token=False,
                norm=False,
            )

            result = []

            for feat in feats:
                if feat.ndim == 4:
                    feat = (
                        feat.flatten(2)
                        .transpose(1, 2)
                    )

                if feat.shape[1] != expected:
                    raise RuntimeError(
                        f"DINO hub tokens={feat.shape[1]}, "
                        f"expected={expected}"
                    )

                result.append(
                    feat.contiguous()
                )

            return result

        feats = self.model.forward_intermediates(
            x,
            indices=list(layers),
            norm=False,
            stop_early=True,
            output_fmt="NLC",
            intermediates_only=True,
        )

        prefix = int(
            getattr(
                self.model,
                "num_prefix_tokens",
                1 + int(
                    getattr(
                        self.model,
                        "num_reg_tokens",
                        0,
                    )
                ),
            )
        )

        result = []

        for feat in feats:
            if feat.ndim == 4:
                # Most timm versions with NLC won't hit this.
                if feat.shape[1] == self.embed_dim:
                    feat = (
                        feat.flatten(2)
                        .transpose(1, 2)
                    )
                else:
                    feat = feat.reshape(
                        feat.shape[0],
                        -1,
                        feat.shape[-1],
                    )

            if feat.shape[1] != expected:
                if feat.shape[1] >= expected + prefix:
                    feat = feat[
                        :,
                        prefix:prefix + expected,
                    ]
                else:
                    raise RuntimeError(
                        f"DINO timm tokens={feat.shape[1]}, "
                        f"expected={expected}"
                    )

            result.append(
                feat.contiguous()
            )

        return result

    def forward_texture_features(self, x):
        if self.texture_branch is None:
            return []

        return self.texture_branch(x)
