from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint

from .blocks import (
    AggregationBlock,
    Mlp,
    PrototypeBlock,
)

from .encoder import DinoV2Encoder


TRAINABLE_PREFIXES = (
    "bottleneck.",
    "aggregation.",
    "decoder.",
    "prototype_token",
    "map_fusion.",
    "refine_decoder.",
    "encoder.texture_branch.",
)


def resize(x, size):
    return F.interpolate(
        x,
        size=size,
        mode="bilinear",
        align_corners=False,
    )


class ConvGNAct(nn.Module):
    def __init__(
        self,
        cin,
        cout,
    ):
        super().__init__()

        groups = min(8, cout)

        while groups > 1 and cout % groups:
            groups -= 1

        self.block = nn.Sequential(
            nn.Conv2d(
                cin,
                cout,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                groups,
                cout,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class MapFusion(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.logits = nn.Parameter(
            torch.zeros(n)
        )

    def forward(self, maps):
        weights = torch.softmax(
            self.logits,
            0,
        )

        result = (
            maps[0] * weights[0]
        )

        for i in range(
            1,
            len(maps),
        ):
            result = (
                result
                + maps[i]
                * weights[i]
            )

        return result, weights


class PixelRefinementDecoder(nn.Module):
    """
    DINO coarse evidence:
        32x32

          -> 56x56 + texture stride8
          -> 112x112 + texture stride4
          -> full resolution
    """

    def __init__(
        self,
        embed_dim,
        texture_dim=64,
        hidden=128,
        levels=4,
    ):
        super().__init__()

        self.semantic_proj = nn.ModuleList([
            nn.Conv2d(
                embed_dim,
                hidden,
                1,
            )
            for _ in range(levels)
        ])

        self.coarse_fuse = nn.Sequential(
            ConvGNAct(
                hidden * levels + levels,
                hidden,
            ),
            ConvGNAct(
                hidden,
                hidden,
            ),
        )

        self.texture8 = nn.Conv2d(
            texture_dim,
            hidden,
            1,
        )

        self.stage8 = nn.Sequential(
            ConvGNAct(
                hidden * 2,
                hidden,
            ),
            ConvGNAct(
                hidden,
                hidden,
            ),
        )

        self.texture4 = nn.Conv2d(
            texture_dim,
            hidden // 2,
            1,
        )

        self.stage4 = nn.Sequential(
            ConvGNAct(
                hidden
                + hidden // 2,
                hidden // 2,
            ),
            ConvGNAct(
                hidden // 2,
                hidden // 2,
            ),
        )

        self.head = nn.Sequential(
            ConvGNAct(
                hidden // 2,
                hidden // 4,
            ),
            nn.Conv2d(
                hidden // 4,
                1,
                1,
            ),
        )

    def forward(
        self,
        semantic,
        maps,
        texture,
        output_size,
    ):
        base = semantic[0].shape[-2:]

        values = []

        for proj, feat in zip(
            self.semantic_proj,
            semantic,
        ):
            values.append(
                proj(feat.detach())
            )

        for amap in maps:
            values.append(
                resize(
                    amap.detach(),
                    base,
                )
            )

        x = self.coarse_fuse(
            torch.cat(
                values,
                1,
            )
        )

        # texture[1] = stride 8
        t8 = texture[1]

        x = resize(
            x,
            t8.shape[-2:],
        )

        x = self.stage8(
            torch.cat(
                [
                    x,
                    self.texture8(t8),
                ],
                1,
            )
        )

        # texture[0] = stride 4
        t4 = texture[0]

        x = resize(
            x,
            t4.shape[-2:],
        )

        x = self.stage4(
            torch.cat(
                [
                    x,
                    self.texture4(t4),
                ],
                1,
            )
        )

        x = resize(
            x,
            output_size,
        )

        return self.head(x)


class INPFormer(nn.Module):
    def __init__(
        self,
        encoder,
        inp_num=6,
        decoder_depth=8,
        bottleneck_drop=0,
        residual=False,
        grad_checkpoint=False,
        use_learnable_map_fusion=True,
        use_refinement=True,
        refinement_hidden_dim=128,
        coarse_weight=0.35,
        refine_weight=0.65,
        image_topk_ratio=0.001,
    ):
        super().__init__()

        self.encoder = encoder
        self.embed_dim = encoder.embed_dim
        self.num_heads = encoder.num_heads
        self.target_layers = encoder.target_layers

        self.residual = residual
        self.grad_checkpoint = grad_checkpoint

        # 8 DINO intermediate layers -> 4 groups.
        self.fuse_layer_encoder = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ]

        self.fuse_layer_decoder = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ]

        if decoder_depth < 8:
            raise ValueError(
                "Enhanced model requires "
                "decoder_depth >= 8"
            )

        dim = self.embed_dim

        norm = partial(
            nn.LayerNorm,
            eps=1e-8,
        )

        self.bottleneck = nn.ModuleList([
            Mlp(
                dim,
                dim * 4,
                dim,
                drop=bottleneck_drop,
            )
        ])

        self.prototype_token = nn.Parameter(
            torch.randn(
                inp_num,
                dim,
            )
        )

        self.aggregation = nn.ModuleList([
            AggregationBlock(
                dim=dim,
                num_heads=self.num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=norm,
            )
        ])

        self.decoder = nn.ModuleList([
            PrototypeBlock(
                dim=dim,
                num_heads=self.num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=norm,
            )
            for _ in range(
                decoder_depth
            )
        ])

        self.map_fusion = (
            MapFusion(4)
            if use_learnable_map_fusion
            else None
        )

        self.refine_decoder = (
            PixelRefinementDecoder(
                embed_dim=dim,
                texture_dim=(
                    encoder.texture_dim
                ),
                hidden=refinement_hidden_dim,
                levels=4,
            )
            if use_refinement
            else None
        )

        self.coarse_weight = (
            coarse_weight
        )

        self.refine_weight = (
            refine_weight
        )

        self.image_topk_ratio = (
            image_topk_ratio
        )

        self._init_weights()

    def _init_weights(self):
        for module in (
            list(self.bottleneck.modules())
            + list(self.aggregation.modules())
            + list(self.decoder.modules())
        ):
            if isinstance(
                module,
                nn.Linear,
            ):
                trunc_normal_(
                    module.weight,
                    std=0.01,
                    a=-0.03,
                    b=0.03,
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

        trunc_normal_(
            self.prototype_token,
            std=0.02,
        )

    def trainable_parameters(self):
        yield from self.bottleneck.parameters()
        yield from self.aggregation.parameters()
        yield from self.decoder.parameters()

        yield self.prototype_token

        if self.map_fusion is not None:
            yield from self.map_fusion.parameters()

        if self.refine_decoder is not None:
            yield from self.refine_decoder.parameters()

        if (
            self.encoder.texture_branch
            is not None
        ):
            yield from (
                self.encoder
                .texture_branch
                .parameters()
            )

    def trainable_state_dict(self):
        return {
            k: v
            for k, v
            in self.state_dict().items()
            if (
                k == "prototype_token"
                or k.startswith(
                    TRAINABLE_PREFIXES
                )
            )
        }

    @staticmethod
    def fuse(features):
        return torch.stack(
            features,
            1,
        ).mean(1)

    def gather_loss(
        self,
        query,
        keys,
    ):
        distance = (
            1
            - F.cosine_similarity(
                query.unsqueeze(2),
                keys.unsqueeze(1),
                dim=-1,
            )
        )

        return distance.min(
            2
        ).values.mean()

    def _block(
        self,
        block,
        x,
        proto,
    ):
        if (
            self.grad_checkpoint
            and self.training
        ):
            return checkpoint(
                block,
                x,
                proto,
                use_reentrant=False,
            )

        return block(
            x,
            proto,
        )

    def reconstruct(
        self,
        features,
    ):
        batch = (
            features[0].shape[0]
        )

        fused = self.fuse(
            features
        )

        proto = (
            self.prototype_token
            .unsqueeze(0)
            .expand(
                batch,
                -1,
                -1,
            )
        )

        for block in (
            self.aggregation
        ):
            proto = self._block(
                block,
                proto,
                fused,
            )

        gather = self.gather_loss(
            fused,
            proto,
        )

        tokens = fused

        for block in self.bottleneck:
            tokens = block(tokens)

        decoded = []

        for block in self.decoder:
            tokens = self._block(
                block,
                tokens,
                proto,
            )

            decoded.append(
                tokens
            )

        decoded = decoded[::-1]

        en = [
            self.fuse([
                features[i]
                for i in group
            ])
            for group
            in self.fuse_layer_encoder
        ]

        de = [
            self.fuse([
                decoded[i]
                for i in group
            ])
            for group
            in self.fuse_layer_decoder
        ]

        if self.residual:
            de = [
                e.detach() + d
                for e, d
                in zip(en, de)
            ]

        return en, de, gather

    def _spatial(
        self,
        features,
        h,
        w,
    ):
        result = []

        for x in features:
            b, n, c = x.shape

            if n != h * w:
                raise RuntimeError(
                    f"{n} tokens != {h}*{w}"
                )

            result.append(
                x.transpose(
                    1,
                    2,
                ).reshape(
                    b,
                    c,
                    h,
                    w,
                )
            )

        return result

    def forward(
        self,
        x,
        return_maps=False,
    ):
        dino = (
            self.encoder
            .forward_features(
                x,
                self.target_layers,
            )
        )

        en, de, gather = (
            self.reconstruct(
                dino
            )
        )

        patch = (
            self.encoder.patch_size
        )

        h = x.shape[-2] // patch
        w = x.shape[-1] // patch

        en = self._spatial(
            en,
            h,
            w,
        )

        de = self._spatial(
            de,
            h,
            w,
        )

        if not return_maps:
            return en, de, gather

        output_size = x.shape[-2:]

        maps = []

        for e, d in zip(
            en,
            de,
        ):
            amap = (
                1
                - F.cosine_similarity(
                    e.float(),
                    d.float(),
                    dim=1,
                )
            ).unsqueeze(1)

            maps.append(
                resize(
                    amap,
                    output_size,
                )
            )

        if self.map_fusion:
            coarse, map_weights = (
                self.map_fusion(
                    maps
                )
            )
        else:
            coarse = torch.stack(
                maps,
                0,
            ).mean(0)

            map_weights = None

        texture = (
            self.encoder
            .forward_texture_features(x)
        )

        refine_logits = None
        refine_map = None

        if self.refine_decoder:
            if len(texture) != 2:
                raise RuntimeError(
                    "Texture branch must provide "
                    "stride-4 and stride-8 features"
                )

            refine_logits = (
                self.refine_decoder(
                    en,
                    maps,
                    texture,
                    output_size,
                )
            )

            refine_map = torch.sigmoid(
                refine_logits
            )

            coarse_prob = (
                coarse / 2
            ).clamp(
                0,
                1,
            )

            denom = max(
                self.coarse_weight
                + self.refine_weight,
                1e-8,
            )

            anomaly_map = (
                self.coarse_weight
                * coarse_prob
                + self.refine_weight
                * refine_map
            ) / denom
        else:
            anomaly_map = coarse

        return {
            "en": en,
            "de": de,
            "g_loss": gather,
            "level_maps": maps,
            "coarse_map": coarse,
            "map_weights": map_weights,
            "texture_features": texture,
            "refine_logits": refine_logits,
            "refine_map": refine_map,
            "anomaly_map": anomaly_map,
        }

    @torch.no_grad()
    def predict(
        self,
        x,
        out_size=None,
    ):
        self.eval()

        output = self(
            x,
            return_maps=True,
        )

        amap = output[
            "anomaly_map"
        ]

        if out_size is not None:
            amap = resize(
                amap,
                (
                    out_size,
                    out_size,
                )
                if isinstance(
                    out_size,
                    int,
                )
                else out_size,
            )

        flat = amap.flatten(1)

        k = max(
            1,
            int(
                flat.shape[1]
                * self.image_topk_ratio
            ),
        )

        score = torch.topk(
            flat,
            k,
            1,
        ).values.mean(1)

        return amap, score


def anomaly_map_from_features(
    en,
    de,
    out_size=448,
):
    maps = []

    size = (
        (out_size, out_size)
        if isinstance(out_size, int)
        else out_size
    )

    for e, d in zip(en, de):
        amap = (
            1
            - F.cosine_similarity(
                e.float(),
                d.float(),
                dim=1,
            )
        ).unsqueeze(1)

        maps.append(
            resize(
                amap,
                size,
            )
        )

    return torch.stack(
        maps,
        0,
    ).mean(0)


def topk_score(
    amap,
    max_ratio=0.001,
):
    flat = amap.flatten(1)

    k = max(
        1,
        int(
            flat.shape[1]
            * max_ratio
        ),
    )

    return torch.topk(
        flat,
        min(k, flat.shape[1]),
        1,
    ).values.mean(1)


def build_model(
    encoder_name="dinov2reg_vit_base_14",
    inp_num=6,
    decoder_depth=8,
    bottleneck_drop=0,
    residual=False,
    encoder_source="auto",
    grad_checkpoint=False,
    use_learnable_map_fusion=True,
    use_refinement=True,
    refinement_hidden_dim=128,
    coarse_weight=0.35,
    refine_weight=0.65,
    image_topk_ratio=0.001,
):
    encoder = DinoV2Encoder(
        encoder_name,
        source=encoder_source,
        use_texture_branch=use_refinement,
        texture_base_dim=32,
        texture_dim=64,
    )

    return INPFormer(
        encoder=encoder,
        inp_num=inp_num,
        decoder_depth=decoder_depth,
        bottleneck_drop=bottleneck_drop,
        residual=residual,
        grad_checkpoint=grad_checkpoint,
        use_learnable_map_fusion=use_learnable_map_fusion,
        use_refinement=use_refinement,
        refinement_hidden_dim=refinement_hidden_dim,
        coarse_weight=coarse_weight,
        refine_weight=refine_weight,
        image_topk_ratio=image_topk_ratio,
    )
