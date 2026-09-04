"""INP reconstruction + pixel localization losses."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F


def _scale_grad(x, factor):
    return x * factor.to(
        x.dtype
    ).expand_as(x)


def global_cosine_hm_adaptive(
    en,
    de,
    y=3.0,
):
    if not en:
        raise ValueError(
            "Empty feature list"
        )

    loss = en[0].new_zeros(())

    for teacher, student in zip(
        en,
        de,
    ):
        teacher = teacher.detach()

        with torch.no_grad():
            distance = (
                1.0
                - F.cosine_similarity(
                    teacher.float(),
                    student.detach().float(),
                    dim=1,
                )
            ).unsqueeze(1)

            mean = (
                distance
                .mean(
                    (-2, -1),
                    keepdim=True,
                )
                .clamp_min(1e-6)
            )

            factor = (
                distance / mean
            ).pow(y).clamp(
                0,
                32,
            )

        rec = (
            1.0
            - F.cosine_similarity(
                teacher.flatten(1).float(),
                student.flatten(1).float(),
                dim=1,
            )
        ).mean()

        loss += rec

        if student.requires_grad:
            student.register_hook(
                partial(
                    _scale_grad,
                    factor=factor,
                )
            )

    return loss / len(en)


def total_loss(
    en,
    de,
    gather_loss,
    gather_weight=0.2,
    y=3.0,
):
    return (
        global_cosine_hm_adaptive(
            en,
            de,
            y,
        )
        + gather_weight
        * gather_loss
    )


def _target(target, size):
    if target.ndim == 3:
        target = target.unsqueeze(1)

    if target.shape[-2:] != size:
        target = F.interpolate(
            target.float(),
            size=size,
            mode="nearest",
        )

    return target.float()


def pixel_anomaly_loss(
    logits,
    target,
    focal_weight=1.0,
    dice_weight=0.5,
    hard_negative_weight=0.03,
    boundary_weight=0.05,
    hard_negative_ratio=0.01,
):
    target = _target(
        target,
        logits.shape[-2:],
    ).to(logits.device)

    x = logits.float()
    y = target.float()

    # Focal
    bce = F.binary_cross_entropy_with_logits(
        x,
        y,
        reduction="none",
    )

    p = torch.sigmoid(x)
    pt = p * y + (1 - p) * (1 - y)
    alpha = 0.25 * y + 0.75 * (1 - y)

    focal = (
        alpha
        * (1 - pt).pow(2)
        * bce
    ).mean()

    # Dice
    pf = p.flatten(1)
    yf = y.flatten(1)

    valid = yf.sum(1) > 0

    if valid.any():
        intersection = (
            pf[valid] * yf[valid]
        ).sum(1)

        dice = 1 - (
            (
                2 * intersection + 1e-6
            )
            / (
                pf[valid].sum(1)
                + yf[valid].sum(1)
                + 1e-6
            )
        ).mean()
    else:
        dice = x.sum() * 0

    # Hard negative
    negatives = []

    for pi, yi in zip(p, y):
        values = pi.flatten()[
            yi.flatten() < 0.5
        ]

        if values.numel():
            k = max(
                1,
                int(
                    values.numel()
                    * hard_negative_ratio
                ),
            )

            negatives.append(
                torch.topk(
                    values,
                    min(k, values.numel()),
                ).values.square().mean()
            )

    hn = (
        torch.stack(negatives).mean()
        if negatives
        else x.sum() * 0
    )

    # Boundary
    gt_edge = (
        F.max_pool2d(
            y,
            3,
            1,
            1,
        )
        + F.max_pool2d(
            -y,
            3,
            1,
            1,
        )
    ).clamp(0, 1)

    pred_edge = (
        F.max_pool2d(
            p,
            3,
            1,
            1,
        )
        + F.max_pool2d(
            -p,
            3,
            1,
            1,
        )
    ).clamp(0, 1)

    edge_valid = (
        gt_edge.flatten(1).sum(1)
        > 0
    )

    if edge_valid.any():
        pe = pred_edge.flatten(1)[edge_valid]
        ge = gt_edge.flatten(1)[edge_valid]

        inter = (pe * ge).sum(1)

        boundary = 1 - (
            (
                2 * inter + 1e-6
            )
            / (
                pe.sum(1)
                + ge.sum(1)
                + 1e-6
            )
        ).mean()
    else:
        boundary = x.sum() * 0

    losses = {
        "focal": focal_weight * focal,
        "dice": dice_weight * dice,
        "hard_negative": (
            hard_negative_weight * hn
        ),
        "boundary": (
            boundary_weight * boundary
        ),
    }

    losses["total"] = sum(
        losses.values()
    )

    return losses
