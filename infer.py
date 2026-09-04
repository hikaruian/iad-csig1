#!/usr/bin/env python3
"""
Inference for enhanced INP-Former.

Supports:
    - learned pixel refinement
    - high-resolution texture branch
    - horizontal-flip TTA
    - legacy NormalStats post-processing
    - optional view gating
    - single GPU / torchrun DDP inference

Single GPU:
    python infer.py \
        --test-root /data/CSIG/Test_A \
        --ckpt runs/inpformer_v3/best.pth

2 GPUs:
    torchrun \
        --standalone \
        --nproc_per_node=2 \
        infer.py \
        --test-root /data/CSIG/Test_A \
        --ckpt runs/inpformer_v3/best.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil

from pathlib import Path

import numpy as np
import torch

from PIL import Image

from torch.utils.data import (
    DataLoader,
    DistributedSampler,
)

from tqdm import tqdm


from src.checkpoint import (
    torch_load,
)

from src.data import (
    CSIGImageDataset,
    CSIGSampleDataset,
    build_transform,
)

from src.dist_utils import (
    all_gather_object,
    barrier,
    cleanup,
    init_distributed,
    is_main,
    make_autocast,
    setup_seed,
)

from src.encoder import (
    prefetch_encoder_weights,
)

from src.model import (
    build_model,
)

from src.postprocess import (
    calibrate_scale,
    maps_to_uint8,
    smooth_map,
    squash_score,
)

from src.refine import (
    NormalStats,
    apply_view_gate,
    apply_view_refine,
)

from src.submission import (
    zip_submission,
)


# ============================================================
# Arguments
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description=(
            "Enhanced INP-Former inference"
        )
    )

    p.add_argument(
        "--test-root",
        type=str,
        required=True,
    )

    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
    )

    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/submission",
    )

    p.add_argument(
        "--zip",
        type=str,
        default="outputs/my_submission.zip",
    )

    # --------------------------------------------------------
    # Architecture fallback
    #
    # Checkpoint args have priority.
    # --------------------------------------------------------

    p.add_argument(
        "--encoder",
        type=str,
        default="",
    )

    p.add_argument(
        "--encoder-source",
        type=str,
        default="auto",
        choices=[
            "auto",
            "hub",
            "timm",
        ],
    )

    p.add_argument(
        "--image-size",
        type=int,
        default=448,
    )

    p.add_argument(
        "--inp-num",
        type=int,
        default=6,
    )

    p.add_argument(
        "--decoder-depth",
        type=int,
        default=8,
    )

    p.add_argument(
        "--bottleneck-drop",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--residual",
        action="store_true",
    )

    p.add_argument(
        "--refinement-hidden-dim",
        type=int,
        default=128,
    )

    p.add_argument(
        "--coarse-weight",
        type=float,
        default=0.35,
    )

    p.add_argument(
        "--refine-weight",
        type=float,
        default=0.65,
    )

    p.add_argument(
        "--image-topk-ratio",
        type=float,
        default=0.001,
    )

    # --------------------------------------------------------
    # inference
    # --------------------------------------------------------

    p.add_argument(
        "--samples-per-batch",
        type=int,
        default=1,
        help=(
            "Physical samples per GPU. "
            "One sample has 5 views. "
            "T4 16G: start from 1."
        ),
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=True,
    )

    p.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
    )

    p.add_argument(
        "--tta-flip",
        action="store_true",
    )

    # --------------------------------------------------------
    # anomaly scoring
    # --------------------------------------------------------

    p.add_argument(
        "--max-ratio",
        type=float,
        default=0.001,
        help=(
            "Top-K pixel ratio for image score. "
            "Recommended search: "
            "0.0005/0.001/0.005/0.01"
        ),
    )

    p.add_argument(
        "--reduce",
        type=str,
        default="max",
        choices=[
            "max",
            "mean",
            "lse",
        ],
    )

    # --------------------------------------------------------
    # pixel postprocess
    # --------------------------------------------------------

    p.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help=(
            "Gaussian smoothing. "
            "For learned refinement start with 0."
        ),
    )

    p.add_argument(
        "--mask-scale",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--train-root",
        type=str,
        default="",
    )

    p.add_argument(
        "--calibrate-n",
        type=int,
        default=256,
    )

    # --------------------------------------------------------
    # Legacy refinement
    #
    # New network already contains learned refinement.
    # Therefore default OFF.
    # --------------------------------------------------------

    p.add_argument(
        "--legacy-refine",
        dest="legacy_refine",
        action="store_true",
        default=False,
    )

    p.add_argument(
        "--no-legacy-refine",
        dest="legacy_refine",
        action="store_false",
    )

    p.add_argument(
        "--gamma",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--border",
        type=int,
        default=16,
    )

    p.add_argument(
        "--fg-gate",
        dest="fg_gate",
        action="store_true",
        default=False,
    )

    p.add_argument(
        "--no-fg-gate",
        dest="fg_gate",
        action="store_false",
    )

    p.add_argument(
        "--stats-path",
        type=str,
        default="",
    )

    p.add_argument(
        "--stats-per-class",
        type=int,
        default=20,
    )

    # --------------------------------------------------------
    # View gate
    # --------------------------------------------------------

    p.add_argument(
        "--view-gate",
        dest="view_gate",
        action="store_true",
        default=False,
    )

    p.add_argument(
        "--no-view-gate",
        dest="view_gate",
        action="store_false",
    )

    p.add_argument(
        "--gate-k",
        type=float,
        default=1.25,
    )

    p.add_argument(
        "--gate-temp",
        type=float,
        default=0.35,
    )

    p.add_argument(
        "--gate-floor",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--gate-hard",
        dest="gate_hard",
        action="store_true",
        default=True,
    )

    p.add_argument(
        "--gate-soft",
        dest="gate_hard",
        action="store_false",
    )

    p.add_argument(
        "--gate-mode",
        type=str,
        default="max",
        choices=[
            "max",
            "z",
        ],
    )

    p.add_argument(
        "--gate-margin",
        type=float,
        default=1.12,
    )

    return p.parse_args()


# ============================================================
# Checkpoint architecture
# ============================================================

def checkpoint_args(
    path: str,
):

    checkpoint = torch_load(
        path,
        map_location="cpu",
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and isinstance(
            checkpoint.get(
                "args"
            ),
            dict,
        )
    ):

        result = dict(
            checkpoint["args"]
        )

    else:

        result = {}

    del checkpoint

    return result


def get_arg(
    checkpoint_cfg,
    cli_value,
    name,
    default,
):

    value = checkpoint_cfg.get(
        name,
        None,
    )

    if value is not None:
        return value

    if cli_value is not None:
        return cli_value

    return default


# ============================================================
# Model loader
# ============================================================

def load_enhanced_model(
    args,
    device,
):

    ckpt = torch_load(
        args.ckpt,
        map_location="cpu",
    )

    cfg = (
        ckpt.get(
            "args",
            {},
        )
        if isinstance(
            ckpt,
            dict,
        )
        else {}
    )

    encoder_name = (
        args.encoder
        or cfg.get(
            "encoder",
            "dinov2reg_vit_base_14",
        )
    )

    image_size = int(
        cfg.get(
            "image_size",
            args.image_size,
        )
    )

    inp_num = int(
        cfg.get(
            "inp_num",
            args.inp_num,
        )
    )

    decoder_depth = int(
        cfg.get(
            "decoder_depth",
            args.decoder_depth,
        )
    )

    bottleneck_drop = float(
        cfg.get(
            "bottleneck_drop",
            args.bottleneck_drop,
        )
    )

    residual = bool(
        cfg.get(
            "residual",
            args.residual,
        )
    )

    refinement_hidden_dim = int(
        cfg.get(
            "refinement_hidden_dim",
            args.refinement_hidden_dim,
        )
    )

    coarse_weight = float(
        cfg.get(
            "coarse_weight",
            args.coarse_weight,
        )
    )

    refine_weight = float(
        cfg.get(
            "refine_weight",
            args.refine_weight,
        )
    )

    image_topk_ratio = float(
        cfg.get(
            "image_topk_ratio",
            args.image_topk_ratio,
        )
    )

    # Previous final train.py fixed map fusion=False.
    # If future training saves this argument, respect it.
    use_map_fusion = bool(
        cfg.get(
            "learnable_map_fusion",
            False,
        )
    )

    use_refinement = bool(
        cfg.get(
            "refinement",
            True,
        )
    )

    model = build_model(

        encoder_name=(
            encoder_name
        ),

        inp_num=(
            inp_num
        ),

        decoder_depth=(
            decoder_depth
        ),

        bottleneck_drop=(
            bottleneck_drop
        ),

        residual=(
            residual
        ),

        encoder_source=(
            args.encoder_source
        ),

        grad_checkpoint=False,

        use_learnable_map_fusion=(
            use_map_fusion
        ),

        use_refinement=(
            use_refinement
        ),

        refinement_hidden_dim=(
            refinement_hidden_dim
        ),

        coarse_weight=(
            coarse_weight
        ),

        refine_weight=(
            refine_weight
        ),

        image_topk_ratio=(
            image_topk_ratio
        ),
    )

    # --------------------------------------------------------
    # State dict
    # --------------------------------------------------------

    if (
        isinstance(
            ckpt,
            dict,
        )
        and "model" in ckpt
    ):

        state = ckpt[
            "model"
        ]

    else:

        state = ckpt

    # DDP compatibility.
    clean_state = {}

    for key, value in state.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        clean_state[
            key
        ] = value

    missing, unexpected = (
        model.load_state_dict(
            clean_state,
            strict=False,
        )
    )

    # --------------------------------------------------------
    # Critical checks
    # --------------------------------------------------------

    critical_missing = [
        k
        for k in missing
        if (
            k.startswith(
                "refine_decoder."
            )
            or k.startswith(
                "encoder.texture_branch."
            )
        )
    ]

    if (
        use_refinement
        and critical_missing
    ):

        raise RuntimeError(
            "Checkpoint does not contain the "
            "trained enhanced refinement architecture.\n"
            "Missing examples:\n"
            + "\n".join(
                critical_missing[:20]
            )
        )

    model.to(
        device
    )

    model.eval()

    info = {

        "arch": {
            "encoder":
                encoder_name,

            "image_size":
                image_size,

            "inp_num":
                inp_num,

            "decoder_depth":
                decoder_depth,

            "residual":
                residual,

            "refinement":
                use_refinement,

            "map_fusion":
                use_map_fusion,
        },

        "args":
            cfg,

        "missing":
            missing,

        "unexpected":
            unexpected,
    }

    del ckpt

    return (
        model,
        info,
        image_size,
        encoder_name,
    )


# ============================================================
# Enhanced prediction
# ============================================================

@torch.no_grad()
def predict_views(
    model,
    images,
    image_size,
    tta_flip=False,
    use_amp=True,
):
    """
    images:
        [B,3,H,W]

    return:
        numpy [B,H,W]

    IMPORTANT:
        Uses the FINAL learned anomaly map, not the old
        anomaly_map_from_features().
    """

    with make_autocast(
        use_amp
    ):

        output = model(
            images,
            return_maps=True,
        )

        amap = output[
            "anomaly_map"
        ]

        if tta_flip:

            flipped = torch.flip(
                images,
                dims=[-1],
            )

            output_flip = model(
                flipped,
                return_maps=True,
            )

            amap_flip = output_flip[
                "anomaly_map"
            ]

            amap_flip = torch.flip(
                amap_flip,
                dims=[-1],
            )

            amap = (
                amap
                + amap_flip
            ) * 0.5

    if (
        amap.shape[-2:]
        != (
            image_size,
            image_size,
        )
    ):

        amap = torch.nn.functional.interpolate(
            amap,
            size=(
                image_size,
                image_size,
            ),
            mode="bilinear",
            align_corners=False,
        )

    return (
        amap[:, 0]
        .float()
        .cpu()
        .numpy()
    )


# ============================================================
# TopK view score
# ============================================================

def view_score(
    amap: np.ndarray,
    ratio: float,
) -> float:

    flat = amap.reshape(
        -1
    )

    ratio = min(
        max(
            float(ratio),
            1.0 / flat.size,
        ),
        1.0,
    )

    k = max(
        1,
        int(
            flat.size
            * ratio
        ),
    )

    if k >= flat.size:
        return float(
            flat.mean()
        )

    # Efficient Top-K mean.
    threshold_index = (
        flat.size - k
    )

    values = np.partition(
        flat,
        threshold_index,
    )[
        threshold_index:
    ]

    return float(
        values.mean()
    )


# ============================================================
# Physical-sample score
# ============================================================

def reduce_view_scores(
    scores,
    mode,
):

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    if mode == "max":

        return float(
            scores.max()
        )

    if mode == "mean":

        return float(
            scores.mean()
        )

    # LogSumExp-like smooth max.
    # Temperature deliberately moderate.
    temperature = 10.0

    m = float(
        scores.max()
    )

    return float(
        m
        + np.log(
            np.exp(
                (
                    scores
                    - m
                )
                * temperature
            ).mean()
            + 1e-12
        )
        / temperature
    )


# ============================================================
# Scale calibration
# ============================================================

def auto_scale(
    model,
    args,
    device,
):

    if not args.train_root:

        print(
            "[scale] no train-root; "
            "using scale=1.0 because the learned "
            "refinement map is already probability-like."
        )

        return 1.0

    ds = CSIGImageDataset(

        args.train_root,

        transform=build_transform(
            args.image_size,
            is_train=False,
        ),

        image_size=(
            args.image_size
        ),

        synthetic_anomaly=False,
    )

    n = min(
        args.calibrate_n,
        len(ds),
    )

    indices = np.linspace(
        0,
        len(ds) - 1,
        n,
    ).astype(
        int
    )

    maps = []

    batch = []

    model.eval()

    for i in tqdm(
        indices,
        desc="calibrate",
        ncols=80,
    ):

        image, _ = ds[
            int(i)
        ]

        batch.append(
            image
        )

        if (
            len(batch)
            >= max(
                1,
                args.samples_per_batch
                * 5,
            )
        ):

            x = torch.stack(
                batch,
                0,
            ).to(
                device
            )

            batch_maps = (
                predict_views(
                    model,
                    x,
                    args.image_size,
                    tta_flip=False,
                    use_amp=args.amp,
                )
            )

            maps.append(
                batch_maps
            )

            batch = []

    if batch:

        x = torch.stack(
            batch,
            0,
        ).to(
            device
        )

        maps.append(
            predict_views(
                model,
                x,
                args.image_size,
                False,
                args.amp,
            )
        )

    scale = calibrate_scale(
        maps,
        target_q=0.995,
        target_value=0.55,
    )

    print(
        f"[scale] auto mask_scale="
        f"{scale:.4f}"
    )

    return float(
        scale
    )


# ============================================================
# Normal statistics
# ============================================================

def collect_normal_stats(
    model,
    args,
    device,
):

    from collections import defaultdict

    ds = CSIGSampleDataset(

        args.train_root,

        transform=build_transform(
            args.image_size,
            is_train=False,
        ),

        image_size=(
            args.image_size
        ),
    )

    by_category = defaultdict(
        list
    )

    for i, (
        category,
        _sid,
        _path,
    ) in enumerate(
        ds.samples
    ):

        by_category[
            category
        ].append(
            i
        )

    stats = NormalStats(
        hw=64
    )

    model.eval()

    for category, indices in tqdm(
        sorted(
            by_category.items()
        ),
        desc="normal-stats",
        ncols=80,
    ):

        step = max(
            1,
            len(indices)
            // max(
                1,
                args.stats_per_class,
            ),
        )

        selected = indices[
            ::step
        ][
            :args.stats_per_class
        ]

        for index in selected:

            item = ds[
                index
            ]

            x = item[
                "images"
            ].to(
                device
            )

            maps = predict_views(
                model,
                x,
                args.image_size,
                tta_flip=False,
                use_amp=args.amp,
            )

            if args.sigma > 0:

                maps = smooth_map(
                    maps,
                    sigma=args.sigma,
                )

            for v in range(
                maps.shape[0]
            ):

                score = view_score(
                    maps[v],
                    args.max_ratio,
                )

                stats.update(
                    category,
                    v,
                    maps[v],
                    view_score=score,
                )

    return stats


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    os.environ.setdefault(
        "NCCL_P2P_DISABLE",
        "1",
    )

    os.environ.setdefault(
        "NCCL_IB_DISABLE",
        "1",
    )

    # --------------------------------------------------------
    # Peek architecture before distributed initialization
    # --------------------------------------------------------

    cfg = checkpoint_args(
        args.ckpt
    )

    encoder_name = (
        args.encoder
        or cfg.get(
            "encoder",
            "dinov2reg_vit_base_14",
        )
    )

    prefetch_encoder_weights(

        encoder_name,

        args.encoder_source,

        stamp_dir=str(
            Path(
                args.ckpt
            ).resolve().parent
            / "_prefetch"
        ),
    )

    # --------------------------------------------------------
    # Distributed
    # --------------------------------------------------------

    info = init_distributed()

    setup_seed(
        0,
        rank=info.rank,
    )

    device = info.device

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        model,
        load_info,
        image_size,
        encoder_name,
    ) = load_enhanced_model(
        args,
        device,
    )

    args.image_size = (
        image_size
    )

    if is_main(info):

        print(
            f"[ckpt] loaded "
            f"{args.ckpt}"
        )

        print(
            "[arch]",
            load_info[
                "arch"
            ],
        )

        noncritical_missing = [
            k
            for k in load_info[
                "missing"
            ]
            if not k.startswith(
                "encoder.model."
            )
        ]

        if noncritical_missing:

            print(
                "[ckpt] missing:",
                noncritical_missing[
                    :10
                ],
            )

        if load_info[
            "unexpected"
        ]:

            print(
                "[ckpt] unexpected:",
                load_info[
                    "unexpected"
                ][
                    :10
                ],
            )

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    need_stats = (
        bool(
            args.train_root
        )
        and (
            args.legacy_refine
            or args.view_gate
        )
    )

    stats = None

    checkpoint_tag = (
        Path(
            args.ckpt
        ).stem
    )

    default_stats = (
        Path(
            args.out_dir
        ).parent
        / (
            f"normal_stats_"
            f"{checkpoint_tag}_"
            f"s{args.sigma}.npz"
        )
    )

    stats_path = (
        Path(
            args.stats_path
        )
        if args.stats_path
        else default_stats
    )

    scale = (
        float(
            args.mask_scale
        )
        if args.mask_scale > 0
        else 0.0
    )

    success = torch.tensor(
        [1],
        device=device,
        dtype=torch.int32,
    )

    # Only rank 0 calculates calibration data.
    if is_main(info):

        try:

            if need_stats:

                if stats_path.is_file():

                    stats = NormalStats.load(
                        str(
                            stats_path
                        )
                    )

                    print(
                        f"[stats] loaded "
                        f"{stats_path}"
                    )

                else:

                    stats = collect_normal_stats(
                        model,
                        args,
                        device,
                    )

                    stats_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    stats.save(
                        str(
                            stats_path
                        )
                    )

                    print(
                        f"[stats] wrote "
                        f"{stats_path}"
                    )

            if scale <= 0:

                # Learned anomaly_map is probability-like.
                # Start with identity scaling when legacy
                # refinement is disabled.
                if not args.legacy_refine:

                    scale = 1.0

                else:

                    scale = auto_scale(
                        model,
                        args,
                        device,
                    )

            print(
                f"[scale] "
                f"mask_scale={scale:.4f} "
                f"legacy_refine="
                f"{args.legacy_refine}"
            )

        except Exception as exc:

            success[
                0
            ] = 0

            print(
                f"[calibration] failed: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # Synchronize calibration
    # --------------------------------------------------------

    if info.distributed:

        torch.distributed.all_reduce(

            success,

            op=(
                torch.distributed
                .ReduceOp.MIN
            ),
        )

        if int(
            success.item()
        ) == 0:

            raise RuntimeError(
                "Calibration failed"
            )

        scale_tensor = torch.tensor(
            [scale],
            device=device,
            dtype=torch.float32,
        )

        torch.distributed.broadcast(
            scale_tensor,
            src=0,
        )

        scale = float(
            scale_tensor.item()
        )

        barrier()

    # Other ranks load stats from file.
    if (
        need_stats
        and stats is None
        and stats_path.is_file()
    ):

        stats = NormalStats.load(
            str(
                stats_path
            )
        )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    dataset = CSIGSampleDataset(

        args.test_root,

        transform=build_transform(
            image_size,
            is_train=False,
        ),

        image_size=(
            image_size
        ),
    )

    sampler = (

        DistributedSampler(

            dataset,

            num_replicas=(
                info.world_size
            ),

            rank=(
                info.rank
            ),

            shuffle=False,

            drop_last=False,

        )

        if info.distributed

        else None
    )

    loader = DataLoader(

        dataset,

        batch_size=(
            args.samples_per_batch
        ),

        shuffle=False,

        sampler=sampler,

        num_workers=(
            args.num_workers
        ),

        pin_memory=(
            device.type
            == "cuda"
        ),

        drop_last=False,
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = Path(
        args.out_dir
    )

    if is_main(info):

        if output_dir.exists():

            shutil.rmtree(
                output_dir
            )

        (
            output_dir
            / "predicted_masks"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    barrier()

    (
        output_dir
        / "predicted_masks"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    local_rows = []

    n_views_total = 0
    n_views_zeroed = 0

    model.eval()

    iterator = (

        tqdm(
            loader,
            ncols=110,
            desc=f"infer-r{info.rank}",
        )

        if is_main(info)

        else loader
    )

    for batch in iterator:

        images = batch[
            "images"
        ].to(
            device,
            non_blocking=True,
        )

        batch_size = (
            images.shape[0]
        )

        num_views = (
            images.shape[1]
        )

        stacked = images.reshape(
            batch_size
            * num_views,
            *images.shape[
                2:
            ],
        )

        maps = predict_views(

            model,

            stacked,

            args.image_size,

            tta_flip=(
                args.tta_flip
            ),

            use_amp=(
                args.amp
            ),
        )

        maps = maps.reshape(

            batch_size,

            num_views,

            maps.shape[-2],

            maps.shape[-1],
        )

        folders = batch[
            "group_folder"
        ]

        categories = batch[
            "category"
        ]

        # ----------------------------------------------------
        # samples
        # ----------------------------------------------------

        for i in range(
            batch_size
        ):

            category = (
                categories[
                    i
                ]
            )

            raw_maps = maps[
                i
            ]

            # Learned refinement maps should first be evaluated
            # WITHOUT Gaussian smoothing.
            if args.sigma > 0:

                score_maps = smooth_map(
                    raw_maps,
                    sigma=args.sigma,
                )

            else:

                score_maps = (
                    raw_maps
                )

            raw_scores = [

                view_score(
                    score_maps[v],
                    args.max_ratio,
                )

                for v in range(
                    num_views
                )
            ]

            # Physical sample score.
            raw_image_score = (
                reduce_view_scores(
                    raw_scores,
                    args.reduce,
                )
            )

            # IMPORTANT:
            # squash exactly ONCE.
            image_score = squash_score(
                raw_image_score
            )

            # ------------------------------------------------
            # Pixel maps
            # ------------------------------------------------

            if (
                args.legacy_refine
                and stats is not None
            ):

                view_maps = apply_view_refine(

                    raw_maps,

                    images[i],

                    category,

                    stats,

                    sigma=(
                        args.sigma
                    ),

                    gamma=(
                        args.gamma
                    ),

                    border=(
                        args.border
                    ),

                    use_fg=(
                        args.fg_gate
                    ),
                )

            else:

                if args.sigma > 0:

                    view_maps = (
                        score_maps
                    )

                else:

                    view_maps = (
                        raw_maps
                    )

            # ------------------------------------------------
            # View gate
            # ------------------------------------------------

            if (
                args.view_gate
                and stats is not None
            ):

                (
                    view_maps,
                    gates,
                    _,
                ) = apply_view_gate(

                    view_maps,

                    category,

                    stats,

                    max_ratio=(
                        args.max_ratio
                    ),

                    k=args.gate_k,

                    temp=(
                        args.gate_temp
                    ),

                    floor=(
                        args.gate_floor
                    ),

                    scores=(
                        raw_scores
                    ),

                    hard=(
                        args.gate_hard
                    ),

                    mode=(
                        args.gate_mode
                    ),

                    margin=(
                        args.gate_margin
                    ),
                )

                n_views_total += len(
                    gates
                )

                n_views_zeroed += sum(
                    1
                    for gate in gates
                    if gate <= 0.0
                )

            # ------------------------------------------------
            # Save masks
            # ------------------------------------------------

            masks_uint8 = maps_to_uint8(

                view_maps,

                scale=scale,
            )

            group = folders[
                i
            ]

            destination = (
                output_dir
                / "predicted_masks"
                / group
            )

            destination.mkdir(
                parents=True,
                exist_ok=True,
            )

            for view_index in range(
                num_views
            ):

                Image.fromarray(
                    masks_uint8[
                        view_index
                    ],
                    mode="L",
                ).save(
                    destination
                    / (
                        f"{view_index}"
                        "_mask.png"
                    )
                )

            local_rows.append(
                (
                    group,
                    float(
                        image_score
                    ),
                )
            )

    # ========================================================
    # Gather results
    # ========================================================

    gathered_rows = (
        all_gather_object(
            local_rows
        )
    )

    gathered_gate = (
        all_gather_object(
            (
                n_views_zeroed,
                n_views_total,
            )
        )
    )

    # ========================================================
    # Submission
    # ========================================================

    if is_main(info):

        rows = []

        seen = set()

        for rank_rows in (
            gathered_rows
        ):

            for group, score in (
                rank_rows
            ):

                # DistributedSampler may pad samples.
                if group in seen:
                    continue

                seen.add(
                    group
                )

                rows.append(
                    (
                        group,
                        score,
                    )
                )

        rows.sort(
            key=lambda x: x[0]
        )

        csv_path = (
            output_dir
            / "submission.csv"
        )

        values = []

        with open(
            csv_path,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    "group_folder",
                    "anomaly_score",
                ]
            )

            for group, score in rows:

                # Already squashed once.
                score = min(
                    max(
                        float(score),
                        0.0,
                    ),
                    1.0,
                )

                values.append(
                    score
                )

                writer.writerow(
                    [
                        group,
                        f"{score:.8f}",
                    ]
                )

        if values:

            print(
                "[csv] anomaly_score "
                f"min={min(values):.6f} "
                f"max={max(values):.6f}"
            )

        zeroed = sum(
            x
            for x, _
            in gathered_gate
        )

        total_views = sum(
            x
            for _, x
            in gathered_gate
        )

        if total_views:

            print(
                f"[gate] zeroed "
                f"{zeroed}/"
                f"{total_views} views "
                f"({100*zeroed/total_views:.1f}%)"
            )

        metadata = {

            "test_root":
                args.test_root,

            "ckpt":
                args.ckpt,

            "image_size":
                args.image_size,

            "mask_scale":
                scale,

            "sigma":
                args.sigma,

            "max_ratio":
                args.max_ratio,

            "reduce":
                args.reduce,

            "tta_flip":
                args.tta_flip,

            "legacy_refine":
                args.legacy_refine,

            "view_gate":
                args.view_gate,

            "n_samples":
                len(rows),

            "world_size":
                info.world_size,

            "arch":
                load_info[
                    "arch"
                ],
        }

        (
            output_dir
            / "infer_meta.json"
        ).write_text(

            json.dumps(
                metadata,
                indent=2,
            ),

            encoding="utf-8",
        )

        zip_path = (
            zip_submission(
                str(
                    output_dir
                ),
                args.zip,
            )
        )

        print(
            f"[done] {zip_path} "
            f"({len(rows)} samples)"
        )

    barrier()


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    os.environ.setdefault(
        "NCCL_P2P_DISABLE",
        "1",
    )

    os.environ.setdefault(
        "NCCL_IB_DISABLE",
        "1",
    )

    try:

        main()

    finally:

        cleanup()
