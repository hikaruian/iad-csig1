#!/usr/bin/env python3
"""
Diagnose seen-vs-unseen category shift without ground-truth labels.

No submission is generated.

The script compares INP reconstruction-error distributions between:

    seen categories:
        categories contained in src.data.SEEN_CATEGORIES

    unseen categories:
        all other categories in Test

Useful evidence for class shift:
    1. unseen image median anomaly is systematically higher
    2. unseen low-quantile anomaly baseline is higher
    3. unseen map-wide mean anomaly is higher
    4. topK increases together with map median rather than local contrast

Example
-------
CUDA_VISIBLE_DEVICES=0 python diagnose_class_shift.py \
    --test-root /data/CSIG/Test_ALL \
    --ckpt runs/inpformer/best.pth \
    --out outputs/class_shift_report.json \
    --image-size 448 \
    --batch-size 8 \
    --amp
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.checkpoint import load_inpformer, torch_load

from src.data import (
    CSIGSampleDataset,
    SEEN_CATEGORIES,
    build_transform,
)

from src.dist_utils import (
    cleanup,
    init_distributed,
    is_main,
    make_autocast,
    setup_seed,
)

from src.encoder import prefetch_encoder_weights

from src.model import anomaly_map_from_features


# ============================================================
# arguments
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="Diagnose INP class shift"
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
        "--out",
        type=str,
        default="outputs/class_shift_report.json",
    )

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
        "--batch-size",
        type=int,
        default=8,
        help="Number of individual views per forward.",
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--topk-ratio",
        type=float,
        default=0.01,
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

    return p.parse_args()


# ============================================================
# numeric helpers
# ============================================================

def safe_stats(values):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if x.size == 0:

        return {
            "n": 0,
        }

    return {
        "n":
            int(x.size),

        "mean":
            float(x.mean()),

        "std":
            float(x.std()),

        "median":
            float(np.median(x)),

        "q10":
            float(np.quantile(x, 0.10)),

        "q25":
            float(np.quantile(x, 0.25)),

        "q75":
            float(np.quantile(x, 0.75)),

        "q90":
            float(np.quantile(x, 0.90)),
    }


def cohen_d(
    a,
    b,
):
    """
    Standardized difference:

        positive d:
            unseen > seen
    """

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    a = a[
        np.isfinite(a)
    ]

    b = b[
        np.isfinite(b)
    ]

    if (
        a.size < 2
        or b.size < 2
    ):

        return 0.0

    va = a.var(
        ddof=1
    )

    vb = b.var(
        ddof=1
    )

    pooled = math.sqrt(
        (
            (
                a.size - 1
            ) * va
            + (
                b.size - 1
            ) * vb
        )
        / (
            a.size
            + b.size
            - 2
        )
    )

    if pooled < 1e-12:
        return 0.0

    # b = unseen, a = seen
    return float(
        (
            b.mean()
            - a.mean()
        )
        / pooled
    )


# ============================================================
# map statistics
# ============================================================

def compute_map_statistics(
    amap,
    topk_ratio,
):
    """
    amap:
        [B,1,H,W]

    returns one dictionary per image/view.
    """

    flat = amap.float().flatten(
        1
    )

    n = flat.shape[
        1
    ]

    k = max(
        1,
        int(
            n
            * topk_ratio
        ),
    )

    k = min(
        k,
        n,
    )

    mean = flat.mean(
        dim=1
    )

    median = flat.median(
        dim=1
    ).values

    maximum = flat.max(
        dim=1
    ).values

    topk = torch.topk(
        flat,
        k=k,
        dim=1,
    ).values.mean(
        dim=1
    )

    deviation = (
        flat
        - median.unsqueeze(
            1
        )
    ).abs()

    mad = deviation.median(
        dim=1
    ).values

    robust_scale = (
        1.4826
        * mad
    ).clamp_min(
        1e-6
    )

    contrast = (
        topk
        - median
    )

    robust_contrast = (
        contrast
        / robust_scale
    )

    result = []

    for i in range(
        flat.shape[0]
    ):

        result.append({
            "mean":
                float(
                    mean[i].item()
                ),

            "median":
                float(
                    median[i].item()
                ),

            "mad":
                float(
                    mad[i].item()
                ),

            "max":
                float(
                    maximum[i].item()
                ),

            "topk":
                float(
                    topk[i].item()
                ),

            "contrast":
                float(
                    contrast[i].item()
                ),

            "robust_contrast":
                float(
                    robust_contrast[
                        i
                    ].item()
                ),
        })

    return result


# ============================================================
# prediction
# ============================================================

@torch.no_grad()
def predict(
    model,
    images,
    image_size,
    amp,
):

    with make_autocast(
        amp
    ):

        en, de, _ = model(
            images
        )

        amap = (
            anomaly_map_from_features(
                en,
                de,
                out_size=image_size,
            )
        )

    return amap.float()


# ============================================================
# main
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
    # This diagnostic should run on one GPU.
    # --------------------------------------------------------

    info = init_distributed()

    if info.world_size != 1:

        raise RuntimeError(
            "diagnose_class_shift.py should "
            "run on one GPU only."
        )

    setup_seed(
        0,
        rank=0,
    )

    device = info.device

    # --------------------------------------------------------
    # Read checkpoint architecture
    # --------------------------------------------------------

    checkpoint = torch_load(
        args.ckpt,
        map_location="cpu",
    )

    checkpoint_args = (
        checkpoint.get(
            "args",
            {},
        )
        if isinstance(
            checkpoint,
            dict,
        )
        else {}
    )

    encoder_name = (
        args.encoder
        or checkpoint_args.get(
            "encoder",
            "dinov2reg_vit_base_14",
        )
    )

    image_size = int(
        checkpoint_args.get(
            "image_size",
            args.image_size,
        )
    )

    del checkpoint

    prefetch_encoder_weights(
        encoder_name,
        args.encoder_source,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, load_info = (
        load_inpformer(
            args.ckpt,
            device,
            encoder=(
                args.encoder
                or None
            ),
            encoder_source=(
                args.encoder_source
            ),
            inp_num=(
                args.inp_num
            ),
            decoder_depth=(
                args.decoder_depth
            ),
            bottleneck_drop=(
                args.bottleneck_drop
            ),
            residual=(
                args.residual
            ),
        )
    )

    model.eval()

    print(
        f"[model] {encoder_name}"
    )

    print(
        f"[image_size] {image_size}"
    )

    print(
        f"[known seen categories] "
        f"{len(SEEN_CATEGORIES)}"
    )

    # --------------------------------------------------------
    # Dataset
    #
    # One physical sample returns 5 views.
    # --------------------------------------------------------

    dataset = CSIGSampleDataset(

        args.test_root,

        transform=build_transform(
            image_size,
            is_train=False,
        ),

        image_size=image_size,
    )

    # --------------------------------------------------------
    # Flatten views manually.
    #
    # DataLoader sample batch:
    #     [B,5,3,H,W]
    # --------------------------------------------------------

    loader = DataLoader(

        dataset,

        batch_size=max(
            1,
            args.batch_size
            // 5,
        ),

        shuffle=False,

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
    # Per-category observations
    # --------------------------------------------------------

    category_values = defaultdict(
        lambda: defaultdict(
            list
        )
    )

    sample_records = []

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    for batch in tqdm(
        loader,
        desc="diagnose",
        ncols=100,
    ):

        images = batch[
            "images"
        ]

        categories = batch[
            "category"
        ]

        folders = batch[
            "group_folder"
        ]

        b, views = images.shape[
            :2
        ]

        images = images.reshape(
            b * views,
            *images.shape[2:],
        ).to(
            device,
            non_blocking=True,
        )

        maps = predict(
            model,
            images,
            image_size,
            args.amp,
        )

        statistics = (
            compute_map_statistics(
                maps,
                args.topk_ratio,
            )
        )

        # ----------------------------------------------------
        # Restore physical-sample grouping.
        # ----------------------------------------------------

        for i in range(
            b
        ):

            category = (
                categories[i]
            )

            group = (
                folders[i]
            )

            seen = (
                category
                in SEEN_CATEGORIES
            )

            view_records = []

            for view in range(
                views
            ):

                index = (
                    i * views
                    + view
                )

                record = (
                    statistics[
                        index
                    ]
                )

                view_records.append(
                    record
                )

                for key, value in (
                    record.items()
                ):

                    category_values[
                        category
                    ][
                        key
                    ].append(
                        value
                    )

            # Physical-sample aggregate.
            sample_median = float(
                np.mean([
                    x["median"]
                    for x
                    in view_records
                ])
            )

            sample_mean = float(
                np.mean([
                    x["mean"]
                    for x
                    in view_records
                ])
            )

            sample_topk = float(
                np.max([
                    x["topk"]
                    for x
                    in view_records
                ])
            )

            sample_contrast = float(
                np.max([
                    x["contrast"]
                    for x
                    in view_records
                ])
            )

            sample_robust = float(
                np.max([
                    x[
                        "robust_contrast"
                    ]
                    for x
                    in view_records
                ])
            )

            sample_records.append({

                "group_folder":
                    group,

                "category":
                    category,

                "seen":
                    seen,

                "median":
                    sample_median,

                "mean":
                    sample_mean,

                "topk":
                    sample_topk,

                "contrast":
                    sample_contrast,

                "robust_contrast":
                    sample_robust,
            })

    # ========================================================
    # Category aggregation
    # ========================================================

    category_report = {}

    for category, metrics in sorted(
        category_values.items()
    ):

        category_report[
            category
        ] = {

            "seen":
                category
                in SEEN_CATEGORIES,

        }

        for key, values in (
            metrics.items()
        ):

            category_report[
                category
            ][key] = safe_stats(
                values
            )

    # ========================================================
    # IMPORTANT:
    # Compare CATEGORY-LEVEL central tendency, not every view.
    #
    # Otherwise a category containing more samples receives
    # greater weight.
    # ========================================================

    metrics = (
        "mean",
        "median",
        "topk",
        "contrast",
        "robust_contrast",
    )

    comparison = {}

    for metric in metrics:

        seen_values = []

        unseen_values = []

        for category, record in (
            category_report.items()
        ):

            value = (
                record[
                    metric
                ][
                    "median"
                ]
            )

            if record[
                "seen"
            ]:

                seen_values.append(
                    value
                )

            else:

                unseen_values.append(
                    value
                )

        comparison[
            metric
        ] = {

            "seen":
                safe_stats(
                    seen_values
                ),

            "unseen":
                safe_stats(
                    unseen_values
                ),

            "unseen_minus_seen":
                float(
                    np.mean(
                        unseen_values
                    )
                    - np.mean(
                        seen_values
                    )
                )
                if (
                    seen_values
                    and unseen_values
                )
                else None,

            "unseen_over_seen":
                float(
                    np.mean(
                        unseen_values
                    )
                    / max(
                        np.mean(
                            seen_values
                        ),
                        1e-12,
                    )
                )
                if (
                    seen_values
                    and unseen_values
                )
                else None,

            "cohen_d":
                cohen_d(
                    seen_values,
                    unseen_values,
                )
                if (
                    seen_values
                    and unseen_values
                )
                else None,
        }

    # ========================================================
    # Count category classification
    # ========================================================

    test_categories = sorted(
        category_report.keys()
    )

    seen_test_categories = [
        x
        for x in test_categories
        if x in SEEN_CATEGORIES
    ]

    unseen_test_categories = [
        x
        for x in test_categories
        if x not in SEEN_CATEGORIES
    ]

    result = {

        "checkpoint":
            args.ckpt,

        "test_root":
            args.test_root,

        "image_size":
            image_size,

        "topk_ratio":
            args.topk_ratio,

        "n_test_categories":
            len(
                test_categories
            ),

        "n_seen_categories":
            len(
                seen_test_categories
            ),

        "n_unseen_categories":
            len(
                unseen_test_categories
            ),

        "seen_categories":
            seen_test_categories,

        "unseen_categories":
            unseen_test_categories,

        "comparison":
            comparison,

        "category_statistics":
            category_report,

        "samples":
            sample_records,
    }

    # ========================================================
    # Output
    # ========================================================

    output = Path(
        args.out
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Human-readable summary
    # ========================================================

    print()
    print("=" * 72)
    print("SEEN vs UNSEEN CATEGORY SHIFT")
    print("=" * 72)

    print(
        f"test categories : "
        f"{len(test_categories)}"
    )

    print(
        f"seen categories : "
        f"{len(seen_test_categories)}"
    )

    print(
        f"unseen categories: "
        f"{len(unseen_test_categories)}"
    )

    print()

    for metric in metrics:

        r = comparison[
            metric
        ]

        if (
            r["seen"]["n"] == 0
            or r["unseen"]["n"]
            == 0
        ):

            continue

        seen_mean = (
            r["seen"][
                "mean"
            ]
        )

        unseen_mean = (
            r["unseen"][
                "mean"
            ]
        )

        ratio = (
            r[
                "unseen_over_seen"
            ]
        )

        d = (
            r[
                "cohen_d"
            ]
        )

        print(
            f"{metric:18s} "
            f"seen={seen_mean:.6f}  "
            f"unseen={unseen_mean:.6f}  "
            f"ratio={ratio:.3f}  "
            f"d={d:+.3f}"
        )

    print()
    print(
        f"Full report: {output}"
    )

    # ========================================================
    # Automatic interpretation
    # ========================================================

    median_result = (
        comparison[
            "median"
        ]
    )

    if (
        median_result["cohen_d"]
        is not None
    ):

        d = (
            median_result[
                "cohen_d"
            ]
        )

        ratio = (
            median_result[
                "unseen_over_seen"
            ]
        )

        print()

        if (
            d >= 0.8
            and ratio >= 1.2
        ):

            print(
                "[diagnosis] STRONG evidence of "
                "category-level reconstruction shift."
            )

        elif (
            d >= 0.5
            and ratio >= 1.1
        ):

            print(
                "[diagnosis] MODERATE evidence of "
                "category-level reconstruction shift."
            )

        elif d > 0.2:

            print(
                "[diagnosis] WEAK evidence of "
                "category-level reconstruction shift."
            )

        else:

            print(
                "[diagnosis] No clear global-offset "
                "class shift detected."
            )


if __name__ == "__main__":

    try:

        main()

    finally:

        cleanup()
