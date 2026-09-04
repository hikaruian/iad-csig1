#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.data import (
    CSIGImageDataset,
    build_transform,
)

from src.dist_utils import (
    barrier,
    cleanup,
    init_distributed,
    is_main,
    make_autocast,
    make_scaler,
    reduce_mean,
    setup_seed,
    unwrap,
)

from src.encoder import (
    prefetch_encoder_weights,
)

from src.losses import (
    pixel_anomaly_loss,
    total_loss,
)

from src.model import build_model

from src.optim import (
    StableAdamW,
    WarmCosineScheduler,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--train-root",
        required=True,
    )

    p.add_argument(
        "--save-dir",
        default="runs/inpformer_v3",
    )

    p.add_argument(
        "--encoder",
        default="dinov2reg_vit_base_14",
    )

    p.add_argument(
        "--encoder-source",
        default="auto",
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
        default=0,
    )

    p.add_argument(
        "--residual",
        action="store_true",
    )

    p.add_argument(
        "--grad-checkpoint",
        action="store_true",
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    p.add_argument(
        "--grad-accum",
        type=int,
        default=4,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
    )

    p.add_argument(
        "--gather-weight",
        type=float,
        default=0.2,
    )

    p.add_argument(
        "--soft-y",
        type=float,
        default=3,
    )

    p.add_argument(
        "--synthetic-prob",
        type=float,
        default=0.8,
    )

    p.add_argument(
        "--pixel-focal-weight",
        type=float,
        default=1,
    )

    p.add_argument(
        "--pixel-dice-weight",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--hard-negative-weight",
        type=float,
        default=0.03,
    )

    p.add_argument(
        "--boundary-weight",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--pixel-warmup-epochs",
        type=int,
        default=10,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--resume",
        default="",
    )

    p.add_argument(
        "--print-freq",
        type=int,
        default=20,
    )

    return p.parse_args()


def log(
    message,
    info,
    save_dir=None,
):
    if not is_main(info):
        return

    print(
        message,
        flush=True,
    )

    if save_dir:
        with open(
            save_dir / "train.log",
            "a",
        ) as file:
            file.write(
                message + "\n"
            )


def main():
    args = parse_args()

    if args.image_size % 14:
        raise ValueError(
            "image-size must be divisible by 14"
        )

    prefetch_encoder_weights(
        args.encoder,
        args.encoder_source,
        str(
            Path(args.save_dir)
            / "_prefetch"
        ),
    )

    info = init_distributed()

    setup_seed(
        args.seed,
        rank=info.rank,
    )

    device = info.device

    save_dir = Path(
        args.save_dir
    )

    if is_main(info):
        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            save_dir
            / "args.json"
        ).write_text(
            json.dumps(
                vars(args),
                indent=2,
            )
        )

    barrier()

    dataset = CSIGImageDataset(
        args.train_root,
        image_size=args.image_size,
        synthetic_anomaly=True,
        synthetic_prob=args.synthetic_prob,
    )

    sampler = (
        DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
        )
        if info.distributed
        else None
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    model = build_model(
        encoder_name=args.encoder,
        inp_num=args.inp_num,
        decoder_depth=args.decoder_depth,
        bottleneck_drop=args.bottleneck_drop,
        residual=args.residual,
        encoder_source=args.encoder_source,
        grad_checkpoint=args.grad_checkpoint,
        use_learnable_map_fusion=False,
        use_refinement=True,
    ).to(device)

    # Do NOT freeze model.encoder here.
    # DINO is already frozen internally.
    # texture_branch must stay trainable.

    if info.distributed:
        model = DDP(
            model,
            device_ids=[
                info.local_rank
            ],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    raw = unwrap(
        model
    )

    trainable = list(
        raw.trainable_parameters()
    )

    optimizer = StableAdamW(
        [{"params": trainable}],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        amsgrad=True,
        eps=1e-10,
    )

    accum = max(
        1,
        args.grad_accum,
    )

    opt_steps = (
        len(loader)
        + accum
        - 1
    ) // accum

    scheduler = WarmCosineScheduler(
        optimizer,
        base_value=args.lr,
        final_value=args.min_lr,
        total_iters=(
            args.epochs
            * opt_steps
        ),
        warmup_iters=min(
            100,
            max(
                10,
                opt_steps,
            ),
        ),
    )

    scaler = make_scaler(
        args.amp
    )

    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        from src.checkpoint import torch_load
        from src.dist_utils import strip_module_prefix

        ckpt = torch_load(
            args.resume,
            map_location="cpu",
        )

        raw.load_state_dict(
            strip_module_prefix(
                ckpt["model"]
            ),
            strict=False,
        )

        start_epoch = ckpt.get(
            "epoch",
            0,
        )

    for epoch in range(
        start_epoch,
        args.epochs,
    ):
        model.train()

        # Wrapper.train() already forces only DINO model to eval.
        raw.encoder.model.eval()

        if sampler:
            sampler.set_epoch(epoch)

        pixel_scale = min(
            1,
            (
                epoch + 1
            )
            / max(
                1,
                args.pixel_warmup_epochs,
            ),
        )

        running = []

        optimizer.zero_grad(
            set_to_none=True,
        )

        iterator = (
            tqdm(
                loader,
                ncols=130,
                desc=(
                    f"{epoch+1}/"
                    f"{args.epochs}"
                ),
            )
            if is_main(info)
            else loader
        )

        start = time.time()

        for step, batch in enumerate(
            iterator
        ):
            clean = batch[
                "clean"
            ].to(
                device,
                non_blocking=True,
            )

            synthetic = batch[
                "synthetic"
            ].to(
                device,
                non_blocking=True,
            )

            mask = batch[
                "mask"
            ].to(
                device,
                non_blocking=True,
            )

            do_step = (
                (step + 1)
                % accum
                == 0
                or step + 1
                == len(loader)
            )

            sync = (
                model.no_sync()
                if (
                    info.distributed
                    and not do_step
                )
                else nullcontext()
            )

            with sync:
                # Clean INP branch
                with make_autocast(
                    args.amp
                ):
                    clean_out = model(
                        clean,
                        return_maps=True,
                    )

                    inp_loss = total_loss(
                        clean_out["en"],
                        clean_out["de"],
                        clean_out["g_loss"],
                        args.gather_weight,
                        args.soft_y,
                    )

                scaler.scale(
                    inp_loss / accum
                ).backward()

                # Synthetic pixel branch
                with make_autocast(
                    args.amp
                ):
                    syn_out = model(
                        synthetic,
                        return_maps=True,
                    )

                    losses = (
                        pixel_anomaly_loss(
                            syn_out[
                                "refine_logits"
                            ],
                            mask,
                            focal_weight=(
                                args.pixel_focal_weight
                                * pixel_scale
                            ),
                            dice_weight=(
                                args.pixel_dice_weight
                                * pixel_scale
                            ),
                            hard_negative_weight=(
                                args.hard_negative_weight
                                * pixel_scale
                            ),
                            boundary_weight=(
                                args.boundary_weight
                                * pixel_scale
                            ),
                        )
                    )

                    pixel_loss = (
                        losses["total"]
                    )

                scaler.scale(
                    pixel_loss / accum
                ).backward()

            total = (
                inp_loss.detach()
                + pixel_loss.detach()
            )

            if do_step:
                scaler.unscale_(
                    optimizer
                )

                nn.utils.clip_grad_norm_(
                    trainable,
                    0.1,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

                scheduler.step()

            reduced = reduce_mean(
                total
            )

            running.append(
                reduced.item()
            )

            if (
                is_main(info)
                and hasattr(
                    iterator,
                    "set_postfix",
                )
                and step
                % args.print_freq
                == 0
            ):
                iterator.set_postfix({
                    "loss":
                        f"{reduced.item():.3f}",
                    "inp":
                        f"{inp_loss.item():.3f}",
                    "focal":
                        f"{losses['focal'].item():.3f}",
                    "dice":
                        f"{losses['dice'].item():.3f}",
                    "lr":
                        f"{optimizer.param_groups[0]['lr']:.2e}",
                })

        mean_loss = float(
            np.mean(running)
        )

        log(
            f"epoch={epoch+1} "
            f"loss={mean_loss:.4f} "
            f"time={time.time()-start:.1f}s",
            info,
            save_dir,
        )

        if is_main(info):
            checkpoint_data = {
                "epoch": epoch + 1,
                "model": raw.state_dict(),
                "trainable": (
                    raw.trainable_state_dict()
                ),
                "optimizer": (
                    optimizer.state_dict()
                ),
                "scheduler": (
                    scheduler.state_dict()
                ),
                "scaler": (
                    scaler.state_dict()
                ),
                "args": vars(args),
                "loss": mean_loss,
                "best_loss": min(
                    best_loss,
                    mean_loss,
                ),
            }

            #torch.save(checkpoint_data,save_dir / "last.pth")

            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(checkpoint_data,save_dir / "best.pth")

        barrier()


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
