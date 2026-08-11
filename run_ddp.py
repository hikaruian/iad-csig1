#!/usr/bin/env python3
"""
DDP (multi-GPU, one process per GPU) entry point.

Usage:
    torchrun --nproc_per_node=4 run_ddp.py --data CSIG --out submission

Each rank:
  1. Initialises DDP and sets its GPU.
  2. Builds the FULL memory bank on ALL training data (banks are small --
     ~200 MB for 50 classes × 16k × 1024-d -- so every rank gets an identical
     copy; this guarantees bit-for-bit consistent scores).
  3. Runs the full CSIGAnomalyPipeline inference on its SHARD of test samples
     using ``DistributedSampler``; each rank writes its own mask PNGs and a
     CSV shard ``_shard_<rank>.csv``.
  4. Rank 0 gathers CSV shards, sorts them deterministically, and zips the
     submission directory.

Accuracy is BIT-FOR-IDENTICAL to ``run.py`` because every rank runs the same
pipeline code path (TTA, CLIP branch, multiview vote, calibration).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import CSIGAnomalyPipeline, PipelineConfig, _imagenet_to_clip
from src.dataset import (
    CSIGSampleDataset,
    build_test_transform,
)
from src.utils import save_mask_png, save_submission_csv
from src.dist_utils import ddp_setup, ddp_cleanup, is_main_process, barrier


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="CSIG")
    ap.add_argument("--out", type=str, default="submission")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--dinov2", type=str, default=None)
    ap.add_argument("--no-clip", action="store_true")
    return ap.parse_args()


def load_yaml_cfg(path):
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()
    ddp_setup()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    is_main = is_main_process()
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42)
    np.random.seed(42)

    cfg = PipelineConfig()
    cfg.train_root = str(Path(args.data) / "Train")
    cfg.test_root = str(Path(args.data) / "Test_A")
    cfg.out_dir = args.out
    cfg.device = f"cuda:{local_rank}"
    cfg.use_dp = False  # Never auto-parallel inside DDP
    if args.config is not None:
        for k, v in load_yaml_cfg(args.config).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    if args.dinov2:
        cfg.dinov2_model = args.dinov2
    if args.no_clip:
        cfg.use_clip = False

    out_dir = Path(cfg.out_dir)
    mask_dir = out_dir / "predicted_masks"
    if is_main:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Build pipeline + memory banks + calibrator (every rank builds the full
    # bank so scores are consistent across ranks).
    if is_main:
        print(f"[ddp rank0] world_size={world_size}, device={cfg.device}")
    pipe = CSIGAnomalyPipeline(cfg)
    pipe.fit(cfg.train_root)
    barrier()

    # Shard test samples across ranks
    test_root = Path(cfg.test_root)
    test_classes = sorted([d.name for d in test_root.iterdir() if d.is_dir()])
    tfm = build_test_transform(cfg.input_size, tta="none")
    test_ds = CSIGSampleDataset(test_root, transform=tfm, classes=test_classes)
    sampler = DistributedSampler(
        test_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    test_ld = DataLoader(
        test_ds, batch_size=1, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    if is_main:
        print(f"[ddp rank0] Running inference on {len(test_ds)} samples "
              f"(~{len(test_ds)//world_size} per rank)...")

    local_rows = []
    for batch in tqdm(test_ld, desc=f"rank{rank} predict", disable=not is_main):
        views = batch["views"].squeeze(0).to(pipe.device)
        cls = batch["cls_name"][0]
        sid = batch["sample_id"][0]
        gf = batch["group_folder"][0]

        views_clip = None
        if pipe.clip is not None:
            # Same CLIP prep as pipeline.predict_and_save: keep 448px
            # resolution so CLIP's patch grid is pixel-registered with
            # DINOv2's (both 32x32), giving a better mask ensemble.
            views_clip = _imagenet_to_clip(views)

        if cls in pipe.patchcore.banks:
            score, masks = pipe._predict_one_sample(views, views_clip, cls)
        else:
            score, masks = pipe._predict_zeroshot(views, views_clip, cls)

        local_rows.append((gf, float(score)))

        sample_mask_dir = mask_dir / cls / sid
        sample_mask_dir.mkdir(parents=True, exist_ok=True)
        for v in range(masks.shape[0]):
            save_mask_png(
                masks[v], sample_mask_dir / f"{v}_mask.png",
                target_size=cfg.input_size, global_lo=0.0, global_hi=1.0,
            )

    barrier()

    # Write CSV shard (all ranks write; filesystem is assumed shared, which
    # is the case for NFS / local SSD with torchrun --nnodes=1).
    shard_path = out_dir / f"_shard_{rank}.csv"
    with open(shard_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_folder", "anomaly_score"])
        for gf, sc in local_rows:
            w.writerow([gf, f"{float(sc):.6f}"])
    barrier()

    if is_main:
        all_rows = []
        for r in range(world_size):
            sp = out_dir / f"_shard_{r}.csv"
            if not sp.exists():
                continue
            with open(sp, newline="") as f:
                rd = csv.reader(f)
                next(rd)
                for row in rd:
                    if len(row) >= 2:
                        all_rows.append((row[0], float(row[1])))
        all_rows.sort(key=lambda x: x[0])
        save_submission_csv(all_rows, out_dir / "submission.csv")

        for r in range(world_size):
            sp = out_dir / f"_shard_{r}.csv"
            if sp.exists():
                sp.unlink()

        zip_path = out_dir.parent / "submission.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out_dir.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(out_dir))
        print(f"[ddp rank0] Done. Wrote {out_dir/'submission.csv'} and {zip_path}")

    barrier()
    ddp_cleanup()


if __name__ == "__main__":
    main()
