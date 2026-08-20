#!/usr/bin/env python3
"""
Main entry point: train memory banks on Train/, then predict on Test_A/ and
produce a ready-to-submit zip.

Examples:
    # Default (DINOv2-L/14 + CLIP ensemble)
    python run.py --data CSIG --out submission

    # Fast ablation (DINOv2-B/14, no CLIP)
    python run.py --data CSIG --out submission_quick --config configs/quick.yaml

    # CPU-only (very slow, for debugging only)
    python run.py --data CSIG --out submission --device cpu
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from dataclasses import fields
from pathlib import Path

import yaml

# --- Clear stale bytecode caches so updated .py files are always picked up ---
# (in long-running notebook/Kaggle sessions, Python may otherwise reuse old
# .pyc files from __pycache__, causing dtype bugs like "expected mat1 and mat2
# to have the same dtype" even after the source is fixed).
for _p in Path(__file__).resolve().parent.rglob("__pycache__"):
    shutil.rmtree(_p, ignore_errors=True)

from src.pipeline import CSIGAnomalyPipeline, PipelineConfig  # noqa: E402


def load_yaml_cfg(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_into_cfg(cfg: PipelineConfig, overrides: dict) -> PipelineConfig:
    from typing import get_type_hints, get_origin, get_args
    hints = get_type_hints(PipelineConfig)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            continue
        # Auto-coerce list -> tuple when the dataclass field expects a tuple
        # (YAML parses [...] as list; dataclass annotation is Tuple[...]).
        anno = hints.get(k, None)
        if isinstance(v, list) and anno is not None:
            origin = get_origin(anno)
            if origin is tuple:
                v = tuple(v)
        setattr(cfg, k, v)
    return cfg


def zip_submission(out_dir: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out_dir))
    print(f"[zip] Wrote {zip_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="CSIG",
                    help="Root folder containing Train/ and Test_A/")
    ap.add_argument("--out", type=str, default="submission")
    ap.add_argument("--config", type=str, default=None,
                    help="Optional YAML config override")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda | cpu (default: auto)")
    ap.add_argument("--no-zip", action="store_true",
                    help="Do not produce submission.zip")
    ap.add_argument("--dinov2", type=str, default=None,
                    help="Override DINOv2 size: vitb14|vitl14|vitg14")
    ap.add_argument("--no-clip", action="store_true",
                    help="Disable CLIP/WinCLIP ensemble branch")
    ap.add_argument("--dp", action="store_true",
                    help="Enable DataParallel across all visible GPUs (single-node, simple)")
    ap.add_argument("--no-amp", action="store_true",
                    help="Disable fp16 autocast (slower, uses more VRAM, more precise)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override DINO batch size (lower this if you still OOM)")
    ap.add_argument("--banks-on-gpu", action="store_true", default=None,
                    help="Prefetch all coreset banks to GPU (faster, uses ~3.2 GB more VRAM)")
    ap.add_argument("--coreset-on-cpu", action="store_true",
                    help="Run coreset selection on CPU (slower, uses ~0.5 GB less GPU temp)")
    ap.add_argument("--dino-micro", type=int, default=None,
                    help="Override DINO inference micro-batch size (views per forward)")
    args = ap.parse_args()

    cfg = PipelineConfig()
    cfg.train_root = str(Path(args.data) / "Train")
    cfg.test_root = str(Path(args.data) / "Test_A")
    cfg.out_dir = args.out

    if args.config is not None:
        overrides = load_yaml_cfg(args.config)
        known = {f.name for f in fields(PipelineConfig)}
        unknown = set(overrides) - known
        if unknown:
            print(f"[warn] Ignoring unknown config keys: {sorted(unknown)}")
        cfg = merge_into_cfg(cfg, overrides)
    if args.device:
        cfg.device = args.device
    if args.dinov2:
        cfg.dinov2_model = args.dinov2
    if args.no_clip:
        cfg.use_clip = False
    if args.dp:
        cfg.use_dp = True
    if args.no_amp:
        cfg.use_amp = False
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.banks_on_gpu:
        cfg.banks_on_gpu = True
    if args.coreset_on_cpu:
        cfg.coreset_on_gpu = False
    if args.dino_micro is not None:
        cfg.dino_view_micro = args.dino_micro

    out_dir = Path(cfg.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] Config: {cfg}")
    pipe = CSIGAnomalyPipeline(cfg)
    pipe.fit(cfg.train_root)
    csv_path, mask_dir = pipe.predict_and_save(cfg.test_root, out_dir)

    if not args.no_zip:
        zip_submission(out_dir, out_dir.parent / "submission.zip")

    print("[main] Done. Ready to submit.")


if __name__ == "__main__":
    sys.exit(main())
