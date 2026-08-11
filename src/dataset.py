"""
CSIG / Real-IAD Variety dataset loader.

Directory layout
----------------
CSIG/
├── Train/<class>/SXXXX/{0..4}.png
└── Test_A/<class>/SXXXX/{0..4}.png

Notes:
    * Train contains NORMAL samples ONLY (unsupervised AD setting).
    * Test_A contains normal + defect samples; labels are NOT provided
      (inference-only).
    * Images are RGB PNGs of variable size (400x400 up to 4400x4400).
    * Each physical sample is represented by 5 camera views (0..4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Normalisation stats
# ---------------------------------------------------------------------------
# ImageNet stats (used for DINOv2 and for general ViT backbones)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CLIP uses its own mean/std (different from ImageNet)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_train_transform(input_size: int = 448,
                           norm: str = "imagenet") -> transforms.Compose:
    """
    Training / feature extraction transform.
    norm: "imagenet" for DINOv2, "clip" for OpenCLIP.
    """
    mean = IMAGENET_MEAN if norm == "imagenet" else CLIP_MEAN
    std = IMAGENET_STD if norm == "imagenet" else CLIP_STD
    return transforms.Compose([
        transforms.Resize((input_size, input_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_test_transform(input_size: int = 448, tta: str = "none",
                         norm: str = "imagenet") -> transforms.Compose:
    """
    Inference transform.
    norm: "imagenet" for DINOv2, "clip" for CLIP.
    """
    mean = IMAGENET_MEAN if norm == "imagenet" else CLIP_MEAN
    std = IMAGENET_STD if norm == "imagenet" else CLIP_STD

    ops: List[Callable] = [
        transforms.Resize((input_size, input_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if tta in ("hflip", "hvflip"):
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    if tta in ("vflip", "hvflip"):
        ops.append(transforms.RandomVerticalFlip(p=1.0))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------
class CSIGImageDataset(Dataset):
    """
    Image-level dataset: returns one image at a time.
    Used for (a) building per-class memory banks and (b) extracting dense
    patch features during inference.
    """

    def __init__(self, root: str | Path, transform: Optional[Callable] = None,
                 classes: Optional[List[str]] = None):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()]) \
            if classes is None else classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[Path, int, str, str]] = []  # img, cls_idx, cls, sample_id
        for cls in self.classes:
            cls_path = self.root / cls
            if not cls_path.exists():
                continue
            for sample_dir in sorted(cls_path.iterdir()):
                if not sample_dir.is_dir():
                    continue
                for img_path in sorted(sample_dir.glob("*.png")):
                    self.samples.append((img_path, self.class_to_idx[cls],
                                         cls, sample_dir.name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, cls_idx, cls, sample_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        view_id = int(img_path.stem)  # 0..4
        return {
            "image": img,
            "cls_idx": torch.tensor(cls_idx, dtype=torch.long),
            "cls_name": cls,
            "sample_id": sample_id,
            "view_id": view_id,
            "img_path": str(img_path),
        }


class CSIGSampleDataset(Dataset):
    """
    Sample-level dataset: returns all 5 views of a single physical sample.
    Useful for multi-view feature aggregation at inference time.
    """

    def __init__(self, root: str | Path, transform: Optional[Callable] = None,
                 classes: Optional[List[str]] = None):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()]) \
            if classes is None else classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[str, str, int]] = []  # cls, sample_id, cls_idx
        for cls in self.classes:
            cls_path = self.root / cls
            if not cls_path.exists():
                continue
            for sample_dir in sorted(cls_path.iterdir()):
                if sample_dir.is_dir():
                    self.samples.append((cls, sample_dir.name, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        cls, sample_id, cls_idx = self.samples[idx]
        views = []
        for v in range(5):
            p = self.root / cls / sample_id / f"{v}.png"
            img = Image.open(p).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            views.append(img)
        views = torch.stack(views, dim=0)  # (5, 3, H, W)
        return {
            "views": views,
            "cls_idx": torch.tensor(cls_idx, dtype=torch.long),
            "cls_name": cls,
            "sample_id": sample_id,
            "group_folder": f"{cls}/{sample_id}",
        }


def discover_classes(train_root: str | Path) -> List[str]:
    """Return the sorted list of known (training) classes."""
    return sorted([d.name for d in Path(train_root).iterdir() if d.is_dir()])


def get_patch_hw(input_size: int, patch_size: int = 14) -> Tuple[int, int]:
    """Return (H_p, W_p) of a ViT patch grid for a given input size."""
    assert input_size % patch_size == 0, \
        f"input_size {input_size} must be divisible by patch_size {patch_size}"
    n = input_size // patch_size
    return n, n
