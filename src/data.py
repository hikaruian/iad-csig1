"""CSIG training/inference datasets with synthetic pixel anomalies."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageOps,
)

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIEWS = (0, 1, 2, 3, 4)


def build_transform(
    image_size=448,
    is_train=False,
):
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=(
                transforms.InterpolationMode.BICUBIC
            ),
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ])


def discover_samples(root: Path):
    root = Path(root)

    if not root.is_dir():
        raise FileNotFoundError(root)

    result = []

    for cat in sorted(
        p for p in root.iterdir()
        if p.is_dir()
    ):
        for sample in sorted(
            p for p in cat.iterdir()
            if p.is_dir()
        ):
            if any(
                x.is_file()
                for x in sample.iterdir()
            ):
                result.append(
                    (
                        cat.name,
                        sample.name,
                        sample,
                    )
                )

    if not result:
        raise RuntimeError(
            f"No samples in {root}"
        )

    return result


def group_folder(cat, sid):
    return f"{cat}/{sid}"


def view_path(directory, view):
    for name in (
        f"{view}.png",
        f"{view}.PNG",
        f"{view:02d}.png",
        f"{view}.jpg",
        f"{view}.jpeg",
    ):
        path = directory / name

        if path.is_file():
            return path

    raise FileNotFoundError(
        f"view={view} missing in {directory}"
    )


def _load(path):
    with Image.open(path) as im:
        image = im.convert("RGB")
        image.load()

    return image


def _to_tensor(image):
    return TF.normalize(
        TF.to_tensor(image),
        IMAGENET_MEAN,
        IMAGENET_STD,
    )


def _mask_tensor(mask):
    arr = (
        np.asarray(mask) > 127
    ).astype(np.float32)

    return torch.from_numpy(
        arr
    ).unsqueeze(0)


def _make_mask(size):
    kind = random.choices(
        ["blob", "rectangle", "scratch", "spot"],
        weights=[0.25, 0.15, 0.40, 0.20],
    )[0]

    mask = Image.new(
        "L",
        (size, size),
        0,
    )

    if kind == "blob":
        s = random.choice(
            [4, 8, 16, 32]
        )

        noise = (
            np.random.rand(s, s) * 255
        ).astype(np.uint8)

        temp = Image.fromarray(
            noise
        ).resize(
            (size, size),
            Image.Resampling.BICUBIC,
        )

        threshold = random.uniform(
            0.6,
            0.8,
        ) * 255

        arr = (
            np.asarray(temp)
            > threshold
        ).astype(np.uint8) * 255

        return (
            Image.fromarray(arr),
            kind,
        )

    draw = ImageDraw.Draw(mask)

    if kind == "rectangle":
        lo = max(3, size // 100)
        hi = max(lo, size // 6)

        w = random.randint(lo, hi)
        h = random.randint(lo, hi)

        x = random.randint(
            0,
            max(0, size - w),
        )

        y = random.randint(
            0,
            max(0, size - h),
        )

        draw.rectangle(
            (x, y, x + w, y + h),
            fill=255,
        )

    elif kind == "spot":
        r = random.randint(
            max(2, size // 160),
            max(3, size // 30),
        )

        x = random.randint(
            r,
            max(r, size - r - 1),
        )

        y = random.randint(
            r,
            max(r, size - r - 1),
        )

        draw.ellipse(
            (
                x - r,
                y - r,
                x + r,
                y + r,
            ),
            fill=255,
        )

    else:
        x = float(
            random.randint(0, size - 1)
        )

        y = float(
            random.randint(0, size - 1)
        )

        angle = random.uniform(
            0,
            2 * math.pi,
        )

        length = random.randint(
            max(10, size // 20),
            max(20, size // 3),
        )

        width = random.randint(
            1,
            max(2, size // 120),
        )

        points = []

        curvature = random.uniform(
            -0.004,
            0.004,
        )

        for _ in range(length):
            if not (
                0 <= x < size
                and 0 <= y < size
            ):
                break

            points.append(
                (int(x), int(y))
            )

            angle += curvature
            x += math.cos(angle)
            y += math.sin(angle)

        if len(points) > 1:
            draw.line(
                points,
                fill=255,
                width=width,
            )

    return mask, kind


def _self_texture(image):
    mode = random.choice(
        [
            "brightness",
            "contrast",
            "color",
            "noise",
            "blur",
        ]
    )

    if mode == "brightness":
        return ImageEnhance.Brightness(
            image
        ).enhance(
            random.choice([
                random.uniform(0.4, 0.7),
                random.uniform(1.3, 1.8),
            ])
        )

    if mode == "contrast":
        return ImageEnhance.Contrast(
            image
        ).enhance(
            random.uniform(0.4, 2.0)
        )

    if mode == "color":
        return ImageEnhance.Color(
            image
        ).enhance(
            random.uniform(0.2, 2.0)
        )

    if mode == "blur":
        return image.filter(
            ImageFilter.GaussianBlur(
                random.uniform(1, 4)
            )
        )

    arr = np.asarray(
        image,
        dtype=np.float32,
    )

    arr += np.random.normal(
        0,
        random.uniform(10, 40),
        arr.shape,
    )

    return Image.fromarray(
        np.clip(
            arr,
            0,
            255,
        ).astype(np.uint8)
    )


class CSIGImageDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        image_size=448,
        synthetic_anomaly=False,
        synthetic_prob=0.8,
        **kwargs,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.transform = (
            transform
            or build_transform(
                image_size,
                True,
            )
        )

        self.synthetic_anomaly = (
            synthetic_anomaly
        )

        self.synthetic_prob = (
            synthetic_prob
        )

        self.items = []

        for cat, sid, directory in discover_samples(
            self.root
        ):
            for view in VIEWS:
                try:
                    self.items.append(
                        (
                            cat,
                            sid,
                            view,
                            view_path(
                                directory,
                                view,
                            ),
                        )
                    )
                except FileNotFoundError:
                    pass

        self.classes = sorted({
            x[0] for x in self.items
        })

        self.class_to_idx = {
            c: i
            for i, c in enumerate(
                self.classes
            )
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cat, sid, view, path = (
            self.items[idx]
        )

        image = _load(path)

        if not self.synthetic_anomaly:
            return (
                self.transform(image),
                self.class_to_idx[cat],
            )

        clean = image.resize(
            (
                self.image_size,
                self.image_size,
            ),
            Image.Resampling.BICUBIC,
        )

        mask = Image.new(
            "L",
            clean.size,
            0,
        )

        synthetic = clean.copy()
        kind = "normal"

        if random.random() < self.synthetic_prob:
            for _ in range(10):
                mask, kind = _make_mask(
                    self.image_size
                )

                ratio = (
                    np.asarray(mask) > 127
                ).mean()

                if 0.00005 <= ratio <= 0.30:
                    break

            texture = _self_texture(
                clean
            )

            alpha = random.uniform(
                0.55,
                1.0,
            )

            mixed = Image.blend(
                clean,
                texture,
                alpha,
            )

            synthetic = Image.composite(
                mixed,
                clean,
                mask,
            )

        return {
            "clean": _to_tensor(clean),
            "synthetic": _to_tensor(synthetic),
            "mask": _mask_tensor(mask),
            "class_idx": self.class_to_idx[cat],
            "category": cat,
            "sample_id": sid,
            "view_id": view,
            "anomaly_type": kind,
        }


class CSIGSampleDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        image_size=448,
    ):
        self.root = Path(root)

        self.transform = (
            transform
            or build_transform(
                image_size,
                False,
            )
        )

        self.samples = discover_samples(
            self.root
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cat, sid, directory = (
            self.samples[idx]
        )

        views = []

        for view in VIEWS:
            views.append(
                self.transform(
                    _load(
                        view_path(
                            directory,
                            view,
                        )
                    )
                )
            )

        return {
            "images": torch.stack(
                views
            ),
            "group_folder": group_folder(
                cat,
                sid,
            ),
            "category": cat,
            "sample_id": sid,
        }


def list_group_folders(root):
    return [
        group_folder(c, s)
        for c, s, _
        in discover_samples(Path(root))
    ]
