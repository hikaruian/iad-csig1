"""Build the official CSIG zip: submission.csv + predicted_masks/."""

from __future__ import annotations

import csv
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

MASK_SIZE = (448, 448)
VIEW_COUNT = 5


def _to_u8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.dtype != np.uint8:
        if mask.max() <= 1.0 + 1e-6:
            mask = np.clip(mask, 0.0, 1.0) * 255.0
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != MASK_SIZE:
        mask = np.asarray(Image.fromarray(mask, mode="L").resize(MASK_SIZE, Image.BILINEAR))
    return mask


class SubmissionBuilder:
    def __init__(self):
        self.samples: List[tuple] = []

    def add_sample(self, group_folder: str, anomaly_score: float, masks: Optional[Sequence[np.ndarray]] = None):
        if masks is None:
            masks = [np.zeros(MASK_SIZE, dtype=np.uint8) for _ in range(VIEW_COUNT)]
        if len(masks) != VIEW_COUNT:
            raise ValueError(f"{group_folder}: expected {VIEW_COUNT} masks, got {len(masks)}")
        self.samples.append((group_folder.replace("\\", "/").strip("/"), float(anomaly_score), list(masks)))

    def save(self, out_dir: str, zip_path: Optional[str] = None) -> Path:
        out = Path(out_dir)
        if out.exists():
            shutil.rmtree(out)
        pred_dir = out / "predicted_masks"
        pred_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out / "submission.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["group_folder", "anomaly_score"])
            for gf, score, _ in self.samples:
                writer.writerow([gf, f"{score:.8f}"])

        for gf, _, masks in self.samples:
            dest = pred_dir / gf
            dest.mkdir(parents=True, exist_ok=True)
            for i, mask in enumerate(masks):
                Image.fromarray(_to_u8(mask), mode="L").save(dest / f"{i}_mask.png")

        if zip_path:
            zip_p = Path(zip_path)
            zip_p.parent.mkdir(parents=True, exist_ok=True)
            if zip_p.exists():
                zip_p.unlink()
            with zipfile.ZipFile(zip_p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(csv_path, arcname="submission.csv")
                for png in sorted(pred_dir.rglob("*.png")):
                    zf.write(png, arcname=str(Path("predicted_masks") / png.relative_to(pred_dir)))
            return zip_p
        return out


def write_sample(out_root: Path, group_folder: str, score: float, masks_u8: Sequence[np.ndarray], scores_fh):
    """Incremental writer used by infer.py so we don't keep 3750×5 maps in RAM."""
    scores_fh.write(f"{group_folder},{score:.8f}\n")
    dest = out_root / "predicted_masks" / group_folder
    dest.mkdir(parents=True, exist_ok=True)
    for i, mask in enumerate(masks_u8):
        Image.fromarray(_to_u8(mask), mode="L").save(dest / f"{i}_mask.png")


def zip_submission(src_dir: str, zip_path: str) -> Path:
    src = Path(src_dir)
    zip_p = Path(zip_path)
    zip_p.parent.mkdir(parents=True, exist_ok=True)
    if zip_p.exists():
        zip_p.unlink()
    csv_path = src / "submission.csv"
    pred = src / "predicted_masks"
    if not csv_path.exists() or not pred.exists():
        raise FileNotFoundError("src_dir must contain submission.csv and predicted_masks/")
    with zipfile.ZipFile(zip_p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="submission.csv")
        for png in sorted(pred.rglob("*.png")):
            zf.write(png, arcname=str(Path("predicted_masks") / png.relative_to(pred)))
    return zip_p
