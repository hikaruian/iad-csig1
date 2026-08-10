#!/usr/bin/env python3
"""
Offline evaluator.

Given a folder of predictions produced by `run.py` and an (optional) ground-
truth label file, compute image-level AUROC / AUPR and pixel-level AUROC /
AUPR / F1-max.  This is useful for cross-validation on your own split.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score


def _load_csv(p: Path):
    rows = {}
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["group_folder"]] = float(r["anomaly_score"])
    return rows


def _load_mask(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0


def per_image_metrics(scores, labels):
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    return auroc, aupr


def per_pixel_metrics(mask_dir: Path, gt_dir: Path, groups):
    aurocs, auprs, f1maxs = [], [], []
    for g in groups:
        preds, gts = [], []
        for v in range(5):
            pm = mask_dir / g / f"{v}_mask.png"
            gm = gt_dir / g / f"{v}_gt.png"
            if not pm.exists() or not gm.exists():
                continue
            preds.append(_load_mask(pm).reshape(-1))
            gts.append(np.asarray(Image.open(gm).convert("L"),
                                  dtype=np.float32).reshape(-1) / 255.0)
        if not preds:
            continue
        p = np.concatenate(preds)
        gt = (np.concatenate(gts) > 0.5).astype(np.int64)
        if gt.sum() == 0 or gt.sum() == gt.size:
            continue
        aurocs.append(roc_auc_score(gt, p))
        auprs.append(average_precision_score(gt, p))
        # F1 max over thresholds
        thr = np.linspace(0, 1, 101)
        best = 0.0
        for t in thr:
            pred_bin = (p >= t).astype(np.int64)
            tp = (pred_bin * gt).sum()
            fp = (pred_bin * (1 - gt)).sum()
            fn = ((1 - pred_bin) * gt).sum()
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            best = max(best, f1)
        f1maxs.append(best)
    return (float(np.mean(aurocs)), float(np.mean(auprs)),
            float(np.mean(f1maxs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=str, required=True)
    ap.add_argument("--gt-csv", type=str, required=True,
                    help="CSV with group_folder,label (0/1) for image-level GT.")
    ap.add_argument("--gt-mask-dir", type=str, default=None,
                    help="Directory of per-view GT masks (optional).")
    args = ap.parse_args()

    sub = Path(args.submission)
    pred_csv = sub / "submission.csv"
    pred_mask_dir = sub / "predicted_masks"
    gt = _load_csv(Path(args.gt_csv))
    pred = _load_csv(pred_csv)
    groups = sorted(set(gt.keys()) & set(pred.keys()))
    labels = np.array([gt[g] for g in groups])
    scores = np.array([pred[g] for g in groups])

    i_auroc, i_aupr = per_image_metrics(scores, labels)
    print(f"I-AUROC: {i_auroc:.4f}  I-AUPR: {i_aupr:.4f}")
    s_cls = 0.5 * i_auroc + 0.5 * i_aupr
    print(f"S_cls:   {s_cls:.4f}")

    if args.gt_mask_dir is not None:
        p_auroc, p_aupr, p_f1 = per_pixel_metrics(pred_mask_dir,
                                                  Path(args.gt_mask_dir),
                                                  groups)
        print(f"P-AUROC: {p_auroc:.4f}  P-AUPR: {p_aupr:.4f}  P-F1max: {p_f1:.4f}")
        s_seg = (p_auroc + p_aupr + p_f1) / 3.0
        s_total = 100.0 * (0.4 * s_cls + 0.6 * s_seg)
        print(f"S_seg:   {s_seg:.4f}")
        print(f"S total: {s_total:.2f}/100")


if __name__ == "__main__":
    main()
