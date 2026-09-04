"""Local metrics matching the CSIG / TIANCHI evaluation protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

SEEN_CATEGORIES = {
    "3_adapter", "DVD_switch", "D_sub_connector", "PLCC_socket", "VR_joystick",
    "accurate_detection_switch", "battery", "blade_switch", "boost_converter_module",
    "button_battery_holder", "circuit_breaker", "connector_housing_female",
    "crimp_st_cable_mount_box", "dc_jack", "dc_power_connector", "detection_switch",
    "effect_transistor", "electronic_watch_movement", "ffc_connector_plug",
    "ingot_buckle", "laser_diode", "lego_pin_connector_plate", "limit_switch",
    "lithium_battery_plug", "littel_fuse", "lock", "miniature_lifting_motor",
    "mobile_charging_connector", "motor_bracket", "motor_gear_reducer", "motor_plug",
    "pencil_sharpener", "pinboard_connector", "potentiometer", "power_jack",
    "power_strip_socket", "purple_clay_pot", "retaining_ring", "rheostat",
    "self_lock_switch", "silicon_cell_sensor", "single_switch", "smd_receiver_module",
    "suction_cup", "toy_tire", "travel_switch", "vacuum_switch",
    "vehicle_harness_conductor", "vibration_motor", "wireless_receiver_module",
}

MASK_SIZE = (448, 448)
VIEW_COUNT = 5


def _read_mask(path: Path, must_448: bool) -> np.ndarray:
    if not path.exists():
        return np.zeros(MASK_SIZE, dtype=np.float32)
    mask = Image.open(path).convert("L")
    if mask.size != MASK_SIZE:
        if must_448:
            raise ValueError(f"prediction mask must be 448x448: {path}")
        mask = mask.resize(MASK_SIZE, Image.NEAREST)
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    if not np.isfinite(arr).all():
        raise ValueError(f"illegal values in {path}")
    return arr


def _f1_max(y_true, y_pred) -> float:
    p, r, _ = precision_recall_curve(y_true, y_pred)
    f1 = (2 * p * r) / (p + r + 1e-8)
    return float(np.max(f1))


def calculate_metrics(standard_dir: str, submission_dir: str) -> dict:
    std = Path(standard_dir)
    sub = Path(submission_dir)
    gt_df = pd.read_csv(std / "ground_truth.csv", encoding="utf-8-sig")
    sub_df = pd.read_csv(sub / "submission.csv", encoding="utf-8-sig")
    merged = pd.merge(gt_df, sub_df, on="group_folder", how="left")
    if merged["anomaly_score"].isnull().any():
        raise ValueError("submission.csv is missing some group_folder rows")
    merged["category"] = merged["group_folder"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]

    cat_metrics = {}
    for cat, cat_data in merged.groupby("category"):
        y_true = cat_data["label"].values.astype(np.int8)
        y_pred = cat_data["anomaly_score"].values.astype(np.float32)
        metrics = {}
        if len(np.unique(y_true)) > 1:
            metrics["I-AUROC"] = float(roc_auc_score(y_true, y_pred))
            metrics["I-AP"] = float(average_precision_score(y_true, y_pred))
        gt_masks, pr_masks = [], []
        for group in cat_data["group_folder"]:
            g = str(group).replace("\\", "/").strip("/")
            for i in range(VIEW_COUNT):
                gt_masks.append(_read_mask(std / "masks" / g / f"{i}_mask.png", must_448=False))
                pr_masks.append(_read_mask(sub / "predicted_masks" / g / f"{i}_mask.png", must_448=True))
        gt = np.stack(gt_masks).reshape(-1)
        pr = np.stack(pr_masks).reshape(-1)
        gt_bin = (gt > 0).astype(np.int8)
        if gt_bin.min() != gt_bin.max():
            metrics["P-AUROC"] = float(roc_auc_score(gt_bin, pr))
            metrics["P-AP"] = float(average_precision_score(gt_bin, pr))
            metrics["P-F1max"] = _f1_max(gt_bin, pr)
        cat_metrics[cat] = metrics

    seen = [c for c in cat_metrics if c in SEEN_CATEGORIES]
    unseen = [c for c in cat_metrics if c not in SEEN_CATEGORIES]
    if not seen:
        seen = list(cat_metrics.keys())

    def avg(cats, names):
        vals = []
        for n in names:
            xs = [cat_metrics[c][n] for c in cats if n in cat_metrics[c]]
            if xs:
                vals.append(float(np.mean(xs)))
        return float(np.mean(vals)) if vals else 0.0

    s_cls = avg(seen, ["I-AUROC", "I-AP"])
    s_seg = avg(list(cat_metrics.keys()), ["P-AUROC", "P-AP", "P-F1max"])
    if unseen:
        s_zs = avg(unseen, ["I-AUROC", "I-AP", "P-AUROC", "P-AP", "P-F1max"])
        score = 100.0 * (0.3 * s_cls + 0.5 * s_seg + 0.2 * s_zs)
    else:
        s_zs = None
        # User brief: 0.4 cls + 0.6 seg. Toolkit default: 0.3 / 0.7.
        # We report BOTH so you can match whichever judge script is used.
        score_user = 100.0 * (0.4 * s_cls + 0.6 * s_seg)
        score_toolkit = 100.0 * (0.3 * s_cls + 0.7 * s_seg)
        score = score_user

    out = {
        "score": score,
        "score_user_formula_0.4_0.6": 100.0 * (0.4 * s_cls + 0.6 * s_seg),
        "score_toolkit_formula_0.3_0.7": 100.0 * (0.3 * s_cls + 0.7 * s_seg),
        "S_cls": s_cls,
        "S_seg": s_seg,
        "S_zs": s_zs,
        "per_category": cat_metrics,
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--standard-dir", required=True)
    p.add_argument("--submission-dir", required=True)
    p.add_argument("--out", default="metrics.json")
    args = p.parse_args()
    metrics = calculate_metrics(args.standard_dir, args.submission_dir)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_category"}, indent=2))


if __name__ == "__main__":
    main()
