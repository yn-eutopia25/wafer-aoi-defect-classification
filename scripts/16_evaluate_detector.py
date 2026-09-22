#!/usr/bin/env python3
"""
16_evaluate_detector.py — 目标检测评估

对 models/detector/baseline_best.pt 在 val / test 划分上进行完整评估。

输出 (reports/detection_evaluation/):
  - standard_metrics_<split>.json    标准指标 (mAP50 / mAP50-95 / 每类 AP / P / R)
  - ultralytics_val_<split>/         ultralytics 自动生成的 PR/F1/P/R 曲线 + 混淆矩阵
  - prediction_instances.csv         每条预测的匹配结果 (含 error_type)
  - missed_instances.csv             每条漏检 GT
  - class_C_metrics_<split>.json     C 类小目标专项指标
  - class_A_metrics_<split>.json     A 类专项指标 (含图片级 has_A)
  - count_statistics_<split>.json    数量统计 (MAE / RMSE)
  - count_scatter_<split>.png        predicted vs gt 散点
  - per_image_count_error_<split>.csv
  - class_thresholds_val.json        (仅 val) 阈值分析
  - test_evaluation_lock.json        (仅 test) 评估锁

用法:
    # val 评估 (默认)
    python scripts/16_evaluate_detector.py --split val

    # test 评估 (需显式确认)
    python scripts/16_evaluate_detector.py --split test --confirm-test-evaluation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("eval_detector")

# =============================================================================
# 常量
# =============================================================================
CLASS_CODES = ["A", "B", "C", "D", "E", "F"]
CLASS_NAMES = {
    "A": "residue_cleaning",
    "B": "edge_glue",
    "C": "particle",
    "D": "pad_abnormal",
    "E": "surface_damage",
    "F": "pi_broken",
}
# YOLO 类索引 ↔ 短码
YOLO_IDX_TO_CODE = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}
CODE_TO_YOLO_IDX = {v: k for k, v in YOLO_IDX_TO_CODE.items()}

# 评估参数
CONF_PRED = 0.001      # 完整 PR 曲线用极低 conf
IOU_NMS = 0.5          # NMS IoU
MAX_DET = 300
MATCH_IOU = 0.5        # 标准匹配阈值
LOOSE_IOU = 0.3        # 宽松匹配阈值 (同时报告)
COUNT_CONF = 0.25      # 数量统计使用的预测 conf 阈值

# C 类按 bbox 面积分组 (px^2)
C_AREA_BINS = {"small": (0, 100), "medium": (100, 400), "large": (400, float("inf"))}


# =============================================================================
# 工具函数
# =============================================================================
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _try_git_hash() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def iou_box(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """计算两个 xyxy 框的 IoU。"""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def center_in_box(cx: float, cy: float, box: Tuple[float, float, float, float]) -> bool:
    return (box[0] <= cx <= box[2]) and (box[1] <= cy <= box[3])


def box_area(box: Tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


# =============================================================================
# 数据加载
# =============================================================================
def load_splits(splits_csv: Path) -> Dict[str, str]:
    """返回 {image_id: split}。"""
    with open(splits_csv, "r", encoding="utf-8-sig", newline="") as f:
        return {r["image_id"]: r["split"] for r in csv.DictReader(f)}


def load_gt_instances(instances_csv: Path, split: str, splits_csv: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    加载某 split 的 GT 实例，按 image_id 分组。
    只保留 A-F 且非 pseudo_normal 的实例。
    返回 {image_id: [ {ann_id, label, box, area}, ... ]}。
    """
    split_map = load_splits(splits_csv)
    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(instances_csv, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("is_pseudo_normal", "False") == "True":
                continue
            code = r["label_code"]
            if code not in CLASS_CODES:
                continue
            img_id = r["image_id"]
            if split_map.get(img_id) != split:
                continue
            box = (float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"]))
            by_image[img_id].append({
                "ann_id": r["ann_id"],
                "label": code,
                "box": box,
                "area": float(r["bbox_area_px"]),
            })
    return by_image


def split_image_paths(detection_dir: Path, split: str) -> List[Path]:
    img_dir = detection_dir / "images" / split
    return sorted(img_dir.glob("*.jpg"))


# =============================================================================
# 标准指标 (ultralytics val)
# =============================================================================
def run_ultralytics_val(model, data_yaml: Path, split: str, save_dir: Path,
                        imgsz: int) -> Dict[str, Any]:
    """运行 ultralytics 内置 val，获取标准指标 + 自动曲线/混淆矩阵。"""
    logger.info("运行 ultralytics val (split=%s, conf=%s, iou=%s, class-aware NMS)...",
                split, CONF_PRED, IOU_NMS)
    # 注意: agnostic_nms=False (默认) → class-aware NMS，不同类别重叠框可同时保留
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        conf=CONF_PRED,
        iou=IOU_NMS,
        max_det=MAX_DET,
        agnostic_nms=False,      # class-aware NMS
        plots=True,
        save=False,
        save_json=False,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        verbose=False,
        device=_select_device(),
    )

    box = metrics.box
    per_class = {}
    for i, code in enumerate(CLASS_CODES):
        per_class[code] = {
            "name": CLASS_NAMES[code],
            "yolo_idx": i,
            "AP50": float(box.ap50[i]) if i < len(box.ap50) else 0.0,
            "AP50_95": float(box.ap[i]) if i < len(box.ap) else 0.0,
            "precision": float(box.p[i]) if i < len(box.p) else 0.0,
            "recall": float(box.r[i]) if i < len(box.r) else 0.0,
        }
    return {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "mean_precision": float(box.mp),
        "mean_recall": float(box.mr),
        "per_class": per_class,
        "ultralytics_save_dir": str(save_dir),
    }


def _select_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


# =============================================================================
# 自定义匹配
# =============================================================================
def match_image(gts: List[Dict[str, Any]], preds: List[Dict[str, Any]],
                iou_thr: float = MATCH_IOU) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    对单张图执行匹配。
    返回:
      pred_records: 每条预测的匹配记录
      missed:       漏检 GT 记录
      matched_gt_ids: 已匹配的 GT ann_id 集合 (在 iou_thr 下)
    匹配原则: 同类别优先按最高 IoU 一对一贪心匹配 (IoU >= iou_thr)。
    """
    # 1. 计算所有同类 (pred, gt) 对的 IoU
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            if p["label"] == g["label"]:
                v = iou_box(p["box"], g["box"])
                if v > 0:
                    pairs.append((v, pi, gi))
    pairs.sort(key=lambda x: -x[0])

    matched_p: set = set()
    matched_g: set = set()
    tp_pairs: Dict[int, Tuple[int, float]] = {}  # pi -> (gi, iou)

    # 2. 贪心一对一匹配 (IoU >= iou_thr)
    for v, pi, gi in pairs:
        if pi in matched_p or gi in matched_g:
            continue
        if v >= iou_thr:
            tp_pairs[pi] = (gi, v)
            matched_p.add(pi)
            matched_g.add(gi)

    # 3. 为每条预测生成记录
    pred_records: List[Dict[str, Any]] = []
    for pi, p in enumerate(preds):
        cx, cy = box_center(p["box"])
        # 最佳同类 GT
        best_same = max(
            ((gi, g, iou_box(p["box"], g["box"])) for gi, g in enumerate(gts) if g["label"] == p["label"]),
            key=lambda x: x[2], default=None,
        )
        # 最佳异类 GT
        best_diff = max(
            ((gi, g, iou_box(p["box"], g["box"])) for gi, g in enumerate(gts) if g["label"] != p["label"]),
            key=lambda x: x[2], default=None,
        )

        if pi in tp_pairs:
            gi, v = tp_pairs[pi]
            g = gts[gi]
            err = "correct"
            matched_ann_id = g["ann_id"]
            matched_gt_label = g["label"]
            iou_val = v
            ch = center_in_box(cx, cy, g["box"])
        elif best_same is not None and best_same[2] >= LOOSE_IOU:
            # 同类但 IoU 不足 0.5 (定位偏差)
            g = best_same[1]
            err = "localization_error"
            matched_ann_id = g["ann_id"]
            matched_gt_label = g["label"]
            iou_val = best_same[2]
            ch = center_in_box(cx, cy, g["box"])
        elif best_diff is not None and best_diff[2] >= LOOSE_IOU:
            # 异类空间匹配 → 分类错误
            g = best_diff[1]
            err = "classification_error"
            matched_ann_id = g["ann_id"]
            matched_gt_label = g["label"]
            iou_val = best_diff[2]
            ch = center_in_box(cx, cy, g["box"])
        else:
            err = "false_positive"
            matched_ann_id = ""
            matched_gt_label = ""
            iou_val = best_same[2] if best_same else 0.0
            ch = center_in_box(cx, cy, best_same[1]["box"]) if best_same else False

        pred_records.append({
            "image_id": p["image_id"],
            "prediction_id": p["pred_id"],
            "predicted_label": p["label"],
            "confidence": round(float(p["conf"]), 6),
            "x_min": round(p["box"][0], 2),
            "y_min": round(p["box"][1], 2),
            "x_max": round(p["box"][2], 2),
            "y_max": round(p["box"][3], 2),
            "matched_ann_id": matched_ann_id,
            "matched_gt_label": matched_gt_label,
            "iou": round(iou_val, 4),
            "center_hit": bool(ch),
            "error_type": err,
        })

    # 4. 漏检 GT
    missed_records: List[Dict[str, Any]] = []
    for gi, g in enumerate(gts):
        if gi in matched_g:
            continue
        # 找最佳同类预测 IoU (用于分析)
        best_pred = max(
            ((p, iou_box(p["box"], g["box"])) for p in preds if p["label"] == g["label"]),
            key=lambda x: x[1], default=None,
        )
        best_pred_iou = best_pred[1] if best_pred else 0.0
        best_pred_conf = best_pred[0]["conf"] if best_pred else 0.0
        # 是否有任何预测中心落入该 GT
        any_center = any(center_in_box(*box_center(p["box"]), g["box"]) for p in preds)
        missed_records.append({
            "image_id": g.get("image_id", ""),
            "ann_id": g["ann_id"],
            "gt_label": g["label"],
            "x_min": round(g["box"][0], 2),
            "y_min": round(g["box"][1], 2),
            "x_max": round(g["box"][2], 2),
            "y_max": round(g["box"][3], 2),
            "bbox_area": round(g["area"], 2),
            "best_pred_iou": round(best_pred_iou, 4),
            "best_pred_conf": round(best_pred_conf, 6),
            "center_hit_by_pred": bool(any_center),
            "error_hint": "small_object" if g["label"] == "C" and g["area"] < C_AREA_BINS["medium"][0]
                          else ("low_iou_overlap" if best_pred_iou >= LOOSE_IOU else "no_overlap"),
        })

    return pred_records, missed_records, matched_g


def run_predictions(model, image_paths: List[Path],
                    imgsz: int) -> Dict[str, List[Dict[str, Any]]]:
    """运行预测，返回 {image_id: [ {pred_id, label, conf, box}, ... ]}。"""
    logger.info("运行预测 (%d 张图, conf=%s, iou=%s, max_det=%s, class-aware NMS)...",
                len(image_paths), CONF_PRED, IOU_NMS, MAX_DET)
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        conf=CONF_PRED,
        iou=IOU_NMS,
        max_det=MAX_DET,
        agnostic_nms=False,   # class-aware NMS
        verbose=False,
        save=False,
        device=_select_device(),
    )
    by_image: Dict[str, List[Dict[str, Any]]] = {}
    for idx, r in enumerate(results):
        # 注意: 传入路径列表时 ultralytics 会把 r.path 设为 'image0' 等通用名，
        # 因此用输入路径列表的索引还原真实 image_id。
        img_id = image_paths[idx].stem
        preds: List[Dict[str, Any]] = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                yolo_idx = int(cls[i])
                code = YOLO_IDX_TO_CODE.get(yolo_idx, "?")
                preds.append({
                    "pred_id": f"{img_id}_pred_{i:03d}",
                    "image_id": img_id,
                    "label": code,
                    "conf": float(conf[i]),
                    "box": (float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3])),
                })
        by_image[img_id] = preds
    return by_image


# =============================================================================
# 写 CSV
# =============================================================================
def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    logger.info("写入 %s (%d 行)", path, len(rows))


# =============================================================================
# C 类小目标专项
# =============================================================================
def compute_C_metrics(pred_records: List[Dict], missed_records: List[Dict],
                      gt_by_image: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """C 类小目标专项指标。"""
    # GT C 总数
    gt_c = [g for gts in gt_by_image.values() for g in gts if g["label"] == "C"]
    n_gt_c = len(gt_c)

    # C 预测按 image_id 索引 (携带预测框)
    c_preds_by_img: Dict[str, List[Dict]] = defaultdict(list)
    for r in pred_records:
        if r["predicted_label"] == "C":
            c_preds_by_img[r["image_id"]].append(r)

    # 对每个 C GT，找最佳同类预测 IoU / 是否被中心命中
    matched_50: set = set()
    matched_30: set = set()
    center_hit_gts: set = set()
    best_iou_by_ann: Dict[str, float] = {}

    for g in gt_c:
        img = g.get("image_id", "")
        gbox = g["box"]
        preds = c_preds_by_img.get(img, [])
        best_iou = 0.0
        any_center = False
        for p in preds:
            pbox = (float(p["x_min"]), float(p["y_min"]), float(p["x_max"]), float(p["y_max"]))
            v = iou_box(pbox, gbox)
            if v > best_iou:
                best_iou = v
            cx, cy = box_center(pbox)
            if center_in_box(cx, cy, gbox):
                any_center = True
        if best_iou >= MATCH_IOU:
            matched_50.add(g["ann_id"])
        if best_iou >= LOOSE_IOU:
            matched_30.add(g["ann_id"])
        if any_center:
            center_hit_gts.add(g["ann_id"])
        best_iou_by_ann[g["ann_id"]] = best_iou

    recall_50 = len(matched_50) / n_gt_c if n_gt_c else 0.0
    recall_30 = len(matched_30) / n_gt_c if n_gt_c else 0.0
    recall_center = len(center_hit_gts) / n_gt_c if n_gt_c else 0.0

    # 按面积分组 (IoU>=0.5 匹配)
    by_bin: Dict[str, Dict[str, int]] = {b: {"n_gt": 0, "tp": 0} for b in C_AREA_BINS}
    for g in gt_c:
        b = next((name for name, (lo, hi) in C_AREA_BINS.items() if lo <= g["area"] < hi), "large")
        by_bin[b]["n_gt"] += 1
        if g["ann_id"] in matched_50:
            by_bin[b]["tp"] += 1
    bin_recall = {b: (v["tp"] / v["n_gt"] if v["n_gt"] else 0.0) for b, v in by_bin.items()}

    # 漏检 / 误检 gallery 索引
    missed_c = [r for r in missed_records if r["gt_label"] == "C"]
    fp_c = [r for r in pred_records if r["predicted_label"] == "C"
            and r["error_type"] in ("false_positive", "classification_error")]

    return {
        "n_gt_C": n_gt_c,
        "tp_iou50": len(matched_50),
        "tp_iou30": len(matched_30),
        "tp_center_hit": len(center_hit_gts),
        "recall_iou50": round(recall_50, 4),
        "recall_iou30": round(recall_30, 4),
        "recall_center_hit": round(recall_center, 4),
        "area_bins": {b: {"n_gt": v["n_gt"], "tp": v["tp"],
                          "recall": round(bin_recall[b], 4)} for b, v in by_bin.items()},
        "area_bin_thresholds_px2": {b: list(C_AREA_BINS[b]) for b in C_AREA_BINS},
        "missed_gallery_count": len(missed_c),
        "false_positive_gallery_count": len(fp_c),
        "missed_ann_ids": [r["ann_id"] for r in missed_c],
        "false_positive_pred_ids": [r["prediction_id"] for r in fp_c],
    }


# =============================================================================
# A 类专项
# =============================================================================
def compute_A_metrics(pred_records: List[Dict], missed_records: List[Dict],
                      gt_by_image: Dict[str, List[Dict]],
                      preds_by_image: Dict[str, List[Dict]],
                      count_conf: float = COUNT_CONF) -> Dict[str, Any]:
    """A 类专项指标。"""
    # 标准 bbox AP @0.5 / @0.3 recall (按唯一 GT 计数)
    gt_a = [g for gts in gt_by_image.values() for g in gts if g["label"] == "A"]
    n_gt_a = len(gt_a)

    # A 预测按 image_id 索引
    a_preds_by_img: Dict[str, List[Dict]] = defaultdict(list)
    for r in pred_records:
        if r["predicted_label"] == "A":
            a_preds_by_img[r["image_id"]].append(r)

    matched_50: set = set()
    matched_30: set = set()
    for g in gt_a:
        img = g.get("image_id", "")
        gbox = g["box"]
        preds = a_preds_by_img.get(img, [])
        best_iou = 0.0
        for p in preds:
            pbox = (float(p["x_min"]), float(p["y_min"]), float(p["x_max"]), float(p["y_max"]))
            v = iou_box(pbox, gbox)
            if v > best_iou:
                best_iou = v
        if best_iou >= MATCH_IOU:
            matched_50.add(g["ann_id"])
        if best_iou >= LOOSE_IOU:
            matched_30.add(g["ann_id"])

    recall_50 = len(matched_50) / n_gt_a if n_gt_a else 0.0
    recall_30 = len(matched_30) / n_gt_a if n_gt_a else 0.0

    # 图片级 has_A
    images_with_a_gt = {img for img, gts in gt_by_image.items() if any(g["label"] == "A" for g in gts)}
    images_with_a_pred = set()
    for img, preds in preds_by_image.items():
        if any(p["label"] == "A" and p["conf"] >= count_conf for p in preds):
            images_with_a_pred.add(img)

    all_images = set(gt_by_image.keys()) | set(preds_by_image.keys())
    tp_img = len(images_with_a_gt & images_with_a_pred)
    fp_img = len(images_with_a_pred - images_with_a_gt)
    fn_img = len(images_with_a_gt - images_with_a_pred)
    img_precision = tp_img / (tp_img + fp_img) if (tp_img + fp_img) else 0.0
    img_recall = tp_img / (tp_img + fn_img) if (tp_img + fn_img) else 0.0
    img_f1 = 2 * img_precision * img_recall / (img_precision + img_recall) if (img_precision + img_recall) else 0.0

    # 边界偏差 gallery: A 漏检或定位偏差的实例
    boundary_issues = [r for r in missed_records if r["gt_label"] == "A"]
    boundary_issues += [r for r in pred_records
                        if r["predicted_label"] == "A" and r["error_type"] == "localization_error"]

    # A 是否更适合图片级/区域级建模
    if img_recall > recall_50 + 0.1:
        recommendation = "image_level_or_region_level"
        recommendation_reason = f"图片级 recall ({img_recall:.3f}) 显著高于 bbox recall ({recall_50:.3f})，A 类边界主观性强，更适合图片级/区域级建模。"
    else:
        recommendation = "bbox_level"
        recommendation_reason = f"图片级 recall ({img_recall:.3f}) 与 bbox recall ({recall_50:.3f}) 接近，bbox 检测即可。"

    return {
        "n_gt_A": n_gt_a,
        "bbox_recall_iou50": round(recall_50, 4),
        "bbox_recall_iou30": round(recall_30, 4),
        "image_level": {
            "n_images_with_A_gt": len(images_with_a_gt),
            "n_images_with_A_pred": len(images_with_a_pred),
            "precision": round(img_precision, 4),
            "recall": round(img_recall, 4),
            "f1": round(img_f1, 4),
            "count_conf_threshold": count_conf,
        },
        "boundary_deviation_gallery_count": len(boundary_issues),
        "boundary_deviation_ann_ids": [r.get("ann_id", r.get("prediction_id")) for r in boundary_issues],
        "modeling_recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
    }


# =============================================================================
# 数量统计
# =============================================================================
def compute_count_statistics(pred_records: List[Dict], missed_records: List[Dict],
                             gt_by_image: Dict[str, List[Dict]],
                             preds_by_image: Dict[str, List[Dict]],
                             split: str, report_dir: Path,
                             count_conf: float = COUNT_CONF) -> Dict[str, Any]:
    """每张图片比较人工与预测的每类数量。"""
    all_images = sorted(set(gt_by_image.keys()) | set(preds_by_image.keys()))
    per_image_rows: List[Dict[str, Any]] = []
    gt_counts = {c: [] for c in CLASS_CODES}
    pred_counts = {c: [] for c in CLASS_CODES}

    for img in all_images:
        gts = gt_by_image.get(img, [])
        preds = [p for p in preds_by_image.get(img, []) if p["conf"] >= count_conf]
        row = {"image_id": img}
        total_gt = 0; total_pred = 0
        for c in CLASS_CODES:
            ng = sum(1 for g in gts if g["label"] == c)
            np_ = sum(1 for p in preds if p["label"] == c)
            row[f"gt_{c}"] = ng
            row[f"pred_{c}"] = np_
            row[f"err_{c}"] = np_ - ng
            gt_counts[c].append(ng)
            pred_counts[c].append(np_)
            total_gt += ng
            total_pred += np_
        row["gt_total"] = total_gt
        row["pred_total"] = total_pred
        row["err_total"] = total_pred - total_gt
        per_image_rows.append(row)

    # MAE / RMSE
    per_class_stats = {}
    for c in CLASS_CODES:
        g = np.array(gt_counts[c], dtype=float)
        p = np.array(pred_counts[c], dtype=float)
        err = p - g
        per_class_stats[c] = {
            "mae": round(float(np.mean(np.abs(err))), 4),
            "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "gt_total": int(g.sum()),
            "pred_total": int(p.sum()),
        }
    gt_total_arr = np.array([r["gt_total"] for r in per_image_rows], dtype=float)
    pred_total_arr = np.array([r["pred_total"] for r in per_image_rows], dtype=float)
    total_err = pred_total_arr - gt_total_arr
    total_stats = {
        "mae": round(float(np.mean(np.abs(total_err))), 4),
        "rmse": round(float(np.sqrt(np.mean(total_err ** 2))), 4),
        "gt_total": int(gt_total_arr.sum()),
        "pred_total": int(pred_total_arr.sum()),
        "count_conf_threshold": count_conf,
    }

    # 散点图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for ax, c in zip(axes.flat, CLASS_CODES):
            ax.scatter(gt_counts[c], pred_counts[c], alpha=0.6, s=40)
            mx = max(max(gt_counts[c]), max(pred_counts[c]), 1)
            ax.plot([0, mx], [0, mx], "r--", lw=1)
            ax.set_title(f"{c} ({CLASS_NAMES[c]})")
            ax.set_xlabel("GT count")
            ax.set_ylabel("Pred count")
            ax.set_xlim(-0.5, mx + 0.5)
            ax.set_ylim(-0.5, mx + 0.5)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Count scatter ({split}, conf>={count_conf})", fontsize=13)
        fig.tight_layout()
        scatter_path = report_dir / f"count_scatter_{split}.png"
        fig.savefig(scatter_path, dpi=120)
        plt.close(fig)
        logger.info("散点图: %s", scatter_path)
    except Exception as e:
        logger.warning("散点图绘制失败: %s", e)

    # 写 CSV
    csv_path = report_dir / f"per_image_count_error_{split}.csv"
    fieldnames = (["image_id"]
                  + [f"gt_{c}" for c in CLASS_CODES]
                  + [f"pred_{c}" for c in CLASS_CODES]
                  + [f"err_{c}" for c in CLASS_CODES]
                  + ["gt_total", "pred_total", "err_total"])
    write_csv(csv_path, fieldnames, per_image_rows)

    return {
        "per_class": per_class_stats,
        "total": total_stats,
        "n_images": len(all_images),
        "scatter_path": str(report_dir / f"count_scatter_{split}.png"),
        "per_image_csv": str(csv_path),
    }


# =============================================================================
# 阈值分析 (仅 val)
# =============================================================================
def threshold_analysis(pred_records: List[Dict], gt_by_image: Dict[str, List[Dict]],
                       report_dir: Path) -> Dict[str, Any]:
    """每类寻找最大 F1 阈值与目标 recall 阈值。"""
    n_gt_by_class = {c: sum(1 for gts in gt_by_image.values() for g in gts if g["label"] == c) for c in CLASS_CODES}
    thresholds = np.arange(0.05, 0.91, 0.05)
    target_recalls = [0.5, 0.7, 0.9]
    result: Dict[str, Any] = {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    except Exception:
        axes = None

    for idx, c in enumerate(CLASS_CODES):
        # 取该类的所有预测 (按 conf 降序)，TP = error_type==correct 且 matched_gt_label==c
        cls_preds = [r for r in pred_records if r["predicted_label"] == c]
        cls_preds.sort(key=lambda r: -r["confidence"])
        n_gt = n_gt_by_class[c]
        is_tp = [1 if (r["error_type"] == "correct" and r["matched_gt_label"] == c) else 0 for r in cls_preds]
        confs = [r["confidence"] for r in cls_preds]

        rows = []
        best_f1 = -1.0; best_thr = 0.05
        for thr in thresholds:
            tp = sum(t for t, cf in zip(is_tp, confs) if cf >= thr)
            fp = sum(1 for t, cf in zip(is_tp, confs) if cf >= thr and t == 0)
            fn = n_gt - tp
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / n_gt if n_gt else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            rows.append({"threshold": round(float(thr), 4), "precision": round(p, 4),
                         "recall": round(r, 4), "f1": round(f1, 4),
                         "tp": tp, "fp": fp, "fn": fn})
            if f1 > best_f1:
                best_f1 = f1; best_thr = float(thr)

        # 目标 recall 阈值 (满足 r>=target 的最大阈值，取最高 conf 仍达标的)
        target_thrs = {}
        for tgt in target_recalls:
            ok = [row for row in rows if row["recall"] >= tgt]
            target_thrs[f"recall_{tgt}"] = round(max(r["threshold"] for r in ok), 4) if ok else None

        # 绘图
        if axes is not None:
            ax = axes.flat[idx]
            ax.plot([r["threshold"] for r in rows], [r["precision"] for r in rows], label="P", marker=".")
            ax.plot([r["threshold"] for r in rows], [r["recall"] for r in rows], label="R", marker=".")
            ax.plot([r["threshold"] for r in rows], [r["f1"] for r in rows], label="F1", marker=".")
            ax.axvline(best_thr, color="gray", ls="--", lw=0.8)
            ax.set_title(f"{c} (best F1@{best_thr:.2f}={best_f1:.3f})")
            ax.set_xlabel("conf threshold")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        result[c] = {
            "n_gt": n_gt,
            "best_f1_threshold": round(best_thr, 4),
            "best_f1": round(best_f1, 4),
            "target_recall_thresholds": target_thrs,
            "curve": rows,
        }

    if axes is not None:
        fig.suptitle("Per-class threshold analysis (val)", fontsize=13)
        fig.tight_layout()
        curve_path = report_dir / "threshold_curves_val.png"
        fig.savefig(curve_path, dpi=120)
        plt.close(fig)
        logger.info("阈值曲线: %s", curve_path)

    return result


# =============================================================================
# test 锁
# =============================================================================
def write_test_lock(report_dir: Path, model_path: Path, split_image_list_path: Path,
                    threshold_path: Optional[Path]) -> Path:
    lock = {
        "evaluation_time": datetime.now(timezone.utc).isoformat(),
        "git_commit": _try_git_hash(),
        "model_path": str(model_path),
        "model_sha256": _sha256_file(model_path),
        "split_image_list_path": str(split_image_list_path),
        "split_sha256": _sha256_file(split_image_list_path),
        "threshold_file": str(threshold_path) if threshold_path else "",
        "threshold_sha256": _sha256_file(threshold_path) if threshold_path and threshold_path.exists() else "",
        "note": "test 评估锁定。若模型或阈值变化，必须作为新正式实验，不得覆盖原 test 结果。",
    }
    lock_path = report_dir / "test_evaluation_lock.json"
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("test 锁已写入: %s", lock_path)
    return lock_path


def check_test_lock(report_dir: Path, model_path: Path, threshold_path: Optional[Path]) -> None:
    """若已有 test 锁，校验模型/阈值 hash 是否变化。"""
    lock_path = report_dir / "test_evaluation_lock.json"
    if not lock_path.exists():
        return
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    cur_model_hash = _sha256_file(model_path)
    cur_thr_hash = _sha256_file(threshold_path) if threshold_path and threshold_path.exists() else ""
    if lock.get("model_sha256") != cur_model_hash:
        logger.error("检测到模型变化 (lock=%s, current=%s)。", lock.get("model_sha256", "")[:12], cur_model_hash[:12])
        logger.error("test 结果已锁定，不得覆盖。请将本次作为新正式实验 (新目录/新模型名)。")
        sys.exit(2)
    if lock.get("threshold_sha256") and lock.get("threshold_sha256") != cur_thr_hash:
        logger.error("检测到阈值文件变化。test 结果已锁定，不得覆盖。")
        sys.exit(2)
    logger.warning("test 锁已存在且模型/阈值未变化 → 允许重跑 (将覆盖相同结果)。")


# =============================================================================
# 主流程
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="目标检测评估")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--confirm-test-evaluation", action="store_true",
                   help="确认 test 集评估 (test 必须传入)")
    p.add_argument("--model", type=str, default="models/detector/baseline_best.pt")
    p.add_argument("--detection-dir", type=str, default="data/derived/detection")
    p.add_argument("--data-yaml", type=str, default="data/derived/detection/data.yaml")
    p.add_argument("--instances", type=str, default="data/annotations/instances.csv")
    p.add_argument("--splits", type=str, default="data/derived/splits.csv")
    p.add_argument("--output-dir", type=str, default="reports/detection_evaluation")
    p.add_argument("--imgsz", type=int, default=640,
                   help="评估和预测输入尺寸；比较实验时需显式保持一致")
    args = p.parse_args()

    if args.split == "test" and not args.confirm_test_evaluation:
        logger.error("test 集评估需要 --confirm-test-evaluation 确认 (test 结果将锁定)。")
        sys.exit(1)

    root = Path(__file__).resolve().parent.parent
    model_path = (root / args.model).resolve()
    detection_dir = (root / args.detection_dir).resolve()
    data_yaml = (root / args.data_yaml).resolve()
    instances_csv = (root / args.instances).resolve()
    splits_csv = (root / args.splits).resolve()
    report_dir = (root / args.output_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        logger.error("模型不存在: %s", model_path)
        sys.exit(1)

    # test 阈值文件依赖
    threshold_path = report_dir / "class_thresholds_val.json"
    if args.split == "test":
        if not threshold_path.exists():
            logger.error("test 评估需要先在 val 上运行得到 %s", threshold_path)
            logger.error("请先: python scripts/16_evaluate_detector.py --split val")
            sys.exit(1)
        check_test_lock(report_dir, model_path, threshold_path)

    # 加载模型
    from ultralytics import YOLO
    logger.info("加载模型: %s", model_path)
    model = YOLO(str(model_path))

    # ---- 1. 标准指标 (ultralytics val + 曲线) ----
    ultra_save_dir = report_dir / f"ultralytics_val_{args.split}"
    standard = run_ultralytics_val(model, data_yaml, args.split, ultra_save_dir, args.imgsz)
    standard["imgsz"] = args.imgsz
    (report_dir / f"standard_metrics_{args.split}.json").write_text(
        json.dumps(standard, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("标准指标: mAP50=%.4f, mAP50-95=%.4f, mP=%.4f, mR=%.4f",
                standard["mAP50"], standard["mAP50_95"], standard["mean_precision"], standard["mean_recall"])

    # ---- 2. 自定义匹配 ----
    image_paths = split_image_paths(detection_dir, args.split)
    logger.info("%s split: %d 张图", args.split, len(image_paths))
    gt_by_image = load_gt_instances(instances_csv, args.split, splits_csv)
    n_gt_total = sum(len(v) for v in gt_by_image.values())
    logger.info("GT 实例 (A-F, 非 pseudo-normal): %d", n_gt_total)

    preds_by_image = run_predictions(model, image_paths, args.imgsz)

    all_pred_records: List[Dict[str, Any]] = []
    all_missed_records: List[Dict[str, Any]] = []
    # 给 GT 记录补 image_id (用于 missed CSV)
    for img, gts in gt_by_image.items():
        for g in gts:
            g["image_id"] = img

    for img in sorted(set(gt_by_image.keys()) | set(preds_by_image.keys())):
        gts = gt_by_image.get(img, [])
        preds = preds_by_image.get(img, [])
        p_rec, m_rec, _ = match_image(gts, preds, iou_thr=MATCH_IOU)
        # 补 image_id 到 missed
        for m in m_rec:
            m["image_id"] = img
        all_pred_records.extend(p_rec)
        all_missed_records.extend(m_rec)

    write_csv(report_dir / "prediction_instances.csv",
              ["image_id", "prediction_id", "predicted_label", "confidence",
               "x_min", "y_min", "x_max", "y_max",
               "matched_ann_id", "matched_gt_label", "iou", "center_hit", "error_type"],
              all_pred_records)
    write_csv(report_dir / "missed_instances.csv",
              ["image_id", "ann_id", "gt_label", "x_min", "y_min", "x_max", "y_max",
               "bbox_area", "best_pred_iou", "best_pred_conf", "center_hit_by_pred", "error_hint"],
              all_missed_records)

    # 错误类型统计
    err_counts = defaultdict(int)
    for r in all_pred_records:
        err_counts[r["error_type"]] += 1
    logger.info("预测错误分布: %s", dict(err_counts))
    logger.info("漏检 GT: %d", len(all_missed_records))

    # ---- 3. C 类专项 ----
    c_metrics = compute_C_metrics(all_pred_records, all_missed_records, gt_by_image)
    (report_dir / f"class_C_metrics_{args.split}.json").write_text(
        json.dumps(c_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("C 类: R@0.5=%.3f, R@0.3=%.3f, R_center=%.3f, missed=%d, fp=%d",
                c_metrics["recall_iou50"], c_metrics["recall_iou30"], c_metrics["recall_center_hit"],
                c_metrics["missed_gallery_count"], c_metrics["false_positive_gallery_count"])

    # ---- 4. A 类专项 ----
    a_metrics = compute_A_metrics(all_pred_records, all_missed_records, gt_by_image, preds_by_image)
    (report_dir / f"class_A_metrics_{args.split}.json").write_text(
        json.dumps(a_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("A 类: bbox_R@0.5=%.3f, img_R=%.3f, img_F1=%.3f, 建议=%s",
                a_metrics["bbox_recall_iou50"], a_metrics["image_level"]["recall"],
                a_metrics["image_level"]["f1"], a_metrics["modeling_recommendation"])

    # ---- 5. 数量统计 ----
    count_stats = compute_count_statistics(all_pred_records, all_missed_records,
                                           gt_by_image, preds_by_image, args.split, report_dir)
    (report_dir / f"count_statistics_{args.split}.json").write_text(
        json.dumps(count_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("数量统计: 总 MAE=%.3f, RMSE=%.3f", count_stats["total"]["mae"], count_stats["total"]["rmse"])

    # ---- 6. 阈值分析 (仅 val) ----
    if args.split == "val":
        thr = threshold_analysis(all_pred_records, gt_by_image, report_dir)
        threshold_path.write_text(json.dumps(thr, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("阈值分析已写入: %s", threshold_path)

    # ---- 7. test 锁 ----
    if args.split == "test":
        split_list_path = detection_dir / "split_image_lists" / "test.txt"
        write_test_lock(report_dir, model_path, split_list_path, threshold_path)

    # ---- 汇总 markdown ----
    write_summary_md(report_dir, args.split, standard, c_metrics, a_metrics, count_stats,
                     err_counts, len(all_missed_records), len(all_pred_records), n_gt_total)
    logger.info("评估完成。报告目录: %s", report_dir)


def write_summary_md(report_dir: Path, split: str, standard: Dict, c_metrics: Dict,
                     a_metrics: Dict, count_stats: Dict, err_counts: Dict,
                     n_missed: int, n_preds: int, n_gt: int) -> None:
    lines = [
        f"# 检测评估报告 — {split.upper()}",
        "",
        f"- GT 实例 (A-F): {n_gt}",
        f"- 预测总数 (conf>={CONF_PRED}): {n_preds}",
        f"- 漏检 GT: {n_missed}",
        "",
        "## 标准指标 (ultralytics val, class-aware NMS)",
        f"- mAP50: **{standard['mAP50']:.4f}**",
        f"- mAP50-95: **{standard['mAP50_95']:.4f}**",
        f"- mean Precision: {standard['mean_precision']:.4f}",
        f"- mean Recall: {standard['mean_recall']:.4f}",
        "",
        "### 每类指标",
        "| 类别 | AP50 | AP50-95 | Precision | Recall |",
        "|------|------|---------|-----------|--------|",
    ]
    for c in CLASS_CODES:
        pc = standard["per_class"][c]
        lines.append(f"| {c} ({pc['name']}) | {pc['AP50']:.4f} | {pc['AP50_95']:.4f} | {pc['precision']:.4f} | {pc['recall']:.4f} |")
    lines += [
        "",
        f"曲线/混淆矩阵见 `ultralytics_val_{split}/` (BoxPR_curve.png / BoxF1_curve.png / BoxP_curve.png / BoxR_curve.png / confusion_matrix.png)。",
        "",
        "## 自定义匹配错误分布 (prediction_instances.csv)",
        "| error_type | 数量 |",
        "|------------|------|",
    ]
    for k in ["correct", "localization_error", "classification_error", "false_positive"]:
        lines.append(f"| {k} | {err_counts.get(k, 0)} |")
    lines += [
        "",
        f"漏检 GT (missed_instances.csv): {n_missed}",
        "",
        "## C 类小目标专项",
        f"- GT 数: {c_metrics['n_gt_C']}",
        f"- IoU@0.5 recall: {c_metrics['recall_iou50']:.4f}",
        f"- IoU@0.3 recall: {c_metrics['recall_iou30']:.4f}",
        f"- center-hit recall: {c_metrics['recall_center_hit']:.4f}",
        "",
        "### 按面积分组",
        "| 分组 | 面积范围 (px²) | GT | TP | recall |",
        "|------|----------------|----|----|--------|",
    ]
    for b, v in c_metrics["area_bins"].items():
        lo, hi = C_AREA_BINS[b]
        rng = f"[{lo}, {int(hi) if hi != float('inf') else '∞'})"
        lines.append(f"| {b} | {rng} | {v['n_gt']} | {v['tp']} | {v['recall']:.4f} |")
    lines += [
        "",
        "## A 类专项",
        f"- GT 数: {a_metrics['n_gt_A']}",
        f"- bbox recall@0.5: {a_metrics['bbox_recall_iou50']:.4f}",
        f"- bbox recall@0.3: {a_metrics['bbox_recall_iou30']:.4f}",
        f"- 图片级 has_A: P={a_metrics['image_level']['precision']:.4f}, R={a_metrics['image_level']['recall']:.4f}, F1={a_metrics['image_level']['f1']:.4f}",
        f"- 建模建议: **{a_metrics['modeling_recommendation']}** — {a_metrics['recommendation_reason']}",
        "",
        "## 数量统计",
        f"- 总缺陷 count MAE: {count_stats['total']['mae']:.4f}",
        f"- 总缺陷 count RMSE: {count_stats['total']['rmse']:.4f}",
        f"- (count 阈值 conf>={count_stats['total']['count_conf_threshold']})",
        "",
        "### 每类 count MAE / RMSE",
        "| 类别 | GT | Pred | MAE | RMSE |",
        "|------|-----|------|-----|------|",
    ]
    for c in CLASS_CODES:
        s = count_stats["per_class"][c]
        lines.append(f"| {c} | {s['gt_total']} | {s['pred_total']} | {s['mae']:.4f} | {s['rmse']:.4f} |")
    lines += [
        "",
        f"散点图: count_scatter_{split}.png",
        f"逐图误差: per_image_count_error_{split}.csv",
    ]
    if split == "val":
        lines += ["", "## 阈值分析 (仅 val)", "见 class_thresholds_val.json 与 threshold_curves_val.png"]
    if split == "test":
        lines += ["", "## test 锁", "见 test_evaluation_lock.json (模型/阈值变化后不得覆盖)"]
    (report_dir / f"evaluation_summary_{split}.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
