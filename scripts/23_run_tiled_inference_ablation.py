#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C 类小目标切片推理消融（仅验证集）。

完整图预测保持不变，只把重叠切片产生的 C 类候选与完整图 C 类结果合并，
再做 class-aware NMS。脚本不读取 test，不修改模型或人工标注。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["A", "B", "C", "D", "E", "F"]
DEFAULT_THRESHOLDS = {
    "A": 0.25,
    "B": 0.15,
    "C": 0.10,
    "D": 0.30,
    "E": 0.20,
    "F": 0.50,
}


@dataclass(frozen=True)
class Detection:
    image_id: str
    label: str
    confidence: float
    box: Tuple[float, float, float, float]
    source: str


@dataclass(frozen=True)
class GroundTruth:
    image_id: str
    ann_id: str
    label: str
    box: Tuple[float, float, float, float]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_number_list(text: str, caster):
    return [caster(item.strip()) for item in text.split(",") if item.strip()]


def load_split_map(path: Path) -> Dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["image_id"]: row["split"] for row in csv.DictReader(handle)}


def load_ground_truth(instances_path: Path, split_map: Dict[str, str],
                      split: str) -> Dict[str, List[GroundTruth]]:
    result: Dict[str, List[GroundTruth]] = defaultdict(list)
    with instances_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if split_map.get(row["image_id"]) != split:
                continue
            if row["label_code"] not in CLASSES:
                continue
            if str(row["is_pseudo_normal"]).lower() == "true":
                continue
            result[row["image_id"]].append(
                GroundTruth(
                    image_id=row["image_id"],
                    ann_id=row["ann_id"],
                    label=row["label_code"],
                    box=(
                        float(row["x_min"]),
                        float(row["y_min"]),
                        float(row["x_max"]),
                        float(row["y_max"]),
                    ),
                )
            )
    return result


def load_full_predictions(path: Path) -> Dict[str, List[Detection]]:
    result: Dict[str, List[Detection]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["image_id"]].append(
                Detection(
                    image_id=row["image_id"],
                    label=row["predicted_label"],
                    confidence=float(row["confidence"]),
                    box=(
                        float(row["x_min"]),
                        float(row["y_min"]),
                        float(row["x_max"]),
                        float(row["y_max"]),
                    ),
                    source="full",
                )
            )
    return result


def filter_full_predictions(
    predictions: Dict[str, List[Detection]],
    thresholds: Dict[str, float],
) -> Dict[str, List[Detection]]:
    return {
        image_id: [
            det for det in detections
            if det.confidence >= thresholds[det.label]
        ]
        for image_id, detections in predictions.items()
    }


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    union += max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def center_in(box: Sequence[float], target: Sequence[float]) -> bool:
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = target
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return tx1 <= cx <= tx2 and ty1 <= cy <= ty2


def nms(detections: List[Detection], threshold: float) -> List[Detection]:
    if not detections:
        return []
    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: List[Detection] = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        ordered = [candidate for candidate in ordered if iou(current.box, candidate.box) < threshold]
    return kept


def starts_for_axis(length: int, tile_size: int, overlap: float) -> List[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return sorted(set(starts))


def tile_image(image: np.ndarray, tile_size: int, overlap: float):
    height, width = image.shape[:2]
    tiles = []
    for y0 in starts_for_axis(height, tile_size, overlap):
        for x0 in starts_for_axis(width, tile_size, overlap):
            x1, y1 = min(width, x0 + tile_size), min(height, y0 + tile_size)
            tiles.append((image[y0:y1, x0:x1].copy(), x0, y0, x1, y1, width, height))
    return tiles


def is_internal_border_prediction(
    local_box: Sequence[float],
    tile_meta: Tuple[int, int, int, int, int, int],
    margin: int,
) -> bool:
    x0, y0, x1, y1, image_width, image_height = tile_meta
    bx1, by1, bx2, by2 = local_box
    cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    tile_width, tile_height = x1 - x0, y1 - y0
    if x0 > 0 and cx < margin:
        return True
    if x1 < image_width and cx > tile_width - margin:
        return True
    if y0 > 0 and cy < margin:
        return True
    if y1 < image_height and cy > tile_height - margin:
        return True
    return False


def run_tiled_c_predictions(
    model,
    image_paths: List[Path],
    tile_size: int,
    overlap: float,
    imgsz: int,
    min_conf: float,
    border_margin: int,
    device: str,
) -> Tuple[Dict[str, List[Detection]], float]:
    start = time.perf_counter()
    result: Dict[str, List[Detection]] = defaultdict(list)
    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"无法读取图片: {image_path}")
        tile_records = tile_image(image, tile_size, overlap)
        tile_arrays = [record[0] for record in tile_records]
        predictions = model.predict(
            source=tile_arrays,
            imgsz=imgsz,
            conf=min_conf,
            iou=0.5,
            max_det=300,
            agnostic_nms=False,
            device=device,
            verbose=False,
        )
        for prediction, record in zip(predictions, tile_records):
            _, x0, y0, x1, y1, image_width, image_height = record
            boxes = prediction.boxes
            for box_index in range(len(boxes)):
                class_index = int(boxes.cls[box_index])
                if CLASSES[class_index] != "C":
                    continue
                local_box = tuple(float(value) for value in boxes.xyxy[box_index].tolist())
                if is_internal_border_prediction(
                    local_box,
                    (x0, y0, x1, y1, image_width, image_height),
                    border_margin,
                ):
                    continue
                bx1, by1, bx2, by2 = local_box
                global_box = (
                    max(0.0, bx1 + x0),
                    max(0.0, by1 + y0),
                    min(float(image_width), bx2 + x0),
                    min(float(image_height), by2 + y0),
                )
                result[image_path.stem].append(
                    Detection(
                        image_id=image_path.stem,
                        label="C",
                        confidence=float(boxes.conf[box_index]),
                        box=global_box,
                        source=f"tile_{tile_size}_{overlap:.2f}",
                    )
                )
        if index % 10 == 0:
            print(f"  tile={tile_size}, overlap={overlap:.2f}: {index}/{len(image_paths)}")
    return result, time.perf_counter() - start


def match_count(
    ground_truth: List[GroundTruth],
    predictions: List[Detection],
    label: str,
    criterion: str,
    threshold: float,
) -> Tuple[int, int, int]:
    gt_items = [item for item in ground_truth if item.label == label]
    pred_items = sorted(
        [item for item in predictions if item.label == label],
        key=lambda item: item.confidence,
        reverse=True,
    )
    unmatched = set(range(len(gt_items)))
    true_positive = 0
    for prediction in pred_items:
        best_index = None
        best_score = -1.0
        for gt_index in unmatched:
            if criterion == "center":
                score = 1.0 if center_in(prediction.box, gt_items[gt_index].box) else 0.0
            else:
                score = iou(prediction.box, gt_items[gt_index].box)
            if score >= threshold and score > best_score:
                best_score = score
                best_index = gt_index
        if best_index is not None:
            unmatched.remove(best_index)
            true_positive += 1
    false_positive = len(pred_items) - true_positive
    false_negative = len(gt_items) - true_positive
    return true_positive, false_positive, false_negative


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(
    image_ids: List[str],
    gt_by_image: Dict[str, List[GroundTruth]],
    pred_by_image: Dict[str, List[Detection]],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    class_f1 = []
    total_count_errors = []
    c_count_errors = []
    for label in CLASSES:
        totals = np.zeros(3, dtype=np.int64)
        for image_id in image_ids:
            totals += np.asarray(
                match_count(
                    gt_by_image.get(image_id, []),
                    pred_by_image.get(image_id, []),
                    label,
                    "iou",
                    0.5,
                )
            )
        precision, recall, f1 = prf(*totals.tolist())
        metrics[f"P_{label}"] = precision
        metrics[f"R_{label}"] = recall
        metrics[f"F1_{label}"] = f1
        class_f1.append(f1)
    for criterion, threshold, suffix in [
        ("iou", 0.3, "iou30"),
        ("center", 1.0, "center"),
    ]:
        totals = np.zeros(3, dtype=np.int64)
        for image_id in image_ids:
            totals += np.asarray(
                match_count(
                    gt_by_image.get(image_id, []),
                    pred_by_image.get(image_id, []),
                    "C",
                    criterion,
                    threshold,
                )
            )
        metrics[f"C_recall_{suffix}"] = prf(*totals.tolist())[1]
    for image_id in image_ids:
        gt_items = gt_by_image.get(image_id, [])
        pred_items = pred_by_image.get(image_id, [])
        total_count_errors.append(abs(len(pred_items) - len(gt_items)))
        gt_c = sum(item.label == "C" for item in gt_items)
        pred_c = sum(item.label == "C" for item in pred_items)
        c_count_errors.append(abs(pred_c - gt_c))
    metrics["macro_F1"] = float(np.mean(class_f1))
    metrics["total_count_MAE"] = float(np.mean(total_count_errors))
    metrics["C_count_MAE"] = float(np.mean(c_count_errors))
    metrics["prediction_count"] = float(sum(len(pred_by_image.get(image_id, [])) for image_id in image_ids))
    return metrics


def combine_predictions(
    image_ids: List[str],
    full_predictions: Dict[str, List[Detection]],
    tile_predictions: Dict[str, List[Detection]],
    tile_conf: float,
    nms_iou: float,
) -> Dict[str, List[Detection]]:
    combined: Dict[str, List[Detection]] = {}
    for image_id in image_ids:
        full = list(full_predictions.get(image_id, []))
        non_c = [item for item in full if item.label != "C"]
        c_items = [item for item in full if item.label == "C"]
        c_items.extend(
            item for item in tile_predictions.get(image_id, [])
            if item.confidence >= tile_conf
        )
        combined[image_id] = non_c + nms(c_items, nms_iou)
    return combined


def write_prediction_csv(path: Path, predictions: Dict[str, List[Detection]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_id", "prediction_id", "predicted_label", "confidence",
        "x_min", "y_min", "x_max", "y_max", "source",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image_id in sorted(predictions):
            ordered = sorted(predictions[image_id], key=lambda item: item.confidence, reverse=True)
            for index, item in enumerate(ordered):
                writer.writerow(
                    {
                        "image_id": image_id,
                        "prediction_id": f"{image_id}_tiled_{index:03d}",
                        "predicted_label": item.label,
                        "confidence": round(item.confidence, 6),
                        "x_min": round(item.box[0], 2),
                        "y_min": round(item.box[1], 2),
                        "x_max": round(item.box[2], 2),
                        "y_max": round(item.box[3], 2),
                        "source": item.source,
                    }
                )


def draw_panel(
    image: np.ndarray,
    title: str,
    gt_items: List[GroundTruth],
    predictions: List[Detection],
) -> np.ndarray:
    panel = image.copy()
    for gt in gt_items:
        if gt.label != "C":
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in gt.box]
        cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 220, 0), 2)
    for prediction in predictions:
        if prediction.label != "C":
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in prediction.box]
        color = (255, 100, 0) if prediction.source == "full" else (220, 0, 220)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 1)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(panel, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return panel


def per_image_c_tp(
    image_id: str,
    gt_items: List[GroundTruth],
    predictions: List[Detection],
) -> int:
    return match_count(gt_items, predictions, "C", "iou", 0.5)[0]


def make_gallery(
    output_dir: Path,
    image_paths: List[Path],
    gt_by_image: Dict[str, List[GroundTruth]],
    baseline: Dict[str, List[Detection]],
    best: Dict[str, List[Detection]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = []
    image_path_map = {path.stem: path for path in image_paths}
    for image_id in image_path_map:
        baseline_tp = per_image_c_tp(
            image_id, gt_by_image.get(image_id, []), baseline.get(image_id, [])
        )
        best_tp = per_image_c_tp(
            image_id, gt_by_image.get(image_id, []), best.get(image_id, [])
        )
        ranked.append((best_tp - baseline_tp, image_id, baseline_tp, best_tp))
    selected = sorted(ranked, reverse=True)[:6] + sorted(ranked)[:4]
    seen = set()
    for delta, image_id, baseline_tp, best_tp in selected:
        if image_id in seen:
            continue
        seen.add(image_id)
        image = cv2.imread(str(image_path_map[image_id]))
        if image is None:
            continue
        gt_items = gt_by_image.get(image_id, [])
        left = draw_panel(
            image,
            f"baseline C TP={baseline_tp}",
            gt_items,
            baseline.get(image_id, []),
        )
        right = draw_panel(
            image,
            f"tiled C TP={best_tp}, delta={delta:+d}",
            gt_items,
            best.get(image_id, []),
        )
        combined = np.hstack([left, right])
        cv2.imwrite(str(output_dir / f"{image_id}_delta_{delta:+d}.jpg"), combined)


def main() -> None:
    parser = argparse.ArgumentParser(description="C 类切片推理消融（val only）")
    parser.add_argument(
        "--model",
        default="models/detector/epoch150_20260720/baseline_best.pt",
    )
    parser.add_argument(
        "--full-predictions",
        default="reports/detection_evaluation/epoch150_20260720/prediction_instances.csv",
    )
    parser.add_argument("--instances", default="data/annotations/instances.csv")
    parser.add_argument("--splits", default="data/derived/splits.csv")
    parser.add_argument("--image-dir", default="data/derived/detection/images/val")
    parser.add_argument("--output-dir", default="reports/optimization_codex/tiled_inference")
    parser.add_argument("--tile-sizes", default="256,320,384")
    parser.add_argument("--overlaps", default="0.20")
    parser.add_argument("--tile-conf-thresholds", default="0.03,0.05,0.10,0.15,0.20")
    parser.add_argument("--nms-thresholds", default="0.30,0.50")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-tile-conf", type=float, default=0.01)
    parser.add_argument("--border-margin", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    model_path = (ROOT / args.model).resolve()
    predictions_path = (ROOT / args.full_predictions).resolve()
    instances_path = (ROOT / args.instances).resolve()
    splits_path = (ROOT / args.splits).resolve()
    image_dir = (ROOT / args.image_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for required in [model_path, predictions_path, instances_path, splits_path, image_dir]:
        if not required.exists():
            raise FileNotFoundError(required)

    tile_sizes = parse_number_list(args.tile_sizes, int)
    overlaps = parse_number_list(args.overlaps, float)
    tile_conf_thresholds = parse_number_list(args.tile_conf_thresholds, float)
    nms_thresholds = parse_number_list(args.nms_thresholds, float)
    split_map = load_split_map(splits_path)
    gt_by_image = load_ground_truth(instances_path, split_map, "val")
    image_paths = sorted(image_dir.glob("*.jpg"))
    image_ids = [path.stem for path in image_paths]
    if len(image_paths) != 30:
        raise RuntimeError(f"预期 val 有 30 张图，实际 {len(image_paths)}")

    full_raw = load_full_predictions(predictions_path)
    full_filtered = filter_full_predictions(full_raw, DEFAULT_THRESHOLDS)
    baseline_metrics = evaluate(image_ids, gt_by_image, full_filtered)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    rows = [
        {
            "experiment": "baseline_full",
            "tile_size": 0,
            "overlap": 0.0,
            "tile_conf": DEFAULT_THRESHOLDS["C"],
            "nms_iou": 0.5,
            "inference_seconds": 0.0,
            **baseline_metrics,
        }
    ]
    cached_predictions: Dict[Tuple[int, float], Dict[str, List[Detection]]] = {}
    cached_seconds: Dict[Tuple[int, float], float] = {}

    for tile_size in tile_sizes:
        for overlap in overlaps:
            cache_path = output_dir / f"tile_raw_{tile_size}_ov{overlap:.2f}.csv"
            if args.reuse_cache and cache_path.exists():
                tile_predictions = load_full_predictions(cache_path)
                seconds = 0.0
            else:
                tile_predictions, seconds = run_tiled_c_predictions(
                    model=model,
                    image_paths=image_paths,
                    tile_size=tile_size,
                    overlap=overlap,
                    imgsz=args.imgsz,
                    min_conf=args.min_tile_conf,
                    border_margin=args.border_margin,
                    device=args.device,
                )
                write_prediction_csv(cache_path, tile_predictions)
            cached_predictions[(tile_size, overlap)] = tile_predictions
            cached_seconds[(tile_size, overlap)] = seconds
            for tile_conf in tile_conf_thresholds:
                for nms_iou in nms_thresholds:
                    combined = combine_predictions(
                        image_ids,
                        full_filtered,
                        tile_predictions,
                        tile_conf,
                        nms_iou,
                    )
                    metrics = evaluate(image_ids, gt_by_image, combined)
                    rows.append(
                        {
                            "experiment": "full_plus_C_tiles",
                            "tile_size": tile_size,
                            "overlap": overlap,
                            "tile_conf": tile_conf,
                            "nms_iou": nms_iou,
                            "inference_seconds": seconds,
                            **metrics,
                        }
                    )

    result_path = output_dir / "ablation_results.csv"
    fields = list(rows[0].keys())
    with result_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    candidates = [row for row in rows if row["experiment"] != "baseline_full"]
    best = max(
        candidates,
        key=lambda row: (
            row["macro_F1"],
            row["F1_C"],
            row["R_C"],
            -row["C_count_MAE"],
        ),
    )
    best_key = (int(best["tile_size"]), float(best["overlap"]))
    best_predictions = combine_predictions(
        image_ids,
        full_filtered,
        cached_predictions[best_key],
        float(best["tile_conf"]),
        float(best["nms_iou"]),
    )
    write_prediction_csv(output_dir / "predictions_best_val.csv", best_predictions)
    make_gallery(
        output_dir / "gallery",
        image_paths,
        gt_by_image,
        full_filtered,
        best_predictions,
    )

    best_config = {
        "model": str(model_path.relative_to(ROOT)),
        "model_sha256": sha256(model_path),
        "split": "val",
        "imgsz": args.imgsz,
        "tile_size": int(best["tile_size"]),
        "overlap": float(best["overlap"]),
        "tile_conf": float(best["tile_conf"]),
        "nms_iou": float(best["nms_iou"]),
        "border_margin": args.border_margin,
        "selection_objective": "macro_F1, then C F1/recall, then C count MAE",
        "baseline_metrics": baseline_metrics,
        "best_metrics": {
            key: value
            for key, value in best.items()
            if key not in {"experiment"}
        },
    }
    (output_dir / "best_config.json").write_text(
        json.dumps(best_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    delta_macro = float(best["macro_F1"]) - baseline_metrics["macro_F1"]
    delta_c_f1 = float(best["F1_C"]) - baseline_metrics["F1_C"]
    delta_c_recall = float(best["R_C"]) - baseline_metrics["R_C"]
    report_lines = [
        "# C 类切片推理消融（VAL）",
        "",
        "本实验只在验证集选择配置，未读取 test。完整图 A/B/D/E/F 预测保持不变，"
        "仅将重叠切片的 C 类候选与完整图 C 类结果融合。",
        "",
        "## 基线",
        "",
        f"- Macro-F1: {baseline_metrics['macro_F1']:.4f}",
        f"- C Precision / Recall / F1: {baseline_metrics['P_C']:.4f} / "
        f"{baseline_metrics['R_C']:.4f} / {baseline_metrics['F1_C']:.4f}",
        f"- C center-hit recall: {baseline_metrics['C_recall_center']:.4f}",
        f"- C count MAE: {baseline_metrics['C_count_MAE']:.4f}",
        "",
        "## 最佳切片配置",
        "",
        f"- tile_size: {int(best['tile_size'])}",
        f"- overlap: {float(best['overlap']):.2f}",
        f"- tile confidence: {float(best['tile_conf']):.2f}",
        f"- C NMS IoU: {float(best['nms_iou']):.2f}",
        f"- Macro-F1: {float(best['macro_F1']):.4f} ({delta_macro:+.4f})",
        f"- C Precision / Recall / F1: {float(best['P_C']):.4f} / "
        f"{float(best['R_C']):.4f} / {float(best['F1_C']):.4f}",
        f"- C F1 变化: {delta_c_f1:+.4f}",
        f"- C Recall 变化: {delta_c_recall:+.4f}",
        f"- C center-hit recall: {float(best['C_recall_center']):.4f}",
        f"- C count MAE: {float(best['C_count_MAE']):.4f}",
        "",
        "## 结论规则",
        "",
        "- 若 Macro-F1 和 C F1 同时提升，切片推理可作为最终候选方案。",
        "- 若只提高 Recall 但显著降低 Precision/F1，则只作为高召回人工复核模式。",
        "- 若没有稳定提升，则保留 640 完整图基线，避免增加部署复杂度。",
    ]
    (output_dir / "tiled_inference_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
