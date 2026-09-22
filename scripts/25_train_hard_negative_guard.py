#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用训练图真实检测框训练 hard-negative 背景守门员（val only 选型）。

守门员只回答“该预测框是否对应某个人工缺陷区域”，不重判 A-F 类别。
它从训练图检测器输出中学习真实候选与背景误报的差异，再分别作用于：
1) 完整图 + 逐类阈值基线；
2) 已选出的完整图 + C 切片方案。

不读取 test，不修改模型、原图、标注或既有划分。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from aoi_defect.patch_features import PatchFeatureExtractor  # noqa: E402


CLASSES = ["A", "B", "C", "D", "E", "F"]
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
DEFAULT_THRESHOLDS = {
    "A": 0.25,
    "B": 0.15,
    "C": 0.10,
    "D": 0.30,
    "E": 0.20,
    "F": 0.50,
}
CONTEXT = {
    "A": (1.2, 96),
    "B": (1.8, 96),
    "C": (3.0, 96),
    "D": (1.5, 128),
    "E": (1.8, 96),
    "F": (1.8, 96),
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
    label: str
    box: Tuple[float, float, float, float]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_map(path: Path) -> Dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["image_id"]: row["split"] for row in csv.DictReader(handle)}


def load_ground_truth(
    path: Path,
    split_map: Dict[str, str],
    split: str,
) -> Dict[str, List[GroundTruth]]:
    result: Dict[str, List[GroundTruth]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
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


def load_predictions(
    path: Path,
    split_map: Dict[str, str],
    split: str,
    thresholds: Dict[str, float] | None,
) -> Dict[str, List[Detection]]:
    result: Dict[str, List[Detection]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]
            if split_map.get(image_id) != split:
                continue
            label = row["predicted_label"]
            confidence = float(row["confidence"])
            if thresholds is not None and confidence < thresholds[label]:
                continue
            result[image_id].append(
                Detection(
                    image_id=image_id,
                    label=label,
                    confidence=confidence,
                    box=(
                        float(row["x_min"]),
                        float(row["y_min"]),
                        float(row["x_max"]),
                        float(row["y_max"]),
                    ),
                    source=row.get("source", "full") or "full",
                )
            )
    return result


def flatten(predictions: Dict[str, List[Detection]]) -> List[Detection]:
    return [
        detection
        for image_id in sorted(predictions)
        for detection in predictions[image_id]
    ]


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def center_in(box: Sequence[float], target: Sequence[float]) -> bool:
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = target
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return tx1 <= cx <= tx2 and ty1 <= cy <= ty2


def spatial_binary_label(
    detection: Detection,
    ground_truth: List[GroundTruth],
) -> int:
    """1=真实缺陷邻域，0=明确背景，-1=空间歧义。"""
    if not ground_truth:
        return 0
    maximum_iou = max(iou(detection.box, item.box) for item in ground_truth)
    center_inside = any(center_in(detection.box, item.box) for item in ground_truth)
    if maximum_iou >= 0.30 or center_inside:
        return 1
    if maximum_iou < 0.02:
        return 0
    return -1


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片: {path}")
    return image


def crop_context(
    image: np.ndarray,
    detection: Detection,
) -> Tuple[np.ndarray, Dict[str, float]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = detection.box
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    factor, minimum = CONTEXT[detection.label]
    half = max(minimum / 2.0, factor * max(box_width, box_height) / 2.0)
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    crop_x1 = max(0, int(math.floor(center_x - half)))
    crop_y1 = max(0, int(math.floor(center_y - half)))
    crop_x2 = min(width, int(math.ceil(center_x + half)))
    crop_y2 = min(height, int(math.ceil(center_y + half)))
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        crop = np.zeros((128, 128, 3), dtype=np.uint8)
    geometry = {
        "bbox_width": box_width,
        "bbox_height": box_height,
        "bbox_area": box_width * box_height,
        "patch_area": max(1.0, float((crop_x2 - crop_x1) * (crop_y2 - crop_y1))),
        "image_width": float(width),
        "image_height": float(height),
        "center_x": center_x,
        "center_y": center_y,
    }
    return crop, geometry


def add_deployment_features(
    base_features: np.ndarray,
    detection: Detection,
    geometry: Dict[str, float],
) -> np.ndarray:
    width = geometry["image_width"]
    height = geometry["image_height"]
    box_width = geometry["bbox_width"]
    box_height = geometry["bbox_height"]
    area_ratio = geometry["bbox_area"] / max(1.0, width * height)
    one_hot = np.zeros(len(CLASSES), dtype=np.float32)
    one_hot[CLASS_TO_INDEX[detection.label]] = 1.0
    extra = np.asarray(
        [
            detection.confidence,
            geometry["center_x"] / width,
            geometry["center_y"] / height,
            box_width / width,
            box_height / height,
            area_ratio,
            math.log1p(area_ratio),
            math.log(max(box_width / box_height, 1e-6)),
            1.0 if detection.source.startswith("tile") else 0.0,
            *one_hot.tolist(),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base_features.astype(np.float32, copy=False), extra])


def extract_detection_features(
    detections: List[Detection],
    image_dir: Path,
    cache_path: Path,
    source_hash: str,
) -> np.ndarray:
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("source_hash") == source_hash
            and metadata.get("detection_count") == len(detections)
        ):
            cached = np.load(cache_path)
            print(f"[cache] {cache_path.name}: {cached.shape}")
            return cached

    extractor = PatchFeatureExtractor(resize_to=128)
    features: List[np.ndarray] = []
    image_cache: Dict[str, np.ndarray] = {}
    start = time.perf_counter()
    for index, detection in enumerate(detections, start=1):
        if detection.image_id not in image_cache:
            image_cache[detection.image_id] = read_image(
                image_dir / f"{detection.image_id}.jpg"
            )
        crop, geometry = crop_context(image_cache[detection.image_id], detection)
        base = extractor.extract(
            crop,
            bbox_width=geometry["bbox_width"],
            bbox_height=geometry["bbox_height"],
            bbox_area=geometry["bbox_area"],
            patch_area=geometry["patch_area"],
            polygon_area=geometry["bbox_area"],
        )
        features.append(add_deployment_features(base, detection, geometry))
        if index % 500 == 0:
            print(
                f"  features {index}/{len(detections)} "
                f"({time.perf_counter() - start:.1f}s)"
            )
    matrix = np.asarray(features, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix)
    metadata_path.write_text(
        json.dumps(
            {
                "source_hash": source_hash,
                "detection_count": len(detections),
                "feature_dim": int(matrix.shape[1]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return matrix


def make_guard(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=350,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )


def choose_oof_threshold(
    labels: np.ndarray,
    defect_probabilities: np.ndarray,
    min_positive_retention: float,
) -> Tuple[float, List[Dict[str, float]]]:
    rows = []
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.50, 101),
                np.quantile(defect_probabilities, np.linspace(0.0, 0.5, 101)),
            ]
        )
    )
    for threshold in sorted(candidates):
        keep = defect_probabilities >= threshold
        positive_retention = float(np.mean(keep[labels == 1]))
        negative_rejection = float(np.mean(~keep[labels == 0]))
        rows.append(
            {
                "threshold": float(threshold),
                "positive_retention": positive_retention,
                "negative_rejection": negative_rejection,
            }
        )
    feasible = [
        row for row in rows
        if row["positive_retention"] >= min_positive_retention
    ]
    best = max(
        feasible,
        key=lambda row: (row["negative_rejection"], row["threshold"]),
    )
    return float(best["threshold"]), rows


def match_count(
    ground_truth: List[GroundTruth],
    predictions: List[Detection],
    label: str,
    threshold: float = 0.5,
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
            score = iou(prediction.box, gt_items[gt_index].box)
            if score >= threshold and score > best_score:
                best_score = score
                best_index = gt_index
        if best_index is not None:
            unmatched.remove(best_index)
            true_positive += 1
    return true_positive, len(pred_items) - true_positive, len(gt_items) - true_positive


def prf(triplet: Sequence[int]) -> Tuple[float, float, float]:
    tp, fp, fn = triplet
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
    class_recall = []
    total_tp = 0
    count_errors = []
    c_count_errors = []
    for label in CLASSES:
        totals = np.zeros(3, dtype=np.int64)
        for image_id in image_ids:
            totals += np.asarray(
                match_count(
                    gt_by_image.get(image_id, []),
                    pred_by_image.get(image_id, []),
                    label,
                )
            )
        precision, recall, f1 = prf(totals.tolist())
        metrics[f"P_{label}"] = precision
        metrics[f"R_{label}"] = recall
        metrics[f"F1_{label}"] = f1
        metrics[f"TP_{label}"] = float(totals[0])
        class_f1.append(f1)
        class_recall.append(recall)
        total_tp += int(totals[0])
    for image_id in image_ids:
        gt_items = gt_by_image.get(image_id, [])
        pred_items = pred_by_image.get(image_id, [])
        count_errors.append(abs(len(pred_items) - len(gt_items)))
        gt_c = sum(item.label == "C" for item in gt_items)
        pred_c = sum(item.label == "C" for item in pred_items)
        c_count_errors.append(abs(pred_c - gt_c))
    metrics["macro_F1"] = float(np.mean(class_f1))
    metrics["macro_recall"] = float(np.mean(class_recall))
    metrics["total_TP"] = float(total_tp)
    metrics["total_count_MAE"] = float(np.mean(count_errors))
    metrics["C_count_MAE"] = float(np.mean(c_count_errors))
    metrics["prediction_count"] = float(
        sum(len(pred_by_image.get(image_id, [])) for image_id in image_ids)
    )
    return metrics


def apply_guard(
    detections: List[Detection],
    probabilities: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, List[Detection]], int]:
    result: Dict[str, List[Detection]] = defaultdict(list)
    removed = 0
    for detection, probability in zip(detections, probabilities):
        if probability < threshold:
            removed += 1
            continue
        result[detection.image_id].append(detection)
    return result, removed


def write_predictions(
    path: Path,
    predictions: Dict[str, List[Detection]],
    probabilities: Dict[Tuple, float] | None = None,
) -> None:
    fields = [
        "image_id", "prediction_id", "predicted_label", "confidence",
        "x_min", "y_min", "x_max", "y_max", "source", "guard_defect_probability",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image_id in sorted(predictions):
            for index, detection in enumerate(
                sorted(
                    predictions[image_id],
                    key=lambda item: item.confidence,
                    reverse=True,
                )
            ):
                key = (
                    detection.image_id,
                    detection.label,
                    detection.confidence,
                    detection.box,
                    detection.source,
                )
                writer.writerow(
                    {
                        "image_id": image_id,
                        "prediction_id": f"{image_id}_guard_{index:03d}",
                        "predicted_label": detection.label,
                        "confidence": round(detection.confidence, 6),
                        "x_min": round(detection.box[0], 2),
                        "y_min": round(detection.box[1], 2),
                        "x_max": round(detection.box[2], 2),
                        "y_max": round(detection.box[3], 2),
                        "source": detection.source,
                        "guard_defect_probability": (
                            round(probabilities[key], 6)
                            if probabilities is not None and key in probabilities
                            else ""
                        ),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="训练检测框 hard-negative 背景守门员")
    parser.add_argument(
        "--train-predictions",
        default="reports/optimization_phase2/gpu/train_preds_ep150_640.csv",
    )
    parser.add_argument(
        "--val-predictions",
        default="reports/detection_evaluation/epoch150_20260720/prediction_instances.csv",
    )
    parser.add_argument(
        "--tiled-val-predictions",
        default="reports/optimization_codex/tiled_inference/predictions_best_val.csv",
    )
    parser.add_argument("--instances", default="data/annotations/instances.csv")
    parser.add_argument("--splits", default="data/derived/splits.csv")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument(
        "--output-dir",
        default="reports/optimization_codex/hard_negative_guard",
    )
    parser.add_argument(
        "--model-out",
        default="models/hard_negative_guard/guard_bundle.joblib",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-oof-positive-retention", type=float, default=0.995)
    args = parser.parse_args()

    paths = {
        "train_predictions": (ROOT / args.train_predictions).resolve(),
        "val_predictions": (ROOT / args.val_predictions).resolve(),
        "tiled_predictions": (ROOT / args.tiled_val_predictions).resolve(),
        "instances": (ROOT / args.instances).resolve(),
        "splits": (ROOT / args.splits).resolve(),
        "image_dir": (ROOT / args.image_dir).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = (ROOT / args.output_dir).resolve()
    model_out = (ROOT / args.model_out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "data/derived/hard_negative_guard"
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_map = load_split_map(paths["splits"])
    train_gt = load_ground_truth(paths["instances"], split_map, "train")
    val_gt = load_ground_truth(paths["instances"], split_map, "val")
    train_predictions = load_predictions(
        paths["train_predictions"], split_map, "train", DEFAULT_THRESHOLDS
    )
    val_predictions = load_predictions(
        paths["val_predictions"], split_map, "val", DEFAULT_THRESHOLDS
    )
    tiled_predictions = load_predictions(
        paths["tiled_predictions"], split_map, "val", None
    )
    train_flat_all = flatten(train_predictions)
    train_labels_all = np.asarray(
        [
            spatial_binary_label(item, train_gt.get(item.image_id, []))
            for item in train_flat_all
        ],
        dtype=np.int8,
    )
    keep_train = train_labels_all >= 0
    train_flat = [
        item for item, keep in zip(train_flat_all, keep_train)
        if keep
    ]
    train_labels = train_labels_all[keep_train]
    train_groups = np.asarray([item.image_id for item in train_flat])
    print(
        f"train candidates={len(train_flat_all)}, usable={len(train_flat)}, "
        f"positive={int((train_labels == 1).sum())}, "
        f"hard_negative={int((train_labels == 0).sum())}, "
        f"ambiguous={int((train_labels_all < 0).sum())}"
    )

    train_hash = sha256(paths["train_predictions"])
    train_features = extract_detection_features(
        train_flat,
        paths["image_dir"],
        cache_dir / f"train_{train_hash[:12]}.npy",
        train_hash,
    )

    group_folds = GroupKFold(n_splits=5)
    oof_probabilities = np.zeros(len(train_labels), dtype=np.float64)
    for fold, (train_indices, valid_indices) in enumerate(
        group_folds.split(train_features, train_labels, groups=train_groups),
        start=1,
    ):
        model = make_guard(args.seed + fold)
        model.fit(train_features[train_indices], train_labels[train_indices])
        oof_probabilities[valid_indices] = model.predict_proba(
            train_features[valid_indices]
        )[:, 1]
        print(f"  GroupKFold {fold}/5 complete")
    oof_threshold, oof_rows = choose_oof_threshold(
        train_labels,
        oof_probabilities,
        args.min_oof_positive_retention,
    )
    print(f"OOF conservative guard threshold={oof_threshold:.6f}")

    with (output_dir / "oof_threshold_curve.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(oof_rows[0].keys()))
        writer.writeheader()
        writer.writerows(oof_rows)

    final_model = make_guard(args.seed)
    final_model.fit(train_features, train_labels)

    val_variants = {
        "full": val_predictions,
        "full_plus_C_tiles": tiled_predictions,
    }
    thresholds = sorted(
        set(
            [
                0.0,
                oof_threshold,
                0.005,
                0.01,
                0.02,
                0.03,
                0.05,
                0.075,
                0.10,
                0.15,
                0.20,
                0.30,
                0.40,
                0.50,
            ]
        )
    )
    val_image_ids = sorted(
        image_id for image_id, split in split_map.items()
        if split == "val"
    )
    all_rows = []
    best_by_variant = {}
    probability_maps = {}
    detections_by_variant = {}
    probabilities_by_variant = {}

    for variant, predictions in val_variants.items():
        flat = flatten(predictions)
        source_path = (
            paths["val_predictions"]
            if variant == "full"
            else paths["tiled_predictions"]
        )
        source_hash = sha256(source_path)
        features = extract_detection_features(
            flat,
            paths["image_dir"],
            cache_dir / f"{variant}_{source_hash[:12]}.npy",
            source_hash,
        )
        probabilities = final_model.predict_proba(features)[:, 1]
        detections_by_variant[variant] = flat
        probabilities_by_variant[variant] = probabilities
        probability_maps[variant] = {
            (
                item.image_id,
                item.label,
                item.confidence,
                item.box,
                item.source,
            ): float(probability)
            for item, probability in zip(flat, probabilities)
        }
        baseline_metrics = evaluate(val_image_ids, val_gt, predictions)
        baseline_tp = baseline_metrics["total_TP"]
        baseline_recalls = {
            label: baseline_metrics[f"R_{label}"] for label in CLASSES
        }
        variant_rows = []
        for threshold in thresholds:
            guarded, removed = apply_guard(flat, probabilities, threshold)
            metrics = evaluate(val_image_ids, val_gt, guarded)
            retained_tp = (
                metrics["total_TP"] / baseline_tp if baseline_tp > 0 else 1.0
            )
            minimum_recall_delta = min(
                metrics[f"R_{label}"] - baseline_recalls[label]
                for label in CLASSES
            )
            row = {
                "variant": variant,
                "guard_threshold": threshold,
                "removed": removed,
                "removed_ratio": removed / len(flat) if flat else 0.0,
                "retained_TP": retained_tp,
                "minimum_class_recall_delta": minimum_recall_delta,
                **metrics,
            }
            all_rows.append(row)
            variant_rows.append((row, guarded))

        safe = [
            item for item in variant_rows
            if item[0]["retained_TP"] >= 0.98
            and item[0]["minimum_class_recall_delta"] >= -0.02
        ]
        best_row, best_predictions = max(
            safe,
            key=lambda item: (
                item[0]["macro_F1"],
                -item[0]["total_count_MAE"],
                item[0]["removed"],
            ),
        )
        best_by_variant[variant] = {
            "baseline_metrics": baseline_metrics,
            "best_row": best_row,
            "predictions": best_predictions,
        }
        write_predictions(
            output_dir / f"predictions_{variant}_guarded_val.csv",
            best_predictions,
            probability_maps[variant],
        )

    fields = list(all_rows[0].keys())
    with (output_dir / "guard_ablation.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    bundle = {
        "model": final_model,
        "classes": [0, 1],
        "meaning": {"0": "hard_negative_background", "1": "defect_region"},
        "feature_layout": "PatchFeatureExtractor + detector/geometry/source/label one-hot",
        "oof_threshold": oof_threshold,
        "thresholds_selected_on_val": {
            variant: result["best_row"]["guard_threshold"]
            for variant, result in best_by_variant.items()
        },
        "train_prediction_sha256": train_hash,
        "instances_sha256": sha256(paths["instances"]),
        "split_sha256": sha256(paths["splits"]),
        "seed": args.seed,
    }
    joblib.dump(bundle, model_out)

    summary = {
        "train_candidates_total": len(train_flat_all),
        "train_candidates_used": len(train_flat),
        "train_positive": int((train_labels == 1).sum()),
        "train_hard_negative": int((train_labels == 0).sum()),
        "train_ambiguous": int((train_labels_all < 0).sum()),
        "oof_positive_retention_target": args.min_oof_positive_retention,
        "oof_guard_threshold": oof_threshold,
        "variants": {
            variant: {
                "baseline": result["baseline_metrics"],
                "selected": result["best_row"],
                "macro_F1_delta": (
                    result["best_row"]["macro_F1"]
                    - result["baseline_metrics"]["macro_F1"]
                ),
                "count_MAE_delta": (
                    result["best_row"]["total_count_MAE"]
                    - result["baseline_metrics"]["total_count_MAE"]
                ),
            }
            for variant, result in best_by_variant.items()
        },
    }
    (output_dir / "guard_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 检测框 hard-negative 背景守门员（VAL）",
        "",
        "训练数据来自 150-epoch 正式检测器在 train 图上的真实预测框，"
        "不再使用随机背景模拟。守门员只允许删除预测，不重判 A-F 类别。",
        "",
        "## 训练数据",
        "",
        f"- 阈值后 train 候选：{len(train_flat_all)}",
        f"- 可用候选：{len(train_flat)}",
        f"- 缺陷邻域正样本：{int((train_labels == 1).sum())}",
        f"- 明确背景 hard negatives：{int((train_labels == 0).sum())}",
        f"- 空间歧义、未训练：{int((train_labels_all < 0).sum())}",
        f"- OOF 正样本保留目标：{args.min_oof_positive_retention:.3f}",
        f"- OOF 建议阈值：{oof_threshold:.6f}",
        "",
        "## 验证结果",
        "",
        "| 方案 | 守门阈值 | 删除框 | Macro-F1 | 变化 | TP保留 | 总count MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, result in best_by_variant.items():
        baseline_metrics = result["baseline_metrics"]
        selected = result["best_row"]
        report_lines.append(
            f"| {variant} | {selected['guard_threshold']:.4f} | "
            f"{int(selected['removed'])} | {selected['macro_F1']:.4f} | "
            f"{selected['macro_F1'] - baseline_metrics['macro_F1']:+.4f} | "
            f"{selected['retained_TP']:.4f} | "
            f"{selected['total_count_MAE']:.4f} |"
        )
    tiled_delta = summary["variants"]["full_plus_C_tiles"]["macro_F1_delta"]
    retain = summary["variants"]["full_plus_C_tiles"]["selected"]["retained_TP"]
    keep_guard = tiled_delta >= 0.003 and retain >= 0.98
    report_lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                "**守门员满足收益与召回约束，可进入最终候选。**"
                if keep_guard
                else
                "**守门员收益不足或损害召回，不进入最终方案。**"
            ),
            "",
            "保留规则：在已选 C 切片方案上 Macro-F1 至少提高 0.003，"
            "且基线 TP 保留率不低于 98%。",
            "",
            "> 该阈值只在 val 上选择；本脚本未读取 test。",
        ]
    )
    (output_dir / "hard_negative_guard_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(f"model saved: {model_out.relative_to(ROOT)}")
    for variant, result in best_by_variant.items():
        baseline = result["baseline_metrics"]
        selected = result["best_row"]
        print(
            f"{variant}: threshold={selected['guard_threshold']:.4f}, "
            f"macroF1={baseline['macro_F1']:.4f}->{selected['macro_F1']:.4f}, "
            f"removed={int(selected['removed'])}, "
            f"retainedTP={selected['retained_TP']:.4f}, "
            f"countMAE={baseline['total_count_MAE']:.4f}"
            f"->{selected['total_count_MAE']:.4f}"
        )


if __name__ == "__main__":
    main()
