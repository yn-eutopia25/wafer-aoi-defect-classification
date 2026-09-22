#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 C 类切片推理改进做图片级 bootstrap 稳健性检验。

默认只读取验证集既有预测，不运行模型。也可在最终方案已经冻结后显式对
test 做一次报告性 bootstrap；test 模式不允许用于重新选配置。
以 image_id 为抽样单位，保留同一图内目标的相关性。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
class BoxItem:
    label: str
    confidence: float
    box: Tuple[float, float, float, float]


def load_split_ids(path: Path, split: str) -> List[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sorted(
            row["image_id"]
            for row in csv.DictReader(handle)
            if row["split"] == split
        )


def load_ground_truth(
    path: Path,
    image_ids: set[str],
) -> Dict[str, List[BoxItem]]:
    result: Dict[str, List[BoxItem]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image_id"] not in image_ids:
                continue
            if row["label_code"] not in CLASSES:
                continue
            if str(row["is_pseudo_normal"]).lower() == "true":
                continue
            result[row["image_id"]].append(
                BoxItem(
                    label=row["label_code"],
                    confidence=1.0,
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
    thresholds: Dict[str, float] | None,
) -> Dict[str, List[BoxItem]]:
    result: Dict[str, List[BoxItem]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row["predicted_label"]
            confidence = float(row["confidence"])
            if thresholds is not None and confidence < thresholds[label]:
                continue
            result[row["image_id"]].append(
                BoxItem(
                    label=label,
                    confidence=confidence,
                    box=(
                        float(row["x_min"]),
                        float(row["y_min"]),
                        float(row["x_max"]),
                        float(row["y_max"]),
                    ),
                )
            )
    return result


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


def match_triplet(
    gt_items: List[BoxItem],
    pred_items: List[BoxItem],
    label: str,
    criterion: str = "iou",
    threshold: float = 0.5,
) -> np.ndarray:
    gt_label = [item for item in gt_items if item.label == label]
    pred_label = sorted(
        [item for item in pred_items if item.label == label],
        key=lambda item: item.confidence,
        reverse=True,
    )
    unmatched = set(range(len(gt_label)))
    true_positive = 0
    for prediction in pred_label:
        best_index = None
        best_score = -1.0
        for gt_index in unmatched:
            if criterion == "center":
                score = 1.0 if center_in(prediction.box, gt_label[gt_index].box) else 0.0
            else:
                score = iou(prediction.box, gt_label[gt_index].box)
            if score >= threshold and score > best_score:
                best_score = score
                best_index = gt_index
        if best_index is not None:
            unmatched.remove(best_index)
            true_positive += 1
    return np.asarray(
        [
            true_positive,
            len(pred_label) - true_positive,
            len(gt_label) - true_positive,
        ],
        dtype=np.int64,
    )


def build_image_statistics(
    image_ids: List[str],
    gt_by_image: Dict[str, List[BoxItem]],
    pred_by_image: Dict[str, List[BoxItem]],
) -> Dict[str, np.ndarray]:
    class_triplets = np.zeros((len(image_ids), len(CLASSES), 3), dtype=np.int64)
    c_iou30 = np.zeros((len(image_ids), 3), dtype=np.int64)
    c_center = np.zeros((len(image_ids), 3), dtype=np.int64)
    total_count_error = np.zeros(len(image_ids), dtype=np.float64)
    c_count_error = np.zeros(len(image_ids), dtype=np.float64)
    for image_index, image_id in enumerate(image_ids):
        gt_items = gt_by_image.get(image_id, [])
        pred_items = pred_by_image.get(image_id, [])
        for class_index, label in enumerate(CLASSES):
            class_triplets[image_index, class_index] = match_triplet(
                gt_items, pred_items, label
            )
        c_iou30[image_index] = match_triplet(
            gt_items, pred_items, "C", "iou", 0.3
        )
        c_center[image_index] = match_triplet(
            gt_items, pred_items, "C", "center", 1.0
        )
        total_count_error[image_index] = abs(len(pred_items) - len(gt_items))
        gt_c = sum(item.label == "C" for item in gt_items)
        pred_c = sum(item.label == "C" for item in pred_items)
        c_count_error[image_index] = abs(pred_c - gt_c)
    return {
        "class_triplets": class_triplets,
        "c_iou30": c_iou30,
        "c_center": c_center,
        "total_count_error": total_count_error,
        "c_count_error": c_count_error,
    }


def prf_from_triplets(triplets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = triplets[..., 0].astype(np.float64)
    fp = triplets[..., 1].astype(np.float64)
    fn = triplets[..., 2].astype(np.float64)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return precision, recall, f1


def bootstrap_metrics(
    statistics: Dict[str, np.ndarray],
    sample_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    triplets = statistics["class_triplets"][sample_indices].sum(axis=1)
    _, class_recall, class_f1 = prf_from_triplets(triplets)
    _, c_iou30_recall, _ = prf_from_triplets(
        statistics["c_iou30"][sample_indices].sum(axis=1)
    )
    _, c_center_recall, _ = prf_from_triplets(
        statistics["c_center"][sample_indices].sum(axis=1)
    )
    return {
        "macro_F1": class_f1.mean(axis=1),
        "C_F1": class_f1[:, CLASSES.index("C")],
        "C_Recall_IoU50": class_recall[:, CLASSES.index("C")],
        "C_Recall_IoU30": c_iou30_recall,
        "C_Center_Recall": c_center_recall,
        "Total_Count_MAE": statistics["total_count_error"][sample_indices].mean(axis=1),
        "C_Count_MAE": statistics["c_count_error"][sample_indices].mean(axis=1),
    }


def observed_metrics(statistics: Dict[str, np.ndarray]) -> Dict[str, float]:
    triplets = statistics["class_triplets"].sum(axis=0)
    _, class_recall, class_f1 = prf_from_triplets(triplets)
    _, c_iou30, _ = prf_from_triplets(statistics["c_iou30"].sum(axis=0))
    _, c_center, _ = prf_from_triplets(statistics["c_center"].sum(axis=0))
    return {
        "macro_F1": float(class_f1.mean()),
        "C_F1": float(class_f1[CLASSES.index("C")]),
        "C_Recall_IoU50": float(class_recall[CLASSES.index("C")]),
        "C_Recall_IoU30": float(c_iou30),
        "C_Center_Recall": float(c_center),
        "Total_Count_MAE": float(statistics["total_count_error"].mean()),
        "C_Count_MAE": float(statistics["c_count_error"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="验证切片推理改进的图片级 bootstrap 稳健性")
    parser.add_argument("--instances", default="data/annotations/instances.csv")
    parser.add_argument("--splits", default="data/derived/splits.csv")
    parser.add_argument(
        "--baseline-predictions",
        default="reports/detection_evaluation/epoch150_20260720/prediction_instances.csv",
    )
    parser.add_argument(
        "--tiled-predictions",
        default="reports/optimization_codex/tiled_inference/predictions_best_val.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/optimization_codex/tiled_inference/robustness",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--confirm-test-reporting",
        action="store_true",
        help="确认只对已冻结最终方案做 test 报告，不进行选型或调参",
    )
    args = parser.parse_args()

    if args.split == "test" and not args.confirm_test_reporting:
        raise SystemExit("test 报告需要显式传入 --confirm-test-reporting")

    paths = {
        "instances": (ROOT / args.instances).resolve(),
        "splits": (ROOT / args.splits).resolve(),
        "baseline": (ROOT / args.baseline_predictions).resolve(),
        "tiled": (ROOT / args.tiled_predictions).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_ids = load_split_ids(paths["splits"], args.split)
    if len(image_ids) != 30:
        raise RuntimeError(f"预期 30 张 val 图片，实际 {len(image_ids)}")
    image_id_set = set(image_ids)
    gt_by_image = load_ground_truth(paths["instances"], image_id_set)
    baseline = load_predictions(paths["baseline"], DEFAULT_THRESHOLDS)
    tiled = load_predictions(paths["tiled"], None)

    baseline_statistics = build_image_statistics(image_ids, gt_by_image, baseline)
    tiled_statistics = build_image_statistics(image_ids, gt_by_image, tiled)
    baseline_observed = observed_metrics(baseline_statistics)
    tiled_observed = observed_metrics(tiled_statistics)

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.integers(
        0,
        len(image_ids),
        size=(args.bootstrap_samples, len(image_ids)),
    )
    baseline_boot = bootstrap_metrics(baseline_statistics, sample_indices)
    tiled_boot = bootstrap_metrics(tiled_statistics, sample_indices)

    higher_is_better = {
        "macro_F1": True,
        "C_F1": True,
        "C_Recall_IoU50": True,
        "C_Recall_IoU30": True,
        "C_Center_Recall": True,
        "Total_Count_MAE": False,
        "C_Count_MAE": False,
    }
    rows = []
    distributions: Dict[str, np.ndarray] = {}
    for metric, is_higher_better in higher_is_better.items():
        raw_delta = tiled_boot[metric] - baseline_boot[metric]
        benefit_delta = raw_delta if is_higher_better else -raw_delta
        distributions[metric] = benefit_delta
        low, median, high = np.quantile(benefit_delta, [0.025, 0.5, 0.975])
        observed_raw_delta = tiled_observed[metric] - baseline_observed[metric]
        observed_benefit = observed_raw_delta if is_higher_better else -observed_raw_delta
        rows.append(
            {
                "metric": metric,
                "direction": "higher_better" if is_higher_better else "lower_better",
                "baseline_observed": baseline_observed[metric],
                "tiled_observed": tiled_observed[metric],
                "observed_benefit": observed_benefit,
                "bootstrap_median_benefit": float(median),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "probability_benefit_gt_0": float(np.mean(benefit_delta > 0)),
            }
        )

    csv_path = output_dir / "bootstrap_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_result = {
        "split": args.split,
        "image_count": len(image_ids),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "sampling_unit": "image_id",
        "baseline_observed": baseline_observed,
        "tiled_observed": tiled_observed,
        "metrics": rows,
        "limitation": "仅量化当前30张验证图的抽样不确定性，不代表跨晶圆或跨批次泛化。",
    }
    (output_dir / "bootstrap_robustness.json").write_text(
        json.dumps(json_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_metrics = ["macro_F1", "C_F1", "C_Recall_IoU50", "Total_Count_MAE"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, metric in zip(axes.flat, plot_metrics):
        values = distributions[metric]
        axis.hist(values, bins=45, color="#3274a1", alpha=0.85)
        axis.axvline(0.0, color="#b22222", linestyle="--", linewidth=1.5)
        axis.axvline(float(np.median(values)), color="#1b7f3a", linewidth=1.5)
        axis.set_title(f"{metric}: benefit distribution")
        axis.set_xlabel("benefit (>0 means tiled is better)")
        axis.set_ylabel("bootstrap samples")
    fig.tight_layout()
    fig.savefig(output_dir / "bootstrap_benefit_distributions.png", dpi=180)
    plt.close(fig)

    row_map = {row["metric"]: row for row in rows}
    robust_core = (
        row_map["C_F1"]["ci95_low"] > 0
        and row_map["C_Recall_IoU50"]["ci95_low"] > 0
        and row_map["C_Count_MAE"]["probability_benefit_gt_0"] >= 0.90
    )
    report_lines = [
        "# C 类切片推理稳健性检验",
        "",
        f"- 验证图片：{len(image_ids)} 张",
        f"- 图片级 bootstrap：{args.bootstrap_samples} 次",
        f"- 随机种子：{args.seed}",
        f"- 数据划分：{args.split}",
        "- 正收益定义：F1/Recall 上升，MAE 下降。",
        "",
        "| 指标 | 基线 | 切片 | 观测收益 | 95% CI | P(收益>0) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['metric']} | {row['baseline_observed']:.4f} | "
            f"{row['tiled_observed']:.4f} | {row['observed_benefit']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
            f"{row['probability_benefit_gt_0']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                (
                    "**通过预设稳健性标准：切片推理保留为最终候选。**"
                    if robust_core
                    else
                    "**未完全通过预设稳健性标准：切片推理只作为可选高召回模式。**"
                )
                if args.split == "val"
                else
                (
                    "**test 报告支持验证集结论。**"
                    if robust_core
                    else
                    "**test 报告未完全复现验证集的稳健性标准；不据此重新调参。**"
                )
            ),
            "",
            "判定标准：C F1 与 C IoU@0.5 Recall 的 95% bootstrap "
            "置信区间下界均大于 0，且 C count MAE 改善概率不低于 90%。",
            "",
            f"> 限制：本检验只衡量当前 30 张 {args.split} 图片的抽样不确定性，"
            "不能替代跨晶圆、跨批次外部验证；test 结果不用于重新选择配置。",
        ]
    )
    (output_dir / "bootstrap_robustness.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(f"PASS: 已生成 {csv_path.relative_to(ROOT)}")
    print(f"robust_core={robust_core}")
    for row in rows:
        print(
            f"{row['metric']}: benefit={row['observed_benefit']:+.4f}, "
            f"CI=[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}], "
            f"P={row['probability_benefit_gt_0']:.3f}"
        )


if __name__ == "__main__":
    main()
