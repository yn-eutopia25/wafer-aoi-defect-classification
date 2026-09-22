#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成最终优化实验图表、全类别 GT/Prediction 对照图与 HTML 索引。

只读取已经锁定的 test 结果，不运行模型、不改变预测、不做参数选择。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
COLORS = {
    "A": (46, 139, 87),
    "B": (0, 165, 255),
    "C": (220, 120, 20),
    "D": (180, 60, 180),
    "E": (40, 40, 220),
    "F": (180, 180, 30),
}
THRESHOLDS = {
    "A": 0.25,
    "B": 0.15,
    "C": 0.10,
    "D": 0.30,
    "E": 0.20,
    "F": 0.50,
}


def load_tiled_module():
    script_path = ROOT / "scripts/23_run_tiled_inference_ablation.py"
    spec = importlib.util.spec_from_file_location("aoi_final_report_tiled", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片: {path}")
    return image


def title_strip(image: np.ndarray, title: str) -> np.ndarray:
    strip = np.full((34, image.shape[1], 3), 26, dtype=np.uint8)
    cv2.putText(
        strip,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([strip, image])


def draw_gt(image: np.ndarray, items) -> np.ndarray:
    panel = image.copy()
    for item in sorted(
        items,
        key=lambda value: (value.box[2] - value.box[0]) * (value.box[3] - value.box[1]),
        reverse=True,
    ):
        x1, y1, x2, y2 = [int(round(value)) for value in item.box]
        color = COLORS[item.label]
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            panel,
            item.label,
            (x1 + 2, max(12, y1 + 13)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return panel


def draw_predictions(image: np.ndarray, items) -> np.ndarray:
    panel = image.copy()
    for item in sorted(
        items,
        key=lambda value: (value.box[2] - value.box[0]) * (value.box[3] - value.box[1]),
        reverse=True,
    ):
        x1, y1, x2, y2 = [int(round(value)) for value in item.box]
        color = COLORS[item.label]
        thickness = 2 if item.confidence >= 0.50 else 1
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            panel,
            f"{item.label} {item.confidence:.2f}",
            (x1 + 2, max(12, y1 + 13)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )
    return panel


def image_metrics(tiled, gt_items, pred_items) -> Dict[str, float]:
    totals = np.zeros(3, dtype=np.int64)
    for label in tiled.CLASSES:
        totals += np.asarray(
            tiled.match_count(gt_items, pred_items, label, "iou", 0.5)
        )
    tp, fp, fn = [int(value) for value in totals]
    precision, recall, f1 = tiled.prf(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def unique_take(items: List[Tuple], count: int, used: set[str]) -> List[Tuple]:
    selected = []
    for item in items:
        image_id = item[1]
        if image_id in used:
            continue
        selected.append(item)
        used.add(image_id)
        if len(selected) >= count:
            break
    return selected


def make_charts(
    output_dir: Path,
    standard_metrics: dict,
    baseline_metrics: dict,
    final_metrics: dict,
) -> None:
    labels = ["A", "B", "C", "D", "E", "F"]
    baseline_f1 = [baseline_metrics[f"F1_{label}"] for label in labels]
    final_f1 = [final_metrics[f"F1_{label}"] for label in labels]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, baseline_f1, width, label="Full-image baseline", color="#6c8ebf")
    axis.bar(x + width / 2, final_f1, width, label="Final C-tiled", color="#82b366")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 0.8)
    axis.set_ylabel("F1 @ fixed deployment thresholds")
    axis.set_title("Per-class test F1")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_f1_test.png", dpi=190)
    plt.close(fig)

    key_names = ["Macro-F1", "C Precision", "C Recall", "C F1"]
    baseline_values = [
        baseline_metrics["macro_F1"],
        baseline_metrics["P_C"],
        baseline_metrics["R_C"],
        baseline_metrics["F1_C"],
    ]
    final_values = [
        final_metrics["macro_F1"],
        final_metrics["P_C"],
        final_metrics["R_C"],
        final_metrics["F1_C"],
    ]
    x = np.arange(len(key_names))
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, baseline_values, width, label="Baseline", color="#6c8ebf")
    axis.bar(x + width / 2, final_values, width, label="Final tiled", color="#82b366")
    axis.set_xticks(x, key_names)
    axis.set_ylim(0, 0.7)
    axis.set_ylabel("Score")
    axis.set_title("Frozen test working-point comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for index, (base, final) in enumerate(zip(baseline_values, final_values)):
        axis.text(index - width / 2, base + 0.012, f"{base:.3f}", ha="center", fontsize=8)
        axis.text(index + width / 2, final + 0.012, f"{final:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "final_working_point_comparison.png", dpi=190)
    plt.close(fig)

    ap50 = [
        standard_metrics["per_class"][label]["AP50"]
        for label in labels
    ]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    bars = axis.bar(labels, ap50, color=["#3c9d74", "#e5a84b", "#6c8ebf", "#9970ab", "#d95f5f", "#baa53b"])
    axis.axhline(standard_metrics["mAP50"], color="#333333", linestyle="--", label=f"mAP50={standard_metrics['mAP50']:.3f}")
    axis.set_ylim(0, 0.8)
    axis.set_ylabel("AP50")
    axis.set_title("YOLO26n standard test AP50")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for bar, value in zip(bars, ap50):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "standard_test_ap50.png", dpi=190)
    plt.close(fig)

    models = ["150ep/640", "1500ep/640", "150ep/1280"]
    map50 = [0.4020, 0.3970, 0.3947]
    map5095 = [0.2145, 0.2120, 0.2012]
    x = np.arange(len(models))
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(x - width / 2, map50, width, label="val mAP50", color="#6c8ebf")
    axis.bar(x + width / 2, map5095, width, label="val mAP50-95", color="#b4c7e7")
    axis.set_xticks(x, models)
    axis.set_ylim(0, 0.48)
    axis.set_ylabel("Score")
    axis.set_title("Training-scale ablation (validation)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_scale_ablation.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成最终优化实验可视化报告")
    parser.add_argument("--instances", default="data/annotations/instances.csv")
    parser.add_argument("--splits", default="data/derived/splits.csv")
    parser.add_argument("--image-dir", default="data/derived/detection/images/test")
    parser.add_argument(
        "--baseline-predictions",
        default="reports/optimization_codex/final_test/prediction_instances.csv",
    )
    parser.add_argument(
        "--final-predictions",
        default="reports/optimization_codex/final_test_tiled/predictions_final_tiled_test.csv",
    )
    parser.add_argument(
        "--standard-metrics",
        default="reports/optimization_codex/final_test/standard_metrics_test.json",
    )
    parser.add_argument(
        "--final-comparison",
        default="reports/optimization_codex/final_test_tiled/final_test_comparison.json",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/optimization_codex/final_visual_report",
    )
    args = parser.parse_args()

    paths = {
        key: (ROOT / value).resolve()
        for key, value in {
            "instances": args.instances,
            "splits": args.splits,
            "image_dir": args.image_dir,
            "baseline": args.baseline_predictions,
            "final": args.final_predictions,
            "standard": args.standard_metrics,
            "comparison": args.final_comparison,
        }.items()
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = (ROOT / args.output_dir).resolve()
    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    tiled = load_tiled_module()
    split_map = tiled.load_split_map(paths["splits"])
    gt_by_image = tiled.load_ground_truth(paths["instances"], split_map, "test")
    baseline_raw = tiled.load_full_predictions(paths["baseline"])
    baseline = tiled.filter_full_predictions(baseline_raw, THRESHOLDS)
    final = tiled.load_full_predictions(paths["final"])
    image_paths = sorted(paths["image_dir"].glob("*.jpg"))
    image_path_map = {path.stem: path for path in image_paths}
    image_ids = sorted(image_path_map)

    rows = []
    for image_id in image_ids:
        baseline_metrics = image_metrics(
            tiled,
            gt_by_image.get(image_id, []),
            baseline.get(image_id, []),
        )
        final_metrics = image_metrics(
            tiled,
            gt_by_image.get(image_id, []),
            final.get(image_id, []),
        )
        gt_c = sum(item.label == "C" for item in gt_by_image.get(image_id, []))
        rows.append(
            {
                "image_id": image_id,
                "gt_count": len(gt_by_image.get(image_id, [])),
                "gt_C_count": gt_c,
                "baseline_f1": baseline_metrics["f1"],
                "final_f1": final_metrics["f1"],
                "f1_delta": final_metrics["f1"] - baseline_metrics["f1"],
                "final_tp": final_metrics["tp"],
                "final_fp": final_metrics["fp"],
                "final_fn": final_metrics["fn"],
            }
        )

    ranked_best = sorted(
        [(row["final_f1"], row["image_id"], row) for row in rows if row["gt_count"] >= 5],
        reverse=True,
    )
    ranked_improvement = sorted(
        [(row["f1_delta"], row["image_id"], row) for row in rows],
        reverse=True,
    )
    ranked_difficult = sorted(
        [(row["final_f1"], row["image_id"], row) for row in rows if row["gt_count"] >= 5]
    )
    ranked_dense_c = sorted(
        [(row["gt_C_count"], row["image_id"], row) for row in rows],
        reverse=True,
    )
    used: set[str] = set()
    selections = []
    for category, ranked, count in [
        ("best", ranked_best, 4),
        ("improved", ranked_improvement, 4),
        ("difficult", ranked_difficult, 4),
        ("dense_C", ranked_dense_c, 4),
    ]:
        for rank_value, image_id, row in unique_take(ranked, count, used):
            selections.append((category, rank_value, image_id, row))

    selection_rows = []
    for category, rank_value, image_id, row in selections:
        image = read_image(image_path_map[image_id])
        original_panel = title_strip(image.copy(), f"Original | {image_id}")
        gt_panel = title_strip(
            draw_gt(image, gt_by_image.get(image_id, [])),
            f"Ground truth | n={row['gt_count']}",
        )
        final_panel = title_strip(
            draw_predictions(image, final.get(image_id, [])),
            (
                f"Final prediction | F1={row['final_f1']:.3f} "
                f"TP/FP/FN={row['final_tp']}/{row['final_fp']}/{row['final_fn']}"
            ),
        )
        combined = np.hstack([original_panel, gt_panel, final_panel])
        filename = f"{category}_{image_id}_delta_{row['f1_delta']:+.3f}.jpg"
        cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tofile(
            gallery_dir / filename
        )
        selection_rows.append(
            {
                "category": category,
                "image_id": image_id,
                "gallery_file": f"gallery/{filename}",
                **row,
            }
        )

    with (output_dir / "gallery_selection.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selection_rows)

    standard_metrics = json.loads(paths["standard"].read_text(encoding="utf-8"))
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    baseline_metrics = comparison["baseline_metrics"]
    final_metrics = comparison["final_tiled_metrics"]
    make_charts(output_dir, standard_metrics, baseline_metrics, final_metrics)

    category_names = {
        "best": "检测较好案例",
        "improved": "切片改善案例",
        "difficult": "困难案例",
        "dense_C": "颗粒密集案例",
    }
    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>AOI 最终检测报告</title>",
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1500px;margin:24px auto;color:#222}"
        "img{max-width:100%;border:1px solid #ddd;margin:8px 0 24px}.metric{display:inline-block;padding:10px 16px;"
        "margin:4px;background:#f2f5f8;border-radius:6px}h2{border-bottom:2px solid #567;padding-bottom:6px}</style>",
        "</head><body><h1>AOI 缺陷检测最终 test 展示</h1>",
        f"<div class='metric'>标准 mAP50: {standard_metrics['mAP50']:.3f}</div>",
        f"<div class='metric'>标准 mAP50-95: {standard_metrics['mAP50_95']:.3f}</div>",
        f"<div class='metric'>最终 Macro-F1: {final_metrics['macro_F1']:.3f}</div>",
        f"<div class='metric'>最终 C F1: {final_metrics['F1_C']:.3f}</div>",
        "<p>所有配置均在 train/val 上确定。本页只展示锁定 test 结果，未据此调参。</p>",
    ]
    for category in ["best", "improved", "dense_C", "difficult"]:
        html.append(f"<h2>{category_names[category]}</h2>")
        for row in selection_rows:
            if row["category"] != category:
                continue
            html.append(
                f"<h3>{row['image_id']} | F1={row['final_f1']:.3f} | "
                f"Δ={row['f1_delta']:+.3f}</h3>"
            )
            html.append(f"<img src='{row['gallery_file']}' alt='{row['image_id']}'>")
    html.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(html), encoding="utf-8")

    print(f"generated: {output_dir.relative_to(ROOT)}")
    print(f"gallery images: {len(selection_rows)}")


if __name__ == "__main__":
    main()
