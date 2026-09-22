#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对已经在 val 冻结的 C 类切片方案做一次最终 test 评估。

本脚本不搜索 tile size、overlap、阈值或 NMS；所有参数必须来自 val 的
best_config.json 与冻结的 class_thresholds_val.json。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tiled_module():
    script_path = ROOT / "scripts/23_run_tiled_inference_ablation.py"
    spec = importlib.util.spec_from_file_location("aoi_tiled_ablation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_comparison_csv(path: Path, baseline: dict, final: dict) -> None:
    rows = []
    keys = sorted(set(baseline) | set(final))
    for key in keys:
        if not isinstance(baseline.get(key), (int, float)):
            continue
        rows.append(
            {
                "metric": key,
                "baseline": baseline.get(key, ""),
                "final_tiled": final.get(key, ""),
                "delta": final.get(key, 0.0) - baseline.get(key, 0.0),
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结配置的 C 切片最终 test 评估")
    parser.add_argument("--confirm-test-evaluation", action="store_true")
    parser.add_argument(
        "--model",
        default="models/detector/epoch150_20260720/baseline_best.pt",
    )
    parser.add_argument(
        "--raw-test-predictions",
        default="reports/optimization_codex/final_test/prediction_instances.csv",
    )
    parser.add_argument(
        "--val-config",
        default="reports/optimization_codex/tiled_inference/best_config.json",
    )
    parser.add_argument(
        "--threshold-config",
        default="reports/optimization_codex/final_test/class_thresholds_val.json",
    )
    parser.add_argument("--instances", default="data/annotations/instances.csv")
    parser.add_argument("--splits", default="data/derived/splits.csv")
    parser.add_argument("--image-dir", default="data/derived/detection/images/test")
    parser.add_argument(
        "--output-dir",
        default="reports/optimization_codex/final_test_tiled",
    )
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if not args.confirm_test_evaluation:
        raise SystemExit("必须显式传入 --confirm-test-evaluation")

    model_path = (ROOT / args.model).resolve()
    raw_predictions_path = (ROOT / args.raw_test_predictions).resolve()
    val_config_path = (ROOT / args.val_config).resolve()
    threshold_config_path = (ROOT / args.threshold_config).resolve()
    instances_path = (ROOT / args.instances).resolve()
    splits_path = (ROOT / args.splits).resolve()
    image_dir = (ROOT / args.image_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "final_test_tiled_lock.json"
    if lock_path.exists():
        raise SystemExit(
            f"最终 test 已锁定：{lock_path}。不得覆盖；如需复核请使用新目录。"
        )
    for required in [
        model_path,
        raw_predictions_path,
        val_config_path,
        threshold_config_path,
        instances_path,
        splits_path,
        image_dir,
    ]:
        if not required.exists():
            raise FileNotFoundError(required)

    val_config = json.loads(val_config_path.read_text(encoding="utf-8"))
    threshold_config = json.loads(threshold_config_path.read_text(encoding="utf-8"))
    thresholds = {
        key: float(value)
        for key, value in threshold_config["thresholds"].items()
    }
    if val_config.get("split") != "val":
        raise RuntimeError("切片配置不是从 val 冻结的，拒绝 test 评估")

    tiled = load_tiled_module()
    split_map = tiled.load_split_map(splits_path)
    gt_by_image = tiled.load_ground_truth(instances_path, split_map, "test")
    image_paths = sorted(image_dir.glob("*.jpg"))
    image_ids = [path.stem for path in image_paths]
    if len(image_paths) != 30:
        raise RuntimeError(f"预期 30 张 test 图片，实际 {len(image_paths)}")

    full_raw = tiled.load_full_predictions(raw_predictions_path)
    full_filtered = tiled.filter_full_predictions(full_raw, thresholds)
    baseline_metrics = tiled.evaluate(image_ids, gt_by_image, full_filtered)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    tile_size = int(val_config["tile_size"])
    overlap = float(val_config["overlap"])
    imgsz = int(val_config["imgsz"])
    border_margin = int(val_config["border_margin"])
    tile_conf = float(val_config["tile_conf"])
    nms_iou = float(val_config["nms_iou"])
    tile_predictions, inference_seconds = tiled.run_tiled_c_predictions(
        model=model,
        image_paths=image_paths,
        tile_size=tile_size,
        overlap=overlap,
        imgsz=imgsz,
        min_conf=0.01,
        border_margin=border_margin,
        device=args.device,
    )
    tiled.write_prediction_csv(output_dir / "tile_raw_test.csv", tile_predictions)
    final_predictions = tiled.combine_predictions(
        image_ids,
        full_filtered,
        tile_predictions,
        tile_conf,
        nms_iou,
    )
    final_metrics = tiled.evaluate(image_ids, gt_by_image, final_predictions)
    tiled.write_prediction_csv(
        output_dir / "predictions_final_tiled_test.csv",
        final_predictions,
    )
    tiled.make_gallery(
        output_dir / "gallery",
        image_paths,
        gt_by_image,
        full_filtered,
        final_predictions,
    )

    result = {
        "split": "test",
        "selection_split": "val",
        "configuration": {
            "thresholds": thresholds,
            "tile_size": tile_size,
            "overlap": overlap,
            "imgsz": imgsz,
            "border_margin": border_margin,
            "tile_conf": tile_conf,
            "nms_iou": nms_iou,
        },
        "baseline_metrics": baseline_metrics,
        "final_tiled_metrics": final_metrics,
        "delta": {
            key: final_metrics[key] - baseline_metrics[key]
            for key in final_metrics
            if isinstance(final_metrics[key], (int, float))
        },
        "tile_inference_seconds_30_images": inference_seconds,
    }
    (output_dir / "final_test_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_comparison_csv(
        output_dir / "final_test_comparison.csv",
        baseline_metrics,
        final_metrics,
    )

    report_lines = [
        "# 冻结 C 类切片方案最终 TEST 评估",
        "",
        "本配置完全由 train/val 确定；test 只评估一次，不进行参数搜索。",
        "",
        "## 冻结配置",
        "",
        f"- 全图模型：`{args.model}`",
        f"- 逐类阈值：{thresholds}",
        f"- C tile：{tile_size}×{tile_size}",
        f"- overlap：{overlap:.2f}",
        f"- tile confidence：{tile_conf:.2f}",
        f"- C NMS IoU：{nms_iou:.2f}",
        "",
        "## 固定工作点结果",
        "",
        "| 指标 | 完整图基线 | 最终切片方案 | 变化 |",
        "|---|---:|---:|---:|",
        f"| Macro-F1 | {baseline_metrics['macro_F1']:.4f} | "
        f"{final_metrics['macro_F1']:.4f} | "
        f"{final_metrics['macro_F1'] - baseline_metrics['macro_F1']:+.4f} |",
        f"| C Precision | {baseline_metrics['P_C']:.4f} | "
        f"{final_metrics['P_C']:.4f} | "
        f"{final_metrics['P_C'] - baseline_metrics['P_C']:+.4f} |",
        f"| C Recall@IoU0.5 | {baseline_metrics['R_C']:.4f} | "
        f"{final_metrics['R_C']:.4f} | "
        f"{final_metrics['R_C'] - baseline_metrics['R_C']:+.4f} |",
        f"| C F1 | {baseline_metrics['F1_C']:.4f} | "
        f"{final_metrics['F1_C']:.4f} | "
        f"{final_metrics['F1_C'] - baseline_metrics['F1_C']:+.4f} |",
        f"| C IoU@0.3 Recall | {baseline_metrics['C_recall_iou30']:.4f} | "
        f"{final_metrics['C_recall_iou30']:.4f} | "
        f"{final_metrics['C_recall_iou30'] - baseline_metrics['C_recall_iou30']:+.4f} |",
        f"| C center-hit Recall | {baseline_metrics['C_recall_center']:.4f} | "
        f"{final_metrics['C_recall_center']:.4f} | "
        f"{final_metrics['C_recall_center'] - baseline_metrics['C_recall_center']:+.4f} |",
        f"| 总 count MAE | {baseline_metrics['total_count_MAE']:.4f} | "
        f"{final_metrics['total_count_MAE']:.4f} | "
        f"{final_metrics['total_count_MAE'] - baseline_metrics['total_count_MAE']:+.4f} |",
        f"| C count MAE | {baseline_metrics['C_count_MAE']:.4f} | "
        f"{final_metrics['C_count_MAE']:.4f} | "
        f"{final_metrics['C_count_MAE'] - baseline_metrics['C_count_MAE']:+.4f} |",
        "",
        "## 每类 F1",
        "",
        "| 类别 | 基线 | 最终 |",
        "|---|---:|---:|",
    ]
    for label in tiled.CLASSES:
        report_lines.append(
            f"| {label} | {baseline_metrics[f'F1_{label}']:.4f} | "
            f"{final_metrics[f'F1_{label}']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            f"- 30 张图切片额外推理时间：{inference_seconds:.3f} s",
            "",
            "> 注意：这里是已冻结逐类阈值下的部署工作点指标，"
            "不能替代标准 mAP；标准 test mAP 见相邻 `final_test/` 目录。",
        ]
    )
    (output_dir / "FINAL_TEST_TILED_REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    lock = {
        "evaluation_time": datetime.now(timezone.utc).isoformat(),
        "model_sha256": sha256(model_path),
        "raw_test_predictions_sha256": sha256(raw_predictions_path),
        "val_config_sha256": sha256(val_config_path),
        "threshold_config_sha256": sha256(threshold_config_path),
        "splits_sha256": sha256(splits_path),
        "result_sha256": sha256(output_dir / "final_test_comparison.json"),
        "note": "Frozen test evaluation. Do not overwrite or tune from this result.",
    }
    lock_path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"baseline macroF1={baseline_metrics['macro_F1']:.4f}, "
        f"final={final_metrics['macro_F1']:.4f}, "
        f"delta={final_metrics['macro_F1'] - baseline_metrics['macro_F1']:+.4f}"
    )
    print(
        f"C F1={baseline_metrics['F1_C']:.4f}->{final_metrics['F1_C']:.4f}; "
        f"C recall={baseline_metrics['R_C']:.4f}->{final_metrics['R_C']:.4f}"
    )
    print(f"lock written: {lock_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
