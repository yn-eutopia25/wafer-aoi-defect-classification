#!/usr/bin/env python3
"""
13_evaluate_patch_classifier.py — 评估已训练的 patch 分类器

用法:
    # val 集评估 (默认)
    python scripts/13_evaluate_patch_classifier.py

    # test 集评估 (需要显式确认)
    python scripts/13_evaluate_patch_classifier.py --split test --confirm-test-evaluation
"""

from __future__ import annotations

import argparse, csv, json, logging, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aoi_defect.metrics import compute_all_metrics, key_confusion_pairs
from aoi_defect.patch_features import PatchFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("evaluate")

CLASS_NAMES = ["A", "B", "C", "D", "E", "F"]


def load_manifest(path: Path, split: str, splits_csv: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("use_for_defect_six_class") == "True" and r["train_label"] in CLASS_NAMES]

    with open(splits_csv, "r", encoding="utf-8-sig", newline="") as f:
        split_map = {r["image_id"]: r["split"] for r in csv.DictReader(f)}
    return [r for r in rows if split_map.get(r["image_id"]) == split]


def load_model(model_dir: Path):
    model = joblib.load(model_dir / "best_model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    return model, scaler


def extract_features(rows: List[Dict], project_root: Path):
    extractor = PatchFeatureExtractor(resize_to=128)
    X, y = [], []
    for r in rows:
        try:
            fp = project_root / r["raw_patch_path"]
            raw_bytes = fp.read_bytes()
            img = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.dtype != np.uint8:
                img = np.zeros((128, 128, 3), dtype=np.uint8)
            feats = extractor.extract(img,
                bbox_width=float(r["bbox_width"]), bbox_height=float(r["bbox_height"]),
                bbox_area=float(r["bbox_area_px"]),
                patch_area=float(r["crop_width"]) * float(r["crop_height"]),
                polygon_area=float(r["target_area_px"]))
            X.append(feats); y.append(CLASS_NAMES.index(r["train_label"]))
        except Exception as e:
            logger.warning("跳过 %s: %s", r.get("patch_id","?"), e)
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    p = argparse.ArgumentParser(description="评估 patch 分类器")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--confirm-test-evaluation", action="store_true", help="确认 test 集评估")
    p.add_argument("--manifest", type=str, default="data/derived/patch_manifest.csv")
    p.add_argument("--splits", type=str, default="data/derived/splits.csv")
    p.add_argument("--model_dir", type=str, default="models/patch_classifier")
    p.add_argument("--patches_root", type=str, default="data/derived/patches")
    p.add_argument("--output_dir", type=str, default="reports/patch_classification")
    args = p.parse_args()

    if args.split == "test" and not args.confirm_test_evaluation:
        logger.error("test 集评估需要 --confirm-test-evaluation 确认")
        sys.exit(1)

    root = Path(__file__).resolve().parent.parent
    manifest_path = (root / args.manifest).resolve()
    splits_path = (root / args.splits).resolve()
    model_dir = (root / args.model_dir).resolve()
    patches_root = (root / args.patches_root).resolve()
    output_dir = (root / args.output_dir).resolve()

    if not (model_dir / "best_model.joblib").exists():
        logger.error("模型不存在: %s", model_dir / "best_model.joblib")
        logger.error("请先运行 scripts/12_train_patch_baselines.py --run-full")
        sys.exit(1)

    logger.info("加载模型: %s", model_dir)
    model, scaler = load_model(model_dir)

    logger.info("加载 %s 集...", args.split)
    rows = load_manifest(manifest_path, args.split, splits_path)
    logger.info("%s 集: %d patches", args.split, len(rows))

    X, y_true = extract_features(rows, root)
    X_s = scaler.transform(X)

    y_pred = model.predict(X_s)
    y_proba = model.predict_proba(X_s) if hasattr(model, "predict_proba") else None

    metrics = compute_all_metrics(y_true, y_pred, y_proba, CLASS_NAMES)

    # Report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        f"# {args.split.upper()} 集评估报告",
        f"样本数: {len(rows)}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}",
        f"Macro F1: {metrics['macro_f1']:.4f}",
        f"Weighted F1: {metrics['weighted_f1']:.4f}",
        f"Top-2 Accuracy: {metrics.get('top2_accuracy', 0):.4f}",
        "",
        "## 每类指标",
        "| 类别 | Precision | Recall | F1 | Support |",
        "|------|-----------|--------|-----|---------|",
    ]
    for cn in CLASS_NAMES:
        pc = metrics.get("per_class", {}).get(cn, {})
        report_lines.append(f"| {cn} | {pc.get('precision',0):.4f} | {pc.get('recall',0):.4f} | {pc.get('f1',0):.4f} | {pc.get('support',0)} |")
    report_lines.append("")
    report_lines.append("## 关键混淆对")
    pairs = key_confusion_pairs(y_true, y_pred, CLASS_NAMES)
    for pair, cnt in list(pairs.items())[:10]:
        report_lines.append(f"- {pair}: {cnt}")

    (output_dir / f"{args.split}_evaluation_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    # Per-class metrics CSV
    with open(output_dir / "per_class_metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
        rows_csv = []
        for cn in CLASS_NAMES:
            pc = metrics.get("per_class", {}).get(cn, {})
            rows_csv.append({"class": cn, **pc})
        w = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        w.writeheader(); w.writerows(rows_csv)

    # Predictions CSV
    with open(output_dir / f"{args.split}_predictions.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patch_id", "true_label", "pred_label"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"patch_id": r["patch_id"], "true_label": CLASS_NAMES[y_true[i]],
                         "pred_label": CLASS_NAMES[y_pred[i]]})

    logger.info("=" * 50)
    logger.info("评估完成: Accuracy=%.4f Balanced=%.4f MacroF1=%.4f",
                 metrics["accuracy"], metrics["balanced_accuracy"], metrics["macro_f1"])
    logger.info("报告: %s/%s_evaluation_report.md", output_dir, args.split)


if __name__ == "__main__":
    main()
