#!/usr/bin/env python3
"""
12_train_patch_baselines.py — Patch 缺陷六分类 baseline 训练

训练 A-F 六分类的多个 baseline 模型，仅使用 use_for_defect_six_class=true 的 patch。
支持传统特征 (颜色/纹理/几何) + 预训练 ResNet18 embedding。

用法:
    python scripts/12_train_patch_baselines.py --smoke-test
    python scripts/12_train_patch_baselines.py --run-full
    python scripts/12_train_patch_baselines.py --run-full --use-resnet
"""

from __future__ import annotations

import argparse, csv, json, logging, os, shutil, sys, tempfile, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aoi_defect.patch_features import PatchFeatureExtractor
from aoi_defect.patch_models import (train_sklearn_models, ResNetEmbedder,
                                       train_on_embeddings, save_model)
from aoi_defect.metrics import compute_all_metrics, key_confusion_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_baselines")

CLASS_NAMES = ["A", "B", "C", "D", "E", "F"]


def load_patch_manifest(path: Path, smoke: int = 0) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("use_for_defect_six_class") == "True" and r["train_label"] in CLASS_NAMES]
    if smoke:
        # per-class at most smoke
        by_cls = defaultdict(list)
        for r in rows:
            by_cls[r["train_label"]].append(r)
        rows = []
        for lc in CLASS_NAMES:
            rows.extend(by_cls[lc][:smoke])
    return rows


def load_split_map(splits_csv: Path) -> Dict[str, str]:
    with open(splits_csv, "r", encoding="utf-8-sig", newline="") as f:
        return {r["image_id"]: r["split"] for r in csv.DictReader(f)}


def extract_handcrafted(rows: List[Dict], project_root: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    extractor = PatchFeatureExtractor(resize_to=128)
    X_list, y_list, paths = [], [], []
    for i, r in enumerate(rows):
        if i % 500 == 0: logger.info("  提取特征 %d/%d", i, len(rows))
        try:
            fp = project_root / r["raw_patch_path"]
            # Read as bytes to avoid path issues, then decode
            raw_bytes = fp.read_bytes()
            img_np = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_np is None or img_np.dtype != np.uint8:
                img_np = np.zeros((128, 128, 3), dtype=np.uint8)
            bbox_info = {
                "bbox_width": float(r["bbox_width"]), "bbox_height": float(r["bbox_height"]),
                "bbox_area": float(r["bbox_area_px"]),
                "patch_area": float(r["crop_width"]) * float(r["crop_height"]),
                "polygon_area": float(r["target_area_px"]),
            }
            feats = extractor.extract(img_np, **bbox_info)
            X_list.append(feats)
            y_list.append(CLASS_NAMES.index(r["train_label"]))
            paths.append(str(fp))
        except Exception as e:
            logger.warning("跳过 %s: %s", r.get("patch_id", "?"), e)
    return np.array(X_list, dtype=np.float32), np.array(y_list), paths


def save_error_gallery(rows: List[Dict], y_true: np.ndarray, y_pred: np.ndarray,
                       project_root: Path, output_dir: Path):
    """保存错误 patch 图集。"""
    for lc in CLASS_NAMES:
        (output_dir / "error_gallery" / lc).mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        true_lc = CLASS_NAMES[y_true[i]]; pred_lc = CLASS_NAMES[y_pred[i]]
        if true_lc == pred_lc: continue
        try:
            fp = project_root / r["raw_patch_path"]
            from PIL import Image
            img_np = np.array(Image.open(fp).convert("RGB"))
            out = output_dir / "error_gallery" / true_lc / f"{r['patch_id']}_{true_lc}_as_{pred_lc}.jpg"
            Image.fromarray(img_np).save(out)
        except Exception: pass


def generate_report(metrics: Dict, model_name: str, output_dir: Path, n_train: int, n_val: int):
    lines = [
        "# Validation Summary", f"模型: {model_name}", f"训练样本: {n_train}, 验证样本: {n_val}",
        "", "## 整体指标",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Balanced Accuracy: {metrics['balanced_accuracy']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Weighted F1: {metrics['weighted_f1']:.4f}",
        f"- Top-2 Accuracy: {metrics.get('top2_accuracy', 0):.4f}",
        "", "## 每类指标",
        "| 类别 | Precision | Recall | F1 | Support |",
        "|------|-----------|--------|-----|---------|",
    ]
    for cn in CLASS_NAMES:
        pc = metrics.get("per_class", {}).get(cn, {})
        lines.append(f"| {cn} | {pc.get('precision',0):.4f} | {pc.get('recall',0):.4f} | {pc.get('f1',0):.4f} | {pc.get('support',0)} |")
    (output_dir / "validation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告: %s", output_dir / "validation_summary.md")


def main():
    p = argparse.ArgumentParser(description="Patch 缺陷六分类 baseline 训练")
    p.add_argument("--smoke-test", action="store_true", help="Smoke test: 每类最多 10 个")
    p.add_argument("--run-full", action="store_true", help="完整训练")
    p.add_argument("--use-resnet", action="store_true", help="同时使用 ResNet18 embedding")
    p.add_argument("--manifest", type=str, default="data/derived/patch_manifest.csv")
    p.add_argument("--splits", type=str, default="data/derived/splits.csv")
    p.add_argument("--patches_root", type=str, default="data/derived/patches")
    p.add_argument("--output_dir", type=str, default="reports/patch_classification")
    args = p.parse_args()

    if not args.smoke_test and not args.run_full:
        logger.error("请指定 --smoke-test 或 --run-full")
        sys.exit(1)

    root = Path(__file__).resolve().parent.parent
    manifest_path = (root / args.manifest).resolve()
    splits_path = (root / args.splits).resolve()
    patches_root = (root / args.patches_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    smoke_n = 10 if args.smoke_test else 0
    all_rows = load_patch_manifest(manifest_path, smoke=smoke_n)
    splits_map = load_split_map(splits_path)

    train_rows = [r for r in all_rows if splits_map.get(r["image_id"]) == "train"]
    val_rows = [r for r in all_rows if splits_map.get(r["image_id"]) == "val"]

    logger.info("Train: %d patches, Val: %d patches", len(train_rows), len(val_rows))

    # --- 实验 1: 传统特征 ---
    logger.info("=== 实验 1: 传统特征 baseline ===")
    X_train, y_train, _ = extract_handcrafted(train_rows, root)
    X_val, y_val, val_paths = extract_handcrafted(val_rows, root)

    result1 = train_sklearn_models(X_train, y_train, X_val, y_val, CLASS_NAMES,
                                    feature_names=[])

    # 保存
    model_dir = root / "models" / "patch_classifier"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_model(result1["best_model"], result1["scaler"], {
        "experiment": "handcrafted",
        "class_names": CLASS_NAMES,
        "model_name": result1["model_name"],
        "val_metrics": {k: v for k, v in result1["val_metrics"].items() if isinstance(v, (int, float, str))},
        "val_per_class": result1["val_metrics"].get("per_class", {}),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }, model_dir)

    generate_report(result1["val_metrics"], result1["model_name"], output_dir,
                    len(train_rows), len(val_rows))

    if args.use_resnet:
        # --- 实验 2: ResNet18 embedding ---
        logger.info("=== 实验 2: ResNet18 embedding baseline ===")
        emb = ResNetEmbedder(device="cpu")

        # Extract embeddings
        train_raw_paths = [root / r["raw_patch_path"] for r in train_rows]
        val_raw_paths = [root / r["raw_patch_path"] for r in val_rows]
        X_emb_train = emb.extract_batch(train_raw_paths)
        X_emb_val = emb.extract_batch(val_raw_paths)

        result2 = train_on_embeddings(X_emb_train, y_train, X_emb_val, y_val, CLASS_NAMES)

        model_dir2 = root / "models" / "patch_classifier_resnet"
        model_dir2.mkdir(parents=True, exist_ok=True)
        save_model(result2["best_model"], result2["scaler"], {
            "experiment": "resnet18",
            "class_names": CLASS_NAMES,
            "model_name": result2["model_name"],
            "weight_name": emb.weight_name,
            "val_metrics": {k: v for k, v in result2["val_metrics"].items() if isinstance(v, (int, float, str))},
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }, model_dir2)
        (model_dir2 / "weight_info.json").write_text(
            json.dumps({"name": emb.weight_name, "hash": emb.weight_hash}, ensure_ascii=False, indent=2))

    # Error gallery
    y_pred_best = result1["best_model"].predict(result1["scaler"].transform(X_val))
    save_error_gallery(val_rows, y_val, y_pred_best, root, output_dir)

    # Model comparison CSV
    with open(output_dir / "model_comparison.csv", "w", encoding="utf-8-sig", newline="") as f:
        comp = result1["models_comparison"]
        if args.use_resnet:
            comp += result2["models_comparison"]
        if comp:
            w = csv.DictWriter(f, fieldnames=list(comp[0].keys()))
            w.writeheader(); w.writerows(comp)

    logger.info("=" * 50)
    logger.info("训练完成! 输出: %s", output_dir)
    if args.smoke_test:
        # 清理 smoke 临时模型
        shutil.rmtree(str(model_dir), ignore_errors=True)
        logger.info("Smoke test 模型已清理")


if __name__ == "__main__":
    main()
