#!/usr/bin/env python3
"""
15_train_detector.py — Ultralytics YOLO nano 目标检测训练封装

训练 A-F 六分类的 YOLO nano 检测模型，使用 YOLO 格式检测数据集。

用法:
    # 打印配置 (不训练)
    python scripts/15_train_detector.py

    # Smoke test (1 epoch, 少量数据)
    python scripts/15_train_detector.py --smoke-test

    # 完整训练
    python scripts/15_train_detector.py --run-full --model yolo11n.pt --device auto

    # 断点续训
    python scripts/15_train_detector.py --resume runs/detection/baseline/weights/last.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_detector")


# =============================================================================
# 工具
# =============================================================================
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def _try_git_hash() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"

def _resolve_device(device: str) -> str:
    """解析 auto → 实际设备。"""
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _get_weight_sha256(model_path: str) -> str:
    """获取模型权重的 SHA256 (如果文件存在)。"""
    p = Path(model_path)
    if p.exists():
        return _sha256_file(p)
    # 如果是预训练模型名 (如 yolo11n.pt)，尝试从缓存读取
    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    cached = cache_dir / p.name
    if cached.exists():
        return _sha256_file(cached)
    return "unknown (will download)"

def _safe_workers() -> int:
    """根据 OS 设置安全的 worker 数。"""
    if platform.system() == "Windows":
        return 0  # Windows 上 worker > 0 可能有问题
    return min(4, os.cpu_count() or 2)


# =============================================================================
# 训练配置
# =============================================================================
def build_train_config(args, device: str) -> Dict[str, Any]:
    """构建 YOLO train 参数。"""
    workers = _safe_workers()

    # 保守增强设置
    config = {
        "data": str(args.data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "seed": args.seed,
        "deterministic": True,
        "workers": workers,
        "patience": args.patience,
        "pretrained": True,
        "save": True,
        "plots": True,
        "val": True,
        "max_det": 300,
        "project": str(args.runs_dir),
        "name": args.run_name,
        "exist_ok": args.exist_ok,
        # 保守增强
        "hsv_h": 0.005,
        "hsv_s": 0.10,
        "hsv_v": 0.10,
        "degrees": 0.0,
        "translate": 0.05,
        "scale": 0.20,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mosaic": 0.25,
        "mixup": 0.0,
        "close_mosaic": 10,
    }

    # smoke test 覆盖
    if args.smoke_test:
        config["epochs"] = 1
        config["batch"] = 4
        config["patience"] = 0
        config["name"] = "smoke_test"
        config["exist_ok"] = True

    return config


# =============================================================================
# 增强说明
# =============================================================================
AUGMENTATION_RATIONALE = """
## 增强策略说明

| 参数 | 值 | 理由 |
|------|-----|------|
| hsv_h | 0.005 | A 类依赖真实亮度，不做强色相扰动 |
| hsv_s | 0.10 | 轻微饱和度变化 |
| hsv_v | 0.10 | 轻微亮度变化 |
| degrees | 0 | 不旋转 |
| translate | 0.05 | 极小平移 |
| scale | 0.20 | 适度缩放 (C 为小目标) |
| shear | 0 | 不剪切 |
| perspective | 0 | 不透视 |
| fliplr | 0.5 | 水平翻转 |
| flipud | 0 | 不垂直翻转 |
| mosaic | 0.25 | 低概率 mosaic，避免破坏空间关系 |
| mixup | 0 | 不 mixup |
| close_mosaic | 10 | 最后 10 epoch 关闭 mosaic |
"""


# =============================================================================
# run_manifest
# =============================================================================
def generate_run_manifest(model: str, model_hash: str, config: Dict,
                          device: str, data_yaml: Path, runs_dir: Path,
                          output_dir: Path, args_list: List[str]):
    """生成 run_manifest.json。"""
    import ultralytics
    import torch

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _try_git_hash(),
        "command": " ".join(args_list),
        "python_version": sys.version,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "device": device,
        "seed": config.get("seed", 42),
        "model": model,
        "model_hash": model_hash,
        "data_yaml": str(data_yaml),
        "data_yaml_hash": _sha256_file(data_yaml) if data_yaml.exists() else "missing",
        "config": config,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("run_manifest: %s", manifest_path)


# =============================================================================
# model_card
# =============================================================================
def generate_model_card(best_pt_path: Path, val_metrics: Dict, config: Dict, output_path: Path):
    """生成 model_card.md。"""
    lines = [
        "# YOLO Nano 检测模型卡片",
        "",
        "## 训练类别",
        "| ID | 代码 | 名称 |",
        "|----|------|------|",
        "| 0 | A | residue_cleaning |",
        "| 1 | B | edge_glue |",
        "| 2 | C | particle |",
        "| 3 | D | pad_abnormal |",
        "| 4 | E | surface_damage |",
        "| 5 | F | pi_broken |",
        "",
        "**不包含**: G (uncertain_other), N1-N3 (pseudo-normal)",
        "",
        "## 数据",
        f"- 数据集: {config.get('data', '')}",
        f"- Split: train=140, val=30, test=30 (test 未参与训练)",
        f"- 图片总数: 200",
        f"- 标注实例: 3807 (A-F)",
        "",
        "## 训练配置",
        f"- 模型: {config.get('model', 'yolo11n.pt')}",
        f"- Epochs: {config.get('epochs', 120)}",
        f"- imgsz: {config.get('imgsz', 640)}",
        f"- batch: {config.get('batch', 'auto')}",
        f"- device: {config.get('device', 'cpu')}",
        f"- seed: {config.get('seed', 42)}",
        f"- patience: {config.get('patience', 20)}",
        "",
        AUGMENTATION_RATIONALE,
        "",
        "## 验证集指标",
    ]
    for k, v in val_metrics.items():
        if isinstance(v, (int, float)):
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
        else:
            lines.append(f"- {k}: {v}")

    lines += [
        "",
        "## 已知限制",
        "1. 只有一个晶圆/批次的数据，不能证明跨批次泛化能力",
        "2. A 类 (胶残留) 边界具有主观性，不同标注者可能不一致",
        "3. C 类 (颗粒) 为小目标，检测难度较高",
        "4. E 和 F 视觉特征有重叠，容易混淆",
        "5. 训练集仅 140 张图片，可能存在过拟合风险",
        "6. 测试集 (30 张) 仅用于最终评估，不用于调参",
        "",
        "## 使用限制",
        "- 此模型仅用于 AOI 缺陷检测辅助，不替代人工复核",
        "- 不适用于训练数据分布之外的图片",
        "- 模型输出需要后处理和置信度阈值调整",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("model_card: %s", output_path)


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Ultralytics YOLO nano 目标检测训练封装")
    p.add_argument("--smoke-test", action="store_true", help="Smoke test: 1 epoch, 少量数据")
    p.add_argument("--run-full", action="store_true", help="完整训练")
    p.add_argument("--resume", type=str, default=None, help="从 last.pt 断点续训")
    p.add_argument("--model", type=str, default="yolo26n.pt",
                   help="模型名称或 .pt 路径 (默认 yolo26n.pt)")
    p.add_argument("--data-yaml", type=str, default="data/derived/detection/data.yaml")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|0|1")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-dir", type=str, default="runs/detection")
    p.add_argument("--run-name", type=str, default="", help="运行名称 (默认带时间戳)")
    p.add_argument("--exist-ok", action="store_true", default=False, help="允许覆盖同名 run")
    p.add_argument("--report-dir", type=str, default="reports/detection_training")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_yaml = (root / args.data_yaml).resolve()
    runs_dir = (root / args.runs_dir).resolve()
    report_dir = (root / args.report_dir).resolve()

    # 如果没指定运行模式，只打印配置
    if not args.smoke_test and not args.run_full and not args.resume:
        logger.info("未指定运行模式。配置预览:")
        device = _resolve_device(args.device)
        config = build_train_config(args, device)
        for k, v in config.items():
            print(f"  {k}: {v}")
        logger.info("使用 --smoke-test 或 --run-full 启动训练")
        return

    # 检查数据集
    if not data_yaml.exists():
        logger.error("data.yaml 不存在: %s", data_yaml)
        logger.error("请先运行 scripts/14_export_detection_dataset.py")
        sys.exit(1)

    # 设备
    device = _resolve_device(args.device)
    logger.info("设备: %s", device)

    # 运行名称
    run_name = args.run_name
    if not run_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"baseline_{ts}" if args.run_full else f"smoke_{ts}"

    # 更新 args
    args.run_name = run_name
    args.runs_dir = str(runs_dir.relative_to(root)) if runs_dir.is_relative_to(root) else str(runs_dir)

    # 配置
    config = build_train_config(args, device)
    logger.info("训练配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # 导入 ultralytics
    try:
        from ultralytics import YOLO
        import ultralytics
    except ImportError:
        logger.error("缺少 ultralytics。请运行: pip install ultralytics")
        sys.exit(1)

    logger.info("ultralytics 版本: %s", ultralytics.__version__)

    # 加载模型
    if args.resume:
        logger.info("断点续训: %s", args.resume)
        model = YOLO(args.resume)
    else:
        logger.info("加载模型: %s", args.model)
        try:
            model = YOLO(args.model)
        except Exception as e:
            logger.error("模型加载失败: %s", e)
            if "yolo26" in args.model.lower():
                logger.error("当前 ultralytics %s 可能不支持 %s", ultralytics.__version__, args.model)
                logger.error("请尝试: --model yolo11n.pt")
            sys.exit(1)

    # 获取权重哈希
    model_hash = _get_weight_sha256(args.model if not args.resume else args.resume)
    logger.info("模型权重 SHA256: %s", model_hash[:16] + "...")

    # 训练
    logger.info("=" * 50)
    logger.info("开始训练...")
    logger.info("  Run name: %s", run_name)
    logger.info("  Device: %s", device)
    logger.info("  Epochs: %s", config["epochs"])
    logger.info("  ⚠️ 不读取 test 数据")

    try:
        results = model.train(**config)
    except Exception as e:
        logger.error("训练失败: %s", e)
        sys.exit(1)

    # 获取训练结果路径
    run_path = runs_dir / run_name
    if not run_path.exists():
        # ultralytics 可能把 run 放在其他地方
        logger.warning("训练输出不在预期路径: %s", run_path)
        # 尝试在 runs/detection 下找
        possible = list(runs_dir.glob(f"*{run_name}*"))
        if possible:
            run_path = possible[0]

    # 生成 run_manifest
    generate_run_manifest(
        model=args.model if not args.resume else args.resume,
        model_hash=model_hash,
        config=config,
        device=device,
        data_yaml=data_yaml,
        runs_dir=runs_dir,
        output_dir=run_path,
        args_list=sys.argv,
    )

    # 如果是完整训练，复制 best.pt 到 models/detector/
    if args.run_full:
        best_pt = run_path / "weights" / "best.pt"
        if best_pt.exists():
            models_dir = root / "models" / "detector"
            models_dir.mkdir(parents=True, exist_ok=True)
            dst = models_dir / "baseline_best.pt"
            shutil.copy2(str(best_pt), str(dst))
            logger.info("最佳模型: %s", dst)

            # 生成 model_card
            val_metrics = {}
            results_csv = run_path / "results.csv"
            if results_csv.exists():
                import csv
                with open(results_csv, "r") as f:
                    rows = list(csv.DictReader(f))
                    if rows:
                        last = rows[-1]
                        for k in ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]:
                            if k in last:
                                val_metrics[k] = float(last[k])

            generate_model_card(dst, val_metrics, config, models_dir / "model_card.md")
        else:
            logger.warning("best.pt 未找到: %s", best_pt)

    # 报告
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "training_summary.md").write_text(
        f"# 检测训练报告\n\n"
        f"- Run: {run_name}\n"
        f"- Model: {args.model}\n"
        f"- Device: {device}\n"
        f"- Epochs: {config['epochs']}\n"
        f"- Output: {run_path}\n"
        f"- Ultralytics: {ultralytics.__version__}\n"
        f"- Smoke test: {args.smoke_test}\n",
        encoding="utf-8")

    logger.info("=" * 50)
    logger.info("训练完成!")
    logger.info("  Run: %s", run_path)
    logger.info("  报告: %s", report_dir)

    # 清理 smoke test
    if args.smoke_test:
        logger.info("Smoke test 完成 (不保存正式模型)")


if __name__ == "__main__":
    main()
