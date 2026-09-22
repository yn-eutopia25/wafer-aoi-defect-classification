#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21_train_detector_hires.py — 高分辨率检测重训 (二期 C 类小目标专项)

与 15 号脚本相同的保守增强与随机性配置, 仅提升输入分辨率。
动机: C 类 bbox 中位 18x20 px, 640 输入下仅 ~28px; 1280 输入下 ~56px,
IoU 对像素偏移的敏感度减半 —— 直接针对 "找得到、框不准" 的定位瓶颈。

用法:
  python scripts/21_train_detector_hires.py --imgsz 1280 --batch 4 --epochs 150
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch, ultralytics

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"hires{args.imgsz}_{stamp}"
    cfg = dict(
        data=str(ROOT / "data/derived/detection/data.yaml"),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device="0", seed=42, deterministic=True, workers=0,
        patience=0, pretrained=True, save=True, plots=True, val=True,
        max_det=300, project=str(ROOT / "runs/detection_hires"),
        name=run_name, exist_ok=False,
        hsv_h=0.005, hsv_s=0.1, hsv_v=0.1, degrees=0.0, translate=0.05,
        scale=0.2, shear=0.0, perspective=0.0, fliplr=0.5, flipud=0.0,
        mosaic=0.25, mixup=0.0, close_mosaic=10,
    )
    print(f"train {run_name}: imgsz={args.imgsz} batch={args.batch} epochs={args.epochs}")
    model = YOLO(args.model)
    model.train(**cfg)

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"

    # archive: models/detector/imgszXXXX_ep150_<date>/
    arch = ROOT / "models" / "detector" / f"imgsz{args.imgsz}_ep{args.epochs}_{stamp[:8]}"
    arch.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(best, arch / "baseline_best.pt")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "experiment": "phase2_hires_training",
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seed": 42, "model": args.model,
        "best_weights_hash": sha256(best),
        "run_dir": str(run_dir), "config": {k: str(v) for k, v in cfg.items()},
    }
    (arch / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"archived -> {arch}")
    print("TRAINING_DONE")


if __name__ == "__main__":
    main()
