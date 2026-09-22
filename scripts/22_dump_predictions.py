#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_dump_predictions.py — 用指定权重对某个 split 全量出框并导出 CSV

用途 (二期 GPU 实验):
  - 在 train 图上收集真实检测框 -> 级联二阶段 hard-negative 训练
  - 在 val 图上导出新模型预测 -> 离线专项评估 (C 类 center-hit 等)

用法:
  python scripts/22_dump_predictions.py --weights models/detector/epoch150_20260720/baseline_best.pt \
      --split train --imgsz 640 --out reports/optimization_phase2/gpu/train_preds_ep150_640.csv
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["A", "B", "C", "D", "E", "F"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    model = YOLO(str(ROOT / args.weights))
    img_dir = ROOT / "data/derived/detection/images" / args.split
    imgs = sorted(img_dir.glob("*.jpg"))
    print(f"split={args.split} images={len(imgs)} imgsz={args.imgsz} conf={args.conf}")
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(imgs):
        res = model.predict(source=str(p), imgsz=args.imgsz, conf=args.conf,
                            iou=0.5, max_det=300, agnostic_nms=False,
                            device=0, verbose=False)[0]
        b = res.boxes
        for k in range(len(b)):
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[k].tolist()]
            rows.append({"image_id": p.stem, "predicted_label": CLASSES[int(b.cls[k])],
                         "confidence": round(float(b.conf[k]), 6),
                         "x_min": round(x1, 2), "y_min": round(y1, 2),
                         "x_max": round(x2, 2), "y_max": round(y2, 2)})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(imgs)}  boxes so far: {len(rows)}")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"saved {len(rows)} boxes -> {out_path}")


if __name__ == "__main__":
    main()
