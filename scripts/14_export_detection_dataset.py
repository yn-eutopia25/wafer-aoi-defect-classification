#!/usr/bin/env python3
"""
14_export_detection_dataset.py — 目标检测数据集导出器 (YOLO 格式)

导出 YOLO 格式的目标检测数据集：
  - data/derived/detection/images/{train,val,test}/
  - data/derived/detection/labels/{train,val,test}/
  - data/derived/detection/data.yaml
  - data/derived/detection/dataset_manifest.csv
  - reports/detection_dataset/

用法:
    python scripts/14_export_detection_dataset.py
    python scripts/14_export_detection_dataset.py --link-mode hardlink
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("export_detection")

# =============================================================================
# 常量
# =============================================================================
CLASS_IDS = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "E": 4, "F": 5}
CLASS_IDS = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
CLASS_NAMES = {0: "residue_cleaning", 1: "edge_glue", 2: "particle",
               3: "pad_abnormal", 4: "surface_damage", 5: "pi_broken"}
DEFECT_CODES = ["A", "B", "C", "D", "E", "F"]
EXPECTED_COUNTS = {"A": 638, "B": 271, "C": 1852, "D": 388, "E": 234, "F": 424}

MANIFEST_FIELDS = [
    "image_id", "filename", "split", "source_image_path", "derived_image_path",
    "label_path", "image_width", "image_height", "box_count",
    "count_A", "count_B", "count_C", "count_D", "count_E", "count_F",
    "source_sha256", "derived_sha256",
]


# =============================================================================
# 工具
# =============================================================================
def _nk(s: str) -> Tuple:
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s))

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def _write_csv(path: Path, fields: List[str], rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=str(path.parent), prefix=".tmp_")
    with os.fdopen(tmp_fd, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow({k: str(v) for k, v in r.items()})
    os.replace(tmp_path, path)

def _load_instances(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def _load_splits(path: Path) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return {r["image_id"]: r["split"] for r in csv.DictReader(f)}

def _load_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}


# =============================================================================
# 预览图
# =============================================================================
PREVIEW_COLORS = {
    "A": (255, 107, 107), "B": (255, 169, 77), "C": (255, 212, 59),
    "D": (105, 219, 124), "E": (116, 192, 252), "F": (218, 119, 242),
}

def _draw_preview(img_path: Path, boxes: List[Tuple[str, int, int, int, int]], out_path: Path):
    """画 bbox 预览图。"""
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: return
    # 按 area 从大到小排序
    boxes_sorted = sorted(boxes, key=lambda b: (b[3]-b[1])*(b[4]-b[2]), reverse=True)
    for lc, x1, y1, x2, y2 in boxes_sorted:
        color = PREVIEW_COLORS.get(lc, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, lc, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img)
    if ok: buf.tofile(str(out_path))


# =============================================================================
# 主逻辑
# =============================================================================
def export_detection_dataset(
    instances_csv: Path, splits_csv: Path, images_manifest_csv: Path,
    images_dir: Path, output_root: Path, report_dir: Path,
    link_mode: str = "hardlink",
):
    instances = _load_instances(instances_csv)
    splits = _load_splits(splits_csv)
    manifest_map = _load_manifest(images_manifest_csv)

    # Group instances by image_id
    inst_by_img: Dict[str, List[Dict]] = defaultdict(list)
    for r in instances:
        inst_by_img[r["image_id"]].append(r)

    # 输出目录
    det_root = output_root / "detection"
    img_dirs = {s: det_root / "images" / s for s in ["train", "val", "test"]}
    lbl_dirs = {s: det_root / "labels" / s for s in ["train", "val", "test"]}
    for d in list(img_dirs.values()) + list(lbl_dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    # 统计
    all_image_ids = sorted(splits.keys(), key=_nk)
    manifest_rows: List[Dict] = []
    class_box_counts: Dict[str, Dict[str, int]] = {s: {c: 0 for c in DEFECT_CODES} for s in ["train", "val", "test"]}
    empty_label_images: List[Dict] = []
    overlap_stats: List[Dict] = []
    duplicate_bboxes: List[Dict] = []
    actual_counts: Counter = Counter()
    errors: List[str] = []
    rng = np.random.RandomState(42)

    for iid in all_image_ids:
        split = splits[iid]
        mr = manifest_map.get(iid, {})
        filename = mr.get("filename", "")
        iw, ih = int(mr.get("width", 0)), int(mr.get("height", 0))
        src_path = images_dir / filename

        if not src_path.exists():
            errors.append(f"{iid}: 图片不存在 {src_path}")
            continue

        # 收集该图的缺陷 bbox (A-F only)
        all_anns = inst_by_img.get(iid, [])
        defect_anns = [a for a in all_anns if a["label_code"] in DEFECT_CODES]

        # 检查重复 bbox (相同类别+相同坐标)
        seen_bboxes: Dict[Tuple, str] = {}
        for a in defect_anns:
            lc = a["label_code"]
            x1, y1, x2, y2 = map(int, [a["x_min"], a["y_min"], a["x_max"], a["y_max"]])
            key = (lc, x1, y1, x2, y2)
            if key in seen_bboxes:
                duplicate_bboxes.append({
                    "image_id": iid, "ann_id_1": seen_bboxes[key],
                    "ann_id_2": a["ann_id"], "label_code": lc,
                    "bbox": f"[{x1},{y1},{x2},{y2}]",
                })
            else:
                seen_bboxes[key] = a["ann_id"]

        # 生成 YOLO labels
        yolo_lines: List[str] = []
        boxes_for_preview: List[Tuple[str, int, int, int, int]] = []
        per_class_counts: Dict[str, int] = {c: 0 for c in DEFECT_CODES}

        for a in defect_anns:
            lc = a["label_code"]
            x1, y1, x2, y2 = map(int, [a["x_min"], a["y_min"], a["x_max"], a["y_max"]])
            # polygon 也用外接 bbox (instances.csv 已经是 bbox 格式)
            cls_id = CLASS_IDS[lc]
            xc = (x1 + x2) / 2.0 / iw if iw > 0 else 0
            yc = (y1 + y2) / 2.0 / ih if ih > 0 else 0
            wn = (x2 - x1) / iw if iw > 0 else 0
            hn = (y2 - y1) / ih if ih > 0 else 0
            # clamp 0-1
            xc = max(0.0, min(1.0, xc)); yc = max(0.0, min(1.0, yc))
            wn = max(0.0, min(1.0, wn)); hn = max(0.0, min(1.0, hn))
            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            per_class_counts[lc] += 1
            actual_counts[lc] += 1
            class_box_counts[split][lc] += 1
            boxes_for_preview.append((lc, x1, y1, x2, y2))

        # 写 label 文件 (空图片写空 txt)
        label_path = lbl_dirs[split] / (iid + ".txt")
        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        if not yolo_lines:
            empty_label_images.append({"image_id": iid, "split": split, "reason": "no_defect_boxes"})

        # 链接/复制图片
        dst_path = img_dirs[split] / filename
        if dst_path.exists():
            dst_path.unlink()
        if link_mode == "hardlink":
            try:
                os.link(str(src_path), str(dst_path))
            except OSError as e:
                errors.append(f"{iid}: hardlink 失败 ({e})，请尝试 --link-mode copy")
                continue
        elif link_mode == "symlink":
            os.symlink(str(src_path), str(dst_path))
        else:  # copy
            shutil.copy2(str(src_path), str(dst_path))

        src_sha = _sha256_file(src_path)
        dst_sha = _sha256_file(dst_path)

        manifest_rows.append({
            "image_id": iid, "filename": filename, "split": split,
            "source_image_path": str(src_path), "derived_image_path": str(dst_path),
            "label_path": str(label_path), "image_width": iw, "image_height": ih,
            "box_count": len(yolo_lines),
            **{f"count_{c}": per_class_counts[c] for c in DEFECT_CODES},
            "source_sha256": src_sha[:16], "derived_sha256": dst_sha[:16],
        })

    # 重叠统计 (同图内 A-F 之间)
    for iid in all_image_ids:
        defect_anns = [a for a in inst_by_img.get(iid, []) if a["label_code"] in DEFECT_CODES]
        for i in range(len(defect_anns)):
            for j in range(i+1, len(defect_anns)):
                a, b = defect_anns[i], defect_anns[j]
                ax1, ay1, ax2, ay2 = map(int, [a["x_min"], a["y_min"], a["x_max"], a["y_max"]])
                bx1, by1, bx2, by2 = map(int, [b["x_min"], b["y_min"], b["x_max"], b["y_max"]])
                x1, y1 = max(ax1, bx1), max(ay1, by1)
                x2, y2 = min(ax2, bx2), min(ay2, by2)
                if x2 > x1 and y2 > y1:
                    inter = (x2-x1) * (y2-y1)
                    area_a = (ax2-ax1) * (ay2-ay1)
                    area_b = (bx2-bx1) * (by2-by1)
                    iou = inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0
                    if iou > 0.3:
                        overlap_stats.append({
                            "image_id": iid, "label_1": a["label_code"], "label_2": b["label_code"],
                            "iou": round(iou, 4),
                        })

    # data.yaml
    yaml_content = (
        f"path: {det_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 6\n"
        f"names:\n"
    )
    for cid, cname in CLASS_NAMES.items():
        yaml_content += f"  {cid}: {cname}\n"
    (det_root / "data.yaml").write_text(yaml_content, encoding="utf-8")

    # class_map.json
    (det_root / "class_map.json").write_text(
        json.dumps({str(k): v for k, v in CLASS_NAMES.items()}, ensure_ascii=False, indent=2), encoding="utf-8")

    # dataset_manifest.csv
    _write_csv(det_root / "dataset_manifest.csv", MANIFEST_FIELDS, manifest_rows)

    # split_image_lists
    sl_dir = det_root / "split_image_lists"
    sl_dir.mkdir(parents=True, exist_ok=True)
    for s in ["train", "val", "test"]:
        ids = [r["image_id"] for r in manifest_rows if r["split"] == s]
        (sl_dir / f"{s}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    # 报告
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / "class_box_counts.csv",
                ["split"] + [f"count_{c}" for c in DEFECT_CODES],
                [{"split": s, **{f"count_{c}": class_box_counts[s][c] for c in DEFECT_CODES}}
                 for s in ["train", "val", "test"]])
    _write_csv(report_dir / "empty_label_report.csv",
                ["image_id", "split", "reason"], empty_label_images)
    _write_csv(report_dir / "overlap_statistics.csv",
                ["image_id", "label_1", "label_2", "iou"], overlap_stats)

    # Preview
    preview_root = report_dir / "preview"
    for s in ["train", "val", "test"]:
        (preview_root / s).mkdir(parents=True, exist_ok=True)
        s_rows = [r for r in manifest_rows if r["split"] == s]
        n_preview = min(10, len(s_rows))
        if n_preview > 0:
            indices = rng.choice(len(s_rows), size=n_preview, replace=False)
            for idx in indices:
                r = s_rows[idx]
                iid = r["image_id"]
                defect_anns = [a for a in inst_by_img.get(iid, []) if a["label_code"] in DEFECT_CODES]
                boxes = [(a["label_code"], int(a["x_min"]), int(a["y_min"]), int(a["x_max"]), int(a["y_max"]))
                         for a in defect_anns]
                _draw_preview(images_dir / r["filename"], boxes, preview_root / s / f"{iid}_preview.jpg")

    # 验证
    val_errors: List[str] = []
    # 实例数验证
    for c in DEFECT_CODES:
        actual = actual_counts.get(c, 0)
        expected = EXPECTED_COUNTS[c]
        if actual != expected:
            val_errors.append(f"类别 {c} 实例数 {actual} != 期望 {expected}")

    # SHA256 一致性
    sha_mismatch = 0
    for r in manifest_rows:
        if r["source_sha256"] != r["derived_sha256"]:
            sha_mismatch += 1
            val_errors.append(f"{r['image_id']}: SHA256 不一致")

    # 重复 bbox
    if duplicate_bboxes:
        _write_csv(report_dir / "duplicate_bbox_report.csv",
                    ["image_id", "ann_id_1", "ann_id_2", "label_code", "bbox"], duplicate_bboxes)
        val_errors.append(f"发现 {len(duplicate_bboxes)} 条完全重复 bbox，请人工确认")

    # 验证报告
    val_lines = [
        "=" * 60, "Detection Dataset Validation Report", "=" * 60, "",
        f"Total images: {len(manifest_rows)}",
        f"Train: {sum(1 for r in manifest_rows if r['split']=='train')}",
        f"Val: {sum(1 for r in manifest_rows if r['split']=='val')}",
        f"Test: {sum(1 for r in manifest_rows if r['split']=='test')}",
        f"Empty labels: {len(empty_label_images)}",
        f"Overlap pairs (IoU>0.3): {len(overlap_stats)}",
        f"Duplicate bboxes: {len(duplicate_bboxes)}",
        f"SHA256 mismatches: {sha_mismatch}", "",
        "Class box counts:",
    ]
    for s in ["train", "val", "test"]:
        val_lines.append(f"  {s}: {class_box_counts[s]}")
    val_lines.append(f"\nTotal instance counts: {dict(actual_counts)}")
    val_lines.append(f"Expected: {EXPECTED_COUNTS}")
    val_lines.append("")
    if val_errors:
        val_lines.append(f"ERRORS ({len(val_errors)}):")
        for e in val_errors: val_lines.append(f"  ERROR: {e}")
    else:
        val_lines.append("✓ All validations passed.")
    val_lines.append("=" * 60)
    (report_dir / "validation_report.txt").write_text("\n".join(val_lines), encoding="utf-8")

    # Summary
    summary_lines = [
        "# Detection Dataset Export Summary", "",
        f"- Images: {len(manifest_rows)} (train={sum(1 for r in manifest_rows if r['split']=='train')}, "
        f"val={sum(1 for r in manifest_rows if r['split']=='val')}, "
        f"test={sum(1 for r in manifest_rows if r['split']=='test')})",
        f"- Link mode: {link_mode}",
        f"- Empty labels: {len(empty_label_images)}",
        f"- Overlap pairs: {len(overlap_stats)}",
        f"- Duplicate bboxes: {len(duplicate_bboxes)}",
        f"- Instance counts: {dict(actual_counts)}",
        f"- Errors: {len(val_errors)}",
        "",
        "## Output",
        f"- Dataset: {det_root}",
        f"- Reports: {report_dir}",
    ]
    (report_dir / "export_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    logger.info("=" * 50)
    logger.info("导出完成: %d 张图片, %d 个 bbox", len(manifest_rows), sum(actual_counts.values()))
    logger.info("  Train: %d, Val: %d, Test: %d",
                 sum(1 for r in manifest_rows if r['split']=='train'),
                 sum(1 for r in manifest_rows if r['split']=='val'),
                 sum(1 for r in manifest_rows if r['split']=='test'))
    logger.info("  实例数: %s", dict(actual_counts))
    if val_errors:
        for e in val_errors:
            logger.error("  %s", e)
        return False
    logger.info("  ✓ 所有验证通过")
    return True


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="目标检测数据集导出器 (YOLO)")
    p.add_argument("--instances_csv", type=str, default="data/annotations/instances.csv")
    p.add_argument("--splits_csv", type=str, default="data/derived/splits.csv")
    p.add_argument("--manifest_csv", type=str, default="data/metadata/images_manifest.csv")
    p.add_argument("--images_dir", type=str, default="data/images")
    p.add_argument("--output_root", type=str, default="data/derived")
    p.add_argument("--report_dir", type=str, default="reports/detection_dataset")
    p.add_argument("--link-mode", type=str, default="hardlink", choices=["hardlink", "symlink", "copy"])
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    ok = export_detection_dataset(
        instances_csv=(root / args.instances_csv).resolve(),
        splits_csv=(root / args.splits_csv).resolve(),
        images_manifest_csv=(root / args.manifest_csv).resolve(),
        images_dir=(root / args.images_dir).resolve(),
        output_root=(root / args.output_root).resolve(),
        report_dir=(root / args.report_dir).resolve(),
        link_mode=args.link_mode,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
