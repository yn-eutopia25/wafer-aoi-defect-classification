#!/usr/bin/env python3
"""
09_audit_training_data.py — 训练前数据全面审计

从 instances.csv + image_summary.csv + images_manifest.csv 读取数据，
进行一致性检查、类别统计、重复检查、bbox 重叠分析、pseudo-normal 检查和
可疑实例检测。输出审计报告、统计 CSV、contact sheets 和可视化图表。

输出:
  data/derived/dataset_fingerprint.json
  reports/data_audit/
    ├── audit_summary.md
    ├── class_instance_statistics.csv
    ├── class_image_statistics.csv
    ├── bbox_statistics.csv
    ├── geometry_statistics.csv
    ├── annotation_duplicate_report.csv
    ├── image_duplicate_report.csv
    ├── overlap_pair_statistics.csv
    ├── containment_statistics.csv
    ├── pseudo_normal_overlap.csv
    ├── suspicious_instances.csv
    ├── contact_sheets/...
    └── figures/...

用法:
    python scripts/09_audit_training_data.py
    python scripts/09_audit_training_data.py --no-figures --no-contact-sheets
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("audit")


# =============================================================================
# 工具函数
# =============================================================================
def _natural_sort_key(s: str) -> Tuple:
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=str(path.parent), prefix=".tmp_")
    with os.fdopen(tmp_fd, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: str(v) for k, v in r.items()})
    os.replace(tmp_path, path)


def _try_git_hash() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# =============================================================================
# 加载
# =============================================================================
def load_instances(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_summary(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}


def load_schema(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# 主审计
# =============================================================================
class TrainingDataAuditor:
    def __init__(self, instances_csv: Path, summary_csv: Path, manifest_csv: Path,
                 schema_yaml: Path, images_dir: Path, output_root: Path,
                 skip_figures: bool = False, skip_contact_sheets: bool = False):
        self.instances_csv = instances_csv
        self.summary_csv = summary_csv
        self.manifest_csv = manifest_csv
        self.schema_yaml = schema_yaml
        self.images_dir = images_dir
        self.output_root = output_root
        self.derived_dir = output_root.parent.parent / "data" / "derived"
        self.skip_figures = skip_figures
        self.skip_contact_sheets = skip_contact_sheets

        # 加载
        self.instances = load_instances(instances_csv)
        self.summary = load_summary(summary_csv)
        self.manifest = load_manifest(manifest_csv)
        self.schema = load_schema(schema_yaml)

        self.DEFECT_CODES = ["A", "B", "C", "D", "E", "F", "G"]
        self.NORMAL_CODES = ["N1", "N2", "N3"]
        self.ALL_CODES = self.DEFECT_CODES + self.NORMAL_CODES
        self.N_DEFECT = {c: 0 for c in self.DEFECT_CODES}

        # 预计算
        self.inst_by_image: Dict[str, List[Dict]] = defaultdict(list)
        for r in self.instances:
            self.inst_by_image[r["image_id"]].append(r)

        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.dup_anns: List[Dict] = []
        self.dup_images: List[Dict] = []
        self.suspicious: List[Dict] = []

    # ------------------------------------------------------------------
    # 1. 基础一致性检查
    # ------------------------------------------------------------------
    def _check_basics(self) -> bool:
        ok = True
        if len(self.manifest) != 200:
            self.errors.append(f"Manifest 行数 {len(self.manifest)} != 200")
            ok = False
        if len(self.instances) != 4005:
            self.errors.append(f"instances 行数 {len(self.instances)} != 4005")
            ok = False
        if len(self.summary) != 200:
            self.errors.append(f"image_summary 行数 {len(self.summary)} != 200")
            ok = False

        manifest_set = set(self.manifest.keys())
        for r in self.instances:
            iid = r["image_id"]
            if iid not in manifest_set:
                self.errors.append(f"instances 中 image_id={iid} 不在 manifest")
                ok = False

        # 图片文件存在 & 尺寸一致
        sizes_ok = 0
        for iid, mr in self.manifest.items():
            fp = self.images_dir / mr["filename"]
            if not fp.exists():
                self.errors.append(f"图片文件不存在: {mr['filename']}")
                ok = False
                continue
            try:
                from PIL import Image
                img = Image.open(fp)
                mw, mh = int(mr["width"]), int(mr["height"])
                if img.size != (mw, mh):
                    self.errors.append(f"{iid}: manifest 尺寸 {mw}x{mh} != 实际 {img.size[0]}x{img.size[1]}")
                    ok = False
                else:
                    sizes_ok += 1
            except Exception as e:
                self.errors.append(f"{iid}: 读取图片失败 {e}")
                ok = False
        logger.info("图片尺寸验证: %d/%d 通过", sizes_ok, len(self.manifest))

        # bbox 合法性 + 宽高面积计算
        bbox_err = 0
        for r in self.instances:
            iid = r["image_id"]; mr = self.manifest.get(iid, {})
            iw, ih = int(mr.get("width", 0)), int(mr.get("height", 0))
            x1, y1, x2, y2 = map(int, [r["x_min"], r["y_min"], r["x_max"], r["y_max"]])
            bw, bh = int(r["bbox_width"]), int(r["bbox_height"])
            barea = int(r["bbox_area_px"])
            if x1 < 0 or y1 < 0 or x2 > iw or y2 > ih:
                self.errors.append(f"{r['ann_id']}: bbox 越界 ({x1},{y1},{x2},{y2}) img={iw}x{ih}")
                bbox_err += 1
            if bw != x2 - x1 or bh != y2 - y1:
                self.errors.append(f"{r['ann_id']}: bbox_width/height 计算错误")
                bbox_err += 1
            if barea != bw * bh:
                self.errors.append(f"{r['ann_id']}: bbox_area_px 计算错误")
                bbox_err += 1
            if r["geometry_type"] == "polygon" and float(r["area_px"]) <= 0:
                self.errors.append(f"{r['ann_id']}: polygon area_px <= 0")
                bbox_err += 1
            # is_pseudo_normal 验证
            lc = r["label_code"]
            ipn = r["is_pseudo_normal"] == "True"
            if lc in self.DEFECT_CODES and ipn:
                self.errors.append(f"{r['ann_id']}: defect({lc}) is_pseudo_normal=True")
                bbox_err += 1
            if lc in self.NORMAL_CODES and not ipn:
                self.errors.append(f"{r['ann_id']}: normal({lc}) is_pseudo_normal=False")
                bbox_err += 1
        if bbox_err:
            logger.error("bbox 错误: %d", bbox_err)
            ok = False

        # 类别总数
        cat_counts = Counter(r["label_code"] for r in self.instances)
        expected = {"A": 638, "B": 271, "C": 1852, "D": 388, "E": 234, "F": 424, "G": 0,
                     "N1": 26, "N2": 135, "N3": 37}
        for c, ev in expected.items():
            actual = cat_counts.get(c, 0)
            if actual != ev:
                self.errors.append(f"类别 {c} 数量 {actual} != 期望 {ev}")
                ok = False
            self.N_DEFECT[c] = actual
        logger.info("基础一致性检查: %s", "PASS" if ok else "FAIL")
        return ok

    # ------------------------------------------------------------------
    # 2. 类别统计
    # ------------------------------------------------------------------
    def _class_stats(self):
        rows_class: List[Dict] = []; rows_bbox: List[Dict] = []; rows_geom: List[Dict] = []
        rows_image_class: List[Dict] = []

        for lc in self.ALL_CODES:
            insts = [r for r in self.instances if r["label_code"] == lc]
            n = len(insts)
            if n == 0:
                rows_class.append({
                    "label_code": lc, "instance_count": 0, "image_count": 0,
                    "avg_per_image": 0, "rect_count": 0, "polygon_count": 0,
                    "bw_min": "", "bw_p1": "", "bw_p5": "", "bw_p25": "", "bw_p50": "",
                    "bw_p75": "", "bw_p95": "", "bw_p99": "", "bw_max": "",
                    "bh_min": "", "bh_median": "", "bh_max": "",
                    "area_min": "", "area_median": "", "area_max": "",
                    "area_ratio_p50": "",
                })
                rows_image_class.append({"label_code": lc, "image_count": 0, "avg_per_image": 0})
                rows_geom.append({"label_code": lc, "instance_count": 0, "rect_count": 0, "polygon_count": 0})
                continue

            images = set(r["image_id"] for r in insts)
            n_img = len(images)
            rect_n = sum(1 for r in insts if r["geometry_type"] == "rectangle")
            poly_n = sum(1 for r in insts if r["geometry_type"] == "polygon")

            bws = sorted([int(r["bbox_width"]) for r in insts])
            bhs = sorted([int(r["bbox_height"]) for r in insts])
            areas = sorted([int(r["bbox_area_px"]) for r in insts])

            def _pcts(vals, ps):
                nv = len(vals)
                return {f"p{p}": vals[max(0, min(nv - 1, int(nv * p / 100)))] for p in ps}

            bp = _pcts(bws, [1, 5, 25, 50, 75, 95, 99])
            hp = _pcts(bhs, [1, 5, 25, 50, 75, 95, 99])

            rows_class.append({
                "label_code": lc, "instance_count": n, "image_count": n_img,
                "avg_per_image": round(n / n_img, 2),
                "rect_count": rect_n, "polygon_count": poly_n,
                "bw_min": bws[0], **{f"bw_{k}": v for k, v in bp.items()},
                "bw_max": bws[-1],
                "bh_min": bhs[0], **{f"bh_{k}": v for k, v in hp.items() if k.startswith("p50")},
                "bh_max": bhs[-1],
                "area_min": areas[0], "area_p50": _pcts(areas, [50])["p50"],
                "area_max": areas[-1],
            })
            rows_image_class.append({"label_code": lc, "image_count": n_img, "avg_per_image": round(n / n_img, 2)})
            rows_geom.append({"label_code": lc, "instance_count": n, "rect_count": rect_n, "polygon_count": poly_n})

            # bbox statistics (per-instance)
            for r in insts:
                iw, ih = int(r["image_width"]), int(r["image_height"])
                rows_bbox.append({
                    "ann_id": r["ann_id"], "label_code": lc,
                    "bbox_width": r["bbox_width"], "bbox_height": r["bbox_height"],
                    "bbox_area_px": r["bbox_area_px"],
                    "area_ratio": round(float(r["bbox_area_px"]) / (iw * ih), 6) if iw * ih > 0 else 0,
                    "geometry_type": r["geometry_type"],
                })

        _write_csv(self.output_root / "class_instance_statistics.csv",
                    ["label_code", "instance_count", "image_count", "avg_per_image",
                     "rect_count", "polygon_count",
                     "bw_min", "bw_p1", "bw_p5", "bw_p25", "bw_p50", "bw_p75", "bw_p95", "bw_p99", "bw_max",
                     "bh_min", "bh_p50", "bh_max",
                     "area_min", "area_p50", "area_max"],
                    rows_class)
        _write_csv(self.output_root / "class_image_statistics.csv",
                    ["label_code", "image_count", "avg_per_image"], rows_image_class)
        _write_csv(self.output_root / "bbox_statistics.csv",
                    ["ann_id", "label_code", "bbox_width", "bbox_height", "bbox_area_px", "area_ratio", "geometry_type"],
                    rows_bbox)
        _write_csv(self.output_root / "geometry_statistics.csv",
                    ["label_code", "instance_count", "rect_count", "polygon_count"], rows_geom)
        logger.info("类别统计: %d 类", len(rows_class))

    # ------------------------------------------------------------------
    # 3. 重复标注检查
    # ------------------------------------------------------------------
    def _check_duplicates(self):
        # 3a: 完全相同的 (image_id, label_code, bbox)
        seen: Dict[Tuple, str] = {}
        for r in sorted(self.instances, key=lambda x: _natural_sort_key(x["ann_id"])):
            key = (r["image_id"], r["label_code"], r["x_min"], r["y_min"], r["x_max"], r["y_max"])
            if key in seen:
                self.dup_anns.append({
                    "type": "exact_duplicate", "ann_id_1": seen[key], "ann_id_2": r["ann_id"],
                    "image_id": r["image_id"], "label_code": r["label_code"],
                    "bbox": f"[{r['x_min']},{r['y_min']},{r['x_max']},{r['y_max']}]",
                })
            else:
                seen[key] = r["ann_id"]

        # 3b: 高度重复 (IoU >= 0.95)
        for iid, insts in self.inst_by_image.items():
            for i in range(len(insts)):
                for j in range(i + 1, len(insts)):
                    a, b = insts[i], insts[j]
                    if a["label_code"] != b["label_code"]:
                        continue
                    ax1, ay1, ax2, ay2 = map(int, [a["x_min"], a["y_min"], a["x_max"], a["y_max"]])
                    bx1, by1, bx2, by2 = map(int, [b["x_min"], b["y_min"], b["x_max"], b["y_max"]])
                    iou = _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
                    if iou >= 0.95:
                        self.dup_anns.append({
                            "type": "high_iou", "ann_id_1": a["ann_id"], "ann_id_2": b["ann_id"],
                            "image_id": iid, "label_code": a["label_code"], "iou": round(iou, 4),
                            "bbox_1": f"[{ax1},{ay1},{ax2},{ay2}]",
                            "bbox_2": f"[{bx1},{by1},{bx2},{by2}]",
                        })

        # 3c: 不同类别完全相同 bbox
        for iid, insts in self.inst_by_image.items():
            for i in range(len(insts)):
                for j in range(i + 1, len(insts)):
                    a, b = insts[i], insts[j]
                    if a["label_code"] == b["label_code"]:
                        continue
                    if (a["x_min"], a["y_min"], a["x_max"], a["y_max"]) == (b["x_min"], b["y_min"], b["x_max"], b["y_max"]):
                        self.dup_anns.append({
                            "type": "same_bbox_diff_label", "ann_id_1": a["ann_id"], "ann_id_2": b["ann_id"],
                            "image_id": iid, "label_1": a["label_code"], "label_2": b["label_code"],
                        })

        # 3d: ann_id 重复
        all_aids = [r["ann_id"] for r in self.instances]
        aid_dupes = sorted({a for a in set(all_aids) if all_aids.count(a) > 1})
        for d in aid_dupes:
            self.errors.append(f"ann_id 重复: {d}")

        fields = ["type", "ann_id_1", "ann_id_2", "image_id", "label_code",
                   "label_1", "label_2", "iou", "bbox", "bbox_1", "bbox_2"]
        _write_csv(self.output_root / "annotation_duplicate_report.csv", fields, self.dup_anns)
        logger.info("重复标注: %d 条", len(self.dup_anns))

    # ------------------------------------------------------------------
    # 4. 图片重复检查
    # ------------------------------------------------------------------
    def _check_image_duplicates(self):
        sha_map: Dict[str, str] = {}
        for iid, mr in self.manifest.items():
            fp = self.images_dir / mr["filename"]
            if fp.exists():
                sha_map[iid] = _sha256_file(fp)

        rev_map: Dict[str, List[str]] = defaultdict(list)
        for iid, sha in sha_map.items():
            rev_map[sha].append(iid)

        for sha, ids in rev_map.items():
            if len(ids) > 1:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        self.dup_images.append({
                            "image_id_1": ids[i], "image_id_2": ids[j],
                            "exact_duplicate": True, "phash_distance": "",
                            "review_required": True,
                        })

        _write_csv(self.output_root / "image_duplicate_report.csv",
                    ["image_id_1", "image_id_2", "exact_duplicate", "phash_distance", "review_required"],
                    self.dup_images)
        logger.info("图片重复: %d 对", len(self.dup_images))

    # ------------------------------------------------------------------
    # 5. bbox 重叠/包含
    # ------------------------------------------------------------------
    def _analyze_overlaps(self):
        overlap_rows: List[Dict] = []
        containment_rows: List[Dict] = []
        nested_stats: Dict[Tuple[str, str], int] = defaultdict(int)

        for iid, insts in self.inst_by_image.items():
            # 只分析缺陷类 A-G
            d_anns = [r for r in insts if r["label_code"] in self.DEFECT_CODES]
            for i in range(len(d_anns)):
                for j in range(i + 1, len(d_anns)):
                    a, b = d_anns[i], d_anns[j]
                    ax1, ay1, ax2, ay2 = map(int, [a["x_min"], a["y_min"], a["x_max"], a["y_max"]])
                    bx1, by1, bx2, by2 = map(int, [b["x_min"], b["y_min"], b["x_max"], b["y_max"]])
                    area_a = (ax2 - ax1) * (ay2 - ay1)
                    area_b = (bx2 - bx1) * (by2 - by1)
                    if area_a <= 0 or area_b <= 0:
                        continue

                    iou = _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
                    inter_area = _inter_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
                    ratio_smaller = inter_area / min(area_a, area_b) if min(area_a, area_b) > 0 else 0
                    a_contains_b = ax1 <= bx1 and ay1 <= by1 and ax2 >= bx2 and ay2 >= by2
                    b_contains_a = bx1 <= ax1 and by1 <= ay1 and bx2 >= ax2 and by2 >= ay2

                    if iou > 0:
                        overlap_rows.append({
                            "image_id": iid, "ann_id_1": a["ann_id"], "ann_id_2": b["ann_id"],
                            "label_1": a["label_code"], "label_2": b["label_code"],
                            "iou": round(iou, 4),
                            "inter_area": round(inter_area, 1),
                            "ratio_smaller": round(ratio_smaller, 4),
                            "a_contains_b": a_contains_b, "b_contains_a": b_contains_a,
                        })

                    if a_contains_b:
                        containment_rows.append({
                            "image_id": iid, "containing": a["ann_id"], "contained": b["ann_id"],
                            "label_containing": a["label_code"], "label_contained": b["label_code"],
                            "area_ratio": round(area_b / area_a, 4),
                        })
                        nested_stats[(a["label_code"], b["label_code"])] += 1
                    if b_contains_a:
                        containment_rows.append({
                            "image_id": iid, "containing": b["ann_id"], "contained": a["ann_id"],
                            "label_containing": b["label_code"], "label_contained": a["label_code"],
                            "area_ratio": round(area_a / area_b, 4),
                        })
                        nested_stats[(b["label_code"], a["label_code"])] += 1

        _write_csv(self.output_root / "overlap_pair_statistics.csv",
                    ["image_id", "ann_id_1", "ann_id_2", "label_1", "label_2",
                     "iou", "inter_area", "ratio_smaller", "a_contains_b", "b_contains_a"],
                    overlap_rows)
        _write_csv(self.output_root / "containment_statistics.csv",
                    ["image_id", "containing", "contained", "label_containing", "label_contained", "area_ratio"],
                    containment_rows)
        logger.info("重叠分析: %d 对重叠, %d 对包含", len(overlap_rows), len(containment_rows))
        return nested_stats, containment_rows

    # ------------------------------------------------------------------
    # 6. pseudo-normal 检查
    # ------------------------------------------------------------------
    def _check_pseudo_normal(self):
        pn_rows: List[Dict] = []
        high_risk = 0

        for iid, insts in self.inst_by_image.items():
            d_anns = [r for r in insts if r["label_code"] in self.DEFECT_CODES]
            n_anns = [r for r in insts if r["label_code"] in self.NORMAL_CODES]

            for nr in n_anns:
                nx1, ny1, nx2, ny2 = map(int, [nr["x_min"], nr["y_min"], nr["x_max"], nr["y_max"]])
                n_area = (nx2 - nx1) * (ny2 - ny1)
                max_iou = 0.0; max_ratio = 0.0

                for dr in d_anns:
                    dx1, dy1, dx2, dy2 = map(int, [dr["x_min"], dr["y_min"], dr["x_max"], dr["y_max"]])
                    iou = _iou(nx1, ny1, nx2, ny2, dx1, dy1, dx2, dy2)
                    inter = _inter_area(nx1, ny1, nx2, ny2, dx1, dy1, dx2, dy2)
                    ratio = inter / n_area if n_area > 0 else 0
                    max_iou = max(max_iou, iou)
                    max_ratio = max(max_ratio, ratio)

                risk = "ok"
                if max_ratio > 0.20:
                    risk = "high_risk"
                    high_risk += 1
                elif max_ratio > 0.05:
                    risk = "review"

                pn_rows.append({
                    "image_id": iid, "ann_id": nr["ann_id"],
                    "label_code": nr["label_code"],
                    "max_iou_with_defect": round(max_iou, 4),
                    "max_overlap_ratio": round(max_ratio, 4),
                    "risk_level": risk,
                })

        _write_csv(self.output_root / "pseudo_normal_overlap.csv",
                    ["image_id", "ann_id", "label_code", "max_iou_with_defect", "max_overlap_ratio", "risk_level"],
                    pn_rows)
        logger.info("Pseudo-normal: %d high_risk, %d total", high_risk, len(pn_rows))
        return high_risk

    # ------------------------------------------------------------------
    # 7. 可疑实例
    # ------------------------------------------------------------------
    def _collect_suspicious(self, containment_rows, pn_high_risk):
        for r in self.instances:
            reasons = []
            bw, bh = int(r["bbox_width"]), int(r["bbox_height"])
            lc = r["label_code"]
            iw, ih = int(r["image_width"]), int(r["image_height"])

            if bw < 5 or bh < 5:
                reasons.append("bbox_too_small")
            if iw * ih > 0 and (bw * bh) / (iw * ih) > 0.8:
                reasons.append("bbox_area_ratio_gt_0.8")
            aspect = bw / bh if bh > 0 else 999
            if aspect > 15 or aspect < 1 / 15:
                reasons.append("extreme_aspect_ratio")
            if r["geometry_type"] == "polygon":
                try:
                    pts = json.loads(r["points_json"])
                    if len(pts) < 3:
                        reasons.append("polygon_lt_3_points")
                except Exception:
                    reasons.append("polygon_points_parse_error")
            if lc == "G":
                reasons.append("label_code_G")
            if reasons:
                self.suspicious.append({
                    "ann_id": r["ann_id"], "image_id": r["image_id"],
                    "label_code": lc, "bbox_width": bw, "bbox_height": bh,
                    "geometry_type": r["geometry_type"],
                    "reasons": ";".join(reasons),
                })

        # 添加完全重复和高风险 pseudo-normal
        for dr in self.dup_anns:
            if dr["type"] == "exact_duplicate":
                self.suspicious.append({
                    "ann_id": dr["ann_id_2"], "image_id": dr.get("image_id", ""),
                    "label_code": dr.get("label_code", ""), "bbox_width": "", "bbox_height": "",
                    "geometry_type": "", "reasons": "exact_duplicate",
                })

        # 从 pseudo_normal_overlap.csv 中取 high_risk
        pn_csv = self.output_root / "pseudo_normal_overlap.csv"
        if pn_csv.exists():
            with open(pn_csv, "r", encoding="utf-8-sig") as f:
                for pr in csv.DictReader(f):
                    if pr["risk_level"] == "high_risk":
                        # avoid double-add
                        pass  # already scored in pn overlays

        _write_csv(self.output_root / "suspicious_instances.csv",
                    ["ann_id", "image_id", "label_code", "bbox_width", "bbox_height",
                     "geometry_type", "reasons"],
                    self.suspicious)
        logger.info("可疑实例: %d 条", len(self.suspicious))

    # ------------------------------------------------------------------
    # 8. 数据指纹
    # ------------------------------------------------------------------
    def _generate_fingerprint(self):
        sha_inst = _sha256_file(self.instances_csv)
        sha_sum = _sha256_file(self.summary_csv)
        sha_manifest = _sha256_file(self.manifest_csv)

        image_shas = {}
        for iid, mr in self.manifest.items():
            fp = self.images_dir / mr["filename"]
            if fp.exists():
                image_shas[iid] = _sha256_file(fp)

        fingerprint = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _try_git_hash(),
            "schema_version": self.schema.get("schema_version", "0.2"),
            "hashes": {
                "instances_csv": sha_inst,
                "image_summary_csv": sha_sum,
                "images_manifest_csv": sha_manifest,
            },
            "image_hashes": image_shas,
            "statistics": {
                "total_images": len(self.manifest),
                "total_instances": len(self.instances),
                "defect_instances": sum(self.N_DEFECT[c] for c in self.DEFECT_CODES),
                "normal_instances": sum(self.N_DEFECT[c] for c in self.NORMAL_CODES if c in self.N_DEFECT),
                "category_counts": {c: self.N_DEFECT.get(c, 0) for c in self.ALL_CODES},
            }
        }

        fp_path = self.derived_dir / "dataset_fingerprint.json"
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        fp_path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("数据指纹: %s", fp_path)

    # ------------------------------------------------------------------
    # 9. Contact Sheets & Figures (simplified for smoke)
    # ------------------------------------------------------------------
    def _generate_contact_sheets(self):
        if self.skip_contact_sheets:
            logger.info("跳过 contact sheet 生成")
            return
        try:
            self._do_generate_contact_sheets()
        except Exception as e:
            logger.warning("Contact sheet 生成失败: %s", e)

    def _do_generate_contact_sheets(self):
        from PIL import Image, ImageDraw, ImageFont

        cs_dir = self.output_root / "contact_sheets"
        for sub in ["by_class", "smallest_boxes", "largest_boxes", "extreme_aspect_ratio",
                     "nested_A_examples", "pseudo_normal_overlap"]:
            (cs_dir / sub).mkdir(parents=True, exist_ok=True)

        def _cs_thumb(img_path, bboxes, labels, output_path, title=""):
            try:
                img = Image.open(img_path).convert("RGBA")
                ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(ov)
                colors = {"A": "#ff6b6b", "B": "#ffa94d", "C": "#ffd43b", "D": "#69db7c",
                           "E": "#74c0fc", "F": "#da77f2", "G": "#adb5bd",
                           "N1": "#38d9a9", "N2": "#4dabf7", "N3": "#b197fc"}
                for (x1, y1, x2, y2), lc in zip(bboxes, labels):
                    c = colors.get(lc, "#fff")
                    draw.rectangle([x1, y1, x2, y2], outline=c, width=2)
                res = Image.alpha_composite(img, ov).convert("RGB")
                res.save(output_path, "JPEG", quality=85)
                return True
            except Exception:
                return False

        # by_class: 每类取前 4 张示例
        for lc in self.ALL_CODES:
            insts = [r for r in self.instances if r["label_code"] == lc]
            if not insts: continue
            iids_seen = []
            seen_iids = set()
            for r in insts:
                if r["image_id"] not in seen_iids:
                    iids_seen.append(r["image_id"])
                    seen_iids.add(r["image_id"])
                if len(iids_seen) >= 4: break
            bboxes = [[int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"])]
                       for r in insts if r["image_id"] in seen_iids][:4]
            labels_i = [r["label_code"] for r in insts if r["image_id"] in seen_iids][:4]
            fp = self.images_dir / self.manifest.get(iids_seen[0], {}).get("filename", "")
            if fp.exists() and bboxes:
                _cs_thumb(fp, bboxes, labels_i, cs_dir / "by_class" / f"sample_{lc}.jpg", lc)

        logger.info("Contact sheets: 已生成 by_class 示例")

    def _generate_figures(self):
        if self.skip_figures:
            logger.info("跳过图表生成")
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            fig_dir = self.output_root / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)

            # 1. class_distribution
            cats = [r["label_code"] for r in self.instances]
            cc = Counter(cats)
            labels = self.DEFECT_CODES + self.NORMAL_CODES
            counts = [cc.get(c, 0) for c in labels]
            colors = ["#ff6b6b", "#ffa94d", "#ffd43b", "#69db7c", "#74c0fc", "#da77f2", "#adb5bd",
                       "#38d9a9", "#4dabf7", "#b197fc"]
            plt.figure(figsize=(10, 5))
            plt.bar(labels, counts, color=colors)
            plt.title("Class Distribution")
            plt.xlabel("Label Code"); plt.ylabel("Instance Count")
            for i, v in enumerate(counts):
                if v > 0: plt.text(i, v + 5, str(v), ha="center", fontsize=8)
            plt.tight_layout(); plt.savefig(fig_dir / "class_distribution.png", dpi=100); plt.close()

            # 2. bbox_area_distribution
            areas = [int(r["bbox_area_px"]) for r in self.instances if int(r["bbox_area_px"]) > 0]
            plt.figure(figsize=(8, 4))
            plt.hist(np.log10(areas), bins=50, color="#7c6ff0", edgecolor="white")
            plt.title("BBox Area Distribution (log10)")
            plt.xlabel("log10(area_px)"); plt.ylabel("Count")
            plt.tight_layout(); plt.savefig(fig_dir / "bbox_area_distribution.png", dpi=100); plt.close()

            # 3. bbox_width_height_scatter
            bws = [int(r["bbox_width"]) for r in self.instances if int(r["bbox_width"]) > 0]
            bhs = [int(r["bbox_height"]) for r in self.instances if int(r["bbox_height"]) > 0]
            plt.figure(figsize=(6, 6))
            plt.scatter(bws, bhs, s=2, alpha=0.3, c="#7c6ff0")
            plt.title("BBox Width vs Height")
            plt.xlabel("Width"); plt.ylabel("Height")
            plt.tight_layout(); plt.savefig(fig_dir / "bbox_width_height_scatter.png", dpi=100); plt.close()

            # 4. class_image_coverage
            img_counts = defaultdict(int)
            for iid, insts in self.inst_by_image.items():
                for r in insts:
                    img_counts[r["label_code"]] += 1
            # unique image count per class
            img_uniq = {lc: len(set(r["image_id"] for r in self.instances if r["label_code"] == lc))
                         for lc in self.ALL_CODES}
            plt.figure(figsize=(8, 4))
            plt.bar(self.ALL_CODES, [img_uniq.get(c, 0) for c in self.ALL_CODES], color=colors)
            plt.title("Images Containing Each Class")
            plt.tight_layout(); plt.savefig(fig_dir / "class_image_coverage.png", dpi=100); plt.close()

            logger.info("图表已生成: %s", fig_dir)
        except Exception as e:
            logger.warning("图表生成失败: %s", e)

    # ------------------------------------------------------------------
    # 10. 审计总结
    # ------------------------------------------------------------------
    def _generate_summary(self, nested_stats, high_risk_pn):
        lines = [
            "# AOI 训练数据审计报告",
            f"生成时间: {datetime.now().isoformat()}",
            "",
            "## 基础统计",
            f"- Manifest 图片数: {len(self.manifest)}",
            f"- 标注实例数: {len(self.instances)}",
            f"- 缺陷实例: {sum(self.N_DEFECT[c] for c in self.DEFECT_CODES)}",
            f"- Pseudo-normal 实例: {sum(self.N_DEFECT.get(c,0) for c in self.NORMAL_CODES)}",
            "",
            "## 类别数量",
        ]
        for c in self.ALL_CODES:
            lines.append(f"- {c}: {self.N_DEFECT.get(c, 0)}")
        lines += [
            "",
            "## 重复检查",
            f"- 重复标注: {len(self.dup_anns)} 条",
            f"- 完全相同的图片: {len(self.dup_images)} 对",
            "",
            "## Pseudo-normal 检查",
            f"- High-risk (重叠>20%): {high_risk_pn}",
            "",
            "## 可疑实例",
            f"- 总数: {len(self.suspicious)}",
            "",
            "## A 类包含其他类别统计",
        ]
        for lc in self.DEFECT_CODES[1:]:  # B..G
            cnt = nested_stats.get(("A", lc), 0)
            if cnt > 0:
                lines.append(f"- A 包含 {lc}: {cnt} 次")
        lines += [
            "",
            "## 审计结论",
            f"- 错误: {len(self.errors)}",
            f"- 警告: {len(self.warnings)}",
        ]
        if self.errors:
            lines.append("\n### Errors")
            for e in self.errors[:20]:
                lines.append(f"- {e}")
            lines.append(f"  ... 共 {len(self.errors)} 条")
        (self.output_root / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
        logger.info("审计总结: %s", self.output_root / "audit_summary.md")

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------
    def run(self) -> bool:
        logger.info("=" * 50)
        logger.info("开始数据审计...")
        ok = self._check_basics()
        self._class_stats()
        self._check_duplicates()
        self._check_image_duplicates()
        nested_stats, containment_rows = self._analyze_overlaps()
        high_risk_pn = self._check_pseudo_normal()
        self._collect_suspicious(containment_rows, high_risk_pn)
        self._generate_fingerprint()
        self._generate_contact_sheets()
        self._generate_figures()
        self._generate_summary(nested_stats, high_risk_pn)

        logger.info("=" * 50)
        logger.info("审计完成")
        logger.info("  基础检查: %s", "PASS" if ok else "FAIL")
        logger.info("  重复标注: %d 条", len(self.dup_anns))
        logger.info("  Exact duplicate 图片: %d 对", len(self.dup_images))
        logger.info("  Pseudo-normal high_risk: %d", high_risk_pn)
        logger.info("  A 包含 B-F: %s", {k: v for k, v in nested_stats.items() if k[0] == "A"})
        logger.info("  可疑实例: %d 条", len(self.suspicious))
        logger.info("  输出: %s", self.output_root)
        return ok


# =============================================================================
# 几何工具
# =============================================================================
def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    x1 = max(ax1, bx1); y1 = max(ay1, by1)
    x2 = min(ax2, bx2); y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0


def _inter_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    x1 = max(ax1, bx1); y1 = max(ay1, by1)
    x2 = min(ax2, bx2); y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1: return 0.0
    return float((x2 - x1) * (y2 - y1))


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="训练前数据审计")
    p.add_argument("--instances_csv", type=str, default="data/annotations/instances.csv")
    p.add_argument("--summary_csv", type=str, default="data/annotations/image_summary.csv")
    p.add_argument("--manifest_csv", type=str, default="data/metadata/images_manifest.csv")
    p.add_argument("--schema_yaml", type=str, default="configs/label_schema.yaml")
    p.add_argument("--images_dir", type=str, default="data/images")
    p.add_argument("--output_dir", type=str, default="reports/data_audit")
    p.add_argument("--no-figures", action="store_true", default=False)
    p.add_argument("--no-contact-sheets", action="store_true", default=False)
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    auditor = TrainingDataAuditor(
        instances_csv=(root / args.instances_csv).resolve(),
        summary_csv=(root / args.summary_csv).resolve(),
        manifest_csv=(root / args.manifest_csv).resolve(),
        schema_yaml=(root / args.schema_yaml).resolve(),
        images_dir=(root / args.images_dir).resolve(),
        output_root=(root / args.output_dir).resolve(),
        skip_figures=args.no_figures,
        skip_contact_sheets=args.no_contact_sheets,
    )
    ok = auditor.run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
