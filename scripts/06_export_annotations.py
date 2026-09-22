#!/usr/bin/env python3
"""
06_export_annotations.py — 导出标注数据为 CSV

读取 annotations.json + images_manifest.csv + label_schema.yaml，
导出两个 CSV（UTF-8-SIG 编码，原子替换）：
  1. data/annotations/instances.csv    — 每个标注实例一行 (20 字段)
  2. data/annotations/image_summary.csv — 每张图一行 (32 字段)

用法:
    python scripts/06_export_annotations.py
    python scripts/06_export_annotations.py --manifest_csv data/metadata/images_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("export")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists(): logger.error("不存在: %s", path); sys.exit(1)
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def load_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists(): logger.warning("Manifest 不存在: %s", path); return {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return {r.get("image_id", ""): r for r in csv.DictReader(f)}

def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists(): logger.error("Schema 不存在: %s", path); sys.exit(1)
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)

def build_maps(schema: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    label_map: Dict[str, str] = {}; display_map: Dict[str, str] = {}
    for section in ("defect_classes", "normal_classes"):
        for code, info in schema.get(section, {}).items():
            label_map[code] = info.get("name", "")
            display_map[code] = info.get("display_name", "")
    return label_map, display_map

def _natural_sort_key(s: str) -> Tuple:
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s))

def _atomic_write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=str(path.parent), prefix=".tmp_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path): os.unlink(tmp_path)
        raise


INSTANCE_FIELDS = [
    "ann_id","image_id","filename","image_width","image_height",
    "label_code","label","display_name","is_pseudo_normal",
    "geometry_type","points_json",
    "x_min","y_min","x_max","y_max",
    "bbox_width","bbox_height","bbox_area_px","area_px","image_status",
]


def export_instances(data: Dict[str, Any], manifest_map: Dict[str, Dict[str, str]],
                     label_map: Dict[str, str], display_map: Dict[str, str], output_path: Path) -> int:
    images = data.get("images", {})
    m_order = sorted(manifest_map.keys(), key=_natural_sort_key)
    rows: List[Dict[str, str]] = []
    for iid in m_order:
        rec = images.get(iid, {}); mr = manifest_map.get(iid, {})
        fn = rec.get("filename") or mr.get("filename", "")
        iw = int(mr.get("width", 0)); ih = int(mr.get("height", 0))
        st = rec.get("status", "")
        anns = sorted(rec.get("annotations", []), key=lambda a: _natural_sort_key(a.get("ann_id", "")))
        for ann in anns:
            lc = ann.get("label_code", "")
            bbox = ann.get("bbox", [0,0,0,0])
            x1,y1,x2,y2 = (int(bbox[i]) if i < len(bbox) else 0 for i in range(4))
            bw, bh = max(0,x2-x1), max(0,y2-y1)
            barea = bw * bh
            area = float(ann.get("area_px", 0))
            gt = ann.get("geometry_type", "rectangle")
            if gt == "rectangle": area = float(barea) if area == 0 else area
            rows.append({
                "ann_id": ann.get("ann_id",""), "image_id": iid, "filename": fn,
                "image_width": str(iw), "image_height": str(ih),
                "label_code": lc, "label": label_map.get(lc, ann.get("label","")),
                "display_name": display_map.get(lc, ""),
                "is_pseudo_normal": "True" if ann.get("is_pseudo_normal") else "False",
                "geometry_type": gt,
                "points_json": json.dumps(ann.get("points",[]), ensure_ascii=False),
                "x_min": str(x1), "y_min": str(y1), "x_max": str(x2), "y_max": str(y2),
                "bbox_width": str(bw), "bbox_height": str(bh),
                "bbox_area_px": str(barea), "area_px": str(area),
                "image_status": st,
            })
    _atomic_write_csv(output_path, INSTANCE_FIELDS, rows)
    logger.info("instances.csv → %s (%d 条)", output_path, len(rows))
    return len(rows)


SUMMARY_FIELDS = [
    "image_id","filename","image_width","image_height","image_status",
    "total_annotation_count","defect_instance_count","pseudo_normal_instance_count",
    "unique_defect_class_count","has_multiple_defect_instances","has_multiple_defect_classes",
    "defect_labels_present","pseudo_normal_labels_present",
    "has_A","has_B","has_C","has_D","has_E","has_F","has_G",
    "count_A","count_B","count_C","count_D","count_E","count_F","count_G",
    "has_N1","has_N2","has_N3","count_N1","count_N2","count_N3",
]
DEFECT_ALL = ["A","B","C","D","E","F","G"]
NORMAL_ALL = ["N1","N2","N3"]


def export_summary(data: Dict[str, Any], manifest_map: Dict[str, Dict[str, str]], output_path: Path) -> int:
    images = data.get("images", {})
    m_order = sorted(manifest_map.keys(), key=_natural_sort_key)
    rows: List[Dict[str, str]] = []
    for iid in m_order:
        rec = images.get(iid, {}); mr = manifest_map.get(iid, {})
        fn = rec.get("filename") or mr.get("filename", "")
        iw, ih = mr.get("width","0"), mr.get("height","0")
        st = rec.get("status", "")
        anns = rec.get("annotations", [])
        d_anns = [a for a in anns if not a.get("is_pseudo_normal")]
        n_anns = [a for a in anns if a.get("is_pseudo_normal")]
        total, nd, nn = len(anns), len(d_anns), len(n_anns)
        dc: Dict[str,int] = {}; nc: Dict[str,int] = {}
        for a in d_anns:
            lc = a.get("label_code","")
            if lc in DEFECT_ALL: dc[lc] = dc.get(lc,0)+1
        for a in n_anns:
            lc = a.get("label_code","")
            if lc in NORMAL_ALL: nc[lc] = nc.get(lc,0)+1
        pd = [c for c in DEFECT_ALL if dc.get(c,0)>0]
        uniq = len(pd)
        row = {
            "image_id":iid,"filename":fn,"image_width":iw,"image_height":ih,
            "image_status":st,
            "total_annotation_count":str(total),
            "defect_instance_count":str(nd),
            "pseudo_normal_instance_count":str(nn),
            "unique_defect_class_count":str(uniq),
            "has_multiple_defect_instances":"1" if nd>1 else "0",
            "has_multiple_defect_classes":"1" if uniq>1 else "0",
            "defect_labels_present":";".join(pd),
            "pseudo_normal_labels_present":";".join([c for c in NORMAL_ALL if nc.get(c,0)>0]),
        }
        for c in DEFECT_ALL:
            row[f"has_{c}"] = "1" if dc.get(c,0)>0 else "0"
            row[f"count_{c}"] = str(dc.get(c,0))
        for c in NORMAL_ALL:
            row[f"has_{c}"] = "1" if nc.get(c,0)>0 else "0"
            row[f"count_{c}"] = str(nc.get(c,0))
        rows.append(row)
    _atomic_write_csv(output_path, SUMMARY_FIELDS, rows)
    logger.info("image_summary.csv → %s (%d 行)", output_path, len(rows))
    return len(rows)


def main():
    p = argparse.ArgumentParser(description="导出标注数据为 CSV")
    p.add_argument("--annotations_json", type=str, default="data/annotations/annotations.json")
    p.add_argument("--manifest_csv", type=str, default="data/metadata/images_manifest.csv")
    p.add_argument("--schema_yaml", type=str, default="configs/label_schema.yaml")
    p.add_argument("--instances_csv", type=str, default="data/annotations/instances.csv")
    p.add_argument("--summary_csv", type=str, default="data/annotations/image_summary.csv")
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    data = load_json((root/args.annotations_json).resolve())
    mm = load_manifest((root/args.manifest_csv).resolve())
    schema = load_schema((root/args.schema_yaml).resolve())
    lm, dm = build_maps(schema)
    logger.info("Manifest: %d 张, 标注图片: %d 张", len(mm), len(data.get("images",{})))
    n1 = export_instances(data, mm, lm, dm, (root/args.instances_csv).resolve())
    n2 = export_summary(data, mm, (root/args.summary_csv).resolve())
    logger.info("导出完成: %d 个实例, %d 张图片汇总", n1, n2)

if __name__ == "__main__": main()
