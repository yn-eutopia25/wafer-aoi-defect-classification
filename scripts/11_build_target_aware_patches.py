#!/usr/bin/env python3
"""
11_build_target_aware_patches.py — 目标感知 patch 数据集生成器

从标注实例裁剪 patch，支持多种策略：
  - B-F 类：context-aware 正方形 crop + raw/focused/mask
  - A 类：clean tile 采样 + 可选 focused bbox
  - N1-N3：normal patch（不参与缺陷六分类）

用法:
    python scripts/11_build_target_aware_patches.py
    python scripts/11_build_target_aware_patches.py --no-contact-sheets --smoke 50
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("build_patches")

# =============================================================================
# 配置
# =============================================================================
# B-F 类的 context factor + min crop
CLASS_CONFIG = {
    "B": {"context": 1.8, "min_crop": 96},
    "C": {"context": 3.0, "min_crop": 96},
    "D": {"context": 1.5, "min_crop": 128},
    "E": {"context": 1.8, "min_crop": 96},
    "F": {"context": 1.8, "min_crop": 96},
}

A_TILE_SIZE = 128
A_TILE_MAX_PER_ANN = 3
A_TILE_MAX_PER_IMG = 10
A_TILE_MIN_AREA_RATIO = 0.85
A_TILE_MAX_OVERLAP_RATIO = 0.02

RANDOM_SEED = 42

PATCH_MANIFEST_FIELDS = [
    "patch_id", "image_id", "ann_id", "filename", "split",
    "source_label", "train_label", "is_pseudo_normal",
    "patch_strategy",
    "raw_patch_path", "focused_patch_path", "mask_path",
    "source_x_min", "source_y_min", "source_x_max", "source_y_max",
    "crop_x_min", "crop_y_min", "crop_x_max", "crop_y_max",
    "crop_width", "crop_height", "bbox_width", "bbox_height",
    "bbox_area_px", "target_area_px", "other_overlap_ratio",
    "use_for_defect_six_class", "use_for_normal_defect",
    "excluded_reason",
]

# =============================================================================
# 工具
# =============================================================================
def _natural_sort_key(s: str) -> Tuple:
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s))

def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=str(path.parent), prefix=".tmp_")
    with os.fdopen(tmp_fd, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: str(v) for k, v in r.items()})
    os.replace(tmp_path, path)

def load_instances(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def load_splits(path: Path) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return {r["image_id"]: r["split"] for r in csv.DictReader(f)}

def load_pn_overlap(path: Path) -> Dict[str, float]:
    """返回 {ann_id: max_overlap_ratio}。"""
    m: Dict[str, float] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                m[r["ann_id"]] = float(r["max_overlap_ratio"])
    return m

# =============================================================================
# 核心: patch 生成器
# =============================================================================
class PatchBuilder:
    def __init__(self, instances_csv: Path, splits_csv: Path,
                 images_dir: Path, pn_overlap_csv: Path,
                 output_root: Path, skip_contact: bool = False,
                 smoke: int = 0):
        self.instances = load_instances(instances_csv)
        self.splits = load_splits(splits_csv)
        self.images_dir = images_dir
        self.pn_overlap = load_pn_overlap(pn_overlap_csv)
        self.output_root = output_root
        self.skip_contact = skip_contact
        self.smoke = smoke
        self.rng = np.random.RandomState(RANDOM_SEED)

        self.patch_id_counter = 0
        self.manifest_rows: List[Dict[str, Any]] = []
        self.excluded_rows: List[Dict[str, Any]] = []
        self.class_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.a_tiles_per_img: Dict[str, int] = defaultdict(int)

        # Group by image_id
        self.inst_by_image: Dict[str, List[Dict]] = defaultdict(list)
        for r in self.instances:
            self.inst_by_image[r["image_id"]].append(r)

    def _next_patch_id(self, iid: str) -> str:
        self.patch_id_counter += 1
        return f"{iid}_patch_{self.patch_id_counter:05d}"

    # ---- 通用: 加载图片 + 缓存 ----
    _img_cache: Dict[str, np.ndarray] = {}

    def _load_image(self, filename: str) -> np.ndarray:
        if filename not in self._img_cache:
            fp = self.images_dir / filename
            # cv2.imread 不支持中文路径, 使用 imdecode
            img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"图片无法读取: {fp}")
            self._img_cache[filename] = img
        return self._img_cache[filename]

    # ---- 核心 crop ----
    def _crop_patch(self, img: np.ndarray, cx: int, cy: int, half: int) -> Tuple[np.ndarray, int, int, int, int]:
        """从 img 中以 (cx,cy) 为中心裁出 (2*half) × (2*half) 的正方形。

        Returns: (patch, crop_x1, crop_y1, crop_x2, crop_y2)
        使用 reflect padding 处理边界。
        """
        h, w = img.shape[:2]
        x1 = cx - half; y1 = cy - half
        x2 = x1 + 2 * half; y2 = y1 + 2 * half

        # reflect padding
        pad_left = max(0, -x1); pad_top = max(0, -y1)
        pad_right = max(0, x2 - w); pad_bottom = max(0, y2 - h)

        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(w, x2); y2c = min(h, y2)
        cropped = img[y1c:y2c, x1c:x2c]

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            cropped = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)

        return cropped, x1, y1, x2, y2

    # ---- mask 生成 ----
    def _build_mask(self, patch_h: int, patch_w: int, ann: Dict, crop_x1: int, crop_y1: int) -> np.ndarray:
        """生成目标区域 mask (0/255)，坐标相对于 patch。"""
        mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
        gt = ann["geometry_type"]
        bbox = [int(ann[k]) for k in ("x_min", "y_min", "x_max", "y_max")]

        if gt == "rectangle":
            x1 = max(0, bbox[0] - crop_x1)
            y1 = max(0, bbox[1] - crop_y1)
            x2 = min(patch_w, bbox[2] - crop_x1)
            y2 = min(patch_h, bbox[3] - crop_y1)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
        elif gt == "polygon":
            try:
                pts = json.loads(ann["points_json"])
                pts_local = np.array([[[p[0] - crop_x1, p[1] - crop_y1]] for p in pts], dtype=np.int32)
                cv2.fillPoly(mask, [pts_local], 255)
            except Exception:
                pass
        return mask

    # ---- focused patch ----
    def _make_focused(self, raw_patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """目标区域外轻度 Gaussian blur。"""
        h, w = raw_patch.shape[:2]
        ks = max(3, min(101, (min(h, w) // 20) | 1))
        blurred = cv2.GaussianBlur(raw_patch, (ks, ks), 0)
        mask_3c = mask[:, :, np.newaxis] // 255
        focused = (raw_patch.astype(np.float32) * mask_3c + blurred.astype(np.float32) * (1 - mask_3c)).astype(np.uint8)
        return focused

    # ---- save helpers ----
    def _save_patch(self, subdir: str, split: str, name: str, img: np.ndarray) -> str:
        """保存图片到 output/subdir/split/name，返回相对路径。"""
        d = self.output_root / "patches" / subdir / split
        d.mkdir(parents=True, exist_ok=True)
        fp = d / name
        # 使用 imencode 避免中文路径问题
        ext = name.rsplit(".", 1)[-1]
        ok, buf = cv2.imencode(f".{ext}", img)
        if not ok:
            raise IOError(f"图片编码失败: {fp}")
        buf.tofile(str(fp))
        return f"data/derived/patches/{subdir}/{split}/{name}"

    # ==================================================================
    # B-F 类
    # ==================================================================
    def _process_defect_bf(self, ann: Dict, split: str):
        lc = ann["label_code"]
        if lc not in CLASS_CONFIG:
            return
        cfg = CLASS_CONFIG[lc]
        bbox = [int(ann[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2

        half = max(cfg["min_crop"] // 2, int(cfg["context"] * max(bw, bh) / 2))

        img = self._load_image(ann["filename"])
        raw, x1, y1, x2, y2 = self._crop_patch(img, cx, cy, half)
        mask = self._build_mask(raw.shape[0], raw.shape[1], ann, x1, y1)
        focused = self._make_focused(raw, mask)

        patch_id = self._next_patch_id(ann["image_id"])
        rname = f"{patch_id}.jpg"; mname = f"{patch_id}_mask.png"

        raw_path = self._save_patch("raw", split, rname, raw)
        mask_path = self._save_patch("masks", split, mname, mask)
        focused_path = self._save_patch("focused", split, rname, focused)

        self.manifest_rows.append({
            "patch_id": patch_id, "image_id": ann["image_id"], "ann_id": ann["ann_id"],
            "filename": ann["filename"], "split": split,
            "source_label": lc, "train_label": lc, "is_pseudo_normal": False,
            "patch_strategy": "context_aware",
            "raw_patch_path": raw_path, "focused_patch_path": focused_path, "mask_path": mask_path,
            "source_x_min": bbox[0], "source_y_min": bbox[1], "source_x_max": bbox[2], "source_y_max": bbox[3],
            "crop_x_min": x1, "crop_y_min": y1, "crop_x_max": x2, "crop_y_max": y2,
            "crop_width": raw.shape[1], "crop_height": raw.shape[0],
            "bbox_width": bw, "bbox_height": bh, "bbox_area_px": bw * bh,
            "target_area_px": int(mask.sum() // 255),
            "other_overlap_ratio": 0.0,
            "use_for_defect_six_class": True, "use_for_normal_defect": False, "excluded_reason": "",
        })
        self.class_counts[lc][split] += 1

    # ==================================================================
    # A 类
    # ==================================================================
    def _process_defect_A(self, ann: Dict, split: str, all_defects_in_img: List[Dict]):
        iid = ann["image_id"]; img = self._load_image(ann["filename"])
        iw, ih = int(ann["image_width"]), int(ann["image_height"])
        bbox_A = [int(ann[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
        a_area = (bbox_A[2] - bbox_A[0]) * (bbox_A[3] - bbox_A[1])
        img_area = iw * ih

        # 收集同图中 B-F 的 bbox
        bf_bboxes = []
        for dr in all_defects_in_img:
            lc = dr["label_code"]
            if lc in CLASS_CONFIG and dr["ann_id"] != ann["ann_id"]:
                bx = [int(dr[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
                bx[0] = max(0, bx[0] - 3); bx[1] = max(0, bx[1] - 3)
                bx[2] = min(iw, bx[2] + 3); bx[3] = min(ih, bx[3] + 3)
                bf_bboxes.append(bx)

        # 计算 other_overlap_ratio
        other_area = 0
        canv = np.zeros((ih, iw), dtype=np.uint8)
        for bx in bf_bboxes:
            canv[bx[1]:bx[3], bx[0]:bx[2]] = 1
        a_mask = np.zeros((ih, iw), dtype=np.uint8)
        a_mask[bbox_A[1]:bbox_A[3], bbox_A[0]:bbox_A[2]] = 1
        other_area = int((canv * a_mask).sum())
        overlap_ratio = other_area / a_area if a_area > 0 else 0

        # 整框 patch (条件)
        if overlap_ratio <= 0.05 and a_area / img_area <= 0.25:
            half = max(96 // 2, int(max(bbox_A[2] - bbox_A[0], bbox_A[3] - bbox_A[1]) * 1.2 / 2))
            cx, cy = (bbox_A[0] + bbox_A[2]) // 2, (bbox_A[1] + bbox_A[3]) // 2
            raw, x1, y1, x2, y2 = self._crop_patch(img, cx, cy, half)
            mask = self._build_mask(raw.shape[0], raw.shape[1], ann, x1, y1)
            focused = self._make_focused(raw, mask)

            pid = self._next_patch_id(iid)
            rname = f"{pid}.jpg"; mname = f"{pid}_mask.png"
            raw_path = self._save_patch("raw", split, rname, raw)
            mask_path = self._save_patch("masks", split, mname, mask)
            focused_path = self._save_patch("focused", split, rname, focused)

            self.manifest_rows.append({
                "patch_id": pid, "image_id": iid, "ann_id": ann["ann_id"],
                "filename": ann["filename"], "split": split,
                "source_label": "A", "train_label": "A", "is_pseudo_normal": False,
                "patch_strategy": "A_focused_bbox",
                "raw_patch_path": raw_path, "focused_patch_path": focused_path, "mask_path": mask_path,
                "source_x_min": bbox_A[0], "source_y_min": bbox_A[1], "source_x_max": bbox_A[2], "source_y_max": bbox_A[3],
                "crop_x_min": x1, "crop_y_min": y1, "crop_x_max": x2, "crop_y_max": y2,
                "crop_width": raw.shape[1], "crop_height": raw.shape[0],
                "bbox_width": bbox_A[2] - bbox_A[0], "bbox_height": bbox_A[3] - bbox_A[1],
                "bbox_area_px": a_area, "target_area_px": int(mask.sum() // 255),
                "other_overlap_ratio": round(overlap_ratio, 4),
                "use_for_defect_six_class": True, "use_for_normal_defect": False, "excluded_reason": "",
            })
            self.class_counts["A"][split] += 1

        # Tile sampling
        tiles_found = 0
        tile_attempts = 0
        while tiles_found < A_TILE_MAX_PER_ANN and tile_attempts < 30:
            tile_attempts += 1
            if self.a_tiles_per_img[iid] >= A_TILE_MAX_PER_IMG:
                break
            tx = self.rng.randint(bbox_A[0] - A_TILE_SIZE // 2, bbox_A[2] - A_TILE_SIZE // 2 + 1)
            ty = self.rng.randint(bbox_A[1] - A_TILE_SIZE // 2, bbox_A[3] - A_TILE_SIZE // 2 + 1)
            tx = max(0, min(tx, iw - A_TILE_SIZE))
            ty = max(0, min(ty, ih - A_TILE_SIZE))

            tile_img = img[ty:ty + A_TILE_SIZE, tx:tx + A_TILE_SIZE]
            tile_a_mask = a_mask[ty:ty + A_TILE_SIZE, tx:tx + A_TILE_SIZE]
            tile_bf_mask = canv[ty:ty + A_TILE_SIZE, tx:tx + A_TILE_SIZE]

            a_pixels = int(tile_a_mask.sum())
            bf_pixels = int(tile_bf_mask.sum())
            total_pixels = A_TILE_SIZE * A_TILE_SIZE

            if a_pixels / total_pixels < A_TILE_MIN_AREA_RATIO:
                continue
            if bf_pixels / total_pixels > A_TILE_MAX_OVERLAP_RATIO:
                continue

            # 检查是否近纯黑/纯白
            gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
            if gray.std() < 10 or gray.mean() > 240 or gray.mean() < 15:
                continue

            pid = self._next_patch_id(iid)
            rname = f"{pid}.jpg"

            raw_path = self._save_patch("raw", split, rname, tile_img)
            mask = np.where(tile_a_mask > 0, 255, 0).astype(np.uint8)
            mname = f"{pid}_mask.png"
            mask_path = self._save_patch("masks", split, mname, mask)
            focused = self._make_focused(tile_img, mask)
            focused_path = self._save_patch("focused", split, rname, focused)

            self.manifest_rows.append({
                "patch_id": pid, "image_id": iid, "ann_id": ann["ann_id"],
                "filename": ann["filename"], "split": split,
                "source_label": "A", "train_label": "A", "is_pseudo_normal": False,
                "patch_strategy": "A_clean_tile",
                "raw_patch_path": raw_path, "focused_patch_path": focused_path, "mask_path": mask_path,
                "source_x_min": bbox_A[0], "source_y_min": bbox_A[1], "source_x_max": bbox_A[2], "source_y_max": bbox_A[3],
                "crop_x_min": tx, "crop_y_min": ty, "crop_x_max": tx + A_TILE_SIZE, "crop_y_max": ty + A_TILE_SIZE,
                "crop_width": A_TILE_SIZE, "crop_height": A_TILE_SIZE,
                "bbox_width": bbox_A[2] - bbox_A[0], "bbox_height": bbox_A[3] - bbox_A[1],
                "bbox_area_px": a_area, "target_area_px": int(mask.sum() // 255),
                "other_overlap_ratio": round(overlap_ratio, 4),
                "use_for_defect_six_class": True, "use_for_normal_defect": False, "excluded_reason": "",
            })
            self.class_counts["A"][split] += 1
            self.a_tiles_per_img[iid] += 1
            tiles_found += 1

        # 如果没有找到任何 tile 且没有生成整框 patch，记录 excluded
        if tiles_found == 0 and (overlap_ratio > 0.05 or a_area / img_area > 0.25 or True):
            # 检查这个 annotation 是否已经有 patch 生成
            has_patches = any(r["ann_id"] == ann["ann_id"] for r in self.manifest_rows)
            if not has_patches:
                self.excluded_rows.append({
                    "ann_id": ann["ann_id"], "image_id": iid, "label_code": "A",
                    "reason": "A_NO_CLEAN_TILE",
                    "note": f"overlap_ratio={overlap_ratio:.4f} a_area_ratio={a_area/img_area:.4f}",
                })

    # ==================================================================
    # Pseudo-normal
    # ==================================================================
    def _process_normal(self, ann: Dict, split: str):
        iid = ann["image_id"]; lc = ann["label_code"]
        overlap = self.pn_overlap.get(ann["ann_id"], 0.0)

        if overlap > 0.05:
            self.excluded_rows.append({
                "ann_id": ann["ann_id"], "image_id": iid, "label_code": lc,
                "reason": "NORMAL_OVERLAP_HIGH", "note": f"overlap_ratio={overlap:.4f}",
            })
            return

        bbox = [int(ann[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        half = max(64, int(2.0 * max(bw, bh) / 2))

        img = self._load_image(ann["filename"])
        raw, x1, y1, x2, y2 = self._crop_patch(img, cx, cy, half)
        mask = self._build_mask(raw.shape[0], raw.shape[1], ann, x1, y1)
        focused = self._make_focused(raw, mask)

        pid = self._next_patch_id(iid)
        rname = f"{pid}.jpg"; mname = f"{pid}_mask.png"
        raw_path = self._save_patch("raw", split, rname, raw)
        mask_path = self._save_patch("masks", split, mname, mask)
        focused_path = self._save_patch("focused", split, rname, focused)

        self.manifest_rows.append({
            "patch_id": pid, "image_id": iid, "ann_id": ann["ann_id"],
            "filename": ann["filename"], "split": split,
            "source_label": lc, "train_label": "normal", "is_pseudo_normal": True,
            "patch_strategy": "context_aware",
            "raw_patch_path": raw_path, "focused_patch_path": focused_path, "mask_path": mask_path,
            "source_x_min": bbox[0], "source_y_min": bbox[1], "source_x_max": bbox[2], "source_y_max": bbox[3],
            "crop_x_min": x1, "crop_y_min": y1, "crop_x_max": x2, "crop_y_max": y2,
            "crop_width": raw.shape[1], "crop_height": raw.shape[0],
            "bbox_width": bw, "bbox_height": bh, "bbox_area_px": bw * bh,
            "target_area_px": int(mask.sum() // 255),
            "other_overlap_ratio": overlap,
            "use_for_defect_six_class": False, "use_for_normal_defect": True, "excluded_reason": "",
        })
        self.class_counts["N_all"][split] += 1

    # ==================================================================
    # 运行
    # ==================================================================
    def run(self):
        logger.info("开始构建 patch 数据集...")
        total_processed = 0

        for iid in sorted(self.inst_by_image.keys(), key=_natural_sort_key):
            split = self.splits.get(iid, "train")
            all_anns = self.inst_by_image[iid]
            # Gather all defects for A overlap computation
            defect_anns_in_img = [r for r in all_anns if r["is_pseudo_normal"] != "True"]

            for ann in all_anns:
                lc = ann["label_code"]
                if lc == "G":
                    continue
                if self.smoke and total_processed >= self.smoke:
                    break

                if lc in ("B", "C", "D", "E", "F"):
                    self._process_defect_bf(ann, split)
                    total_processed += 1
                elif lc == "A":
                    self._process_defect_A(ann, split, defect_anns_in_img)
                    total_processed += 1
                elif lc in ("N1", "N2", "N3"):
                    self._process_normal(ann, split)
                    total_processed += 1
            if self.smoke and total_processed >= self.smoke:
                break

        # Write manifest
        _write_csv(self.output_root / "patch_manifest.csv", PATCH_MANIFEST_FIELDS, self.manifest_rows)
        logger.info("patch_manifest.csv → %d 条", len(self.manifest_rows))

        # Excluded
        if self.excluded_rows:
            _write_csv(self.output_root / ".." / ".." / "reports" / "patch_dataset" / "excluded_instances.csv",
                        ["ann_id", "image_id", "label_code", "reason", "note"], self.excluded_rows)

        # Summary
        self._generate_report()

        return len(self.manifest_rows)

    def _generate_report(self):
        out = self.output_root.parent.parent / "reports" / "patch_dataset"
        out.mkdir(parents=True, exist_ok=True)

        lines = ["# Patch 数据集报告", f"生成时间: {datetime.now().isoformat()}", ""]
        lines.append("## 各类 patch 数量")
        for lc in sorted(self.class_counts.keys()):
            d = self.class_counts[lc]
            lines.append(f"- {lc}: total={sum(d.values())} {dict(d)}")

        lines.append(f"\n## 总计: {len(self.manifest_rows)} patches")
        lines.append(f"\n## Excluded: {len(self.excluded_rows)} 实例")
        lines.append("\n## 策略分布")
        strat = defaultdict(int)
        for r in self.manifest_rows:
            strat[r["patch_strategy"]] += 1
        for k, v in sorted(strat.items()):
            lines.append(f"- {k}: {v}")

        (out / "patch_summary.md").write_text("\n".join(lines), encoding="utf-8")
        logger.info("报告: %s", out / "patch_summary.md")


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="目标感知 patch 数据集生成器")
    p.add_argument("--instances_csv", type=str, default="data/annotations/instances.csv")
    p.add_argument("--splits_csv", type=str, default="data/derived/splits.csv")
    p.add_argument("--images_dir", type=str, default="data/images")
    p.add_argument("--pn_overlap", type=str, default="reports/data_audit/pseudo_normal_overlap.csv")
    p.add_argument("--output_dir", type=str, default="data/derived")
    p.add_argument("--no-contact-sheets", action="store_true", default=True,
                   help="跳过 contact sheet 生成")
    p.add_argument("--smoke", type=int, default=0,
                   help="只生成前 N 个 patch 用于快速测试 (0=全部)")
    p.add_argument("--run_full", action="store_true", default=False,
                   help="生成全部 4005 个 patch (耗时较长)")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    builder = PatchBuilder(
        instances_csv=(root / args.instances_csv).resolve(),
        splits_csv=(root / args.splits_csv).resolve(),
        images_dir=(root / args.images_dir).resolve(),
        pn_overlap_csv=(root / args.pn_overlap).resolve(),
        output_root=(root / args.output_dir).resolve(),
        skip_contact=args.no_contact_sheets,
        smoke=args.smoke if not args.run_full else 0,
    )

    n = builder.run()
    logger.info("=" * 50)
    logger.info("完成! 生成 %d 个 patch", n)
    logger.info("输出: %s/patches/", args.output_dir)
    logger.info("Manifest: %s/patch_manifest.csv", args.output_dir)


if __name__ == "__main__":
    main()
