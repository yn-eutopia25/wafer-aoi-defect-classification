"""annotation_utils.py — 标注数据结构与 annotations.json 管理。

提供:
  - AnnotationManager: 标注数据 CRUD、持久化、进度管理
  - 从 images_manifest.csv 初始化 annotations.json
  - 实例增删改查、整图汇总自动计算
  - 导出 instances.csv / image_summary.csv
  - 标注数据校验
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .io_utils import load_yaml, setup_logger

logger = setup_logger("annotation_utils")


# =============================================================================
# Schema 常量 —— 从 label_schema.yaml 加载，同时提供硬编码回退
# =============================================================================
_DEFECT_SHORT_TO_LONG: Dict[str, str] = {}
_DEFECT_LONG_TO_SHORT: Dict[str, str] = {}
_DEFECT_LONG_NAMES: Dict[str, str] = {}
_DEFECT_CLASSES: List[str] = []
_NORMAL_CLASSES: Dict[str, str] = {}
_ALL_CLASSES: Set[str] = set()
_SHAPE_TYPES: List[str] = ["rectangle", "polygon"]
_SEVERITY_LEVELS: Dict[int, str] = {}
_REGION_TYPES: List[str] = ["surface", "edge", "pad", "background", "unknown"]
_REVIEW_STATUSES: List[str] = ["pending", "in_progress", "reviewed", "flagged"]


def _init_schema(project_root: Optional[Path] = None):
    """从 label_schema.yaml 加载类别定义（模块级缓存，一次加载）。"""
    global _DEFECT_SHORT_TO_LONG, _DEFECT_LONG_TO_SHORT, _DEFECT_LONG_NAMES
    global _DEFECT_CLASSES, _NORMAL_CLASSES, _ALL_CLASSES
    global _SEVERITY_LEVELS, _REGION_TYPES, _SHAPE_TYPES, _REVIEW_STATUSES

    if _DEFECT_CLASSES:
        return  # 已加载

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    schema_path = project_root / "configs" / "label_schema.yaml"
    if schema_path.exists():
        try:
            schema = load_yaml(schema_path)
            _DEFECT_SHORT_TO_LONG = schema.get("defect_short_to_long", {})
            _DEFECT_LONG_TO_SHORT = schema.get("defect_long_to_short", {})
            _DEFECT_LONG_NAMES = schema.get("defect_class_names", {})
            _DEFECT_CLASSES = schema.get("defect_classes", [])
            _NORMAL_CLASSES = schema.get("normal_classes", {})
            _SHAPE_TYPES = schema.get("shape_types", ["rectangle", "polygon"])
            _SEVERITY_LEVELS = {int(k): v for k, v in schema.get("severity_levels", {}).items()}
            _REGION_TYPES = schema.get("region_types", _REGION_TYPES)
            _REVIEW_STATUSES = schema.get("review_statuses", _REVIEW_STATUSES)
        except Exception as e:
            logger.warning("加载 label_schema.yaml 失败: %s，使用内置默认值。", e)

    # 确保有回退默认值
    if not _DEFECT_SHORT_TO_LONG:
        _DEFECT_SHORT_TO_LONG = {
            "A": "residue_cleaning", "B": "edge_glue", "C": "particle",
            "D": "pad_abnormal", "E": "surface_damage", "F": "uncertain_other",
        }
    if not _DEFECT_LONG_TO_SHORT:
        _DEFECT_LONG_TO_SHORT = {v: k for k, v in _DEFECT_SHORT_TO_LONG.items()}
    if not _DEFECT_CLASSES:
        _DEFECT_CLASSES = list(_DEFECT_SHORT_TO_LONG.values())
    if not _NORMAL_CLASSES:
        _NORMAL_CLASSES = {
            "pseudo_normal_surface": "局部正常芯片表面",
            "pseudo_normal_pad": "局部正常植球开口/pad区",
            "pseudo_normal_edge": "局部正常边缘区域",
        }
    if not _SEVERITY_LEVELS:
        _SEVERITY_LEVELS = {1: "mild", 2: "moderate", 3: "severe"}

    _ALL_CLASSES = set(_DEFECT_CLASSES) | set(_NORMAL_CLASSES.keys())


# =============================================================================
# 工具函数
# =============================================================================
def _now_iso() -> str:
    """返回当前 UTC ISO8601 时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _make_instance_template(
    image_id: str,
    annotation_id: str,
) -> Dict[str, Any]:
    """创建一个空的标注实例模板。"""
    return {
        "annotation_id": annotation_id,
        "image_id": image_id,
        "category": "",
        "category_short": "",
        "shape_type": "rectangle",
        "bbox": {"x_min": 0.0, "y_min": 0.0, "x_max": 0.0, "y_max": 0.0},
        "polygon_points": [],
        "severity": None,
        "region": "",
        "is_pseudo_normal": False,
        "note": "",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def _make_image_entry(
    image_id: str,
    filename: str,
) -> Dict[str, Any]:
    """创建一个空的图片条目模板。"""
    return {
        "image_id": image_id,
        "filename": filename,
        "review_status": "pending",
        "reviewer": "",
        "note": "",
        "instances": [],
    }


def _make_empty_annotations() -> Dict[str, Any]:
    """创建一个空的 annotations.json 顶层结构。"""
    return {
        "version": "2.0",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "images": {},
        "meta": {
            "total_images": 0,
            "total_reviewed": 0,
            "total_instances": 0,
            "total_defect_instances": 0,
            "total_normal_instances": 0,
            "category_counts": {},
            "last_modified": _now_iso(),
        },
        "progress": {
            "current_image_index": 0,
            "current_image_id": "",
            "reviewed_image_ids": [],
        },
    }


# =============================================================================
# AnnotationManager
# =============================================================================
class AnnotationManager:
    """annotations.json 的完整 CRUD 管理器。

    用法:
        mgr = AnnotationManager("data/annotations/annotations.json", project_root)
        mgr.init_from_manifest("data/metadata/images_manifest.csv")
        mgr.add_instance("AOI_NG_0001", category="particle", bbox={...})
        mgr.save()
        mgr.export_csv("data/annotations/")
    """

    def __init__(self, json_path: Path, project_root: Optional[Path] = None):
        _init_schema(project_root)

        self.json_path = Path(json_path)
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent

        # 标注计数器（全局递增）
        self._next_ann_id: int = 1

        if self.json_path.exists():
            self.data = self._load()
            self._sync_counter()
        else:
            self.data = _make_empty_annotations()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        """从文件加载 annotations.json。"""
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("已加载 annotations.json (%d 张图片)。", len(data.get("images", {})))
        return data

    def save(self) -> None:
        """保存 annotations.json，自动更新时间戳和元数据。"""
        self._update_meta()
        self.data["updated_at"] = _now_iso()
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        logger.debug("annotations.json 已保存。")

    def _sync_counter(self):
        """同步全局 annotation_id 计数器。"""
        max_id = 0
        for img in self.data.get("images", {}).values():
            for inst in img.get("instances", []):
                aid = inst.get("annotation_id", "")
                if aid.startswith("ANN_"):
                    try:
                        num = int(aid[4:])
                        if num > max_id:
                            max_id = num
                    except ValueError:
                        pass
        self._next_ann_id = max_id + 1

    def _next_annotation_id(self) -> str:
        """分配下一个 annotation_id。"""
        aid = f"ANN_{self._next_ann_id:05d}"
        self._next_ann_id += 1
        return aid

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def init_from_manifest(self, manifest_path: Path) -> int:
        """从 images_manifest.csv 初始化 images 条目（仅添加尚不存在的 image_id）。

        Returns:
            新添加的 image_id 数量。
        """
        if not manifest_path.exists():
            logger.error("Manifest 不存在: %s", manifest_path)
            return 0

        with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        images = self.data.setdefault("images", {})
        added = 0
        for row in rows:
            image_id = row.get("image_id", "")
            filename = row.get("filename", "")
            if image_id and image_id not in images:
                images[image_id] = _make_image_entry(image_id, filename)
                added += 1

        # 设置进度初始值
        progress = self.data.setdefault("progress", {})
        if not progress.get("current_image_id"):
            image_ids = self.sorted_image_ids()
            if image_ids:
                progress["current_image_id"] = image_ids[0]
                progress["current_image_index"] = 0

        self.data["meta"]["total_images"] = len(images)
        self.save()
        logger.info("从 manifest 初始化了 %d 个新 image_id（共 %d 张）。", added, len(images))
        return added

    # ------------------------------------------------------------------
    # 图片级操作
    # ------------------------------------------------------------------
    def get_image(self, image_id: str) -> Optional[Dict[str, Any]]:
        """获取某个 image_id 的完整条目。"""
        return self.data.get("images", {}).get(image_id)

    def get_or_create_image(self, image_id: str, filename: str = "") -> Dict[str, Any]:
        """获取或创建图片条目。"""
        images = self.data.setdefault("images", {})
        if image_id not in images:
            images[image_id] = _make_image_entry(image_id, filename)
        return images[image_id]

    def set_image_review_status(self, image_id: str, status: str):
        """更新图片审核状态。"""
        img = self.get_image(image_id)
        if img:
            if status in _REVIEW_STATUSES:
                img["review_status"] = status
            else:
                logger.warning("无效 review_status: %s", status)
            self.save()

    def set_image_note(self, image_id: str, note: str):
        """更新图片备注。"""
        img = self.get_image(image_id)
        if img:
            img["note"] = note
            self.save()

    def sorted_image_ids(self) -> List[str]:
        """返回按 image_id 自然排序的图片 ID 列表。"""
        ids = list(self.data.get("images", {}).keys())
        ids.sort(key=_natural_sort_key)
        return ids

    # ------------------------------------------------------------------
    # 实例 CRUD
    # ------------------------------------------------------------------
    def add_instance(
        self,
        image_id: str,
        category: str = "",
        category_short: str = "",
        shape_type: str = "rectangle",
        bbox: Optional[Dict[str, float]] = None,
        polygon_points: Optional[List[List[float]]] = None,
        severity: Optional[int] = None,
        region: str = "",
        is_pseudo_normal: bool = False,
        note: str = "",
    ) -> str:
        """为一个 image_id 添加标注实例。返回 annotation_id。"""
        img = self.get_or_create_image(image_id)

        # 自动推导 short/long category
        if category and not category_short:
            category_short = _DEFECT_LONG_TO_SHORT.get(category, "")
        elif category_short and not category:
            category = _DEFECT_SHORT_TO_LONG.get(category_short.upper(), category_short)

        ann_id = self._next_annotation_id()
        inst = _make_instance_template(image_id, ann_id)
        inst.update({
            "category": category,
            "category_short": category_short,
            "shape_type": shape_type if shape_type in _SHAPE_TYPES else "rectangle",
            "bbox": bbox or {"x_min": 0.0, "y_min": 0.0, "x_max": 0.0, "y_max": 0.0},
            "polygon_points": polygon_points or [],
            "severity": severity if severity in _SEVERITY_LEVELS else None,
            "region": region if region in _REGION_TYPES else "",
            "is_pseudo_normal": bool(is_pseudo_normal),
            "note": note,
        })

        img.setdefault("instances", []).append(inst)
        # 首次添加实例时更新审核状态
        if img.get("review_status") == "pending":
            img["review_status"] = "in_progress"
        self.save()
        logger.info("添加实例 %s → %s", ann_id, image_id)
        return ann_id

    def update_instance(
        self,
        image_id: str,
        annotation_id: str,
        **kwargs,
    ) -> bool:
        """更新一个标注实例的字段。返回是否成功。"""
        img = self.get_image(image_id)
        if not img:
            return False

        for inst in img.get("instances", []):
            if inst["annotation_id"] == annotation_id:
                for key, value in kwargs.items():
                    if key in inst:
                        inst[key] = value
                inst["updated_at"] = _now_iso()
                self.save()
                return True
        return False

    def delete_instance(self, image_id: str, annotation_id: str) -> bool:
        """删除一个标注实例。返回是否成功。"""
        img = self.get_image(image_id)
        if not img:
            return False

        instances = img.get("instances", [])
        for i, inst in enumerate(instances):
            if inst["annotation_id"] == annotation_id:
                instances.pop(i)
                # 如果该图无剩余实例，回到 pending
                if not instances:
                    img["review_status"] = "pending"
                self.save()
                logger.info("删除实例 %s", annotation_id)
                return True
        return False

    def get_instances(self, image_id: str) -> List[Dict[str, Any]]:
        """获取某图片的所有标注实例。"""
        img = self.get_image(image_id)
        return img.get("instances", []) if img else []

    def clear_instances(self, image_id: str) -> int:
        """清空某图片的所有实例。返回删除数量。"""
        img = self.get_image(image_id)
        if img:
            count = len(img.get("instances", []))
            img["instances"] = []
            img["review_status"] = "pending"
            self.save()
            return count
        return 0

    # ------------------------------------------------------------------
    # 进度
    # ------------------------------------------------------------------
    def set_progress(self, image_id: str, index: int):
        """更新当前进度。"""
        progress = self.data.setdefault("progress", {})
        progress["current_image_id"] = image_id
        progress["current_image_index"] = index
        # 标记当前为 reviewed
        reviewed = set(progress.get("reviewed_image_ids", []))
        reviewed.add(image_id)
        progress["reviewed_image_ids"] = sorted(reviewed, key=_natural_sort_key)
        self.save()

    def get_progress(self) -> Dict[str, Any]:
        """获取当前进度信息。"""
        return self.data.get("progress", {
            "current_image_index": 0,
            "current_image_id": "",
            "reviewed_image_ids": [],
        })

    # ------------------------------------------------------------------
    # 元数据更新
    # ------------------------------------------------------------------
    def _update_meta(self):
        """根据当前 images 数据重新计算 meta 统计。"""
        images = self.data.get("images", {})
        total = len(images)
        reviewed = sum(
            1 for img in images.values()
            if img.get("review_status") in ("reviewed", "flagged")
        )
        total_instances = 0
        total_defect = 0
        total_normal = 0
        cat_counter: Counter = Counter()

        for img in images.values():
            for inst in img.get("instances", []):
                total_instances += 1
                if inst.get("is_pseudo_normal"):
                    total_normal += 1
                else:
                    total_defect += 1
                cat = inst.get("category", "")
                if cat:
                    cat_counter[cat] += 1

        self.data["meta"] = {
            "total_images": total,
            "total_reviewed": reviewed,
            "total_instances": total_instances,
            "total_defect_instances": total_defect,
            "total_normal_instances": total_normal,
            "category_counts": dict(cat_counter),
            "last_modified": _now_iso(),
        }

    def get_meta(self) -> Dict[str, Any]:
        """获取当前元数据统计。"""
        self._update_meta()
        return self.data.get("meta", {})

    # ------------------------------------------------------------------
    # 导出 CSV
    # ------------------------------------------------------------------
    def export_instances_csv(self, output_path: Path) -> int:
        """将所有标注实例导出为 instances.csv。返回导出行数。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "annotation_id", "image_id", "category", "category_short",
            "shape_type", "x_min", "y_min", "x_max", "y_max",
            "polygon_points", "severity", "region", "is_pseudo_normal",
            "note", "created_at", "updated_at",
        ]

        rows = []
        for img in self.data.get("images", {}).values():
            for inst in img.get("instances", []):
                bbox = inst.get("bbox", {})
                rows.append({
                    "annotation_id": inst.get("annotation_id", ""),
                    "image_id": inst.get("image_id", ""),
                    "category": inst.get("category", ""),
                    "category_short": inst.get("category_short", ""),
                    "shape_type": inst.get("shape_type", ""),
                    "x_min": bbox.get("x_min", ""),
                    "y_min": bbox.get("y_min", ""),
                    "x_max": bbox.get("x_max", ""),
                    "y_max": bbox.get("y_max", ""),
                    "polygon_points": json.dumps(inst.get("polygon_points", [])),
                    "severity": inst.get("severity", ""),
                    "region": inst.get("region", ""),
                    "is_pseudo_normal": inst.get("is_pseudo_normal", False),
                    "note": inst.get("note", ""),
                    "created_at": inst.get("created_at", ""),
                    "updated_at": inst.get("updated_at", ""),
                })

        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("instances.csv 已导出: %s (%d 条)", output_path, len(rows))
        return len(rows)

    def export_image_summary_csv(self, output_path: Path) -> int:
        """导出 image_summary.csv —— 由实例自动汇总整图标签。返回导出行数。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "image_id", "filename", "review_status", "reviewer",
            "num_instances", "num_defect_instances", "num_normal_instances",
            "main_label", "secondary_labels", "category_counts",
            "has_multiple_defects", "note",
        ]

        rows = []
        for image_id in self.sorted_image_ids():
            img = self.get_image(image_id)
            if not img:
                continue
            instances = img.get("instances", [])
            num_instances = len(instances)
            num_defect = sum(1 for i in instances if not i.get("is_pseudo_normal"))
            num_normal = sum(1 for i in instances if i.get("is_pseudo_normal"))

            # 按出现次数统计缺陷类别
            defect_cats = [
                i["category"] for i in instances
                if i.get("category") and not i.get("is_pseudo_normal")
            ]
            cat_counter = Counter(defect_cats)
            cat_counts_json = json.dumps(dict(cat_counter), ensure_ascii=False)

            # 主类别 = 出现最多的
            if cat_counter:
                main_label = cat_counter.most_common(1)[0][0]
                secondary = [c for c in cat_counter if c != main_label]
                has_multiple = 1 if len(secondary) > 0 else 0
            else:
                main_label = ""
                secondary = []
                has_multiple = 0

            rows.append({
                "image_id": image_id,
                "filename": img.get("filename", ""),
                "review_status": img.get("review_status", ""),
                "reviewer": img.get("reviewer", ""),
                "num_instances": num_instances,
                "num_defect_instances": num_defect,
                "num_normal_instances": num_normal,
                "main_label": main_label,
                "secondary_labels": ";".join(secondary),
                "category_counts": cat_counts_json,
                "has_multiple_defects": has_multiple,
                "note": img.get("note", ""),
            })

        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("image_summary.csv 已导出: %s (%d 行)", output_path, len(rows))
        return len(rows)

    def export_all_csv(self, output_dir: Path):
        """一次性导出 instances.csv 和 image_summary.csv。"""
        self.export_instances_csv(output_dir / "instances.csv")
        self.export_image_summary_csv(output_dir / "image_summary.csv")

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def validate(self) -> Tuple[int, List[str]]:
        """校验 annotations.json 数据完整性。返回 (错误数, 错误消息列表)。"""
        errors: List[str] = []
        images = self.data.get("images", {})

        for image_id, img in images.items():
            prefix = f"[{image_id}]"

            # 必填字段
            if not img.get("filename"):
                errors.append(f"{prefix}: filename 为空")

            status = img.get("review_status", "")
            if status and status not in _REVIEW_STATUSES:
                errors.append(f"{prefix}: 无效 review_status='{status}'")

            for inst in img.get("instances", []):
                aid = inst.get("annotation_id", "?")
                iprefix = f"{prefix}/{aid}"

                cat = inst.get("category", "")
                if cat and cat not in _ALL_CLASSES:
                    errors.append(f"{iprefix}: 无效 category='{cat}'")

                shape = inst.get("shape_type", "")
                if shape and shape not in _SHAPE_TYPES:
                    errors.append(f"{iprefix}: 无效 shape_type='{shape}'")

                bbox = inst.get("bbox", {})
                if bbox:
                    xmin = bbox.get("x_min", 0)
                    ymin = bbox.get("y_min", 0)
                    xmax = bbox.get("x_max", 0)
                    ymax = bbox.get("y_max", 0)
                    for name, v in [("x_min", xmin), ("y_min", ymin), ("x_max", xmax), ("y_max", ymax)]:
                        if v < 0.0 or v > 1.0:
                            errors.append(f"{iprefix}: {name}={v} 超出 [0,1]")
                    if xmin >= xmax:
                        errors.append(f"{iprefix}: x_min >= x_max")
                    if ymin >= ymax:
                        errors.append(f"{iprefix}: y_min >= y_max")

                sev = inst.get("severity")
                if sev is not None and sev not in _SEVERITY_LEVELS:
                    errors.append(f"{iprefix}: 无效 severity={sev}")

                region = inst.get("region", "")
                if region and region not in _REGION_TYPES:
                    errors.append(f"{iprefix}: 无效 region='{region}'")

        logger.info("校验完成: %d 个错误。", len(errors))
        return len(errors), errors

    # ------------------------------------------------------------------
    # 统计查询
    # ------------------------------------------------------------------
    def get_summary(self) -> Dict[str, Any]:
        """获取标注概览摘要。"""
        meta = self.get_meta()
        progress = self.get_progress()
        return {
            "meta": meta,
            "progress": {
                "current": progress.get("current_image_id", ""),
                "reviewed_count": len(progress.get("reviewed_image_ids", [])),
            },
        }

    def get_all_instances_flat(self) -> List[Dict[str, Any]]:
        """返回所有实例的扁平列表。"""
        result = []
        for img in self.data.get("images", {}).values():
            result.extend(img.get("instances", []))
        return result


# =============================================================================
# 辅助
# =============================================================================
def _natural_sort_key(s: str) -> Tuple:
    """自然排序键。"""
    import re
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s))


# =============================================================================
# 便捷工厂函数
# =============================================================================
def create_annotation_manager(
    json_path: str = "data/annotations/annotations.json",
    manifest_path: str = "data/metadata/images_manifest.csv",
    project_root: Optional[Path] = None,
) -> AnnotationManager:
    """一站式工厂：创建 AnnotationManager 并从 manifest 初始化。

    Args:
        json_path: annotations.json 路径（相对于 project_root）
        manifest_path: images_manifest.csv 路径（相对于 project_root）
        project_root: 项目根目录，默认自动检测

    Returns:
        已初始化的 AnnotationManager 实例
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    json_full = project_root / json_path
    manifest_full = project_root / manifest_path

    mgr = AnnotationManager(json_full, project_root)

    if manifest_full.exists():
        mgr.init_from_manifest(manifest_full)
    else:
        logger.warning("Manifest 不存在: %s，跳过初始化。", manifest_full)

    return mgr
