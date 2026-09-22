"""annotation_store.py — 标注数据持久化存储。

提供 AnnotationStore 类，管理 annotations.json 的完整生命周期：
  - 原子写入（临时文件 + rename）
  - 自动备份（带时间戳）
  - 图片记录管理
  - 标注实例 CRUD（rectangle / polygon）
  - 坐标自动推导（points ↔ bbox）、面积计算（shoelace）
  - 进度查询、自然排序

用法:
    from aoi_defect.annotation_store import AnnotationStore

    store = AnnotationStore("data/annotations/annotations.json", "data/annotations/backups")
    store.add_annotation("AOI_NG_0001", "AOI_NG_0001.jpg", {
        "label_code": "C",
        "geometry_type": "rectangle",
        "bbox": [120, 80, 300, 230],
        "severity": 2,
        "region": "surface",
    })
    store.save()
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def _setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger


logger = _setup_logger("annotation_store")


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "0.2"

VALID_DEFECT_SHORTS = {"A", "B", "C", "D", "E", "F", "G"}
VALID_NORMAL_SHORTS = {"N1", "N2", "N3"}
VALID_LABEL_CODES = VALID_DEFECT_SHORTS | VALID_NORMAL_SHORTS
VALID_GEOMETRY_TYPES = {"rectangle", "polygon"}
VALID_REGIONS = {"surface", "edge", "pad", "background", "unknown", ""}
VALID_SEVERITY = {1, 2, 3, None}
VALID_QUALITY = {"good", "uncertain", "bad", ""}
VALID_STATUSES = {"unstarted", "in_progress", "done", "skipped", "uncertain"}

LABEL_CODE_TO_NAME: Dict[str, str] = {
    "A": "residue_cleaning",
    "B": "edge_glue",
    "C": "particle",
    "D": "pad_abnormal",
    "E": "surface_damage",
    "F": "pi_broken",
    "G": "uncertain_other",
    "N1": "pseudo_normal_surface",
    "N2": "pseudo_normal_pad",
    "N3": "pseudo_normal_edge",
}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _natural_sort_key(s: str) -> Tuple:
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s))


# ---------------------------------------------------------------------------
# 坐标推导
# ---------------------------------------------------------------------------
def _rect_points_from_bbox(bbox: List[float]) -> List[List[int]]:
    """从 bbox [x_min, y_min, x_max, y_max] 推导四个角点（int）。"""
    x_min, y_min, x_max, y_max = bbox
    return [
        [int(x_min), int(y_min)],
        [int(x_max), int(y_min)],
        [int(x_max), int(y_max)],
        [int(x_min), int(y_max)],
    ]


def _bbox_from_points(points: List[List[float]]) -> List[int]:
    """从多边形顶点推导外接矩形 [x_min, y_min, x_max, y_max]（int）。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _shoelace_area(points: List[List[float]]) -> float:
    """Shoelace formula 计算多边形面积。"""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _compute_area(geometry_type: str, bbox: List[int],
                  points: Optional[List[List[int]]] = None) -> float:
    """计算标注面积（像素平方）。"""
    if geometry_type == "rectangle" and len(bbox) == 4:
        x_min, y_min, x_max, y_max = bbox
        return float(max(0, x_max - x_min) * max(0, y_max - y_min))
    if geometry_type == "polygon" and points and len(points) >= 3:
        return _shoelace_area(points)
    return 0.0


# =============================================================================
# AnnotationStore
# =============================================================================
class AnnotationStore:
    """标注数据持久化存储管理器。

    Parameters
    ----------
    annotations_path : str
        annotations.json 文件路径。
    backup_dir : str
        自动备份目录。

    Usage::

        store = AnnotationStore("data/annotations/annotations.json", "data/annotations/backups")
        aid = store.add_annotation("AOI_NG_0001", "AOI_NG_0001.jpg", {
            "label_code": "C",
            "geometry_type": "rectangle",
            "bbox": [120, 80, 300, 230],
        })
        store.save()
    """

    def __init__(
        self,
        annotations_path: str = "data/annotations/annotations.json",
        backup_dir: str = "data/annotations/backups",
        project_root: Optional[Path] = None,
    ):
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent.parent
        self._project_root = Path(project_root)

        self.json_path = (self._project_root / annotations_path).resolve()
        self.backups_dir = (self._project_root / backup_dir).resolve()

        self.data: Dict[str, Any] = self.load()

    # ------------------------------------------------------------------
    # 空结构
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_structure() -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "images": {},
        }

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def load(self) -> Dict[str, Any]:
        """从文件加载 JSON；文件不存在或损坏则创建空结构。"""
        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data.setdefault("images", {})
                data.setdefault("schema_version", SCHEMA_VERSION)
                data.setdefault("updated_at", _now_iso())
                logger.info("已加载 annotations.json (%d 张图片)。", len(data["images"]))
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.error("加载 annotations.json 失败: %s，使用空结构。", e)
                return self._empty_structure()
        else:
            logger.info("annotations.json 不存在，创建空结构。")
            return self._empty_structure()

    def save(self) -> None:
        """原子保存：备份 → 写临时文件 → replace。

        如果程序崩溃，旧文件或新文件至少有一个完整。
        """
        self.data["updated_at"] = _now_iso()

        # 1. 备份
        self._backup_current()

        # 2. 确保目录存在
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. 原子写入
        tmp_path = self.json_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            tmp_path.replace(self.json_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        logger.debug("annotations.json 已保存。")

    def _backup_current(self) -> Optional[Path]:
        """备份当前文件到 backups/。"""
        if not self.json_path.exists() or self.json_path.stat().st_size == 0:
            return None
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = self.backups_dir / f"annotations_{ts}.json"
        try:
            shutil.copy2(self.json_path, dst)
            logger.info("已备份: %s", dst.name)
            return dst
        except OSError as e:
            logger.warning("备份失败: %s", e)
            return None

    # ------------------------------------------------------------------
    # 图片记录
    # ------------------------------------------------------------------
    def _ensure_image(self, image_id: str, filename: str) -> Dict[str, Any]:
        """获取或创建图片记录（直接引用）。"""
        images: Dict[str, Any] = self.data.setdefault("images", {})

        if image_id not in images:
            images[image_id] = {
                "image_id": image_id,
                "filename": filename,
                "status": "unstarted",
                "image_note": "",
                "annotations": [],
            }
        else:
            rec = images[image_id]
            if not rec.get("filename"):
                rec["filename"] = filename
            rec.setdefault("image_note", "")

        return images[image_id]

    def get_image_record(self, image_id: str, filename: str = "") -> Dict[str, Any]:
        """获取图片记录（深拷贝）。"""
        return deepcopy(self._ensure_image(image_id, filename))

    def get_image_ids(self) -> List[str]:
        """返回所有 image_id（自然排序）。"""
        ids = list(self.data.get("images", {}).keys())
        ids.sort(key=_natural_sort_key)
        return ids

    # ------------------------------------------------------------------
    # 状态 & 备注
    # ------------------------------------------------------------------
    def update_image_status(self, image_id: str, status: str) -> bool:
        """更新图片 status。"""
        if status not in VALID_STATUSES:
            logger.warning("无效 status: '%s'", status)
            return False
        images = self.data.get("images", {})
        if image_id not in images:
            logger.warning("image_id 不存在: %s", image_id)
            return False
        images[image_id]["status"] = status
        return True

    def update_image_note(self, image_id: str, note: str) -> bool:
        """更新图片 image_note。"""
        images = self.data.get("images", {})
        if image_id not in images:
            logger.warning("image_id 不存在: %s", image_id)
            return False
        images[image_id]["image_note"] = note
        return True

    def get_done_image_ids(self) -> List[str]:
        """返回 status="done" 的 image_id。"""
        done = [iid for iid, rec in self.data.get("images", {}).items()
                if rec.get("status") == "done"]
        done.sort(key=_natural_sort_key)
        return done

    def get_first_unfinished_image_id(self, image_ids: List[str]) -> Optional[str]:
        """返回第一个 status != "done" 的 image_id。"""
        images = self.data.get("images", {})
        for iid in image_ids:
            rec = images.get(iid)
            if rec is None or rec.get("status") != "done":
                return iid
        return None

    # ------------------------------------------------------------------
    # 标注列表
    # ------------------------------------------------------------------
    def list_annotations(self, image_id: str) -> List[Dict[str, Any]]:
        """返回某图片的所有标注实例（深拷贝）。"""
        rec = self.data.get("images", {}).get(image_id)
        return deepcopy(rec.get("annotations", [])) if rec else []

    # ------------------------------------------------------------------
    # ann_id
    # ------------------------------------------------------------------
    def next_ann_id(self, image_id: str) -> str:
        """生成下一个 ann_id: AOI_NG_0001_ann_0001。"""
        rec = self.data.get("images", {}).get(image_id)
        anns = rec.get("annotations", []) if rec else []
        prefix = f"{image_id}_ann_"
        max_idx = 0
        for a in anns:
            aid = a.get("ann_id", "")
            if aid.startswith(prefix):
                try:
                    max_idx = max(max_idx, int(aid[len(prefix):]))
                except ValueError:
                    pass
        return f"{prefix}{max_idx + 1:04d}"

    # ------------------------------------------------------------------
    # 标注 CRUD
    # ------------------------------------------------------------------
    def add_annotation(
        self,
        image_id: str,
        filename: str,
        annotation_dict: Dict[str, Any],
    ) -> Optional[str]:
        """添加标注实例。

        annotation_dict 字段:
          - label_code : str (必填) — A–F 或 N1–N3
          - geometry_type : str — "rectangle" / "polygon"
          - bbox : [x_min, y_min, x_max, y_max] — int 像素坐标
          - points : [[x1,y1], ...] — polygon 顶点（≥3）
          - severity : int|None — 1/2/3
          - region : str — surface/edge/pad/background/unknown
          - quality : str — good/uncertain/bad
          - note : str

        Returns
        -------
        str or None — ann_id；失败返回 None。

        Notes
        -----
        - rectangle: 提供 bbox → 自动推导 points、area_px
        - polygon:   提供 points → 自动推导 bbox、area_px
        - 所有坐标转 int
        - N1/N2/N3 → is_pseudo_normal=true
        """
        # ---- 类型 ----
        if not isinstance(annotation_dict, dict):
            logger.error("annotation_dict 必须为 dict，收到 %s", type(annotation_dict).__name__)
            return None

        # ---- label_code ----
        label_code = str(annotation_dict.get("label_code", "")).strip()
        if label_code not in VALID_LABEL_CODES:
            logger.error("无效 label_code: '%s'，合法值: %s", label_code, VALID_LABEL_CODES)
            return None

        label = LABEL_CODE_TO_NAME.get(label_code, label_code)
        is_pseudo_normal = label_code in VALID_NORMAL_SHORTS

        # ---- geometry_type ----
        geometry_type = annotation_dict.get("geometry_type", "rectangle")
        if geometry_type not in VALID_GEOMETRY_TYPES:
            logger.error("无效 geometry_type: '%s'", geometry_type)
            return None

        # ---- 坐标 ----
        points: List[List[int]] = []
        bbox: List[int] = []

        if geometry_type == "rectangle":
            raw_bbox = annotation_dict.get("bbox")
            if not raw_bbox or len(raw_bbox) != 4:
                logger.error("rectangle 需要 bbox [x_min, y_min, x_max, y_max]")
                return None
            try:
                bbox = [int(round(float(v))) for v in raw_bbox]
            except (ValueError, TypeError) as e:
                logger.error("bbox 格式错误: %s", e)
                return None
            points = _rect_points_from_bbox(bbox)

        elif geometry_type == "polygon":
            raw_points = annotation_dict.get("points")
            if not raw_points or len(raw_points) < 3:
                logger.error("polygon 需要至少 3 个顶点")
                return None
            try:
                points = [[int(round(float(p[0]))), int(round(float(p[1])))]
                          for p in raw_points]
            except (ValueError, TypeError, IndexError) as e:
                logger.error("points 格式错误: %s", e)
                return None
            bbox = _bbox_from_points(points)

        # ---- 面积 ----
        area_px = _compute_area(geometry_type, bbox, points)

        # ---- 可选字段 ----
        severity = annotation_dict.get("severity")
        if severity is not None:
            try:
                severity = int(severity)
            except (ValueError, TypeError):
                severity = None
            if severity not in VALID_SEVERITY:
                logger.warning("severity=%s 非法，设为 None", severity)
                severity = None

        region = str(annotation_dict.get("region", ""))
        if region not in VALID_REGIONS:
            region = ""

        quality = str(annotation_dict.get("quality", "good"))
        if quality not in VALID_QUALITY:
            quality = "good"

        note = str(annotation_dict.get("note", ""))

        # ---- 构建 ----
        ann_id = self.next_ann_id(image_id)
        now = _now_iso()

        instance: Dict[str, Any] = {
            "ann_id": ann_id,
            "label_code": label_code,
            "label": label,
            "geometry_type": geometry_type,
            "points": points,
            "bbox": bbox,
            "area_px": area_px,
            "severity": severity,
            "region": region,
            "quality": quality,
            "is_pseudo_normal": is_pseudo_normal,
            "note": note,
            "created_at": now,
            "updated_at": now,
        }

        # ---- 写入 ----
        img = self._ensure_image(image_id, filename)
        # 如果该图片之前已标记为 done，此次是重新标注，先清空旧标注
        if img.get("status") == "done":
            img["annotations"] = []
            logger.info("图片 %s 标记为 done 但被重新标注，已清空旧标注。", image_id)
        img.setdefault("annotations", []).append(instance)
        if img.get("status") == "unstarted":
            img["status"] = "in_progress"

        logger.info("添加 %s → %s (code=%s, bbox=%s, area=%.0f px²)",
                     ann_id, image_id, label_code, bbox, area_px)
        return ann_id

    def delete_annotation(self, image_id: str, ann_id: str) -> bool:
        """删除一个标注实例。"""
        images = self.data.get("images", {})
        if image_id not in images:
            logger.warning("image_id 不存在: %s", image_id)
            return False

        anns = images[image_id].get("annotations", [])
        for i, a in enumerate(anns):
            if a.get("ann_id") == ann_id:
                anns.pop(i)
                if not anns:
                    images[image_id]["status"] = "unstarted"
                logger.info("删除 %s (image=%s)", ann_id, image_id)
                return True

        logger.warning("ann_id 不存在: %s", ann_id)
        return False

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        """标注统计概览。"""
        images = self.data.get("images", {})
        total = len(images)
        done = len(self.get_done_image_ids())
        n_inst = 0
        cats: Dict[str, int] = {}
        statuses: Dict[str, int] = {}

        for rec in images.values():
            for a in rec.get("annotations", []):
                n_inst += 1
                lc = a.get("label_code", "")
                if lc:
                    cats[lc] = cats.get(lc, 0) + 1
            st = rec.get("status", "unstarted")
            statuses[st] = statuses.get(st, 0) + 1

        return {
            "total_images": total,
            "total_done": done,
            "total_instances": n_inst,
            "category_counts": cats,
            "status_counts": statuses,
            "updated_at": self.data.get("updated_at", ""),
        }
