"""label_utils.py — 标签体系与校验工具。

提供:
  - 加载 label_schema.yaml
  - 获取合法类别集合
  - 整图标签 & bbox 标签校验
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .io_utils import load_yaml, setup_logger

logger = setup_logger("label_utils")


# ---------------------------------------------------------------------------
# Schema 加载
# ---------------------------------------------------------------------------
class LabelSchema:
    """标签体系封装，从 configs/label_schema.yaml 加载。"""

    def __init__(self, yaml_path: Path):
        self.yaml_path = yaml_path
        schema = load_yaml(yaml_path)

        # 缺陷类别（兼容旧版列表 & 新版映射）
        self.defect_classes: List[str] = schema.get("defect_classes", [])
        self.defect_short_to_long: Dict[str, str] = schema.get("defect_short_to_long", {})
        self.defect_long_to_short: Dict[str, str] = schema.get("defect_long_to_short", {})
        self.defect_class_names: Dict[str, str] = schema.get("defect_class_names", {})

        # 正常类别
        raw_normals = schema.get("normal_classes", {})
        if isinstance(raw_normals, list):
            self.normal_classes: List[str] = raw_normals
        else:
            self.normal_classes: List[str] = list(raw_normals.keys())

        # 标注相关
        self.shape_types: List[str] = schema.get("shape_types", ["rectangle", "polygon"])
        self.severity_levels: Dict[int, str] = {
            int(k): v for k, v in schema.get("severity_levels", {}).items()
        }
        self.region_types: List[str] = schema.get("region_types", [])
        self.review_statuses: List[str] = schema.get("review_statuses", [])

        # CSV 导出字段
        self.instances_csv_fields: List[str] = schema.get("instances_csv_fields", [])
        self.image_summary_csv_fields: List[str] = schema.get("image_summary_csv_fields", [])

        # 兼容旧版字段
        self.image_level_fields: List[str] = schema.get("image_level_fields", [])
        self.bbox_fields: List[str] = schema.get("bbox_fields", [])

        # 快速集合
        self._defect_set: Set[str] = set(self.defect_classes)
        self._normal_set: Set[str] = set(self.normal_classes)
        self._all_set: Set[str] = self._defect_set | self._normal_set

    @property
    def all_classes(self) -> List[str]:
        return self.defect_classes + self.normal_classes

    def is_valid_label(self, label: str) -> bool:
        return label in self._all_set

    def is_defect(self, label: str) -> bool:
        return label in self._defect_set

    def is_normal(self, label: str) -> bool:
        return label in self._normal_set

    def field_names_image(self) -> List[str]:
        return self.image_level_fields

    def field_names_bbox(self) -> List[str]:
        return self.bbox_fields


def load_schema(project_root: Path) -> LabelSchema:
    """从项目根目录加载标签 schema。"""
    schema_path = project_root / "configs" / "label_schema.yaml"
    if not schema_path.exists():
        raise FileNotFoundError(f"标签 schema 不存在: {schema_path}")
    return LabelSchema(schema_path)


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def validate_image_labels_row(
    row: Dict[str, str],
    schema: LabelSchema,
    valid_image_ids: Set[str],
    row_num: int,
) -> List[str]:
    """校验一行 image_labels 记录，返回错误消息列表。"""
    errors: List[str] = []
    prefix = f"image_labels 行 {row_num}"

    image_id = row.get("image_id", "").strip()

    if not image_id:
        errors.append(f"{prefix}: image_id 为空")
        return errors

    if valid_image_ids and image_id not in valid_image_ids:
        errors.append(f"{prefix}: image_id='{image_id}' 不在 manifest 中")

    main_label = row.get("main_label", "").strip()
    if main_label and not schema.is_valid_label(main_label):
        errors.append(
            f"{prefix}: 无效 main_label='{main_label}'，"
            f"合法值: {schema.all_classes}"
        )

    secondary = row.get("secondary_labels", "").strip()
    if secondary:
        for s in secondary.split(";"):
            s = s.strip()
            if s and not schema.is_valid_label(s):
                errors.append(
                    f"{prefix}: 无效 secondary_label='{s}'"
                )

    severity = row.get("severity", "").strip()
    if severity and severity not in {"none", "mild", "moderate", "severe"}:
        errors.append(f"{prefix}: severity 无效: '{severity}'")

    has_multi = row.get("has_multiple_defects", "").strip()
    if has_multi and has_multi not in {"0", "1"}:
        errors.append(f"{prefix}: has_multiple_defects 应为 0 或 1")

    review = row.get("review_status", "").strip()
    if review and review not in {"pending", "reviewed", "confirmed", "flagged"}:
        errors.append(f"{prefix}: review_status 无效: '{review}'")

    return errors


def validate_bbox_row(
    row: Dict[str, str],
    schema: LabelSchema,
    valid_image_ids: Set[str],
    row_num: int,
) -> List[str]:
    """校验一行 bbox_labels 记录，返回错误消息列表。"""
    errors: List[str] = []
    prefix = f"bbox_labels 行 {row_num}"

    image_id = row.get("image_id", "").strip()
    if not image_id:
        errors.append(f"{prefix}: image_id 为空")
        return errors

    if valid_image_ids and image_id not in valid_image_ids:
        errors.append(f"{prefix}: image_id='{image_id}' 不在 manifest 中")

    label = row.get("label", "").strip()
    if not label:
        errors.append(f"{prefix}: label 为空")
    elif not schema.is_valid_label(label):
        errors.append(
            f"{prefix}: 无效 label='{label}'，合法值: {schema.all_classes}"
        )

    # 坐标
    for field in ["x_min", "y_min", "x_max", "y_max"]:
        val = row.get(field, "").strip()
        if not val:
            errors.append(f"{prefix}: {field} 为空")
            continue
        try:
            fval = float(val)
            if fval < 0.0 or fval > 1.0:
                errors.append(f"{prefix}: {field}={fval} 超出 [0,1]")
        except ValueError:
            errors.append(f"{prefix}: {field} 不是有效数字: '{val}'")

    # x_min < x_max, y_min < y_max
    try:
        x_min = float(row.get("x_min", 0))
        x_max = float(row.get("x_max", 0))
        y_min = float(row.get("y_min", 0))
        y_max = float(row.get("y_max", 0))
        if x_min >= x_max:
            errors.append(f"{prefix}: x_min >= x_max")
        if y_min >= y_max:
            errors.append(f"{prefix}: y_min >= y_max")
    except ValueError:
        pass

    # is_pseudo_normal
    ipn = row.get("is_pseudo_normal", "").strip()
    if ipn and ipn not in {"0", "1"}:
        errors.append(f"{prefix}: is_pseudo_normal 应为 0 或 1")

    # quality
    quality = row.get("quality", "").strip()
    if quality and quality not in {"good", "blur", "low_contrast", "overexposed"}:
        errors.append(f"{prefix}: quality 无效: '{quality}'")

    return errors
