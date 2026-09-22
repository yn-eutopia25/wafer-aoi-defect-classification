#!/usr/bin/env python3
"""
04_validate_labels.py — 标签合法性校验

校验 image_labels.csv 和 bbox_labels.csv:
  - 列名是否与 label_schema.yaml 一致
  - 必填字段非空
  - 类别值在合法集合内
  - bbox 坐标在 [0,1] 范围，且 x_min < x_max, y_min < y_max
  - image_id 是否在 manifest 中存在

用法:
    python scripts/04_validate_labels.py
    python scripts/04_validate_labels.py --image-labels data/labels/image_labels.csv --bbox-labels data/labels/bbox_labels.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Set

# 确保 src/ 在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aoi_defect.io_utils import (
    get_project_root,
    read_csv_skip_comments,
    setup_logger,
)
from aoi_defect.label_utils import (
    LabelSchema,
    load_schema,
    validate_image_labels_row,
    validate_bbox_row,
)

logger = setup_logger("validate_labels")


def load_valid_ids(manifest_path: Path) -> Set[str]:
    """从 manifest 加载合法 image_id 集合。"""
    if not manifest_path.exists():
        logger.warning("Manifest 不存在: %s，跳过 ID 校验。", manifest_path)
        return set()

    ids: Set[str] = set()
    for row in read_csv_skip_comments(manifest_path):
        fid = row.get("new_filename", row.get("filename", "")).strip()
        if fid:
            ids.add(fid)

    logger.info("从 manifest 加载 %d 个合法 ID。", len(ids))
    return ids


def validate_file(
    path: Path,
    schema: LabelSchema,
    valid_ids: Set[str],
    expected_fields: list,
    label: str,
    row_validator,
) -> int:
    """通用校验函数。返回错误数。"""
    logger.info("--- 校验 %s: %s ---", label, path)

    if not path.exists():
        logger.error("文件不存在: %s", path)
        return 1

    rows = read_csv_skip_comments(path)
    if not rows:
        logger.info("文件为空（未添加数据），视为合法。")
        return 0

    errors = 0
    logger.info("共 %d 行数据。", len(rows))

    # 检查列名
    actual_cols = set(rows[0].keys())
    expected_cols = set(expected_fields)
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols
    if missing:
        logger.error("缺少列: %s", missing)
        errors += 1
    if extra:
        logger.warning("多余列 (无害): %s", extra)

    for i, row in enumerate(rows, start=1):
        row_errors = row_validator(row, schema, valid_ids, i + 1)  # +1 for header
        for e in row_errors:
            logger.error("  %s", e)
        errors += len(row_errors)

    if errors == 0:
        logger.info("✓ %s 校验通过。", label)
    else:
        logger.error("✗ %s 发现 %d 个错误。", label, errors)

    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="校验标注 CSV 文件")
    parser.add_argument("--image-labels", type=str, default="data/labels/image_labels.csv")
    parser.add_argument("--bbox-labels", type=str, default="data/labels/bbox_labels.csv")
    parser.add_argument("--manifest", type=str, default="data/metadata/image_manifest.csv")
    args = parser.parse_args()

    project_root = get_project_root()
    manifest_path = (project_root / args.manifest).resolve()
    image_labels_path = (project_root / args.image_labels).resolve()
    bbox_labels_path = (project_root / args.bbox_labels).resolve()

    logger.info("项目根目录: %s", project_root)

    schema = load_schema(project_root)
    valid_ids = load_valid_ids(manifest_path)

    total_errors = 0

    # 校验 image_labels.csv
    total_errors += validate_file(
        path=image_labels_path,
        schema=schema,
        valid_ids=valid_ids,
        expected_fields=schema.field_names_image(),
        label="image_labels",
        row_validator=validate_image_labels_row,
    )

    # 校验 bbox_labels.csv
    total_errors += validate_file(
        path=bbox_labels_path,
        schema=schema,
        valid_ids=valid_ids,
        expected_fields=schema.field_names_bbox(),
        label="bbox_labels",
        row_validator=validate_bbox_row,
    )

    logger.info("=" * 50)
    if total_errors == 0:
        logger.info("✓ 所有标签校验通过！")
        sys.exit(0)
    else:
        logger.error("✗ 共发现 %d 个错误，请修正后重新校验。", total_errors)
        sys.exit(1)


if __name__ == "__main__":
    main()
