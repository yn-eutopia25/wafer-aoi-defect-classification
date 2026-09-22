#!/usr/bin/env python3
"""
00_create_label_templates.py — 生成标注 CSV 模板

根据 configs/label_schema.yaml 定义的字段和类别，生成：
  1. data/labels/image_labels.csv — 整图分类标签表（需先有 manifest）
  2. data/labels/bbox_labels.csv  — 局部 bbox 标注表（空模板）

用法:
    python scripts/00_create_label_templates.py
    python scripts/00_create_label_templates.py --manifest data/metadata/image_manifest.csv
"""

import argparse
import sys
from pathlib import Path

# 确保 src/ 在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aoi_defect.io_utils import (
    ensure_dir,
    get_project_root,
    read_csv_skip_comments,
    setup_logger,
    write_csv_with_comments,
)
from aoi_defect.label_utils import load_schema

logger = setup_logger("create_templates")


def build_image_labels_template(manifest_path: Path, output_path: Path, schema):
    """根据 manifest 生成整图分类标签模板。"""
    if not manifest_path.exists():
        logger.error("Manifest 不存在: %s", manifest_path)
        logger.error("请先运行 scripts/01_build_manifest.py")
        sys.exit(1)

    rows = read_csv_skip_comments(manifest_path)
    logger.info("从 manifest 读取到 %d 条记录。", len(rows))

    # 构建模板行
    template_rows = []
    for row in rows:
        fid = row.get("new_filename", row.get("filename", "")).strip()
        if not fid:
            continue
        template_rows.append({
            "image_id": fid,
            "original_filename": row.get("original_filename", ""),
            "filename": fid,
            "main_label": "",
            "secondary_labels": "",
            "severity": "",
            "has_multiple_defects": "",
            "review_status": "pending",
            "reviewer": "",
            "note": "",
        })

    # 注释头
    comments = [
        "=== 整图分类标签表 (image_labels.csv) ===",
        "标注说明:",
        "  - image_id: 图片唯一 ID（对应 data/images/ 下的文件）",
        "  - main_label: 主缺陷类别，从以下选项中选择:",
    ]
    for c in schema.defect_classes:
        comments.append(f"      {c}")
    comments += [
        "  - secondary_labels: 次要标签，多个用分号(;)分隔",
        "  - severity: none / mild / moderate / severe",
        "  - has_multiple_defects: 0=否, 1=是",
        "  - review_status: pending / reviewed / confirmed / flagged",
        "  - 每张图一行。main_label 为空表示待标注。",
        "==============================================",
    ]

    example = {
        "image_id": "aoi_ng_00001.jpg",
        "original_filename": "XXXX.jpg",
        "filename": "aoi_ng_00001.jpg",
        "main_label": "particle",
        "secondary_labels": "edge_glue",
        "severity": "moderate",
        "has_multiple_defects": "1",
        "review_status": "reviewed",
        "reviewer": "your_name",
        "note": "底部有颗粒，同时边缘有溢胶",
    }

    write_csv_with_comments(
        output_path,
        fieldnames=schema.field_names_image(),
        rows=template_rows,
        comments=comments,
        example_row=example,
    )
    logger.info("整图分类标签模板已保存: %s (%d 行)", output_path, len(template_rows))


def build_bbox_labels_template(output_path: Path, schema):
    """生成 bbox 标注空模板（不含数据行，标注者按需添加）。"""
    comments = [
        "=== 局部 bbox 标注表 (bbox_labels.csv) ===",
        "标注说明:",
        "  - image_id: 对应图片 ID",
        "  - label: 缺陷或正常区域类别",
        "    缺陷类: " + ", ".join(schema.defect_classes),
        "    正常类: " + ", ".join(schema.normal_classes),
        "  - x_min, y_min, x_max, y_max: 归一化坐标 (0.0~1.0)",
        "  - region: 区域位置描述 (如 chip_center, edge_top)",
        "  - is_pseudo_normal: 0=缺陷框, 1=局部正常框",
        "  - quality: good / blur / low_contrast / overexposed",
        "  - 每张图可有多行。",
        "==============================================",
    ]

    example = {
        "image_id": "aoi_ng_00001.jpg",
        "label": "particle",
        "x_min": "0.12",
        "y_min": "0.34",
        "x_max": "0.56",
        "y_max": "0.78",
        "region": "chip_center",
        "is_pseudo_normal": "0",
        "quality": "good",
        "note": "中心区域黑色颗粒",
    }

    # 空行（标注者按需添加）
    empty_rows: list = []

    write_csv_with_comments(
        output_path,
        fieldnames=schema.field_names_bbox(),
        rows=empty_rows,
        comments=comments,
        example_row=example,
    )
    logger.info("Bbox 标注模板已保存: %s (0 行，请按需添加)", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="生成标注 CSV 模板")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/metadata/image_manifest.csv",
        help="Manifest CSV 路径 (默认: data/metadata/image_manifest.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/labels",
        help="标签输出目录 (默认: data/labels)",
    )
    args = parser.parse_args()

    project_root = get_project_root()
    manifest_path = (project_root / args.manifest).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    ensure_dir(output_dir)

    logger.info("项目根目录: %s", project_root)

    schema = load_schema(project_root)
    logger.info("已加载标签 schema: %d 缺陷类 + %d 正常类",
                 len(schema.defect_classes), len(schema.normal_classes))

    # 1. 整图分类标签
    build_image_labels_template(manifest_path, output_dir / "image_labels.csv", schema)

    # 2. bbox 标注模板
    build_bbox_labels_template(output_dir / "bbox_labels.csv", schema)


if __name__ == "__main__":
    main()
