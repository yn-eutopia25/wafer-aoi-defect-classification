#!/usr/bin/env python3
"""
03_make_contact_sheets.py — 生成缩略图 contact sheet

从 data/images/ 读取已重命名的图片，生成：
  1. 单张缩略图 (缓存到 reports/thumbnails/)
  2. 整体 contact sheet 大图 (保存到 reports/contact_sheets/)

用法:
    python scripts/03_make_contact_sheets.py
    python scripts/03_make_contact_sheets.py --images-dir data/images --output-dir reports/contact_sheets --grid-cols 10 --thumb-size 200
"""

import argparse
import sys
from pathlib import Path

# 确保 src/ 在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aoi_defect.image_utils import collect_images, build_contact_sheet
from aoi_defect.io_utils import get_project_root, setup_logger

logger = setup_logger("contact_sheets")


def make_contact_sheets(
    images_dir: Path,
    output_dir: Path,
    thumb_size: int = 200,
    grid_cols: int = 10,
):
    """主入口：收集图片 → 生成缩略图 → 拼接 contact sheet。"""
    if not images_dir.exists():
        logger.error("图片目录不存在: %s", images_dir)
        logger.error("请先运行 scripts/02_rename_images.py")
        sys.exit(1)

    image_files = collect_images(images_dir)
    if not image_files:
        logger.error("在 %s 中未找到图片。", images_dir)
        sys.exit(1)

    logger.info("找到 %d 张图片。", len(image_files))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "contact_sheet.jpg"

    build_contact_sheet(
        image_paths=image_files,
        output_path=output_path,
        thumb_size=thumb_size,
        grid_cols=grid_cols,
        thumb_dir=output_dir / "thumbnails",
        add_numbering=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="生成缩略图 contact sheet")
    parser.add_argument("--images-dir", type=str, default="data/images")
    parser.add_argument("--output-dir", type=str, default="reports/contact_sheets")
    parser.add_argument("--thumb-size", type=int, default=200)
    parser.add_argument("--grid-cols", type=int, default=10)
    args = parser.parse_args()

    project_root = get_project_root()
    images_dir = (project_root / args.images_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    logger.info("图片目录: %s", images_dir)
    logger.info("输出目录: %s", output_dir)

    make_contact_sheets(images_dir, output_dir, args.thumb_size, args.grid_cols)


if __name__ == "__main__":
    main()
