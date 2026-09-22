#!/usr/bin/env python3
"""
01_build_manifest.py — 扫描原始图片，生成 raw_manifest.csv

遍历 data/raw_original/ 下所有图片文件（.jpg .jpeg .png .bmp .tif .tiff），
提取元数据并写入 raw_manifest.csv。不修改任何原始图片。

输出字段:
  raw_index            从 1 开始的序号
  original_filename    原始文件名
  original_relpath     相对路径
  width                图片宽度 (px)
  height               图片高度 (px)
  mode                 PIL 图像模式 (RGB / L / ...)
  file_size_bytes      文件大小 (字节)
  md5                  文件 MD5 哈希
  suggested_image_id   建议新 ID，格式 AOI_NG_0001
  suggested_filename   建议新文件名，如 AOI_NG_0001.jpg
  parse_wafer_id       文件名中解析出的 wafer/WFR 字段，否则空
  parse_datetime       文件名中解析出的 14 位时间戳 (yyyyMMddHHmmss)，否则空
  note                 备注（损坏图片写 error 信息）

用法:
    python scripts/01_build_manifest.py
    python scripts/01_build_manifest.py --input_dir data/raw_original --output_csv data/metadata/raw_manifest.csv
"""

import argparse
import csv
import hashlib
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# 确保 src/ 可导入
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def setup_logger(name: str) -> logging.Logger:
    """创建带时间戳的 logger。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logger("build_manifest")


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ID_PREFIX = "AOI_NG_"
ID_ZFILL = 4

CSV_FIELDNAMES = [
    "raw_index",
    "original_filename",
    "original_relpath",
    "width",
    "height",
    "mode",
    "file_size_bytes",
    "md5",
    "suggested_image_id",
    "suggested_filename",
    "parse_wafer_id",
    "parse_datetime",
    "note",
]


# ---------------------------------------------------------------------------
# 文件收集（稳定排序）
# ---------------------------------------------------------------------------
def _natural_key(s: str) -> Tuple:
    """自然排序键：数字部分按数值比较，如 W1 < W2 < W10。"""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", s)
    )


def collect_images(root: Path) -> List[Path]:
    """收集目录下所有支持的图片文件，按文件名自然排序。"""
    files: List[Path] = []
    for ext in SUPPORTED_EXTS:
        files.extend(root.glob(f"*{ext}"))
        files.extend(root.glob(f"*{ext.upper()}"))
    files = sorted(set(files), key=lambda p: _natural_key(p.name))
    return files


# ---------------------------------------------------------------------------
# 图片元数据（含异常捕获）
# ---------------------------------------------------------------------------
def get_image_info(file_path: Path) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    """安全读取图片元数据。

    Returns:
        (width, height, mode, error_message)
        成功时 error_message 为 None。
    """
    try:
        with Image.open(file_path) as img:
            w, h = img.size
            mode = img.mode
            return w, h, mode, None
    except Exception as e:
        return None, None, None, f"error: {type(e).__name__}: {e}"


def compute_md5(file_path: Path) -> str:
    """计算文件 MD5 哈希。"""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 文件名解析
# ---------------------------------------------------------------------------
_WAFER_RE = re.compile(r"(?:WFR|wafer)(?:[-_][A-Za-z0-9]+){0,2}", re.IGNORECASE)
_DATETIME_RE = re.compile(r"(\d{14})")


def parse_wafer_id(filename: str) -> str:
    """从文件名中提取 wafer / WFR 标识字段。

    匹配形如 WFR_NO_BLT000、wafer_01 等模式。
    """
    m = _WAFER_RE.search(filename)
    return m.group(0) if m else ""


def parse_datetime_14(filename: str) -> str:
    """从文件名中提取 14 位合法时间戳 (yyyyMMddHHmmss)。

    仅当能被 datetime.strptime 成功解析时才返回，否则返回空字符串。
    """
    m = _DATETIME_RE.search(filename)
    if not m:
        return ""
    ts = m.group(1)
    try:
        datetime.strptime(ts, "%Y%m%d%H%M%S")
        return ts
    except ValueError:
        return ""


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------
def build_manifest(input_dir: Path, output_csv: Path):
    """扫描图片 → 构建 manifest → 写入 CSV。"""
    if not input_dir.exists():
        logger.error("输入目录不存在: %s", input_dir)
        sys.exit(1)

    image_files = collect_images(input_dir)
    if not image_files:
        logger.error("在 %s 中未找到任何支持的图片文件。", input_dir)
        sys.exit(1)

    logger.info("扫描到 %d 张图片。", len(image_files))

    # 确保输出目录存在
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    success_count = 0
    error_count = 0

    # 项目根目录：input_dir 是 data/raw_original，向上两级
    project_root = input_dir.resolve().parent.parent

    for idx, img_path in enumerate(image_files, start=1):
        original_filename = img_path.name

        # 相对路径（相对于项目根目录）
        try:
            original_relpath = str(img_path.resolve().relative_to(project_root).as_posix())
        except ValueError:
            original_relpath = str(img_path)

        file_size = img_path.stat().st_size

        # 图片元数据（损坏图片会被捕获）
        w, h, mode, error_msg = get_image_info(img_path)

        # 建议 ID / 文件名
        image_id = f"{ID_PREFIX}{idx:0{ID_ZFILL}d}"
        suggested_filename = f"{image_id}{img_path.suffix.lower()}"

        # MD5
        try:
            md5_hash = compute_md5(img_path)
        except Exception as e:
            md5_hash = ""
            if not error_msg:
                error_msg = f"error: MD5 failed: {e}"

        # 解析字段
        wafer = parse_wafer_id(original_filename)
        dt_14 = parse_datetime_14(original_filename)

        if error_msg:
            error_count += 1
            note = error_msg
            row = {
                "raw_index": str(idx),
                "original_filename": original_filename,
                "original_relpath": original_relpath,
                "width": "",
                "height": "",
                "mode": "",
                "file_size_bytes": str(file_size),
                "md5": md5_hash,
                "suggested_image_id": image_id,
                "suggested_filename": suggested_filename,
                "parse_wafer_id": wafer,
                "parse_datetime": dt_14,
                "note": note,
            }
        else:
            success_count += 1
            row = {
                "raw_index": str(idx),
                "original_filename": original_filename,
                "original_relpath": original_relpath,
                "width": str(w) if w is not None else "",
                "height": str(h) if h is not None else "",
                "mode": str(mode) if mode else "",
                "file_size_bytes": str(file_size),
                "md5": md5_hash,
                "suggested_image_id": image_id,
                "suggested_filename": suggested_filename,
                "parse_wafer_id": wafer,
                "parse_datetime": dt_14,
                "note": "",
            }

        rows.append(row)

    # 写入 CSV（UTF-8-SIG，方便 Excel 直接打开中文）
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    # 汇总输出
    logger.info("Manifest 已保存: %s", output_csv)
    logger.info("=" * 50)
    logger.info("  图片总数: %d", len(image_files))
    logger.info("  成功读取: %d", success_count)
    logger.info("  读取失败: %d", error_count)

    # 统计摘要
    if success_count > 0:
        widths = [int(r["width"]) for r in rows if r["width"]]
        heights = [int(r["height"]) for r in rows if r["height"]]
        sizes = [int(r["file_size_bytes"]) for r in rows]
        if widths:
            logger.info("--- 尺寸统计 ---")
            logger.info("  宽度范围: %d – %d px", min(widths), max(widths))
            logger.info("  高度范围: %d – %d px", min(heights), max(heights))
        if sizes:
            logger.info("  文件大小: %.1f – %.1f KB", min(sizes) / 1024, max(sizes) / 1024)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="扫描原始 AOI 图片，生成 raw_manifest.csv 清单"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw_original",
        help="输入目录，包含原始图片 (默认: data/raw_original)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/metadata/raw_manifest.csv",
        help="输出 CSV 路径 (默认: data/metadata/raw_manifest.csv)",
    )
    args = parser.parse_args()

    # 所有路径基于脚本所在项目根目录
    project_root = Path(__file__).resolve().parent.parent
    input_dir = (project_root / args.input_dir).resolve()
    output_csv = (project_root / args.output_csv).resolve()

    logger.info("项目根目录: %s", project_root)
    logger.info("输入目录:   %s", input_dir)
    logger.info("输出文件:   %s", output_csv)

    build_manifest(input_dir, output_csv)


if __name__ == "__main__":
    main()
