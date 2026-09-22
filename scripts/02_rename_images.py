#!/usr/bin/env python3
"""
02_rename_images.py — 复制原始图片并统一重命名

读取 raw_manifest.csv，将 data/raw_original/ 下的图片复制到 data/images/，
按 AOI_NG_0001.jpg 格式重命名。原始文件不做任何修改。

同时输出 images_manifest.csv，记录每张图片的复制状态，并验证 MD5 一致性。

用法:
    python scripts/02_rename_images.py
    python scripts/02_rename_images.py --overwrite
    python scripts/02_rename_images.py --manifest_csv data/metadata/raw_manifest.csv --output_dir data/images --output_csv data/metadata/images_manifest.csv
"""

import argparse
import csv
import hashlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logger("rename_images")


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
IMAGES_MANIFEST_FIELDS = [
    "image_id",
    "filename",
    "image_relpath",
    "original_filename",
    "original_relpath",
    "width",
    "height",
    "md5_original",
    "md5_copied",
    "parse_wafer_id",
    "parse_datetime",
    "copy_status",
    "note",
]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def compute_md5(file_path: Path) -> str:
    """计算文件 MD5。"""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    """读取 raw_manifest.csv（UTF-8-SIG 编码）。"""
    if not path.exists():
        logger.error("Manifest 文件不存在: %s", path)
        logger.error("请先运行 scripts/01_build_manifest.py")
        sys.exit(1)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.error("Manifest 为空。")
        sys.exit(1)

    return rows


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------
def rename_images(
    manifest_csv: Path,
    raw_dir: Path,
    output_dir: Path,
    output_csv: Path,
    overwrite: bool,
    project_root: Path,
):
    """核心：读取 manifest → 复制 → 校验 MD5 → 输出 images_manifest.csv。"""
    rows = read_manifest(manifest_csv)
    logger.info("从 manifest 读取到 %d 条记录。", len(rows))

    # 确保目标目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "success": 0, "skipped": 0, "failed": 0}
    out_rows: List[Dict[str, str]] = []

    for row in rows:
        stats["total"] += 1

        # --- 从 manifest 提取字段 ---
        original_filename = row.get("original_filename", "")
        original_relpath = row.get("original_relpath", "")
        suggested_filename = row.get("suggested_filename", "")
        image_id = row.get("suggested_image_id", "")
        width = row.get("width", "")
        height = row.get("height", "")
        md5_original = row.get("md5", "")
        wafer_id = row.get("parse_wafer_id", "")
        dt_14 = row.get("parse_datetime", "")
        note_existing = row.get("note", "")

        # 源文件路径
        source_path = project_root / original_relpath

        # 目标路径
        dest_path = output_dir / suggested_filename
        image_relpath = f"data/images/{suggested_filename}"

        # --- 构建输出基准行 ---
        base_row = {
            "image_id": image_id,
            "filename": suggested_filename,
            "image_relpath": image_relpath,
            "original_filename": original_filename,
            "original_relpath": original_relpath,
            "width": width,
            "height": height,
            "md5_original": md5_original,
            "md5_copied": "",
            "parse_wafer_id": wafer_id,
            "parse_datetime": dt_14,
            "copy_status": "",
            "note": note_existing,
        }

        # --- 源文件检查 ---
        if not source_path.exists():
            out_row = dict(base_row)
            out_row["copy_status"] = "failed"
            out_row["note"] = _append_note(note_existing, "source not found")
            out_rows.append(out_row)
            stats["failed"] += 1
            logger.warning("源文件不存在: %s", source_path)
            continue

        # --- 目标已存在检查 ---
        if dest_path.exists() and not overwrite:
            # 跳过，但仍验证已存在文件的 MD5
            try:
                md5_copied = compute_md5(dest_path)
                out_row = dict(base_row)
                out_row["md5_copied"] = md5_copied
                if md5_copied == md5_original:
                    out_row["copy_status"] = "skipped"
                else:
                    out_row["copy_status"] = "skipped"
                    out_row["note"] = _append_note(
                        out_row["note"],
                        "MD5 mismatch with original — may need re-copy with --overwrite",
                    )
                    logger.warning("%s: 已存在但 MD5 不一致", suggested_filename)
                out_rows.append(out_row)
                stats["skipped"] += 1
                continue
            except Exception as e:
                out_row = dict(base_row)
                out_row["copy_status"] = "failed"
                out_row["note"] = _append_note(note_existing, f"read existing failed: {e}")
                out_rows.append(out_row)
                stats["failed"] += 1
                logger.warning("读取已存在文件失败: %s: %s", suggested_filename, e)
                continue

        # --- 执行复制 ---
        try:
            shutil.copy2(source_path, dest_path)
        except Exception as e:
            out_row = dict(base_row)
            out_row["copy_status"] = "failed"
            out_row["note"] = _append_note(note_existing, f"copy failed: {e}")
            out_rows.append(out_row)
            stats["failed"] += 1
            logger.error("复制失败: %s → %s: %s", source_path, dest_path, e)
            continue

        # --- 验证复制后的 MD5 ---
        try:
            md5_copied = compute_md5(dest_path)
        except Exception as e:
            out_row = dict(base_row)
            out_row["copy_status"] = "failed"
            out_row["note"] = _append_note(note_existing, f"MD5 verify failed: {e}")
            out_rows.append(out_row)
            stats["failed"] += 1
            logger.error("复制后 MD5 校验失败: %s: %s", suggested_filename, e)
            continue

        if md5_copied == md5_original:
            out_row = dict(base_row)
            out_row["md5_copied"] = md5_copied
            out_row["copy_status"] = "success"
            out_rows.append(out_row)
            stats["success"] += 1
        else:
            out_row = dict(base_row)
            out_row["md5_copied"] = md5_copied
            out_row["copy_status"] = "failed"
            out_row["note"] = _append_note(
                note_existing,
                f"MD5 mismatch: original={md5_original} copied={md5_copied}",
            )
            out_rows.append(out_row)
            stats["failed"] += 1
            logger.error("%s: MD5 不一致！", suggested_filename)

    # --- 写入 images_manifest.csv（UTF-8-SIG） ---
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=IMAGES_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    # --- 汇总输出 ---
    logger.info("Images manifest 已保存: %s", output_csv)
    logger.info("=" * 50)
    logger.info("  应复制数量: %d", stats["total"])
    logger.info("  成功:       %d", stats["success"])
    logger.info("  跳过:       %d", stats["skipped"])
    logger.info("  失败:       %d", stats["failed"])
    logger.info("  输出目录:    %s", output_dir)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _append_note(existing: str, msg: str) -> str:
    """在已有 note 后追加信息，用分号分隔。"""
    if not existing:
        return msg
    return f"{existing}; {msg}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="复制原始 AOI 图片并统一重命名"
    )
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default="data/metadata/raw_manifest.csv",
        help="输入 manifest CSV 路径 (默认: data/metadata/raw_manifest.csv)",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw_original",
        help="原始图片目录 (默认: data/raw_original)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/images",
        help="输出图片目录 (默认: data/images)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/metadata/images_manifest.csv",
        help="输出 images manifest CSV 路径 (默认: data/metadata/images_manifest.csv)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="覆盖 data/images/ 中已存在的文件 (默认: 跳过已存在)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    manifest_csv = (project_root / args.manifest_csv).resolve()
    raw_dir = (project_root / args.raw_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_csv = (project_root / args.output_csv).resolve()

    logger.info("项目根目录: %s", project_root)
    logger.info("Manifest:   %s", manifest_csv)
    logger.info("原始目录:    %s", raw_dir)
    logger.info("输出目录:    %s", output_dir)
    logger.info("输出 CSV:   %s", output_csv)
    logger.info("覆盖模式:    %s", "是" if args.overwrite else "否（跳过已存在）")

    rename_images(manifest_csv, raw_dir, output_dir, output_csv, args.overwrite, project_root)


if __name__ == "__main__":
    main()
