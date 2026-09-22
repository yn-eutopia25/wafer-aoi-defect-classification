"""io_utils.py — 通用 I/O 工具函数。

提供:
  - 项目根目录解析
  - 目录自动创建
  - CSV 安全读写（跳过注释行）
  - YAML 加载
  - 日志配置
"""

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# 项目根目录
# ---------------------------------------------------------------------------
def get_project_root() -> Path:
    """返回项目根目录（相对于本文件位于 src/aoi_defect/ 下两级）。"""
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# 目录
# ---------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    """确保目录存在，不存在则递归创建。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """创建带统一格式的 logger。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    """加载 YAML 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def read_csv_skip_comments(path: Path) -> List[Dict[str, str]]:
    """读取 CSV 文件，跳过以 # 开头的注释行。

    返回 list[dict]，每个 dict 以表头列为 key。
    """
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        data_lines = [row for row in reader if row and not row[0].strip().startswith("#")]

    if not data_lines:
        return []

    fieldnames = [fn.strip() for fn in data_lines[0]]
    rows: List[Dict[str, str]] = []
    for values in data_lines[1:]:
        row = {fn: values[i].strip() if i < len(values) else "" for i, fn in enumerate(fieldnames)}
        rows.append(row)

    return rows


def write_csv_with_comments(
    path: Path,
    fieldnames: List[str],
    rows: List[Dict[str, str]],
    comments: Optional[List[str]] = None,
    example_row: Optional[Dict[str, str]] = None,
) -> None:
    """写入带注释头的 CSV 文件。

    Parameters
    ----------
    path : Path
        输出 CSV 路径。
    fieldnames : list[str]
        表头列名。
    rows : list[dict]
        数据行。
    comments : list[str] | None
        写入数据前的注释行（不含 # 前缀）。
    example_row : dict | None
        注释掉的示例行。
    """
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        if comments:
            for c in comments:
                writer.writerow([f"# {c}"])
            writer.writerow(["#"])

        if example_row:
            example_values = [example_row.get(fn, "") for fn in fieldnames]
            writer.writerow(["# " + ",".join(str(v) for v in example_values)])
            writer.writerow(["# （以上为示例行，请删除或替换）"])
            writer.writerow(["#"])

        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([row.get(fn, "") for fn in fieldnames])
