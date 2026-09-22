"""image_utils.py — 图像处理工具函数。

提供:
  - 图片尺寸获取
  - MD5 哈希计算
  - 缩略图生成
  - Contact sheet 拼接
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .io_utils import setup_logger

logger = setup_logger("image_utils")

# 支持的图片扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# 文件收集
# ---------------------------------------------------------------------------
def collect_images(directory: Path) -> List[Path]:
    """收集目录下所有支持的图片文件，排序返回。"""
    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(set(files), key=lambda p: p.name)


# ---------------------------------------------------------------------------
# 元数据
# ---------------------------------------------------------------------------
def compute_md5(file_path: Path) -> str:
    """计算文件 MD5 哈希。"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_image_size(file_path: Path) -> Optional[Tuple[int, int]]:
    """获取图片 (width, height)，失败返回 None。"""
    try:
        with Image.open(file_path) as img:
            return img.size
    except Exception:
        logger.warning(f"无法读取图片尺寸: {file_path.name}")
        return None


# ---------------------------------------------------------------------------
# 缩略图
# ---------------------------------------------------------------------------
def make_thumbnail(
    img_path: Path,
    size: int = 200,
) -> Image.Image:
    """生成正方形缩略图（保持比例、居中裁剪）。"""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return img


def build_contact_sheet(
    image_paths: List[Path],
    output_path: Path,
    thumb_size: int = 200,
    grid_cols: int = 10,
    thumb_dir: Optional[Path] = None,
    add_numbering: bool = True,
) -> Path:
    """生成 contact sheet 大图。

    Parameters
    ----------
    image_paths : list[Path]
        图片路径列表。
    output_path : Path
        输出大图路径。
    thumb_size : int
        单张缩略图尺寸。
    grid_cols : int
        每行列数。
    thumb_dir : Path | None
        缩略图缓存目录，None 则生成到 output_path 同级的 thumbnails/。
    add_numbering : bool
        是否在缩略图上叠加编号。

    Returns
    -------
    Path
        生成的 contact sheet 路径。
    """
    if thumb_dir is None:
        thumb_dir = output_path.parent / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # 生成缩略图
    logger.info("生成缩略图 (%d 张)...", len(image_paths))
    for img_path in tqdm(image_paths, desc="缩略图", unit="张"):
        out = thumb_dir / f"thumb_{img_path.stem}.jpg"
        if out.exists():
            continue
        try:
            thumb = make_thumbnail(img_path, thumb_size)
            thumb.save(out, "JPEG", quality=85)
        except Exception as e:
            logger.warning(f"缩略图生成失败: {img_path.name}: {e}")

    # 拼接
    total = len(image_paths)
    grid_rows = (total + grid_cols - 1) // grid_cols
    canvas_w = grid_cols * thumb_size
    canvas_h = grid_rows * thumb_size
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))

    logger.info("拼接 contact sheet (%d × %d)...", grid_cols, grid_rows)
    placeholder = Image.new("RGB", (thumb_size, thumb_size), color=(60, 60, 60))

    for idx, img_path in enumerate(tqdm(image_paths, desc="拼接", unit="张")):
        thumb_file = thumb_dir / f"thumb_{img_path.stem}.jpg"
        try:
            thumb = Image.open(thumb_file)
        except Exception:
            thumb = placeholder

        col = idx % grid_cols
        row = idx // grid_cols
        canvas.paste(thumb, (col * thumb_size, row * thumb_size))

    # 编号
    if add_numbering:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", size=10)
        except Exception:
            font = ImageFont.load_default()

        for idx in range(total):
            col = idx % grid_cols
            row = idx // grid_cols
            x = col * thumb_size + 3
            y = (row + 1) * thumb_size - 14
            draw.text((x, y), str(idx + 1), fill=(255, 255, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=90)
    logger.info("Contact sheet 已保存: %s", output_path)
    return output_path
