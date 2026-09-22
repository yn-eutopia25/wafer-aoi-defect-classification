"""patch_features.py — 传统图像特征提取器。

从 patch 图像中提取手工设计的特征向量，用于传统 ML 基线。
支持: RGB/HSV/Lab 颜色统计、Sobel 边缘、Laplacian 方差、
LBP 纹理、HOG 梯度、GLCM 共现矩阵、几何特征。

用法:
    from aoi_defect.patch_features import PatchFeatureExtractor
    feats = PatchFeatureExtractor().extract(patch_img, bbox_info)
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


class PatchFeatureExtractor:
    """Patch 图像特征提取器。

    Parameters
    ----------
    resize_to : int | None
        特征提取前 resize 到的正方形尺寸。None 表示不 resize。
    include_lbp : bool
        是否提取 LBP 特征。
    include_hog : bool
        是否提取 HOG 特征。
    include_glcm : bool
        是否提取 GLCM 特征。
    include_geometry : bool
        是否提取几何 (bbox) 特征。
    """

    def __init__(self, resize_to: int = 128, include_lbp: bool = True,
                 include_hog: bool = True, include_glcm: bool = True,
                 include_geometry: bool = True):
        self.resize_to = resize_to
        self.include_lbp = include_lbp
        self.include_hog = include_hog
        self.include_glcm = include_glcm
        self.include_geometry = include_geometry

    def extract(self, patch_img: np.ndarray,
                bbox_width: float = 0, bbox_height: float = 0,
                bbox_area: float = 0, patch_area: float = 0,
                polygon_area: float = 0) -> np.ndarray:
        """提取特征向量。出错时返回零向量。"""
        try:
            return self._extract_impl(patch_img, bbox_width, bbox_height,
                                       bbox_area, patch_area, polygon_area)
        except Exception as e:
            # 返回与正常提取相同长度的零向量
            n = len(self.feature_names())
            return np.zeros(n, dtype=np.float32)

    def _extract_impl(self, patch_img: np.ndarray,
                bbox_width: float = 0, bbox_height: float = 0,
                bbox_area: float = 0, patch_area: float = 0,
                polygon_area: float = 0) -> np.ndarray:
        # Ensure uint8 BGR, contiguous
        if patch_img.dtype != np.uint8:
            if patch_img.dtype in (np.float32, np.float64):
                patch_img = (np.clip(patch_img, 0, 255)).astype(np.uint8)
            else:
                patch_img = patch_img.astype(np.uint8)
        patch_img = np.ascontiguousarray(patch_img)
        if len(patch_img.shape) == 2:
            patch_img = cv2.cvtColor(patch_img, cv2.COLOR_GRAY2BGR)
        elif patch_img.shape[2] == 4:
            patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGRA2BGR)
        if self.resize_to and (patch_img.shape[0] != self.resize_to or patch_img.shape[1] != self.resize_to):
            try:
                patch_img = cv2.resize(patch_img, (self.resize_to, self.resize_to))
            except Exception:
                patch_img = cv2.resize(patch_img.astype(np.float32), (self.resize_to, self.resize_to)).astype(np.uint8)

        features: List[float] = []

        # 1. RGB 统计
        for ch in range(3):
            ch_data = patch_img[:, :, ch].astype(np.float32)
            features.extend(_channel_stats(ch_data))
            features.append(float(np.mean(ch_data)))
            features.append(float(np.std(ch_data)))

        # 2. HSV 统计
        hsv = cv2.cvtColor(patch_img, cv2.COLOR_BGR2HSV)
        for ch in range(3):
            ch_data = hsv[:, :, ch].astype(np.float32)
            features.append(float(np.mean(ch_data)))
            features.append(float(np.std(ch_data)))

        # 3. Lab 统计
        lab = cv2.cvtColor(patch_img, cv2.COLOR_BGR2LAB)
        for ch in range(3):
            ch_data = lab[:, :, ch].astype(np.float32)
            features.append(float(np.mean(ch_data)))
            features.append(float(np.std(ch_data)))

        # 4. 灰度统计
        gray = cv2.cvtColor(patch_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        features.append(float(np.mean(gray)))
        features.append(float(np.std(gray)))
        features.extend(_channel_stats(gray))

        # 5. Sobel — use uint8 input, CV_32F output (OpenCV 4.13 不支持 32F→64F)
        gray_u8 = gray.astype(np.uint8)
        sx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sx ** 2 + sy ** 2).astype(np.float32)
        features.append(float(np.mean(sobel_mag)))
        features.append(float(np.std(sobel_mag)))
        features.append(float(np.sum(sobel_mag > np.percentile(sobel_mag, 90)) / max(sobel_mag.size, 1)))

        # 6. Laplacian variance
        lap = cv2.Laplacian(gray_u8, cv2.CV_32F)
        features.append(float(lap.var()))

        # 7. LBP
        if self.include_lbp:
            lbp = _lbp_histogram(gray)
            features.extend(lbp)

        # 8. HOG
        if self.include_hog:
            hog = _hog_features(gray, self.resize_to or 128)
            features.extend(hog.tolist())

        # 9. GLCM
        if self.include_glcm:
            glcm_feats = _glcm_features(gray)
            features.extend(glcm_feats)

        # 10. 几何
        if self.include_geometry:
            features.append(float(bbox_width))
            features.append(float(bbox_height))
            features.append(float(bbox_width / max(bbox_height, 1)))
            features.append(float(bbox_area / max(patch_area, 1)))
            features.append(float(polygon_area / max(bbox_area, 1)))

        return np.array(features, dtype=np.float32)

    def feature_names(self) -> List[str]:
        """返回特征名列表 (与 extract 输出对齐)。"""
        names = []
        for ch in ["R", "G", "B"]:
            for pct in ["p10", "p25", "p50", "p75", "p90"]:
                names.append(f"{ch}_{pct}")
            names.append(f"{ch}_mean"); names.append(f"{ch}_std")
        for ch in ["H", "S", "V"]:
            names.append(f"{ch}_mean"); names.append(f"{ch}_std")
        for ch in ["L", "A", "B"]:
            names.append(f"{ch}_mean"); names.append(f"{ch}_std")
        names += ["gray_mean", "gray_std", "gray_p10", "gray_p25", "gray_p50", "gray_p75", "gray_p90"]
        names += ["sobel_mean", "sobel_std", "sobel_edge_ratio", "laplacian_var"]
        if self.include_lbp:
            names += [f"lbp_{i}" for i in range(10)]
        if self.include_hog:
            names += [f"hog_{i}" for i in range(36)]
        if self.include_glcm:
            names += ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]
        if self.include_geometry:
            names += ["bbox_width", "bbox_height", "bbox_aspect", "bbox_area_ratio", "poly_area_ratio"]
        return names


# =============================================================================
# 内部帮助函数
# =============================================================================
def _channel_stats(data: np.ndarray) -> List[float]:
    ps = [10, 25, 50, 75, 90]
    vals = np.percentile(data, ps)
    return vals.tolist()


def _lbp_histogram(gray: np.ndarray) -> List[float]:
    """简单 LBP 直方图 (10 bins uniform pattern-like)"""
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = gray[y, x]
            code = 0
            for i, (dy, dx) in enumerate([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]):
                if gray[y + dy, x + dx] >= center:
                    code |= (1 << i)
            lbp[y, x] = code % 10
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9))
    return (hist / hist.sum()).tolist()


def _hog_features(gray: np.ndarray, size: int) -> np.ndarray:
    """HOG 特征 (简化版, 4×4 cells)."""
    try:
        win_size = (size, size)
        block_size = (size // 2, size // 2)
        cell_size = (size // 4, size // 4)
        block_stride = cell_size
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, 9)
        gray_uint8 = gray.astype(np.uint8)
        result = hog.compute(gray_uint8)
        if result is None:
            return np.zeros(36, dtype=np.float32)
        return result.flatten()
    except Exception:
        return np.zeros(36, dtype=np.float32)


def _glcm_features(gray: np.ndarray) -> List[float]:
    """GLCM 5 个特征 (contrast, dissimilarity, homogeneity, energy, correlation)"""
    try:
        gray_uint8 = np.clip(gray / 4, 0, 63).astype(np.uint8)
        # 向量化计算水平方向共生矩阵
        a = gray_uint8[:, :-1].ravel().astype(np.int64)
        b = gray_uint8[:, 1:].ravel().astype(np.int64)
        # 构建 64x64 共生矩阵
        glcm = np.zeros((64, 64), dtype=np.float64)
        np.add.at(glcm, (a, b), 1)
        glcm /= max(glcm.sum(), 1)

        i = np.arange(64, dtype=np.float64)
        I, J = np.meshgrid(i, i)
        contrast = float(np.sum(glcm * (I - J) ** 2))
        dissimilarity = float(np.sum(glcm * np.abs(I - J)))
        homogeneity = float(np.sum(glcm / (1 + (I - J) ** 2)))
        energy = float(np.sum(glcm ** 2))
        mu_i = np.sum(glcm * I); mu_j = np.sum(glcm * J)
        si = np.sqrt(np.sum(glcm * (I - mu_i) ** 2))
        sj = np.sqrt(np.sum(glcm * (J - mu_j) ** 2))
        correlation = float(np.sum(glcm * (I - mu_i) * (J - mu_j)) / max(si * sj, 1e-6))
        return [contrast, dissimilarity, homogeneity, energy, correlation]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
