"""patch_models.py — Baseline 模型定义与训练。

提供:
  - train_sklearn_models(): 传统特征 + StandardScaler + SVM/LR/RF
  - ResNetEmbedder: 冻结 ResNet18 特征提取器
  - extract_resnet_embeddings(): 批量提取 embedding

用法:
    from aoi_defect.patch_models import train_sklearn_models, ResNetEmbedder
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision.models import resnet18, ResNet18_Weights

from .patch_features import PatchFeatureExtractor
from .metrics import compute_all_metrics

logger = logging.getLogger("patch_models")


# =============================================================================
# 实验 1: 传统特征 + sklearn
# =============================================================================
def train_sklearn_models(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    class_names: List[str],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """训练多个 sklearn 模型，返回最佳模型及性能。

    Returns:
        { "best_model": model, "scaler": scaler, "model_name": "...",
          "val_metrics": {...}, "models_comparison": [...] }
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    models = [
        ("SVM_RBF", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)),
        ("LogisticRegression", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ("RandomForest", RandomForestClassifier(class_weight="balanced_subsample", n_estimators=100, random_state=42)),
    ]

    best_model = None
    best_name = ""
    best_score = -1
    best_scaler = scaler
    comparison = []

    for name, model in models:
        logger.info("  训练 %s ...", name)
        t0 = time.time()
        model.fit(X_train_s, y_train)
        t1 = time.time()

        y_pred = model.predict(X_val_s)
        y_proba = model.predict_proba(X_val_s) if hasattr(model, "predict_proba") else None
        metrics = compute_all_metrics(y_val, y_pred, y_proba, class_names)

        comparison.append({
            "model": name, "train_time_s": round(t1 - t0, 2),
            **{k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))},
        })

        bacc = metrics.get("balanced_accuracy", 0)
        if bacc > best_score:
            best_score = bacc
            best_model = model
            best_name = name

        logger.info("    %s balanced_acc=%.4f", name, bacc)

    return {
        "best_model": best_model,
        "scaler": best_scaler,
        "model_name": best_name,
        "val_metrics": compute_all_metrics(y_val, best_model.predict(X_val_s),
                                            best_model.predict_proba(X_val_s) if hasattr(best_model, "predict_proba") else None,
                                            class_names),
        "models_comparison": comparison,
    }


# =============================================================================
# 实验 2: ResNet18 embedding
# =============================================================================
class ResNetEmbedder:
    """冻结 ResNet18 作为特征提取器。

    Parameters
    ----------
    device : str
        "cpu" 或 "cuda"
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.fc = torch.nn.Identity()  # 去掉分类层
        self.model.to(self.device)
        self.model.eval()
        self.transform = weights.transforms()

        # 记录权重信息
        self.weight_name = "ResNet18_Weights.DEFAULT"
        self.weight_hash = "torchvision_pretrained"

    def extract(self, patch_img: np.ndarray) -> np.ndarray:
        """从 BGR patch 提取 512 维 embedding。"""
        rgb = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(tensor)
        return emb.cpu().numpy().flatten()

    def extract_batch(self, patch_paths: List[Path]) -> np.ndarray:
        """批量提取 embedding。"""
        embs = []
        for i, p in enumerate(patch_paths):
            if i % 100 == 0:
                logger.info("    embedding %d/%d", i, len(patch_paths))
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            embs.append(self.extract(img))
        return np.array(embs, dtype=np.float32)


def train_on_embeddings(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """在 ResNet embeddings 上训练 LR 和 Linear SVM。"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    models = [
        ("LogisticRegression_ResNet", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ("LinearSVM_ResNet", SVC(kernel="linear", class_weight="balanced", probability=True, random_state=42)),
    ]

    best_model = None; best_name = ""; best_score = -1
    comparison = []

    for name, model in models:
        logger.info("  训练 %s ...", name)
        t0 = time.time()
        model.fit(X_train_s, y_train)
        t1 = time.time()

        y_pred = model.predict(X_val_s)
        y_proba = model.predict_proba(X_val_s) if hasattr(model, "predict_proba") else None
        metrics = compute_all_metrics(y_val, y_pred, y_proba, class_names)

        comparison.append({
            "model": name, "train_time_s": round(t1 - t0, 2),
            **{k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))},
        })

        bacc = metrics.get("balanced_accuracy", 0)
        if bacc > best_score: best_score = bacc; best_model = model; best_name = name

    return {
        "best_model": best_model, "scaler": scaler, "model_name": best_name,
        "val_metrics": compute_all_metrics(y_val, best_model.predict(X_val_s),
                                            best_model.predict_proba(X_val_s) if hasattr(best_model, "predict_proba") else None,
                                            class_names),
        "models_comparison": comparison,
    }


def save_model(model: Any, scaler: Any, config: Dict, output_dir: Path):
    """保存模型和相关配置。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "best_model.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    (output_dir / "feature_config.json").write_text(json.dumps(config.get("feature_config", {}), ensure_ascii=False, indent=2))
    (output_dir / "label_map.json").write_text(json.dumps(config.get("label_map", {}), ensure_ascii=False, indent=2))
    (output_dir / "run_manifest.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    logger.info("模型已保存: %s", output_dir)
